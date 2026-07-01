"""RAG leggero sul codebase locale: chunking + retrieval BM25 (solo stdlib).

Idea: un modello piccolo risponde molto meglio se gli si dà il pezzo GIUSTO del
TUO codice. Niente dipendenze pesanti: tokenizzazione regex + BM25 in puro Python.
"""

import math
import os
import re
from collections import Counter

__all__ = ["chunk_text", "index_paths", "CodeIndex", "build_context"]

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def chunk_text(path: str, text: str, max_lines: int = 40) -> list[dict]:
    """Spezza il testo in chunk di al massimo `max_lines` righe."""
    lines = (text or "").splitlines()
    chunks = []
    for start in range(0, max(len(lines), 1), max_lines):
        body = "\n".join(lines[start : start + max_lines])
        if body.strip():
            chunks.append({"path": path, "start_line": start + 1, "text": body})
    return chunks


def index_paths(
    root: str,
    exts: tuple = (".cs", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".py"),
    max_lines: int = 40,
) -> list[dict]:
    """Indicizza ricorsivamente i file con estensione in `exts` sotto `root`."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {".git", "node_modules", "__pycache__", "outputs", ".venv"}
        ]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                full = os.path.join(dirpath, fn)
                try:
                    with open(full, encoding="utf-8", errors="ignore") as fh:
                        out.extend(chunk_text(full, fh.read(), max_lines=max_lines))
                except OSError:
                    continue
    return out


class CodeIndex:
    """Indice BM25 in memoria sui chunk di codice."""

    def __init__(self, chunks, doc_tokens, df, avgdl, k1=1.5, b=0.75):
        self.chunks = chunks
        self.doc_tokens = doc_tokens
        self.df = df
        self.avgdl = avgdl
        self.n = len(chunks)
        self.k1 = k1
        self.b = b

    @classmethod
    def build(cls, chunks: list[dict]) -> "CodeIndex":
        doc_tokens = [_tokens(c["text"]) for c in chunks]
        df = Counter()
        for toks in doc_tokens:
            for term in set(toks):
                df[term] += 1
        avgdl = (sum(len(t) for t in doc_tokens) / len(doc_tokens)) if doc_tokens else 0.0
        return cls(chunks, doc_tokens, df, avgdl)

    def _idf(self, term: str) -> float:
        n_q = self.df.get(term, 0)
        if n_q == 0:
            return 0.0
        return math.log(1 + (self.n - n_q + 0.5) / (n_q + 0.5))

    def search(self, query: str, k: int = 4) -> list[dict]:
        q_terms = _tokens(query)
        scores = []
        for i, toks in enumerate(self.doc_tokens):
            if not toks:
                scores.append((0.0, i))
                continue
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self._idf(term) * (f * (self.k1 + 1)) / denom
            scores.append((s, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [self.chunks[i] for s, i in scores[:k] if s > 0]


def build_context(chunks: list[dict], max_chars: int = 2000) -> str:
    """Formatta i chunk recuperati come blocco di contesto, troncando a max_chars."""
    parts = []
    used = 0
    for c in chunks:
        header = f"# {c['path']}:{c['start_line']}\n"
        block = header + c["text"]
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)
