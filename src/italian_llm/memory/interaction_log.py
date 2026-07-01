"""Log persistente delle interazioni + retrieval BM25 + export SFT (solo stdlib).

Il ciclo di "apprendimento ricorsivo" ha due gambe:

1. **Memoria immediata (questo modulo + scripts/chat_learn.py)**: ogni scambio
   domanda/risposta viene salvato in JSONL; alle domande successive i turni
   passati piu' rilevanti (BM25, riuso di rag/code_index) vengono iniettati nel
   contesto. Nessun retrain: il modello "ricorda" da subito.
2. **Apprendimento nei pesi**: `to_sft_examples` converte il log nello schema
   SFT del repo; il dataset alimenta il retrain periodico QLoRA (es. notebook
   Kaggle) e il modello aggiornato torna in Ollama. Vedi docs/continuous-learning.md.
"""

import json
import os
from datetime import UTC, datetime

from italian_llm.data.prompts import SYSTEM_DEFAULT
from italian_llm.data.schema import SFTExample
from italian_llm.rag.code_index import CodeIndex

__all__ = [
    "append_interaction",
    "load_interactions",
    "retrieve_memory",
    "memory_context",
    "to_sft_examples",
]


def load_interactions(path: str) -> list[dict]:
    """Legge il log JSONL delle interazioni; lista vuota se il file non esiste."""
    if not path or not os.path.exists(path):
        return []
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue  # riga corrotta: la memoria resta usabile
            if isinstance(rec, dict):
                records.append(rec)
    return records


def append_interaction(
    path: str,
    question: str,
    answer: str,
    *,
    model: str = "",
    tags: list[str] | None = None,
) -> dict:
    """Accoda uno scambio al log e ritorna il record scritto."""
    existing = load_interactions(path)
    rec = {
        "id": f"mem-{len(existing) + 1:06d}",
        "ts": datetime.now(UTC).isoformat(),
        "model": model,
        "question": question,
        "answer": answer,
        "tags": list(tags or []),
    }
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def _as_chunks(records: list[dict]) -> list[dict]:
    """Mappa le interazioni nella forma chunk usata da CodeIndex."""
    chunks = []
    for rec in records:
        text = f"D: {rec.get('question', '')}\nR: {rec.get('answer', '')}"
        chunks.append({"path": str(rec.get("id", "")), "start_line": 1, "text": text})
    return chunks


def retrieve_memory(records: list[dict], query: str, k: int = 3) -> list[dict]:
    """Ritorna le interazioni passate piu' rilevanti per `query` (BM25)."""
    if not records:
        return []
    index = CodeIndex.build(_as_chunks(records))
    hits = index.search(query, k=k)
    by_id = {str(rec.get("id", "")): rec for rec in records}
    return [by_id[c["path"]] for c in hits if c["path"] in by_id]


def memory_context(records: list[dict], query: str, k: int = 3, max_chars: int = 2000) -> str:
    """Blocco di contesto pronto da iniettare nel system prompt ('' se nulla)."""
    parts = []
    used = 0
    for rec in retrieve_memory(records, query, k=k):
        block = (
            f"[{rec.get('id', '?')}] D: {rec.get('question', '')}\n" f"R: {rec.get('answer', '')}"
        )
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def to_sft_examples(
    records: list[dict],
    *,
    system: str = SYSTEM_DEFAULT,
    domain: str = "chat",
) -> list[dict]:
    """Converte il log nello schema SFT del repo (per il retrain periodico)."""
    out = []
    for rec in records:
        question = (rec.get("question") or "").strip()
        answer = (rec.get("answer") or "").strip()
        if not question or not answer:
            continue
        example = SFTExample(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            id=f"{rec.get('id', 'mem-unknown')}-sft",
            source_type="user_interaction",
            domain=domain,
            difficulty="medium",
            teacher_name=str(rec.get("model", "")),
        )
        out.append(example.to_dict())
    return out
