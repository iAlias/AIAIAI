#!/usr/bin/env python
"""Legge i dati grezzi (JSONL/TXT), normalizza il testo e scrive un corpus intermedio JSONL."""

from __future__ import annotations

import os, sys  # noqa: E401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse  # noqa: E402
import glob  # noqa: E402

from italian_llm.config import get, load_config  # noqa: E402
from italian_llm.logging_utils import get_logger, setup_logging  # noqa: E402
from italian_llm.utils.io import read_jsonl, write_jsonl  # noqa: E402

logger = get_logger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join("configs", "data", "corpus.yaml")

# Documenti minimi grezzi: scartiamo testi banali già in questa fase (filtro morbido).
_RAW_MIN_CHARS = 40


def _abspath(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _iter_raw_documents(raw_dir: str):
    """Itera documenti grezzi da tutti i .jsonl e .txt sotto raw_dir (ricorsivo).

    JSONL: ogni riga con campo testo (text/content/body) diventa un documento.
    TXT: il file viene spezzato in paragrafi (doppio newline) -> più documenti.
    """
    jsonl_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.jsonl"), recursive=True))
    txt_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.txt"), recursive=True))

    for path in jsonl_files:
        source = os.path.splitext(os.path.basename(path))[0]
        try:
            for i, rec in enumerate(read_jsonl(path)):
                text = rec.get("text") or rec.get("content") or rec.get("body") or ""
                if not isinstance(text, str):
                    continue
                yield {
                    "id": rec.get("id") or f"{source}-{i}",
                    "text": text,
                    "source": rec.get("source") or source,
                    "domain": rec.get("domain") or "general",
                }
        except Exception as exc:
            logger.warning("Lettura di '%s' fallita: %s", path, exc)

    for path in txt_files:
        source = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                blob = fh.read()
        except Exception as exc:
            logger.warning("Lettura del file di testo '%s' fallita: %s", path, exc)
            continue
        paragraphs = [p for p in blob.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            yield {"id": f"{source}-txt-{i}", "text": para, "source": source, "domain": "general"}


def build(config_path: str, in_dir: str | None, out_path: str | None) -> int:
    """Costruisce il corpus intermedio normalizzato; ritorna il numero di documenti scritti."""
    from italian_llm.data.cleaning import normalize_unicode, remove_boilerplate  # lazy (modulo puro)

    cfg = load_config(config_path)
    do_normalize = bool(get(cfg, "corpus.filters.normalize_unicode", True))
    do_boilerplate = bool(get(cfg, "corpus.filters.remove_boilerplate", True))

    raw_dir = _abspath(in_dir or get(cfg, "paths.data_dir", "data/raw"))
    out_file = _abspath(out_path or get(cfg, "corpus.output.interim_path", "data/interim/corpus.jsonl"))

    if not os.path.isdir(raw_dir):
        logger.error("Directory dati grezzi inesistente: %s (esegui prima download_open_data.py).", raw_dir)
        return 0

    logger.info("Costruzione corpus intermedio: input=%s output=%s", raw_dir, out_file)

    n_in = 0
    n_short = 0
    n_empty = 0

    def _processed_rows():
        nonlocal n_in, n_short, n_empty
        for doc in _iter_raw_documents(raw_dir):
            n_in += 1
            text = doc["text"]
            if do_normalize:
                text = normalize_unicode(text)
            if do_boilerplate:
                text = remove_boilerplate(text)
            text = text.strip()
            if not text:
                n_empty += 1
                continue
            if len(text) < _RAW_MIN_CHARS:
                n_short += 1
                continue
            doc["text"] = text
            doc["n_chars"] = len(text)
            yield doc

    n_out = write_jsonl(out_file, _processed_rows())

    logger.info("=== Corpus intermedio costruito ===")
    logger.info("  documenti letti     : %d", n_in)
    logger.info("  scartati (vuoti)    : %d", n_empty)
    logger.info("  scartati (corti)    : %d", n_short)
    logger.info("  documenti scritti   : %d", n_out)
    logger.info("  output              : %s", out_file)
    return n_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalizza i dati grezzi in un corpus intermedio JSONL.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML del corpus (default: %(default)s).")
    parser.add_argument("--in-dir", default=None, help="Directory dati grezzi (default: paths.data_dir).")
    parser.add_argument("--out", default=None, help="Percorso JSONL di output (default: data/interim/corpus.jsonl).")
    parser.add_argument("--log-level", default="INFO", help="Livello di log (default: INFO).")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    try:
        n = build(_abspath(args.config), args.in_dir, args.out)
    except Exception as exc:
        logger.error("Errore nella costruzione del corpus: %s", exc)
        return 1
    return 0 if n > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
