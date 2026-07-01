#!/usr/bin/env python
"""Impacchetta il corpus processato in blocchi di testo a lunghezza fissa per il Continued Pre-Training (CPT)."""

from __future__ import annotations

import os  # noqa: E401
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse  # noqa: E402

from italian_llm.config import get, load_config  # noqa: E402
from italian_llm.logging_utils import get_logger, setup_logging  # noqa: E402
from italian_llm.utils.io import read_jsonl, write_jsonl  # noqa: E402

logger = get_logger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join("configs", "data", "corpus.yaml")

# Separatore di documento inserito tra testi concatenati (riconoscibile e neutro).
_DOC_SEP = "\n\n"
# Stima grezza caratteri-per-token per l'italiano (per derivare block-chars dai token).
_CHARS_PER_TOKEN = 4


def _abspath(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def pack(
    config_path: str,
    in_path: str | None,
    out_path: str | None,
    block_chars: int | None,
    block_tokens: int | None,
    min_block_chars: int,
) -> dict:
    """Concatena e impacchetta i documenti in blocchi ~block_chars; ritorna statistiche."""
    cfg = load_config(config_path)

    in_file = _abspath(
        in_path or get(cfg, "corpus.output.processed_path", "data/processed/corpus.jsonl")
    )
    out_file = _abspath(
        out_path or get(cfg, "corpus.output.cpt_path", "data/processed/cpt_blocks.jsonl")
    )

    if block_chars is None:
        if block_tokens is not None:
            block_chars = int(block_tokens) * _CHARS_PER_TOKEN
        else:
            block_chars = 4096  # ~1024 token: blocco CPT compatto e ragionevole

    if not os.path.isfile(in_file):
        logger.error(
            "Corpus processato non trovato: %s (esegui prima clean_dedup_filter.py).", in_file
        )
        return {"blocks": 0}

    logger.info("Packing CPT: input=%s output=%s block_chars=%d", in_file, out_file, block_chars)

    n_docs = 0
    total_chars = 0
    buffer: list[str] = []
    buffer_len = 0
    block_index = 0

    def _flush():
        """Costruisce un blocco dal buffer corrente e lo restituisce (o None se vuoto)."""
        nonlocal buffer, buffer_len, block_index
        if not buffer:
            return None
        text = _DOC_SEP.join(buffer).strip()
        buffer = []
        buffer_len = 0
        if len(text) < min_block_chars:
            return None
        block = {"id": f"cpt-{block_index:06d}", "text": text, "n_chars": len(text)}
        block_index += 1
        return block

    def _blocks():
        nonlocal n_docs, total_chars, buffer, buffer_len
        for rec in read_jsonl(in_file):
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            n_docs += 1
            total_chars += len(text)

            # Spezza documenti molto lunghi su confini di parola per non superare il blocco.
            words = text.split(" ")
            chunk: list[str] = []
            chunk_len = 0
            for w in words:
                add = len(w) + 1
                if chunk_len + add > block_chars and chunk:
                    piece = " ".join(chunk)
                    buffer.append(piece)
                    buffer_len += len(piece) + len(_DOC_SEP)
                    chunk, chunk_len = [], 0
                    if buffer_len >= block_chars:
                        blk = _flush()
                        if blk:
                            yield blk
                chunk.append(w)
                chunk_len += add
            if chunk:
                piece = " ".join(chunk)
                buffer.append(piece)
                buffer_len += len(piece) + len(_DOC_SEP)
                if buffer_len >= block_chars:
                    blk = _flush()
                    if blk:
                        yield blk
        # Ultimo blocco residuo.
        blk = _flush()
        if blk:
            yield blk

    n_blocks = write_jsonl(out_file, _blocks())

    logger.info("=== Packing CPT completato ===")
    logger.info("  documenti letti    : %d", n_docs)
    logger.info("  caratteri totali   : %d", total_chars)
    logger.info("  blocchi scritti    : %d", n_blocks)
    logger.info("  ~token stimati     : %d", total_chars // _CHARS_PER_TOKEN)
    logger.info("  output             : %s", out_file)
    return {"docs": n_docs, "blocks": n_blocks, "chars": total_chars}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Impacchetta il corpus in blocchi di testo per il CPT."
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, help="YAML del corpus (default: %(default)s)."
    )
    parser.add_argument(
        "--in", dest="in_path", default=None, help="Corpus processato (default da config)."
    )
    parser.add_argument(
        "--out", default=None, help="JSONL blocchi CPT (default: data/processed/cpt_blocks.jsonl)."
    )
    parser.add_argument(
        "--block-chars", type=int, default=None, help="Caratteri per blocco (default: 4096)."
    )
    parser.add_argument(
        "--block-tokens", type=int, default=None, help="Token per blocco (deriva i caratteri)."
    )
    parser.add_argument(
        "--min-block-chars", type=int, default=200, help="Lunghezza minima di un blocco."
    )
    parser.add_argument("--log-level", default="INFO", help="Livello di log (default: INFO).")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    try:
        stats = pack(
            _abspath(args.config),
            args.in_path,
            args.out,
            args.block_chars,
            args.block_tokens,
            args.min_block_chars,
        )
    except Exception as exc:
        logger.error("Errore nel packing CPT: %s", exc)
        return 1
    return 0 if stats.get("blocks", 0) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
