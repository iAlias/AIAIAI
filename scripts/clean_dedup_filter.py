#!/usr/bin/env python
"""Filtra il corpus intermedio (lingua, qualità, lunghezza, boilerplate) e deduplica -> corpus processato."""

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


def _abspath(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def clean(config_path: str, in_path: str | None, out_path: str | None) -> dict:
    """Applica i filtri di pulizia/dedup e scrive il corpus processato; ritorna le statistiche."""
    from italian_llm.data.cleaning import (  # lazy import (modulo puro)
        dedup_exact,
        dedup_near,
        is_italian,
        italian_score,
        length_ok,
        normalize_unicode,
        quality_score,
        remove_boilerplate,
    )

    cfg = load_config(config_path)
    f = get(cfg, "corpus.filters", {}) or {}
    min_chars = int(f.get("min_chars", 200))
    max_chars = int(f.get("max_chars", 20000))
    italian_threshold = float(f.get("italian_threshold", 0.5))
    quality_min = float(f.get("quality_min", 0.35))
    do_dedup_exact = bool(f.get("dedup_exact", True))
    do_dedup_near = bool(f.get("dedup_near", True))
    dedup_threshold = float(f.get("dedup_threshold", 0.9))
    do_boilerplate = bool(f.get("remove_boilerplate", True))
    do_normalize = bool(f.get("normalize_unicode", True))

    in_file = _abspath(
        in_path or get(cfg, "corpus.output.interim_path", "data/interim/corpus.jsonl")
    )
    out_file = _abspath(
        out_path or get(cfg, "corpus.output.processed_path", "data/processed/corpus.jsonl")
    )

    if not os.path.isfile(in_file):
        logger.error(
            "Corpus intermedio non trovato: %s (esegui prima build_italian_corpus.py).", in_file
        )
        return {"input": 0, "kept": 0}

    stats = {
        "input": 0,
        "drop_length": 0,
        "drop_non_italian": 0,
        "drop_low_quality": 0,
        "drop_dedup_exact": 0,
        "drop_dedup_near": 0,
        "kept": 0,
    }

    logger.info("Pulizia corpus: input=%s output=%s", in_file, out_file)
    logger.info(
        "Filtri: min=%d max=%d italian>=%.2f quality>=%.2f near=%.2f",
        min_chars,
        max_chars,
        italian_threshold,
        quality_min,
        dedup_threshold,
    )

    survivors = []
    for rec in read_jsonl(in_file):
        stats["input"] += 1
        text = rec.get("text", "")
        if not isinstance(text, str) or not text.strip():
            stats["drop_length"] += 1
            continue
        if do_normalize:
            text = normalize_unicode(text)
        if do_boilerplate:
            text = remove_boilerplate(text)
        text = text.strip()

        if not length_ok(text, min_chars=min_chars, max_chars=max_chars):
            stats["drop_length"] += 1
            continue
        if not is_italian(text, threshold=italian_threshold):
            stats["drop_non_italian"] += 1
            continue
        q = quality_score(text)
        if q < quality_min:
            stats["drop_low_quality"] += 1
            continue

        rec["text"] = text
        rec["n_chars"] = len(text)
        rec["italian_score"] = round(italian_score(text), 4)
        rec["quality_score"] = round(q, 4)
        survivors.append(rec)

    # Dedup esatto, poi near (entrambi sul testo normalizzato).
    if do_dedup_exact:
        before = len(survivors)
        survivors = dedup_exact(survivors, key=lambda r: r.get("text", ""))
        stats["drop_dedup_exact"] = before - len(survivors)
    if do_dedup_near:
        before = len(survivors)
        survivors = dedup_near(
            survivors, key=lambda r: r.get("text", ""), threshold=dedup_threshold
        )
        stats["drop_dedup_near"] = before - len(survivors)

    stats["kept"] = write_jsonl(out_file, survivors)

    logger.info("=== Pulizia corpus completata ===")
    for k, v in stats.items():
        logger.info("  %-18s : %d", k, v)
    logger.info("  output             : %s", out_file)
    if stats["input"]:
        ratio = 100.0 * stats["kept"] / stats["input"]
        logger.info("  tasso di tenuta    : %.1f%%", ratio)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Filtra e deduplica il corpus intermedio verso data/processed."
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, help="YAML del corpus (default: %(default)s)."
    )
    parser.add_argument(
        "--in", dest="in_path", default=None, help="Corpus intermedio (default da config)."
    )
    parser.add_argument(
        "--out", default=None, help="Corpus processato (default: data/processed/corpus.jsonl)."
    )
    parser.add_argument("--log-level", default="INFO", help="Livello di log (default: INFO).")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    try:
        stats = clean(_abspath(args.config), args.in_path, args.out)
    except Exception as exc:
        logger.error("Errore nella pulizia del corpus: %s", exc)
        return 1
    return 0 if stats.get("kept", 0) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
