#!/usr/bin/env python
"""Unisce SFT umani + sintetici, valida lo schema, deduplica, filtra e produce gli split train/valid/test."""

from __future__ import annotations

import os, sys  # noqa: E401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse  # noqa: E402
import glob  # noqa: E402
import random  # noqa: E402

from italian_llm.config import get, load_config  # noqa: E402
from italian_llm.logging_utils import get_logger, setup_logging  # noqa: E402
from italian_llm.utils.io import read_jsonl, write_jsonl  # noqa: E402

logger = get_logger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join("configs", "data", "sft.yaml")


def _abspath(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _sibling(path: str, suffix: str) -> str:
    return os.path.join(os.path.dirname(path), f"sft_{suffix}.jsonl")


def _discover_inputs(processed_dir: str, exclude: set[str]) -> list[str]:
    """Trova i file SFT candidati in data/processed e data/raw (esclusi gli output)."""
    patterns = [
        os.path.join(processed_dir, "*sft*.jsonl"),
        os.path.join(processed_dir, "synth_sft*.jsonl"),
        os.path.join(_abspath("data/raw"), "*sft*.jsonl"),
    ]
    found: list[str] = []
    for pat in patterns:
        for p in glob.glob(pat):
            ap = os.path.abspath(p)
            if ap not in exclude and ap not in found:
                found.append(ap)
    return sorted(found)


def _last_role(messages: list, role: str) -> str:
    """Restituisce il contenuto dell'ultimo messaggio con il ruolo indicato."""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == role:
            return str(m.get("content", ""))
    return ""


def prepare(config_path: str, inputs: list[str] | None, out_train: str | None,
            out_valid: str | None, out_test: str | None, seed: int | None) -> dict:
    """Esegue merge + validazione + filtri + dedup + split; ritorna statistiche."""
    from italian_llm.data.cleaning import dedup_near, is_italian, quality_score  # lazy (puro)
    from italian_llm.data.schema import validate_jsonl, validate_record  # lazy (puro)

    cfg = load_config(config_path)
    seed = int(seed if seed is not None else get(cfg, "sft.output.seed", 42))
    rng = random.Random(seed)

    filt = get(cfg, "sft.filters", {}) or {}
    italian_threshold = float(filt.get("italian_threshold", 0.5))
    quality_min = float(filt.get("quality_min", 0.4))
    min_chars = int(filt.get("min_chars", 20))
    max_chars = int(filt.get("max_chars", 20000))
    do_dedup_near = bool(filt.get("dedup_near", True))
    dedup_threshold = float(filt.get("dedup_threshold", 0.9))
    drop_unsafe = bool(filt.get("drop_unsafe", True))

    train_path = _abspath(out_train or get(cfg, "sft.output.train_path", "data/processed/sft_train.jsonl"))
    valid_path = _abspath(out_valid or get(cfg, "sft.output.valid_path", "data/processed/sft_valid.jsonl"))
    test_path = _abspath(out_test or get(cfg, "sft.output.test_path", _sibling(train_path, "test")))
    valid_ratio = float(get(cfg, "sft.output.valid_ratio", 0.02))
    test_ratio = float(get(cfg, "sft.output.test_ratio", valid_ratio))

    processed_dir = _abspath(get(cfg, "paths.processed_dir", "data/processed"))
    output_set = {os.path.abspath(p) for p in (train_path, valid_path, test_path)}

    # Sorgenti: esplicite (--inputs) oppure auto-discovery escludendo gli output.
    if inputs:
        in_files = [_abspath(p) for p in inputs]
    else:
        in_files = _discover_inputs(processed_dir, output_set)
    in_files = [p for p in in_files if os.path.isfile(p) and os.path.abspath(p) not in output_set]

    if not in_files:
        logger.error("Nessun file SFT di input trovato. Genera prima i dati (create_instruction_dataset / synthesize).")
        return {"total_in": 0, "kept": 0}

    logger.info("Preparazione SFT da %d file: %s", len(in_files), [os.path.basename(p) for p in in_files])

    stats = {
        "total_in": 0,
        "drop_invalid": 0,
        "drop_unsafe": 0,
        "drop_length": 0,
        "drop_non_italian": 0,
        "drop_low_quality": 0,
        "drop_dedup": 0,
        "kept": 0,
    }

    merged: list[dict] = []
    for path in in_files:
        # Report di validazione a livello di file (counts/errors).
        try:
            report = validate_jsonl(path)
            logger.info("  %s -> %s", os.path.basename(path), report)
        except Exception as exc:
            logger.warning("validate_jsonl fallita su %s: %s", path, exc)

        for rec in read_jsonl(path):
            stats["total_in"] += 1
            ok, err = validate_record(rec)
            if not ok:
                stats["drop_invalid"] += 1
                logger.debug("Record invalido scartato (%s): %s", os.path.basename(path), err)
                continue

            messages = rec.get("messages", [])
            assistant = _last_role(messages, "assistant").strip()
            user = _last_role(messages, "user").strip()
            if drop_unsafe and rec.get("safety_tag") == "refuse":
                stats["drop_unsafe"] += 1
                continue
            if len(assistant) < min_chars or len(assistant) > max_chars:
                stats["drop_length"] += 1
                continue
            if not is_italian(assistant, threshold=italian_threshold):
                stats["drop_non_italian"] += 1
                continue
            if quality_score(assistant) < quality_min:
                stats["drop_low_quality"] += 1
                continue

            # Chiave di dedup: testo utente + risposta (ultimo turno).
            rec["_dedup_key"] = (user + " ||| " + assistant)
            merged.append(rec)

    # Dedup near sui contenuti (rimuove varianti quasi identiche tra umani/sintetici).
    if do_dedup_near and merged:
        before = len(merged)
        merged = dedup_near(merged, key=lambda r: r.get("_dedup_key", ""), threshold=dedup_threshold)
        stats["drop_dedup"] = before - len(merged)

    for r in merged:
        r.pop("_dedup_key", None)

    if not merged:
        logger.error("Nessun esempio SFT è sopravvissuto ai filtri.")
        return stats

    # Split deterministico.
    rng.shuffle(merged)
    n = len(merged)
    n_valid = max(1, int(n * valid_ratio)) if n >= 10 else 0
    n_test = max(1, int(n * test_ratio)) if n >= 10 else 0
    valid = merged[:n_valid]
    test = merged[n_valid:n_valid + n_test]
    train = merged[n_valid + n_test:]

    n_tr = write_jsonl(train_path, train)
    n_va = write_jsonl(valid_path, valid)
    n_te = write_jsonl(test_path, test)
    stats["kept"] = n

    logger.info("=== Preparazione SFT completata ===")
    for k, v in stats.items():
        logger.info("  %-18s : %d", k, v)
    logger.info("  train/valid/test   : %d / %d / %d", n_tr, n_va, n_te)
    logger.info("  output             : %s | %s | %s", train_path, valid_path, test_path)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unisce, valida, deduplica e splitta i dati SFT.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML SFT (default: %(default)s).")
    parser.add_argument("--inputs", default=None,
                        help="File JSONL di input separati da virgola (default: auto-discovery).")
    parser.add_argument("--out-train", default=None, help="Output train (default da config).")
    parser.add_argument("--out-valid", default=None, help="Output valid (default da config).")
    parser.add_argument("--out-test", default=None, help="Output test (default: sft_test.jsonl accanto al train).")
    parser.add_argument("--seed", type=int, default=None, help="Seme (default da config).")
    parser.add_argument("--log-level", default="INFO", help="Livello di log (default: INFO).")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    inputs = [p.strip() for p in args.inputs.split(",") if p.strip()] if args.inputs else None
    try:
        stats = prepare(_abspath(args.config), inputs, args.out_train, args.out_valid, args.out_test, args.seed)
    except Exception as exc:
        logger.error("Errore nella preparazione SFT: %s", exc)
        return 1
    return 0 if stats.get("kept", 0) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
