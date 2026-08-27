#!/usr/bin/env python
"""CLI per il Supervised Fine-Tuning (SFT) in QLoRA del modello istruito italiano."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse
import json

from italian_llm.config import load_config
from italian_llm.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def _cast(value: str):
    """Interpreta un valore CLI come JSON (int/float/bool/null/lista), altrimenti stringa."""
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def parse_overrides(pairs: list[str]) -> dict:
    """Converte ['a.b=c', 'x=1'] in {'a': {'b': 'c'}, 'x': 1} per il deep-merge della config."""
    out: dict = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--set richiede il formato CHIAVE=VALORE, ricevuto: {item!r}")
        key, _, raw = item.partition("=")
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _cast(raw.strip())
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Avvia il Supervised Fine-Tuning (SFT) in QLoRA su dati istruzione italiani.",
    )
    p.add_argument(
        "--config",
        default="configs/train/sft_qwen9b_lora.yaml",
        help="Percorso YAML della configurazione SFT (default: %(default)s).",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="CHIAVE=VALORE",
        help="Override puntato della config, ripetibile (es. --set train.epochs=1).",
    )
    p.add_argument("--log-level", default="INFO", help="Livello di logging (default: INFO).")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Risolve e stampa la config senza avviare il training.",
    )
    return p


def main(argv: list[str] | None = None) -> str | None:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    overrides = parse_overrides(args.overrides)
    cfg = load_config(args.config, overrides=overrides or None)
    logger.info("Config SFT caricata da '%s'.", args.config)

    if args.dry_run:
        logger.info("Modalita' dry-run: configurazione risolta di seguito.")
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return None

    from italian_llm.training.sft import run_sft

    logger.info("Avvio SFT...")
    out_dir = run_sft(cfg)
    logger.info("SFT completato. Adapter/artefatti in: %s", out_dir)
    print(out_dir)
    return out_dir


if __name__ == "__main__":
    main()
