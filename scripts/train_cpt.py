#!/usr/bin/env python
"""CLI per il continued pre-training (CPT) del modello base Qwen su corpus italiano."""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import argparse
import json

from italian_llm.config import load_config
from italian_llm.logging_utils import setup_logging, get_logger

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
        description="Avvia il continued pre-training (CPT) sul corpus italiano grezzo/sintetico.",
    )
    p.add_argument(
        "--config",
        default="configs/train/cpt_qwen9b.yaml",
        help="Percorso YAML della configurazione CPT (default: %(default)s).",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="CHIAVE=VALORE",
        help="Override puntato della config, ripetibile (es. --set train.lr=1e-4).",
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
    logger.info("Config CPT caricata da '%s'.", args.config)

    if args.dry_run:
        logger.info("Modalita' dry-run: configurazione risolta di seguito.")
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return None

    # Import pesante (torch/transformers/trl) ritardato fino all'avvio effettivo.
    from italian_llm.training.cpt import run_cpt

    logger.info("Avvio CPT...")
    out_dir = run_cpt(cfg)
    logger.info("CPT completato. Artefatti in: %s", out_dir)
    print(out_dir)
    return out_dir


if __name__ == "__main__":
    main()
