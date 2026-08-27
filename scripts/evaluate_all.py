#!/usr/bin/env python
"""CLI di valutazione: esegue il runner, stampa un riepilogo leggibile e salva il report JSON."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse
import json

from italian_llm.config import get, load_config
from italian_llm.logging_utils import get_logger, setup_logging
from italian_llm.utils.io import ensure_dir

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
        description="Valuta un modello/adapter sul set italiano e produce un report aggregato.",
    )
    p.add_argument(
        "--config",
        default="configs/eval/eval.yaml",
        help="Percorso YAML della configurazione di valutazione (default: %(default)s).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Percorso del report JSON (default: eval.output.report_path della config).",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="CHIAVE=VALORE",
        help="Override puntato della config, ripetibile (es. --set eval.batch_size=1).",
    )
    p.add_argument("--log-level", default="INFO", help="Livello di logging (default: INFO).")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Risolve e stampa la config senza eseguire la valutazione.",
    )
    return p


def _print_summary(report: dict) -> None:
    """Stampa un riepilogo tabellare delle metriche numeriche del report."""
    # Le metriche aggregate possono stare al top-level o sotto la chiave 'metrics'.
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else report
    print("\n=== Riepilogo valutazione ===")
    rows = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append((key, f"{value:.4f}"))
        elif isinstance(value, str):
            rows.append((key, value))
    if not rows:
        print("(nessuna metrica numerica nel report)")
        return
    width = max(len(name) for name, _ in rows)
    for name, val in rows:
        print(f"  {name.ljust(width)} : {val}")
    print("=============================\n")


def main(argv: list[str] | None = None) -> dict | None:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    overrides = parse_overrides(args.overrides)
    cfg = load_config(args.config, overrides=overrides or None)
    logger.info("Config valutazione caricata da '%s'.", args.config)

    if args.dry_run:
        logger.info("Modalita' dry-run: configurazione risolta di seguito.")
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return None

    from italian_llm.evaluation.runner import run_eval

    logger.info("Avvio valutazione...")
    report = run_eval(cfg)

    out_path = args.out or get(cfg, "eval.output.report_path", "outputs/eval/report.json")
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        ensure_dir(parent)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info("Report salvato in: %s", out_path)

    _print_summary(report)
    return report


if __name__ == "__main__":
    main()
