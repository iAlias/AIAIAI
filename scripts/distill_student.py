#!/usr/bin/env python
"""CLI di distillazione verso lo student 3B: modalita' 'data' (default) o 'logits' (sperimentale)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse
import json

from italian_llm.config import get, load_config
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
        description="Distilla il teacher allineato verso lo student compatto (data o logits).",
    )
    p.add_argument(
        "--config",
        default="configs/distill/student_3b.yaml",
        help="Percorso YAML della configurazione di distillazione (default: %(default)s).",
    )
    p.add_argument(
        "--mode",
        choices=["data", "logits"],
        default=None,
        help="Modalita' di distillazione; sovrascrive distill.mode nella config.",
    )
    p.add_argument(
        "--no-train",
        action="store_true",
        help="In modalita' 'data' costruisce solo il dataset, senza avviare l'SFT dello student.",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="CHIAVE=VALORE",
        help="Override puntato della config, ripetibile (es. --set distill.temperature=1.5).",
    )
    p.add_argument("--log-level", default="INFO", help="Livello di logging (default: INFO).")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Risolve e stampa la config senza eseguire la distillazione.",
    )
    return p


def _run_data_distill(cfg: dict, no_train: bool) -> str:
    """Costruisce il dataset distillato e (salvo --no-train) avvia l'SFT dello student."""
    from italian_llm.distillation.data_distill import build_distill_dataset

    logger.info("Costruzione dataset di distillazione (risposte del teacher)...")
    distill_path = build_distill_dataset(cfg)
    logger.info("Dataset distillato pronto: %s", distill_path)

    if no_train:
        logger.info("Flag --no-train attivo: salto l'SFT dello student.")
        print(distill_path)
        return distill_path

    # Costruisce una config SFT per lo student riusando run_sft:
    # - il modello da addestrare e' lo student
    # - i dati di training sono il dataset distillato appena prodotto
    sft_cfg = dict(cfg)
    sft_cfg["model"] = {
        "name": get(cfg, "student.name", get(cfg, "model.name")),
        "dtype": get(cfg, "student.dtype", "bfloat16"),
        "trust_remote_code": get(cfg, "student.trust_remote_code", True),
        "quantization": get(cfg, "student.quantization", {}),
    }
    data_cfg = dict(get(cfg, "data", {}) or {})
    data_cfg["train_path"] = distill_path
    data_cfg.setdefault("valid_path", get(cfg, "data.valid_path", ""))
    data_cfg["format"] = get(cfg, "data.format", "chat")
    sft_cfg["data"] = data_cfg

    from italian_llm.training.sft import run_sft

    logger.info("Avvio SFT dello student sul dataset distillato...")
    out_dir = run_sft(sft_cfg)
    logger.info("Distillazione 'data' completata. Student in: %s", out_dir)
    print(out_dir)
    return out_dir


def _run_logits_distill(cfg: dict) -> str:
    """Esegue la distillazione su logits (KL), sperimentale e con limiti documentati."""
    from italian_llm.distillation.logits_distill import run_logits_distill

    logger.warning(
        "Distillazione 'logits' sperimentale: richiede teacher e student con stesso tokenizer; "
        "vedi i limiti documentati nel modulo logits_distill."
    )
    out_dir = run_logits_distill(cfg)
    logger.info("Distillazione 'logits' completata. Student in: %s", out_dir)
    print(out_dir)
    return out_dir


def main(argv: list[str] | None = None) -> str | None:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    overrides = parse_overrides(args.overrides)
    if args.mode is not None:
        overrides.setdefault("distill", {})["mode"] = args.mode

    cfg = load_config(args.config, overrides=overrides or None)
    mode = get(cfg, "distill.mode", "data")
    logger.info("Config distillazione caricata da '%s' (modalita': %s).", args.config, mode)

    if args.dry_run:
        logger.info("Modalita' dry-run: configurazione risolta di seguito.")
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return None

    if mode == "logits":
        return _run_logits_distill(cfg)
    return _run_data_distill(cfg, no_train=args.no_train)


if __name__ == "__main__":
    main()
