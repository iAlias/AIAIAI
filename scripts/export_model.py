#!/usr/bin/env python
"""CLI per fondere un adapter LoRA nel modello base e salvare un modello HF unificato."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse

from italian_llm.logging_utils import get_logger, setup_logging
from italian_llm.utils.io import ensure_dir

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fonde (merge) un adapter LoRA nel modello base e salva i pesi HF risultanti.",
    )
    p.add_argument("--base", required=True, help="Modello base HF (nome o percorso locale).")
    p.add_argument(
        "--adapter", required=True, help="Directory dell'adapter LoRA (PEFT) da fondere."
    )
    p.add_argument("--out", required=True, help="Directory di output per il modello fuso.")
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Precisione con cui caricare/salvare i pesi (default: float16).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device per il merge (cpu consigliato per modelli grandi; 'cuda'/'auto' opzionali).",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Abilita trust_remote_code nel caricamento (necessario per alcuni modelli).",
    )
    p.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Dimensione massima per shard nel salvataggio safetensors (default: 5GB).",
    )
    p.add_argument("--log-level", default="INFO", help="Livello di logging (default: INFO).")
    return p


def main(argv: list[str] | None = None) -> str:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    # Import pesanti (e opzionali) ritardati: errore chiaro se mancano.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Dipendenze mancanti per l'export: installa 'torch' e 'transformers' "
            "(pip install torch transformers)."
        ) from exc
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit(
            "Dipendenza mancante per l'export: installa 'peft' (pip install peft)."
        ) from exc

    if not os.path.isdir(args.adapter):
        raise SystemExit(f"Directory adapter non trovata: '{args.adapter}'.")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        args.dtype
    ]

    logger.info(
        "Carico modello base '%s' (dtype=%s, device=%s)...", args.base, args.dtype, args.device
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if args.device not in ("auto",):
        model = model.to(args.device)

    logger.info("Applico l'adapter LoRA da '%s'...", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)

    logger.info("Fondo l'adapter nei pesi base (merge_and_unload)...")
    model = model.merge_and_unload()

    ensure_dir(args.out)
    logger.info("Salvo il modello fuso in '%s'...", args.out)
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size=args.max_shard_size)

    # Salva anche il tokenizer per ottenere una cartella self-contained.
    try:
        tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=args.trust_remote_code)
        tok.save_pretrained(args.out)
        logger.info("Tokenizer salvato accanto al modello fuso.")
    except Exception as exc:  # tokenizer assente non e' fatale per i pesi
        logger.warning(
            "Impossibile salvare il tokenizer (%s). Copialo manualmente se necessario.", exc
        )

    logger.info("Export completato: %s", args.out)
    print(args.out)
    return args.out


if __name__ == "__main__":
    main()
