#!/usr/bin/env python
"""Scarica (lazy) alcune sorgenti italiane aperte da HuggingFace; offline genera un corpus sintetico di bootstrap."""

from __future__ import annotations

import os, sys  # noqa: E401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse  # noqa: E402

from italian_llm.config import get, load_config  # noqa: E402
from italian_llm.logging_utils import get_logger, setup_logging  # noqa: E402
from italian_llm.utils.io import write_jsonl  # noqa: E402
from italian_llm.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join("configs", "data", "corpus.yaml")

# Argomenti tipici di HF per le sorgenti note (usati come default se non in config).
_HF_DATASET_ARGS = {
    "wikimedia/wikipedia": {"name": "20231101.it"},
    "oscar-corpus/OSCAR-2301": {"language": "it"},
}

# Argomenti di "seme" per il fallback sintetico: argomenti italiani neutri e leciti.
_BOOTSTRAP_TOPICS = [
    "la storia delle città italiane",
    "come funziona la posta elettronica certificata",
    "una ricetta tradizionale della cucina regionale",
    "consigli pratici per organizzare il lavoro d'ufficio",
    "le basi della programmazione in Python",
    "il funzionamento dei trasporti pubblici locali",
    "come scrivere una email formale efficace",
    "principi di base della fotografia digitale",
    "la gestione del bilancio familiare",
    "il ciclo dell'acqua e l'ambiente",
    "buone pratiche per l'assistenza clienti",
    "una guida introduttiva al giardinaggio",
]


def _abspath(path: str) -> str:
    """Risolve un percorso relativo rispetto alla root del repo."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _download_hf_source(source: dict, raw_dir: str, max_docs: int) -> int:
    """Tenta lo streaming di un dataset HF e scrive righe {text,...}; ritorna il conteggio.

    Tutto in lazy import: se 'datasets' manca o non c'è rete, solleva e il chiamante
    registra un warning e prosegue col fallback.
    """
    from datasets import load_dataset  # lazy: heavy/network

    name = source["name"]
    path = source.get("path", "")
    split = source.get("split", "train")
    text_field = source.get("text_field", "text")

    kwargs = dict(_HF_DATASET_ARGS.get(path, {}))
    if source.get("config"):
        # Una sorgente può specificare esplicitamente il config HF.
        kwargs["name"] = source["config"]

    logger.info("Sorgente '%s': load_dataset(%s, %s, split=%s, streaming=True)", name, path, kwargs, split)
    ds = load_dataset(path, split=split, streaming=True, **kwargs)

    out_path = os.path.join(raw_dir, f"{name}.jsonl")
    rows = []
    for i, ex in enumerate(ds):
        if i >= max_docs:
            break
        text = (ex.get(text_field) or "").strip()
        if not text:
            continue
        rows.append({"id": f"{name}-{i}", "text": text, "source": name, "domain": "general"})
    written = write_jsonl(out_path, rows)
    logger.info("Sorgente '%s': scritti %d documenti in %s", name, written, out_path)
    return written


def _generate_synthetic_corpus(raw_dir: str, n_docs: int, seed: int) -> int:
    """Genera un piccolo corpus sintetico italiano col mock teacher (100% offline)."""
    import random

    from italian_llm.data.prompts import SYSTEM_DEFAULT, build_messages
    from italian_llm.data.synthetic import get_teacher

    rng = random.Random(seed)
    teacher = get_teacher("mock")  # deterministico, nessuna rete
    logger.info("Generazione corpus sintetico di bootstrap con teacher '%s' (%d documenti).", teacher.name, n_docs)

    rows = []
    for i in range(n_docs):
        topic = _BOOTSTRAP_TOPICS[i % len(_BOOTSTRAP_TOPICS)]
        # Piccole variazioni per evitare documenti identici.
        flavor = rng.choice(["in modo chiaro e ordinato", "con esempi concreti", "per un lettore non esperto"])
        user = f"Scrivi un breve testo informativo in italiano su {topic}, {flavor}."
        messages = build_messages(SYSTEM_DEFAULT, user)
        try:
            text = teacher.generate(messages, max_new_tokens=400, temperature=0.7).strip()
        except Exception as exc:  # il mock non dovrebbe fallire, ma restiamo robusti
            logger.warning("Generazione sintetica fallita per il doc %d: %s", i, exc)
            text = f"Testo informativo in italiano su {topic}. {flavor.capitalize()}."
        rows.append({"id": f"synthetic-{i}", "text": text, "source": "synthetic_fallback", "domain": "general"})

    out_path = os.path.join(raw_dir, "synthetic_bootstrap.jsonl")
    written = write_jsonl(out_path, rows)
    logger.info("Corpus sintetico: scritti %d documenti in %s", written, out_path)
    return written


def download(config_path: str, out_dir: str | None, max_docs: int, synthetic_docs: int,
             force_synthetic: bool, seed: int) -> int:
    """Esegue il download delle sorgenti abilitate; in caso di necessità genera il fallback."""
    cfg = load_config(config_path)
    set_seed(seed)

    raw_dir = _abspath(out_dir or get(cfg, "paths.data_dir", "data/raw"))
    os.makedirs(raw_dir, exist_ok=True)

    sources = get(cfg, "corpus.sources", []) or []
    total = 0
    n_ok = 0
    for source in sources:
        if not source.get("enabled", True):
            logger.info("Sorgente '%s' disabilitata: salto.", source.get("name"))
            continue
        stype = source.get("type", "huggingface")
        try:
            if stype == "huggingface":
                got = _download_hf_source(source, raw_dir, max_docs)
            elif stype == "synthetic":
                # Gestita esplicitamente più sotto come fallback unico.
                logger.info("Sorgente sintetica '%s': verrà generata dal fallback.", source.get("name"))
                continue
            else:
                logger.info("Sorgente '%s' di tipo '%s' non scaricabile qui: salto.", source.get("name"), stype)
                continue
        except Exception as exc:
            logger.warning("Download della sorgente '%s' fallito (%s): proseguo col fallback.",
                           source.get("name"), exc)
            continue
        total += got
        if got > 0:
            n_ok += 1

    # Fallback sintetico: se nulla è stato scaricato o se richiesto esplicitamente.
    if force_synthetic or total == 0:
        if total == 0:
            logger.warning("Nessun dato reale scaricato: genero un corpus sintetico di bootstrap.")
        total += _generate_synthetic_corpus(raw_dir, synthetic_docs, seed)
    else:
        logger.info("Scaricati dati reali da %d sorgenti (%d documenti).", n_ok, total)

    logger.info("=== Download completato: %d documenti totali in %s ===", total, raw_dir)
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scarica sorgenti italiane aperte o genera un corpus sintetico.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML del corpus (default: %(default)s).")
    parser.add_argument("--out", default=None, help="Directory di output (default: paths.data_dir o data/raw).")
    parser.add_argument("--max-docs", type=int, default=2000, help="Max documenti per sorgente HF.")
    parser.add_argument("--synthetic-docs", type=int, default=120, help="Documenti del fallback sintetico.")
    parser.add_argument("--force-synthetic", action="store_true", help="Genera sempre anche il fallback sintetico.")
    parser.add_argument("--seed", type=int, default=42, help="Seme per la generazione sintetica.")
    parser.add_argument("--log-level", default="INFO", help="Livello di log (default: INFO).")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    config_path = _abspath(args.config)
    try:
        n = download(config_path, args.out, args.max_docs, args.synthetic_docs, args.force_synthetic, args.seed)
    except Exception as exc:
        logger.error("Errore irreversibile nel download dati: %s", exc)
        return 1
    return 0 if n > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
