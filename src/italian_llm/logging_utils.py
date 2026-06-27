"""Utility di logging centralizzate: setup idempotente + factory get_logger."""

from __future__ import annotations

import logging
import os
import sys

# Formato pulito e leggibile, in stile training-loop minimale ma informativo.
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Flag interno per rendere setup_logging idempotente: chiamarlo piu' volte
# non deve aggiungere handler duplicati (che causerebbero righe ripetute).
_CONFIGURED = False


def _coerce_level(level: str | int) -> int:
    """Converte un livello (stringa o int) in un intero logging valido."""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        # Permetti sia "INFO" sia "info" sia un numero come stringa.
        named = logging.getLevelName(level.upper())
        if isinstance(named, int):
            return named
        try:
            return int(level)
        except (TypeError, ValueError):
            return logging.INFO
    return logging.INFO


def setup_logging(level: str = "INFO") -> None:
    """Configura il root logger una sola volta con un handler su stderr.

    Idempotente: chiamate ripetute aggiornano solo il livello, senza
    aggiungere handler duplicati. Il livello puo' essere sovrascritto via
    variabile d'ambiente ITALIAN_LLM_LOG_LEVEL.
    """
    global _CONFIGURED

    env_level = os.environ.get("ITALIAN_LLM_LOG_LEVEL")
    effective_level = _coerce_level(env_level if env_level else level)

    root = logging.getLogger()

    if _CONFIGURED:
        # Gia' configurato: aggiorna solo il livello e usciamo.
        root.setLevel(effective_level)
        for handler in root.handlers:
            handler.setLevel(effective_level)
        return

    # Rimuovi eventuali handler preesistenti (es. configurati da librerie)
    # per garantire un output coerente e un singolo stream.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    handler.setLevel(effective_level)

    root.addHandler(handler)
    root.setLevel(effective_level)

    # Riduci il rumore delle librerie heavy piu' chiacchierone.
    for noisy in ("urllib3", "filelock", "huggingface_hub", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Restituisce un logger nominale, garantendo il setup di base.

    Se setup_logging non e' ancora stato chiamato, lo invochiamo con i
    default cosi' i moduli/script funzionano subito senza boilerplate.
    """
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
