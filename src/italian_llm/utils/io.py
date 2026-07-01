"""Lettura/scrittura JSONL tollerante agli errori e helper per le directory."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator

from italian_llm.logging_utils import get_logger

logger = get_logger(__name__)


def ensure_dir(path: str) -> str:
    """Crea la directory indicata (e i genitori) se non esiste, e la ritorna.

    Accetta sia un percorso di directory sia un percorso di file: in
    quest'ultimo caso crea la directory contenitrice. Idempotente.
    """
    # Euristica: se il path ha un'estensione lo trattiamo come file e
    # creiamo la cartella padre; altrimenti lo trattiamo come directory.
    _, ext = os.path.splitext(path)
    target = os.path.dirname(path) if ext else path
    if target:
        os.makedirs(target, exist_ok=True)
    return target


def read_jsonl(path: str) -> Iterator[dict]:
    """Generatore che produce un dict per ogni riga JSON valida del file.

    Le righe vuote vengono ignorate silenziosamente; le righe malformate
    vengono saltate con un warning, senza interrompere la lettura. Encoding
    sempre UTF-8.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File JSONL non trovato: '{path}'.")

    skipped = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped += 1
                logger.warning("Riga %d di '%s' non valida, saltata: %s", lineno, path, exc)
                continue
            if not isinstance(obj, dict):
                skipped += 1
                logger.warning("Riga %d di '%s' non e' un oggetto JSON, saltata.", lineno, path)
                continue
            yield obj

    if skipped:
        logger.info("Lettura di '%s' completata con %d righe saltate.", path, skipped)


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    """Scrive un iterabile di dict come JSONL UTF-8, una riga per record.

    Crea le directory mancanti. Restituisce il numero di record scritti.
    `ensure_ascii=False` per preservare gli accenti italiani in chiaro.
    """
    ensure_dir(path)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            if not isinstance(row, dict):
                logger.warning("Record non-dict ignorato in scrittura su '%s': %r", path, type(row))
                continue
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
            count += 1
    logger.debug("Scritti %d record JSONL in '%s'.", count, path)
    return count
