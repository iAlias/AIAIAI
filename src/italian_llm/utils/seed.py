"""Seeding riproducibile: stdlib sempre, numpy e torch in lazy import."""

from __future__ import annotations

import os
import random

from italian_llm.logging_utils import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Imposta il seed per random, numpy e torch (se disponibili).

    Lo stdlib `random` e PYTHONHASHSEED vengono sempre impostati. numpy e
    torch sono importati in modo lazy: se non installati, vengono saltati
    senza errori (i moduli puri devono restare importabili senza torch).
    """
    seed = int(seed)

    # Sempre disponibile: stdlib.
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # numpy: opzionale.
    try:
        import numpy as np  # lazy

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy assente o errore minore
        logger.debug("numpy non disponibile: seeding numpy saltato.")

    # torch: opzionale e pesante.
    try:
        import torch  # lazy

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.debug("Seed torch impostato a %d (cuda disponibile: %s).", seed, torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch assente su questo host
        logger.debug("torch non disponibile: seeding torch saltato.")

    logger.info("Seed globale impostato a %d.", seed)
