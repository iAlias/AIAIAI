"""Pacchetto di training: CPT, SFT e allineamento alle preferenze (ORPO/DPO)."""

from __future__ import annotations

import importlib
from typing import Any

# Mappa simbolo pubblico -> modulo che lo definisce. Il caricamento e' pigro
# (PEP 562) cosi' "import italian_llm.training" non tira dentro torch/transformers
# ne' forza l'import dei sotto-moduli finche' non servono davvero.
_EXPORTS = {
    "run_cpt": "italian_llm.training.cpt",
    "run_sft": "italian_llm.training.sft",
    "run_preference": "italian_llm.training.preference",
    "load_model_and_tokenizer": "italian_llm.training.common",
    "apply_lora": "italian_llm.training.common",
    "build_lora_config": "italian_llm.training.common",
    "get_bnb_config": "italian_llm.training.common",
    "count_trainable_params": "italian_llm.training.common",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Risolve pigramente i simboli pubblici importando il modulo giusto."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'italian_llm.training' has no attribute '{name}'")
    module = importlib.import_module(target)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
