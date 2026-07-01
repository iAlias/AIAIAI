"""Fill-in-the-middle (FIM) per Qwen2.5-Coder: completamento da editor.

Dai codice prima (prefix) e dopo (suffix) il cursore; il modello riempie il mezzo.
Feature locale che i chatbot cloud non offrono bene. Qui solo la logica pura dei
token; la generazione raw vive in scripts/fim_complete.py (transformers lazy).
"""

import re

__all__ = ["build_fim_prompt", "strip_fim", "FIM_PREFIX", "FIM_SUFFIX", "FIM_MIDDLE"]

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"

_SPECIAL_RE = re.compile(
    r"<\|(endoftext|fim_pad|fim_prefix|fim_suffix|fim_middle|im_end|im_start)\|>"
)


def build_fim_prompt(prefix: str, suffix: str) -> str:
    """Prompt FIM nel formato Qwen2.5-Coder."""
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"


def strip_fim(text: str) -> str:
    """Taglia l'output al primo token speciale e rimuove residui."""
    if not text:
        return ""
    m = _SPECIAL_RE.search(text)
    if m:
        text = text[: m.start()]
    return text
