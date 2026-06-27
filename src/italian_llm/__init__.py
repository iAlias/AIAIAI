"""Pacchetto principale dell'LLM italiano specializzato (italian_llm)."""

# La versione del pacchetto. Usata da pyproject e per logging diagnostico.
__version__ = "0.1.0"

# Re-export comodo: get_logger e' usato praticamente ovunque negli script.
# logging_utils importa solo stdlib, quindi questo re-export e' sicuro anche
# senza torch installato.
from italian_llm.logging_utils import get_logger

__all__ = ["__version__", "get_logger"]
