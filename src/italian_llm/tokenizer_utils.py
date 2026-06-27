"""Caricamento tokenizer (lazy transformers) e formattazione chat con fallback."""

from __future__ import annotations

from typing import Any

from italian_llm.logging_utils import get_logger

logger = get_logger(__name__)


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = True) -> Any:
    """Carica un tokenizer HuggingFace, importando transformers in modo lazy.

    Garantisce la presenza di un pad_token: se mancante, riusa l'eos_token
    (pratica standard per i modelli causali tipo Qwen). L'import e' interno
    alla funzione cosi' il modulo resta importabile senza transformers.
    """
    try:
        from transformers import AutoTokenizer  # lazy import pesante
    except ImportError as exc:  # pragma: no cover - dipende dall'ambiente
        raise ImportError(
            "transformers non e' installato: impossibile caricare il tokenizer. "
            "Installa le dipendenze heavy (vedi requirements/extra)."
        ) from exc

    logger.info("Carico tokenizer da '%s'.", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    # Molti modelli causali non definiscono un pad_token: lo impostiamo
    # all'eos_token per abilitare il padding nei batch di training/inferenza.
    if getattr(tokenizer, "pad_token", None) is None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos is not None:
            tokenizer.pad_token = eos
            logger.info("pad_token mancante: impostato uguale a eos_token ('%s').", eos)
        else:
            # Fallback estremo: aggiungiamo un token di padding dedicato.
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            logger.warning("Ne' pad_token ne' eos_token presenti: aggiunto '<|pad|>'.")

    return tokenizer


def format_chat(tokenizer: Any, messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Formatta una lista di messaggi in un prompt testuale.

    Usa il chat_template del tokenizer quando disponibile; in assenza (o in
    caso di errore) ricade su italian_llm.data.prompts.render_plain, che
    produce un formato <|system|>/<|user|>/<|assistant|> coerente.
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception as exc:  # template presente ma incompatibile
            logger.warning("apply_chat_template fallito (%s): uso render_plain.", exc)

    # Fallback puro-Python, importato lazy per evitare cicli a import-time.
    from italian_llm.data.prompts import render_plain

    return render_plain(messages, add_generation_prompt=add_generation_prompt)
