"""Componenti condivisi del training: caricamento modello/tokenizer, QLoRA e LoRA."""

from __future__ import annotations

import os
from typing import Any

from italian_llm.config import get
from italian_llm.logging_utils import get_logger

logger = get_logger(__name__)

# Messaggio unico mostrato quando manca lo stack pesante (torch/transformers/...).
# Le dipendenze sono importate SOLO dentro le funzioni: i moduli restano
# importabili anche su questa macchina Windows senza GPU/torch installati.
_INSTALL_HINT = (
    "Dipendenze di training mancanti o non importabili.\n"
    "Installa i requisiti del progetto con:\n"
    "    pip install -r requirements.txt\n"
    "oppure, come minimo:\n"
    "    pip install torch transformers peft accelerate datasets trl bitsandbytes"
)

# Moduli LoRA di default per i modelli della famiglia Qwen2.5 (attn + MLP).
# Usati se cfg['lora']['target_modules'] non e' specificato.
_DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

__all__ = [
    "require_torch",
    "require_transformers",
    "require_peft",
    "require_trl",
    "require_datasets",
    "resolve_dtype",
    "get_bnb_config",
    "load_model_and_tokenizer",
    "build_lora_config",
    "apply_lora",
    "count_trainable_params",
    "set_gradient_checkpointing",
    "save_artifacts",
    "is_quantized",
    "select_precision",
    "select_optimizer",
    "filter_kwargs",
    "resolve_output_dir",
]


# --------------------------------------------------------------------------- #
# Import pigri con messaggio d'errore chiaro
# --------------------------------------------------------------------------- #
def require_torch():
    """Importa torch o solleva un RuntimeError con istruzioni d'installazione."""
    try:
        import torch  # noqa: PLC0415

        return torch
    except ImportError as exc:  # pragma: no cover - dipende dall'ambiente
        raise RuntimeError(_INSTALL_HINT) from exc


def require_transformers():
    """Importa il modulo transformers o solleva un RuntimeError chiaro."""
    try:
        import transformers  # noqa: PLC0415

        return transformers
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(_INSTALL_HINT) from exc


def require_peft():
    """Importa peft o solleva un RuntimeError chiaro."""
    try:
        import peft  # noqa: PLC0415

        return peft
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(_INSTALL_HINT) from exc


def require_trl():
    """Importa trl o solleva un RuntimeError chiaro."""
    try:
        import trl  # noqa: PLC0415

        return trl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(_INSTALL_HINT) from exc


def require_datasets():
    """Importa datasets (HuggingFace) o solleva un RuntimeError chiaro."""
    try:
        import datasets  # noqa: PLC0415

        return datasets
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(_INSTALL_HINT) from exc


# --------------------------------------------------------------------------- #
# Helper su dtype / quantizzazione / precisione
# --------------------------------------------------------------------------- #
def resolve_dtype(name: Any, torch_mod) -> Any:
    """Converte un nome di dtype (stringa) nel corrispondente torch.dtype.

    Restituisce la stringa "auto" se richiesto, e bfloat16 come fallback
    prudente per nomi non riconosciuti.
    """
    if name is None:
        return "auto"
    key = str(name).strip().lower()
    mapping = {
        "float32": torch_mod.float32,
        "fp32": torch_mod.float32,
        "float": torch_mod.float32,
        "float16": torch_mod.float16,
        "fp16": torch_mod.float16,
        "half": torch_mod.float16,
        "bfloat16": torch_mod.bfloat16,
        "bf16": torch_mod.bfloat16,
        "auto": "auto",
    }
    return mapping.get(key, torch_mod.bfloat16)


def is_quantized(cfg: dict) -> bool:
    """True se la config richiede il caricamento in 4bit (QLoRA)."""
    return bool(get(cfg, "model.quantization.load_in_4bit", False))


def get_bnb_config(cfg: dict):
    """Costruisce un BitsAndBytesConfig per QLoRA, o None se non richiesto.

    Legge model.quantization.* dalla config. Avvisa (senza bloccare) se la
    libreria bitsandbytes non e' importabile sulla macchina corrente: l'errore
    vero emergera' solo a runtime quando si tenta davvero il caricamento 4bit.
    """
    quant = get(cfg, "model.quantization", {}) or {}
    if not quant.get("load_in_4bit", False):
        return None

    torch = require_torch()
    require_transformers()
    try:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(_INSTALL_HINT) from exc

    try:
        import bitsandbytes  # noqa: F401,PLC0415
    except Exception:  # pragma: no cover - tipico su Windows/CPU
        logger.warning(
            "bitsandbytes non disponibile: il caricamento in 4bit fallira' "
            "a runtime su questa macchina (serve Linux + CUDA)."
        )

    compute_dtype = resolve_dtype(quant.get("bnb_4bit_compute_dtype", "bfloat16"), torch)
    if compute_dtype == "auto":
        compute_dtype = torch.bfloat16  # bnb richiede un dtype concreto

    logger.info(
        "Config quantizzazione 4bit: quant_type=%s compute_dtype=%s double_quant=%s",
        quant.get("bnb_4bit_quant_type", "nf4"),
        compute_dtype,
        quant.get("bnb_4bit_use_double_quant", True),
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(quant.get("bnb_4bit_use_double_quant", True)),
    )


def select_precision(cfg: dict, torch_mod) -> tuple[bool, bool]:
    """Decide i flag (bf16, fp16) per il training in base a dtype e hardware.

    Su CPU (es. questa macchina di build) entrambi restano False -> fp32.
    """
    dtype_name = str(get(cfg, "model.dtype", "bfloat16")).lower()
    if not torch_mod.cuda.is_available():
        return False, False
    bf16_ok = bool(getattr(torch_mod.cuda, "is_bf16_supported", lambda: False)())
    if dtype_name in ("bfloat16", "bf16"):
        return (True, False) if bf16_ok else (False, True)
    if dtype_name in ("float16", "fp16", "half"):
        return False, True
    return False, False


def select_optimizer(cfg: dict) -> str:
    """Sceglie l'optimizer: paged AdamW 8bit con QLoRA (se bnb c'e'), altrimenti adamw_torch."""
    if is_quantized(cfg):
        try:
            import bitsandbytes  # noqa: F401,PLC0415

            return "paged_adamw_8bit"
        except Exception:
            return "adamw_torch"
    return "adamw_torch"


def filter_kwargs(cls, desired: dict) -> dict:
    """Mantiene solo le chiavi accettate dall'__init__ di `cls`.

    Serve a essere robusti tra versioni diverse di transformers/trl, dove alcuni
    parametri vengono rinominati (es. evaluation_strategy -> eval_strategy).
    """
    import inspect  # noqa: PLC0415

    try:
        params = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):  # pragma: no cover
        return dict(desired)
    return {k: v for k, v in desired.items() if k in params}


# --------------------------------------------------------------------------- #
# Caricamento modello + tokenizer
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(cfg: dict):
    """Carica modello causale e tokenizer onorando dtype e quantizzazione 4bit.

    Ritorna (model, tokenizer). Import pesanti tutti pigri qui dentro.
    """
    torch = require_torch()
    require_transformers()
    try:
        from transformers import AutoModelForCausalLM  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(_INSTALL_HINT) from exc

    from italian_llm.tokenizer_utils import load_tokenizer  # noqa: PLC0415

    model_name = get(cfg, "model.name")
    if not model_name:
        raise ValueError("Config priva di 'model.name': impossibile caricare il modello.")

    dtype = resolve_dtype(get(cfg, "model.dtype", "bfloat16"), torch)
    trust = bool(get(cfg, "model.trust_remote_code", False))
    bnb_config = get_bnb_config(cfg)

    kwargs: dict = {"trust_remote_code": trust, "torch_dtype": dtype}
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
        # Con i pesi in 4bit serve un device_map per il placement (default "auto").
        kwargs["device_map"] = get(cfg, "model.device_map", "auto")
    else:
        device_map = get(cfg, "model.device_map", None)
        if device_map:
            kwargs["device_map"] = device_map

    logger.info(
        "Carico il modello base '%s' (dtype=%s, 4bit=%s, trust_remote_code=%s)",
        model_name,
        dtype,
        bnb_config is not None,
        trust,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    tokenizer = load_tokenizer(model_name)
    # Allinea il pad token sul modello (il tokenizer_utils garantisce un pad token).
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None and hasattr(model, "config"):
        model.config.pad_token_id = pad_id

    return model, tokenizer


# --------------------------------------------------------------------------- #
# LoRA / QLoRA
# --------------------------------------------------------------------------- #
def build_lora_config(cfg: dict):
    """Crea un peft.LoraConfig (CAUSAL_LM) dai parametri cfg['lora']."""
    try:
        from peft import LoraConfig, TaskType  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(_INSTALL_HINT) from exc

    lora_cfg = get(cfg, "lora", {}) or {}
    target_modules = lora_cfg.get("target_modules") or _DEFAULT_TARGET_MODULES
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(target_modules),
        bias="none",
    )


def apply_lora(model, cfg: dict):
    """Avvolge il modello con adattatori LoRA, preparando il path k-bit se 4bit.

    Restituisce il modello PEFT. La gradient checkpointing NON viene attivata qui
    (la gestiscono i Trainer/loop a valle), ma assicuriamo input_require_grads
    cosi' che il checkpointing funzioni anche con i moduli LoRA.
    """
    try:
        from peft import get_peft_model, prepare_model_for_kbit_training  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(_INSTALL_HINT) from exc

    quantized = bool(getattr(model, "is_loaded_in_4bit", False)) or bool(
        getattr(model, "is_loaded_in_8bit", False)
    )
    if quantized:
        # Stabilizza il training in 4bit (layernorm in fp32, output in fp32, ecc.).
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    peft_config = build_lora_config(cfg)
    model = get_peft_model(model, peft_config)

    # Necessario perche' il gradient checkpointing propaghi i gradienti agli input
    # quando solo gli adapter sono allenabili.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable, total = count_trainable_params(model)
    pct = 100.0 * trainable / max(total, 1)
    logger.info(
        "LoRA applicata: r=%s alpha=%s dropout=%s | parametri allenabili %s/%s (%.4f%%)",
        peft_config.r,
        peft_config.lora_alpha,
        peft_config.lora_dropout,
        f"{trainable:,}",
        f"{total:,}",
        pct,
    )
    return model


def count_trainable_params(model) -> tuple[int, int]:
    """Conta (parametri_allenabili, parametri_totali) del modello."""
    trainable = 0
    total = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return trainable, total


def set_gradient_checkpointing(model, enabled: bool):
    """Attiva/lascia inalterato il gradient checkpointing su `model`.

    Quando attivo disabilita la KV-cache (incompatibile col checkpointing) e
    garantisce input_require_grads. Ritorna sempre il modello.
    """
    if not enabled:
        return model
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:  # versioni vecchie senza il kwarg
            model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    logger.info("Gradient checkpointing abilitato (use_cache=False).")
    return model


# --------------------------------------------------------------------------- #
# Output / salvataggio
# --------------------------------------------------------------------------- #
def resolve_output_dir(cfg: dict, default_name: str) -> str:
    """Risolve la cartella di output: train.output_dir oppure output_root/<default>."""
    out = get(cfg, "train.output_dir")
    if not out:
        root = str(get(cfg, "project.output_root", "outputs"))
        out = os.path.join(root, default_name)
    return out


def save_artifacts(model, tokenizer, output_dir: str) -> str:
    """Salva pesi/adapter e tokenizer in `output_dir`. Ritorna il path."""
    from italian_llm.utils.io import ensure_dir  # noqa: PLC0415

    ensure_dir(output_dir)
    model.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    logger.info("Artefatti salvati in: %s", output_dir)
    return output_dir
