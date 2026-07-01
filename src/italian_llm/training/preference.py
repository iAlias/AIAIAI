"""Allineamento alle preferenze (ORPO default, DPO opzionale) per ridurre l'over-refusal."""

from __future__ import annotations

from italian_llm.config import get
from italian_llm.logging_utils import get_logger
from italian_llm.training.common import (
    build_lora_config,
    filter_kwargs,
    load_model_and_tokenizer,
    require_datasets,
    require_torch,
    require_trl,
    resolve_output_dir,
    save_artifacts,
    select_optimizer,
    select_precision,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Costruzione dataset di preferenze
# --------------------------------------------------------------------------- #
def _build_pairs(rows: list[dict], tokenizer) -> list[dict]:
    """Trasforma le righe di preferenza in record {prompt, chosen, rejected}.

    Il prompt viene reso col chat template includendo un system prompt
    "bilanciato" (filosofia del progetto: meno rifiuti su richieste lecite,
    niente disclaimer superflui). chosen/rejected restano le risposte grezze:
    tipicamente `chosen` e' la risposta utile e `rejected` un over-refusal.
    """
    from italian_llm.data.prompts import build_messages  # noqa: PLC0415
    from italian_llm.safety.policy import balanced_system_prompt  # noqa: PLC0415
    from italian_llm.tokenizer_utils import format_chat  # noqa: PLC0415

    system = balanced_system_prompt()
    pairs: list[dict] = []
    skipped = 0
    for row in rows:
        user = row.get("prompt", "")
        chosen = row.get("chosen", "")
        rejected = row.get("rejected", "")
        if not (user and chosen and rejected):
            skipped += 1
            continue
        messages = build_messages(system, user)
        prompt_text = format_chat(tokenizer, messages, add_generation_prompt=True)
        pairs.append({"prompt": prompt_text, "chosen": chosen, "rejected": rejected})

    logger.info("Dataset preferenze: %d coppie valide, %d scartate.", len(pairs), skipped)
    if not pairs:
        raise ValueError("Nessuna coppia di preferenze valida: servono prompt/chosen/rejected.")
    return pairs


def _to_hf_dataset(pairs: list[dict]):
    """Converte la lista di coppie in un datasets.Dataset."""
    datasets = require_datasets()
    return datasets.Dataset.from_list(pairs)


# --------------------------------------------------------------------------- #
# Selezione del metodo (ORPO / DPO)
# --------------------------------------------------------------------------- #
def _resolve_method(cfg: dict) -> str:
    """Ritorna 'orpo' (default) o 'dpo' in base alla config."""
    method = str(get(cfg, "preference.method", get(cfg, "train.method", "orpo"))).lower()
    if method not in ("orpo", "dpo"):
        logger.warning("Metodo preferenze '%s' non riconosciuto: uso ORPO.", method)
        method = "orpo"
    return method


def _desired_args(cfg: dict, output_dir: str, torch, method: str, has_eval: bool) -> dict:
    """Costruisce il dizionario di parametri per ORPOConfig/DPOConfig (poi filtrato)."""
    bf16, fp16 = select_precision(cfg, torch)
    save_steps = max(1, int(get(cfg, "train.save_steps", 200)))
    beta = float(get(cfg, "preference.beta", 0.1))
    max_len = int(get(cfg, "train.max_seq_len", 1024))
    max_prompt_len = int(get(cfg, "preference.max_prompt_length", max_len // 2))

    desired = {
        "output_dir": output_dir,
        "num_train_epochs": int(get(cfg, "train.epochs", 1)),
        "per_device_train_batch_size": int(get(cfg, "train.batch_size", 1)),
        "gradient_accumulation_steps": max(1, int(get(cfg, "train.grad_accum", 16))),
        "learning_rate": float(get(cfg, "train.lr", 5e-6)),
        "warmup_ratio": float(get(cfg, "train.warmup_ratio", 0.03)),
        "weight_decay": float(get(cfg, "train.weight_decay", 0.0)),
        "logging_steps": max(1, int(get(cfg, "train.logging_steps", 10))),
        "save_steps": save_steps,
        "save_total_limit": int(get(cfg, "train.save_total_limit", 2)),
        "lr_scheduler_type": str(get(cfg, "train.lr_scheduler_type", "cosine")),
        "max_grad_norm": float(get(cfg, "train.max_grad_norm", 1.0)),
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": bool(get(cfg, "train.gradient_checkpointing", True)),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": select_optimizer(cfg),
        "report_to": [],
        "remove_unused_columns": False,
        "seed": int(get(cfg, "project.seed", 42)),
        # Lunghezze sequenza (nomi compatibili tra versioni trl).
        "max_length": max_len,
        "max_prompt_length": max_prompt_len,
        "max_completion_length": int(
            get(cfg, "preference.max_completion_length", max_len - max_prompt_len)
        ),
        "beta": beta,
    }
    if method == "orpo":
        # In ORPO il termine di odds-ratio e' pesato da 'beta' (lambda).
        desired["beta"] = float(get(cfg, "preference.lambda", beta))
    if has_eval:
        desired["per_device_eval_batch_size"] = int(get(cfg, "train.batch_size", 1))
        desired["eval_strategy"] = "steps"
        desired["evaluation_strategy"] = "steps"
        desired["eval_steps"] = save_steps
    return desired


def _construct_trainer(trainer_cls, base_kwargs: dict, tokenizer, peft_config, needs_ref: bool):
    """Istanzia il trainer trl gestendo le differenze di firma tra versioni."""
    import inspect  # noqa: PLC0415

    params = set(inspect.signature(trainer_cls.__init__).parameters)
    kwargs = dict(base_kwargs)
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    if "peft_config" in params:
        kwargs["peft_config"] = peft_config
    if needs_ref and "ref_model" in params:
        # Con PEFT il modello di riferimento e' il base con adapter disabilitati.
        kwargs["ref_model"] = None
    return trainer_cls(**kwargs)


# --------------------------------------------------------------------------- #
# Entrypoint pubblico
# --------------------------------------------------------------------------- #
def run_preference(cfg: dict) -> str:
    """Esegue il training di preferenza (ORPO/DPO) e ritorna la cartella di output."""
    torch = require_torch()
    require_trl()
    from italian_llm.utils.io import read_jsonl  # noqa: PLC0415
    from italian_llm.utils.seed import set_seed  # noqa: PLC0415

    set_seed(int(get(cfg, "project.seed", 42)))

    train_path = get(cfg, "data.train_path")
    if not train_path:
        raise ValueError("Config priva di 'data.train_path' per il training di preferenza.")
    valid_path = get(cfg, "data.valid_path")
    output_dir = resolve_output_dir(cfg, "preference")
    method = _resolve_method(cfg)

    # --- Modello base (la LoRA la applica il trainer trl via peft_config) ---
    model, tokenizer = load_model_and_tokenizer(cfg)
    if hasattr(model, "config"):
        model.config.use_cache = False
    peft_config = build_lora_config(cfg)

    # --- Dati ---
    train_pairs = _build_pairs(list(read_jsonl(train_path)), tokenizer)
    train_ds = _to_hf_dataset(train_pairs)
    eval_ds = None
    if valid_path:
        eval_ds = _to_hf_dataset(_build_pairs(list(read_jsonl(valid_path)), tokenizer))

    # --- Selezione classe trainer/config ---
    if method == "dpo":
        from trl import DPOConfig, DPOTrainer  # noqa: PLC0415

        config_cls, trainer_cls, needs_ref = DPOConfig, DPOTrainer, True
    else:
        from trl import ORPOConfig, ORPOTrainer  # noqa: PLC0415

        config_cls, trainer_cls, needs_ref = ORPOConfig, ORPOTrainer, False

    desired = _desired_args(cfg, output_dir, torch, method, has_eval=eval_ds is not None)
    args = config_cls(**filter_kwargs(config_cls, desired))

    base_kwargs = {"model": model, "args": args, "train_dataset": train_ds, "eval_dataset": eval_ds}
    trainer = _construct_trainer(trainer_cls, base_kwargs, tokenizer, peft_config, needs_ref)

    logger.info(
        "Avvio training preferenze [%s]: %d coppie, beta=%s, output=%s "
        "(framing anti over-refusal su richieste lecite).",
        method.upper(),
        len(train_pairs),
        getattr(args, "beta", "n/d"),
        output_dir,
    )
    trainer.train()

    # --- Salvataggio finale (adapter + tokenizer) ---
    save_artifacts(model, tokenizer, output_dir)
    logger.info("Training preferenze completato. Output in: %s", output_dir)
    return output_dir
