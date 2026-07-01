"""Supervised Fine-Tuning (SFT) su dataset chat JSONL con masking del prompt."""

from __future__ import annotations

from italian_llm.config import get
from italian_llm.logging_utils import get_logger
from italian_llm.training.common import (
    apply_lora,
    filter_kwargs,
    load_model_and_tokenizer,
    require_torch,
    require_transformers,
    resolve_output_dir,
    save_artifacts,
    select_optimizer,
    select_precision,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Encoding con masking del prompt
# --------------------------------------------------------------------------- #
def _encode_example(
    messages: list[dict], tokenizer, max_seq_len: int
) -> dict[str, list[int]] | None:
    """Tokenizza una conversazione mascherando tutto tranne le risposte assistant.

    Strategia stabile a prefisso (valida per i template Qwen e per il fallback
    render_plain): per ogni turno assistant si calcola lo span [start:end] dove
    `start` = lunghezza del prompt fino a quel turno (con generation prompt) ed
    `end` = lunghezza includendo la risposta. Solo quei token finiscono nei
    labels; tutto il resto resta -100.
    """
    from italian_llm.tokenizer_utils import format_chat  # noqa: PLC0415

    full_text = format_chat(tokenizer, messages, add_generation_prompt=False)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if not full_ids:
        return None

    labels = [-100] * len(full_ids)
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prompt_prefix = format_chat(tokenizer, messages[:i], add_generation_prompt=True)
        start = len(tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"])
        with_resp = format_chat(tokenizer, messages[: i + 1], add_generation_prompt=False)
        end = len(tokenizer(with_resp, add_special_tokens=False)["input_ids"])
        for j in range(start, min(end, len(full_ids))):
            labels[j] = full_ids[j]

    # Troncatura a max_seq_len.
    input_ids = full_ids[:max_seq_len]
    labels = labels[:max_seq_len]

    # Scarta esempi in cui, dopo la troncatura, non resta nulla da supervisionare.
    if all(t == -100 for t in labels):
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _build_dataset(rows: list[dict], tokenizer, max_seq_len: int) -> list[dict[str, list[int]]]:
    """Costruisce la lista di esempi tokenizzati a partire dalle righe SFT JSONL."""
    examples: list[dict[str, list[int]]] = []
    skipped = 0
    for row in rows:
        messages = row.get("messages")
        if not messages:
            skipped += 1
            continue
        enc = _encode_example(messages, tokenizer, max_seq_len)
        if enc is None:
            skipped += 1
            continue
        examples.append(enc)
    logger.info("Dataset SFT: %d esempi validi, %d scartati.", len(examples), skipped)
    if not examples:
        raise ValueError("Nessun esempio SFT valido dopo l'encoding: controlla i dati.")
    return examples


class _MaskingCollator:
    """Collator con right-padding: pad sugli input, -100 sui labels, 0 sull'attention."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = getattr(tokenizer, "eos_token_id", 0) or 0

    def __call__(self, features: list[dict[str, list[int]]]):
        import torch  # noqa: PLC0415

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            attention.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# Costruzione del trainer (trl SFTTrainer con fallback HF Trainer)
# --------------------------------------------------------------------------- #
def _training_desired(cfg: dict, output_dir: str, torch, has_eval: bool) -> dict:
    """Dizionario "desiderata" per TrainingArguments/SFTConfig (poi filtrato per versione)."""
    bf16, fp16 = select_precision(cfg, torch)
    save_steps = max(1, int(get(cfg, "train.save_steps", 200)))
    desired = {
        "output_dir": output_dir,
        "num_train_epochs": int(get(cfg, "train.epochs", 1)),
        "per_device_train_batch_size": int(get(cfg, "train.batch_size", 1)),
        "gradient_accumulation_steps": max(1, int(get(cfg, "train.grad_accum", 16))),
        "learning_rate": float(get(cfg, "train.lr", 2e-4)),
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
    }
    if has_eval:
        desired["per_device_eval_batch_size"] = int(get(cfg, "train.batch_size", 1))
        # Chiavi sinonime tra versioni: il filtro terra' quella supportata.
        desired["eval_strategy"] = "steps"
        desired["evaluation_strategy"] = "steps"
        desired["eval_steps"] = save_steps
    return desired


def _build_trl_trainer(cfg, model, tokenizer, desired, train_ds, eval_ds, collator, max_seq_len):
    """Tenta di costruire un trl SFTTrainer sul dataset gia' tokenizzato."""
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    sft_desired = dict(desired)
    sft_desired["packing"] = False
    sft_desired["max_seq_length"] = max_seq_len
    sft_desired["max_length"] = max_seq_len
    sft_desired["dataset_kwargs"] = {"skip_prepare_dataset": True}
    args = SFTConfig(**filter_kwargs(SFTConfig, sft_desired))

    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
    }
    # Nome dell'argomento tokenizer cambiato in processing_class nelle versioni recenti.
    import inspect  # noqa: PLC0415

    params = set(inspect.signature(SFTTrainer.__init__).parameters)
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**kwargs)


def _build_hf_trainer(cfg, model, tokenizer, desired, train_ds, eval_ds, collator):
    """Costruisce un transformers.Trainer classico (fallback robusto)."""
    from transformers import Trainer, TrainingArguments  # noqa: PLC0415

    args = TrainingArguments(**filter_kwargs(TrainingArguments, desired))
    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
    }
    import inspect  # noqa: PLC0415

    params = set(inspect.signature(Trainer.__init__).parameters)
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


# --------------------------------------------------------------------------- #
# Entrypoint pubblico
# --------------------------------------------------------------------------- #
def run_sft(cfg: dict) -> str:
    """Esegue il fine-tuning supervisionato QLoRA e ritorna la cartella di output."""
    torch = require_torch()
    require_transformers()
    from italian_llm.utils.io import read_jsonl  # noqa: PLC0415
    from italian_llm.utils.seed import set_seed  # noqa: PLC0415

    set_seed(int(get(cfg, "project.seed", 42)))

    train_path = get(cfg, "data.train_path")
    if not train_path:
        raise ValueError("Config priva di 'data.train_path' per l'SFT.")
    valid_path = get(cfg, "data.valid_path")
    max_seq_len = int(get(cfg, "train.max_seq_len", 2048))
    output_dir = resolve_output_dir(cfg, "sft")

    # --- Modello + tokenizer + LoRA ---
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)
    if hasattr(model, "config"):
        model.config.use_cache = False  # incompatibile col gradient checkpointing

    # --- Dati ---
    train_rows = list(read_jsonl(train_path))
    train_ds = _build_dataset(train_rows, tokenizer, max_seq_len)
    eval_ds = None
    if valid_path:
        eval_rows = list(read_jsonl(valid_path))
        eval_ds = _build_dataset(eval_rows, tokenizer, max_seq_len)

    collator = _MaskingCollator(tokenizer)
    desired = _training_desired(cfg, output_dir, torch, has_eval=eval_ds is not None)

    # --- Trainer: prova trl SFTTrainer, altrimenti HF Trainer ---
    prefer = str(get(cfg, "train.trainer", "auto")).lower()
    trainer = None
    if prefer in ("auto", "trl"):
        try:
            trainer = _build_trl_trainer(
                cfg, model, tokenizer, desired, train_ds, eval_ds, collator, max_seq_len
            )
            logger.info("SFT: uso trl SFTTrainer.")
        except Exception as exc:  # pragma: no cover - dipende dalle versioni
            if prefer == "trl":
                raise RuntimeError(
                    "trl SFTTrainer richiesto ma non utilizzabile: "
                    f"{exc}. Reinstalla trl o usa train.trainer=hf."
                ) from exc
            logger.warning("trl SFTTrainer non utilizzabile (%s); passo a HF Trainer.", exc)
    if trainer is None:
        trainer = _build_hf_trainer(cfg, model, tokenizer, desired, train_ds, eval_ds, collator)
        logger.info("SFT: uso transformers.Trainer.")

    logger.info(
        "Avvio SFT: %d esempi, max_seq_len=%d, output=%s", len(train_ds), max_seq_len, output_dir
    )
    trainer.train()

    # --- Salvataggio finale (adapter + tokenizer) ---
    save_artifacts(model, tokenizer, output_dir)
    logger.info("SFT completato. Output in: %s", output_dir)
    return output_dir
