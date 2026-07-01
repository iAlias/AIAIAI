"""Continued Pre-Training (CPT) su corpus italiano con loop minimale stile nanoGPT."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterator
from contextlib import nullcontext

from italian_llm.config import get
from italian_llm.logging_utils import get_logger
from italian_llm.training.common import (
    apply_lora,
    load_model_and_tokenizer,
    require_torch,
    resolve_output_dir,
    save_artifacts,
    select_precision,
    set_gradient_checkpointing,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Caricamento corpus -> blocchi di token
# --------------------------------------------------------------------------- #
def _iter_documents(path: str) -> Iterator[str]:
    """Itera i documenti di testo da un file (.jsonl/.txt) o da una cartella.

    - .jsonl: usa il campo 'text', altrimenti concatena 'messages[*].content',
      altrimenti il campo 'content'.
    - .txt: ogni documento e' separato da una riga vuota (doppio newline).
    """
    if os.path.isdir(path):
        files: list[str] = []
        for root, _dirs, names in os.walk(path):
            for name in names:
                if name.endswith((".jsonl", ".txt")):
                    files.append(os.path.join(root, name))
        files.sort()
    else:
        files = [path]

    for fpath in files:
        if fpath.endswith(".jsonl"):
            from italian_llm.utils.io import read_jsonl  # noqa: PLC0415

            for row in read_jsonl(fpath):
                text = row.get("text")
                if not text and row.get("messages"):
                    text = "\n".join(str(m.get("content", "")) for m in row.get("messages", []))
                if not text and row.get("content"):
                    text = row.get("content")
                if text and str(text).strip():
                    yield str(text).strip()
        else:
            with open(fpath, encoding="utf-8") as fh:
                content = fh.read()
            for doc in content.split("\n\n"):
                doc = doc.strip()
                if doc:
                    yield doc


def _build_token_blocks(documents: Iterator[str], tokenizer, block_size: int) -> list[list[int]]:
    """Tokenizza e impacchetta i documenti in blocchi contigui di `block_size` token.

    I documenti sono separati dal token EOS; il resto piu' corto di un blocco viene
    scartato per mantenere blocchi di lunghezza uniforme (semplice e veloce).
    """
    eos_id = getattr(tokenizer, "eos_token_id", None)
    buffer: list[int] = []
    blocks: list[list[int]] = []
    n_docs = 0

    for doc in documents:
        ids = tokenizer(doc, add_special_tokens=False)["input_ids"]
        buffer.extend(ids)
        if eos_id is not None:
            buffer.append(eos_id)
        n_docs += 1
        while len(buffer) >= block_size:
            blocks.append(buffer[:block_size])
            buffer = buffer[block_size:]

    logger.info(
        "Corpus tokenizzato: %d documenti -> %d blocchi da %d token (~%d token totali).",
        n_docs,
        len(blocks),
        block_size,
        len(blocks) * block_size,
    )
    return blocks


def _collate(batch: list[list[int]]):
    """Impila una lista di blocchi (liste di int) in tensori per il forward causale."""
    import torch  # noqa: PLC0415

    input_ids = torch.tensor(batch, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


# --------------------------------------------------------------------------- #
# Loop di training
# --------------------------------------------------------------------------- #
def run_cpt(cfg: dict) -> str:
    """Esegue il continued pre-training LoRA e ritorna la cartella di output.

    Loop custom in spirito Karpathy: AdamW + scheduler coseno con warmup,
    gradient accumulation, logging compatto (step, loss, lr, token/s).
    """
    torch = require_torch()
    from torch.utils.data import DataLoader  # noqa: PLC0415
    from transformers import get_cosine_schedule_with_warmup  # noqa: PLC0415

    from italian_llm.utils.seed import set_seed  # noqa: PLC0415

    seed = int(get(cfg, "project.seed", 42))
    set_seed(seed)

    # --- Iperparametri (con default prudenti per QLoRA) ---
    train_path = get(cfg, "data.train_path")
    if not train_path:
        raise ValueError("Config priva di 'data.train_path' per il CPT.")
    output_dir = resolve_output_dir(cfg, "cpt")
    epochs = int(get(cfg, "train.epochs", 1))
    batch_size = int(get(cfg, "train.batch_size", 1))
    grad_accum = max(1, int(get(cfg, "train.grad_accum", 16)))
    block_size = int(get(cfg, "train.max_seq_len", 1024))
    lr = float(get(cfg, "train.lr", 2e-4))
    warmup_ratio = float(get(cfg, "train.warmup_ratio", 0.03))
    weight_decay = float(get(cfg, "train.weight_decay", 0.0))
    logging_steps = max(1, int(get(cfg, "train.logging_steps", 10)))
    save_steps = max(1, int(get(cfg, "train.save_steps", 200)))
    grad_clip = float(get(cfg, "train.max_grad_norm", 1.0))
    gradient_checkpointing = bool(get(cfg, "train.gradient_checkpointing", True))

    # --- Modello + tokenizer + LoRA ---
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)
    set_gradient_checkpointing(model, gradient_checkpointing)

    # --- Dati ---
    blocks = _build_token_blocks(_iter_documents(train_path), tokenizer, block_size)
    if not blocks:
        raise ValueError(
            f"Nessun blocco di token costruito da '{train_path}': corpus vuoto o piu' "
            f"corto di max_seq_len={block_size}."
        )
    loader = DataLoader(
        blocks,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        drop_last=False,
    )

    # --- Device / precisione ---
    placed = getattr(model, "hf_device_map", None) is not None
    if placed:
        device = next(model.parameters()).device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    bf16, fp16 = select_precision(cfg, torch)
    amp_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else None)
    use_amp = device.type == "cuda" and amp_dtype is not None
    try:
        from torch.amp import GradScaler  # noqa: PLC0415

        scaler = GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    except Exception:  # pragma: no cover - torch piu' vecchi
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # --- Optimizer + scheduler ---
    steps_per_epoch = math.ceil(len(loader) / grad_accum)
    total_optim_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_optim_steps * warmup_ratio)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_optim_steps)

    logger.info(
        "Avvio CPT: epochs=%d batch=%d grad_accum=%d -> %d step di ottimizzazione "
        "(warmup=%d), device=%s, amp=%s",
        epochs,
        batch_size,
        grad_accum,
        total_optim_steps,
        warmup_steps,
        device,
        ("bf16" if bf16 else "fp16" if fp16 else "fp32"),
    )

    # --- Training ---
    model.train()
    global_step = 0
    running_loss = 0.0
    tokens_since_log = 0
    t_log = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        n_batches = len(loader)
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            )
            with amp_ctx:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = out.loss / grad_accum

            scaler.scale(loss).backward()
            running_loss += out.loss.item()
            tokens_since_log += int(input_ids.numel())

            is_boundary = ((i + 1) % grad_accum == 0) or (i + 1 == n_batches)
            if not is_boundary:
                continue

            # Passo di ottimizzazione vero e proprio.
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % logging_steps == 0:
                elapsed = max(1e-6, time.time() - t_log)
                tok_per_s = tokens_since_log / elapsed
                avg_loss = running_loss / (logging_steps * grad_accum)
                cur_lr = scheduler.get_last_lr()[0]
                logger.info(
                    "step %5d/%d | epoch %d | loss %.4f | lr %.2e | %.0f tok/s",
                    global_step,
                    total_optim_steps,
                    epoch + 1,
                    avg_loss,
                    cur_lr,
                    tok_per_s,
                )
                running_loss = 0.0
                tokens_since_log = 0
                t_log = time.time()

            if global_step % save_steps == 0:
                ckpt = os.path.join(output_dir, f"checkpoint-{global_step}")
                save_artifacts(model, tokenizer, ckpt)

    # --- Salvataggio finale ---
    save_artifacts(model, tokenizer, output_dir)
    logger.info("CPT completato. Output in: %s", output_dir)
    return output_dir
