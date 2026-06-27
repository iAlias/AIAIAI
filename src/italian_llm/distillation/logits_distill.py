"""Logits distillation sperimentale: KL su soft-label teacher->student (torch lazy).

ATTENZIONE — questa modalita' e' SPERIMENTALE e costosa:
  * Richiede teacher e studente con LO STESSO TOKENIZER/VOCABOLARIO: la KL si
    calcola sui logits token-per-token, quindi gli indici di vocabolario devono
    corrispondere. Con vocabolari diversi i logits non sono confrontabili.
  * Tiene DUE modelli in memoria contemporaneamente (teacher in inferenza,
    studente in training): l'occupazione di VRAM e' elevata.
  * E' piu' lenta e fragile della data distillation.

Per questi motivi la modalita' RACCOMANDATA e di DEFAULT del progetto e' la
DATA DISTILLATION (italian_llm.distillation.data_distill.build_distill_dataset),
piu' robusta, economica e indipendente dal vocabolario. Usa la logits
distillation solo per esperimenti mirati con modelli della stessa famiglia.

Loss: L = alpha * KL_softT + (1 - alpha) * CE_hard
  - KL_softT: KL( student/T || teacher/T ) * T^2 (soft targets, distillazione classica)
  - CE_hard:  cross-entropy causale sui token reali (label)
"""

import os

from italian_llm.config import get
from italian_llm.logging_utils import get_logger

logger = get_logger(__name__)


def _build_examples(train_path, tokenizer, max_seq_len, max_examples):
    """Tokenizza gli esempi SFT in input_ids/labels (causal LM)."""
    from italian_llm.utils.io import read_jsonl
    from italian_llm.tokenizer_utils import format_chat

    examples = []
    for rec in read_jsonl(train_path):
        messages = rec.get("messages")
        if not messages:
            continue
        text = format_chat(tokenizer, messages, add_generation_prompt=False)
        enc = tokenizer(text, truncation=True, max_length=max_seq_len)
        ids = enc["input_ids"]
        if len(ids) < 2:
            continue
        examples.append(ids)
        if max_examples and len(examples) >= max_examples:
            break
    return examples


def _collate(batch_ids, pad_id):
    """Padding manuale di un batch; labels = input_ids con -100 sul padding."""
    import torch

    max_len = max(len(x) for x in batch_ids)
    input_ids, attn, labels = [], [], []
    for ids in batch_ids:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
        labels.append(ids + [-100] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def run_logits_distill(cfg: dict) -> str:
    """Esegue la distillazione su logits e restituisce la cartella di output.

    Config attesa (distill/*.yaml):
      teacher.name        -> modello teacher (id HF o path)
      student.name        -> modello studente (deve condividere il tokenizer!)
      student.dtype       -> dtype del training (default bfloat16/float32 su CPU)
      distill.temperature -> T per i soft target (default 2.0)
      distill.alpha       -> peso della KL vs CE (default 0.5)
      distill.lr          -> learning rate (default 1e-5)
      distill.max_steps   -> passi massimi (default 50, esperimento)
      distill.batch_size  -> dimensione batch (default 1)
      distill.max_examples-> tetto sugli esempi caricati
      data.train_path     -> JSONL SFT di training
      train.output_dir / distill.output_dir -> cartella di salvataggio
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    from italian_llm.tokenizer_utils import load_tokenizer
    from italian_llm.utils.io import ensure_dir

    teacher_name = get(cfg, "teacher.name")
    student_name = get(cfg, "student.name") or get(cfg, "model.name")
    if not teacher_name or not student_name:
        raise ValueError("Servono 'teacher.name' e 'student.name' nella config.")

    temperature = float(get(cfg, "distill.temperature", 2.0) or 2.0)
    alpha = float(get(cfg, "distill.alpha", 0.5) or 0.5)
    lr = float(get(cfg, "distill.lr", 1e-5) or 1e-5)
    max_steps = int(get(cfg, "distill.max_steps", 50) or 50)
    batch_size = int(get(cfg, "distill.batch_size", 1) or 1)
    max_seq_len = int(get(cfg, "distill.max_seq_len", get(cfg, "train.max_seq_len", 1024)) or 1024)
    max_examples = get(cfg, "distill.max_examples")
    max_examples = int(max_examples) if max_examples else None

    output_dir = (get(cfg, "distill.output_dir")
                  or get(cfg, "train.output_dir")
                  or os.path.join(get(cfg, "project.output_root", "outputs"), "logits_distill"))
    ensure_dir(output_dir)

    train_path = get(cfg, "data.train_path")
    if not train_path or not os.path.exists(train_path):
        raise FileNotFoundError(f"data.train_path non trovato: {train_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_str = get(cfg, "student.dtype", "bfloat16")
    if device == "cpu":
        torch_dtype = torch.float32  # bf16/fp16 instabili/lenti su CPU
    else:
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}.get(str(dtype_str).lower(), torch.bfloat16)

    logger.info("Logits distillation (SPERIMENTALE): teacher=%s student=%s device=%s T=%.2f alpha=%.2f",
                teacher_name, student_name, device, temperature, alpha)
    logger.warning("Ricorda: teacher e studente DEVONO condividere il tokenizer/vocabolario. "
                   "La data distillation resta la modalita' consigliata.")

    # ----- Tokenizer (dello studente; il teacher deve condividerlo) -----
    tokenizer = load_tokenizer(student_name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # ----- Modelli -----
    logger.info("Carico teacher (eval, no-grad)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch_dtype, trust_remote_code=True)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    logger.info("Carico studente (train)...")
    student = AutoModelForCausalLM.from_pretrained(
        student_name, torch_dtype=torch_dtype, trust_remote_code=True)
    student.to(device)

    # LoRA opzionale per ridurre i parametri allenabili (consigliato).
    if get(cfg, "lora") is not None:
        try:
            from italian_llm.training.common import apply_lora

            student = apply_lora(student, cfg)
            logger.info("LoRA applicata allo studente.")
        except Exception as e:  # pragma: no cover
            logger.warning("LoRA non applicata (%s); proseguo in full fine-tuning.", e)

    if get(cfg, "train.gradient_checkpointing"):
        try:
            student.gradient_checkpointing_enable()
            student.config.use_cache = False
        except Exception:
            pass
    student.train()

    # Allineamento vocabolario: usiamo il minimo comune per la KL.
    v_student = student.get_output_embeddings().weight.shape[0]
    v_teacher = teacher.get_output_embeddings().weight.shape[0]
    v_kl = min(v_student, v_teacher)
    if v_student != v_teacher:
        logger.warning("Vocabolari diversi (student=%d, teacher=%d): KL calcolata sui primi %d "
                       "token. I risultati possono essere inaffidabili.", v_student, v_teacher, v_kl)

    # ----- Dati -----
    examples = _build_examples(train_path, tokenizer, max_seq_len, max_examples)
    if not examples:
        raise RuntimeError("Nessun esempio valido trovato per la distillazione.")
    logger.info("Esempi di training: %d", len(examples))

    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=lr)

    # ----- Loop di training (chiaro e minimale, stile nanoGPT adattato) -----
    step = 0
    T = temperature
    done = False
    while not done:
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            input_ids, attn, labels = _collate(batch, pad_id)
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                t_logits = teacher(input_ids=input_ids, attention_mask=attn).logits

            s_logits = student(input_ids=input_ids, attention_mask=attn).logits

            # Shift causale: si predice il token t+1 dalla posizione t.
            s_shift = s_logits[:, :-1, :]
            t_shift = t_logits[:, :-1, :]
            lbl_shift = labels[:, 1:]
            valid = (lbl_shift != -100)

            # --- Soft loss (KL sui logits temperati, vocab allineato) ---
            s_soft = F.log_softmax(s_shift[..., :v_kl] / T, dim=-1)
            t_soft = F.softmax(t_shift[..., :v_kl] / T, dim=-1)
            kl_tok = F.kl_div(s_soft, t_soft, reduction="none").sum(dim=-1)  # [B, T-1]
            denom = valid.sum().clamp(min=1)
            soft_loss = (kl_tok * valid).sum() / denom * (T * T)

            # --- Hard loss (CE sui token reali, vocab pieno dello studente) ---
            hard_loss = F.cross_entropy(
                s_shift.reshape(-1, s_shift.size(-1)).float(),
                lbl_shift.reshape(-1),
                ignore_index=-100,
            )

            loss = alpha * soft_loss + (1.0 - alpha) * hard_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in student.parameters() if p.requires_grad), 1.0)
            optimizer.step()

            step += 1
            if step % max(1, int(get(cfg, "train.logging_steps", 5) or 5)) == 0:
                logger.info("step %d/%d | loss %.4f (soft %.4f, hard %.4f)",
                            step, max_steps, float(loss), float(soft_loss), float(hard_loss))

            if step >= max_steps:
                done = True
                break

    # ----- Salvataggio -----
    logger.info("Training terminato a %d step. Salvo in %s", step, output_dir)
    try:
        student.save_pretrained(output_dir)
    except Exception as e:  # pragma: no cover
        logger.warning("save_pretrained ha avuto un problema (%s); provo a salvare lo state_dict.", e)
        import torch as _torch

        _torch.save(student.state_dict(), os.path.join(output_dir, "student_state_dict.pt"))
    tokenizer.save_pretrained(output_dir)
    return output_dir
