# Apprendimento continuo — il modello impara da ogni domanda

Obiettivo: ogni domanda che fai al modello lo rende migliore sulle successive.
Il ciclo ha **due gambe**, con onestà su cosa succede subito e cosa richiede un retrain.

## Gamba 1 — Memoria immediata (zero retrain, funziona ora)

`scripts/chat_learn.py` salva **ogni scambio** in `data/memory/interactions.jsonl`
e, a ogni nuova domanda, recupera via BM25 gli scambi passati più rilevanti e li
inietta nel contesto. Il modello "ricorda" da subito, senza toccare i pesi.

```bash
# REPL con memoria (richiede Ollama attivo)
python scripts/chat_learn.py --ollama-model coder-local

# Domanda singola
python scripts/chat_learn.py --question "come gestisco il retry su HttpClient in C#?" --ollama-model coder-local

# Disattivare la memoria per un confronto A/B
python scripts/chat_learn.py --question "..." --no-memory
```

Parametri utili: `--k` (scambi iniettati, default 3), `--memory` (percorso log),
`--system`, `--temperature`, `--ollama-host`.

**Privacy**: `data/memory/` è esclusa da git (`.gitignore`). Le tue conversazioni
restano solo sulla tua macchina.

## Gamba 2 — Apprendimento nei pesi (retrain periodico)

La memoria contestuale non modifica il modello. Per consolidare davvero ciò che
chiedi più spesso, il log si converte in dataset SFT (schema del repo) e alimenta
il retrain QLoRA gratuito su Kaggle:

```bash
# 1) Esporta il log come dataset SFT
python scripts/chat_learn.py --memory data/memory/interactions.jsonl \
  --export-sft data/processed/sft_from_memory.jsonl

# 2) (opzionale) unisci al dataset coding esistente
#    entrambi rispettano lo stesso schema: basta concatenare i JSONL

# 3) Retrain su Kaggle (gratis, ~30h GPU/settimana):
#    notebooks/kaggle_qlora_coder.ipynb con DATA_PATH sul tuo dataset

# 4) Merge + quantizza + reimporta in Ollama
python scripts/quantize_gguf.py --in /path/merged --out outputs/gguf/coder-v2.q4_k_m.gguf
python scripts/export_ollama_coding.py --gguf outputs/gguf/coder-v2.q4_k_m.gguf --name coder-local
```

Cadenza consigliata: ogni 200–500 interazioni, o quando noti lacune ricorrenti.

## Cosa NON è (onestà)

- **Non è online learning nei pesi**: su CPU senza GPU il fine-tuning per singola
  domanda non è praticabile (minuti/ore per step, rischio catastrophic forgetting).
- La gamba 1 dà l'effetto "ricorda tutto" **subito**; la gamba 2 consolida nei
  pesi **periodicamente**. Insieme formano il ciclo ricorsivo completo:
  domanda → memoria → (batch) → retrain → modello migliore → domanda...

## Componenti

| Pezzo | File |
|------|------|
| Log + retrieval + export | `src/italian_llm/memory/interaction_log.py` |
| Chat CLI con memoria | `scripts/chat_learn.py` |
| Indice BM25 riusato | `src/italian_llm/rag/code_index.py` |
| Test | `tests/test_interaction_memory.py` |
