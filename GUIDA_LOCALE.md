# Guida locale — cosa fare adesso e come usare il modello sul tuo PC

Guida pratica per questo host (Windows, CPU-only, Ollama). Tutti i comandi si
lanciano da PowerShell nella cartella del repo (`D:\AI Code\AIAIAI`).

---

## 0. Prerequisiti (una volta sola)

```powershell
# 1) Ollama attivo e modello presente
ollama list                      # deve elencare coder-local o qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:1.5b   # se manca

# 2) Dipendenze Python leggere (niente torch)
pip install -e ".[dev]" numpy requests pydantic -c constraints.txt
```

Se `coder-local` non esiste più in Ollama, ricrealo (serve un GGUF già
quantizzato, vedi `docs/coding-model.md`) oppure usa direttamente
`qwen2.5-coder:1.5b` in tutti i comandi qui sotto.

---

## 1. Uso quotidiano

### A. Chat con memoria — il modello impara da ogni domanda (consigliata)

```powershell
python scripts/chat_learn.py --ollama-model coder-local
```

- Ogni scambio viene salvato in `data/memory/interactions.jsonl` (solo sul tuo
  PC, mai in git).
- Alle domande successive gli scambi passati più pertinenti vengono reiniettati
  nel contesto: più la usi, più "ti conosce".
- Domanda singola senza REPL: `--question "come faccio X?"`.
- Confronto senza memoria: aggiungi `--no-memory`.

### B. Chat semplice (senza memoria)

```powershell
ollama run coder-local
```

### C. Domande sul TUO codice (RAG)

```powershell
python scripts/rag_ask.py --root src --question "dove viene calcolato il BM25?" --ollama-model coder-local
```

Cambia `--root` per puntare a un altro progetto (es. `--root "D:\MioProgetto"`).

### D. Autocomplete in mezzo al codice (FIM)

```powershell
python scripts/fim_complete.py --prefix "def somma(a, b):`n    " --suffix "`n" --ollama-model coder-local
```

### E. Misurare il modello (numeri veri, non mock)

```powershell
# smoke test veloce (2 problemi)
python scripts/run_coding_eval.py --config configs/eval/eval_coding_ollama.yaml

# baseline complete (~2 h ciascuna su questa CPU)
python scripts/run_coding_eval.py --config configs/eval/eval_coding_ollama_humaneval.yaml
python scripts/run_coding_eval.py --config configs/eval/eval_coding_ollama_humaneval_js.yaml

# se una run si interrompe, riprendi senza rigenerare i task già fatti
python scripts/run_coding_eval.py --config configs/eval/eval_coding_ollama_humaneval.yaml `
  --resume-from outputs/eval/coding_report_humaneval_preds.jsonl
```

Report in `outputs/eval/coding_report_humaneval*.json`; le risposte grezze in
`outputs/eval/*_preds.jsonl`. Se Ollama è spento la run **si interrompe** con un
messaggio chiaro invece di scrivere un `pass@1` senza significato.

Tempi attesi su questa CPU: ~40–60 s a risposta con il modello 1.5B. È normale.

I numeri già misurati (agosto 2026) sono nel README, sezione **Baseline
misurata**: pass@1 0.628 su HumanEval e 0.596 su humaneval-js.

---

## 2. Ciclo di apprendimento nei pesi (ogni ~200–500 interazioni)

La memoria (punto 1A) funziona subito ma non modifica il modello. Per
consolidare davvero ciò che chiedi più spesso:

```powershell
# 1) Esporta il log come dataset SFT
python scripts/chat_learn.py --export-sft data/processed/sft_from_memory.jsonl

# 2) (opzionale) aggiungi il dataset coding
python scripts/build_coding_sft.py
#    poi concatena i due JSONL (stesso schema)

# 3) Retrain GRATIS su Kaggle (~30h GPU/settimana):
#    carica notebooks/kaggle_qlora_coder.ipynb su kaggle.com,
#    carica il JSONL come dataset, imposta DATA_PATH, esegui.

# 4) Scarica i pesi merged, quantizza e reimporta in Ollama:
python scripts/quantize_gguf.py --in <cartella_modello_merged> --out outputs/gguf/coder-v2.q4_k_m.gguf
python scripts/export_ollama_coding.py --gguf outputs/gguf/coder-v2.q4_k_m.gguf --name coder-local
```

Da qui in poi `coder-local` è la versione che ha imparato da te. Il ciclo si
ripete all'infinito: usa → accumula → retrain → usa.

Dettagli e onestà sui limiti: `docs/continuous-learning.md`.

---

## 3. Cosa fare adesso, in ordine

1. **Merge della PR** `chore/production-hygiene` su GitHub
   (https://github.com/iAlias/AIAIAI/compare/main...chore/production-hygiene se
   non è ancora aperta). Al merge parte la CI e il badge nel README diventa verde.
2. **pre-commit** (una volta): `pip install pre-commit && pre-commit install && pre-commit autoupdate`.
3. **Usa la chat con memoria** (1A) come strumento quotidiano per 2–3 settimane:
   accumula interazioni reali.
4. ~~**Baseline misurata**~~ — **fatta** (agosto 2026): pass@1 **0.628** su
   HumanEval (164 problemi Python) e **0.596** su humaneval-js (161 problemi
   JavaScript), con `coder-local` q4 su CPU. Dettagli e riproduzione nel README,
   sezione *Baseline misurata*. È il metro di confronto per ogni fine-tuning
   futuro: rilancia gli stessi due comandi dopo un retrain e confronta.
5. **Primo retrain** (sezione 2) quando il log supera ~200 interazioni; rilancia
   l'eval e confronta col baseline: se migliora, hai la prova che il ciclo funziona.
6. **Traccia italiana V1 (opzionale, richiede GPU)**: SFT del 7B su Kaggle T4 o
   GPU a noleggio (~0,20–0,40 €/h per 24 GB): `python scripts/train_sft.py
   --config configs/train/sft_qwen9b_lora.yaml` su Linux+CUDA, poi `make eval`.
   Solo dopo questo passo ha senso pubblicare pesi su Hugging Face.

---

## 4. Problemi comuni

| Sintomo | Causa/Rimedio |
|---|---|
| `Errore Ollama: ... non raggiungibile` | Ollama spento: apri l'app Ollama o `ollama serve`; verifica con `ollama list` |
| Risposte lentissime | Normale su CPU. Quantizzazione più aggressiva (`q3_k_m`) = più veloce, qualità un po' più bassa |
| `pass_at_1: 0.0` con `generation_mode: "mock"` | Config sbagliata o Ollama spento: usa `eval_coding_ollama.yaml` |
| `ModuleNotFoundError: italian_llm` | Manca l'install: `pip install -e .` |
| Il modello non "ricorda" | Stai usando `ollama run` diretto (1B) invece di `chat_learn.py` (1A), o hai passato `--no-memory` |
