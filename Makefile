# ===========================================================================
# Makefile — Italian LLM
# ---------------------------------------------------------------------------
# Orchestratore della pipeline end-to-end. Target principali (vedi `make help`).
# Pensato per Linux + CUDA, ma i target "leggeri" (setup, corpus, sft-data,
# synth con mock teacher, test, clean) girano anche senza GPU.
#
# Usa `python` (non `python3`) per compatibilita': su molti ambienti conda/venv
# l'eseguibile e' `python`. Se sul tuo sistema serve `python3`, esporta:
#     make PY=python3 <target>
#
# Su Windows (PowerShell) puoi invocare gli script direttamente, es:
#     python scripts/train_sft.py --config configs/train/sft_qwen9b_lora.yaml
# ===========================================================================

PY ?= python
PIP ?= $(PY) -m pip

# Config di default per ogni stadio della pipeline (percorsi REALI in configs/).
CFG_CORPUS   ?= configs/data/corpus.yaml
CFG_SFT_DATA ?= configs/data/sft.yaml
CFG_CPT      ?= configs/train/cpt_qwen9b.yaml
CFG_SFT      ?= configs/train/sft_qwen9b_lora.yaml
CFG_ORPO     ?= configs/train/orpo_qwen9b.yaml
CFG_DISTILL  ?= configs/distill/student_3b.yaml
CFG_EVAL     ?= configs/eval/eval.yaml
CFG_EVAL_CODING ?= configs/eval/eval_coding.yaml
CFG_EVAL_CODING_OLLAMA ?= configs/eval/eval_coding_ollama.yaml

# Modello per l'inferenza CLI (run_inference.py richiede --model).
MODEL ?= Qwen/Qwen2.5-7B

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# help — elenco auto-generato dai commenti "## " accanto ai target.
# ---------------------------------------------------------------------------
.PHONY: help
help: ## Mostra questo aiuto (lista dei target)
	@echo ""
	@echo "Italian LLM — comandi disponibili:"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Esempi:"
	@echo "  make setup            # installa dipendenze + pacchetto editable"
	@echo "  make synth            # genera dati sintetici col mock teacher (no rete)"
	@echo "  make sft              # fine-tuning SFT (QLoRA) con config di default"
	@echo "  make PY=python3 test  # override dell'eseguibile Python"
	@echo ""

# ---------------------------------------------------------------------------
# setup / ambiente
# ---------------------------------------------------------------------------
.PHONY: setup
setup: ## Installa le dipendenze e il pacchetto in modalita' editable
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "Setup completato. Copia .env.example in .env e personalizza se necessario."

# ---------------------------------------------------------------------------
# Pipeline dati (gira senza GPU; mock teacher di default, no rete)
# ---------------------------------------------------------------------------
.PHONY: download
download: ## [0] Scarica/genera i dati grezzi (fallback sintetico offline)
	$(PY) scripts/download_open_data.py --config $(CFG_CORPUS)

.PHONY: corpus
corpus: ## [1] Costruisce il corpus italiano da data/raw (pulizia base)
	$(PY) scripts/build_italian_corpus.py --config $(CFG_CORPUS)

.PHONY: clean-data
clean-data: ## [1b] Pulizia profonda + dedup del corpus (interim -> processed)
	$(PY) scripts/clean_dedup_filter.py --config $(CFG_CORPUS)

.PHONY: cpt-data
cpt-data: ## [1c] Prepara i blocchi per il Continued Pre-Training
	$(PY) scripts/prepare_continued_pretraining_data.py --config $(CFG_CORPUS)

.PHONY: instruct-data
instruct-data: ## [2a] Deriva istruzioni dal corpus processato
	$(PY) scripts/create_instruction_dataset.py --config $(CFG_SFT_DATA)

.PHONY: sft-data
sft-data: ## [2] Costruisce/pulisce il dataset SFT (train/valid/test JSONL)
	$(PY) scripts/prepare_sft_data.py --config $(CFG_SFT_DATA)

.PHONY: synth
synth: ## [3] Genera dati sintetici dai teacher (mock di default, no rete)
	$(PY) scripts/synthesize_with_teachers.py --config $(CFG_SFT_DATA)

# ---------------------------------------------------------------------------
# Training (richiedono GPU per i pesi veri; senza GPU fanno smoke run)
# ---------------------------------------------------------------------------
.PHONY: cpt
cpt: ## [4] Continued Pre-Training sul corpus italiano
	$(PY) scripts/train_cpt.py --config $(CFG_CPT)

.PHONY: sft
sft: ## [5] Supervised Fine-Tuning (QLoRA) sui dati istruzione
	$(PY) scripts/train_sft.py --config $(CFG_SFT)

.PHONY: orpo
orpo: ## [6] Allineamento alle preferenze (ORPO di default, DPO opzionale)
	$(PY) scripts/train_dpo_or_orpo.py --config $(CFG_ORPO)

.PHONY: distill
distill: ## Distillazione verso uno student piu' piccolo (data/logits)
	$(PY) scripts/distill_student.py --config $(CFG_DISTILL)

# ---------------------------------------------------------------------------
# Valutazione / inferenza / serving
# ---------------------------------------------------------------------------
.PHONY: eval
eval: ## [7] Valuta il modello (aderenza, italianita', verbosita', rifiuti...)
	$(PY) scripts/evaluate_all.py --config $(CFG_EVAL)

.PHONY: infer
infer: ## Inferenza interattiva da CLI (usa MODEL=<hf-id-o-path>)
	$(PY) scripts/run_inference.py --model $(MODEL)

.PHONY: serve
serve: ## Avvia il server vLLM (Linux + CUDA; vedi scripts/launch_vllm.sh)
	bash scripts/launch_vllm.sh

# ---------------------------------------------------------------------------
# Traccia coding locale (Ollama / Qwen2.5-Coder)
# ---------------------------------------------------------------------------
.PHONY: coding-data
coding-data: ## Costruisce il dataset SFT per il coder locale
	$(PY) scripts/build_coding_sft.py

.PHONY: coding-eval
coding-eval: ## Eval pass@1 del coder (mock senza modello reale)
	$(PY) scripts/run_coding_eval.py --config $(CFG_EVAL_CODING)

.PHONY: coding-eval-ollama
coding-eval-ollama: ## Eval pass@1 del coder via Ollama (modello reale locale)
	$(PY) scripts/run_coding_eval.py --config $(CFG_EVAL_CODING_OLLAMA)

# ---------------------------------------------------------------------------
# Qualita' del codice
# ---------------------------------------------------------------------------
.PHONY: test
test: ## Esegue la suite di test (pytest)
	$(PY) -m pytest

.PHONY: lint
lint: ## Lint con ruff e check formattazione con black
	$(PY) -m ruff check src scripts tests
	$(PY) -m black --check src scripts tests

.PHONY: format
format: ## Formatta il codice con black e applica i fix di ruff
	$(PY) -m black src scripts tests
	$(PY) -m ruff check --fix src scripts tests

.PHONY: clean
clean: ## Rimuove cache, build e artefatti temporanei (NON tocca data/ e outputs/)
	$(PY) -c "import shutil,glob,os; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True)]"
	$(PY) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache','.ruff_cache','.mypy_cache','build','dist')]"
	$(PY) -c "import shutil,glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('*.egg-info')]"
	@echo "Pulizia completata."
