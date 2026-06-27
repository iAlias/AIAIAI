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
#     python scripts/train_sft.py --config configs/train/sft.yaml
# ===========================================================================

PY ?= python
PIP ?= $(PY) -m pip

# Config di default per ogni stadio della pipeline.
CFG_BASE     ?= configs/base.yaml
CFG_CPT      ?= configs/train/cpt.yaml
CFG_SFT      ?= configs/train/sft.yaml
CFG_ORPO     ?= configs/train/orpo.yaml
CFG_DISTILL  ?= configs/distill/student_3b.yaml
CFG_EVAL     ?= configs/eval/eval.yaml
CFG_SERVE    ?= configs/serving/vllm.yaml

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
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
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
# Pipeline dati (girano senza GPU)
# ---------------------------------------------------------------------------
.PHONY: corpus
corpus: ## [1] Prepara il corpus per il Continued Pre-Training (pulizia + dedup)
	$(PY) scripts/build_corpus.py --config $(CFG_BASE)

.PHONY: sft-data
sft-data: ## [2] Costruisce/pulisce il dataset SFT in formato chat JSONL
	$(PY) scripts/build_sft_data.py --config $(CFG_BASE)

.PHONY: synth
synth: ## [3] Genera dati sintetici dai teacher (mock di default, no rete)
	$(PY) scripts/synthesize.py --config $(CFG_BASE)

# ---------------------------------------------------------------------------
# Training (richiedono GPU per i pesi veri; con base sintetica girano in mock)
# ---------------------------------------------------------------------------
.PHONY: cpt
cpt: ## [4] Continued Pre-Training sul corpus italiano
	$(PY) scripts/train_cpt.py --config $(CFG_CPT)

.PHONY: sft
sft: ## [5] Supervised Fine-Tuning (QLoRA) sui dati istruzione
	$(PY) scripts/train_sft.py --config $(CFG_SFT)

.PHONY: orpo
orpo: ## [6] Allineamento alle preferenze (ORPO di default, DPO opzionale)
	$(PY) scripts/train_preference.py --config $(CFG_ORPO)

.PHONY: distill
distill: ## Distillazione verso uno student piu' piccolo (data/logits)
	$(PY) scripts/run_distill.py --config $(CFG_DISTILL)

# ---------------------------------------------------------------------------
# Valutazione / inferenza / serving
# ---------------------------------------------------------------------------
.PHONY: eval
eval: ## [7] Valuta il modello (aderenza, italianita', verbosita', rifiuti...)
	$(PY) scripts/run_eval.py --config $(CFG_EVAL)

.PHONY: infer
infer: ## Inferenza interattiva da CLI (transformers, fallback senza GPU)
	$(PY) scripts/infer.py --config $(CFG_EVAL)

.PHONY: serve
serve: ## Avvia il server di inferenza (vLLM se disponibile)
	$(PY) scripts/serve.py --config $(CFG_SERVE)

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
