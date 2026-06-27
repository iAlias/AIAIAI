#!/usr/bin/env bash
# Bootstrap del progetto Italian LLM: crea il virtualenv, installa le dipendenze,
# prepara il file .env e stampa i passi successivi. Pensato per Linux + CUDA ma
# i passi "leggeri" (pacchetto editable, dati sintetici col mock teacher) girano
# ovunque. Le dipendenze pesanti (torch, transformers, vllm...) possono fallire
# offline: in quel caso lo script continua con l'installazione "core".
#
# Uso:
#   bash scripts/bootstrap_project.sh           # setup completo (best effort)
#   CORE_ONLY=1 bash scripts/bootstrap_project.sh   # solo dipendenze leggere
#   VENV_DIR=.venv311 bash scripts/bootstrap_project.sh
set -euo pipefail

# --- posizionamento sulla root del repo ------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${VENV_DIR:-.venv}"
CORE_ONLY="${CORE_ONLY:-0}"

echo "==> Italian LLM :: bootstrap"
echo "    repo root : ${REPO_ROOT}"
echo "    venv dir  : ${VENV_DIR}"

# --- selezione interprete Python -------------------------------------------
PYTHON_BIN=""
for cand in python3.11 python3 python; do
  if command -v "${cand}" >/dev/null 2>&1; then
    PYTHON_BIN="${cand}"
    break
  fi
done
if [ -z "${PYTHON_BIN}" ]; then
  echo "ERRORE: nessun interprete Python trovato (python3.11/python3/python)." >&2
  exit 1
fi
echo "    python    : $(${PYTHON_BIN} --version 2>&1) (${PYTHON_BIN})"

# --- creazione virtualenv ---------------------------------------------------
if [ ! -d "${VENV_DIR}" ]; then
  echo "==> Creazione virtualenv in '${VENV_DIR}'"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "==> Virtualenv '${VENV_DIR}' già presente, riutilizzo."
fi

# Path agli eseguibili del venv (compatibile Linux/macOS e Git-Bash su Windows).
if [ -x "${VENV_DIR}/bin/python" ]; then
  VENV_PY="${VENV_DIR}/bin/python"
elif [ -x "${VENV_DIR}/Scripts/python.exe" ]; then
  VENV_PY="${VENV_DIR}/Scripts/python.exe"
else
  echo "ERRORE: interprete del venv non trovato in '${VENV_DIR}'." >&2
  exit 1
fi

# --- aggiornamento pip + pacchetto in editable (core, sempre) --------------
echo "==> Aggiornamento pip/setuptools/wheel"
"${VENV_PY}" -m pip install -U pip setuptools wheel

echo "==> Installazione del pacchetto 'italian_llm' in modalità editable (core)"
"${VENV_PY}" -m pip install -e .

# --- dipendenze pesanti (best effort) --------------------------------------
if [ "${CORE_ONLY}" = "1" ]; then
  echo "==> CORE_ONLY=1: salto le dipendenze pesanti (torch/transformers/...)."
else
  echo "==> Installazione dipendenze complete da requirements.txt (best effort)"
  if "${VENV_PY}" -m pip install -r requirements.txt; then
    echo "    Dipendenze complete installate."
  else
    echo "    AVVISO: installazione di requirements.txt fallita (offline o no GPU?)." >&2
    echo "    Il repo resta usabile in modalità leggera: config, dati sintetici (mock teacher), test." >&2
    echo "    Per il training reale installa torch/transformers su Linux+CUDA, es:" >&2
    echo "      ${VENV_PY} -m pip install torch --index-url https://download.pytorch.org/whl/cu121" >&2
  fi
fi

# --- preparazione file .env -------------------------------------------------
if [ ! -f ".env.example" ]; then
  echo "==> .env.example assente: genero un template di default."
  cat > ".env.example" <<'ENV_EOF'
# Configurazione locale Italian LLM (copiare in .env e personalizzare).
# Provider del teacher per la sintesi dati: mock | openai_compat | hf_local
TEACHER_PROVIDER=mock
# Teacher OpenAI-compatibile (lasciare vuoto per usare il mock 100% offline)
TEACHER_BASE_URL=
TEACHER_API_KEY=
TEACHER_MODEL=
# Cache e token HuggingFace (opzionali)
HF_HOME=.hf_cache
HUGGINGFACE_HUB_TOKEN=
# Livello di logging (DEBUG/INFO/WARNING/ERROR)
ITALIAN_LLM_LOG_LEVEL=INFO
ENV_EOF
fi

if [ ! -f ".env" ]; then
  echo "==> Creazione di .env da .env.example"
  cp ".env.example" ".env"
else
  echo "==> .env già presente, lo lascio invariato."
fi

# --- placeholder directory dati --------------------------------------------
mkdir -p data/raw data/interim data/processed data/eval outputs
for d in data/raw data/interim data/processed data/eval outputs; do
  [ -f "${d}/.gitkeep" ] || : > "${d}/.gitkeep"
done

# --- prossimi passi ---------------------------------------------------------
cat <<NEXT

==> Bootstrap completato.

Prossimi passi:
  1) Attiva il virtualenv:
       source ${VENV_DIR}/bin/activate        # Linux/macOS
       ${VENV_DIR}\\Scripts\\activate           # Windows PowerShell
  2) (Opzionale) Rileva GPU/CUDA:
       bash scripts/setup_env.sh
  3) Genera i dati (100% offline, mock teacher):
       python scripts/download_open_data.py --config configs/data/corpus.yaml
       python scripts/build_italian_corpus.py --config configs/data/corpus.yaml
       python scripts/clean_dedup_filter.py --config configs/data/corpus.yaml
       python scripts/create_instruction_dataset.py --config configs/data/sft.yaml
       python scripts/synthesize_with_teachers.py --kind sft --n 200 --out data/processed/synth_sft.jsonl
       python scripts/prepare_sft_data.py --config configs/data/sft.yaml
  4) Esegui i test:
       python -m pytest

Suggerimento: rivedi e personalizza '.env' (provider teacher, token HF, ecc.).
NEXT
