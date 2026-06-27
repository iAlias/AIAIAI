#!/usr/bin/env bash
# Rilevamento ambiente (opzionale): mostra GPU/CUDA e lo stato di torch/transformers.
# Non installa nulla e non fallisce mai (utile come diagnostica pre-training).
#
# Uso:
#   bash scripts/setup_env.sh
#
# Variabili:
#   VENV_DIR   directory del virtualenv da usare per i check Python (default .venv)
set -uo pipefail   # niente -e: lo script è puramente diagnostico e tollerante.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${VENV_DIR:-.venv}"

echo "==> Italian LLM :: rilevamento ambiente"
echo "    OS        : $(uname -s 2>/dev/null || echo sconosciuto) $(uname -r 2>/dev/null || true)"

# --- selezione interprete Python (preferisci il venv se presente) ----------
PYTHON_BIN=""
if [ -x "${VENV_DIR}/bin/python" ]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
elif [ -x "${VENV_DIR}/Scripts/python.exe" ]; then
  PYTHON_BIN="${VENV_DIR}/Scripts/python.exe"
else
  for cand in python3.11 python3 python; do
    if command -v "${cand}" >/dev/null 2>&1; then
      PYTHON_BIN="${cand}"
      break
    fi
  done
fi
if [ -n "${PYTHON_BIN}" ]; then
  echo "    python    : $(${PYTHON_BIN} --version 2>&1) (${PYTHON_BIN})"
else
  echo "    python    : NON trovato"
fi

# --- driver NVIDIA / CUDA ---------------------------------------------------
echo ""
echo "==> GPU / CUDA"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "    nvidia-smi rilevato:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    | sed 's/^/      /' || nvidia-smi 2>/dev/null | sed -n '1,12p'
else
  echo "    nvidia-smi NON trovato: nessuna GPU NVIDIA visibile (training pesante non disponibile)."
fi

if command -v nvcc >/dev/null 2>&1; then
  echo "    nvcc      : $(nvcc --version 2>/dev/null | grep -i release | sed 's/^ *//')"
else
  echo "    nvcc      : toolkit CUDA non in PATH (ok se usi i wheel torch+cuXX)."
fi

# --- check torch/transformers (best effort, mai fatale) --------------------
echo ""
echo "==> Stack PyTorch / Transformers"
if [ -n "${PYTHON_BIN}" ]; then
  "${PYTHON_BIN}" - <<'PYCHECK' || echo "    (check Python terminato con avvisi)"
import importlib

def ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception as exc:  # libreria assente: normale in modalità leggera
        return f"NON installato ({type(exc).__name__})"

torch_ver = ver("torch")
print(f"    torch        : {torch_ver}")
try:
    import torch  # noqa
    print(f"    torch.cuda   : disponibile={torch.cuda.is_available()} "
          f"(device={torch.cuda.device_count() if torch.cuda.is_available() else 0}, "
          f"cuda={getattr(torch.version, 'cuda', None)})")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"      - GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception:
    print("    torch.cuda   : n/d (torch non importabile)")

print(f"    transformers : {ver('transformers')}")
print(f"    datasets     : {ver('datasets')}")
print(f"    peft         : {ver('peft')}")
print(f"    trl          : {ver('trl')}")
print(f"    bitsandbytes : {ver('bitsandbytes')}")
print(f"    vllm         : {ver('vllm')}")
PYCHECK
else
  echo "    Interprete Python non disponibile: salto i check delle librerie."
fi

echo ""
echo "==> Diagnostica completata."
echo "    Se le librerie pesanti risultano 'NON installato', il repo resta usabile"
echo "    in modalità leggera (config, dati sintetici mock, test). Per il training"
echo "    reale installa lo stack su Linux+CUDA: pip install -r requirements.txt"
