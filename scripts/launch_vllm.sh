#!/usr/bin/env bash
# Avvia un server vLLM (API OpenAI-compatibile) leggendo i parametri da
# configs/serving/vllm.yaml. Gli argomenti extra passati a questo script vengono
# inoltrati tal quali a "vllm serve".
#
# Nota: i valori provengono da configs/serving/vllm.yaml (chiavi sotto "serving.*").
# Override del file di config tramite la variabile d'ambiente VLLM_CONFIG.
#
# Esempi:
#   scripts/launch_vllm.sh
#   VLLM_CONFIG=configs/serving/vllm.yaml scripts/launch_vllm.sh --enable-prefix-caching
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${VLLM_CONFIG:-${REPO_ROOT}/configs/serving/vllm.yaml}"

# Legge una chiave puntata dallo YAML (richiede python3 + PyYAML); ritorna il default in caso d'errore.
cfg_get() {
  local key="$1" default="$2"
  if ! command -v python3 >/dev/null 2>&1; then
    echo "$default"
    return 0
  fi
  python3 - "$CONFIG" "$key" "$default" <<'PY' 2>/dev/null || echo "$default"
import sys
try:
    import yaml
except Exception:
    print(sys.argv[3]); sys.exit(0)
path, dotted, default = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
except Exception:
    print(default); sys.exit(0)
node = cfg
for part in dotted.split("."):
    if isinstance(node, dict) and part in node:
        node = node[part]
    else:
        print(default); sys.exit(0)
print(node)
PY
}

MODEL="$(cfg_get serving.model Qwen/Qwen2.5-7B)"
HOST="$(cfg_get serving.host 0.0.0.0)"
PORT="$(cfg_get serving.port 8000)"
MAX_LEN="$(cfg_get serving.max_model_len 4096)"
DTYPE="$(cfg_get serving.dtype bfloat16)"
GPU_UTIL="$(cfg_get serving.gpu_memory_utilization 0.85)"
TP="$(cfg_get serving.tensor_parallel_size 1)"
MAX_SEQS="$(cfg_get serving.max_num_seqs 16)"
SERVED_NAME="$(cfg_get serving.served_model_name italian-llm)"

CMD=(vllm serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_LEN"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_UTIL"
  --tensor-parallel-size "$TP"
  --max-num-seqs "$MAX_SEQS"
  --served-model-name "$SERVED_NAME")

if ! command -v vllm >/dev/null 2>&1; then
  echo "[!] 'vllm' non trovato nel PATH. Installa con: pip install vllm" >&2
  echo "    Poi avvia manualmente:" >&2
  echo "    ${CMD[*]} $*" >&2
  exit 1
fi

echo "[i] Config: ${CONFIG}"
echo "[i] Avvio: ${CMD[*]} $*"
exec "${CMD[@]}" "$@"
