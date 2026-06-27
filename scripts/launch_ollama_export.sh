#!/usr/bin/env bash
# Crea un Modelfile da un file GGUF e registra il modello in Ollama.
# Se 'ollama' non e' installato, stampa le istruzioni manuali esatte.
#
# Uso:
#   scripts/launch_ollama_export.sh <modello.gguf> [nome-modello]
#
# Esempio:
#   scripts/launch_ollama_export.sh outputs/gguf/italian-llm.q4_k_m.gguf italian-llm
set -euo pipefail

GGUF="${1:-}"
NAME="${2:-italian-llm}"

if [[ -z "${GGUF}" ]]; then
  echo "Uso: $0 <modello.gguf> [nome-modello]" >&2
  exit 2
fi
if [[ ! -f "${GGUF}" ]]; then
  echo "[!] File GGUF non trovato: ${GGUF}" >&2
  exit 1
fi

WORKDIR="$(cd "$(dirname "${GGUF}")" && pwd)"
GGUF_ABS="${WORKDIR}/$(basename "${GGUF}")"
MODELFILE="${WORKDIR}/Modelfile"

# Template chat in stile Qwen (ChatML) + system prompt italiano e parametri di sampling.
cat > "${MODELFILE}" <<EOF
FROM ${GGUF_ABS}

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM "Sei un assistente italiano: diretto, utile e poco verboso. Niente disclaimer superflui."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
EOF

echo "[i] Modelfile scritto in: ${MODELFILE}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "[!] 'ollama' non trovato nel PATH. Installa da https://ollama.com" >&2
  echo "    Poi registra il modello manualmente con:" >&2
  echo "    ollama create ${NAME} -f ${MODELFILE}" >&2
  echo "    ollama run ${NAME}" >&2
  exit 1
fi

echo "[i] Registro il modello '${NAME}' in Ollama..."
ollama create "${NAME}" -f "${MODELFILE}"
echo "[ok] Fatto. Avvia con: ollama run ${NAME}"
