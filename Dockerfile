# ===========================================================================
# Dockerfile — inferenza CPU (transformers) per Italian LLM.
#
# Questa immagine copre l'inferenza CLI su CPU (student piccoli / smoke run):
#   docker build -t italian-llm .
#   docker run --rm -it italian-llm --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Ciao"
#
# Per il serving GPU ad alto throughput NON usare questa immagine: usa
# l'immagine ufficiale vLLM montando i pesi (vedi scripts/launch_vllm.sh e
# configs/serving/vllm.yaml).
# ===========================================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf

WORKDIR /app

# Layer dipendenze: prima i file di vincolo per sfruttare la cache Docker.
COPY constraints.txt pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install -U pip \
    && pip install -e . numpy requests pydantic -c constraints.txt \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install "transformers>=4.44" "peft>=0.12" "accelerate>=0.33" sentencepiece

COPY scripts ./scripts
COPY configs ./configs

# I pesi NON sono nell'immagine: si scaricano da HF al primo run (volume /cache)
# o si montano localmente (-v /path/pesi:/models).
VOLUME ["/cache", "/models"]

ENTRYPOINT ["python", "scripts/run_inference.py"]
CMD ["--help"]
