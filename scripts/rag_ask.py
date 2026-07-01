#!/usr/bin/env python
"""Domanda al coder con contesto RAG dal tuo codebase locale."""

import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

import argparse

from italian_llm.data.prompts import build_messages, coding_system
from italian_llm.evaluation.runner import _Predictor
from italian_llm.logging_utils import get_logger, setup_logging
from italian_llm.rag.code_index import CodeIndex, build_context, index_paths

logger = get_logger(__name__)


def main(argv=None):
    p = argparse.ArgumentParser(description="Chiedi al coder con RAG sul tuo codice.")
    p.add_argument("--root", default=".", help="Cartella del codebase da indicizzare")
    p.add_argument("--question", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument(
        "--ollama-model",
        default=None,
        help="Nome di un modello gia' registrato in Ollama (usa questo invece di --model)",
    )
    p.add_argument("--ollama-host", default="http://localhost:11434")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    chunks = index_paths(args.root)
    logger.info("Indicizzati %d chunk da %s", len(chunks), args.root)
    idx = CodeIndex.build(chunks)
    hits = idx.search(args.question, k=args.k)
    context = build_context(hits)

    predictor = _Predictor(
        {
            "eval": {
                "model_path": args.model,
                "adapter": "",
                "max_new_tokens": 512,
                "temperature": 0.2,
                "ollama_model": args.ollama_model,
                "ollama_host": args.ollama_host,
            }
        }
    )
    predictor.init()
    user = (
        f"Contesto dal codebase:\n{context}\n\nDomanda: {args.question}"
        if context
        else args.question
    )
    answer = predictor.predict(build_messages(coding_system(), user))
    # predictor.mode e' riletto DOPO predict(): se il backend scelto in init()
    # fallisce a runtime (es. Ollama irraggiungibile), predict() degrada al
    # MockTeacher e aggiorna predictor.mode di conseguenza.
    print(f"[mode={predictor.mode}]\n{answer}")


if __name__ == "__main__":
    main()
