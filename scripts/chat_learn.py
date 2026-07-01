#!/usr/bin/env python
"""Chat locale che impara da ogni scambio: memoria BM25 + log per il retrain.

Ogni domanda/risposta finisce in un log JSONL (default: data/memory/). Alle
domande successive gli scambi passati piu' rilevanti vengono iniettati nel
contesto (apprendimento immediato, senza retrain). Con --export-sft il log
diventa un dataset SFT nello schema del repo, pronto per il retrain QLoRA
periodico (vedi docs/continuous-learning.md).

Richiede solo Ollama in esecuzione (niente torch/transformers).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from italian_llm.data.prompts import SYSTEM_DEFAULT
from italian_llm.memory.interaction_log import (
    append_interaction,
    load_interactions,
    memory_context,
    to_sft_examples,
)
from italian_llm.serving import ollama_client

DEFAULT_MEMORY = os.path.join("data", "memory", "interactions.jsonl")


def _ask(question: str, args) -> str:
    """Un turno: recupera memoria, interroga Ollama, registra lo scambio."""
    system = args.system
    if not args.no_memory:
        ctx = memory_context(load_interactions(args.memory), question, k=args.k)
        if ctx:
            system = (
                f"{system}\n\n"
                "Memoria di scambi precedenti con questo utente (usali se pertinenti):\n"
                f"{ctx}"
            )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    answer = ollama_client.chat(
        args.ollama_model,
        messages,
        host=args.ollama_host,
        temperature=args.temperature,
    )
    append_interaction(args.memory, question, answer, model=args.ollama_model)
    return answer


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Chat via Ollama con memoria persistente delle interazioni."
    )
    p.add_argument("--question", default=None, help="Domanda singola; se assente avvia la REPL.")
    p.add_argument(
        "--memory", default=DEFAULT_MEMORY, help=f"Log JSONL (default: {DEFAULT_MEMORY})"
    )
    p.add_argument("--k", type=int, default=3, help="Scambi passati da iniettare (default: 3).")
    p.add_argument(
        "--no-memory", action="store_true", help="Non iniettare la memoria nel contesto."
    )
    p.add_argument(
        "--export-sft",
        default=None,
        metavar="OUT_JSONL",
        help="Converte il log in dataset SFT (schema repo) ed esce.",
    )
    p.add_argument("--ollama-model", default="coder-local", help="Modello registrato in Ollama.")
    p.add_argument("--ollama-host", default="http://localhost:11434")
    p.add_argument("--system", default=SYSTEM_DEFAULT, help="System prompt di base.")
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args(argv)

    if args.export_sft:
        examples = to_sft_examples(load_interactions(args.memory), system=args.system)
        os.makedirs(os.path.dirname(os.path.abspath(args.export_sft)), exist_ok=True)
        import json

        with open(args.export_sft, "w", encoding="utf-8") as fh:
            for rec in examples:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Esportati {len(examples)} esempi SFT in {args.export_sft}")
        return 0

    try:
        if args.question is not None:
            print(_ask(args.question, args))
            return 0
        print("Chat con memoria (invio a vuoto o 'exit' per uscire).")
        while True:
            try:
                question = input("tu> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question or question.lower() in {"exit", "quit"}:
                break
            print(_ask(question, args))
        return 0
    except ollama_client.OllamaError as e:
        raise SystemExit(
            f"Errore Ollama: {e}\nVerifica che Ollama sia in esecuzione (ollama list)."
        ) from e


if __name__ == "__main__":
    main()
