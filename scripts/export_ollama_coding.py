#!/usr/bin/env python
"""Genera un Modelfile Ollama per un GGUF, con system prompt di coding, e (se
ollama c'e') registra il modello."""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import argparse
import shutil
import subprocess

from italian_llm.data.prompts import coding_system

_TEMPLATE = (
    '{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}'
    '{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}'
    '<|im_start|>assistant\n{{ .Response }}<|im_end|>\n'
)


def build_modelfile(gguf_path: str, system: str, *, temperature: float = 0.2,
                    top_p: float = 0.9) -> str:
    """Testo del Modelfile Ollama (template ChatML Qwen + system + sampling)."""
    sys_escaped = system.replace('"', '\\"')
    return (
        f"FROM {gguf_path}\n\n"
        f'TEMPLATE """{_TEMPLATE}"""\n\n'
        f'SYSTEM "{sys_escaped}"\n\n'
        f"PARAMETER temperature {temperature}\n"
        f"PARAMETER top_p {top_p}\n"
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|im_start|>"\n'
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Esporta un GGUF coder in Ollama.")
    p.add_argument("--gguf", required=True, help="Percorso del file .gguf")
    p.add_argument("--name", default="coder-local", help="Nome del modello Ollama")
    p.add_argument("--languages", default="C#,JavaScript,HTML,CSS",
                   help="Linguaggi principali (CSV) per il system prompt")
    p.add_argument("--temperature", type=float, default=0.2)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.isfile(args.gguf):
        raise SystemExit(f"GGUF non trovato: {args.gguf}")
    langs = [s.strip() for s in args.languages.split(",") if s.strip()]
    system = coding_system(langs)
    text = build_modelfile(os.path.abspath(args.gguf), system, temperature=args.temperature)
    modelfile = os.path.join(os.path.dirname(os.path.abspath(args.gguf)), "Modelfile")
    with open(modelfile, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[i] Modelfile scritto: {modelfile}")
    if shutil.which("ollama"):
        subprocess.run(["ollama", "create", args.name, "-f", modelfile], check=True)
        print(f"[ok] Modello '{args.name}' registrato. Avvia: ollama run {args.name}")
    else:
        print("[!] 'ollama' non nel PATH. Installa da https://ollama.com, poi:")
        print(f"    ollama create {args.name} -f {modelfile}")
        print(f"    ollama run {args.name}")


if __name__ == "__main__":
    main()
