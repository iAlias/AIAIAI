#!/usr/bin/env python
"""Completamento FIM: legge prefix e suffix, stampa il 'mezzo' generato.

Richiede transformers + il modello coder (gira dove c'e' il modello). Import lazy.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from italian_llm.serving.fim import build_fim_prompt, strip_fim


def main(argv=None):
    p = argparse.ArgumentParser(description="FIM completion con Qwen2.5-Coder.")
    p.add_argument("--prefix", required=True, help="Codice prima del cursore")
    p.add_argument("--suffix", default="", help="Codice dopo il cursore")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=128)
    args = p.parse_args(argv)

    prompt = build_fim_prompt(args.prefix, args.suffix)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise SystemExit(f"transformers non disponibile ({e}). Esegui dove c'e' il modello.") from e

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", trust_remote_code=True
    )
    inputs = tok(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
    print(strip_fim(gen))


if __name__ == "__main__":
    main()
