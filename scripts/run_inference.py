#!/usr/bin/env python
"""CLI di inferenza: prompt singolo oppure REPL conversazionale tramite serving.Generator."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse

from italian_llm.data.prompts import SYSTEM_DEFAULT, build_messages
from italian_llm.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genera testo con il modello italiano: prompt singolo o REPL conversazionale.",
    )
    p.add_argument("--model", required=True, help="Modello base HF (nome o percorso locale).")
    p.add_argument("--adapter", default=None, help="Adapter LoRA opzionale da applicare.")
    p.add_argument(
        "--prompt", default=None, help="Prompt utente singolo; se assente avvia la REPL."
    )
    p.add_argument(
        "--system", default=SYSTEM_DEFAULT, help="System prompt (default: SYSTEM_DEFAULT italiano)."
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=512, help="Token massimi generati (default: 512)."
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperatura di campionamento (default: 0.7).",
    )
    p.add_argument(
        "--top-p", type=float, default=0.9, help="Nucleus sampling top-p (default: 0.9)."
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device per l'inferenza ('auto', 'cuda', 'cpu'); default: auto.",
    )
    p.add_argument("--log-level", default="INFO", help="Livello di logging (default: INFO).")
    return p


def _gen_kwargs(args: argparse.Namespace) -> dict:
    """Costruisce i kwargs di generazione, abilitando il sampling solo se temperature>0."""
    do_sample = args.temperature is not None and args.temperature > 0.0
    kw = {"max_new_tokens": args.max_new_tokens, "do_sample": do_sample}
    if do_sample:
        kw["temperature"] = args.temperature
        kw["top_p"] = args.top_p
    return kw


def _run_once(generator, args: argparse.Namespace) -> str:
    """Genera una risposta per un prompt singolo e la stampa."""
    messages = build_messages(args.system, args.prompt)
    text = generator.generate(messages, **_gen_kwargs(args))
    print(text)
    return text


def _run_repl(generator, args: argparse.Namespace) -> None:
    """Avvia una REPL conversazionale con storico messaggi e comandi /reset, /system, /exit."""
    print("REPL inferenza italiana. Comandi: /reset (azzera storico), /system <testo>, /exit.\n")
    system = args.system
    history: list[dict] = []
    gen_kw = _gen_kwargs(args)
    while True:
        try:
            user = input("tu> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nUscita.")
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            print("Uscita.")
            break
        if user == "/reset":
            history = []
            print("[storico azzerato]")
            continue
        if user.startswith("/system"):
            system = user[len("/system") :].strip() or SYSTEM_DEFAULT
            history = []
            print("[system aggiornato; storico azzerato]")
            continue

        # Compone i messaggi: system + storico + turno corrente.
        messages = (
            [{"role": "system", "content": system}] + history + [{"role": "user", "content": user}]
        )
        reply = generator.generate(messages, **gen_kw)
        print(f"ai> {reply}\n")
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": reply})


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    from italian_llm.serving.inference import Generator

    logger.info(
        "Carico il modello '%s'%s...",
        args.model,
        f" + adapter '{args.adapter}'" if args.adapter else "",
    )
    generator = Generator(args.model, adapter=args.adapter, device=args.device)

    if args.prompt is not None:
        _run_once(generator, args)
    else:
        _run_repl(generator, args)


if __name__ == "__main__":
    main()
