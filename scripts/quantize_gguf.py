#!/usr/bin/env python
"""CLI per convertire un modello HF in GGUF e quantizzarlo con llama.cpp (o istruzioni manuali)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse
import shutil
import subprocess

from italian_llm.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

# Possibili nomi dello script di conversione HF->GGUF nelle varie versioni di llama.cpp.
_CONVERT_SCRIPTS = ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py")
# Possibili nomi del binario di quantizzazione.
_QUANT_BINARIES = ("llama-quantize", "quantize")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Converte un modello HF in GGUF e lo quantizza tramite llama.cpp.",
    )
    p.add_argument(
        "--in", dest="in_dir", required=True, help="Cartella del modello HF (merge SFT/ORPO)."
    )
    p.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Percorso del file GGUF quantizzato di output.",
    )
    p.add_argument("--quant", default="q4_k_m", help="Tipo di quantizzazione (default: q4_k_m).")
    p.add_argument(
        "--llama-cpp",
        default=os.environ.get("LLAMA_CPP_DIR", ""),
        help="Cartella di llama.cpp (default: variabile d'ambiente LLAMA_CPP_DIR).",
    )
    p.add_argument("--log-level", default="INFO", help="Livello di logging (default: INFO).")
    return p


def _find_convert_script(llama_cpp_dir: str) -> str | None:
    """Cerca lo script di conversione in llama_cpp_dir e nelle sue sottocartelle note."""
    candidates = []
    if llama_cpp_dir:
        for name in _CONVERT_SCRIPTS:
            candidates.append(os.path.join(llama_cpp_dir, name))
    for name in _CONVERT_SCRIPTS:
        found = shutil.which(name)
        if found:
            candidates.append(found)
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _find_quant_binary(llama_cpp_dir: str) -> str | None:
    """Cerca il binario di quantizzazione su PATH o nelle build dir di llama.cpp."""
    for name in _QUANT_BINARIES:
        found = shutil.which(name)
        if found:
            return found
    if llama_cpp_dir:
        for sub in ("", "build/bin", "build"):
            for name in _QUANT_BINARIES:
                for ext in ("", ".exe"):
                    path = os.path.join(llama_cpp_dir, sub, name + ext)
                    if os.path.isfile(path):
                        return path
    return None


def _manual_instructions(in_dir: str, out_path: str, quant: str) -> None:
    """Stampa istruzioni manuali esatte quando llama.cpp non e' disponibile."""
    f16_path = os.path.splitext(out_path)[0] + ".f16.gguf"
    print("\n[!] llama.cpp non trovato: esegui manualmente questi passi.\n")
    print("# 1) Clona e compila llama.cpp")
    print("git clone https://github.com/ggerganov/llama.cpp")
    print(
        "cmake -B llama.cpp/build -S llama.cpp && cmake --build llama.cpp/build -j --config Release"
    )
    print("pip install -r llama.cpp/requirements.txt\n")
    print("# 2) Converti il modello HF in GGUF f16")
    print(f"python llama.cpp/convert_hf_to_gguf.py {in_dir} --outfile {f16_path} --outtype f16\n")
    print("# 3) Quantizza al formato desiderato")
    print(f"llama.cpp/build/bin/llama-quantize {f16_path} {out_path} {quant.upper()}\n")
    print("# Suggerimento: imposta LLAMA_CPP_DIR alla cartella di llama.cpp per automatizzare.")


def main(argv: list[str] | None = None) -> str | None:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if not os.path.isdir(args.in_dir):
        raise SystemExit(f"Cartella modello HF non trovata: '{args.in_dir}'.")

    convert_script = _find_convert_script(args.llama_cpp)
    quant_bin = _find_quant_binary(args.llama_cpp)

    if not convert_script or not quant_bin:
        logger.warning(
            "Toolchain llama.cpp incompleta (convert=%s, quantize=%s).",
            bool(convert_script),
            bool(quant_bin),
        )
        _manual_instructions(args.in_dir, args.out_path, args.quant)
        return None

    out_parent = os.path.dirname(os.path.abspath(args.out_path))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    f16_path = os.path.splitext(args.out_path)[0] + ".f16.gguf"

    # Passo 1: conversione HF -> GGUF f16.
    convert_cmd = [
        sys.executable,
        convert_script,
        args.in_dir,
        "--outfile",
        f16_path,
        "--outtype",
        "f16",
    ]
    logger.info("Conversione HF->GGUF f16: %s", " ".join(convert_cmd))
    try:
        subprocess.run(convert_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Conversione fallita (exit {exc.returncode}). Vedi log llama.cpp."
        ) from exc

    # Passo 2: quantizzazione f16 -> formato richiesto.
    quant_cmd = [quant_bin, f16_path, args.out_path, args.quant.upper()]
    logger.info("Quantizzazione %s: %s", args.quant, " ".join(quant_cmd))
    try:
        subprocess.run(quant_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Quantizzazione fallita (exit {exc.returncode}).") from exc

    logger.info("GGUF quantizzato pronto: %s", args.out_path)
    print(args.out_path)
    return args.out_path


if __name__ == "__main__":
    main()
