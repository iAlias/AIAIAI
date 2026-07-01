#!/usr/bin/env python
"""Costruisce un dataset SFT coding (schema del repo) da fonti aperte, filtrato a
C#/JS/HTML/CSS. Offline: usa un campione locale. Reale: leggi un JSONL esportato
da un dataset aperto (Magicoder/Evol-Instruct-Code/OSS-Instruct) con campi
instruction/response/language, oppure adatta load_open() con `datasets` (lazy)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse

from italian_llm.data.prompts import coding_system
from italian_llm.data.schema import validate_record
from italian_llm.logging_utils import get_logger, setup_logging
from italian_llm.utils.io import read_jsonl, write_jsonl

logger = get_logger(__name__)

_LANG_KEYWORDS = {
    "csharp": ("c#", "csharp", ".net", "dotnet"),
    "javascript": ("javascript", " js ", "node", "typescript"),
    "html": ("html", "markup"),
    "css": ("css", "flexbox", "stylesheet"),
}


def filter_language(text: str) -> str | None:
    """Mappa il testo a uno dei linguaggi target, altrimenti None."""
    t = f" {(text or '').lower()} "
    for lang, kws in _LANG_KEYWORDS.items():
        if any(k in t for k in kws):
            return lang
    return None


def to_sft_example(instruction: str, response: str, language: str, idx: int) -> dict:
    """Record SFT valido (system coding + user + assistant), domain=linguaggio."""
    messages = [
        {"role": "system", "content": coding_system()},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    return {
        "id": f"coding-{language}-{idx}",
        "source_type": "open_dataset",
        "domain": language,
        "difficulty": "medium",
        "messages": messages,
        "quality_score": 0.0,
        "safety_tag": "allow",
        "italian_score": 0.0,
        "teacher_name": "open_dataset",
    }


def build(in_path: str, out_path: str) -> int:
    """Legge {instruction,response,language}; scrive SFT JSONL valido. Ritorna #righe."""
    out = []
    for i, rec in enumerate(read_jsonl(in_path)):
        instr = rec.get("instruction") or rec.get("prompt") or ""
        resp = rec.get("response") or rec.get("output") or ""
        lang = rec.get("language") or filter_language(instr + " " + resp)
        if not instr or not resp or lang not in _LANG_KEYWORDS:
            continue
        ex = to_sft_example(instr, resp, lang, i)
        ok, reason = validate_record(ex)
        if ok:
            out.append(ex)
        else:
            logger.warning("scarto riga %d: %s", i, reason)
    write_jsonl(out_path, out)
    logger.info("Scritte %d righe SFT coding in %s", len(out), out_path)
    return len(out)


def main(argv=None):
    p = argparse.ArgumentParser(description="Build coding SFT dataset (schema repo).")
    p.add_argument("--in", dest="in_path", default="data/processed/coding_sft.sample.jsonl")
    p.add_argument("--out", dest="out_path", default="data/processed/coding_sft.jsonl")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    build(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
