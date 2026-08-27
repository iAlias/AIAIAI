#!/usr/bin/env python
"""Scarica e converte i set di valutazione coding nello schema del repo.

- HumanEval (OpenAI, MIT): 164 problemi Python -> data/eval/humaneval.jsonl
- MultiPL-E humaneval-js (Northeastern/nuprl, MIT): 161 problemi JavaScript,
  traduzione di HumanEval con test `node:assert` -> data/eval/humaneval_js.jsonl

Schema di uscita (uno per riga): task_id, prompt, entry_point, test, language.
I set sono piccoli (< 1 MB) e vengono versionati per rendere l'eval riproducibile
offline; questo script serve a rigenerarli o aggiornarli.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse
import gzip
import io
import json
import re
import urllib.request

from italian_llm.logging_utils import get_logger, setup_logging
from italian_llm.utils.io import write_jsonl

logger = get_logger(__name__)

HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
MULTIPLE_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=nuprl%2FMultiPL-E&config=humaneval-js&split=test&offset={offset}&length=100"
)

_JS_SIGNATURE_RE = re.compile(r"function\s+([A-Za-z_$][\w$]*)\s*\(")


def humaneval_row_to_problem(row: dict) -> dict:
    """Riga HumanEval originale -> problema eseguibile (scarta canonical_solution)."""
    return {
        "task_id": row["task_id"],
        "prompt": row["prompt"],
        "entry_point": row["entry_point"],
        "test": row["test"],
        "language": "python",
    }


def multiple_js_row_to_problem(row: dict) -> dict:
    """Riga MultiPL-E (humaneval-js) -> problema eseguibile; entry_point dall'ultima firma."""
    prompt = row.get("prompt", "")
    names = _JS_SIGNATURE_RE.findall(prompt)
    if not names:
        raise ValueError(f"{row.get('name')}: nessuna firma 'function nome(' nel prompt")
    return {
        "task_id": row["name"],
        "prompt": prompt,
        "entry_point": names[-1],
        "test": row["tests"],
        "language": "javascript",
    }


def _fetch(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def download_humaneval() -> list[dict]:
    raw = _fetch(HUMANEVAL_URL)
    with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def download_multiple_js() -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        body = json.loads(_fetch(MULTIPLE_ROWS_URL.format(offset=offset)).decode("utf-8"))
        page = [item["row"] for item in body.get("rows", [])]
        rows.extend(page)
        total = body.get("num_rows_total")
        offset += len(page)
        if not page or (total is not None and offset >= total):
            return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Costruisce i set di eval coding (HumanEval + JS).")
    p.add_argument("--out-dir", default="data/eval")
    p.add_argument("--skip-python", action="store_true")
    p.add_argument("--skip-js", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    if not args.skip_python:
        problems = [humaneval_row_to_problem(r) for r in download_humaneval()]
        out = os.path.join(args.out_dir, "humaneval.jsonl")
        write_jsonl(out, problems)
        logger.info("HumanEval: %d problemi -> %s", len(problems), out)

    if not args.skip_js:
        problems = [multiple_js_row_to_problem(r) for r in download_multiple_js()]
        out = os.path.join(args.out_dir, "humaneval_js.jsonl")
        write_jsonl(out, problems)
        logger.info("MultiPL-E humaneval-js: %d problemi -> %s", len(problems), out)


if __name__ == "__main__":
    main()
