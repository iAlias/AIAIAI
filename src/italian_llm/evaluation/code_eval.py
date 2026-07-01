"""Harness di valutazione coding eseguibile (pass@1) in stile HumanEval.

Flusso: per ogni problema -> il modello completa il prompt -> si costruisce un
programma (prompt + completamento + test + check) -> si esegue in sandbox
(code_exec.run_python) -> passed bool. L'aggregato pass@k usa
italian_llm.evaluation.metrics.coding_passk sui campi 'passed'.
"""

import re

from italian_llm.evaluation.code_exec import run_python
from italian_llm.utils.io import read_jsonl

__all__ = ["extract_code", "build_program", "load_problems", "evaluate_coding"]

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


def extract_code(raw: str) -> str:
    """Estrae il corpo di codice: primo blocco ``` se presente, altrimenti tutto."""
    if not raw:
        return ""
    m = _FENCE_RE.search(raw)
    if m:
        return m.group(1).rstrip("\n")
    return raw.rstrip("\n")


def build_program(problem: dict, completion: str) -> str:
    """Compone il programma eseguibile in stile HumanEval."""
    prompt = problem.get("prompt", "")
    test = problem.get("test", "")
    entry = problem.get("entry_point", "candidate")
    return f"{prompt}{completion}\n{test}\ncheck({entry})\n"


def load_problems(path: str) -> list[dict]:
    """Carica problemi JSONL con chiavi task_id, prompt, test, entry_point."""
    return list(read_jsonl(path))


def evaluate_coding(problems, predict_fn, timeout: float = 8.0) -> list[dict]:
    """Genera, esegue e valuta. predict_fn(prompt:str)->str (il completamento)."""
    results = []
    for prob in problems:
        raw = predict_fn(prob.get("prompt", ""))
        completion = extract_code(raw)
        program = build_program(prob, completion)
        outcome = run_python(program, timeout=timeout)
        results.append({
            "task_id": prob.get("task_id"),
            "passed": bool(outcome["passed"]),
            "error": outcome["error"],
        })
    return results
