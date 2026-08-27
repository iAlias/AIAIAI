"""Harness di valutazione coding eseguibile (pass@1) in stile HumanEval.

Flusso: per ogni problema -> il modello completa il prompt -> si costruisce un
programma (prompt + completamento + test + check) -> si esegue in sandbox
(code_exec.run_python) -> passed bool. L'aggregato pass@k usa
italian_llm.evaluation.metrics.coding_passk sui campi 'passed'.
"""

import re

from italian_llm.evaluation.code_exec import run_javascript, run_python
from italian_llm.utils.io import read_jsonl

__all__ = [
    "extract_code",
    "build_program",
    "run_program",
    "load_problems",
    "evaluate_coding",
    "RUNNERS",
]

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)

RUNNERS = {"python": run_python, "javascript": run_javascript}


def _language(problem: dict) -> str:
    return (problem.get("language") or "python").lower()


def _declares_entry(language: str, entry: str, completion: str) -> bool:
    """True se il completamento definisce gia' la funzione `entry` a livello top."""
    name = re.escape(entry)
    if language == "javascript":
        pattern = rf"^\s*(?:(?:async\s+)?function\s+{name}\s*\(|(?:const|let|var)\s+{name}\s*=)"
    else:
        pattern = rf"^\s*(?:async\s+)?def\s+{name}\s*\("
    return re.search(pattern, completion, re.MULTILINE) is not None


def _indent_if_flush_left(completion: str, indent: str = "    ") -> str:
    """Un corpo Python restituito senza indentazione viene rientrato di un livello."""
    lines = completion.split("\n")
    first = next((ln for ln in lines if ln.strip()), "")
    if not first or first[0].isspace():
        return completion
    return "\n".join(indent + ln if ln.strip() else ln for ln in lines)


def extract_code(raw: str) -> str:
    """Estrae il corpo di codice: primo blocco ``` se presente, altrimenti tutto."""
    if not raw:
        return ""
    m = _FENCE_RE.search(raw)
    if m:
        return m.group(1).rstrip("\n")
    return raw.rstrip("\n")


def build_program(problem: dict, completion: str) -> str:
    """Compone il programma eseguibile in stile HumanEval.

    Python: prompt + completamento + test + check(entry). Se il modello ha
    restituito la funzione completa, la si accoda al prompt (che conserva import
    e helper) invece di concatenarla dentro la firma; un corpo senza
    indentazione viene rientrato. JavaScript (MultiPL-E): i test si
    auto-invocano; se il completamento dichiara gia' la funzione, sostituisce il
    prompt, altrimenti ne completa il corpo.
    """
    prompt = problem.get("prompt", "")
    test = problem.get("test", "")
    entry = problem.get("entry_point", "candidate")
    language = _language(problem)
    if language == "javascript":
        body = completion if _declares_entry(language, entry, completion) else prompt + completion
        return f"{body}\n{test}\n"
    if _declares_entry(language, entry, completion):
        return f"{prompt}\n{completion}\n{test}\ncheck({entry})\n"
    return f"{prompt}{_indent_if_flush_left(completion)}\n{test}\ncheck({entry})\n"


def run_program(problem: dict, completion: str, timeout: float = 8.0) -> dict:
    """Compone ed esegue il programma con il runner del linguaggio del problema."""
    runner = RUNNERS.get(_language(problem))
    if runner is None:
        return {
            "passed": False,
            "timed_out": False,
            "error": f"linguaggio non supportato: {_language(problem)}",
        }
    return runner(build_program(problem, completion), timeout=timeout)


def load_problems(path: str) -> list[dict]:
    """Carica problemi JSONL con chiavi task_id, prompt, test, entry_point."""
    return list(read_jsonl(path))


def evaluate_coding(problems, predict_fn, timeout: float = 8.0, on_result=None) -> list[dict]:
    """Genera, esegue e valuta. predict_fn(prompt:str)->str (il completamento).

    `on_result`, se fornito, viene chiamato con ogni risultato appena pronto
    (utile per log incrementali su run lunghi).
    """
    results = []
    for prob in problems:
        raw = predict_fn(prob.get("prompt", ""))
        completion = extract_code(raw)
        outcome = run_program(prob, completion, timeout=timeout)
        result = {
            "task_id": prob.get("task_id"),
            "passed": bool(outcome["passed"]),
            "error": outcome["error"],
            "raw": raw,
        }
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results
