"""Loop di auto-riparazione: genera -> esegue test in sandbox -> se fallisce,
reinietta l'errore e ritenta. Trasforma un modello debole in uno piu' affidabile
senza cambiare i pesi: piu' intelligenza dalla stessa rete.
"""

from italian_llm.evaluation.code_eval import build_program, extract_code
from italian_llm.evaluation.code_exec import run_python

__all__ = ["repair_prompt", "solve_with_repair"]


def repair_prompt(problem: dict, last_code: str, error: str) -> str:
    """Prompt di riparazione che mostra codice fallito + errore e chiede la fix."""
    return (
        f"{problem.get('prompt', '')}"
        f"\n# Il tuo tentativo precedente ha fallito i test.\n"
        f"# Codice:\n{last_code}\n"
        f"# Errore:\n{error}\n"
        f"# Correggi: restituisci solo il corpo della funzione corretto.\n"
    )


def solve_with_repair(
    problem: dict, predict_fn, max_attempts: int = 3, timeout: float = 8.0
) -> dict:
    """Tenta fino a max_attempts; ritorna {passed, attempts, code}."""
    prompt = problem.get("prompt", "")
    last_code = ""
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        raw = predict_fn(prompt)
        last_code = extract_code(raw)
        outcome = run_python(build_program(problem, last_code), timeout=timeout)
        if outcome["passed"]:
            return {"passed": True, "attempts": attempt, "code": last_code}
        last_error = outcome["error"] or "test falliti"
        prompt = repair_prompt(problem, last_code, last_error)
    return {"passed": False, "attempts": max_attempts, "code": last_code}
