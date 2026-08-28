"""Esecuzione SANDBOX di codice non fidato in un sottoprocesso isolato.

Sicurezza: il codice generato dal modello non e' fidato. Lo eseguiamo SEMPRE in
un processo separato (mai exec/eval in-process), con timeout e working dir
temporanea. Niente accesso allo stato dell'interprete chiamante.
"""

import os
import shutil
import subprocess
import sys
import tempfile

__all__ = ["run_python", "run_javascript", "run_program_file"]


def run_program_file(argv_prefix: list[str], filename: str, program: str, timeout: float) -> dict:
    """Scrive `program` in una dir temporanea e lo esegue con `argv_prefix + [path]`.

    Ritorna {"passed": bool, "timed_out": bool, "error": str|None}.
    passed=True solo se l'uscita e' 0 entro il timeout.
    """
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(program)
        try:
            proc = subprocess.run(
                [*argv_prefix, path],
                cwd=workdir,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "timed_out": True, "error": "timeout"}
        if proc.returncode == 0:
            return {"passed": True, "timed_out": False, "error": None}
        err = _abbreviate((proc.stderr or proc.stdout or "").strip())
        return {"passed": False, "timed_out": False, "error": err or "non-zero exit"}


def _abbreviate(text: str, keep: int = 400) -> str:
    """Conserva inizio e fine dell'output: Python mette l'errore in coda, Node in testa."""
    if len(text) <= 2 * keep:
        return text
    return text[:keep] + chr(10) + "..." + chr(10) + text[-keep:]


def run_python(program: str, timeout: float = 8.0) -> dict:
    """Esegue `program` come script Python in un sottoprocesso isolato."""
    return run_program_file([sys.executable], "candidate.py", program, timeout)


def run_javascript(program: str, timeout: float = 8.0) -> dict:
    """Esegue `program` con Node.js in un sottoprocesso isolato (richiede `node` nel PATH)."""
    node = shutil.which("node")
    if node is None:
        return {"passed": False, "timed_out": False, "error": "node non trovato nel PATH"}
    return run_program_file([node], "candidate.js", program, timeout)
