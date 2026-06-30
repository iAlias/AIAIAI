"""Esecuzione SANDBOX di codice non fidato in un sottoprocesso isolato.

Sicurezza: il codice generato dal modello non e' fidato. Lo eseguiamo SEMPRE in
un processo separato (mai exec/eval in-process), con timeout e working dir
temporanea. Niente accesso allo stato dell'interprete chiamante.
"""

import os
import subprocess
import sys
import tempfile

__all__ = ["run_python"]


def run_python(program: str, timeout: float = 8.0) -> dict:
    """Esegue `program` come script Python in un sottoprocesso isolato.

    Ritorna {"passed": bool, "timed_out": bool, "error": str|None}.
    passed=True solo se l'uscita e' 0 entro il timeout.
    """
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "candidate.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(program)
        try:
            proc = subprocess.run(
                [sys.executable, path],
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
        err = (proc.stderr or proc.stdout or "").strip()[-500:]
        return {"passed": False, "timed_out": False, "error": err or "non-zero exit"}
