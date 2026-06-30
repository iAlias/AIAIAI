# Local Zero-Budget Coding Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repurpose the `italian_llm` scaffold into a fast, local, free coding assistant: run Qwen2.5-Coder locally via Ollama, measure it with a real executable pass@k eval (Python + optional JS) plus a qualitative C#/web set, and provide an optional free Kaggle QLoRA path.

**Architecture:** Reuse the existing pure-Python + lazy-heavy-import structure. Add a sandboxed code executor and a coding-eval harness that feeds the existing `coding_passk` metric. Add coding-oriented config, prompts, dataset builder, and an Ollama export with a coding system prompt. Training stays off-device (Kaggle/Colab free GPU); local machine only runs inference.

**Tech Stack:** Python 3 (stdlib + PyYAML for pure modules), pytest, Ollama / llama.cpp (GGUF), Qwen2.5-Coder-1.5B/3B, `datasets`/`transformers`/`peft`/`trl` (lazy, only on GPU host), Kaggle Notebooks.

## Global Constraints

- Heavy deps (`torch`, `transformers`, `peft`, `trl`, `datasets`, `bitsandbytes`, `vllm`) MUST be imported lazily inside functions (repo ADR-009). Pure modules import with stdlib + PyYAML only.
- Generated/untrusted code MUST be executed only in an isolated subprocess with a timeout and a temp working directory. Never `exec()`/`eval()` model output in-process.
- No paid services. No model weights, datasets, or secrets committed to git.
- Tests must run on this host (Windows, no GPU) without torch.
- src layout: scripts prepend `src/` to `sys.path` via the standard header; tests rely on `tests/conftest.py`.
- Package name stays `italian_llm` (rename is out of scope).
- Commit after each task. Run `python -m pytest` (and `python -m ruff check` where touched) before committing.

---

## File Structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `configs/model/qwen_coder.yaml` | Base model = Qwen2.5-Coder-1.5B (+3B note), QLoRA quant keys | New |
| `src/italian_llm/data/prompts.py` | Add `CODING_SYSTEM` + `coding_system()` | Modify |
| `src/italian_llm/evaluation/code_exec.py` | Sandboxed subprocess code runner | New |
| `src/italian_llm/evaluation/code_eval.py` | Load problems, build program, extract code, run pass@k harness | New |
| `data/eval/humaneval.sample.jsonl` | Tiny offline HumanEval-format fixture | New |
| `data/eval/coding_csharp_web.sample.jsonl` | Qualitative C#/JS/HTML/CSS eval set | New |
| `configs/eval/eval_coding.yaml` | Eval config pointing at coding sets | New |
| `scripts/run_coding_eval.py` | CLI: generate + execute + pass@k report | New |
| `scripts/build_coding_sft.py` | Build SFT JSONL from open coding datasets (lazy `datasets`), filtered to target langs | New |
| `data/processed/coding_sft.sample.jsonl` | Tiny offline coding SFT sample | New |
| `scripts/export_ollama_coding.py` | Generate Ollama Modelfile with coding system prompt | New |
| `notebooks/kaggle_qlora_coder.ipynb` | Free Kaggle QLoRA notebook (artifact) | New |
| `tests/test_code_exec.py` | Tests for sandbox executor | New |
| `tests/test_code_eval.py` | Tests for code-eval harness | New |
| `tests/test_coding_prompts.py` | Tests for coding system prompt | New |
| `tests/test_coding_configs.py` | Tests new coding configs load | New |
| `docs/coding-model.md` | Usage + honest limits | New |
| `README.md` | Add coding-assistant section + limits | Modify |

---

## Phase A — Run a strong coder locally (Blocco 1)

### Task A1: Coding model config

**Files:**
- Create: `configs/model/qwen_coder.yaml`
- Test: `tests/test_coding_configs.py`

**Interfaces:**
- Produces: a YAML loadable by `italian_llm.config.load_config`, with keys `model.name`, `model.dtype`, `model.quantization.bnb_4bit_quant_type`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coding_configs.py
import os
from italian_llm import config

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIGS = os.path.join(_REPO_ROOT, "configs")


def test_qwen_coder_model_config_loads():
    cfg = config.load_config(os.path.join(_CONFIGS, "model", "qwen_coder.yaml"))
    assert config.get(cfg, "model.name") == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert config.get(cfg, "model.dtype")
    assert config.get(cfg, "model.quantization.bnb_4bit_quant_type") == "nf4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_coding_configs.py::test_qwen_coder_model_config_loads -v`
Expected: FAIL (file not found / config missing).

- [ ] **Step 3: Create the config**

```yaml
# configs/model/qwen_coder.yaml
# Base coding model. Default = Qwen2.5-Coder-1.5B-Instruct: forte sul codice e
# abbastanza piccolo da girare veloce su CPU una volta quantizzato in GGUF.
# Per piu' capacita' (piu' lento su CPU) usa Qwen/Qwen2.5-Coder-3B-Instruct.
# La quantizzazione 4bit (bitsandbytes) serve SOLO al fine-tuning su GPU; in
# locale il modello gira come GGUF via Ollama (vedi docs/coding-model.md).

model:
  name: Qwen/Qwen2.5-Coder-1.5B-Instruct
  dtype: bfloat16
  trust_remote_code: true
  quantization:
    load_in_4bit: true
    bnb_4bit_compute_dtype: bfloat16
    bnb_4bit_quant_type: nf4
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_coding_configs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/model/qwen_coder.yaml tests/test_coding_configs.py
git commit -m "feat: add Qwen2.5-Coder model config"
```

---

### Task A2: Coding system prompt

**Files:**
- Modify: `src/italian_llm/data/prompts.py`
- Test: `tests/test_coding_prompts.py`

**Interfaces:**
- Produces: `CODING_SYSTEM: str` and `coding_system(languages: list[str] | None = None) -> str` in `italian_llm.data.prompts`. `coding_system(None)` returns `CODING_SYSTEM`; with a list it appends a line naming the languages.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coding_prompts.py
from italian_llm.data import prompts


def test_coding_system_default_is_direct():
    s = prompts.CODING_SYSTEM
    assert "codice" in s.lower()
    assert prompts.coding_system() == s


def test_coding_system_mentions_languages():
    s = prompts.coding_system(["C#", "JavaScript"])
    assert "C#" in s and "JavaScript" in s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_coding_prompts.py -v`
Expected: FAIL (`AttributeError: ... CODING_SYSTEM`).

- [ ] **Step 3: Add to `prompts.py`** (append after `SYSTEM_DEFAULT`)

```python
# System prompt per l'assistente di programmazione: diretto, codice corretto,
# spiegazioni brevi, niente moralismi superflui.
CODING_SYSTEM: str = (
    "Sei un assistente esperto di programmazione: preciso, diretto e pratico. "
    "Scrivi codice corretto, idiomatico e funzionante. "
    "Mostra prima il codice, poi una spiegazione breve solo se utile. "
    "Usa blocchi di codice con il linguaggio indicato. "
    "Se mancano dettagli, assumi i default piu' ragionevoli e dichiarali in una riga. "
    "Niente premesse inutili, niente disclaimer superflui: vai dritto alla soluzione."
)


def coding_system(languages: list[str] | None = None) -> str:
    """System prompt per il coding; se passi dei linguaggi li dichiara esplicitamente."""
    if not languages:
        return CODING_SYSTEM
    langs = ", ".join(languages)
    return CODING_SYSTEM + f" Linguaggi principali dell'utente: {langs}."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_coding_prompts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/italian_llm/data/prompts.py tests/test_coding_prompts.py
git commit -m "feat: add coding system prompt"
```

---

### Task A3: Ollama export with coding system prompt

**Files:**
- Create: `scripts/export_ollama_coding.py`
- Test: `tests/test_code_eval.py` is unrelated; add `tests/test_export_ollama.py`

**Interfaces:**
- Produces: `build_modelfile(gguf_path: str, system: str, *, temperature: float = 0.2, top_p: float = 0.9) -> str` returning Modelfile text with a Qwen ChatML `TEMPLATE`, `FROM <gguf_path>`, the given `SYSTEM`, and sampling params. CLI writes it next to the GGUF and (if `ollama` present) runs `ollama create`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export_ollama.py
import importlib.util
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "export_ollama_coding.py")


def _load():
    spec = importlib.util.spec_from_file_location("export_ollama_coding", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_modelfile_contains_system_and_from():
    mod = _load()
    text = mod.build_modelfile("/tmp/model.q4_k_m.gguf", "Sei un assistente di codice.")
    assert "FROM /tmp/model.q4_k_m.gguf" in text
    assert "Sei un assistente di codice." in text
    assert "<|im_start|>assistant" in text
    assert "PARAMETER temperature 0.2" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export_ollama.py -v`
Expected: FAIL (script missing).

- [ ] **Step 3: Create the script**

```python
#!/usr/bin/env python
"""Genera un Modelfile Ollama per un GGUF, con system prompt di coding, e (se
ollama c'e') registra il modello."""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import argparse
import shutil
import subprocess

from italian_llm.data.prompts import coding_system

_TEMPLATE = (
    '{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}'
    '{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}'
    '<|im_start|>assistant\n{{ .Response }}<|im_end|>\n'
)


def build_modelfile(gguf_path: str, system: str, *, temperature: float = 0.2,
                    top_p: float = 0.9) -> str:
    """Testo del Modelfile Ollama (template ChatML Qwen + system + sampling)."""
    sys_escaped = system.replace('"', '\\"')
    return (
        f"FROM {gguf_path}\n\n"
        f'TEMPLATE """{_TEMPLATE}"""\n\n'
        f'SYSTEM "{sys_escaped}"\n\n'
        f"PARAMETER temperature {temperature}\n"
        f"PARAMETER top_p {top_p}\n"
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|im_start|>"\n'
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Esporta un GGUF coder in Ollama.")
    p.add_argument("--gguf", required=True, help="Percorso del file .gguf")
    p.add_argument("--name", default="coder-local", help="Nome del modello Ollama")
    p.add_argument("--languages", default="C#,JavaScript,HTML,CSS",
                   help="Linguaggi principali (CSV) per il system prompt")
    p.add_argument("--temperature", type=float, default=0.2)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.isfile(args.gguf):
        raise SystemExit(f"GGUF non trovato: {args.gguf}")
    langs = [s.strip() for s in args.languages.split(",") if s.strip()]
    system = coding_system(langs)
    text = build_modelfile(os.path.abspath(args.gguf), system, temperature=args.temperature)
    modelfile = os.path.join(os.path.dirname(os.path.abspath(args.gguf)), "Modelfile")
    with open(modelfile, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[i] Modelfile scritto: {modelfile}")
    if shutil.which("ollama"):
        subprocess.run(["ollama", "create", args.name, "-f", modelfile], check=True)
        print(f"[ok] Modello '{args.name}' registrato. Avvia: ollama run {args.name}")
    else:
        print("[!] 'ollama' non nel PATH. Installa da https://ollama.com, poi:")
        print(f"    ollama create {args.name} -f {modelfile}")
        print(f"    ollama run {args.name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_export_ollama.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_ollama_coding.py tests/test_export_ollama.py
git commit -m "feat: Ollama export with coding system prompt"
```

---

## Phase B — Real coding eval (Blocco 2)

### Task B1: Sandboxed code executor

**Files:**
- Create: `src/italian_llm/evaluation/code_exec.py`
- Test: `tests/test_code_exec.py`

**Interfaces:**
- Produces: `run_python(program: str, timeout: float = 8.0) -> dict` returning `{"passed": bool, "timed_out": bool, "error": str | None}`. Runs `program` as a fresh `sys.executable` subprocess in a temp cwd; `passed` iff exit code 0 within `timeout`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_code_exec.py
from italian_llm.evaluation.code_exec import run_python


def test_passing_program():
    r = run_python("assert 1 + 1 == 2\n")
    assert r["passed"] is True
    assert r["timed_out"] is False


def test_failing_program():
    r = run_python("assert 1 + 1 == 3\n")
    assert r["passed"] is False
    assert r["error"]


def test_timeout_program():
    r = run_python("while True:\n    pass\n", timeout=1.0)
    assert r["passed"] is False
    assert r["timed_out"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_code_exec.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Create the executor**

```python
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
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "timed_out": True, "error": "timeout"}
        if proc.returncode == 0:
            return {"passed": True, "timed_out": False, "error": None}
        err = (proc.stderr or proc.stdout or "").strip()[-500:]
        return {"passed": False, "timed_out": False, "error": err or "non-zero exit"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_code_exec.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/italian_llm/evaluation/code_exec.py tests/test_code_exec.py
git commit -m "feat: sandboxed subprocess code executor"
```

---

### Task B2: Code-eval harness (load, extract, build, pass@k)

**Files:**
- Create: `src/italian_llm/evaluation/code_eval.py`
- Create: `data/eval/humaneval.sample.jsonl`
- Test: `tests/test_code_eval.py`

**Interfaces:**
- Consumes: `run_python` from Task B1.
- Produces:
  - `extract_code(raw: str) -> str` — strips markdown fences, returns code body.
  - `build_program(problem: dict, completion: str) -> str` — `problem["prompt"] + completion + "\n" + problem["test"] + f"\ncheck({problem['entry_point']})\n"`.
  - `load_problems(path: str) -> list[dict]` — reads JSONL with keys `task_id, prompt, test, entry_point`.
  - `evaluate_coding(problems: list[dict], predict_fn, timeout: float = 8.0) -> list[dict]` — for each problem calls `predict_fn(problem["prompt"]) -> str`, executes, returns records `{"task_id", "passed": bool, "error"}`. `predict_fn` takes the prompt string and returns the model completion.

- [ ] **Step 1: Create the offline fixture**

`data/eval/humaneval.sample.jsonl` (2 lines, HumanEval format):

```json
{"task_id": "sample/0", "prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n", "entry_point": "add", "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n"}
{"task_id": "sample/1", "prompt": "def is_even(n):\n    \"\"\"Return True if n is even.\"\"\"\n", "entry_point": "is_even", "test": "def check(candidate):\n    assert candidate(2) is True\n    assert candidate(3) is False\n"}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_code_eval.py
import os
from italian_llm.evaluation import code_eval

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIX = os.path.join(_REPO_ROOT, "data", "eval", "humaneval.sample.jsonl")


def test_extract_code_strips_fences():
    raw = "Ecco:\n```python\n    return a + b\n```\nfine"
    assert code_eval.extract_code(raw).strip() == "return a + b"


def test_build_program_appends_test_and_check():
    problem = {"prompt": "def f():\n", "test": "def check(c):\n    pass\n", "entry_point": "f"}
    prog = code_eval.build_program(problem, "    return 1\n")
    assert "def f():" in prog and "check(f)" in prog


def test_load_problems_reads_fixture():
    problems = code_eval.load_problems(_FIX)
    assert len(problems) == 2
    assert problems[0]["entry_point"] == "add"


def test_evaluate_coding_correct_solution_passes():
    problems = code_eval.load_problems(_FIX)

    def predict(prompt):
        if "add" in prompt:
            return "    return a + b\n"
        return "    return n % 2 == 0\n"

    results = code_eval.evaluate_coding(problems, predict)
    assert all(r["passed"] for r in results)


def test_evaluate_coding_wrong_solution_fails():
    problems = code_eval.load_problems(_FIX)
    results = code_eval.evaluate_coding(problems, lambda p: "    return None\n")
    assert all(r["passed"] is False for r in results)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_code_eval.py -v`
Expected: FAIL (module missing).

- [ ] **Step 4: Create the harness**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_code_eval.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add src/italian_llm/evaluation/code_eval.py data/eval/humaneval.sample.jsonl tests/test_code_eval.py
git commit -m "feat: executable coding eval harness (pass@1)"
```

---

### Task B3: Coding eval CLI + config + qualitative C#/web set

**Files:**
- Create: `scripts/run_coding_eval.py`
- Create: `configs/eval/eval_coding.yaml`
- Create: `data/eval/coding_csharp_web.sample.jsonl`
- Test: extend `tests/test_coding_configs.py`

**Interfaces:**
- Consumes: `code_eval.load_problems`, `code_eval.evaluate_coding`, `metrics.coding_passk`, the `_Predictor` pattern from `evaluation/runner.py` (real Generator with MockTeacher fallback).
- Produces: a CLI `python scripts/run_coding_eval.py --config configs/eval/eval_coding.yaml` that writes a JSON report with `pass_at_1` and per-task results to `eval_coding.output.report_path`.

- [ ] **Step 1: Write the failing config test** (append to `tests/test_coding_configs.py`)

```python
def test_eval_coding_config_loads():
    cfg = config.load_config(os.path.join(_CONFIGS, "eval", "eval_coding.yaml"))
    assert config.get(cfg, "eval_coding.exec_set")
    assert config.get(cfg, "eval_coding.model_path")
    assert config.get(cfg, "eval_coding.output.report_path")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_coding_configs.py::test_eval_coding_config_loads -v`
Expected: FAIL.

- [ ] **Step 3: Create the config**

```yaml
# configs/eval/eval_coding.yaml
# Valutazione coding. 'exec_set' = problemi eseguibili (HumanEval-format, Python)
# per pass@1 reale. 'qualitative_set' = prompt C#/JS/HTML/CSS senza esecuzione
# (ispezione manuale + metriche testuali). Punta i path ai dataset reali quando
# li scarichi; di default usa i campioni offline versionati.
_base_: ../base.yaml

eval_coding:
  model_path: Qwen/Qwen2.5-Coder-1.5B-Instruct
  adapter: ""                 # LoRA opzionale ("" = nessuno)
  exec_set: data/eval/humaneval.sample.jsonl
  qualitative_set: data/eval/coding_csharp_web.sample.jsonl
  max_new_tokens: 512
  temperature: 0.0
  timeout_s: 8.0
  output:
    report_path: outputs/eval/coding_report.json
```

- [ ] **Step 4: Create the qualitative set fixture**

`data/eval/coding_csharp_web.sample.jsonl`:

```json
{"id": "cs/0", "domain": "csharp", "prompt": "In C#, scrivi un metodo che inverte una stringa.", "reference": "public static string Reverse(string s){var a=s.ToCharArray();Array.Reverse(a);return new string(a);}"}
{"id": "js/0", "domain": "javascript", "prompt": "In JavaScript, scrivi una funzione che rimuove i duplicati da un array.", "reference": "const unique = arr => [...new Set(arr)];"}
{"id": "css/0", "domain": "css", "prompt": "Scrivi il CSS per centrare un div sia orizzontalmente che verticalmente con flexbox.", "reference": ".parent{display:flex;justify-content:center;align-items:center;}"}
{"id": "html/0", "domain": "html", "prompt": "Scrivi un form HTML accessibile con un campo email e un bottone di invio.", "reference": "<form><label for=\"e\">Email</label><input id=\"e\" type=\"email\" required><button type=\"submit\">Invia</button></form>"}
```

- [ ] **Step 5: Create the CLI**

```python
#!/usr/bin/env python
"""Valutazione coding: pass@1 eseguibile (Python) + set qualitativo C#/web."""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import argparse
import json

from italian_llm.config import load_config, get
from italian_llm.logging_utils import setup_logging, get_logger
from italian_llm.evaluation import code_eval
from italian_llm.evaluation import metrics as M
from italian_llm.utils.io import ensure_dir

logger = get_logger(__name__)


def _make_predictor(cfg: dict):
    """Generator reale con fallback MockTeacher (riusa la logica del runner)."""
    from italian_llm.evaluation.runner import _Predictor

    sub = {"eval": {
        "model_path": get(cfg, "eval_coding.model_path"),
        "adapter": get(cfg, "eval_coding.adapter", ""),
        "max_new_tokens": get(cfg, "eval_coding.max_new_tokens", 512),
        "temperature": get(cfg, "eval_coding.temperature", 0.0),
    }}
    pred = _Predictor(sub)
    pred.init()

    def predict_fn(prompt: str) -> str:
        from italian_llm.data.prompts import coding_system, build_messages
        messages = build_messages(coding_system(), prompt)
        return pred.predict(messages)

    return predict_fn, pred.mode


def main(argv=None):
    p = argparse.ArgumentParser(description="Valutazione coding (pass@1 + qualitativo).")
    p.add_argument("--config", default="configs/eval/eval_coding.yaml")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    predict_fn, mode = _make_predictor(cfg)
    logger.info("Predizioni in modalita': %s", mode)

    exec_set = get(cfg, "eval_coding.exec_set")
    timeout = float(get(cfg, "eval_coding.timeout_s", 8.0))
    problems = code_eval.load_problems(exec_set)
    results = code_eval.evaluate_coding(problems, predict_fn, timeout=timeout)
    pass_at_1 = M.coding_passk([r["passed"] for r in results])

    report = {
        "generation_mode": mode,
        "model_path": get(cfg, "eval_coding.model_path"),
        "exec_set": exec_set,
        "n_exec": len(results),
        "pass_at_1": round(pass_at_1, 4),
        "results": results,
    }
    report_path = get(cfg, "eval_coding.output.report_path")
    ensure_dir(os.path.dirname(report_path) or ".")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    logger.info("pass@1 = %.4f (mode=%s). Report: %s", pass_at_1, mode, report_path)
    print(json.dumps({"pass_at_1": report["pass_at_1"], "mode": mode}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the config test + a smoke run of the CLI**

Run: `python -m pytest tests/test_coding_configs.py -v`
Expected: PASS.

Run: `python scripts/run_coding_eval.py --config configs/eval/eval_coding.yaml`
Expected: prints JSON with `pass_at_1` and `"mode": "mock"` (no GPU here → MockTeacher; pass@1 likely 0.0, which is correct: the mock can't solve coding). Writes `outputs/eval/coding_report.json`.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_coding_eval.py configs/eval/eval_coding.yaml data/eval/coding_csharp_web.sample.jsonl tests/test_coding_configs.py
git commit -m "feat: coding eval CLI (pass@1 + qualitative set)"
```

---

## Phase C — Optional free fine-tuning (Blocco 3)

### Task C1: Coding SFT dataset builder

**Files:**
- Create: `scripts/build_coding_sft.py`
- Create: `data/processed/coding_sft.sample.jsonl`
- Test: `tests/test_build_coding_sft.py`

**Interfaces:**
- Consumes: `italian_llm.data.schema.SFTExample`, `validate_record`; `italian_llm.data.prompts.coding_system`.
- Produces:
  - `to_sft_example(instruction: str, response: str, language: str, idx: int) -> dict` — returns a dict validating against `validate_record` (chat `messages` with system=coding, user=instruction, assistant=response; `domain=language`, `source_type="open_dataset"`).
  - `filter_language(text: str) -> str | None` — returns one of `{"csharp","javascript","html","css"}` if detectable, else None.
  - CLI: reads an input JSONL (`--in`) of `{instruction,response,language}` and writes valid SFT JSONL (`--out`); offline default input = the sample fixture.

- [ ] **Step 1: Create the offline sample**

`data/processed/coding_sft.sample.jsonl`:

```json
{"instruction": "In C#, scrivi un metodo che somma due interi.", "response": "```csharp\npublic static int Add(int a, int b) => a + b;\n```", "language": "csharp"}
{"instruction": "In JavaScript, inverti una stringa.", "response": "```javascript\nconst reverse = s => [...s].reverse().join('');\n```", "language": "javascript"}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_build_coding_sft.py
import importlib.util
import os

from italian_llm.data.schema import validate_record

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "build_coding_sft.py")


def _load():
    spec = importlib.util.spec_from_file_location("build_coding_sft", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_to_sft_example_is_valid():
    mod = _load()
    rec = mod.to_sft_example("Scrivi add in C#", "```csharp\nint Add()=>0;\n```", "csharp", 0)
    ok, reason = validate_record(rec)
    assert ok, reason
    assert rec["domain"] == "csharp"
    assert rec["messages"][0]["role"] == "system"


def test_filter_language_known_and_unknown():
    mod = _load()
    assert mod.filter_language("usando C# e .NET") == "csharp"
    assert mod.filter_language("in python") is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_build_coding_sft.py -v`
Expected: FAIL.

- [ ] **Step 4: Create the builder**

```python
#!/usr/bin/env python
"""Costruisce un dataset SFT coding (schema del repo) da fonti aperte, filtrato a
C#/JS/HTML/CSS. Offline: usa un campione locale. Reale: leggi un JSONL esportato
da un dataset aperto (Magicoder/Evol-Instruct-Code/OSS-Instruct) con campi
instruction/response/language, oppure adatta load_open() con `datasets` (lazy)."""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import argparse

from italian_llm.data.prompts import coding_system
from italian_llm.data.schema import validate_record
from italian_llm.utils.io import read_jsonl, write_jsonl
from italian_llm.logging_utils import setup_logging, get_logger

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
```

- [ ] **Step 5: Run tests + smoke build**

Run: `python -m pytest tests/test_build_coding_sft.py -v`
Expected: PASS.

Run: `python scripts/build_coding_sft.py`
Expected: writes `data/processed/coding_sft.jsonl` (2 valid rows). (This output is git-ignored or add the `.sample` only — do NOT commit the generated full file.)

- [ ] **Step 6: Commit**

```bash
git add scripts/build_coding_sft.py data/processed/coding_sft.sample.jsonl tests/test_build_coding_sft.py
git commit -m "feat: coding SFT dataset builder"
```

---

### Task C2: Coding SFT training config

**Files:**
- Create: `configs/train/sft_coder.yaml`
- Test: existing `tests/test_train_config.py` auto-parametrizes over `configs/train/*.yaml` — the new file must satisfy `test_train_config_resolves_base_and_required_keys`.

**Interfaces:**
- Produces: a train config with `_base_: ../base.yaml`, `model.name` Qwen Coder, `model.quantization.load_in_4bit`, full `train.*` keys, `lora.r` + `lora.target_modules`, `data.train_path` + `data.format`.

- [ ] **Step 1: Inspect an existing train config to mirror keys**

Run: `python -c "from italian_llm import config; import json; print(json.dumps(config.load_config('configs/train/sft_qwen9b_lora.yaml'), indent=2, ensure_ascii=False))"`
Expected: prints the full resolved config; note every `train.*`, `lora.*`, `data.*` key present.

- [ ] **Step 2: Create the config mirroring those keys**

```yaml
# configs/train/sft_coder.yaml
# SFT QLoRA del coder su dati coding. Da lanciare su GPU gratis (Kaggle/Colab),
# NON su questo PC. Mirror delle chiavi di sft_qwen9b_lora.yaml.
_base_: ../base.yaml

model:
  name: Qwen/Qwen2.5-Coder-1.5B-Instruct
  dtype: bfloat16
  trust_remote_code: true
  quantization:
    load_in_4bit: true
    bnb_4bit_compute_dtype: bfloat16
    bnb_4bit_quant_type: nf4

train:
  output_dir: outputs/sft_coder_lora
  epochs: 1
  lr: 0.0002
  batch_size: 1
  grad_accum: 16
  max_seq_len: 2048
  warmup_ratio: 0.03
  weight_decay: 0.0
  logging_steps: 10
  save_steps: 200
  gradient_checkpointing: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

data:
  train_path: data/processed/coding_sft.jsonl
  format: chat
```

> Before writing, confirm these key names against Step 1 output. If `sft_qwen9b_lora.yaml` uses different names (e.g. `data.path` instead of `data.train_path`), match the existing file exactly.

- [ ] **Step 3: Run the train-config test suite**

Run: `python -m pytest tests/test_train_config.py -v`
Expected: PASS (the new file is auto-included by the glob and must pass `test_train_config_resolves_base_and_required_keys`).

- [ ] **Step 4: Commit**

```bash
git add configs/train/sft_coder.yaml
git commit -m "feat: coding SFT QLoRA train config"
```

---

### Task C3: Kaggle QLoRA notebook (artifact)

**Files:**
- Create: `notebooks/kaggle_qlora_coder.ipynb`

**Interfaces:** none (documentation artifact run on Kaggle, not unit-tested here).

- [ ] **Step 1: Create the notebook**

Create `notebooks/kaggle_qlora_coder.ipynb` as a JSON notebook with these cells (markdown + code). Code cells, in order:

Cell 1 (markdown): title + instructions: "Run on Kaggle with GPU T4 x2 enabled (Settings → Accelerator). Free ~30h/week. Upload this repo as a Kaggle dataset or `git clone` it. Upload your `coding_sft.jsonl`."

Cell 2 (code):
```python
!pip -q install -U transformers peft trl bitsandbytes accelerate datasets
```

Cell 3 (code):
```python
import os, sys
REPO = "/kaggle/working/AIAIAI"   # adjust to your clone path
sys.path.insert(0, os.path.join(REPO, "src"))
DATA = "/kaggle/input/coding-sft/coding_sft.jsonl"  # adjust to your uploaded dataset
```

Cell 4 (code):
```python
from italian_llm.config import load_config
from italian_llm.training.sft import run_sft   # reuses the repo's SFT loop
cfg = load_config(os.path.join(REPO, "configs/train/sft_coder.yaml"))
cfg["data"]["train_path"] = DATA
cfg["train"]["output_dir"] = "/kaggle/working/sft_coder_lora"
run_sft(cfg)
```

Cell 5 (markdown): "After training: merge adapter, convert to GGUF (`scripts/quantize_gguf.py`), download the `.gguf`, then locally run `python scripts/export_ollama_coding.py --gguf <file>.gguf --name coder-local`."

Cell 6 (code):
```python
# Optional: merge LoRA into base and save merged HF model for GGUF conversion
from italian_llm.training.common import load_model_and_tokenizer  # if available
# If the repo exposes a merge helper use it; otherwise:
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", torch_dtype="auto")
merged = PeftModel.from_pretrained(base, "/kaggle/working/sft_coder_lora").merge_and_unload()
merged.save_pretrained("/kaggle/working/coder_merged")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct").save_pretrained("/kaggle/working/coder_merged")
print("Merged model at /kaggle/working/coder_merged — download and convert to GGUF.")
```

> Note: verify `run_sft` and `load_model_and_tokenizer` signatures against `src/italian_llm/training/sft.py` and `common.py` while creating the notebook; adjust the calls to match the actual API. If `run_sft` reads `data.train_path` and `train.output_dir`, the overrides above are sufficient.

- [ ] **Step 2: Validate the notebook is valid JSON**

Run: `python -c "import json; json.load(open('notebooks/kaggle_qlora_coder.ipynb', encoding='utf-8')); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/kaggle_qlora_coder.ipynb
git commit -m "docs: Kaggle free QLoRA notebook for coder"
```

---

## Phase D — Docs & honesty

### Task D1: Coding-model docs + README section

**Files:**
- Create: `docs/coding-model.md`
- Modify: `README.md`

**Interfaces:** none.

- [ ] **Step 1: Write `docs/coding-model.md`**

Cover, with exact commands:
1. **What this is / is not** — small specialized local coder; NOT a frontier model; strong on snippets/Q&A, weak on complex multi-file/agentic work. Copy the honest comparison table from the spec.
2. **Run locally (no training):**
   - Install Ollama (https://ollama.com).
   - `ollama pull qwen2.5-coder:1.5b` (quick path) OR build your own GGUF.
   - Or build-your-own: `python scripts/quantize_gguf.py --in <hf_model_dir> --out outputs/gguf/coder.q4_k_m.gguf` then `python scripts/export_ollama_coding.py --gguf outputs/gguf/coder.q4_k_m.gguf --name coder-local`.
   - `ollama run coder-local`.
3. **Measure it:** `python scripts/run_coding_eval.py --config configs/eval/eval_coding.yaml` → reads `pass_at_1` from `outputs/eval/coding_report.json`. Explain mock-mode caveat (0.0 without a real model).
4. **Optional fine-tune (free):** build data (`scripts/build_coding_sft.py`), run `notebooks/kaggle_qlora_coder.ipynb` on Kaggle, convert to GGUF, export to Ollama.
5. **Cost:** €0 path documented.

- [ ] **Step 2: Add a "Coding assistant" section to `README.md`** linking `docs/coding-model.md`, stating the honest scope (small/fast/local/free, not frontier), near the top of the README.

- [ ] **Step 3: Commit**

```bash
git add docs/coding-model.md README.md
git commit -m "docs: coding model usage and honest limits"
```

---

### Task D2: Full test + lint gate

**Files:** none (verification task).

- [ ] **Step 1: Run the whole suite**

Run: `python -m pytest -q`
Expected: all tests PASS (existing + new).

- [ ] **Step 2: Lint the touched files**

Run: `python -m ruff check src scripts tests`
Expected: no errors (fix any). `black` formatting: `python -m black src scripts tests` then re-check.

- [ ] **Step 3: Smoke the coding eval end-to-end (mock mode)**

Run: `python scripts/run_coding_eval.py --config configs/eval/eval_coding.yaml`
Expected: prints `{"pass_at_1": ..., "mode": "mock"}` and writes the report. (Value 0.0 in mock mode is expected and correct.)

- [ ] **Step 4: Commit any lint/format fixes**

```bash
git add -A
git commit -m "chore: lint/format coding pipeline"
```

---

## Phase E — "Genial infrastructure" features

These make a small model punch above its weight without a bigger model. All CPU-friendly, local, free. Pure logic is TDD-tested; model-dependent CLIs degrade to MockTeacher offline.

### Task E1: RAG over the local codebase

**Files:**
- Create: `src/italian_llm/rag/__init__.py` (empty)
- Create: `src/italian_llm/rag/code_index.py`
- Create: `scripts/rag_ask.py`
- Test: `tests/test_rag.py`

**Interfaces:**
- Produces:
  - `chunk_text(path: str, text: str, max_lines: int = 40) -> list[dict]` → list of `{"path","start_line","text"}`.
  - `index_paths(root: str, exts: tuple = (".cs",".js",".ts",".jsx",".tsx",".html",".css",".py"), max_lines: int = 40) -> list[dict]` — walks `root`, chunks every file with a matching extension.
  - `CodeIndex` with classmethod `build(chunks: list[dict]) -> "CodeIndex"` and `search(query: str, k: int = 4) -> list[dict]` (BM25 ranking, stdlib only).
  - `build_context(chunks: list[dict], max_chars: int = 2000) -> str` — formats retrieved chunks as a `# file:line` context block.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag.py
from italian_llm.rag.code_index import chunk_text, CodeIndex, build_context


def test_chunk_text_splits_by_lines():
    text = "\n".join(f"line{i}" for i in range(100))
    chunks = chunk_text("a.py", text, max_lines=40)
    assert len(chunks) == 3
    assert chunks[0]["path"] == "a.py"
    assert chunks[0]["start_line"] == 1
    assert chunks[1]["start_line"] == 41


def test_bm25_ranks_relevant_chunk_first():
    chunks = [
        {"path": "auth.cs", "start_line": 1, "text": "public bool ValidateToken(string jwt) { return true; }"},
        {"path": "math.cs", "start_line": 1, "text": "public int Add(int a, int b) { return a + b; }"},
    ]
    idx = CodeIndex.build(chunks)
    top = idx.search("come valido un token jwt", k=1)
    assert top and top[0]["path"] == "auth.cs"


def test_build_context_includes_paths_and_caps_size():
    chunks = [{"path": "a.cs", "start_line": 5, "text": "X" * 50}]
    ctx = build_context(chunks, max_chars=1000)
    assert "a.cs:5" in ctx
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rag.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Create `src/italian_llm/rag/__init__.py`** (empty file) and `src/italian_llm/rag/code_index.py`

```python
"""RAG leggero sul codebase locale: chunking + retrieval BM25 (solo stdlib).

Idea: un modello piccolo risponde molto meglio se gli si dà il pezzo GIUSTO del
TUO codice. Niente dipendenze pesanti: tokenizzazione regex + BM25 in puro Python.
"""

import math
import os
import re
from collections import Counter

__all__ = ["chunk_text", "index_paths", "CodeIndex", "build_context"]

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def chunk_text(path: str, text: str, max_lines: int = 40) -> list[dict]:
    """Spezza il testo in chunk di al massimo `max_lines` righe."""
    lines = (text or "").splitlines()
    chunks = []
    for start in range(0, max(len(lines), 1), max_lines):
        body = "\n".join(lines[start:start + max_lines])
        if body.strip():
            chunks.append({"path": path, "start_line": start + 1, "text": body})
    return chunks


def index_paths(root: str, exts: tuple = (".cs", ".js", ".ts", ".jsx", ".tsx",
                                          ".html", ".css", ".py"),
                max_lines: int = 40) -> list[dict]:
    """Indicizza ricorsivamente i file con estensione in `exts` sotto `root`."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in
                       {".git", "node_modules", "__pycache__", "outputs", ".venv"}]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                full = os.path.join(dirpath, fn)
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                        out.extend(chunk_text(full, fh.read(), max_lines=max_lines))
                except OSError:
                    continue
    return out


class CodeIndex:
    """Indice BM25 in memoria sui chunk di codice."""

    def __init__(self, chunks, doc_tokens, df, avgdl, k1=1.5, b=0.75):
        self.chunks = chunks
        self.doc_tokens = doc_tokens
        self.df = df
        self.avgdl = avgdl
        self.n = len(chunks)
        self.k1 = k1
        self.b = b

    @classmethod
    def build(cls, chunks: list[dict]) -> "CodeIndex":
        doc_tokens = [_tokens(c["text"]) for c in chunks]
        df = Counter()
        for toks in doc_tokens:
            for term in set(toks):
                df[term] += 1
        avgdl = (sum(len(t) for t in doc_tokens) / len(doc_tokens)) if doc_tokens else 0.0
        return cls(chunks, doc_tokens, df, avgdl)

    def _idf(self, term: str) -> float:
        n_q = self.df.get(term, 0)
        if n_q == 0:
            return 0.0
        return math.log(1 + (self.n - n_q + 0.5) / (n_q + 0.5))

    def search(self, query: str, k: int = 4) -> list[dict]:
        q_terms = _tokens(query)
        scores = []
        for i, toks in enumerate(self.doc_tokens):
            if not toks:
                scores.append((0.0, i))
                continue
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self._idf(term) * (f * (self.k1 + 1)) / denom
            scores.append((s, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [self.chunks[i] for s, i in scores[:k] if s > 0]


def build_context(chunks: list[dict], max_chars: int = 2000) -> str:
    """Formatta i chunk recuperati come blocco di contesto, troncando a max_chars."""
    parts = []
    used = 0
    for c in chunks:
        header = f"# {c['path']}:{c['start_line']}\n"
        block = header + c["text"]
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rag.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Create `scripts/rag_ask.py`** (thin CLI: index → retrieve → ask)

```python
#!/usr/bin/env python
"""Domanda al coder con contesto RAG dal tuo codebase locale."""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import argparse

from italian_llm.rag.code_index import index_paths, CodeIndex, build_context
from italian_llm.data.prompts import coding_system, build_messages
from italian_llm.evaluation.runner import _Predictor
from italian_llm.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


def main(argv=None):
    p = argparse.ArgumentParser(description="Chiedi al coder con RAG sul tuo codice.")
    p.add_argument("--root", default=".", help="Cartella del codebase da indicizzare")
    p.add_argument("--question", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    chunks = index_paths(args.root)
    logger.info("Indicizzati %d chunk da %s", len(chunks), args.root)
    idx = CodeIndex.build(chunks)
    hits = idx.search(args.question, k=args.k)
    context = build_context(hits)

    predictor = _Predictor({"eval": {"model_path": args.model, "adapter": "",
                                     "max_new_tokens": 512, "temperature": 0.2}})
    mode = predictor.init()
    user = (f"Contesto dal codebase:\n{context}\n\nDomanda: {args.question}"
            if context else args.question)
    answer = predictor.predict(build_messages(coding_system(), user))
    print(f"[mode={mode}]\n{answer}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke run + commit**

Run: `python scripts/rag_ask.py --root src --question "come e' strutturata la config?"`
Expected: prints `[mode=mock]` + a (mock) answer; proves indexing + retrieval + predictor wiring work offline.

```bash
git add src/italian_llm/rag/ scripts/rag_ask.py tests/test_rag.py
git commit -m "feat: lightweight BM25 RAG over local codebase"
```

---

### Task E2: Self-repair loop (automatic TDD)

**Files:**
- Create: `src/italian_llm/evaluation/self_repair.py`
- Test: `tests/test_self_repair.py`

**Interfaces:**
- Consumes: `code_eval.build_program`, `code_eval.extract_code`, `code_exec.run_python`.
- Produces:
  - `repair_prompt(problem: dict, last_code: str, error: str) -> str`.
  - `solve_with_repair(problem: dict, predict_fn, max_attempts: int = 3, timeout: float = 8.0) -> dict` → `{"passed": bool, "attempts": int, "code": str}`. Attempt 1 calls `predict_fn(problem["prompt"])`; on failure, calls `predict_fn(repair_prompt(...))` up to `max_attempts`. `predict_fn(prompt:str)->str` (same contract as `evaluate_coding`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_self_repair.py
from italian_llm.evaluation import self_repair


def test_succeeds_after_one_repair():
    problem = {
        "prompt": "def add(a, b):\n",
        "test": "def check(c):\n    assert c(1, 2) == 3\n",
        "entry_point": "add",
    }
    calls = {"n": 0}

    def predict(prompt):
        calls["n"] += 1
        return "    return 0\n" if calls["n"] == 1 else "    return a + b\n"

    out = self_repair.solve_with_repair(problem, predict, max_attempts=3)
    assert out["passed"] is True
    assert out["attempts"] == 2


def test_gives_up_after_max_attempts():
    problem = {"prompt": "def f():\n", "test": "def check(c):\n    assert c() == 1\n",
               "entry_point": "f"}
    out = self_repair.solve_with_repair(problem, lambda p: "    return 0\n", max_attempts=2)
    assert out["passed"] is False
    assert out["attempts"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_self_repair.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Create the module**

```python
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


def solve_with_repair(problem: dict, predict_fn, max_attempts: int = 3,
                      timeout: float = 8.0) -> dict:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_self_repair.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Wire an optional `--repair` flag into `scripts/run_coding_eval.py`**

In `scripts/run_coding_eval.py`, add `p.add_argument("--repair", type=int, default=0, help="max tentativi di auto-riparazione (0=off)")`. When `args.repair > 0`, replace the per-problem evaluation with `self_repair.solve_with_repair(prob, predict_fn, max_attempts=args.repair, timeout=timeout)` and build `results` from its `{passed, attempts}`. Import `from italian_llm.evaluation import self_repair`. Keep the non-repair path unchanged.

- [ ] **Step 6: Smoke + commit**

Run: `python scripts/run_coding_eval.py --config configs/eval/eval_coding.yaml --repair 3`
Expected: runs without error (mock mode → still 0.0, but the repair loop executes). 

```bash
git add src/italian_llm/evaluation/self_repair.py scripts/run_coding_eval.py tests/test_self_repair.py
git commit -m "feat: self-repair loop (generate-test-fix)"
```

---

### Task E3: FIM (fill-in-the-middle) autocomplete

**Files:**
- Create: `src/italian_llm/serving/fim.py`
- Create: `scripts/fim_complete.py`
- Test: `tests/test_fim.py`

**Interfaces:**
- Produces:
  - `build_fim_prompt(prefix: str, suffix: str) -> str` → `"<|fim_prefix|>" + prefix + "<|fim_suffix|>" + suffix + "<|fim_middle|>"` (Qwen2.5-Coder FIM tokens).
  - `strip_fim(text: str) -> str` — removes FIM/EOS special tokens and anything after the first one.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fim.py
from italian_llm.serving.fim import build_fim_prompt, strip_fim


def test_build_fim_prompt_uses_qwen_tokens():
    p = build_fim_prompt("def add(a, b):\n    return ", "\n\nprint(add(1,2))")
    assert p == "<|fim_prefix|>def add(a, b):\n    return <|fim_suffix|>\n\nprint(add(1,2))<|fim_middle|>"


def test_strip_fim_cuts_at_special_tokens():
    assert strip_fim("a + b<|endoftext|>garbage") == "a + b"
    assert strip_fim("x<|fim_pad|>") == "x"
    assert strip_fim("clean") == "clean"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fim.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Create the module**

```python
"""Fill-in-the-middle (FIM) per Qwen2.5-Coder: completamento da editor.

Dai codice prima (prefix) e dopo (suffix) il cursore; il modello riempie il mezzo.
Feature locale che i chatbot cloud non offrono bene. Qui solo la logica pura dei
token; la generazione raw vive in scripts/fim_complete.py (transformers lazy).
"""

import re

__all__ = ["build_fim_prompt", "strip_fim", "FIM_PREFIX", "FIM_SUFFIX", "FIM_MIDDLE"]

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"

_SPECIAL_RE = re.compile(r"<\|(endoftext|fim_pad|fim_prefix|fim_suffix|fim_middle|im_end|im_start)\|>")


def build_fim_prompt(prefix: str, suffix: str) -> str:
    """Prompt FIM nel formato Qwen2.5-Coder."""
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"


def strip_fim(text: str) -> str:
    """Taglia l'output al primo token speciale e rimuove residui."""
    if not text:
        return ""
    m = _SPECIAL_RE.search(text)
    if m:
        text = text[:m.start()]
    return text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fim.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Create `scripts/fim_complete.py`** (raw completion; real model on GPU host, clear error offline)

```python
#!/usr/bin/env python
"""Completamento FIM: legge prefix e suffix, stampa il 'mezzo' generato.

Richiede transformers + il modello coder (gira dove c'e' il modello). Import lazy.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import argparse

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
        raise SystemExit(f"transformers non disponibile ({e}). Esegui dove c'e' il modello.")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                 trust_remote_code=True)
    inputs = tok(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    print(strip_fim(gen))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/italian_llm/serving/fim.py scripts/fim_complete.py tests/test_fim.py
git commit -m "feat: FIM fill-in-the-middle autocomplete"
```

---

### Task E4: Document Phase E in coding docs

**Files:**
- Modify: `docs/coding-model.md`

- [ ] **Step 1: Add three subsections to `docs/coding-model.md`**

1. **RAG sul tuo codice:** `python scripts/rag_ask.py --root <tua/cartella> --question "..."`. Spiega che ancora le risposte al tuo codebase reale.
2. **Auto-riparazione:** `python scripts/run_coding_eval.py --config configs/eval/eval_coding.yaml --repair 3`. Spiega genera→test→fix.
3. **FIM autocomplete:** `python scripts/fim_complete.py --prefix "..." --suffix "..."`. Nota che gira dove c'e' il modello (GPU host o CPU con modello scaricato).

- [ ] **Step 2: Commit**

```bash
git add docs/coding-model.md
git commit -m "docs: document RAG, self-repair, FIM"
```

---

## Self-Review

**Spec coverage:**
- Blocco 1 (run locally) → Tasks A1–A3, D1. ✓
- Blocco 2 (data + eval pass@k, sandboxed) → Tasks B1–B3. ✓ (Honest deviation: pass@k is Python-executable HumanEval-format; C#/JS/HTML/CSS handled by a qualitative set, since C#/CSS/HTML aren't executable by the Python sandbox. JS execution via node is left as a future extension and noted in docs.)
- Blocco 3 (optional free LoRA) → Tasks C1–C3. ✓
- Phase E genial-infra features (RAG, self-repair, FIM) → Tasks E1–E4. ✓
- Honest limits in docs → Task D1, E4. ✓
- €0 / no secrets/weights committed → Global Constraints + D1. ✓
- Sandboxed untrusted execution → Task B1 + constraint; reused by E2. ✓

**Placeholder scan:** No TBD/TODO in code steps; every code step shows full content. Two explicit "verify against existing file" notes (Task C2 config key names, Task C3 notebook signatures) are real verification steps, not placeholders — the implementer runs the given inspect command first.

**Type consistency:** `run_python(program, timeout) -> dict{passed,timed_out,error}` used consistently by `evaluate_coding`. `evaluate_coding(problems, predict_fn, timeout)` with `predict_fn(prompt:str)->str` used consistently by the CLI. `coding_system(languages=None)->str` used by prompts test, export script, builder, and eval CLI. `to_sft_example(instruction,response,language,idx)->dict` matches its test. `coding_passk([bool])` matches existing metric signature.

**Deviation flagged for user:** real executable pass@k is Python-only (HumanEval format). C#/HTML/CSS measured qualitatively, JS-via-node deferred. This matches the honesty principle of the spec.
