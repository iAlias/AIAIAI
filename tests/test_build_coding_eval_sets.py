import importlib.util
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "build_coding_eval_sets.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("build_coding_eval_sets", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_humaneval_row_to_problem_keeps_exec_fields_and_sets_python():
    mod = _load_script()
    row = {
        "task_id": "HumanEval/0",
        "prompt": "def f():\n",
        "entry_point": "f",
        "test": "def check(c):\n    pass\n",
        "canonical_solution": "    pass\n",
    }
    assert mod.humaneval_row_to_problem(row) == {
        "task_id": "HumanEval/0",
        "prompt": "def f():\n",
        "entry_point": "f",
        "test": "def check(c):\n    pass\n",
        "language": "python",
    }


def test_multiple_js_row_to_problem_derives_entry_point_from_prompt():
    mod = _load_script()
    row = {
        "name": "HumanEval_0_has_close_elements",
        "language": "js",
        "prompt": "// doc\n// more doc\nfunction has_close_elements(numbers, threshold){\n",
        "tests": "const assert = require('node:assert');\n\nfunction test() {\n}\n\ntest();",
        "stop_tokens": ["\nfunction ", "\n/*", "\n//", "\nconsole.log"],
    }
    prob = mod.multiple_js_row_to_problem(row)
    assert prob["task_id"] == "HumanEval_0_has_close_elements"
    assert prob["entry_point"] == "has_close_elements"
    assert prob["language"] == "javascript"
    assert prob["prompt"] == row["prompt"]
    assert prob["test"] == row["tests"]


def test_multiple_js_row_without_function_signature_is_rejected():
    mod = _load_script()
    row = {"name": "x", "prompt": "// nothing here\n", "tests": "test();"}
    try:
        mod.multiple_js_row_to_problem(row)
    except ValueError as exc:
        assert "x" in str(exc)
    else:
        raise AssertionError("atteso ValueError per prompt senza firma")
