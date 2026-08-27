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


def test_evaluate_coding_keeps_raw_prediction():
    problems = code_eval.load_problems(_FIX)
    raw = "```python\n    return a + b\n```"
    results = code_eval.evaluate_coding(problems, lambda p: raw)
    assert results[0]["raw"] == raw


def test_evaluate_coding_invokes_on_result_per_problem():
    problems = code_eval.load_problems(_FIX)
    seen = []
    code_eval.evaluate_coding(problems, lambda p: "    return None\n", on_result=seen.append)
    assert [r["task_id"] for r in seen] == [p["task_id"] for p in problems]
    assert all("passed" in r and "raw" in r for r in seen)


_FIX_JS = os.path.join(_REPO_ROOT, "data", "eval", "humaneval_js.sample.jsonl")


def test_build_program_javascript_uses_completion_when_it_declares_entry():
    problem = code_eval.load_problems(_FIX_JS)[0]
    completion = "function add(a, b){\n  return a + b;\n}"
    prog = code_eval.build_program(problem, completion)
    assert prog.count("function add(") == 1
    assert prog.rstrip().endswith("test();")
    assert "check(" not in prog


def test_build_program_javascript_appends_body_when_no_declaration():
    problem = code_eval.load_problems(_FIX_JS)[0]
    completion = "  return a + b;\n}"
    prog = code_eval.build_program(problem, completion)
    assert prog.startswith(problem["prompt"] + completion)
    assert prog.rstrip().endswith("test();")


def test_build_program_python_indents_flush_left_body():
    problem = {"prompt": "def f(a, b):\n", "test": "def check(c):\n    pass\n", "entry_point": "f"}
    prog = code_eval.build_program(problem, "return a + b")
    assert "def f(a, b):\n    return a + b\n" in prog


def test_build_program_python_full_function_keeps_prompt_imports():
    problem = {
        "prompt": 'from typing import List\n\n\ndef f(xs: List[int]) -> int:\n    """Sum."""\n',
        "test": "def check(c):\n    assert c([1, 2]) == 3\n",
        "entry_point": "f",
    }
    completion = "def f(xs: List[int]) -> int:\n    return sum(xs)\n"
    prog = code_eval.build_program(problem, completion)
    assert prog.startswith("from typing import List")
    assert prog.count("def f(") == 2
    assert code_eval.run_program(problem, completion)["passed"] is True


def test_evaluate_coding_runs_javascript_problems_with_node():
    problems = code_eval.load_problems(_FIX_JS)

    def predict(prompt):
        if "add" in prompt:
            return "```javascript\nfunction add(a, b){\n  return a + b;\n}\n```"
        return "```javascript\nfunction is_even(n){\n  return n % 2 === 0;\n}\n```"

    results = code_eval.evaluate_coding(problems, predict)
    assert all(r["passed"] for r in results)


def test_evaluate_coding_javascript_wrong_solution_fails():
    problems = code_eval.load_problems(_FIX_JS)
    results = code_eval.evaluate_coding(problems, lambda p: "  return null;\n}")
    assert all(r["passed"] is False for r in results)
