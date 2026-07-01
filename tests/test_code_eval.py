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
