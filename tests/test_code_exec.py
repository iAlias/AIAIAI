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
