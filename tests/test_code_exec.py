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


def test_non_utf8_output():
    """Regression test: program emitting invalid UTF-8 bytes should not raise UnicodeDecodeError."""
    # Program writes raw non-UTF8 bytes to stdout, should decode with U+FFFD replacement chars
    program = "import sys\n" "sys.stdout.buffer.write(b'\\xff\\xfe')\n" "sys.exit(0)\n"
    r = run_python(program)
    assert isinstance(r, dict), "run_python should always return a dict, not raise"
    assert r["passed"] is True, "Exit code 0 should be treated as passed"
    assert r["timed_out"] is False
    assert r["error"] is None
