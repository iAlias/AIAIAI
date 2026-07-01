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
    problem = {
        "prompt": "def f():\n",
        "test": "def check(c):\n    assert c() == 1\n",
        "entry_point": "f",
    }
    out = self_repair.solve_with_repair(problem, lambda p: "    return 0\n", max_attempts=2)
    assert out["passed"] is False
    assert out["attempts"] == 2
