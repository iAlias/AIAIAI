from italian_llm.data import prompts


def test_coding_system_default_is_direct():
    s = prompts.CODING_SYSTEM
    assert "codice" in s.lower()
    assert prompts.coding_system() == s


def test_coding_system_mentions_languages():
    s = prompts.coding_system(["C#", "JavaScript"])
    assert "C#" in s and "JavaScript" in s
