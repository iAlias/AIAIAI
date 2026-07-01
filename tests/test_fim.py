from italian_llm.serving.fim import build_fim_prompt, strip_fim


def test_build_fim_prompt_uses_qwen_tokens():
    p = build_fim_prompt("def add(a, b):\n    return ", "\n\nprint(add(1,2))")
    assert (
        p
        == "<|fim_prefix|>def add(a, b):\n    return <|fim_suffix|>\n\nprint(add(1,2))<|fim_middle|>"
    )


def test_strip_fim_cuts_at_special_tokens():
    assert strip_fim("a + b<|endoftext|>garbage") == "a + b"
    assert strip_fim("x<|fim_pad|>") == "x"
    assert strip_fim("clean") == "clean"
