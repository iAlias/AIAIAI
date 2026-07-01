from italian_llm.rag.code_index import CodeIndex, build_context, chunk_text


def test_chunk_text_splits_by_lines():
    text = "\n".join(f"line{i}" for i in range(100))
    chunks = chunk_text("a.py", text, max_lines=40)
    assert len(chunks) == 3
    assert chunks[0]["path"] == "a.py"
    assert chunks[0]["start_line"] == 1
    assert chunks[1]["start_line"] == 41


def test_bm25_ranks_relevant_chunk_first():
    chunks = [
        {
            "path": "auth.cs",
            "start_line": 1,
            "text": "public bool ValidateToken(string jwt) { return true; }",
        },
        {
            "path": "math.cs",
            "start_line": 1,
            "text": "public int Add(int a, int b) { return a + b; }",
        },
    ]
    idx = CodeIndex.build(chunks)
    top = idx.search("come valido un token jwt", k=1)
    assert top and top[0]["path"] == "auth.cs"


def test_build_context_includes_paths_and_caps_size():
    chunks = [{"path": "a.cs", "start_line": 5, "text": "X" * 50}]
    ctx = build_context(chunks, max_chars=1000)
    assert "a.cs:5" in ctx
