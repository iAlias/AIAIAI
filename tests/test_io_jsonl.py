from italian_llm.utils.io import read_jsonl, write_jsonl


def test_write_jsonl_uses_lf_line_endings_on_every_platform(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(str(path), [{"a": 1}, {"b": "è"}])
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")
    assert list(read_jsonl(str(path))) == [{"a": 1}, {"b": "è"}]
