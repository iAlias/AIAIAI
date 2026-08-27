import importlib.util
import os

from italian_llm.data.schema import validate_record

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "build_coding_sft.py")


def _load():
    spec = importlib.util.spec_from_file_location("build_coding_sft", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_to_sft_example_is_valid():
    mod = _load()
    rec = mod.to_sft_example("Scrivi add in C#", "```csharp\nint Add()=>0;\n```", "csharp", 0)
    ok, reason = validate_record(rec)
    assert ok, reason
    assert rec["domain"] == "csharp"
    assert rec["messages"][0]["role"] == "system"


def test_filter_language_known_and_unknown():
    mod = _load()
    assert mod.filter_language("usando C# e .NET") == "csharp"
    assert mod.filter_language("in python") is None


def _write(path, rows):
    import json

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_merges_inputs_and_drops_duplicate_instructions(tmp_path):
    mod = _load()
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write(
        a,
        [
            {
                "instruction": "In C#, somma due interi.",
                "response": "```csharp\nx\n```",
                "language": "csharp",
            },
            {
                "instruction": "In CSS, centra un div.",
                "response": "```css\nx\n```",
                "language": "css",
            },
        ],
    )
    _write(
        b,
        [
            {
                "instruction": "in c#,  somma due interi.",
                "response": "```csharp\ny\n```",
                "language": "csharp",
            },
            {
                "instruction": "In JS, inverti una stringa.",
                "response": "```javascript\nx\n```",
                "language": "javascript",
            },
        ],
    )
    out = tmp_path / "out.jsonl"

    n = mod.build([str(a), str(b)], str(out))

    rows = list(mod.read_jsonl(str(out)))
    assert n == len(rows) == 3
    assert len({r["id"] for r in rows}) == 3


def test_build_stamps_source_type_and_teacher_name(tmp_path):
    mod = _load()
    src = tmp_path / "a.jsonl"
    _write(
        src, [{"instruction": "In C#, x", "response": "```csharp\nx\n```", "language": "csharp"}]
    )
    out = tmp_path / "out.jsonl"

    mod.build([str(src)], str(out), source_type="teacher_synthetic", teacher_name="claude")

    (row,) = list(mod.read_jsonl(str(out)))
    assert row["source_type"] == "teacher_synthetic"
    assert row["teacher_name"] == "claude"


def test_main_accepts_multiple_inputs(tmp_path):
    mod = _load()
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write(a, [{"instruction": "In C#, a", "response": "```csharp\na\n```", "language": "csharp"}])
    _write(b, [{"instruction": "In C#, b", "response": "```csharp\nb\n```", "language": "csharp"}])
    out = tmp_path / "out.jsonl"

    mod.main(["--in", str(a), str(b), "--out", str(out), "--teacher-name", "claude"])

    rows = list(mod.read_jsonl(str(out)))
    assert len(rows) == 2 and all(r["teacher_name"] == "claude" for r in rows)
