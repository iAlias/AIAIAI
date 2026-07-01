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
