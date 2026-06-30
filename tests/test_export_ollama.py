import importlib.util
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "export_ollama_coding.py")


def _load():
    spec = importlib.util.spec_from_file_location("export_ollama_coding", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_modelfile_contains_system_and_from():
    mod = _load()
    text = mod.build_modelfile("/tmp/model.q4_k_m.gguf", "Sei un assistente di codice.")
    assert "FROM /tmp/model.q4_k_m.gguf" in text
    assert "Sei un assistente di codice." in text
    assert "<|im_start|>assistant" in text
    assert "PARAMETER temperature 0.2" in text
