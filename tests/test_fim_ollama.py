import importlib.util
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from italian_llm.serving.fim import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX
from italian_llm.serving.ollama_client import OllamaError, generate


def _fake_response(payload: dict):
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _load_fim_complete():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "scripts", "fim_complete.py")
    spec = importlib.util.spec_from_file_location("fim_complete", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# ollama_client.generate (/api/generate, raw mode per i token FIM)
# ---------------------------------------------------------------------------


def test_generate_sends_raw_payload_and_parses_response():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _fake_response({"response": "  middle();"})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = generate(
            "coder-local",
            "<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>",
            host="http://localhost:11434",
            temperature=0.0,
            max_tokens=64,
        )

    assert out == "  middle();"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["body"]["model"] == "coder-local"
    assert captured["body"]["prompt"].startswith("<|fim_prefix|>")
    assert captured["body"]["raw"] is True
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"]["num_predict"] == 64


def test_generate_raises_on_malformed_response():
    with patch("urllib.request.urlopen", return_value=_fake_response({"nope": 1})):
        with pytest.raises(OllamaError):
            generate("coder-local", "prompt")


def test_generate_raises_on_connection_error():
    with patch("urllib.request.urlopen", side_effect=OSError("refused")):
        with pytest.raises(OllamaError):
            generate("coder-local", "prompt")


# ---------------------------------------------------------------------------
# scripts/fim_complete.py --ollama-model: niente torch, output ripulito
# ---------------------------------------------------------------------------


def test_fim_complete_via_ollama_strips_tokens(capsys):
    mod = _load_fim_complete()
    captured = {}

    def fake_generate(model, prompt, **kw):
        captured["model"] = model
        captured["prompt"] = prompt
        return "return a + b<|fim_pad|>garbage"

    with patch("italian_llm.serving.ollama_client.generate", side_effect=fake_generate):
        mod.main(
            [
                "--prefix",
                "def add(a, b):\n    ",
                "--suffix",
                "\n",
                "--ollama-model",
                "coder-local",
            ]
        )

    out = capsys.readouterr().out
    assert "return a + b" in out
    assert "<|fim_pad|>" not in out
    assert captured["model"] == "coder-local"
    assert captured["prompt"] == (f"{FIM_PREFIX}def add(a, b):\n    {FIM_SUFFIX}\n{FIM_MIDDLE}")
