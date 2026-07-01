import json
from unittest.mock import MagicMock, patch

import pytest

from italian_llm.serving.ollama_client import OllamaError, chat


def _fake_response(payload: dict):
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_chat_sends_expected_payload_and_parses_content():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _fake_response({"message": {"role": "assistant", "content": "ok"}})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = chat(
            "coder-local",
            [{"role": "user", "content": "ciao"}],
            host="http://localhost:11434",
            temperature=0.2,
            timeout=5.0,
        )

    assert out == "ok"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["body"]["model"] == "coder-local"
    assert captured["body"]["messages"] == [{"role": "user", "content": "ciao"}]
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"]["temperature"] == 0.2
    assert captured["timeout"] == 5.0


def test_chat_raises_ollama_error_on_connection_failure():
    def fake_urlopen(req, timeout=None):
        raise OSError("connection refused")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(OllamaError):
            chat("coder-local", [{"role": "user", "content": "ciao"}])


def test_chat_raises_ollama_error_on_malformed_response():
    def fake_urlopen(req, timeout=None):
        return _fake_response({"unexpected": "shape"})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(OllamaError):
            chat("coder-local", [{"role": "user", "content": "ciao"}])
