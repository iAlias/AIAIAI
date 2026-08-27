from unittest.mock import patch

from italian_llm.evaluation.runner import _Predictor


def test_predictor_uses_ollama_when_configured():
    cfg = {"eval": {"ollama_model": "coder-local", "ollama_host": "http://localhost:11434"}}
    pred = _Predictor(cfg)

    mode = pred.init()

    assert mode == "ollama"

    with patch(
        "italian_llm.serving.ollama_client.chat", return_value="codice generato"
    ) as mock_chat:
        out = pred.predict([{"role": "user", "content": "ciao"}])

    assert out == "codice generato"
    mock_chat.assert_called_once()
    _, kwargs = mock_chat.call_args
    assert kwargs["host"] == "http://localhost:11434"


def test_predictor_falls_back_to_mock_when_ollama_unreachable():
    cfg = {"eval": {"ollama_model": "coder-local"}}
    pred = _Predictor(cfg)
    pred.init()

    from italian_llm.serving.ollama_client import OllamaError

    with patch("italian_llm.serving.ollama_client.chat", side_effect=OllamaError("no server")):
        out = pred.predict([{"role": "user", "content": "ciao"}])

    assert isinstance(out, str)
    assert out  # il MockTeacher produce comunque del testo
    assert pred.mode == "mock"  # il fallback aggiorna mode per riflettere la realta'


def test_predictor_without_ollama_model_behaves_as_before():
    cfg = {"eval": {}}
    pred = _Predictor(cfg)

    mode = pred.init()

    assert mode == "mock"


def test_predictor_passes_max_tokens_and_timeout_to_ollama():
    cfg = {"eval": {"ollama_model": "coder-local", "max_new_tokens": 256, "ollama_timeout": 300}}
    pred = _Predictor(cfg)
    pred.init()

    with patch("italian_llm.serving.ollama_client.chat", return_value="ok") as mock_chat:
        pred.predict([{"role": "user", "content": "ciao"}])

    _, kwargs = mock_chat.call_args
    assert kwargs["max_tokens"] == 256
    assert kwargs["timeout"] == 300.0
