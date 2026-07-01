"""Client HTTP minimale per l'API chat di Ollama (solo stdlib, no requests).

Permette a `_Predictor` (evaluation/runner.py) di generare usando un modello
gia' registrato in Ollama (es. via scripts/export_ollama_coding.py) invece
di caricare i pesi via transformers/torch.
"""

import json
import urllib.error
import urllib.request

__all__ = ["OllamaError", "chat"]


class OllamaError(Exception):
    """Errore di comunicazione o risposta inattesa dal server Ollama."""


def chat(
    model: str,
    messages: list[dict],
    *,
    host: str = "http://localhost:11434",
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> str:
    """Chiama /api/chat su un'istanza Ollama locale e ritorna il testo generato.

    Solleva OllamaError se il server non e' raggiungibile o la risposta non
    ha la forma attesa ({"message": {"content": ...}}).
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (OSError, urllib.error.URLError) as e:
        raise OllamaError(f"Ollama non raggiungibile ({host}): {e}") from e
    except json.JSONDecodeError as e:
        raise OllamaError(f"Risposta Ollama non valida (JSON): {e}") from e

    message = body.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise OllamaError(f"Risposta Ollama inattesa: {body!r}")
    return message["content"]
