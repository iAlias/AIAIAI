import importlib.util
import json
import os
from unittest.mock import patch

from italian_llm.data.schema import validate_record
from italian_llm.memory.interaction_log import (
    append_interaction,
    load_interactions,
    memory_context,
    retrieve_memory,
    to_sft_examples,
)


def _load_chat_learn():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "scripts", "chat_learn.py")
    spec = importlib.util.spec_from_file_location("chat_learn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# interaction_log: append / load
# ---------------------------------------------------------------------------


def test_append_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "interactions.jsonl")
    rec = append_interaction(path, "come si usa flexbox?", "Con display: flex.", model="coder")
    assert rec["question"] == "come si usa flexbox?"
    assert rec["answer"] == "Con display: flex."
    assert rec["model"] == "coder"
    assert rec["id"]
    assert rec["ts"]

    loaded = load_interactions(path)
    assert len(loaded) == 1
    assert loaded[0]["question"] == "come si usa flexbox?"


def test_append_assigns_progressive_ids(tmp_path):
    path = str(tmp_path / "interactions.jsonl")
    r1 = append_interaction(path, "domanda uno", "risposta uno")
    r2 = append_interaction(path, "domanda due", "risposta due")
    assert r1["id"] != r2["id"]
    loaded = load_interactions(path)
    assert [r["question"] for r in loaded] == ["domanda uno", "domanda due"]


def test_load_missing_file_returns_empty(tmp_path):
    assert load_interactions(str(tmp_path / "nope.jsonl")) == []


# ---------------------------------------------------------------------------
# interaction_log: retrieval BM25 sulle interazioni passate
# ---------------------------------------------------------------------------


def _sample_records():
    return [
        {
            "id": "mem-1",
            "question": "come apro una connessione postgres in C#?",
            "answer": "Usa NpgsqlConnection con la connection string.",
        },
        {
            "id": "mem-2",
            "question": "come centro un div con flexbox?",
            "answer": "justify-content: center e align-items: center.",
        },
    ]


def test_retrieve_memory_finds_relevant_interaction():
    hits = retrieve_memory(_sample_records(), "connessione postgres", k=1)
    assert len(hits) == 1
    assert hits[0]["id"] == "mem-1"


def test_retrieve_memory_empty_records_no_crash():
    assert retrieve_memory([], "qualsiasi cosa", k=3) == []


def test_memory_context_contains_question_and_answer():
    ctx = memory_context(_sample_records(), "flexbox div", k=1)
    assert "flexbox" in ctx
    assert "align-items" in ctx


def test_memory_context_empty_when_no_match():
    assert memory_context(_sample_records(), "zzz xxyyzz", k=2) == ""


# ---------------------------------------------------------------------------
# interaction_log: export verso dataset SFT (schema del repo)
# ---------------------------------------------------------------------------


def test_to_sft_examples_produce_valid_schema_records():
    examples = to_sft_examples(_sample_records())
    assert len(examples) == 2
    for rec in examples:
        ok, reason = validate_record(rec)
        assert ok, reason
        assert rec["source_type"] == "user_interaction"
        roles = [m["role"] for m in rec["messages"]]
        assert roles == ["system", "user", "assistant"]


def test_to_sft_examples_skips_empty_answers():
    records = [{"id": "mem-1", "question": "ciao", "answer": "   "}]
    assert to_sft_examples(records) == []


# ---------------------------------------------------------------------------
# scripts/chat_learn.py: one-shot con memoria + logging su file
# ---------------------------------------------------------------------------


def test_chat_learn_one_shot_uses_memory_and_appends(tmp_path, capsys):
    mod = _load_chat_learn()
    mem = str(tmp_path / "interactions.jsonl")
    append_interaction(mem, "come apro una connessione postgres in C#?", "Usa NpgsqlConnection.")

    captured = {}

    def fake_chat(model, messages, host="http://localhost:11434", **kw):
        captured["model"] = model
        captured["messages"] = messages
        return "Risposta del modello."

    with patch("italian_llm.serving.ollama_client.chat", side_effect=fake_chat):
        mod.main(
            [
                "--question",
                "errore di connessione postgres, come debuggho?",
                "--memory",
                mem,
                "--ollama-model",
                "coder-local",
            ]
        )

    out = capsys.readouterr().out
    assert "Risposta del modello." in out
    # La memoria rilevante (NpgsqlConnection) e' stata iniettata nel contesto.
    joined = json.dumps(captured["messages"], ensure_ascii=False)
    assert "NpgsqlConnection" in joined
    # La domanda utente resta pulita nell'ultimo turno user.
    assert captured["messages"][-1]["role"] == "user"
    assert captured["messages"][-1]["content"].startswith("errore di connessione")
    # Lo scambio e' stato registrato: ora la memoria contiene 2 interazioni.
    assert len(load_interactions(mem)) == 2


def test_chat_learn_no_memory_flag_skips_context(tmp_path):
    mod = _load_chat_learn()
    mem = str(tmp_path / "interactions.jsonl")
    append_interaction(mem, "domanda su postgres", "risposta su NpgsqlConnection")

    captured = {}

    def fake_chat(model, messages, host="http://localhost:11434", **kw):
        captured["messages"] = messages
        return "ok"

    with patch("italian_llm.serving.ollama_client.chat", side_effect=fake_chat):
        mod.main(
            [
                "--question",
                "altra domanda su postgres",
                "--memory",
                mem,
                "--no-memory",
            ]
        )

    joined = json.dumps(captured["messages"], ensure_ascii=False)
    assert "NpgsqlConnection" not in joined


def test_chat_learn_export_sft_writes_valid_jsonl(tmp_path):
    mod = _load_chat_learn()
    mem = str(tmp_path / "interactions.jsonl")
    append_interaction(mem, "come centro un div?", "Con flexbox: display flex.")
    out_path = str(tmp_path / "sft_from_memory.jsonl")

    mod.main(["--memory", mem, "--export-sft", out_path])

    with open(out_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 1
    ok, reason = validate_record(rows[0])
    assert ok, reason
