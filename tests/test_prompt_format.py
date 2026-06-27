"""Smoke test sui prompt: build_messages, render_plain, SYNTH_PROMPTS, SYSTEM_DEFAULT."""

from italian_llm.data import prompts


def test_build_messages_structure():
    msgs = prompts.build_messages("Sistema", "Domanda utente", "Risposta")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == "Sistema"
    assert msgs[1]["content"] == "Domanda utente"
    assert msgs[2]["content"] == "Risposta"
    # Ogni turno espone esattamente 'role' e 'content'.
    for m in msgs:
        assert set(m.keys()) == {"role", "content"}

    # system None oppure vuoto/blank -> nessun turno di sistema.
    assert [m["role"] for m in prompts.build_messages(None, "Solo utente")] == ["user"]
    assert [m["role"] for m in prompts.build_messages("   ", "Solo utente")] == ["user"]

    # assistant None -> il turno finale non viene aggiunto.
    two = prompts.build_messages("S", "U")
    assert [m["role"] for m in two] == ["system", "user"]


def test_render_plain_contains_tags():
    text = prompts.render_plain(prompts.build_messages("S", "U", "A"))
    assert "<|system|>" in text
    assert "<|user|>" in text
    assert "<|assistant|>" in text

    # add_generation_prompt termina con l'header dell'assistente, pronto a generare.
    gen = prompts.render_plain(
        prompts.build_messages(None, "Ciao"), add_generation_prompt=True
    )
    assert gen.rstrip().endswith("<|assistant|>")
    assert "<|user|>" in gen


def test_synth_prompts_ten_domains():
    expected = {
        "qa", "summary", "rewrite", "coding", "email",
        "helpdesk", "faq", "admin", "dialog", "doc",
    }
    assert set(prompts.SYNTH_PROMPTS.keys()) == expected
    assert len(prompts.SYNTH_PROMPTS) == 10
    # Ogni template e' una stringa non vuota con il placeholder {topic}.
    for dom, tmpl in prompts.SYNTH_PROMPTS.items():
        assert isinstance(tmpl, str)
        assert tmpl.strip()
        assert "{topic}" in tmpl


def test_system_default_non_empty_italian():
    sd = prompts.SYSTEM_DEFAULT
    assert isinstance(sd, str)
    assert len(sd) > 30
    low = sd.lower()
    # Segnali che e' davvero in italiano e in tema (assistente utile, conciso).
    assert "italiano" in low
    assert any(w in low for w in ("assistente", "rispondi", "utile"))
