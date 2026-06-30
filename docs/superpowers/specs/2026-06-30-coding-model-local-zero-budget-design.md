# Coding model locale, veloce, zero-budget — Design

Data: 2026-06-30
Stato: approvato (in attesa di review utente sullo spec scritto)

## Obiettivo

Avere un **assistente di programmazione locale** che:

- gira **veloce** su hardware modesto (Intel i7-10510U, 4 core, 32 GB RAM, **no GPU CUDA**, Intel iGPU);
- è forte su **C#, JavaScript, HTML, CSS** (web stack);
- costa **€0** sia da costruire sia da eseguire;
- è **privato** (offline) e personalizzabile nello stile dall'utente.

Ispirazione dichiarata: filosofia di [antirez/ds4](https://github.com/antirez/ds4) — **specializzato + quantizzato + locale** invece di generico e grande. Adattata però all'hardware reale (ds4 mira a macchine 96 GB+; qui si sta su un ultrabook).

## Non-obiettivi (espliciti)

- **Non** è un sostituto di Claude Opus 4.8 o di modelli frontier. Su coding agentico complesso / multi-file / debugging di sistemi il gap è enorme e resta tale. Si punta a un "junior veloce" utile su snippet, completamento, Q&A sullo stack.
- **Non** si addestra un modello da zero. **Non** si scrive un inference engine in C (sovra-ingegneria, scartato in fase di design).
- **Non** si spende denaro: niente API a pagamento, niente GPU a noleggio a pagamento.

## Realtà di partenza che guida le scelte

1. Hardware locale **non addestra** un LLM: serve solo per **eseguire** modelli piccoli quantizzati.
2. C#/JS/HTML/CSS sono mainstream: **Qwen2.5-Coder è già molto forte** su questi linguaggi. Un fine-tuning su dati aperti **rischia di non migliorare** (o peggiorare) la base. Quindi il fine-tuning è **opzionale** e serve a **personalizzare lo stile**, non a "rendere più intelligente".
3. Il repo esistente (`italian_llm`) è uno scaffold completo orientato all'italiano, con import pesanti lazy e fallback mock: **riusabile** ri-mirando config, dati ed eval verso il coding.

## Architettura della soluzione

Tre blocchi, in ordine di valore:

### Blocco 1 — Esecuzione locale (il 90% del valore, gira subito)

- **Modello base:** `Qwen2.5-Coder-1.5B-Instruct` come **default** (veloce su CPU).
  Opzione documentata: `Qwen2.5-Coder-3B-Instruct` (più capace, più lento).
- **Runtime:** [Ollama](https://ollama.com) (wrapper su llama.cpp), formato **GGUF Q4_K_M**.
- **System prompt coding:** diretto, conciso, niente moralismi superflui — riusa la
  filosofia anti over-refusal già presente nel repo (`safety/policy.py`,
  `data/prompts.py`), adattata a un assistente di programmazione.
- Configurazione serving nel repo (`configs/serving/`) puntata a Ollama locale,
  con il fallback `transformers` già esistente.

### Blocco 2 — Pipeline dati + valutazione coding (gratis, locale)

Ri-mira la pipeline esistente dall'italiano al coding:

- **Config modello:** `configs/model/` punta a Qwen2.5-Coder (al posto di Qwen2.5-7B).
- **Valutazione coding reale:** cablare le metriche **pass@k** già presenti
  (`evaluation/metrics.py`) su benchmark standard **HumanEval** e **MBPP**, più un
  piccolo set proprietario C#/web. Esecuzione sandboxata dei test (per pass@k serve
  eseguire codice generato in modo isolato — vincolo di sicurezza da rispettare).
- **Dati SFT coding:** costruzione di un dataset istruzione da **dataset aperti**
  (es. Magicoder OSS-Instruct, Evol-Instruct-Code, the-stack), **filtrati** ai
  linguaggi target C#/JS/HTML/CSS, in formato chat JSONL secondo lo schema esistente
  (`data/schema.py`). Serve solo al Blocco 3.

### Blocco 3 — Fine-tuning opzionale e gratuito (solo se serve personalizzare)

- **Dove:** **Kaggle Notebooks** (GPU gratis ~30h/settimana, T4/P100) come default;
  Google Colab free come alternativa.
- **Cosa:** **QLoRA** sul modello base usando lo script SFT esistente
  (`training/sft.py`), sul dataset del Blocco 2.
- **Output:** adapter LoRA → merge → **quantizzazione GGUF** (`scripts/quantize_gguf.py`
  già presente) → import in Ollama.
- **Quando lanciarlo:** solo se la base non riflette lo stile/le convenzioni volute.
  È uno step **pronto-ma-non-obbligatorio**, non parte del percorso minimo.

## Flusso dati

```
[dataset aperti] --filtro C#/JS/HTML/CSS--> [SFT JSONL] --(Kaggle, opz.)--> [LoRA]
                                                                              |
[Qwen2.5-Coder-1.5B] --(merge opz.)--> [GGUF Q4_K_M] --> [Ollama locale] --> utente
                                              |
                                        [eval pass@k: HumanEval/MBPP/set C#] --> report
```

## Componenti del repo toccati

| Area | File/dir | Modifica |
|------|----------|----------|
| Config modello | `configs/model/` | nuovo target Qwen2.5-Coder 1.5B/3B |
| Config serving | `configs/serving/` | profilo Ollama locale |
| Config eval | `configs/eval/` | benchmark coding (HumanEval/MBPP + set C#/web) |
| Eval | `src/.../evaluation/` | cablare pass@k su esecuzione test sandboxata |
| Dati | `scripts/` + `src/.../data/` | builder dataset coding da fonti aperte, filtro linguaggi |
| Prompt/safety | `src/.../data/prompts.py`, `safety/policy.py` | system prompt coding |
| Training | `src/.../training/sft.py` | invariato; consumato dal notebook Kaggle |
| Notebook | `notebooks/` | notebook Kaggle QLoRA pronto |
| Docs | `docs/`, `README.md` | riallineare lingua/identità progetto: coding, non italiano |

Nota: il package resta `italian_llm` per ora (rinominare è un refactor a parte, non
necessario all'obiettivo). Va valutato se vale un rename a fine lavoro.

## Gestione errori e vincoli

- **Esecuzione codice per pass@k:** il codice generato dal modello va eseguito in
  **sandbox isolata** con timeout e risorse limitate. Mai eseguire output non fidato
  senza isolamento. Questo è un requisito di sicurezza non negoziabile.
- **No GPU locale:** ogni step pesante (training) gira **fuori** (Kaggle/Colab); in
  locale solo inferenza. Gli import pesanti restano lazy (ADR-009 del repo).
- **Licenze:** verificare licenze di base model (Qwen) e dataset aperti prima dell'uso.
- **Aspettative:** documentare nel README i limiti reali (vedi Non-obiettivi) per non
  promettere prestazioni frontier.

## Criteri di successo

- [ ] Qwen2.5-Coder gira in locale via Ollama e risponde a domande C#/JS/HTML/CSS
      a velocità usabile su questo PC.
- [ ] `make eval` (o equivalente) produce un report **pass@k** su HumanEval/MBPP +
      set proprietario, con esecuzione test sandboxata.
- [ ] Builder dataset coding produce JSONL valido (schema esistente) filtrato ai
      linguaggi target, **offline**, da fonti aperte.
- [ ] Notebook Kaggle QLoRA documentato e pronto (eseguibile a richiesta, gratis).
- [ ] Pipeline GGUF: da adapter/merge a modello importabile in Ollama.
- [ ] README/docs riallineati a "coding assistant", con limiti dichiarati.
- [ ] Tutto a costo **€0**; nessun peso/segreto committato.

## Approcci scartati

- **B (solo uso, zero pipeline):** valido ma non lascia nulla di "tuo" né misurabile.
  Assorbito come Blocco 1 del piano.
- **C (inference engine in C tipo ds4):** mesi di lavoro, non aumenta l'intelligenza
  del modello. YAGNI. Scartato.
- **Fine-tuning come step obbligatorio:** rischia di peggiorare la base forte su
  linguaggi mainstream. Reso opzionale (Blocco 3).
