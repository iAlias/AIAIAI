# Piano dei dati

> Corpus italiano misto, **sourcing legale/aperto**, bootstrap **sintetico**,
> pipeline di **pulizia → filtro lingua/qualità → dedup → tagging → split**, schema
> **JSONL** e design del dataset per la **riduzione dell'over-refusal**.

Documenti collegati:
[architecture.md](./architecture.md) ·
[training-plan.md](./training-plan.md) ·
[distillation-plan.md](./distillation-plan.md) ·
[evaluation-plan.md](./evaluation-plan.md) ·
[operations.md](./operations.md) ·
[decisions.md](./decisions.md)

Codice di riferimento: `src/italian_llm/data/cleaning.py`,
`src/italian_llm/data/schema.py`, `src/italian_llm/data/prompts.py`,
`src/italian_llm/data/synthetic.py`. Config:
`configs/data/corpus.yaml`, `configs/data/sft.yaml`.

---

## 1. Filosofia dei dati

- **Solo fonti aperte/lecite**: corpora con licenza chiara (open/CC, dataset HF con
  licenza compatibile) o **dati propri**. Niente scraping di contenuti protetti,
  niente dati proprietari nel repo. Le licenze delle fonti vanno verificate **prima**
  dell'uso e della ridistribuzione (vedi nota licenze nel README).
- **Bootstrap sintetico**: quando i dati reali scarseggiano, si **genera** col
  teacher (mock offline di default, reale via env). Questo rende la pipeline
  eseguibile **anche completamente offline**.
- **Qualità > quantità**: filtri lingua/qualità severi, dedup aggressivo, tagging per
  dominio/difficoltà. Un dataset piccolo e pulito batte un dataset grande e sporco.
- **Italianità come metrica di prima classe**: `italian_score` e `is_italian` sono
  filtri, non accessori.

Domini di prodotto (allineati a `SYNTH_PROMPTS` in `data/prompts.py` e a
`corpus.domains` / `sft.domain_weights`):
`general, qa, summary, rewrite, coding, email, helpdesk, faq, admin, dialog, doc`.

---

## 2. Sorgenti del corpus (CPT)

Definite in `configs/data/corpus.yaml` → `corpus.sources`. Ogni sorgente ha
`name, type, path, split, text_field, weight, enabled`.

| Sorgente | `type` | Stato default | Licenza/Nota |
|----------|--------|---------------|--------------|
| `wiki_it` | `huggingface` (`wikimedia/wikipedia`, config `20231101.it`) | `enabled: true` | CC BY-SA; ottima base italiana ampia. |
| `oscar_it` | `huggingface` (`oscar-corpus/OSCAR-2301`) | `enabled: false` | richiede **accettazione licenza** → off di default. |
| `local_txt` | `local` (`data/raw/local`, scansione ricorsiva `.txt`/`.jsonl`) | `enabled: true` | dati propri dell'utente. |
| `synthetic_fallback` | `synthetic` (teacher mock) | `enabled: true` | generato localmente se manca tutto il resto. |

I dataset HF sono caricati **lazy** (`datasets`), così il modulo importa senza la
dipendenza. In assenza di rete/dati reali, `synthetic_fallback` garantisce che la
pipeline produca comunque un `corpus.jsonl` valido. `weight` regola il campionamento
relativo tra sorgenti.

> Per il **long-context** servono documenti lunghi reali: è una scelta di sourcing,
> non solo di config (vedi [architecture.md](./architecture.md#62-long-context--parzialmente-plug-and-play)).

---

## 3. Pipeline di pulizia (step e ordine)

Implementata in `src/italian_llm/data/cleaning.py`, guidata da `corpus.filters`. I
filtri sono **idempotenti** e applicati **in quest'ordine**:

1. **Normalizzazione Unicode** — `normalize_unicode(text)`: NFC, normalizzazione di
   spazi/virgolette/trattini, rimozione di caratteri di controllo. (`normalize_unicode: true`)
2. **Rimozione boilerplate** — `remove_boilerplate(text)`: footer, banner cookie,
   menù ripetuti, righe di navigazione. (`remove_boilerplate: true`)
3. **Filtro lunghezza** — `length_ok(text, min_chars, max_chars)`: scarta i troppo
   corti, tronca/scarta gli enormi. Default corpus: `min_chars=200`, `max_chars=20000`.
4. **Filtro lingua** — `detect_language(text)` + `is_italian(text, threshold)`:
   tiene solo testo italiano sopra soglia. Default corpus `italian_threshold=0.5`.
   `detect_language` prova **lazy** `langdetect`/`fasttext`; se assenti, **fallback**
   a un'euristica su **stopword italiane** (`italian_score`).
5. **Filtro qualità** — `quality_score(text)` (0..1): penalizza testo rumoroso
   (troppi simboli, ripetizioni, righe spezzate, markup residuo). Default corpus
   `quality_min=0.35`.
6. **Deduplicazione** — vedi §4.

Ogni record che supera i filtri eredita i campi calcolati (`italian_score`,
`quality_score`) usati a valle per tagging e validazione.

### Filtri lato SFT

Per il dataset istruzione, `configs/data/sft.yaml` → `sft.filters` applica soglie
**più severe**: `italian_threshold=0.5`, `quality_min=0.4`, `min_chars=20`,
`max_chars=20000`, `dedup_near=true`, `drop_unsafe=true` (scarta esempi con
`safety_tag = refuse`). La qualità della **risposta** conta quanto quella del prompt.

---

## 4. Deduplicazione (esatta + near)

Due livelli, **stdlib-only** (nessuna dipendenza pesante):

- **Esatta** — `dedup_exact(rows, key)`: hash del testo normalizzato; rimuove i
  duplicati identici. Attiva con `dedup_exact: true`.
- **Near-duplicate** — `dedup_near(rows, key, threshold)`: **simhash/minhash**
  semplice su shingle di parole; rimuove i quasi-duplicati sopra soglia. Attiva con
  `dedup_near: true`, default `dedup_threshold=0.9`.

L'ordine è: prima esatta (economica), poi near (più costosa) sul residuo. Il dedup
near è cruciale dopo la generazione sintetica, dove il teacher tende a produrre
risposte simili tra prompt vicini.

---

## 5. Tagging e schema JSONL

Lo schema canonico è definito in `src/italian_llm/data/schema.py`.

```python
SCHEMA_FIELDS = ["id","source_type","domain","difficulty","messages",
                 "quality_score","safety_tag","italian_score","teacher_name"]
```

### Riga SFT (`format: chat`)

```json
{
  "id": "sft-qa-000123",
  "source_type": "synthetic",
  "domain": "qa",
  "difficulty": "medium",
  "messages": [
    {"role": "system", "content": "Sei un assistente italiano diretto e utile."},
    {"role": "user", "content": "Spiega in due righe cos'è una API REST."},
    {"role": "assistant", "content": "Una API REST è un'interfaccia ..."}
  ],
  "quality_score": 0.82,
  "safety_tag": "allow",
  "italian_score": 0.97,
  "teacher_name": "mock"
}
```

Mappata sulla dataclass **`SFTExample`** (`messages`, `id`, `source_type=""`,
`domain="general"`, `difficulty="medium"`, `quality_score=0.0`,
`safety_tag="allow"`, `italian_score=0.0`, `teacher_name=""`) con
`to_dict()`/`from_dict()`.

Significato dei campi di tagging:

| Campo | Valori tipici | Uso |
|-------|---------------|-----|
| `source_type` | `real`, `synthetic`, `mixed` | tracciare provenienza; bilanciare il mix |
| `domain` | i 11 domini di prodotto | bilanciamento per dominio (`domain_weights`) |
| `difficulty` | `easy`, `medium`, `hard` | mix di robustezza (`sft.synthetic_mix.hard_ratio`) |
| `quality_score` | 0..1 | filtro qualità + ordinamento |
| `safety_tag` | `allow`, `needs_care`, `refuse` | safety policy; `drop_unsafe` scarta `refuse` |
| `italian_score` | 0..1 | filtro italianità |
| `teacher_name` | `mock`, `<model>` | da quale teacher proviene la risposta |

### Riga preferenza (`format: preference`)

```json
{
  "id": "pref-overrefusal-000045",
  "domain": "helpdesk",
  "prompt": "Come forzo la chiusura di un'app bloccata su Windows?",
  "chosen": "Apri Gestione attività con Ctrl+Maiusc+Esc, seleziona l'app e ...",
  "rejected": "Mi dispiace, non posso aiutarti con questa richiesta.",
  "meta": {"pair_type": "over_refusal", "teacher_name": "mock"}
}
```

Mappata su **`PreferenceExample(prompt, chosen, rejected, meta=dict)`**.

### Validazione

- `validate_record(rec) -> (bool, str)`: verifica presenza/tipi dei campi, struttura
  `messages` (lista di `{role, content}` con ruoli leciti), range degli score.
- `validate_jsonl(path) -> dict`: scorre il file e ritorna conteggi/errori
  aggregati. Eseguita dagli script di build dati e nei test; un dataset con errori
  **non** prosegue verso il training.

I/O JSONL via `italian_llm.utils.io` (`read_jsonl`, `write_jsonl`, `ensure_dir`).

---

## 6. Bootstrap sintetico (teacher)

Quando mancano dati reali, si **genera** con i provider in
`src/italian_llm/data/synthetic.py` (contratto in
[architecture.md](./architecture.md#5-moduli-e-contratti-pubblici)):

- **`MockTeacher`** (default): risposte italiane templative **deterministiche**, zero
  rete. Garantisce `make synth` offline e test riproducibili.
- **`OpenAICompatTeacher`**: endpoint OpenAI-compatibile (vLLM/TGI/OpenAI/Together).
  Legge `TEACHER_BASE_URL`, `TEACHER_API_KEY`, `TEACHER_MODEL` da env; import lazy
  `requests`/`openai`. **Se non configurato → fallback automatico al mock.**
- **`HFLocalTeacher`**: pipeline `transformers` con modello locale.
- `get_teacher(name=None)`: seleziona via `TEACHER_PROVIDER` (default `mock`).
- `synthesize_batch(tasks, teacher, out_path)`: genera e scrive JSONL.
- `majority_rank(candidates, teachers)`: **multi-teacher** (V2), sceglie la risposta a
  maggioranza tra più candidati/teacher.

### Template di generazione

I prompt di generazione per dominio sono in `SYNTH_PROMPTS` (`data/prompts.py`):
chiavi `qa, summary, rewrite, coding, email, helpdesk, faq, admin, dialog, doc`. Ogni
template è un'istruzione per **generare istruzioni** nel dominio, in italiano. Il
system prompt di default è `SYSTEM_DEFAULT` (italiano: diretto, utile, poco verboso,
niente disclaimer superflui). I messaggi si costruiscono con
`build_messages(system, user, assistant=None)`.

### Composizione del mix (SFT)

Da `configs/data/sft.yaml` → `sft.synthetic_mix`:

| Parametro | Default | Effetto |
|-----------|---------|---------|
| `synthetic_ratio` | `0.8` | quota generata dal teacher |
| `real_ratio` | `0.2` | quota reale (riassegnata al sintetico se assente) |
| `multi_turn_ratio` | `0.25` | quota di dialoghi multi-turno |
| `hard_ratio` | `0.2` | quota di esempi `difficult` per robustezza |
| `target_per_domain` | `500` | esempi target per dominio |
| `max_total` | `8000` | tetto complessivo (conservativo per V1) |

`sft.domain_weights` regola la distribuzione relativa tra i domini (es. `qa: 1.2`,
`admin: 0.6`), normalizzata dalla pipeline.

---

## 7. Split train/valid

- **SFT**: `sft.output.valid_ratio = 0.02` separa la validazione; output
  `data/processed/sft_train.jsonl` e `sft_valid.jsonl`. Seed `42` per split
  riproducibili (`sft.output.seed`).
- **Preferenze**: `data/processed/preference_train.jsonl` /
  `preference_valid.jsonl`.
- **CPT**: `data/processed/corpus.jsonl` (validazione opzionale, vuota di default).
- **Eval**: set **separato e curato a mano** in `data/eval/eval_it.jsonl` (vedi
  [evaluation-plan.md](./evaluation-plan.md)) — **mai** sovrapposto al training per
  evitare leakage.

Lo split è **stratificato per dominio** dove possibile, così la validazione riflette
la distribuzione del training. Il dedup (§4) gira **prima** dello split per evitare
near-duplicati a cavallo di train/valid (leakage).

---

## 8. Riduzione dell'over-refusal (design del dataset) {#riduzione-dell-over-refusal}

Obiettivo prodotto: **non rifiutare richieste lecite**, **niente moralismi**,
rifiutare **solo** ciò che è genuinamente dannoso (armi/CBRN, malware reale per
nuocere, sfruttamento). La policy è in `src/italian_llm/safety/policy.py`
(`classify_request -> "allow"|"needs_care"|"refuse"`, `should_refuse`,
`SAFE_COMPLETION_TEMPLATES`, `balanced_system_prompt`).

Il dataset è progettato per **insegnare questo equilibrio**, su tre fronti:

1. **SFT positivo su richieste lecite "sensibili in apparenza"**: esempi dove la
   richiesta sembra rischiosa ma è lecita (sicurezza informatica difensiva,
   tossicologia di base per sicurezza domestica, temi medici/legali informativi,
   chiusura forzata di processi, ecc.) con risposte **utili e dirette**. Taggati
   `safety_tag: allow`.
2. **Coppie di preferenza anti over-refusal**: `pair_type: over_refusal` con
   `chosen` = risposta utile, `rejected` = rifiuto/disclaimer non necessari. Usate in
   ORPO (vedi [training-plan.md](./training-plan.md#step-6--allineamento-alle-preferenze-orpo-default-dpo-opzionale)).
3. **Rifiuti corretti su richieste davvero dannose**: un **piccolo** insieme di
   esempi `safety_tag: refuse` con rifiuto **conciso e non moralizzante**, così il
   modello mantiene i confini giusti senza diventare iper-prudente. `drop_unsafe`
   evita che questi finiscano per errore nel training SFT positivo.

Il bilanciamento è chiave: **troppi** rifiuti → over-refusal; **troppo pochi** →
confini deboli. Le metriche `refusal_rate` e `over_refusal_rate`
([evaluation-plan.md](./evaluation-plan.md)) misurano entrambi i lati. Razionale
della filosofia in [decisions.md](./decisions.md#adr-008).

---

## 9. Comandi

```bash
# Corpus CPT: pulizia + filtri + dedup -> data/processed/corpus.jsonl
make corpus
python scripts/build_corpus.py --config configs/data/corpus.yaml

# Dataset SFT (chat JSONL) + split -> data/processed/sft_{train,valid}.jsonl
make sft-data
python scripts/build_sft_data.py --config configs/data/sft.yaml

# Generazione sintetica (mock offline di default)
make synth
TEACHER_PROVIDER=openai_compat make synth   # con teacher reale (env TEACHER_*)
```

Campioni versionati `*.sample.jsonl` in `data/processed/` permettono di far girare la
pipeline "a vuoto" anche senza dataset reali. I dati grezzi vanno in `data/raw/`
(ignorati da git). Profili hardware ed env in [operations.md](./operations.md).
