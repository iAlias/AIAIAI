# Italian LLM — un modello linguistico specializzato in italiano, end-to-end

> Pipeline completa per costruire, addestrare, allineare, distillare, valutare e
> servire un LLM specializzato in **italiano**, partendo da una base **Qwen ~9B**
> aperta. Pensata per **Linux + CUDA + HuggingFace + PyTorch**, ma il repository
> e' **staticamente valido e importabile anche senza GPU**: ogni componente pesante
> ha un fallback **locale / sintetico / mock** funzionante.

[![CI](https://github.com/iAlias/AIAIAI/actions/workflows/ci.yml/badge.svg)](https://github.com/iAlias/AIAIAI/actions/workflows/ci.yml)

Lingua del prodotto: **italiano**. Lingua del codice: inglese. Identità del
progetto: un assistente italiano **diretto, utile, poco verboso**, che **non
moralizza** e **non rifiuta richieste lecite** (riduzione dell'over-refusal).

> **Stato (luglio 2026).** Pipeline, test e lint verdi senza GPU. **Nessun peso
> addestrato è ancora pubblicato**: i report in `outputs/` generati senza GPU
> provengono da smoke run/mock (etichettati `generation_mode: mock`) e non
> misurano un modello reale. L'unica valutazione con modello reale oggi
> disponibile è la traccia coding via **Ollama**
> (`configs/eval/eval_coding_ollama.yaml`).

---

## Indice

1. [Panoramica](#panoramica)
2. [Assistente di programmazione locale](#assistente-di-programmazione-locale)
3. [Le due tracce: V1 pragmatica e V2 R&D](#le-due-tracce-v1-pragmatica-e-v2-rd)
4. [Struttura del repository](#struttura-del-repository)
5. [Quick start](#quick-start)
6. [Profili hardware](#profili-hardware)
7. [La pipeline in 7 step](#la-pipeline-in-7-step)
8. [Cosa è pronto subito vs cosa è simulato ma cablato](#cosa-è-pronto-subito-vs-cosa-è-simulato-ma-cablato)
9. [Mock teacher e fallback sintetico](#mock-teacher-e-fallback-sintetico)
10. [Documentazione](#documentazione)
11. [Roadmap a 30 giorni (sintesi)](#roadmap-a-30-giorni-sintesi)
12. [Checklist finale](#checklist-finale)
13. [Licenza](#licenza)

---

## Panoramica

Questo repository contiene **tutto il necessario** per portare un modello base
generalista (Qwen ~9B) a un **assistente specializzato in italiano**:

- **Preparazione dati**: pulizia, normalizzazione Unicode, rilevamento lingua,
  scoring di qualità e "italianità", deduplicazione esatta e near-duplicate.
- **Generazione sintetica**: teacher intercambiabili (mock offline, endpoint
  OpenAI-compatibile, modello HF locale) per creare istruzioni e risposte.
- **Continued Pre-Training (CPT)**: adattamento del base al dominio/lingua.
- **Supervised Fine-Tuning (SFT)**: addestramento istruzione in formato chat, QLoRA.
- **Allineamento alle preferenze**: **ORPO** di default (single-stage), **DPO** opzionale.
- **Distillazione**: verso uno student più piccolo (data distillation; logits
  distillation sperimentale, con limiti documentati).
- **Valutazione**: metriche su aderenza, italianità, verbosità, rifiuti,
  over-refusal, ROUGE-L, pass@k coding.
- **Serving**: inferenza con vLLM quando disponibile, con fallback `transformers`.

Il filo conduttore è uno stile **alla Karpathy** (nanoGPT / build-nanogpt):
loop di training leggibili, logging minimale-ma-utile, data prep ordinata —
ma **adattato al fine-tuning HF moderno** (PEFT/QLoRA reale, non un giocattolo).

Tutto il codice "puro" (config, logging, prompt, schema dati, cleaning, safety)
importa con **solo stdlib + PyYAML**. Le dipendenze pesanti (`torch`,
`transformers`, `peft`, `trl`, `datasets`, `bitsandbytes`, `accelerate`, `vllm`)
sono importate **lazy**, dentro le funzioni: il repo resta validabile su questo host
Windows senza GPU.

---

## Assistente di programmazione locale

Il repository include anche uno **strumento specializzato per il coding locale**: un assistente basato su `Qwen2.5-Coder` quantizzato, che gira in **locale senza GPU**, specifico su **C#, JavaScript, HTML, CSS** (web stack).

**Cosa fa:**
- Gira veloce su hardware modesto (Intel i7-10510U, 4 core, 32 GB RAM).
- È forte su snippet, completamento, Q&A dello stack.
- Costa **€0** (sia da costruire sia da eseguire).
- È privato e personalizzabile.

**Cosa non fa:**
- Non è un sostituto di Claude Opus 4.8 o modelli frontier.
- Su coding agentico complesso, multi-file, debugging di sistemi — il gap è enorme.
- È un "junior veloce", non un architetto.

**Quick start:**
```bash
# Opzione 1: usa il modello precostruito (più veloce)
ollama pull qwen2.5-coder:1.5b
ollama run qwen2.5-coder:1.5b

# Opzione 2: costruisci il tuo GGUF con il sistema prompt personalizzato
python scripts/quantize_gguf.py --in <hf_model_dir> --out outputs/gguf/coder.q4_k_m.gguf
python scripts/export_ollama_coding.py --gguf outputs/gguf/coder.q4_k_m.gguf --name coder-local
ollama run coder-local

# Valuta il modello reale via Ollama (pass@1, opzionale --repair 3)
python scripts/run_coding_eval.py --config configs/eval/eval_coding_ollama.yaml

# RAG sul tuo codebase, FIM autocomplete, chat con memoria persistente
python scripts/rag_ask.py --root src --question "..." --ollama-model coder-local
python scripts/fim_complete.py --prefix "def add(a,b):\n    " --ollama-model coder-local
python scripts/chat_learn.py --ollama-model coder-local
```

Per dettagli completi, prompt, fine-tuning opzionale su Kaggle, e limiti onesti, vedi **[`docs/coding-model.md`](./docs/coding-model.md)**. Per il ciclo di **apprendimento continuo** (il modello impara da ogni domanda: memoria BM25 immediata + retrain periodico), vedi **[`docs/continuous-learning.md`](./docs/continuous-learning.md)**.

---

## Le due tracce: V1 pragmatica e V2 R&D

Il progetto è organizzato su **due tracce parallele** che condividono lo stesso
codice ma hanno obiettivi diversi.

### V1 — Traccia pragmatica ("spedire qualcosa di utile")

Obiettivo: ottenere **velocemente** un assistente italiano buono, con il minimo
rischio e il minimo costo.

- Base Qwen ~9B + **QLoRA 4-bit**.
- **SFT** su dati istruzione (reali + sintetici dal teacher) → **ORPO** per lo stile.
- Distillazione **data-based** verso uno student 3B per il deploy economico.
- Valutazione orientata al prodotto (aderenza, italianità, verbosità, over-refusal).
- Serving con vLLM o `transformers`.

Tempi tipici: **giorni**, non settimane. Hardware: una singola GPU 24 GB è sufficiente
per iniziare (profilo *recommended*).

### V2 — Traccia R&D ("spingere la qualità")

Obiettivo: massimizzare la qualità e studiare il comportamento del modello.

- **Continued Pre-Training** su corpus italiano ampio prima dell'SFT.
- Preferenze con **multi-teacher majority ranking** e dataset di preferenze curato.
- **Logits distillation** (KL su logit del teacher) — sperimentale, con limiti noti
  (richiede teacher e student con vocabolario allineato; documentato in `docs/`).
- Ablation su metriche, mix di dati, lunghezza sequenze, sched.
- Hardware: multi-GPU / GPU "serious" (40–80 GB), DeepSpeed ZeRO.

Le due tracce **non si escludono**: V1 produce un baseline spedibile; V2 lo supera
quando ci sono dati e GPU. Lo stesso `config` + gli stessi script servono entrambe,
cambiando solo i file YAML in `configs/`.

---

## Struttura del repository

```
AIAIAI/
├── README.md                  # questo file
├── LICENSE                    # Apache 2.0
├── pyproject.toml             # metadata, src layout, ruff/black/pytest
├── requirements.txt           # dipendenze (core leggero + blocchi pesanti opzionali)
├── constraints.txt            # versioni pinnate per install riproducibili (CI)
├── Makefile                   # orchestratore pipeline (make help)
├── .env.example               # variabili ambiente (teacher, HF, W&B, CUDA)
├── .gitignore
│
├── configs/                   # configurazione YAML (con _base_ + deep merge)
│   ├── base.yaml              # progetto, paths, model, logging
│   ├── data/
│   │   ├── corpus.yaml        # corpus CPT (pulizia, dedup, blocchi)
│   │   └── sft.yaml           # dataset istruzione (split, sintesi)
│   ├── model/
│   │   ├── qwen9b.yaml        # base Qwen2.5-7B ("~9B", vedi ADR-001)
│   │   ├── qwen_coder.yaml    # Qwen2.5-Coder per la traccia coding
│   │   └── student_3b.yaml    # student di distillazione
│   ├── train/
│   │   ├── cpt_qwen9b.yaml    # Continued Pre-Training
│   │   ├── sft_qwen9b_lora.yaml  # Supervised Fine-Tuning (QLoRA)
│   │   ├── sft_coder.yaml     # SFT del coder locale
│   │   └── orpo_qwen9b.yaml   # preferenze (ORPO/DPO)
│   ├── distill/
│   │   └── student_3b.yaml    # distillazione verso student 3B
│   ├── eval/
│   │   ├── eval.yaml          # set + metriche di valutazione
│   │   ├── eval_coding.yaml   # eval coding (mock/transformers)
│   │   └── eval_coding_ollama.yaml  # eval coding con modello reale via Ollama
│   └── serving/
│       └── vllm.yaml          # serving vLLM
│
├── src/italian_llm/           # package (src layout)
│   ├── config.py              # load_config / deep_merge / get / dump_config
│   ├── logging_utils.py       # setup_logging / get_logger
│   ├── tokenizer_utils.py     # load_tokenizer / format_chat
│   ├── utils/                 # io (read/write_jsonl), seed
│   ├── data/                  # schema, prompts, cleaning, synthetic (teacher)
│   ├── training/              # common, cpt, sft, preference (ORPO/DPO)
│   ├── distillation/          # data_distill, logits_distill (sperimentale)
│   ├── evaluation/            # metrics, runner, code_exec, code_eval, self_repair
│   ├── serving/               # inference, fim, ollama_client
│   ├── rag/                   # code_index (BM25 stdlib sul codebase)
│   ├── memory/                # interaction_log (apprendimento continuo)
│   └── safety/                # policy anti over-refusal
│
├── scripts/                   # CLI sottili (argparse) — vedi Makefile
│   ├── download_open_data.py            # [0] dati grezzi (fallback sintetico)
│   ├── build_italian_corpus.py          # [1] corpus CPT
│   ├── clean_dedup_filter.py            # [1b] pulizia profonda + dedup
│   ├── prepare_continued_pretraining_data.py  # [1c] blocchi CPT
│   ├── create_instruction_dataset.py    # [2a] istruzioni dal corpus
│   ├── prepare_sft_data.py              # [2] dataset SFT (train/valid/test)
│   ├── synthesize_with_teachers.py      # [3] generazione sintetica (teacher)
│   ├── train_cpt.py                     # [4]
│   ├── train_sft.py                     # [5]
│   ├── train_dpo_or_orpo.py             # [6]
│   ├── distill_student.py               # distillazione
│   ├── evaluate_all.py                  # [7] valutazione
│   ├── run_inference.py                 # inferenza CLI
│   ├── launch_vllm.sh                   # serving vLLM
│   ├── export_model.py / quantize_gguf.py / export_ollama_coding.py
│   ├── build_coding_sft.py / run_coding_eval.py   # traccia coding
│   ├── rag_ask.py / fim_complete.py / chat_learn.py  # RAG, FIM, memoria
│   └── bootstrap_project.sh / setup_env.sh
│
├── docs/                      # documentazione approfondita (vedi sotto)
├── notebooks/                 # ispezione dataset, report eval, QLoRA su Kaggle
├── tests/                     # test pytest (moduli puri + smoke, niente GPU)
│
├── data/
│   ├── raw/ interim/          # corpora grezzi e intermedi (ignorati da git)
│   ├── processed/ synthetic/  # output generati (ignorati; campioni *.sample.jsonl versionati)
│   ├── memory/                # log interazioni chat_learn (ignorato: privacy)
│   ├── eval/                  # set di valutazione versionati
│   └── manifests/sources.yaml # sorgenti corpus
│
└── outputs/                   # checkpoint, adapter, report (ignorati da git)
```

> I percorsi `data/raw`, `data/interim`, `outputs/*` e i pesi binari
> (`*.safetensors`, `*.bin`, `*.gguf`, ...) sono **esclusi da git**. I placeholder
> `.gitkeep` e i campioni `*.sample.jsonl` restano **versionati** per far girare la
> pipeline a vuoto.

---

## Quick start

> Target Linux + CUDA. Su Windows (questo host) il repo è validabile staticamente
> e gli step "leggeri" girano lo stesso; per i comandi `make` su Windows usa WSL,
> Git Bash, oppure invoca gli script direttamente con `python scripts/...`.

```bash
# 0) Ambiente
cp .env.example .env            # personalizza se hai un teacher/endpoint reale
make setup                      # pip install -r requirements.txt + pip install -e .

# 1) Dati (girano SENZA GPU, usano il mock teacher offline)
make corpus                     # [1] prepara il corpus CPT (pulizia + dedup)
make sft-data                   # [2] costruisce/pulisce il dataset SFT (chat JSONL)
make synth                      # [3] genera dati sintetici dal teacher (mock di default)

# 2) Training (richiede GPU per i pesi veri; in mock fa uno smoke run)
make cpt                        # [4] Continued Pre-Training (V2)
make sft                        # [5] Supervised Fine-Tuning (QLoRA)
make orpo                       # [6] allineamento alle preferenze (ORPO; DPO opzionale)
make distill                    # distillazione verso student 3B

# 3) Valutazione, inferenza, serving
make eval                       # [7] metriche (aderenza, italianita', verbosita', rifiuti)
make infer                      # inferenza interattiva da CLI
make serve                      # server (vLLM se disponibile, altrimenti transformers)

# Qualita' del codice
make test                       # pytest (moduli puri girano senza torch)
make lint                       # ruff + black --check
make help                       # elenco completo dei target
```

Ogni target del `Makefile` invoca lo script corrispondente con una config di
default in `configs/`. Puoi sovrascrivere la config, es:

```bash
make sft CFG_SFT=configs/train/sft_qwen9b_lora.yaml
python scripts/train_sft.py --config configs/train/sft_qwen9b_lora.yaml
```

---

## Profili hardware

Tre profili di riferimento. I tempi sono **ordini di grandezza** per un giro
completo della traccia indicata sul base ~9B (QLoRA salvo diversa nota); variano
molto con dataset, lunghezza sequenze e numero di epoche.

| Profilo         | GPU / VRAM            | RAM    | Disco  | Tempo (giro) | Cosa puoi fare |
|-----------------|-----------------------|--------|--------|--------------|----------------|
| **Minimum**     | nessuna / 8–12 GB     | 16 GB  | ~30 GB | minuti       | Validare il repo, far girare data prep, mock teacher, test, smoke run su modelli tiny. **Nessun training reale del 9B.** |
| **Recommended** | 1× 24 GB (RTX 3090/4090, A5000) | 32–64 GB | ~150 GB | ore | **V1 completa**: SFT QLoRA 4-bit del 9B, ORPO, distillazione data-based su student 3B, valutazione, serving locale. |
| **Serious**     | 1–8× 40–80 GB (A100/H100) | 128 GB+ | 1 TB+ | da ore a giorni | **V2 completa**: CPT su corpus ampio, full/bf16 fine-tuning, multi-teacher, logits distillation, ablation, DeepSpeed ZeRO, serving ad alto throughput. |

Note pratiche:

- **Minimum** non addestra davvero il 9B: serve a **sviluppare e validare la
  pipeline** (dati, prompt, config, metriche) e a far girare i test. Gli script di
  training, in assenza di GPU/pesi, eseguono uno **smoke run** coerente su dati
  sintetici e/o modelli minuscoli, così l'intera catena resta verificabile.
- **Recommended** è il punto dolce per la **traccia V1**: 24 GB bastano per QLoRA
  4-bit sul 9B con `gradient_checkpointing` e batch piccolo + `grad_accum`.
- **Serious** abilita la **traccia V2**: CPT, sequenze lunghe, throughput, e gli
  esperimenti R&D (incluso il logits distillation, che è esigente).
- Disco: i checkpoint del 9B e gli intermedi crescono in fretta; tieni `outputs/`
  su un volume capiente. Gli adapter LoRA invece sono piccoli (decine/centinaia di MB).

---

## La pipeline in 7 step

```
[1] CORPUS  ─▶ [2] SFT-DATA ─▶ [3] SYNTH ─▶ [4] CPT ─▶ [5] SFT ─▶ [6] ORPO ─▶ [7] EVAL
   raccolta/      formato chat    teacher      adatt.    istruz.    preferenze   metriche
   pulizia/       + qualità/      mock o       lingua/   QLoRA      ORPO/DPO     prodotto
   dedup          italianità      reale        dominio                          + report
```

1. **Corpus (CPT)** — `make corpus` → `scripts/build_italian_corpus.py`
   (più `make clean-data` → `scripts/clean_dedup_filter.py` e `make cpt-data` →
   `scripts/prepare_continued_pretraining_data.py`).
   Raccolta e pulizia del testo italiano: normalizzazione Unicode, rimozione
   boilerplate, filtro lingua (`is_italian`), scoring qualità, dedup esatta/near.
   Output in `data/processed/`.

2. **SFT-data** — `make sft-data` → `scripts/prepare_sft_data.py`.
   Costruzione del dataset istruzione in **formato chat JSONL** secondo lo schema:
   `{"id","source_type","domain","difficulty","messages":[...],"quality_score",
   "safety_tag","italian_score","teacher_name"}`. Validazione con `validate_jsonl`.

3. **Synth** — `make synth` → `scripts/synthesize_with_teachers.py`.
   Generazione sintetica di istruzioni/risposte tramite i **teacher**
   (`MockTeacher` di default, `OpenAICompatTeacher`, `HFLocalTeacher`). Supporta
   `majority_rank` multi-teacher per la traccia V2. Nessuna rete richiesta col mock.

4. **CPT** — `make cpt` → `scripts/train_cpt.py` (`run_cpt`).
   Continued Pre-Training del base sul corpus italiano (traccia V2). Quantizzazione
   e dtype dalla config; LoRA opzionale.

5. **SFT** — `make sft` → `scripts/train_sft.py` (`run_sft`).
   Supervised Fine-Tuning in formato chat con **QLoRA 4-bit** (PEFT). Loop leggibile,
   logging essenziale, salvataggio adapter in `outputs/`.

6. **ORPO / preferenze** — `make orpo` → `scripts/train_dpo_or_orpo.py` (`run_preference`).
   Allineamento allo **stile** e alle preferenze: **ORPO** (single-stage, default,
   via `trl`) oppure **DPO**. Dataset di preferenze in formato `{"prompt","chosen",
   "rejected","meta"}`.

7. **Eval** — `make eval` → `scripts/evaluate_all.py` (`run_eval`).
   Valutazione orientata al prodotto: **aderenza** alle istruzioni, **italianità**,
   **verbosità**, **refusal/over-refusal**, **ROUGE-L**, **pass@k** coding. Produce un
   report in `outputs/`.

Step di supporto: **distillazione** (`make distill`), **inferenza** (`make infer`),
**serving** (`make serve`).

---

## Cosa è pronto subito vs cosa è simulato ma cablato

Trasparenza totale: su questo host (Windows, no GPU, no dati proprietari) **alcune
cose funzionano davvero subito**, altre sono **simulate ma completamente cablate**
— cioè il codice, le config e i contratti ci sono e girano end-to-end appena
aggiungi GPU/pesi/dati reali.

### Pronto subito (gira ora, senza GPU)

- **Moduli puri**: `config`, `logging_utils`, `data/prompts`, `data/schema`,
  `data/cleaning`, `safety/policy` — importano con solo stdlib + PyYAML.
- **Caricamento/merge config** YAML con `_base_` + deep merge + override.
- **Data prep**: normalizzazione, filtro lingua euristico, scoring qualità/italianità,
  dedup esatta e near-duplicate (simhash/minhash stdlib).
- **Mock teacher**: risposte italiane templative **deterministiche**, zero rete.
- **Generazione del dataset sintetico** end-to-end col mock (`make synth`).
- **Validazione JSONL** (schema, conteggi, errori).
- **Metriche di valutazione testuali** (aderenza, italianità, verbosità, refusal,
  ROUGE-L) calcolabili su predizioni d'esempio.
- **Safety policy** anti over-refusal (classificazione richiesta, template).
- **Test suite** (`make test`) e **lint** (`make lint`).

### Simulato ma cablato (pronto appena hai GPU/pesi/dati)

- **Pesi del base Qwen ~9B**: non scaricati qui; `model.name` in config punta al repo
  HF, si carica al primo run reale. Senza pesi/GPU, gli script di training fanno uno
  **smoke run** coerente.
- **QLoRA 4-bit / bitsandbytes / DeepSpeed**: importati lazy; attivi su Linux+CUDA.
- **CPT, SFT, ORPO/DPO, logits distillation**: loop completi e configurati; richiedono
  GPU per i pesi veri (il logits distillation è **sperimentale**, limiti in `docs/`).
- **Teacher reale** (`OpenAICompatTeacher` / `HFLocalTeacher`): cablati; bastano le
  variabili in `.env` (`TEACHER_BASE_URL`, `TEACHER_API_KEY`, `TEACHER_MODEL`) o un
  modello HF locale. In assenza, **fallback automatico al mock**.
- **Serving vLLM**: configurato in `configs/serving/vllm.yaml`; su host senza vLLM si
  usa il fallback `transformers`.
- **Corpora/dataset reali**: qui ci sono solo **campioni** `*.sample.jsonl`; i dati
  veri vanno in `data/raw` (ignorati da git) e attraversano la stessa pipeline.

---

## Mock teacher e fallback sintetico

La generazione dati **non dipende da cloud o API proprietarie**. Il provider è
selezionato da `TEACHER_PROVIDER` (default `mock`):

- `mock` → **`MockTeacher`**: risposte italiane templative e **deterministiche**,
  nessuna rete. Perfetto per sviluppare la pipeline e per i test.
- `openai_compat` → **`OpenAICompatTeacher`**: qualsiasi endpoint
  OpenAI-compatibile (vLLM, TGI, OpenAI, Together, ...). Legge `TEACHER_BASE_URL`,
  `TEACHER_API_KEY`, `TEACHER_MODEL`. Import lazy di `requests`/`openai`.
  **Se le variabili non sono impostate, ricade automaticamente sul mock.**
- `hf_local` → **`HFLocalTeacher`**: pipeline `transformers` con un modello locale.

Per la traccia V2, `majority_rank` combina più teacher e sceglie la risposta a
maggioranza. In ogni caso, **niente è obbligatoriamente online**: il mock garantisce
che `make synth` produca dati validi anche completamente offline.

```bash
# Offline, deterministico (default):
TEACHER_PROVIDER=mock make synth

# Con un endpoint reale OpenAI-compatibile:
#   imposta TEACHER_BASE_URL / TEACHER_API_KEY / TEACHER_MODEL in .env
TEACHER_PROVIDER=openai_compat make synth
```

---

## Documentazione

La documentazione approfondita vive in `docs/`:

- [`docs/architecture.md`](./docs/architecture.md) — architettura, flusso dati, contratti dei moduli.
- [`docs/dataset-plan.md`](./docs/dataset-plan.md) — schema dati, pulizia, qualità/italianità, dedup, formati JSONL.
- [`docs/training-plan.md`](./docs/training-plan.md) — CPT, SFT, QLoRA, preferenze ORPO/DPO, iperparametri.
- [`docs/distillation-plan.md`](./docs/distillation-plan.md) — data vs logits distillation e limiti.
- [`docs/evaluation-plan.md`](./docs/evaluation-plan.md) — metriche, set di valutazione, interpretazione dei report.
- [`docs/coding-model.md`](./docs/coding-model.md) — assistente coding locale: uso, eval, RAG, self-repair, FIM.
- [`docs/continuous-learning.md`](./docs/continuous-learning.md) — memoria delle interazioni + retrain periodico.
- [`docs/operations.md`](./docs/operations.md) — operatività: ambienti, hardware, serving, troubleshooting.
- [`docs/decisions.md`](./docs/decisions.md) — registro ADR di tutte le decisioni architetturali.

---

## Roadmap a 30 giorni (sintesi)

Piano indicativo per portare il progetto da repo a modello spedibile (V1) e gettare
le basi della V2. I piani di dettaglio sono in `docs/dataset-plan.md`,
`docs/training-plan.md`, `docs/evaluation-plan.md` e `docs/distillation-plan.md`.

| Settimana | Obiettivo | Output |
|-----------|-----------|--------|
| **1 — Fondamenta & dati** | Ambiente, config, data prep, mock teacher, prime sintesi. Validazione schema e cleaning. | Dataset SFT v0 (reale + sintetico), pipeline dati verde, test verdi. |
| **2 — SFT v1** | SFT QLoRA del 9B sul dataset v0; primo giro di valutazione; iterazione su prompt/qualità. | Adapter SFT v1, report metriche baseline, fix data quality. |
| **3 — Allineamento & valutazione** | Dataset di preferenze; ORPO; riduzione over-refusal; suite di valutazione ampliata. | Modello allineato (stile + safety bilanciata), report comparativo. |
| **4 — Distillazione, serving & hardening** | Distillazione data-based su student 3B; serving vLLM/transformers; ablation leggere; documentazione. | Student 3B servibile, endpoint di inferenza, V1 "spedibile", backlog V2 (CPT, logits distill). |

Milestone trasversali: CI verde (`make test`/`make lint`), config riproducibili,
metriche tracciate, `.env`/segreti gestiti correttamente.

---

## Checklist finale

Prima di considerare un giro "completo", verifica:

- [ ] `make setup` completa senza errori; `.env` creato da `.env.example`.
- [ ] `make test` e `make lint` verdi.
- [ ] Moduli puri importabili **senza** torch (`config`, `prompts`, `schema`,
      `cleaning`, `safety`).
- [ ] `make corpus`, `make sft-data`, `make synth` producono output validi (mock, offline).
- [ ] `validate_jsonl` non segnala errori sui dataset generati.
- [ ] SFT (QLoRA) gira sul profilo *recommended*; adapter salvato in `outputs/`.
- [ ] ORPO eseguito; lo stile è più diretto/meno verboso; **over-refusal ridotto**.
- [ ] `make eval` produce un report con aderenza, italianità, verbosità, refusal,
      over-refusal, ROUGE-L (e pass@k dove applicabile).
- [ ] Distillazione data-based produce uno student 3B funzionante.
- [ ] `make serve` espone l'inferenza (vLLM o fallback transformers).
- [ ] Nessun peso/binario o segreto committato (rispetto del `.gitignore`).
- [ ] Documentazione `docs/` allineata al comportamento effettivo.

---

## Licenza

Distribuito sotto **Apache License 2.0** — vedi [`LICENSE`](./LICENSE).

I modelli base (Qwen) e gli eventuali teacher/dataset di terze parti restano
soggetti alle **rispettive licenze**: verificale prima dell'uso e della
ridistribuzione. Questo repository non include pesi né dati proprietari.
