# italian-llm

**Costruisci il tuo LLM specializzato in italiano — e nel frattempo usa gratis un assistente di programmazione locale.**

[![CI](https://github.com/iAlias/italian-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/iAlias/italian-llm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20|%203.13-blue)](https://www.python.org/)
[![Licenza](https://img.shields.io/badge/licenza-Apache%202.0-green)](LICENSE)
[![Stile](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

🇬🇧 [Read in English](README.md)

Pipeline end-to-end che parte da un modello base **Qwen ~9B** aperto e lo porta a
essere un assistente specializzato in **italiano**: preparazione del corpus,
continued pre-training, fine-tuning supervisionato con QLoRA, allineamento alle
preferenze con ORPO, distillazione, valutazione e serving.

È pensata per Linux + CUDA, ma **l'intero repository si installa, importa ed esegue
i test senza GPU**: ogni componente pesante ha un fallback locale, sintetico o mock
funzionante. Puoi sviluppare e validare tutta la pipeline su un portatile.

L'identità del progetto è un assistente italiano **diretto, utile, poco verboso**,
che **non moralizza** e **non rifiuta richieste lecite** — la riduzione
dell'over-refusal è una metrica di prima classe, non un dettaglio.

> [!NOTE]
> **Stato del progetto.** Pipeline, test e lint verdi senza GPU. **Nessun peso è
> stato addestrato in questo progetto**: i report generati senza un modello reale
> sono smoke run (etichettati `generation_mode: mock`) e non misurano nulla. L'unica
> parte con numeri veri è l'assistente di programmazione locale servito da
> **Ollama** — vedi [Baseline misurata](#baseline-misurata): **pass@1 0.628** su
> HumanEval (164 problemi Python) e **0.596** su `humaneval-js` (161 problemi
> JavaScript), con `Qwen2.5-Coder-1.5B-Instruct` quantizzato a 4 bit, su CPU. La
> traccia italiana (CPT / SFT / ORPO / distillazione) è cablata e testata ma **non
> addestrata**: servono una GPU e un corpus reale.

---

## Indice

- [Provalo in cinque minuti](#provalo-in-cinque-minuti)
- [Baseline misurata](#baseline-misurata)
- [Due modi di usare questo repo](#due-modi-di-usare-questo-repo)
- [Addestra il tuo modello](#addestra-il-tuo-modello)
- [Profili hardware](#profili-hardware)
- [Cosa gira oggi senza GPU](#cosa-gira-oggi-senza-gpu)
- [Teacher e generazione offline](#teacher-e-generazione-offline)
- [Struttura del repository](#struttura-del-repository)
- [Sviluppo](#sviluppo)
- [Documentazione](#documentazione)
- [Licenza](#licenza)

---

## Provalo in cinque minuti

Il repository include un **assistente di programmazione locale**: `Qwen2.5-Coder`
quantizzato, che gira **su CPU** anche su hardware modesto, specializzato sullo stack
web (C#, JavaScript, HTML, CSS). Niente GPU, niente API key, niente abbonamento.

```bash
# La via più rapida — il modello precostruito
ollama pull qwen2.5-coder:1.5b
ollama run qwen2.5-coder:1.5b
```

Poi usa gli strumenti del repo contro quel modello:

```bash
# Interroga il tuo codebase (retrieval BM25, solo stdlib)
python scripts/rag_ask.py --root src --question "come viene fatto il merge della config?" --ollama-model qwen2.5-coder:1.5b

# Completamento fill-in-the-middle
python scripts/fim_complete.py --prefix "def add(a, b):\n    " --ollama-model qwen2.5-coder:1.5b

# Chat con memoria: ogni scambio viene registrato e richiamato nelle domande successive
python scripts/chat_learn.py --ollama-model qwen2.5-coder:1.5b

# Misuralo: pass@1 su un set di coding, con self-repair opzionale
python scripts/run_coding_eval.py --config configs/eval/eval_coding_ollama.yaml
```

Per costruire il **tuo** GGUF con un system prompt personalizzato invece del modello
standard:

```bash
python scripts/quantize_gguf.py --in <hf_model_dir> --out outputs/gguf/coder.q4_k_m.gguf
python scripts/export_ollama_coding.py --gguf outputs/gguf/coder.q4_k_m.gguf --name coder-local
ollama run coder-local
```

**Cosa non è:** è un junior veloce, non un architetto. Va forte su snippet,
completamento e domande sullo stack; non sostituisce un modello frontier sul coding
agentico multi-file o sul debugging di sistemi. Dettagli completi, prompt,
fine-tuning opzionale su Kaggle e limiti onesti in
[`docs/coding-model.md`](./docs/coding-model.md). Per il ciclo di **apprendimento
continuo** — memoria BM25 subito, retrain QLoRA periodico poi — vedi
[`docs/continuous-learning.md`](./docs/continuous-learning.md).

---

## Baseline misurata

I primi numeri **reali** del progetto: nessun mock, nessuna stima.

**Setup.** `coder-local` = `Qwen2.5-Coder-1.5B-Instruct` quantizzato `q4_K_M`
(986 MB) servito da Ollama; CPU Intel i7-10510U (4 core, 32 GB RAM), nessuna GPU;
system prompt di coding del repo, `temperature: 0`, `max_new_tokens: 512`, **un
solo tentativo per problema** (pass@1); il codice generato viene eseguito contro i
test ufficiali in un sottoprocesso isolato con timeout di 8 s.

| Benchmark | Problemi | Risolti | pass@1 |
|---|---|---|---|
| HumanEval (Python) | 164 | 103 | **0.628** |
| MultiPL-E `humaneval-js` (JavaScript, test `node:assert`) | 161 | 96 | **0.596** |

Riproduzione — circa 45 s per problema su questa CPU, quindi ~2 h per set:

```bash
python scripts/build_coding_eval_sets.py   # una volta: scarica i due set in data/eval/
python scripts/run_coding_eval.py --config configs/eval/eval_coding_ollama_humaneval.yaml
python scripts/run_coding_eval.py --config configs/eval/eval_coding_ollama_humaneval_js.yaml
```

Le risposte grezze vengono scritte man mano in `outputs/eval/*_preds.jsonl`: una
run interrotta riparte dai soli task mancanti con `--resume-from`, e una modifica
dell'harness si ri-valuta sulle risposte già salvate con `--rescore-from`, senza
rigenerare nulla. Se Ollama non è raggiungibile la run **si interrompe** invece di
scrivere un `pass@1` privo di significato. I risultati per-task stanno in
[`docs/baselines/`](./docs/baselines/), così un fine-tuning futuro si confronta
task per task e non solo sull'aggregato.

**Come leggere questi numeri.** Quasi tutti i fallimenti sono logica sbagliata del
modello, non attrito dell'harness: 47 `AssertionError` sui 61 fallimenti Python e
55 sui 65 JavaScript, contro un solo errore di sintassi in Python, cinque in
JavaScript e un unico timeout. È il profilo atteso per un 1.5B quantizzato: solido
su funzioni brevi e autocontenute, inaffidabile appena il problema richiede più
passaggi di ragionamento. La quantizzazione a 4 bit e il prompt non in inglese
costano qualche punto rispetto ai numeri pubblicati a piena precisione. Il
confronto utile non è con i modelli frontier, che restano molto più avanti, ma con
il costo: **0 €, nessun dato che lascia il PC, nessuna dipendenza da rete**.

**Cosa questi numeri non dicono.** Non misurano C#, HTML e CSS (il set
`data/eval/coding_csharp_web.sample.jsonl` è qualitativo, va ispezionato a mano),
non misurano lavoro multi-file o agentico e non riguardano nessun modello
addestrato qui: `coder-local` è il modello pubblico quantizzato, usato come
**baseline** che un eventuale fine-tuning deve battere.

---

## Due modi di usare questo repo

| | **Usarlo** | **Addestrare con esso** |
|---|---|---|
| Cosa ottieni | Un assistente coding locale, gratuito e privato | Un LLM specializzato in italiano tuo |
| Hardware | Qualsiasi portatile, solo CPU | Una GPU da 24 GB in su |
| Tempi | Minuti | Giorni |
| Parti da | [Provalo in cinque minuti](#provalo-in-cinque-minuti) | [Addestra il tuo modello](#addestra-il-tuo-modello) |

---

## Addestra il tuo modello

Sette passi, ognuno un target `make` e uno script:

```
[1] CORPUS ─▶ [2] SFT-DATA ─▶ [3] SYNTH ─▶ [4] CPT ─▶ [5] SFT ─▶ [6] ORPO ─▶ [7] EVAL
   raccolta      formato chat   teacher     adatt.     QLoRA     preferenze   metriche
   pulizia       qualità e      mock o      lingua     4-bit     ORPO/DPO     di prodotto
   dedup         italianità     reale       dominio                          + report
```

```bash
# 0) Ambiente
cp .env.example .env      # opzionale: punta a un teacher reale
make setup                # requirements + install editabile

# 1) Dati — girano SENZA GPU, con il mock teacher offline
make corpus               # [1] corpus CPT: pulizia, filtro lingua, dedup
make sft-data             # [2] dataset istruzione in chat JSONL
make synth                # [3] generazione sintetica dal teacher

# 2) Training — serve una GPU per i pesi veri; altrimenti smoke run
make cpt                  # [4] continued pre-training (traccia V2)
make sft                  # [5] fine-tuning supervisionato, QLoRA 4-bit
make orpo                 # [6] allineamento alle preferenze, ORPO (DPO opzionale)
make distill              # distillazione verso uno student 3B

# 3) Valuta e servi
make eval                 # [7] aderenza, italianità, verbosità, rifiuti, ROUGE-L, pass@k
make infer                # inferenza interattiva da CLI
make serve                # vLLM se disponibile, altrimenti transformers
```

Ogni target legge una config di default da `configs/`, sovrascrivibile:

```bash
make sft CFG_SFT=configs/train/sft_qwen9b_lora.yaml
python scripts/train_sft.py --config configs/train/sft_qwen9b_lora.yaml
```

Il progetto procede su **due tracce parallele** che condividono lo stesso codice.
**V1**, pragmatica: QLoRA sul base 9B, SFT e poi ORPO, distillazione data-based su
student 3B — spedibile in giorni con una sola GPU da 24 GB. **V2**, ricerca:
continued pre-training su corpus ampio, multi-teacher majority ranking, logits
distillation sperimentale, ablation, multi-GPU. Fra le due cambiano solo i file YAML
in `configs/`.

Approfondimenti: [`docs/training-plan.md`](./docs/training-plan.md),
[`docs/dataset-plan.md`](./docs/dataset-plan.md),
[`docs/distillation-plan.md`](./docs/distillation-plan.md),
[`docs/evaluation-plan.md`](./docs/evaluation-plan.md).

---

## Profili hardware

Ordini di grandezza per un giro completo della traccia indicata sul base ~9B, QLoRA
salvo diversa nota. I tempi reali variano molto con dataset, lunghezza sequenze ed
epoche.

| Profilo | GPU / VRAM | RAM | Disco | Un giro | Cosa puoi fare |
|---|---|---|---|---|---|
| **Minimum** | nessuna / 8–12 GB | 16 GB | ~30 GB | minuti | Validare il repo, far girare data prep, mock teacher, test, smoke run su modelli tiny. **Nessun training reale del 9B.** |
| **Recommended** | 1× 24 GB (RTX 3090/4090, A5000) | 32–64 GB | ~150 GB | ore | **V1 completa**: SFT QLoRA 4-bit del 9B, ORPO, distillazione data-based su student 3B, valutazione, serving locale. |
| **Serious** | 1–8× 40–80 GB (A100/H100) | 128 GB+ | 1 TB+ | da ore a giorni | **V2 completa**: CPT su corpus ampio, fine-tuning full/bf16, multi-teacher, logits distillation, ablation, DeepSpeed ZeRO. |

Sul profilo **Minimum** gli script di training, in assenza di GPU o pesi, eseguono uno
**smoke run** coerente su dati sintetici o modelli minuscoli: l'intera catena resta
verificabile. Gli adapter LoRA sono piccoli (decine o centinaia di MB); i checkpoint
del 9B no — tieni `outputs/` su un volume capiente.

---

## Cosa gira oggi senza GPU

**Funziona subito:** i moduli puri (`config`, `logging_utils`, `data/prompts`,
`data/schema`, `data/cleaning`, `safety/policy`) che importano con sola stdlib +
PyYAML · il caricamento delle config YAML con `_base_` e deep merge · la data prep con
normalizzazione Unicode, filtro lingua, scoring di qualità e italianità, dedup esatta
e near-duplicate · il mock teacher deterministico, senza rete · la generazione del
dataset sintetico end-to-end · la validazione JSONL · le metriche testuali di
valutazione · la safety policy anti over-refusal · test suite e lint.

**Cablato ma simulato,** pronto appena aggiungi GPU, pesi o dati reali: i pesi del base
Qwen ~9B, caricati dal repo HF al primo run vero · QLoRA 4-bit, bitsandbytes e
DeepSpeed, importati lazy · i loop di CPT, SFT, ORPO/DPO e logits distillation · i
teacher reali via `.env` · il serving vLLM, con fallback `transformers` · i corpora
reali, che attraversano la stessa pipeline.

---

## Teacher e generazione offline

La generazione dei dati non richiede mai un'API cloud. `TEACHER_PROVIDER` seleziona il
backend:

- `mock` (default) — risposte italiane templative e deterministiche, nessuna rete.
  Ideale per sviluppo e test.
- `openai_compat` — qualsiasi endpoint OpenAI-compatibile (vLLM, TGI, OpenAI,
  Together…), configurato con `TEACHER_BASE_URL`, `TEACHER_API_KEY`, `TEACHER_MODEL`.
  **Ricade automaticamente sul mock** se non sono impostate.
- `hf_local` — una pipeline `transformers` locale.

Per la traccia V2, `majority_rank` combina più teacher e sceglie la risposta a
maggioranza.

```bash
TEACHER_PROVIDER=mock make synth            # offline, deterministico
TEACHER_PROVIDER=openai_compat make synth   # endpoint reale, via .env
```

---

## Struttura del repository

```
configs/           configurazione YAML con _base_ e deep merge
  data/ model/ train/ distill/ eval/ serving/
src/italian_llm/   il package
  data/            schema, prompt, cleaning, teacher sintetici
  training/        cpt, sft, preference (ORPO/DPO)
  distillation/    data distillation; logits distillation (sperimentale)
  evaluation/      metriche, runner, esecuzione codice, self-repair
  serving/         inferenza, FIM, client Ollama
  rag/ memory/     indice BM25 del codice, log delle interazioni
  safety/          policy anti over-refusal
scripts/           CLI sottili in argparse, una per passo della pipeline
docs/              architettura, piani, decisioni, operatività
tests/             pytest, moduli puri e smoke test, senza GPU
notebooks/         ispezione dataset, report eval, QLoRA su Kaggle
data/ outputs/     dataset e checkpoint (ignorati da git; i campioni restano)
```

Pesi, binari e dataset generati restano fuori da git; i placeholder `.gitkeep` e i
campioni `*.sample.jsonl` sono versionati perché la pipeline possa girare a vuoto.

---

## Sviluppo

```bash
make test     # pytest — i moduli puri girano senza torch
make lint     # ruff + black --check
make help     # tutti i target disponibili
```

La CI esegue lint, format check e test su **ubuntu e windows × Python 3.12 e 3.13**,
con dipendenze pinnate in `constraints.txt`. Sono inclusi un
`.pre-commit-config.yaml` (ruff, black, igiene dei file, rilevamento chiavi private) e
un `Dockerfile` che costruisce un'immagine di inferenza CPU.

I contributi sono benvenuti: apri una issue che descriva la modifica prima di una pull
request grossa, e mantieni verdi `make test` e `make lint`.

---

## Documentazione

| Documento | Contenuto |
|---|---|
| [`architecture.md`](./docs/architecture.md) | Architettura, flusso dati, contratti dei moduli |
| [`dataset-plan.md`](./docs/dataset-plan.md) | Schema, pulizia, qualità e italianità, dedup, formati JSONL |
| [`training-plan.md`](./docs/training-plan.md) | CPT, SFT, QLoRA, preferenze ORPO/DPO, iperparametri |
| [`distillation-plan.md`](./docs/distillation-plan.md) | Data vs logits distillation e i limiti di ciascuna |
| [`evaluation-plan.md`](./docs/evaluation-plan.md) | Metriche, set di valutazione, lettura dei report |
| [`coding-model.md`](./docs/coding-model.md) | L'assistente coding locale: uso, eval, RAG, self-repair, FIM |
| [`continuous-learning.md`](./docs/continuous-learning.md) | Memoria delle interazioni e retrain periodico |
| [`operations.md`](./docs/operations.md) | Ambienti, hardware, serving, troubleshooting |
| [`decisions.md`](./docs/decisions.md) | Registro ADR delle decisioni architetturali |
| [`roadmap.md`](./docs/roadmap.md) | Roadmap a 30 giorni e checklist di rilascio |

---

## Licenza

**Apache License 2.0** — vedi [`LICENSE`](./LICENSE).

I modelli base (Qwen) e gli eventuali teacher o dataset di terze parti restano
soggetti alle **rispettive licenze**: verificale prima dell'uso e della
ridistribuzione. Questo repository non include pesi né dati proprietari.
