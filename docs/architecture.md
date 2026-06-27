# Architettura del sistema

> Mappa tecnica del progetto **Italian LLM**: backbone, flusso dati, contratti dei
> moduli, le due tracce **V1 / V2** e la nota onesta su **attention backend** e
> **dynamic sparse attention** (cosa è plug-and-play e cosa no).

Documenti collegati:
[training-plan.md](./training-plan.md) ·
[dataset-plan.md](./dataset-plan.md) ·
[distillation-plan.md](./distillation-plan.md) ·
[evaluation-plan.md](./evaluation-plan.md) ·
[operations.md](./operations.md) ·
[decisions.md](./decisions.md)

---

## 1. Obiettivo e principi

Costruire, end-to-end, un **assistente specializzato in italiano** partendo da una
base aperta **Qwen ~9B**. L'identità del prodotto è precisa: risposte **dirette,
utili, poco verbose**, **niente moralismi né disclaimer superflui**, e soprattutto
**riduzione dell'over-refusal** sulle richieste lecite (vedi
`src/italian_llm/safety/policy.py` e [decisions.md](./decisions.md#adr-008)).

Principi architetturali che vincolano tutto il codice:

1. **Src layout**: il package è `italian_llm`, le sorgenti vivono in
   `src/italian_llm/`. `pyproject.toml` imposta `package-dir = {"" = "src"}`.
2. **Import pesanti lazy**: `torch`, `transformers`, `peft`, `trl`, `datasets`,
   `bitsandbytes`, `accelerate`, `vllm` sono importati **dentro le funzioni**, mai a
   livello di modulo. Conseguenza: i moduli "puri" (`config`, `logging_utils`,
   `data/prompts`, `data/schema`, `data/cleaning`, `safety/policy`, gran parte di
   `evaluation/metrics`) importano con **solo stdlib + PyYAML**. Questo è ciò che
   rende il repo **staticamente valido su Windows senza GPU**.
3. **Fallback ovunque**: ogni componente pesante ha un percorso
   **locale / sintetico / mock** funzionante (mock teacher, smoke run di training,
   fallback `transformers` al posto di vLLM, euristica lingua al posto di
   `langdetect`/`fasttext`).
4. **Config dichiarativa**: tutto passa per YAML in `configs/` con `_base_` +
   deep-merge (vedi `src/italian_llm/config.py`).
5. **CLI sottili**: gli script in `scripts/*.py` usano solo `argparse` (stdlib) e
   inseriscono `src/` nel path con un header standard; la logica vera sta nel package.

---

## 2. Vista d'insieme (flusso dati)

```
                       configs/*.yaml  (_base_ + deep merge)
                                  │
                                  ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │                          PREPARAZIONE DATI                              │
  │  data/raw ──▶ cleaning ──▶ language/quality filter ──▶ dedup ──▶ tag    │
  │  synthetic.py (TeacherProvider) ──▶ istruzioni/risposte sintetiche      │
  │  output: data/processed/*.jsonl  (schema in data/schema.py)             │
  └───────────────────────────────────────────────────────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
      corpus.jsonl          sft_train.jsonl      preference_train.jsonl
            │                     │                     │
            ▼                     ▼                     ▼
  ┌──────────────┐      ┌──────────────┐      ┌──────────────────────┐
  │  CPT (V2)    │ ───▶ │  SFT (QLoRA) │ ───▶ │  Preferenze ORPO/DPO  │
  │ training/cpt │      │ training/sft │      │ training/preference   │
  └──────────────┘      └──────────────┘      └──────────────────────┘
                                  │                     │
                                  ▼                     ▼
                          adapter LoRA in outputs/  (SFT, poi ORPO)
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
         ┌─────────────────────┐     ┌────────────────────────┐
         │  Distillazione 3B   │     │  Valutazione (metriche) │
         │  distillation/*     │     │  evaluation/runner.py   │
         └─────────────────────┘     └────────────────────────┘
                    │                           │
                    ▼                           ▼
              student 3B servibile         report.json + preds.jsonl
                    │
                    ▼
         ┌─────────────────────────────────────────┐
         │  Serving: vLLM  (fallback transformers)  │
         │  serving/inference.py · configs/serving  │
         └─────────────────────────────────────────┘
```

La pipeline operativa a **7 step** (corpus → sft-data → synth → cpt → sft → orpo →
eval) è descritta in dettaglio in [training-plan.md](./training-plan.md) e mappata
1:1 sui target del `Makefile`.

---

## 3. Backbone: perché Qwen e la deviazione ~9B → 7B/3B

Il target dichiarato è una **base aperta Qwen ~9B**. Alla data del progetto, nella
linea **Qwen2.5** non esiste un checkpoint "9B" ufficiale: la variante aperta stabile
e ampiamente disponibile su HuggingFace più vicina a quella fascia è
**`Qwen/Qwen2.5-7B`** (7.6B parametri). È quindi il **default** in
`configs/model/qwen9b.yaml` e in `configs/base.yaml`.

Motivazioni della scelta (dettaglio in [decisions.md](./decisions.md#adr-001)):

- **Tokenizer forte sull'italiano** e licenza aperta utilizzabile per fine-tuning.
- **Headroom di VRAM**: la 7B in **QLoRA 4-bit** sta comodamente su una singola GPU
  da 24 GB (profilo *recommended*), lasciando margine per `gradient_checkpointing`,
  sequenze a 2048 e `grad_accum`.
- **Famiglia coerente** tra teacher (~7B allineato) e **student `Qwen/Qwen2.5-3B`**
  (`configs/model/student_3b.yaml`): stesso tokenizer e stessa famiglia, requisito
  importante soprattutto per la **logits distillation** (vedi §6 e
  [distillation-plan.md](./distillation-plan.md)).

Per salire verso i ~9B reali si punta a `Qwen/Qwen2.5-14B` riducendo
batch/sequenza: è un **cambio di una riga** nel campo `model.name` del config,
perché tutto il resto (quantizzazione, LoRA target modules, loop) è invariato.

I valori `model.name` e `model.quantization.*` sono definiti una sola volta e
ereditati da tutti i config di training via `_base_`.

---

## 4. Le due tracce: V1 (pragmatica) e V2 (R&D)

Le due tracce **condividono lo stesso codice e gli stessi script**: cambiano solo i
file YAML in `configs/`. Questa è una scelta architetturale deliberata — niente fork
di codice tra "produzione" e "ricerca".

### V1 — pragmatica ("spedire qualcosa di utile")

| Aspetto        | Scelta V1 |
|----------------|-----------|
| Base           | `Qwen/Qwen2.5-7B` in **QLoRA 4-bit** |
| Pipeline       | SFT (chat) → **ORPO** per lo stile |
| Distillazione  | **data distillation** verso student 3B |
| Hardware       | 1× GPU 24 GB (profilo *recommended*) |
| Tempi          | giorni, non settimane |
| CPT            | **saltato** (si parte dal base già pre-addestrato) |

### V2 — R&D ("spingere la qualità")

| Aspetto        | Scelta V2 |
|----------------|-----------|
| CPT            | **Continued Pre-Training** su corpus italiano ampio prima dell'SFT |
| Preferenze     | dataset curato + **multi-teacher majority ranking** (`majority_rank`) |
| Distillazione  | **logits distillation** (KL sui logit) — sperimentale, limiti noti |
| Hardware       | multi-GPU 40–80 GB, DeepSpeed ZeRO |
| Tempi          | da ore a giorni per giro |

Le due tracce non si escludono: V1 produce un baseline spedibile, V2 lo supera
quando ci sono dati e GPU. Confronto hardware completo in
[operations.md](./operations.md#profili-hardware).

---

## 5. Moduli e contratti pubblici

Il package è organizzato in sottomoduli con confini netti. La colonna **"Puro?"**
indica se il modulo importa **senza** torch (solo stdlib + PyYAML).

| Modulo | Simboli chiave (contratto) | Puro? |
|--------|----------------------------|:-----:|
| `italian_llm.config` | `load_config`, `deep_merge`, `get`, `dump_config` | ✅ |
| `italian_llm.logging_utils` | `setup_logging`, `get_logger` | ✅ |
| `italian_llm.tokenizer_utils` | `load_tokenizer`, `format_chat` | ❌ (lazy `transformers`) |
| `italian_llm.utils.io` | `read_jsonl`, `write_jsonl`, `ensure_dir` | ✅ |
| `italian_llm.utils.seed` | `set_seed` | ✅ (torch/np lazy) |
| `italian_llm.data.schema` | `SCHEMA_FIELDS`, `SFTExample`, `PreferenceExample`, `validate_record`, `validate_jsonl` | ✅ |
| `italian_llm.data.prompts` | `SYSTEM_DEFAULT`, `build_messages`, `render_plain`, `SYNTH_PROMPTS` | ✅ |
| `italian_llm.data.cleaning` | `normalize_unicode`, `detect_language`, `is_italian`, `italian_score`, `quality_score`, `dedup_exact`, `dedup_near` | ✅ (langdetect lazy) |
| `italian_llm.data.synthetic` | `TeacherProvider`, `MockTeacher`, `OpenAICompatTeacher`, `HFLocalTeacher`, `get_teacher`, `synthesize_batch`, `majority_rank` | ⚠️ parziale (mock puro) |
| `italian_llm.training.common` | `load_model_and_tokenizer`, `get_bnb_config`, `apply_lora`, `count_trainable_params` | ❌ |
| `italian_llm.training.cpt` | `run_cpt(cfg) -> str` | ❌ |
| `italian_llm.training.sft` | `run_sft(cfg) -> str` | ❌ |
| `italian_llm.training.preference` | `run_preference(cfg) -> str` | ❌ |
| `italian_llm.distillation.data_distill` | `build_distill_dataset(cfg) -> str` | ❌ |
| `italian_llm.distillation.logits_distill` | `run_logits_distill(cfg) -> str` | ❌ |
| `italian_llm.evaluation.metrics` | `instruction_adherence`, `verbosity_ratio`, `italianity_score`, `refusal_rate`, `over_refusal_rate`, `rouge_l`, `coding_passk`, `is_refusal` | ✅ (in gran parte) |
| `italian_llm.evaluation.runner` | `run_eval(cfg) -> dict` | ❌ (carica modello) |
| `italian_llm.serving.inference` | `Generator(...).generate(...)` | ❌ (lazy `transformers`) |
| `italian_llm.safety.policy` | `classify_request`, `should_refuse`, `SAFE_COMPLETION_TEMPLATES`, `balanced_system_prompt` | ✅ |

**Regola di dipendenza**: i moduli puri non importano mai i moduli pesanti a
livello di modulo. I moduli pesanti possono usare quelli puri liberamente. Questo
grafo aciclico è ciò che garantisce l'import pulito senza GPU.

### Contratto di config (`_base_` + deep merge)

`load_config(path, overrides=None)` risolve la chiave `_base_` **relativamente alla
directory del file YAML**, effettua un **deep merge** (i dizionari si fondono in
profondità; gli scalari e le liste vengono sovrascritti dal figlio) e infine applica
gli `overrides`. `get(cfg, "train.lr", default)` legge per chiave puntata.
`dump_config` riserializza per riproducibilità. Schema delle chiavi in
[dataset-plan.md](./dataset-plan.md) e [training-plan.md](./training-plan.md).

---

## 6. Attention backend: nota di astrazione (onesta)

Domanda ricorrente: *"possiamo cambiare il meccanismo di attenzione per andare più
veloci o più lunghi?"*. La risposta richiede di distinguere **tre cose diverse** che
spesso vengono confuse. Questa sezione è volutamente esplicita su **cosa è
plug-and-play e cosa no**.

### 6.1 Attention backend (kernel) — **plug-and-play**

Il "backend" è solo **l'implementazione del kernel** della stessa identica
operazione matematica (softmax attention). Esempi: eager, SDPA (PyTorch
`scaled_dot_product_attention`), **FlashAttention-2**, xFormers memory-efficient.

- **Cosa cambia**: velocità e uso di memoria. **Non** cambia il modello, i pesi, il
  comportamento o la qualità (a meno di differenze numeriche trascurabili).
- **Come si attiva**: parametro `attn_implementation` al caricamento del modello
  (`transformers`), oppure semplicemente girando su vLLM lato serving. Non richiede
  **alcun retraining**.
- **Stato nel repo**: il caricamento avviene in `training/common.py`
  (`load_model_and_tokenizer`) e in `serving/inference.py`; il backend è scelto in
  modo lazy in base a ciò che è installato (FlashAttention solo Linux+CUDA). Su
  Windows/CPU si ricade automaticamente su eager/SDPA. **Questo è il caso facile.**

### 6.2 Long-context — **parzialmente plug-and-play**

Estendere il contesto oltre la finestra di addestramento (es. da 4k/8k a 32k+) è un
problema **di posizione**, non di kernel:

- **Plug-and-play (entro limiti)**: tecniche di **RoPE scaling** (NTK-aware, YaRN,
  linear/dynamic) permettono di estendere il contesto **senza retraining**, ma con
  **degrado crescente** man mano che ci si allontana dalla lunghezza nativa. Si
  configurano via `rope_scaling` nel config del modello e `max_model_len` lato
  serving (`configs/serving/vllm.yaml`).
- **Richiede lavoro**: per un long-context **di qualità** serve **continued
  pre-training / fine-tuning a contesto lungo** (dati lunghi, sequenze lunghe, costo
  VRAM elevato). Questo è materiale **V2** ed è collegato al CPT
  ([training-plan.md](./training-plan.md)). **Non** è gratis.

### 6.3 Dynamic sparse attention — **R&D, NON plug-and-play**

Per "dynamic sparse attention" si intende cambiare **quali** coppie query/key
vengono effettivamente calcolate (sparsità appresa o dipendente dall'input: routing,
landmark, block-sparse dinamico, ecc.). Qui bisogna essere onesti e separare di nuovo
**tre livelli**, perché hanno costi radicalmente diversi:

| Livello | Cosa cambia | Costo | Plug-and-play? |
|---------|-------------|-------|:--------------:|
| **A. Serving-opt** (KV-cache sparsa a inferenza: eviction, sliding-window, attention sink, quantizzazione KV) | Solo come si gestisce la KV-cache a runtime. Pesi invariati. | Basso/medio. Possibile **degrado** su dipendenze a lungo raggio. | **Quasi**: lato serving, niente retraining. |
| **B. Long-context approx** (pattern sparsi fissi/strutturati per allungare il contesto) | Pattern di attenzione, ma stesso modello | Medio. Spesso serve un **breve fine-tuning** per recuperare qualità. | **No** del tutto. |
| **C. Sparsità appresa/dinamica** (il modello impara la sparsità: cambia l'architettura) | **Architettura** e quindi i pesi | **Alto: retraining pesante** (CPT/from-scratch). | **No.** |

Posizione del progetto:

- Il livello **A** è un'**ottimizzazione di serving** sperimentabile **senza toccare
  il training**: rientra in `serving/` e si valuta con le metriche di **latenza e
  VRAM** descritte in [evaluation-plan.md](./evaluation-plan.md). È la prima cosa da
  provare se servono throughput/contesto in più a costo basso.
- I livelli **B** e **C** sono **R&D a tutti gli effetti (traccia V2)** e
  **retraining-heavy**: vanno trattati come esperimenti dedicati, non come switch di
  config. Il repo li tiene **fuori dal percorso critico V1** e li documenta come
  backlog. Pretendere che siano "un flag" sarebbe disonesto: comportano costo di
  addestramento, rischio di regressione qualità e validazione metrica estesa.

**In sintesi**: cambiare *kernel* (6.1) è gratis e già astratto; allungare il
*contesto* (6.2) costa da poco (RoPE scaling) a molto (CPT lungo); rendere
l'attenzione *sparsa-dinamica* (6.3) è ricerca con retraining. Tenerli separati
evita la trappola del "basta cambiare l'attenzione".

---

## 7. Quantizzazione, LoRA e memoria

Il caricamento del modello in `training/common.py` rispetta `model.quantization`:

- **QLoRA 4-bit** (`load_in_4bit: true`, `bnb_4bit_quant_type: nf4`,
  `bnb_4bit_compute_dtype: bfloat16`) tramite `get_bnb_config(cfg)`.
- **LoRA** applicata con `apply_lora(model, cfg)` sui `target_modules`
  (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`), rango `r=16`,
  `alpha=32`, `dropout=0.05` (default in `configs/train/*.yaml`).
- `count_trainable_params(model)` riporta `(trainable, total)` per il logging.

Conseguenza pratica: si addestrano **solo gli adapter** (decine/centinaia di MB),
non i 7B di pesi. Razionale degli iperparametri in
[training-plan.md](./training-plan.md#iperparametri-razionale).

---

## 8. Serving

Lo strato di serving ha due percorsi:

1. **vLLM** (preferito, alto throughput): configurato in
   `configs/serving/vllm.yaml` (`serving.model`, `host`, `port`, `max_model_len`,
   `dtype`, `gpu_memory_utilization`, `tensor_parallel_size`). Espone un'API
   OpenAI-compatibile. Solo Linux+CUDA.
2. **Fallback `transformers`** (`serving/inference.py`, classe `Generator`):
   inferenza locale lazy, usabile anche per smoke test senza vLLM/GPU.

Dettagli operativi (avvio, troubleshooting) in
[operations.md](./operations.md#serving).

---

## 9. Mappa file ↔ responsabilità

| Area | File sorgente | Config | Script / target |
|------|---------------|--------|-----------------|
| Config | `src/italian_llm/config.py` | `configs/base.yaml` | — |
| Dati: corpus | `data/cleaning.py`, `data/schema.py` | `configs/data/corpus.yaml` | `scripts/build_corpus.py` · `make corpus` |
| Dati: SFT | `data/schema.py`, `data/prompts.py` | `configs/data/sft.yaml` | `scripts/build_sft_data.py` · `make sft-data` |
| Dati: sintetici | `data/synthetic.py` | `configs/data/sft.yaml` | `scripts/synthesize.py` · `make synth` |
| CPT | `training/cpt.py`, `training/common.py` | `configs/train/cpt_qwen9b.yaml` | `scripts/train_cpt.py` · `make cpt` |
| SFT | `training/sft.py`, `training/common.py` | `configs/train/sft_qwen9b_lora.yaml` | `scripts/train_sft.py` · `make sft` |
| Preferenze | `training/preference.py` | `configs/train/orpo_qwen9b.yaml` | `scripts/train_preference.py` · `make orpo` |
| Distillazione | `distillation/data_distill.py`, `distillation/logits_distill.py` | `configs/distill/student_3b.yaml` | `scripts/run_distill.py` · `make distill` |
| Valutazione | `evaluation/metrics.py`, `evaluation/runner.py` | `configs/eval/eval.yaml` | `scripts/run_eval.py` · `make eval` |
| Serving | `serving/inference.py` | `configs/serving/vllm.yaml` | `scripts/serve.py` · `make serve` |
| Safety | `safety/policy.py` | — | (usata da synth/eval/serving) |

> Nota: alcuni nomi di script seguono il `Makefile`/README come riferimento
> canonico user-facing; sono richiamabili sia con `make <target>` sia direttamente
> con `python scripts/<nome>.py --config <yaml>`. Vedi
> [operations.md](./operations.md#make-targets).

---

## 10. Riproducibilità e seed

`project.seed` (default `42`) è propagato da `utils/seed.py:set_seed`, che semina
`random` (stdlib, sempre) e in modo lazy `numpy`/`torch` se presenti. La valutazione
gira **greedy** (`eval.temperature: 0.0`) per report deterministici
([evaluation-plan.md](./evaluation-plan.md)). I config usati in un run vanno
serializzati con `dump_config` accanto ai checkpoint per tracciabilità.

---

## 11. Cosa è pronto vs simulato (sintesi)

| Pronto ora (no GPU) | Simulato ma cablato (serve GPU/pesi/dati) |
|---------------------|-------------------------------------------|
| Moduli puri, config merge, data prep, mock teacher, validazione JSONL, metriche testuali, safety policy, test/lint | Pesi Qwen 7B/3B, QLoRA/bnb, CPT/SFT/ORPO/DPO, logits distillation, teacher reale, serving vLLM |

Dettaglio e checklist in [operations.md](./operations.md) e nel README di progetto.
