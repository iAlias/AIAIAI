# Piano di addestramento

> Pipeline in **7 step** (CPT → SFT → ORPO → distillazione), iperparametri con
> **razionale**, e configurazioni **consapevoli dell'hardware**. Ogni step è mappato
> su uno script `scripts/*.py`, un target `make` e un file `configs/*.yaml`.

Documenti collegati:
[architecture.md](./architecture.md) ·
[dataset-plan.md](./dataset-plan.md) ·
[distillation-plan.md](./distillation-plan.md) ·
[evaluation-plan.md](./evaluation-plan.md) ·
[operations.md](./operations.md) ·
[decisions.md](./decisions.md)

---

## 1. La pipeline in 7 step

```
[1] CORPUS  ─▶ [2] SFT-DATA ─▶ [3] SYNTH ─▶ [4] CPT ─▶ [5] SFT ─▶ [6] ORPO ─▶ [7] EVAL
   pulizia/      formato chat    teacher      adatt.    istruz.    preferenze   metriche
   dedup         + qualità       mock/reale   lingua    QLoRA      ORPO/DPO     prodotto
```

| # | Step | Target | Script | Config | Funzione |
|---|------|--------|--------|--------|----------|
| 1 | Corpus (CPT) | `make corpus` | `scripts/build_corpus.py` | `configs/data/corpus.yaml` | pulizia/dedup → `data/processed/corpus.jsonl` |
| 2 | SFT-data | `make sft-data` | `scripts/build_sft_data.py` | `configs/data/sft.yaml` | dataset chat → `sft_train.jsonl` / `sft_valid.jsonl` |
| 3 | Synth | `make synth` | `scripts/synthesize.py` | `configs/data/sft.yaml` | risposte sintetiche dai teacher |
| 4 | CPT | `make cpt` | `scripts/train_cpt.py` | `configs/train/cpt_qwen9b.yaml` | `run_cpt(cfg) -> str` |
| 5 | SFT | `make sft` | `scripts/train_sft.py` | `configs/train/sft_qwen9b_lora.yaml` | `run_sft(cfg) -> str` |
| 6 | ORPO/Pref | `make orpo` | `scripts/train_preference.py` | `configs/train/orpo_qwen9b.yaml` | `run_preference(cfg) -> str` |
| 7 | Eval | `make eval` | `scripts/run_eval.py` | `configs/eval/eval.yaml` | `run_eval(cfg) -> dict` |

Step di supporto: **distillazione** (`make distill`, `scripts/run_distill.py`,
`configs/distill/student_3b.yaml`), trattata in
[distillation-plan.md](./distillation-plan.md).

Gli step 1–3 (preparazione dati) sono coperti in
[dataset-plan.md](./dataset-plan.md); qui ci concentriamo su 4–7 e sulla
distillazione.

> **Ordine per traccia.** In **V1** si salta il CPT: `[2]→[3]→[5]→[6]→[7]`. In **V2**
> si esegue anche `[1]→[4]` prima dell'SFT. Vedi
> [architecture.md](./architecture.md#le-due-tracce-v1-pragmatica-e-v2-rd).

---

## 2. Step 4 — Continued Pre-Training (CPT)

**Scopo**: adattare il base alla **lingua e ai domini** italiani prima
dell'instruction tuning. È un passo **opzionale (V2)**: ha senso quando si dispone di
un corpus italiano ampio e di GPU serie. Il base Qwen è già fortemente multilingue,
quindi in **V1 il CPT si salta** senza grave perdita.

**Forma**: CPT **in stile QLoRA** (si addestrano adapter LoRA, non l'intero modello)
per restare in VRAM su singola GPU. Dati a testo grezzo (`format: text`, campo
`text`), language modeling causale.

Config di riferimento `configs/train/cpt_qwen9b.yaml`:

| Chiave | Valore | Razionale |
|--------|--------|-----------|
| `train.epochs` | `1` | il CPT vuole **tante token, poche epoche**: ripassare troppo i dati causa overfitting/memorizzazione. |
| `train.max_steps` | `500` | tetto basso per **avvio rapido**; in produzione si alza in funzione delle token disponibili. |
| `train.lr` | `2e-5` | LR **prudente**: il CPT deve spostare poco i pesi per non degradare le capacità generali. |
| `train.batch_size` / `grad_accum` | `1` / `16` | batch effettivo ~16 stando stretti in memoria. |
| `train.max_seq_len` | `2048` | contesto contenuto per ridurre VRAM; per long-context vedi [architecture.md](./architecture.md#62-long-context--parzialmente-plug-and-play). |
| `train.weight_decay` | `0.0` | CPT leggero, nessun decay. |
| `train.gradient_checkpointing` | `true` | scambia ~20-30% di velocità per memoria: abilita batch/seq maggiori. |
| `lora.r/alpha/dropout` | `16/32/0.05` | rango moderato; `alpha/r = 2` è uno scaling standard ed efficace. |

**Output**: adapter in `outputs/cpt_qwen9b/` (`run_cpt` restituisce la dir). Questo
adapter (o il merge) diventa il punto di partenza dell'SFT.

> **Long-context.** Il CPT è anche il luogo dove si fa il **vero** allungamento di
> contesto (sequenze lunghe + dati lunghi), non un semplice flag. Costo VRAM elevato:
> materiale V2. RoPE scaling resta un'alternativa "economica" ma con degrado (vedi
> architecture §6.2).

---

## 3. Step 5 — Supervised Fine-Tuning (SFT)

**Scopo**: insegnare al modello a **seguire istruzioni in formato chat**, con lo
stile target (diretto, utile, poco verboso). È il cuore della **V1**.

**Forma**: **QLoRA 4-bit** (NF4 + compute bf16) + LoRA sugli stessi `target_modules`.
Dati in `format: chat` (campo `messages: [{role, content}, ...]`), con la maschera
di loss tipicamente sulle sole risposte dell'assistant. Il prompt viene reso con il
chat template del tokenizer (`tokenizer_utils.format_chat`), con fallback
`data/prompts.render_plain` (stile `<|system|>/<|user|>/<|assistant|>`).

Config `configs/train/sft_qwen9b_lora.yaml`:

| Chiave | Valore | Razionale |
|--------|--------|-----------|
| `train.epochs` | `3` | per instruction tuning su dataset curato, **2-3 epoche** è la zona dolce; oltre si rischia ripetitività/overfit. |
| `train.lr` | `2e-4` | LR **tipico per LoRA/QLoRA**: gli adapter tollerano LR più alti del full-FT (1e-5..2e-5). |
| `train.batch_size` / `grad_accum` | `2` / `8` | batch effettivo ~16: compromesso stabilità/memoria su 24 GB. |
| `train.max_seq_len` | `2048` | copre la quasi totalità degli esempi istruzione; gli outlier lunghi vengono troncati. |
| `train.warmup_ratio` | `0.03` | warmup breve per stabilizzare i primi step (LR alto su adapter freschi). |
| `train.weight_decay` | `0.0` | gli adapter LoRA si regolarizzano col `dropout`, non col decay. |
| `train.gradient_checkpointing` | `true` | permette batch 2 @ 2048 token sul 7B in 4-bit. |
| `lora.r/alpha` | `16/32` | capacità sufficiente per lo stile/istruzioni senza gonfiare l'adapter. |
| `data.format` | `chat` | esempi conversazionali (`messages[]`). |

**Scheduler**: cosine decay con il warmup indicato è il default ragionevole; lo
imposta il loop di training in `training/sft.py`.

**Output**: adapter SFT in `outputs/sft_qwen9b_lora/`. È il modello base per **ORPO**
e per la **distillazione**.

---

## 4. Step 6 — Allineamento alle preferenze (ORPO default, DPO opzionale)

**Scopo**: allineare **stile e preferenze** (tono diretto, brevità, **niente
over-refusal**) usando coppie `(chosen, rejected)`.

**Perché ORPO di default** (dettaglio in [decisions.md](./decisions.md#adr-005)):

- **Single-stage**: ORPO unisce SFT e preferenza in un'unica loss (NLL + termine di
  odds-ratio). **Non richiede un modello di riferimento separato** come il DPO →
  **meno memoria**, niente secondo forward sul ref model. Ideale su singola GPU.
- **DPO opzionale**: disponibile impostando `preference.method: dpo` in
  `configs/train/orpo_qwen9b.yaml`; TRL gestisce il ref model. Si usa quando si parte
  da un SFT molto forte e si vuole separare nettamente la fase di preferenza.

Config `configs/train/orpo_qwen9b.yaml`:

| Chiave | Valore | Razionale |
|--------|--------|-----------|
| `train.epochs` | `1` | l'allineamento preferenza vuole **poche passate**: troppe degradano le capacità (alignment tax). |
| `train.lr` | `5e-6` | LR **basso**: l'allineamento deve rifinire, non riscrivere. |
| `train.max_seq_len` | `1536` | due risposte per coppia → contesto un po' più corto per stare in memoria. |
| `preference.method` | `orpo` | default single-stage (`dpo` opzionale). |
| `preference.beta` | `0.1` | forza del termine di preferenza (ORPO λ / DPO β): `0.1` è equilibrato. |
| `preference.max_prompt_len` | `768` | tetto sui token del prompt nelle coppie. |
| `preference.max_len` | `1536` | tetto totale prompt+risposta. |

**Punto di partenza**: si parte dall'**adapter SFT** (continua sugli stessi
`target_modules`). Dati `format: preference`, schema
`{"id","domain","prompt","chosen","rejected","meta":{...}}`.

**Ruolo anti over-refusal.** Una fetta del dataset di preferenze è costruita
apposta per **ridurre l'over-refusal**: `chosen` = risposta utile e diretta a una
richiesta **lecita**, `rejected` = rifiuto/moralizzazione non necessari. Il design di
questo sottoinsieme è in
[dataset-plan.md](./dataset-plan.md#riduzione-dell-over-refusal) e la policy in
`src/italian_llm/safety/policy.py`. Si misura con `over_refusal_rate`
([evaluation-plan.md](./evaluation-plan.md)).

**Output**: adapter allineato in `outputs/orpo_qwen9b/`.

---

## 5. Step 7 — Valutazione

Dopo l'ORPO si esegue `make eval` (`scripts/run_eval.py` → `run_eval(cfg)`), che
calcola le metriche in `evaluation/metrics.py` sul set `data/eval/eval_it.jsonl` e
scrive `outputs/eval/report.json` + `outputs/eval/preds.jsonl`. Metriche, set di
valutazione e interpretazione sono in [evaluation-plan.md](./evaluation-plan.md). La
valutazione gira **greedy** (`temperature: 0.0`) per report riproducibili.

Si confronta sempre **prima/dopo** ogni fase (base → SFT → ORPO) sulle stesse
metriche: in particolare ci si aspetta che ORPO **riduca verbosità e over-refusal**
mantenendo o migliorando aderenza e italianità.

---

## 6. Distillazione (step di supporto)

Dopo aver ottenuto un teacher allineato (7B + adapter ORPO) si distilla verso uno
**student `Qwen/Qwen2.5-3B`** per il deploy economico. **Data distillation** di
default (robusta), **logits distillation** sperimentale. Config
`configs/distill/student_3b.yaml`, dettaglio completo in
[distillation-plan.md](./distillation-plan.md).

| Chiave (distill) | Valore | Note |
|------------------|--------|------|
| `distill.mode` | `data` | `data` (default) \| `logits` (sperimentale) |
| `distill.temperature` | `2.0` | softening dei logit (modalità `logits`) |
| `distill.alpha` | `0.5` | peso KL vs cross-entropy (modalità `logits`) |
| `train.epochs` | `2` | lo student impara dalle risposte del teacher |
| `train.lr` | `2e-4` | LoRA LR, come l'SFT |

---

## 7. Iperparametri: razionale d'insieme {#iperparametri-razionale}

Tabella di sintesi per leggere a colpo d'occhio le scelte e il **perché**:

| Fase | LR | Epoche | Seq | Batch eff. | Logica dominante |
|------|----|--------|-----|-----------|------------------|
| **CPT** | `2e-5` | 1 (max 500 step) | 2048 | ~16 | tante token, poche epoche, spostare poco i pesi |
| **SFT** | `2e-4` | 3 | 2048 | ~16 | LR alto da adapter; 2-3 epoche = zona dolce |
| **ORPO** | `5e-6` | 1 | 1536 | ~16 | rifinire stile/preferenze, evitare alignment tax |
| **Distill** | `2e-4` | 2 | 2048 | ~16 | student impara stile del teacher |

Principi trasversali:

1. **LR cala lungo la pipeline** (`2e-4` SFT → `5e-6` ORPO): ogni fase successiva
   modifica **meno** i pesi, per non distruggere quanto appreso prima.
2. **Batch effettivo costante ~16** via `grad_accum`, così il comportamento è
   confrontabile tra profili hardware diversi (cambia il micro-batch, non l'effettivo).
3. **`alpha/r = 2`** stabile per LoRA su tutte le fasi.
4. **`gradient_checkpointing` sempre on** nei default conservativi: si spegne solo se
   si ha VRAM abbondante e si vuole velocità (profilo *serious*).
5. **`weight_decay = 0`**: con LoRA la regolarizzazione la fa il `dropout`.

---

## 8. Configurazioni consapevoli dell'hardware {#hardware-aware}

I default in `configs/` puntano al profilo **recommended** (1× 24 GB). Per gli altri
profili si **sovrascrivono poche chiavi** (via override CLI o copia del YAML). Vedi
la tabella profili in [operations.md](./operations.md#profili-hardware).

### Profilo Minimum (no GPU / 8–12 GB) — smoke run

Obiettivo: **validare la pipeline**, non addestrare il 7B. Gli script di training, in
assenza di GPU/pesi, eseguono uno **smoke run** coerente su dati sintetici/modelli
minuscoli. Suggerimenti se si vuole forzare un giro reale su modello tiny:

```yaml
# override d'esempio (modello tiny + tutto al minimo)
model:
  name: sshleifer/tiny-gpt2     # o altro modello minuscolo locale
  quantization: { load_in_4bit: false }
train:
  max_steps: 20
  batch_size: 1
  grad_accum: 1
  max_seq_len: 256
  gradient_checkpointing: false
```

### Profilo Recommended (1× 24 GB) — V1 completa

I **default così come sono**. QLoRA 4-bit, `gradient_checkpointing: true`,
`batch_size: 2`, `grad_accum: 8`, `max_seq_len: 2048`. Sufficiente per SFT + ORPO del
7B e distillazione data-based sullo student 3B.

### Profilo Serious (40–80 GB, multi-GPU) — V2 completa

Si può **alzare la qualità/velocità**:

```yaml
model:
  dtype: bfloat16
  quantization: { load_in_4bit: false }   # bf16 full-weight se la VRAM basta
train:
  batch_size: 8
  grad_accum: 2
  max_seq_len: 4096
  gradient_checkpointing: false           # più veloce con VRAM abbondante
```

- **Multi-GPU**: `accelerate` / **DeepSpeed ZeRO** (incluso in `requirements.txt`)
  per shardare ottimizzatore/gradiente/pesi. `tensor_parallel_size` lato serving in
  `configs/serving/vllm.yaml`.
- **CPT su corpus ampio** e **logits distillation** diventano praticabili.
- **Sequenze lunghe**: qui si fa il long-context "vero" (vedi §2 e architecture §6.2).

---

## 9. Come lanciare (riepilogo comandi)

```bash
# Preparazione dati (no GPU, mock teacher)
make corpus            # [1]  -> data/processed/corpus.jsonl
make sft-data          # [2]  -> data/processed/sft_{train,valid}.jsonl
make synth             # [3]  -> risposte sintetiche

# Training
make cpt               # [4]  (V2)  -> outputs/cpt_qwen9b/
make sft               # [5]        -> outputs/sft_qwen9b_lora/
make orpo              # [6]        -> outputs/orpo_qwen9b/
make distill           #            -> outputs/distill_student_3b/

# Valutazione
make eval              # [7]  -> outputs/eval/report.json

# Invocazione esplicita equivalente (config reale su disco)
python scripts/train_sft.py --config configs/train/sft_qwen9b_lora.yaml
python scripts/train_preference.py --config configs/train/orpo_qwen9b.yaml
```

Ogni run dovrebbe **serializzare il config effettivo** (`dump_config`) accanto al
checkpoint per riproducibilità, e registrare `count_trainable_params` nel log. Setup
ambiente, env var e troubleshooting in [operations.md](./operations.md).
