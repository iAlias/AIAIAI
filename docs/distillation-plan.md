# Piano di distillazione

> Distillare il teacher allineato (~7B) verso uno **student `Qwen/Qwen2.5-3B`** per
> il deploy economico. **Data distillation** di default (robusta), **logits
> distillation** sperimentale con **limiti documentati**.

Documenti collegati:
[architecture.md](./architecture.md) ·
[training-plan.md](./training-plan.md) ·
[dataset-plan.md](./dataset-plan.md) ·
[evaluation-plan.md](./evaluation-plan.md) ·
[operations.md](./operations.md) ·
[decisions.md](./decisions.md)

Codice: `src/italian_llm/distillation/data_distill.py`
(`build_distill_dataset(cfg) -> str`),
`src/italian_llm/distillation/logits_distill.py`
(`run_logits_distill(cfg) -> str`). Config: `configs/distill/student_3b.yaml`,
`configs/model/student_3b.yaml`. Script/target: `scripts/run_distill.py` /
`make distill`.

---

## 1. Perché distillare e verso cosa

Il teacher (7B base + adapter **ORPO** allineato) è ottimo ma **costoso da servire**.
Per il deploy a basso costo/latenza si addestra uno **student più piccolo** che ne
imiti il comportamento sui domini di prodotto.

**Student = `Qwen/Qwen2.5-3B`** (`configs/model/student_3b.yaml`). Scelta motivata:

- **Stessa famiglia/tokenizer del teacher** (Qwen2.5): requisito **obbligatorio**
  per la logits distillation (vocabolario allineato, vedi §4) e comunque comodo per
  la data distillation.
- **Servibile su poca VRAM** (anche < 12 GB in 4-bit), ottimo per il profilo
  *recommended* e per edge/CPU-limited.
- **Gap di capacità gestibile**: 7B → 3B è un salto sensato; scendere troppo (es.
  < 1B) degrada oltre l'utile sui task generativi italiani.

Razionale della scelta "data distillation di default" in
[decisions.md](./decisions.md#adr-006).

---

## 2. Le due modalità a confronto

`distill.mode` in `configs/distill/student_3b.yaml`: `data` (default) | `logits`.

| Aspetto | **Data distillation** (`mode: data`) | **Logits distillation** (`mode: logits`) |
|---------|--------------------------------------|------------------------------------------|
| Segnale | **risposte (testo)** del teacher | **distribuzione sui logit** del teacher per token |
| Cosa serve dal teacher | solo l'**output testuale** (anche via API) | accesso ai **logit** (modello locale in memoria) |
| Vincolo vocabolario | nessuno (teacher e student possono differire) | **tokenizer/vocabolario identici** teacher↔student |
| Costo | basso/medio (è un SFT su dati del teacher) | **alto**: doppio forward (teacher+student), molta VRAM |
| Robustezza | **alta**, pochi modi di sbagliare | **fragile**: allineamento token, shift, memoria |
| Qualità potenziale | molto buona | potenzialmente superiore (segnale più ricco) **se** ben fatto |
| Stato nel progetto | **default, raccomandato** | **sperimentale**, limiti noti (§5) |

In una riga: **data distillation** è "fai un SFT sulle risposte del teacher";
**logits distillation** è "fai imparare allo student la stessa *distribuzione* del
teacher token per token". La seconda promette di più ma è molto più delicata.

---

## 3. Data distillation (default)

Flusso in due fasi:

### 3.1 Costruzione del dataset distillato

`build_distill_dataset(cfg) -> str` (`data_distill.py`):

1. Legge i **prompt sorgente** da `data.prompts_path` (default
   `data/processed/sft_train.jsonl`): si riusano i prompt SFT come istruzioni.
2. Per ogni prompt, ottiene la risposta dal **teacher** configurato in `teacher`:
   `teacher.name` (es. `Qwen/Qwen2.5-7B`), `teacher.adapter`
   (`outputs/orpo_qwen9b`), `teacher.provider` (`mock | openai_compat | hf_local`),
   con `distill.max_new_tokens` (default `512`). In assenza di teacher reale →
   **fallback al mock**, così il passo è sempre eseguibile.
3. Applica gli **stessi filtri qualità/lingua** della pipeline dati
   ([dataset-plan.md](./dataset-plan.md)) e scrive il dataset distillato in
   `data.distill_path` (`data/processed/distill_student.jsonl`), in **formato chat**
   (`format: chat`), taggando `teacher_name`.

### 3.2 Fine-tuning dello student

Lo student viene addestrato in **QLoRA** sul dataset distillato, esattamente come un
SFT (vedi [training-plan.md](./training-plan.md#step-5--supervised-fine-tuning-sft)).
Parametri da `configs/distill/student_3b.yaml`:

| Chiave | Valore | Note |
|--------|--------|------|
| `student.name` | `Qwen/Qwen2.5-3B` | + quantizzazione 4-bit NF4 |
| `train.epochs` | `2` | lo student impara dalle risposte del teacher |
| `train.lr` | `2e-4` | LoRA LR (come SFT) |
| `train.batch_size`/`grad_accum` | `2`/`8` | batch eff. ~16 |
| `train.max_seq_len` | `2048` | come SFT |
| `lora.r/alpha/dropout` | `16/32/0.05` | identico al resto della pipeline |
| `data.valid_path` | `sft_valid.jsonl` | validazione condivisa |

**Output**: adapter dello student in `outputs/distill_student_3b/`.

**Vantaggio chiave**: il teacher può essere **remoto** (endpoint OpenAI-compatibile)
e si possono **arricchire** le risposte (best-of-n con `majority_rank`, filtri
qualità) prima di addestrare. È la via robusta verso uno student di buona qualità.

---

## 4. Logits distillation (sperimentale)

`run_logits_distill(cfg) -> str` (`logits_distill.py`). Idea: minimizzare la
**divergenza KL** tra la distribuzione del teacher e quella dello student sui token,
con **softening** via temperatura, combinata alla cross-entropy sui target.

Loss (concettuale):

```
L = alpha * KL( softmax(teacher_logits / T) || softmax(student_logits / T) ) * T^2
  + (1 - alpha) * CrossEntropy(student_logits, target)
```

Parametri da `configs/distill/student_3b.yaml`:

| Chiave | Default | Significato |
|--------|---------|-------------|
| `distill.mode` | `logits` | attiva questa modalità |
| `distill.temperature` (`T`) | `2.0` | ammorbidisce le distribuzioni: trasferisce le "dark knowledge" del teacher |
| `distill.alpha` | `0.5` | peso del termine KL vs cross-entropy |

Requisiti operativi:

- **Teacher e student in memoria insieme** (doppio forward): VRAM elevata →
  **profilo serious** (vedi [operations.md](./operations.md#profili-hardware)).
- **Tokenizer/vocabolario identici**: garantito dalla scelta stessa famiglia Qwen2.5
  (7B teacher, 3B student). Con tokenizer diversi i logit **non sono allineabili**
  token-per-token senza mappature lossy.

---

## 5. Limiti documentati della logits distillation {#limiti}

Onestà tecnica: ecco **perché** è marcata sperimentale e quando *non* usarla.

1. **Vincolo di vocabolario rigido.** La KL token-per-token richiede **lo stesso
   spazio di vocabolario** tra teacher e student. Se differiscono (anche solo per
   token speciali aggiunti durante il fine-tuning), l'allineamento salta. Mitigazione
   nel progetto: stessa famiglia Qwen2.5 → vocabolario condiviso.

2. **Costo memoria/compute.** Tenere teacher (7B) e student (3B) **entrambi
   residenti** e fare due forward per batch è molto più oneroso della data
   distillation. Su singola GPU 24 GB è al limite o impraticabile a sequenze piene.

3. **Allineamento di sequenza.** I logit vanno confrontati **sulle stesse posizioni**
   con la **stessa tokenizzazione del testo**. Padding, troncamenti e teacher forcing
   devono coincidere; un disallineamento corrompe silenziosamente il segnale.

4. **Storage/streaming dei logit.** Precomputare e salvare i logit del teacher
   (alternativa al doppio forward) esplode su disco (`vocab_size ≈ 150k` floats per
   token). Si mitiga salvando solo **top-k logit** + softmax sparsa, al prezzo di
   approssimare la distribuzione.

5. **Instabilità.** La scelta di `temperature` e `alpha` è sensibile; valori errati
   producono collasso (lo student ignora la KL) o rumore (segue logit poco
   informativi). Richiede tuning e validazione attenta.

6. **Guadagno non garantito.** In pratica, su task generativi italiani, una **data
   distillation ben fatta** (best-of-n, filtri qualità, abbastanza dati) raggiunge
   spesso una qualità **comparabile** con molto meno rischio. Il vantaggio della
   logits distillation emerge soprattutto con teacher/student della stessa famiglia,
   tanto compute e validazione rigorosa.

**Raccomandazione operativa**: usare **data distillation per default (V1)**. Trattare
la **logits distillation come esperimento V2** su hardware serio, confrontandola con
la baseline data-based sulle **stesse metriche** ([evaluation-plan.md](./evaluation-plan.md))
prima di adottarla.

---

## 6. Valutazione dello student

Lo student si valuta **con lo stesso runner del teacher**
(`scripts/run_eval.py` → `run_eval`), puntando `eval.model_path` allo student
(`Qwen/Qwen2.5-3B`) e `eval.adapter` a `outputs/distill_student_3b`. Metriche chiave
per la distillazione:

- **Qualità mantenuta**: `instruction_adherence`, `italianity_score`, `rouge_l`
  vicini al teacher.
- **Comportamento di stile**: `verbosity_ratio`, `refusal_rate`,
  **`over_refusal_rate`** (lo student deve **ereditare** il comportamento anti
  over-refusal del teacher).
- **Efficienza**: **latenza** e **VRAM** (motivo stesso della distillazione) —
  misurate dal runner, vedi [evaluation-plan.md](./evaluation-plan.md#latenza-e-vram).

Il confronto si fa **teacher vs student** sulle stesse predizioni: si accetta un
piccolo calo di qualità in cambio di un grande guadagno di efficienza.

---

## 7. Comandi

```bash
# Distillazione (default: data; teacher mock se non configurato)
make distill
python scripts/run_distill.py --config configs/distill/student_3b.yaml

# Forzare la modalità logits (sperimentale, hardware serio):
#   imposta distill.mode: logits in configs/distill/student_3b.yaml
#   oppure override CLI se lo script lo supporta

# Valutare lo student distillato
python scripts/run_eval.py --config configs/eval/eval.yaml \
  # (in eval.yaml: model_path=Qwen/Qwen2.5-3B, adapter=outputs/distill_student_3b)
```

Con teacher reale, impostare in `.env`: `TEACHER_PROVIDER`, `TEACHER_BASE_URL`,
`TEACHER_API_KEY`, `TEACHER_MODEL` (vedi [operations.md](./operations.md#variabili-d-ambiente)).
In assenza, la modalità `data` ricade sul **mock** e resta eseguibile offline.
