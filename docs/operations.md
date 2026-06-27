# Operazioni: setup, esecuzione, hardware, serving, troubleshooting

> Guida operativa: **ambiente**, variabili d'ambiente, **target `make`**, **profili
> hardware**, **serving** (vLLM + fallback transformers) e **troubleshooting**.

Documenti collegati:
[architecture.md](./architecture.md) ·
[training-plan.md](./training-plan.md) ·
[dataset-plan.md](./dataset-plan.md) ·
[distillation-plan.md](./distillation-plan.md) ·
[evaluation-plan.md](./evaluation-plan.md) ·
[decisions.md](./decisions.md)

---

## 1. Requisiti

- **Python 3.11** (richiesto da `pyproject.toml`, `requires-python >= 3.11`).
- **Target**: Linux + CUDA per il training/serving reale. **Windows/CPU**: il repo è
  **staticamente valido e importabile**; gli step leggeri girano, quelli pesanti
  fanno smoke run o fallback.
- Dipendenze in `requirements.txt` (core leggero + blocchi pesanti opzionali) e nei
  gruppi `optional-dependencies` di `pyproject.toml` (`training`, `serving`, `eval`,
  `dev`).

---

## 2. Setup dell'ambiente

```bash
# 1) Ambiente virtuale (consigliato conda o venv)
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
#   .venv\Scripts\Activate.ps1                       # Windows PowerShell

# 2) Variabili d'ambiente
cp .env.example .env        # poi personalizza (vedi §3)

# 3) Installazione
make setup                  # pip install -U pip + -r requirements.txt + pip install -e .
```

`make setup` esegue: aggiornamento pip, installazione `requirements.txt`, e
installazione del package in **editable** (`pip install -e .`, src layout).

### Installazione "a la carte" (senza tutto il pesante)

Per il solo uso **statico** (moduli puri, data prep, mock teacher, test) basta:

```bash
pip install -e .            # installa solo PyYAML (dipendenza core)
pip install pytest ruff black   # per test/lint
```

I moduli puri (`config`, `logging_utils`, `data/prompts`, `data/schema`,
`data/cleaning`, `safety/policy`) importano **senza torch**. Per il training reale:

```bash
# Linux + CUDA: usa l'indice ufficiale PyTorch per la build cu12x
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> `bitsandbytes`, `deepspeed`, `vllm` sono **Linux/CUDA**. Su Windows non installarli:
> il codice li importa lazy e ricade sui fallback.

### Nota Windows

I target `make` richiedono `make` + utility Unix (`grep`, `awk`, usate da
`make help`). Su Windows usa **WSL**, **Git Bash**, oppure invoca direttamente gli
script:

```powershell
python scripts/train_sft.py --config configs/train/sft_qwen9b_lora.yaml
```

---

## 3. Variabili d'ambiente {#variabili-d-ambiente}

Si copiano da `.env.example` in `.env`. Categorie principali:

| Variabile | Scopo | Default/Nota |
|-----------|-------|--------------|
| `TEACHER_PROVIDER` | seleziona il teacher per la sintesi | `mock` (offline, deterministico) |
| `TEACHER_BASE_URL` | endpoint OpenAI-compatibile | richiesto per `openai_compat` |
| `TEACHER_API_KEY` | chiave dell'endpoint | richiesto per `openai_compat` |
| `TEACHER_MODEL` | nome modello sull'endpoint | richiesto per `openai_compat` |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | accesso a modelli/dataset gated su HF | per scaricare i pesi Qwen/dataset |
| `HF_HOME` | cache HuggingFace | sposta la cache su disco capiente |
| `CUDA_VISIBLE_DEVICES` | selezione GPU | es. `0` o `0,1` |
| `WANDB_API_KEY` / `WANDB_MODE` | logging esperimenti (opzionale) | `WANDB_MODE=offline` per disabilitare |

**Regola d'oro**: se le `TEACHER_*` non sono impostate, `OpenAICompatTeacher` ricade
**automaticamente sul mock** (vedi [dataset-plan.md](./dataset-plan.md#6-bootstrap-sintetico-teacher)).
Nessun segreto va committato: `.env` è ignorato da git.

```bash
# Esempi
TEACHER_PROVIDER=mock make synth                 # offline
TEACHER_PROVIDER=openai_compat make synth        # con endpoint reale (env TEACHER_*)
CUDA_VISIBLE_DEVICES=0 python scripts/train_sft.py --config configs/train/sft_qwen9b_lora.yaml
```

---

## 4. Target `make` {#make-targets}

Tutti i target sono in `Makefile` (`make help` li elenca auto-generati). Ognuno
invoca uno script con la config di default; la config è sovrascrivibile via variabili
`CFG_*` o invocando lo script a mano.

| Target | Script | Config default | Cosa fa | GPU? |
|--------|--------|----------------|---------|:----:|
| `make setup` | — | — | installa deps + package editable | no |
| `make corpus` | `scripts/build_corpus.py` | `configs/base.yaml` | [1] corpus CPT (pulizia+dedup) | no |
| `make sft-data` | `scripts/build_sft_data.py` | `configs/base.yaml` | [2] dataset SFT chat JSONL | no |
| `make synth` | `scripts/synthesize.py` | `configs/base.yaml` | [3] sintesi (mock di default) | no |
| `make cpt` | `scripts/train_cpt.py` | `configs/train/cpt_qwen9b.yaml` (`CFG_CPT`) | [4] Continued Pre-Training | sì* |
| `make sft` | `scripts/train_sft.py` | `configs/train/sft_qwen9b_lora.yaml` (`CFG_SFT`) | [5] SFT QLoRA | sì* |
| `make orpo` | `scripts/train_preference.py` | `configs/train/orpo_qwen9b.yaml` (`CFG_ORPO`) | [6] preferenze ORPO/DPO | sì* |
| `make distill` | `scripts/run_distill.py` | `configs/distill/student_3b.yaml` (`CFG_DISTILL`) | distillazione student 3B | sì* |
| `make eval` | `scripts/run_eval.py` | `configs/eval/eval.yaml` (`CFG_EVAL`) | [7] valutazione | sì* |
| `make infer` | `scripts/infer.py` | `configs/eval/eval.yaml` | inferenza CLI | sì* |
| `make serve` | `scripts/serve.py` | `configs/serving/vllm.yaml` (`CFG_SERVE`) | serving (vLLM/transformers) | sì* |
| `make test` | — | — | `pytest` (moduli puri senza torch) | no |
| `make lint` | — | — | `ruff check` + `black --check` | no |
| `make format` | — | — | `black` + `ruff --fix` | no |
| `make clean` | — | — | rimuove cache/build (non tocca `data/`/`outputs/`) | no |

\* "GPU sì" = richiede GPU/pesi per il run **reale**; senza, esegue **smoke run**
coerente. Override eseguibile Python: `make PY=python3 <target>`.

> **Nota sui default `CFG_*`.** Il `Makefile` definisce variabili `CFG_*` (es.
> `CFG_SFT`). I file di config reali su disco sono quelli versionati in `configs/`
> (es. `configs/train/sft_qwen9b_lora.yaml`): per essere sempre espliciti e
> riproducibili, invoca gli script con `--config <percorso reale>` o passa
> `make sft CFG_SFT=configs/train/sft_qwen9b_lora.yaml`.

---

## 5. Profili hardware {#profili-hardware}

Tre profili di riferimento. I tempi sono **ordini di grandezza** per un giro della
traccia indicata sul base ~7B (QLoRA salvo nota); variano con dataset, sequenza ed
epoche. Configurazioni per profilo in
[training-plan.md](./training-plan.md#hardware-aware).

| Profilo | GPU / VRAM | RAM | Disco | Tempo (giro) | Cosa puoi fare |
|---------|-----------|-----|-------|--------------|----------------|
| **Minimum** | nessuna / 8–12 GB | 16 GB | ~30 GB | minuti | Validare il repo, data prep, mock teacher, test, smoke run su modelli tiny. **Nessun training reale del 7B.** |
| **Recommended** | 1× 24 GB (RTX 3090/4090, A5000) | 32–64 GB | ~150 GB | ore | **V1 completa**: SFT QLoRA 4-bit del 7B, ORPO, data distillation su student 3B, valutazione, serving locale. |
| **Serious** | 1–8× 40–80 GB (A100/H100) | 128 GB+ | 1 TB+ | ore→giorni | **V2 completa**: CPT su corpus ampio, full/bf16 FT, multi-teacher, **logits distillation**, ablation, DeepSpeed ZeRO, serving ad alto throughput. |

Note pratiche:

- **Minimum** sviluppa/valida la pipeline; i training fanno smoke run su dati
  sintetici/modelli minuscoli. Vedi override "tiny" in
  [training-plan.md](./training-plan.md#hardware-aware).
- **Recommended** è il punto dolce **V1**: 24 GB bastano per QLoRA 4-bit del 7B con
  `gradient_checkpointing` + `grad_accum`.
- **Serious** abilita **V2**: CPT, sequenze lunghe, throughput, logits distillation
  ([distillation-plan.md](./distillation-plan.md#limiti)).
- **Disco**: i checkpoint del 7B e gli intermedi crescono in fretta; tieni `outputs/`
  su un volume capiente. Gli **adapter LoRA** invece sono piccoli (decine/centinaia
  di MB).

### Stima VRAM (QLoRA 4-bit, indicativa)

| Modello | Solo inferenza 4-bit | SFT QLoRA (seq 2048, bs 2) |
|---------|----------------------|----------------------------|
| Qwen2.5-3B | ~3–4 GB | ~8–10 GB |
| Qwen2.5-7B | ~6–7 GB | ~16–20 GB |
| Qwen2.5-14B | ~10–12 GB | ~28–36 GB |

Riduci la VRAM con: `gradient_checkpointing: true`, `batch_size` minore + `grad_accum`
maggiore, `max_seq_len` minore. Aumenta velocità (con VRAM abbondante) spegnendo il
checkpointing e alzando il batch.

---

## 6. Serving {#serving}

Due percorsi, vedi [architecture.md](./architecture.md#8-serving).

### vLLM (preferito, Linux+CUDA)

```bash
make serve
python scripts/serve.py --config configs/serving/vllm.yaml
```

Config `configs/serving/vllm.yaml`:

| Chiave | Default | Note |
|--------|---------|------|
| `serving.model` | `Qwen/Qwen2.5-7B` | base o merge SFT/ORPO da servire |
| `serving.adapter` | `outputs/sft_qwen9b_lora` | LoRA opzionale (`""` per nessuno) |
| `serving.host` / `port` | `0.0.0.0` / `8000` | interfaccia/porta |
| `serving.max_model_len` | `4096` | contesto servito (vedi long-context, architecture §6.2) |
| `serving.dtype` | `bfloat16` | dtype runtime |
| `serving.gpu_memory_utilization` | `0.85` | lascia margine VRAM |
| `serving.tensor_parallel_size` | `1` | >1 per multi-GPU |
| `serving.max_num_seqs` | `16` | richieste concorrenti |
| `serving.served_model_name` | `italian-llm` | nome esposto dall'API |

vLLM espone un'**API OpenAI-compatibile** su `host:port`: si può puntare lì un client
OpenAI (e perfino usarlo come **teacher** `openai_compat` per altre fasi).

### Fallback transformers (senza vLLM/GPU)

`src/italian_llm/serving/inference.py`, classe **`Generator`**:

```python
from italian_llm.serving.inference import Generator
gen = Generator(model_path="Qwen/Qwen2.5-7B", adapter="outputs/orpo_qwen9b")
print(gen.generate("Spiega in due righe cos'è una API REST."))
# accetta anche una lista di messaggi [{role, content}, ...]
```

Import **lazy** di `transformers`; senza GPU/pesi resta usabile per smoke test.
Inferenza interattiva da CLI: `make infer` (`scripts/infer.py`).

---

## 7. Troubleshooting

| Sintomo | Causa probabile | Rimedio |
|---------|-----------------|---------|
| `ImportError: No module named 'italian_llm'` | package non installato editable, o `src/` non nel path | `pip install -e .`; gli script aggiungono `src/` via header `sys.path.insert(...)` |
| `ModuleNotFoundError: torch` su moduli puri | stai importando un modulo pesante | i moduli puri non richiedono torch; per training installa `requirements.txt` |
| `bitsandbytes` non si importa / errori CUDA | su Windows o senza CUDA | `bnb` è Linux/CUDA: usa WSL/Linux, o disattiva `quantization.load_in_4bit` per smoke run |
| `CUDA out of memory` | batch/seq troppo grandi | abbassa `batch_size`, alza `grad_accum`, riduci `max_seq_len`, `gradient_checkpointing: true`, usa 4-bit |
| `make help`/`make ...` fallisce su Windows | manca `make`/utility Unix | usa WSL/Git Bash o invoca `python scripts/...` |
| Teacher reale non risponde / 401 | `TEACHER_*` mancanti o errate | imposta `TEACHER_BASE_URL/API_KEY/MODEL`; in assenza c'è il **fallback mock** |
| `langdetect` assente | dipendenza eval non installata | `cleaning.detect_language` ricade sull'euristica stopword italiane |
| vLLM non installabile | Windows/CPU | usa il **fallback `transformers`** (`Generator`) |
| Modello HF "gated"/403 | manca il token o l'accettazione licenza | imposta `HF_TOKEN` e accetta la licenza sul Hub |
| `make eval` riporta latenza/VRAM "n/d" | nessuna GPU | atteso: le metriche testuali girano comunque ([evaluation-plan.md](./evaluation-plan.md#latenza-e-vram)) |
| Disco pieno durante il training | checkpoint del 7B voluminosi | sposta `outputs/` e `HF_HOME` su volume capiente; alza `save_steps` |

### Diagnostica rapida

```bash
make test                 # i moduli puri devono passare anche senza torch
make lint                 # ruff + black --check
python -c "import italian_llm.config, italian_llm.data.cleaning, italian_llm.safety.policy; print('moduli puri OK')"
python -c "import torch; print(torch.cuda.is_available())"   # verifica CUDA (se installato)
```

---

## 8. Igiene del repository

- **Mai committare** pesi/binari (`*.safetensors`, `*.bin`, `*.gguf`), dati grezzi
  (`data/raw/`, `data/interim/`), `outputs/`, segreti (`.env`). Sono in `.gitignore`.
- **Versionati**: codice, config, docs, `.gitkeep`, campioni `*.sample.jsonl`.
- **Prima di un giro "completo"** segui la checklist finale del README (setup verde,
  test/lint verdi, moduli puri importabili senza torch, dataset validi, adapter
  salvati, eval con report, niente segreti committati).
- **Riproducibilità**: serializza il config effettivo (`dump_config`) accanto ai
  checkpoint; fissa `project.seed`; valuta in greedy.
