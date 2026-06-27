# Registro delle decisioni (ADR)

> Log in stile **ADR** (Architecture Decision Record) di **ogni decisione autonoma**
> presa nel progetto. Ogni voce: **Contesto · Decisione · Razionale · Alternative ·
> Conseguenze**. Le decisioni sono **immutabili**: se cambia qualcosa, si aggiunge una
> nuova ADR che supera la precedente.

Documenti collegati:
[architecture.md](./architecture.md) ·
[training-plan.md](./training-plan.md) ·
[dataset-plan.md](./dataset-plan.md) ·
[distillation-plan.md](./distillation-plan.md) ·
[evaluation-plan.md](./evaluation-plan.md) ·
[operations.md](./operations.md)

| ADR | Decisione | Stato |
|-----|-----------|-------|
| [001](#adr-001) | Base Qwen + deviazione ~9B → Qwen2.5-7B (student 3B) | Accettata |
| [002](#adr-002) | Src layout, package `italian_llm` | Accettata |
| [003](#adr-003) | `argparse` (stdlib) invece di typer/click | Accettata |
| [004](#adr-004) | Config YAML con `_base_` + deep merge (PyYAML) | Accettata |
| [005](#adr-005) | ORPO come default per le preferenze (DPO opzionale) | Accettata |
| [006](#adr-006) | Data distillation di default (logits sperimentale) | Accettata |
| [007](#adr-007) | Mock teacher offline come default della sintesi | Accettata |
| [008](#adr-008) | Filosofia anti over-refusal (safety bilanciata) | Accettata |
| [009](#adr-009) | Import pesanti **lazy** dentro le funzioni | Accettata |
| [010](#adr-010) | QLoRA 4-bit (NF4) come default di training | Accettata |
| [011](#adr-011) | Due tracce V1/V2 sullo stesso codice | Accettata |

---

## ADR-001 — Base model: Qwen e deviazione ~9B → Qwen2.5-7B / student 3B {#adr-001}

**Contesto.** Il brief richiede una base aperta **Qwen ~9B**. Alla data del progetto,
nella linea **Qwen2.5** non esiste un checkpoint "9B" ufficiale e distribuito su
HuggingFace. La fascia disponibile è 0.5B / 1.5B / **3B** / **7B** / 14B / 32B / 72B.
Serve inoltre stare su **singola GPU 24 GB** (profilo *recommended*) per la V1.

**Decisione.** Usare **`Qwen/Qwen2.5-7B`** (7.6B parametri) come base "~9B target", e
**`Qwen/Qwen2.5-3B`** come student di distillazione. Default in
`configs/base.yaml`, `configs/model/qwen9b.yaml`, `configs/model/student_3b.yaml`.

**Razionale.**
- Nessuna 9B ufficiale Qwen2.5: la 7B è la variante aperta stabile **più vicina** al
  target, stessa famiglia/tokenizer.
- **Headroom VRAM**: 7B in QLoRA 4-bit sta comodamente su 24 GB con
  `gradient_checkpointing` e seq 2048.
- **Coerenza teacher↔student** (7B→3B, stesso vocabolario): abilita sia data sia
  **logits** distillation ([distillation-plan.md](./distillation-plan.md)).
- Salire ai ~9B reali è un **cambio di una riga** (`model.name: Qwen/Qwen2.5-14B`)
  riducendo batch/seq, senza toccare il resto della pipeline.

**Alternative considerate.**
- *Qwen2.5-14B come "≥9B"*: più vicino al numero, ma fuori budget VRAM per la V1 su
  24 GB; resta documentato come upgrade.
- *Mistral/Llama ~7-8B*: tokenizer meno forte sull'italiano nella nostra esperienza e
  fuori dal vincolo "famiglia Qwen" del brief.
- *Attendere/forzare una "9B"*: non esiste un checkpoint ufficiale → non perseguibile.

**Conseguenze.** Il "~9B" del brief è interpretato come **fascia**, non numero esatto;
la deviazione è esplicitata in ogni config del modello. La scelta è reversibile.

---

## ADR-002 — Src layout con package `italian_llm` {#adr-002}

**Contesto.** Repo che deve essere **importabile** in modo pulito, testabile e
pacchettizzabile, evitando import accidentali dalla working dir.

**Decisione.** Adottare il **src layout**: sorgenti in `src/italian_llm/`,
`pyproject.toml` con `package-dir = {"" = "src"}` e `packages.find where = ["src"]`.

**Razionale.**
- Evita il classico bug "funziona solo perché eseguo dalla root": il package va
  **installato** (`pip install -e .`) per essere importato, come in produzione.
- Separa nettamente codice sorgente, script, test, config.
- Gli script CLI aggiungono `src/` al path con un header standard
  (`sys.path.insert(0, .../src)`), così funzionano anche senza install.

**Alternative considerate.**
- *Flat layout* (`italian_llm/` nella root): più semplice ma soggetto a import
  impliciti e shadowing; scartato.

**Conseguenze.** Necessario `pip install -e .` (incluso in `make setup`). I test e gli
script usano l'header di path o il package installato.

---

## ADR-003 — `argparse` (stdlib) invece di typer/click {#adr-003}

**Contesto.** Gli script in `scripts/*.py` sono **CLI sottili** che leggono un
`--config` e lanciano una funzione del package. Vincolo globale: i moduli puri devono
girare con **solo stdlib + PyYAML**.

**Decisione.** Usare **`argparse`** (stdlib) per tutti gli script. Niente
`typer`/`click`.

**Razionale.**
- **Zero dipendenze aggiuntive**: coerente col principio "leggero per default".
- Le CLI sono **banali** (poche opzioni): argparse è più che sufficiente.
- Meno superficie di versione/breaking change da gestire.

**Alternative considerate.**
- *typer*: ergonomia migliore, ma trascina `click` e type-machinery non necessari per
  CLI così semplici.
- *click*: idem, dipendenza extra ingiustificata.

**Conseguenze.** CLI verbose-ino ma trasparenti e senza deps. Pattern uniforme:
header path → `argparse` → chiamata alla funzione del package.

---

## ADR-004 — Config YAML con `_base_` + deep merge (PyYAML) {#adr-004}

**Contesto.** Serve una configurazione **dichiarativa, componibile e riproducibile**
condivisa da data prep, training, distillazione, eval, serving. Molti config
condividono `project/paths/model/logging`.

**Decisione.** Config **YAML** con chiave speciale **`_base_`** (include relativo) +
**deep merge**, implementati in `src/italian_llm/config.py`:
`load_config(path, overrides=None)`, `deep_merge(a, b)`, `get(cfg, "a.b.c", default)`,
`dump_config(cfg, path)`. `_base_` è risolto **relativamente alla dir del file**.

**Razionale.**
- **DRY**: `configs/train/*.yaml` ereditano `configs/base.yaml` via
  `_base_: ../base.yaml`, sovrascrivendo solo ciò che cambia.
- **Deep merge** (dict fusi in profondità, scalari/liste sovrascritti dal figlio)
  rende prevedibile l'override.
- **PyYAML** è l'unica dipendenza core: nessun framework di config pesante.
- `get` con chiave puntata e `dump_config` per **riproducibilità** (serializzare il
  config effettivo accanto ai checkpoint).

**Alternative considerate.**
- *Hydra/OmegaConf*: potente ma pesante e con curva di apprendistato; eccessivo qui.
- *Pydantic-settings / dataclass tipizzate*: ottime per validazione ma irrigidiscono
  lo schema mentre la pipeline è ancora in evoluzione.
- *JSON*: niente commenti né include; YAML è più leggibile per i config.

**Conseguenze.** Schema "convenzionale" (non validato a tipi forti): la coerenza delle
chiavi è garantita dalla documentazione (questo doc + i commenti nei YAML) e dai test.

---

## ADR-005 — ORPO come default per l'allineamento alle preferenze (DPO opzionale) {#adr-005}

**Contesto.** Dopo l'SFT serve allineare **stile e preferenze** (tono diretto,
brevità, anti over-refusal) con coppie `(chosen, rejected)`. Vincolo: deve girare su
**singola GPU** quando possibile.

**Decisione.** **ORPO** di default (via `trl`), **single-stage**; **DPO** opzionale via
`preference.method: dpo` in `configs/train/orpo_qwen9b.yaml`.

**Razionale.**
- ORPO unisce SFT e preferenza in **un'unica loss** (NLL + odds-ratio) e **non
  richiede un modello di riferimento separato** → **meno VRAM**, niente secondo
  forward sul ref model.
- Pipeline più semplice: un solo stadio invece di SFT + DPO separati.
- DPO resta a disposizione per chi parte da un SFT molto forte e vuole separare le
  fasi.

**Alternative considerate.**
- *DPO*: standard e robusto, ma richiede ref model (più memoria/complessità) → reso
  **opzionale**.
- *PPO/RLHF online*: molto più costoso e instabile, fuori scope per V1.
- *KTO/IPO*: validi, ma ORPO offre il miglior rapporto semplicità/qualità/memoria per
  il nostro caso.

**Conseguenze.** Default memory-friendly. Il dataset preferenze
([dataset-plan.md](./dataset-plan.md)) serve entrambi i metodi senza modifiche.

---

## ADR-006 — Data distillation di default; logits distillation sperimentale {#adr-006}

**Contesto.** Si vuole uno **student 3B** servibile a basso costo che imiti il teacher
7B allineato. Due approcci: distillare sulle **risposte** (data) o sui **logit**.

**Decisione.** **Data distillation** come modalità di default
(`distill.mode: data`); **logits distillation** disponibile ma **sperimentale**
(`distill.mode: logits`), con limiti documentati.

**Razionale.**
- La data distillation è **robusta**, economica e funziona anche con teacher
  **remoto** (solo output testuale). Permette arricchimenti (best-of-n via
  `majority_rank`, filtri qualità).
- La logits distillation richiede **vocabolario identico**, **doppio forward**, molta
  **VRAM** e tuning delicato di `temperature`/`alpha`; il guadagno non è garantito.
  Vedi [distillation-plan.md](./distillation-plan.md#limiti).

**Alternative considerate.**
- *Solo logits distillation*: potenzialmente superiore ma fragile e costosa → non
  adatta al default V1.
- *Nessuna distillazione*: si servirebbe il 7B, più costoso; contro l'obiettivo di
  deploy economico.

**Conseguenze.** V1 usa data distillation; la logits resta backlog **V2** su hardware
serio, da confrontare con la baseline sulle **stesse metriche**
([evaluation-plan.md](./evaluation-plan.md#6-valutazione-dello-student)).

---

## ADR-007 — Mock teacher offline come default della sintesi {#adr-007}

**Contesto.** Su questo host non esistono GPU/cloud/API proprietarie, ma la pipeline
di generazione dati deve **funzionare end-to-end** e i test devono essere
**deterministici**.

**Decisione.** **`MockTeacher`** come default (`TEACHER_PROVIDER=mock`): risposte
italiane templative **deterministiche**, **zero rete**. I teacher reali
(`OpenAICompatTeacher`, `HFLocalTeacher`) sono **cablati** e si attivano via env; in
loro assenza, **fallback automatico al mock**.

**Razionale.**
- **Eseguibilità offline**: `make synth`, `make sft-data` e la distillazione data
  girano senza rete.
- **Determinismo**: test riproducibili senza dipendere da un servizio esterno.
- **Nessun segreto richiesto** per lo sviluppo della pipeline.

**Alternative considerate.**
- *Richiedere un endpoint reale obbligatorio*: bloccherebbe sviluppo e CI offline.
- *Dati sintetici statici pre-generati*: meno flessibili dei template parametrici del
  mock.

**Conseguenze.** La qualità "vera" dei dati arriva con un teacher reale (env
`TEACHER_*`), ma la **struttura** della pipeline è validabile da subito.

---

## ADR-008 — Filosofia anti over-refusal (safety bilanciata) {#adr-008}

**Contesto.** Molti assistant **rifiutano richieste lecite** e **moraleggiano**,
peggiorando l'esperienza. Il prodotto vuole essere **diretto e utile**, rifiutando
**solo** ciò che è genuinamente dannoso.

**Decisione.** Adottare una **safety bilanciata** in
`src/italian_llm/safety/policy.py`: `classify_request -> "allow"|"needs_care"|
"refuse"`, `should_refuse`, `SAFE_COMPLETION_TEMPLATES`, `balanced_system_prompt`.
Rifiutare **solo** categorie genuinamente dannose (armi/CBRN, malware reale per
nuocere, sfruttamento). Niente disclaimer superflui. Il dataset
([dataset-plan.md](./dataset-plan.md#riduzione-dell-over-refusal)) e l'ORPO sono
progettati per **ridurre l'over-refusal**.

**Razionale.**
- L'over-refusal su richieste lecite è il difetto di UX più comune e misurabile
  (`over_refusal_rate`, [evaluation-plan.md](./evaluation-plan.md)).
- Una policy esplicita e centralizzata evita "moralismo diffuso" sparso nel codice.
- Mantiene comunque **confini netti** sul genuinamente dannoso.

**Alternative considerate.**
- *Safety massimalista* (rifiuta nel dubbio): peggiora UX e italianità percepita;
  scartata.
- *Nessuna policy*: rischio sui contenuti davvero dannosi; scartata.

**Conseguenze.** Si misura **entrambi i lati**: `over_refusal_rate` (da minimizzare) e
`refusal_rate` legittimo (da mantenere sul dannoso). Un peggioramento dell'over-refusal
dopo l'allineamento è un segnale d'allarme da indagare.

---

## ADR-009 — Import delle dipendenze pesanti **lazy**, dentro le funzioni {#adr-009}

**Contesto.** Il repo deve essere **staticamente valido e importabile su Windows senza
GPU**, ma usa `torch`, `transformers`, `peft`, `trl`, `datasets`, `bitsandbytes`,
`accelerate`, `vllm` per il lavoro reale.

**Decisione.** **Mai** importare dipendenze pesanti a livello di modulo. Importarle
**lazy dentro le funzioni** che le usano. I moduli "puri" (`config`, `logging_utils`,
`data/prompts`, `data/schema`, `data/cleaning`, `safety/policy`, gran parte di
`evaluation/metrics`) importano con **solo stdlib + PyYAML**.

**Razionale.**
- **Validità statica** del repo e **test/lint** eseguibili senza GPU/deps pesanti.
- Tempi di import rapidi per i moduli puri e per le CLI leggere.
- Errori "missing dependency" emergono solo quando si usa davvero la feature pesante,
  con messaggi chiari.

**Alternative considerate.**
- *Import top-level con try/except*: rumoroso e fragile; nasconde gli errori.
- *Spezzare in due package (core/heavy)*: complessità di packaging non giustificata.

**Conseguenze.** Disciplina richiesta agli autori dei moduli (nessun import pesante in
testa). Verificato dai test che importano i moduli puri **senza** torch.

---

## ADR-010 — QLoRA 4-bit (NF4) come default di training {#adr-010}

**Contesto.** Addestrare un 7B su singola GPU 24 GB. Il full fine-tuning è fuori
budget di VRAM.

**Decisione.** Default **QLoRA 4-bit**: pesi base in **NF4**
(`bnb_4bit_quant_type: nf4`), compute in **bf16**
(`bnb_4bit_compute_dtype: bfloat16`), adapter **LoRA** su
`q,k,v,o,gate,up,down_proj` (`r=16`, `alpha=32`, `dropout=0.05`). In
`configs/model/qwen9b.yaml` e in tutti i `configs/train/*.yaml`.

**Razionale.**
- **NF4 + compute bf16** è il setup QLoRA standard col miglior rapporto
  qualità/memoria; NF4 batte fp4 in qualità.
- Si addestrano **solo gli adapter** (decine/centinaia di MB), non i 7B di pesi.
- `gradient_checkpointing: true` di default per ulteriore risparmio memoria.

**Alternative considerate.**
- *Full FT bf16*: qualità potenzialmente maggiore ma richiede profilo *serious*; resta
  un'opzione disattivando la quantizzazione su hardware adeguato
  ([training-plan.md](./training-plan.md#hardware-aware)).
- *8-bit*: meno aggressivo, ma 4-bit NF4 dà più headroom con qualità adeguata.

**Conseguenze.** `bitsandbytes` è **Linux/CUDA**: su Windows/CPU la quantizzazione si
disattiva per lo smoke run. La scelta è ribaltabile per profilo hardware.

---

## ADR-011 — Due tracce V1/V2 sullo **stesso** codice {#adr-011}

**Contesto.** Servono sia un percorso **pragmatico spedibile** sia un percorso **R&D**
di qualità superiore, senza mantenere due codebase.

**Decisione.** **Una sola codebase**; la differenza V1/V2 vive **solo nei file YAML**
di `configs/` (e nell'attivazione opzionale di CPT, multi-teacher, logits distill).
Vedi [architecture.md](./architecture.md#4-le-due-tracce-v1-pragmatica-e-v2-rd).

**Razionale.**
- Niente fork "prod vs research": gli stessi script e funzioni
  (`run_cpt/run_sft/run_preference/build_distill_dataset/run_eval`) servono entrambe.
- Riproducibilità: cambiare traccia = cambiare config, non codice.
- Riduce drift e duplicazione.

**Alternative considerate.**
- *Branch/repo separati per V2*: divergenza inevitabile e doppia manutenzione.
- *Flag hardcoded nel codice*: meno trasparenti dei config dichiarativi.

**Conseguenze.** I config devono coprire entrambi gli scenari (default conservativi V1,
override per V2). La documentazione mantiene allineata la corrispondenza
config↔comportamento.

---

> **Come aggiungere una ADR.** Crea una nuova sezione `## ADR-0NN — <titolo> {#adr-0NN}`,
> aggiorna la tabella in cima, e compila Contesto/Decisione/Razionale/Alternative/
> Conseguenze. Non modificare le ADR esistenti: se una decisione cambia, **supérala**
> con una nuova ADR che la referenzia.
