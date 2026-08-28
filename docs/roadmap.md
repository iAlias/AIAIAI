# Roadmap e checklist di rilascio

Documento di lavoro: dove sta andando il progetto e cosa serve perché un giro
completo possa dirsi concluso. Per lo stato attuale, vedi la nota in cima al
[README](../README.md).

---

## Roadmap a 30 giorni

Piano indicativo per portare il progetto da repository a modello spedibile (V1) e
gettare le basi della V2.

| Settimana | Obiettivo | Output |
|-----------|-----------|--------|
| **1 — Fondamenta e dati** | Ambiente, config, data prep, mock teacher, prime sintesi. Validazione schema e cleaning. | Dataset SFT v0 (reale + sintetico), pipeline dati verde, test verdi. |
| **2 — SFT v1** | SFT QLoRA del 9B sul dataset v0; primo giro di valutazione; iterazione su prompt e qualità. | Adapter SFT v1, report metriche baseline, correzioni di data quality. |
| **3 — Allineamento e valutazione** | Dataset di preferenze; ORPO; riduzione dell'over-refusal; suite di valutazione ampliata. | Modello allineato (stile e safety bilanciata), report comparativo. |
| **4 — Distillazione, serving e hardening** | Distillazione data-based su student 3B; serving vLLM/transformers; ablation leggere; documentazione. | Student 3B servibile, endpoint di inferenza, V1 spedibile, backlog V2 (CPT, logits distillation). |

Milestone trasversali: CI verde (`make test`, `make lint`), config riproducibili,
metriche tracciate, `.env` e segreti gestiti correttamente.

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
- [ ] ORPO eseguito; lo stile è più diretto e meno verboso; **over-refusal ridotto**.
- [ ] `make eval` produce un report con aderenza, italianità, verbosità, refusal,
      over-refusal, ROUGE-L (e pass@k dove applicabile).
- [ ] Distillazione data-based produce uno student 3B funzionante.
- [ ] `make serve` espone l'inferenza (vLLM o fallback transformers).
- [ ] Nessun peso, binario o segreto committato (rispetto del `.gitignore`).
- [ ] Documentazione in `docs/` allineata al comportamento effettivo.

---

## Traguardo mancante: pesi pubblicati

Il progetto avrà utenti veri quando esisterà **un modello scaricabile**, non solo
la ricetta per costruirlo. Il primo passo utile non è il 9B: è un adapter QLoRA su
un modello piccolo, addestrato anche per poche ore, pubblicato su Hugging Face con
il report di valutazione prodotto da `make eval`. Da lì il README può dire
"scarica e prova", che è la frase che trasforma uno spettatore in un utente.
