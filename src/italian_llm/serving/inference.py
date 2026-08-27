"""Inferenza con HuggingFace Transformers (import pesanti lazy)."""

from italian_llm.logging_utils import get_logger

logger = get_logger(__name__)

# Default di generazione ragionevoli per un assistente italiano conciso.
_DEFAULT_GEN = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
}


def _resolve_dtype(dtype):
    """Converte una stringa/dtype in torch dtype (torch importato lazy)."""
    import torch

    if dtype is None:
        return None
    if not isinstance(dtype, str):
        return dtype  # gia' un torch.dtype
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "auto": None,
    }
    return mapping.get(dtype.lower(), None)


class Generator:
    """Generatore di testo su un modello causale, con adapter PEFT opzionale.

    Il modello viene caricato in modo pigro (lazy) alla prima generazione, cosi'
    importare questo modulo non richiede torch/transformers installati.

    Parametri:
      model_path: id HF o cartella locale del modello base.
      adapter: cartella di un adapter LoRA/PEFT da applicare (opzionale).
      merge_adapter: se True, fonde l'adapter nel modello (inferenza piu' veloce).
      device: 'cuda' | 'cpu' | None (auto).
      dtype: 'bfloat16' | 'float16' | 'float32' | None.
      load_in_4bit: carica il base in 4-bit (richiede bitsandbytes).
      I restanti kw sono usati come default di generazione.
    """

    def __init__(
        self,
        model_path,
        adapter=None,
        merge_adapter=True,
        device=None,
        dtype="bfloat16",
        load_in_4bit=False,
        trust_remote_code=True,
        **gen_kw,
    ):
        self.model_path = model_path
        self.adapter = adapter
        self.merge_adapter = merge_adapter
        self.device = device
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code
        # Default di generazione: base + override passati al costruttore.
        self.gen_defaults = dict(_DEFAULT_GEN)
        self.gen_defaults.update({k: v for k, v in gen_kw.items() if v is not None})

        self.model = None
        self.tokenizer = None
        self._loaded = False

    # ------------------------------------------------------------------ load
    def _ensure_loaded(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM

        # Device automatico se non specificato.
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch_dtype = _resolve_dtype(self.dtype)
        # Su CPU il bf16/fp16 e' spesso inutile o non supportato: usa float32.
        if self.device == "cpu" and torch_dtype in (torch.float16, torch.bfloat16):
            logger.info("Device CPU: forzo dtype float32 per stabilita'.")
            torch_dtype = torch.float32

        logger.info("Carico tokenizer da %s", self.model_path)
        from italian_llm.tokenizer_utils import load_tokenizer

        self.tokenizer = load_tokenizer(self.model_path)

        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": torch_dtype,
        }
        # Quantizzazione 4-bit opzionale (best-effort).
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype or torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["device_map"] = "auto"
            except Exception as e:  # pragma: no cover
                logger.warning("4-bit non disponibile (%s); carico in precisione piena.", e)

        logger.info(
            "Carico modello da %s (device=%s, dtype=%s)", self.model_path, self.device, torch_dtype
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)

        # Adapter PEFT opzionale.
        if self.adapter:
            from peft import PeftModel

            logger.info("Applico adapter PEFT da %s", self.adapter)
            model = PeftModel.from_pretrained(model, self.adapter)
            if self.merge_adapter:
                try:
                    model = model.merge_and_unload()
                    logger.info("Adapter fuso nel modello base.")
                except Exception as e:  # pragma: no cover
                    logger.warning("Merge dell'adapter fallito (%s); proseguo senza merge.", e)

        # Se non abbiamo usato device_map='auto', spostiamo noi sul device.
        if "device_map" not in model_kwargs:
            model = model.to(self.device)
        model.eval()

        self.model = model
        self._loaded = True
        logger.info("Modello pronto per l'inferenza.")

    # --------------------------------------------------------------- helpers
    def _to_messages(self, prompt):
        """Normalizza str o lista di messaggi in una lista di messaggi chat."""
        if isinstance(prompt, str):
            try:
                from italian_llm.data.prompts import build_messages

                return build_messages(None, prompt)
            except Exception:
                return [{"role": "user", "content": prompt}]
        if isinstance(prompt, list):
            return prompt
        raise TypeError("prompt deve essere str oppure list[dict] (messaggi chat).")

    # -------------------------------------------------------------- generate
    def generate(self, prompt, **gen_kw) -> str:
        """Genera una risposta da uno str (singolo turno utente) o da messaggi chat."""
        self._ensure_loaded()
        import torch

        messages = self._to_messages(prompt)

        # Costruzione del prompt testuale via tokenizer_utils (chat template +
        # fallback a render_plain).
        from italian_llm.tokenizer_utils import format_chat

        text = format_chat(self.tokenizer, messages, add_generation_prompt=True)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        # Merge dei parametri di generazione.
        params = dict(self.gen_defaults)
        params.update({k: v for k, v in gen_kw.items() if v is not None})
        # do_sample coerente con la temperatura.
        temperature = params.get("temperature", 0.7)
        if "do_sample" not in params:
            params["do_sample"] = bool(temperature and temperature > 0)
        if not params["do_sample"]:
            # In modalita' greedy questi parametri vanno rimossi per evitare warning.
            params.pop("temperature", None)
            params.pop("top_p", None)
            params.pop("top_k", None)
        if self.tokenizer.pad_token_id is not None:
            params.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        with torch.no_grad():
            output = self.model.generate(**inputs, **params)

        # Decodifichiamo solo i token nuovi (esclusa la parte di prompt).
        new_tokens = output[0][input_len:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return result.strip()
