from early_exit.models.qwen import QwenModel
from early_exit.models.base import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
from datetime import datetime
from pathlib import Path

class ModelFactory:

    MODELS_DIR = Path("models")

    
    @staticmethod
    def load_class(name: str, quant: str | None = None) -> BaseModel:
        model, tokenizer = ModelFactory.load_model(name, quant)

        if name.lower().startswith("qwen"):
            return QwenModel(model, tokenizer)

        raise ValueError(f"Architecture non prise en charge : {name}")

    @staticmethod
    def load_model(name: str, quant: str | None = None):
        """quant : None (fp16/bf16), '8bit' ou '4bit'."""
        path = ModelFactory.MODELS_DIR / name
        tokenizer = AutoTokenizer.from_pretrained(path)

        kwargs = {"device_map": "auto"}

        if quant == "8bit":
            kwargs["dtype"] = torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif quant == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif quant is None:
            kwargs["dtype"] = torch.bfloat16
        else:
            raise ValueError(f"quant inconnu : {quant}")

        model = AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
        return model, tokenizer
