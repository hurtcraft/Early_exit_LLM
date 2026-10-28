from typing import Any

import torch

from early_exit.models.base import BaseModel

SYSTEM_PROMPT = (
    "Answer concisely and directly, in one sentence at most. "
)
class QwenModel(BaseModel):
    """Adaptateur pour les modèles Qwen compatibles."""

    @property
    def num_layers(self) -> int:
        return len(self.model.model.layers)

    
    def prepare_inputs(self, prompt: str, enable_thinking: bool = False, system_prompt: str | None = SYSTEM_PROMPT):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            return_tensors="pt",
            return_dict=True,
        )
        device = self.model.get_input_embeddings().weight.device
        return {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.embed_tokens(input_ids)

    # def forward_layer(
    #     self,
    #     layer_idx: int,
    #     hidden_states: torch.Tensor,
    #     **kwargs: Any,
    # ) -> tuple[torch.Tensor, dict[str, Any]]:
    #     layer = self.model.model.layers[layer_idx]

    #     outputs = layer(hidden_states, **kwargs)

    #     if isinstance(outputs, tuple):
    #         hidden_states = outputs[0]
    #         auxiliary_outputs = outputs[1:]
    #     else:
    #         auxiliary_outputs = ()

    #     return hidden_states, {"auxiliary_outputs": auxiliary_outputs}
    def forward_layer(self, layer_idx, hidden_states, **kwargs):
        layer = self.model.model.layers[layer_idx]
        outputs = layer(hidden_states, **kwargs)

        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            auxiliary_outputs = outputs[1:]
        else:
            hidden_states = outputs
            auxiliary_outputs = ()

        return hidden_states, {"auxiliary_outputs": auxiliary_outputs}
    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.model.model.norm(hidden_states)

        return self.model.lm_head(normalized)