import torch


class MonitoringHiddenStates:
    """Affiche, couche par couche, la distribution de probabilité des tokens
    (logit lens) à la dernière position. Indépendant de toute stratégie d'early exit."""

    def __init__(self, model, tokenizer, top_k: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k

    # ---------- Entrée ----------

    def _encode(self, prompt: str, use_chat_template: bool) -> torch.Tensor:
        if use_chat_template:
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = inputs["input_ids"]
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        return input_ids.to(self.model.get_input_embeddings().weight.device)

    # ---------- Projection d'une couche vers le vocabulaire ----------

    @torch.no_grad()
    def _layer_probs(self, hidden_states, final_logits) -> list[torch.Tensor]:
        """Retourne une distribution (vocab,) par entrée de hidden_states."""
        text_model = self.model.model
        probs = []
        last_idx = len(hidden_states) - 1

        for i, h in enumerate(hidden_states):
            if i == last_idx:
                # Le dernier hidden state HF a déjà subi la norme finale :
                # on utilise directement les logits officiels du modèle.
                logits = final_logits
            else:
                logits = self.model.lm_head(text_model.norm(h[:, -1, :]))
            probs.append(torch.softmax(logits.float(), dim=-1)[0])
        return probs

    def _format(self, probs: torch.Tensor) -> str:
        top_p, top_i = probs.topk(self.top_k)
        return " | ".join(
            f"{self.tokenizer.decode([i], clean_up_tokenization_spaces=False)!r}: {p * 100:6.2f}%"
            for p, i in zip(top_p.tolist(), top_i.tolist())
        )

    # ---------- API publique ----------

    @torch.no_grad()
    def show(
        self,
        prompt: str,
        max_new_tokens: int = 1,
        use_chat_template: bool = True,
    ) -> str:
        """Génération gloutonne ; pour chaque token généré, affiche la
        distribution top-k à chaque couche."""
        input_ids = self._encode(prompt, use_chat_template)
        generated = []

        print("=" * 110)
        print("Prompt :", prompt)
        print("=" * 110)

        for step in range(max_new_tokens):
            out = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            final_logits = out.logits[:, -1, :]
            probs = self._layer_probs(out.hidden_states, final_logits)

            print(f"\n--- TOKEN {step + 1} ---")
            for layer, p in enumerate(probs):
                label = "emb" if layer == 0 else f"{layer:02d} "
                print(f"Layer {label}| {self._format(p)}")

            next_id = final_logits.argmax(dim=-1, keepdim=True)
            token = self.tokenizer.decode(next_id[0], clean_up_tokenization_spaces=False)
            print(f"\n[Token généré] : {token!r}")
            generated.append(token)

            input_ids = torch.cat([input_ids, next_id], dim=-1)
            if next_id.item() == self.tokenizer.eos_token_id:
                break

        answer = "".join(generated)
        print("\n" + "=" * 110)
        print("Réponse générée :", answer)
        return answer