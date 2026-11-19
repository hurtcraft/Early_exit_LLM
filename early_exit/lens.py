"""Tuned lens (Belrose et al., 2023) pour lire les couches intermédiaires.

Logit lens (actuel) : logits_l = lm_head(norm(h_l))
Tuned lens          : logits_l = lm_head(norm(h_l + A_l h_l + b_l))

Un « translateur » (A_l, b_l) par couche, appris pour que la couche l, lue avec la
tête du modèle, donne la même distribution que la dernière couche
(perte KL(dernière || couche l)). Le modèle reste gelé : le lens est à côté, sur le
chemin de lecture, et ne change pas le calcul des couches suivantes.

Translateurs initialisés à zéro : avant entraînement, tuned lens = logit lens.
La dernière couche (et toute couche sans translateur) est renvoyée telle quelle.

Usage :
    lens = TunedLens.for_adapter(adapter, "qwen3.5-2b")
    lens.fit(adapter, train_sequences)
    lens.evaluate(adapter, val_sequences)
    lens.save("lenses/qwen3.5-2b.pt")
    lens = TunedLens.load("lenses/qwen3.5-2b.pt", adapter)
    logits = adapter.get_logits(lens(h, layer))
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from early_exit.layers import _forward_trace


def _spans(length: int, skip: int, chunk: int):
    """Blocs de positions [a, b) : de `skip` à length - 1 (la dernière position
    n'a pas de token suivant pour la perplexité)."""
    for a in range(skip, length - 1, chunk):
        yield a, min(a + chunk, length - 1)


def _hidden_size(adapter) -> int:
    return adapter.model.get_input_embeddings().weight.shape[1]


def _device(adapter) -> torch.device:
    return adapter.model.get_input_embeddings().weight.device


class TunedLens(nn.Module):

    def __init__(self, hidden_size: int, layers, model_name: str | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = sorted(int(l) for l in layers)
        self.model_name = model_name
        self.meta = {}
        self.translators = nn.ModuleDict(
            {str(l): nn.Linear(hidden_size, hidden_size) for l in self.layers}
        )
        for t in self.translators.values():
            nn.init.zeros_(t.weight)
            nn.init.zeros_(t.bias)

    @classmethod
    def for_adapter(cls, adapter, model_name: str | None = None, min_layer: int | None = None):
        """Lens pour les couches min_layer .. avant-dernière (défaut : moitié de la
        profondeur, comme config.strategy_grid). Paramètres en float32."""
        n = adapter.num_layers
        min_layer = n // 2 if min_layer is None else min_layer
        lens = cls(_hidden_size(adapter), range(min_layer, n - 1), model_name)
        return lens.to(_device(adapter))

    def __contains__(self, layer) -> bool:
        return str(int(layer)) in self.translators

    def forward(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """h : [..., H] hidden state de la couche `layer` (dtype du modèle).
        Retourne h + A h + b dans le dtype de h, prêt pour adapter.get_logits."""
        key = str(int(layer))
        if key not in self.translators:
            return h
        x = h.float()
        return (x + self.translators[key](x)).to(h.dtype)

    # ---------- Entraînement ----------

    def fit(self, adapter, sequences: list, *, layers=None, lr: float = 1.0,
            momentum: float = 0.9, seqs_per_step: int = 8, warmup: float = 0.1,
            clip: float = 1.0, chunk: int = 256, skip: int = 1, desc: str = "tuned lens"):
        """Entraîne les translateurs de `layers` (défaut : tous) sur `sequences`
        (liste de LongTensor [1, L], texte générique). Un pas d'optimisation par
        groupe de `seqs_per_step` séquences, un seul passage sur les données.

        layers : sous-ensemble à entraîner, pour découper l'entraînement en groupes
                 quand tout ne tient pas en VRAM (9B) ; les autres sont inchangés.
        skip   : premières positions ignorées (activations énormes du 1er token).
        Retourne l'historique [(pas, KL moyenne par couche)].
        """
        layers = self.layers if layers is None else sorted(int(l) for l in layers)
        unknown = [l for l in layers if l not in self]
        if unknown:
            raise ValueError(f"couches sans translateur : {unknown}")
        n_steps = len(sequences) // seqs_per_step
        if n_steps == 0:
            raise ValueError(f"{len(sequences)} séquences < seqs_per_step={seqs_per_step}")

        params = [p for l in layers for p in self.translators[str(l)].parameters()]
        opt = torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=True)
        n_warm = max(1, int(warmup * n_steps))

        def schedule(step):  # montée linéaire puis décroissance linéaire jusqu'à 0
            if step < n_warm:
                return (step + 1) / n_warm
            return max(0.0, (n_steps - step) / max(1, n_steps - n_warm))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
        adapter.model.requires_grad_(False)   # modèle gelé
        device = _device(adapter)
        self.train()
        history = []

        bar = tqdm(range(n_steps), desc=desc, unit="pas")
        for step in bar:
            batch = sequences[step * seqs_per_step:(step + 1) * seqs_per_step]
            n_blocks = sum(len(list(_spans(ids.shape[1], skip, chunk))) for ids in batch)
            total = torch.zeros((), device=device)

            for ids in batch:
                ids = ids.to(device)
                with torch.no_grad():
                    trace = _forward_trace(adapter, ids, keep=ids.shape[1])  # [n_layers, L, H]
                for a, b in _spans(ids.shape[1], skip, chunk):
                    with torch.no_grad():
                        target = F.log_softmax(adapter.get_logits(trace[-1, a:b]).float(), dim=-1)
                    for l in layers:
                        logp = F.log_softmax(adapter.get_logits(self(trace[l, a:b], l)).float(), dim=-1)
                        loss = F.kl_div(logp, target, log_target=True, reduction="batchmean") / n_blocks
                        loss.backward()             # libère le graphe de cette couche
                        total += loss.detach()
                del trace

            kl = (total / len(layers)).item()
            if not math.isfinite(kl):
                tqdm.write(f"pas {step} : perte non finie, pas ignoré")
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            history.append((step, kl))
            bar.set_postfix(kl=f"{kl:.3f}", lr=f"{sched.get_last_lr()[0]:.2g}")

        self.eval()
        return history

    # ---------- Évaluation ----------

    @torch.no_grad()
    def evaluate(self, adapter, sequences: list, chunk: int = 256, skip: int = 1) -> dict:
        """Par couche du lens, sur `sequences` (à ne pas utiliser pour l'entraînement) :
        kl   : KL(dernière couche || couche l), moyenne par token
        top1 : part des tokens où le top-1 de la couche l = top-1 de la dernière couche
        ppl  : perplexité (par token) en lisant la couche l
        final_ppl : perplexité de la dernière couche (référence).
        Avant entraînement (translateurs à zéro), ce sont les valeurs du logit lens."""
        self.eval()
        device = _device(adapter)
        n_l = len(self.layers)
        kl = torch.zeros(n_l, dtype=torch.float64, device=device)
        agree = torch.zeros(n_l, dtype=torch.float64, device=device)
        nll = torch.zeros(n_l, dtype=torch.float64, device=device)
        nll_final = torch.zeros((), dtype=torch.float64, device=device)
        n_tok = 0

        for ids in tqdm(sequences, desc="évaluation", unit="séq", leave=False):
            ids = ids.to(device)
            trace = _forward_trace(adapter, ids, keep=ids.shape[1])
            for a, b in _spans(ids.shape[1], skip, chunk):
                final = F.log_softmax(adapter.get_logits(trace[-1, a:b]).float(), dim=-1)
                p_final = final.exp()
                top_final = final.argmax(dim=-1)
                tgt = ids[0, a + 1:b + 1, None]
                nll_final -= final.gather(-1, tgt).sum()
                for j, l in enumerate(self.layers):
                    logp = F.log_softmax(adapter.get_logits(self(trace[l, a:b], l)).float(), dim=-1)
                    kl[j] += (p_final * (final - logp)).sum()
                    agree[j] += (logp.argmax(dim=-1) == top_final).sum()
                    nll[j] -= logp.gather(-1, tgt).sum()
                n_tok += b - a
            del trace

        return {
            "layers": list(self.layers),
            "kl": (kl / n_tok).tolist(),
            "top1": (agree / n_tok).tolist(),
            "ppl": torch.exp(nll / n_tok).tolist(),
            "final_ppl": math.exp(nll_final.item() / n_tok),
            "n_tokens": n_tok,
        }

    # ---------- Sauvegarde ----------

    def save(self, path, **meta) -> Path:
        """Sauvegarde translateurs + métadonnées (types simples uniquement)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model_name,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "meta": {**self.meta, **meta},
        }, path)
        return path

    @classmethod
    def load(cls, path, adapter=None):
        """Charge un lens. Avec `adapter` : vérifie qu'il correspond au modèle
        (taille des hidden states, couches) et le place sur son device."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        lens = cls(ckpt["hidden_size"], ckpt["layers"], ckpt.get("model"))
        lens.load_state_dict(ckpt["state_dict"])
        lens.meta = ckpt.get("meta", {})
        if adapter is not None:
            if lens.hidden_size != _hidden_size(adapter):
                raise ValueError(f"{path} : hidden_size {lens.hidden_size} ≠ modèle "
                                 f"{_hidden_size(adapter)} (lens d'un autre modèle ?)")
            if lens.layers and lens.layers[-1] >= adapter.num_layers:
                raise ValueError(f"{path} : couche {lens.layers[-1]} absente du modèle "
                                 f"({adapter.num_layers} couches)")
            lens = lens.to(_device(adapter))
        return lens.eval()
