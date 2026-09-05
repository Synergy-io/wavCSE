"""mtlkit/pooling.py — layer-pooling strategies, registry-dispatched.

Moved verbatim from `downstream/pooling/pooling.py` (Next Step 2). Only
layer pooling (reducing the transformer-layer axis to one vector per
utterance) lives here — frame pooling stays upstream/frozen, out of scope
(see the design doc's Glossary).

Dispatch is now through `POOLING_REGISTRY` (Eng Review decision 3B's shared
`Registry[T]`) instead of the original per-instance dict
(`self.pooling_operations`), so an unsupported pooling type gets the
registry's standard "valid options: ..." message. The original
`ValueError` contract on an unsupported type is preserved (callers catching
`ValueError` specifically keep working) — only the message's valid-options
list is now generated from the registry instead of hand-maintained.
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mtlkit.registry import Registry

# pooling_type -> Pooling method name. A registry of *method names* (not bound
# methods) because dispatch needs `self` — see get_vector_after_pooling below.
POOLING_REGISTRY: Registry[str] = Registry("pooling strategy")


def _register_builtin_pooling_strategies() -> None:
    for pooling_type, method_name in (
        ("mean", "mean_pooling"),
        ("max", "max_pooling"),
        ("mix", "mixed_pooling"),
        ("lnp", "learned_norm_pooling"),
        ("smp", "softmax_pooling"),
        ("lse", "log_sum_exp_pooling"),
        ("weighted", "weighted_pooling"),
        ("gated", "gated_pooling"),
        ("auto", "auto_pooling"),
        ("sap", "self_attentive_pooling"),
    ):
        POOLING_REGISTRY.register(pooling_type, method_name)


_register_builtin_pooling_strategies()


class Pooling(nn.Module):
    """Unified layer-pooling interface: reduces a sequence of layer-wise
    representations into one fixed-dimensional vector per utterance."""

    def __init__(
        self,
        pooling_type: str,
        pooling_param: Optional[Union[int, float]] = None,
    ):
        super().__init__()

        self.pooling_type = pooling_type
        self.pooling_param = pooling_param

        # stores last forward auxiliaries
        self._last_aux: Dict[str, Any] = {}

        if self.pooling_type not in POOLING_REGISTRY:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        # Learnable position-wise weights used by BOTH weighted and gated
        if self.pooling_type in {"weighted", "gated"}:
            if pooling_param is None:
                raise ValueError(f"{self.pooling_type} pooling requires pooling_param = seq_len")
            self.position_weights = nn.Parameter(torch.ones(int(pooling_param)))

        # Learnable alpha for auto pooling
        if self.pooling_type == "auto":
            init_alpha = 0.01 if pooling_param is None else float(pooling_param)
            self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float))

        # Self-attentive pooling (SAP) needs input_dim (D)
        if self.pooling_type == "sap":
            if pooling_param is None:
                raise ValueError("sap pooling requires pooling_param = input_dim (D)")
            d_model = int(pooling_param)
            self.sap_W = nn.Linear(d_model, d_model)
            self.sap_v = nn.Linear(d_model, 1, bias=False)

    # ---------------- basic ----------------
    def mean_pooling(self, x, d):
        self._last_aux = {}
        return x.mean(dim=d)

    def max_pooling(self, x, d):
        self._last_aux = {}
        return x.max(dim=d).values

    def mixed_pooling(self, x, d):
        self._last_aux = {}
        if self.pooling_param is None:
            raise ValueError("mix pooling requires pooling_param = mix_ratio")
        mix_ratio = float(self.pooling_param)
        return mix_ratio * self.max_pooling(x, d) + (1.0 - mix_ratio) * self.mean_pooling(x, d)

    def learned_norm_pooling(self, x, d):
        self._last_aux = {}
        if self.pooling_param is None:
            raise ValueError("lnp pooling requires pooling_param = p")
        p = int(self.pooling_param)
        n = x.size(d)
        return torch.pow(torch.sum(torch.abs(x) ** p, dim=d) / n, 1.0 / p)

    def softmax_pooling(self, x, d):
        self._last_aux = {}
        if self.pooling_param is None:
            raise ValueError("smp pooling requires pooling_param = lambda")
        lam = float(self.pooling_param)
        w = torch.softmax(lam * x, dim=d)
        return torch.sum(w * x, dim=d)

    def log_sum_exp_pooling(self, x, d):
        self._last_aux = {}
        if self.pooling_param is None:
            raise ValueError("lse pooling requires pooling_param = r")
        r = float(self.pooling_param)
        n = x.size(d)
        return (1.0 / r) * torch.log(torch.sum(torch.exp(r * x), dim=d) / n)

    # ---------------- weighted ----------------
    def weighted_pooling(self, x, d):
        # x: [B, L, D], pool over L
        self._last_aux = {}

        w = F.softmax(self.position_weights, dim=0)          # [L]
        out = torch.sum(x * w.view(1, -1, 1), dim=d)         # [B, D]

        self._last_aux["position_weights"] = self.position_weights.detach()
        self._last_aux["weighted_weights"] = w.detach()

        return out

    # ---------------- gated (vector gate variant) ----------------
    def gated_pooling(self, x, d):
        self._last_aux = {}

        avg = x.mean(dim=d)                 # [B, D]
        mx = x.max(dim=d).values            # [B, D]

        s = torch.sum(x * self.position_weights.view(1, -1, 1), dim=d)  # [B, D]
        g = torch.sigmoid(s)                # [B, D]   (vector gate)

        out = g * mx + (1.0 - g) * avg

        self._last_aux["position_weights"] = self.position_weights.detach()
        self._last_aux["gate"] = g.detach()

        return out

    # ---------------- auto pooling ----------------
    def auto_pooling(self, x, d):
        self._last_aux = {}

        exp_alpha_x = torch.exp(self.alpha * x)
        out = torch.sum(x * exp_alpha_x, dim=d) / (torch.sum(exp_alpha_x, dim=d) + 1e-12)

        self._last_aux["alpha"] = self.alpha.detach()

        return out

    # ---------------- SAP (self-attentive pooling) ----------------
    def self_attentive_pooling(self, x, d):
        # x: [B, L, D], pool over L
        self._last_aux = {}

        h = torch.tanh(self.sap_W(x))              # [B, L, D]
        scores = self.sap_v(h).squeeze(-1)         # [B, L]
        a = torch.softmax(scores, dim=d)           # [B, L]
        out = torch.sum(x * a.unsqueeze(-1), dim=d)  # [B, D]

        self._last_aux["sap_weights"] = a.detach()
        self._last_aux["attn_scores"] = scores.detach()

        return out

    # ---------------- API ----------------
    def get_vector_after_pooling(self, data, dim: int):
        method_name = POOLING_REGISTRY.get(self.pooling_type)
        return getattr(self, method_name)(data, dim)

    def get_last_aux(self) -> Dict[str, Any]:
        """
        Call after forward() to inspect:
          - weighted: position_weights, weighted_weights
          - gated: gate, position_weights
          - auto: alpha
          - sap: sap_weights, attn_scores
        """
        return self._last_aux
