"""mtlkit/compose.py — deterministic declared-order composition of
parameter-sharing strategies on one shared layer (Premise 4).

v1 is validated only by a synthetic fixture strategy (`FixtureScaleStrategy`
below) — no real method (MTRL, Low-Rank, GBC, or otherwise) validates this
seam's shape yet. This is a knowingly accepted risk (see the design doc's
Reviewer Concerns and Eng Review cross-model tension 1: the shape stays
output-space-only, not broadened speculatively to weight-space or
objective-space composition, per the user's explicit decision).

Composition semantics (declared per-strategy via `combine_op`):

    shared_output
         |
         +----------------------+----------------------+
         v (additive)           v (additive)            v (sequential)
    strategy_A(shared_output)  strategy_B(shared_output)  strategy_C(out_so_far)
         |                       |                             |
         +-----------+-----------+                             |
                     v (sum)                                    v
              accumulated_additive                    out = strategy_C(out)
                     |                                          |
                     +------------------ + -----------------------+
                                          v
                            final = out + accumulated_additive

"additive" strategies each see the ORIGINAL shared_output independently and
their outputs are summed once at the end. "sequential" strategies chain:
each one transforms whatever the running `out` is so far, in declared order.
Declared order therefore only matters for sequential strategies among
themselves and for where they fall relative to each additive strategy's
computation of the original shared_output — additive strategies are
order-independent with respect to each other.
"""

from typing import List

import torch
import torch.nn as nn

from mtlkit.registry import Registry

_VALID_COMBINE_OPS = ("additive", "sequential")

COMPOSE_REGISTRY: Registry[type] = Registry("compose strategy")


def register_compose_strategy(key: str):
    """Class decorator: ``@register_compose_strategy("my_strategy")``."""

    def _decorator(cls: type) -> type:
        COMPOSE_REGISTRY.register(key, cls)
        return cls

    return _decorator


def build_compose_strategy(key: str, **kwargs) -> "ComposeStrategy":
    cls = COMPOSE_REGISTRY.get(key)
    return cls(**kwargs)


class ComposeStrategy(nn.Module):
    """A parameter-sharing strategy wrapping a shared layer's output.

    Subclasses set the class attribute ``combine_op`` to ``"additive"`` or
    ``"sequential"`` and implement ``forward``.
    """

    combine_op: str = "additive"

    def forward(self, shared_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def compose(shared_output: torch.Tensor, strategies: List[ComposeStrategy]) -> torch.Tensor:
    """Apply ``strategies`` to ``shared_output`` in the given (declared)
    order, combining each per its own ``combine_op``. See module docstring
    for the exact semantics.
    """
    out = shared_output
    accumulated_additive = torch.zeros_like(shared_output)

    for strategy in strategies:
        op = strategy.combine_op
        if op == "additive":
            accumulated_additive = accumulated_additive + strategy(shared_output)
        elif op == "sequential":
            out = strategy(out)
        else:
            raise ValueError(
                f"Unknown combine_op '{op}' for strategy {type(strategy).__name__}. "
                f"Valid combine_ops: {', '.join(_VALID_COMBINE_OPS)}"
            )

    return out + accumulated_additive


@register_compose_strategy("fixture_scale")
class FixtureScaleStrategy(ComposeStrategy):
    """Synthetic fixture strategy — validates compose()'s wiring only.

    Multiplies the shared trunk output by a single learnable scalar. It
    models no real parameter-sharing method; it exists purely to prove a
    second strategy can be stacked with the shared trunk via declared-order
    composition and train end-to-end without shape or gradient errors (the
    Composition seam's fixture-only success criterion).
    """

    combine_op = "additive"

    def __init__(self, init_scale: float = 0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, shared_output: torch.Tensor) -> torch.Tensor:
        return self.scale * shared_output


@register_compose_strategy("fixture_identity")
class FixtureIdentityStrategy(ComposeStrategy):
    """Synthetic sequential fixture — a no-op pass-through, for testing
    sequential-op chaining without changing numeric output."""

    combine_op = "sequential"

    def forward(self, shared_output: torch.Tensor) -> torch.Tensor:
        return shared_output
