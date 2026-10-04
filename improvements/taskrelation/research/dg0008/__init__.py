"""DG-0008 exact-opportunity ER-target residual diagnostic (implementation).

The approved scientific specification is proposal DP-0008 at the immutable
commit recorded in ``studies/DG-0008/PLAN.md``.  This package is the faithful
implementation of that specification and nothing more:

* :mod:`manifest`  -- canonical identity/opportunity manifest contract
  (clauses 1-12 of the approved proposal), stdlib only.
* :mod:`sampler`   -- manifest-driven exact training/validation datasets.
* :mod:`trainer`   -- fixed-LR five-epoch diagnostic trainer with the
  default-off auxiliary-loss control and deterministic tracing.
* :mod:`gate`      -- fail-closed Stage-1 validity gate.
* :mod:`analysis`  -- frozen seed-level interval classification.

Nothing here authorizes compute, creates an authorization, or reads a
held-out endpoint outside the delayed gate.
"""

SCHEMA_IDENTITY = "dg0008.example-identity.v1"
SCHEMA_OPPORTUNITY = "dg0008.opportunity-manifest.v2"
SCHEMA_RUN_RECORD = "dg0008.run-record.v1"
SCHEMA_GATE = "dg0008.validity-gate.v1"
SCHEMA_ENDPOINT = "dg0008.endpoint.v1"
SCHEMA_ANALYSIS = "dg0008.analysis.v1"

BATCH_SIZE = 2048
EPOCHS = 5
SEEDS = (0, 1, 2, 3, 4)
FOLDS = tuple(range(10))
CELLS = ("ks_er", "si_er")
ARMS = ("pair", "control")

# Component order per cell, clause 6.
CELL_COMPONENTS = {
    "ks_er": ("speechcommand", "iemocap"),
    "si_er": ("voxceleb", "iemocap"),
}
