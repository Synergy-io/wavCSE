# wavCSE Task Relation Learning Research

## Research problem

We study parameter-based multi-task learning for speech using fixed
wavCSE/WavLM embeddings.

Primary tasks:

- KS — keyword spotting
- SI — speaker identification
- ER — emotion recognition

Current research family:

Task Relation Learning.

The current optimization target is to significantly outperform the
reproduced wavCSE baseline while avoiding meaningful regression on any task.

The larger scientific objective is not merely performance.

We must characterize:

task properties
→ task relationships
→ transfer behavior
→ appropriate parameter-based MTL mechanisms.

The final research should support a framework for selecting parameter-based
MTL methods based on observable task relationships.

## Existing evidence

Read these before proposing experiments:

- improvements/taskrelation/01-mtrl/README.md
- improvements/taskrelation/02-lnp/README.md
- improvements/taskrelation/03-gbc/README.md
- improvements/base/POOLING_GRID_SEARCH.md
- improvements/taskrelation/research/STATE.md
- improvements/taskrelation/research/FINDINGS.md

Important established findings must not be rediscovered as if new.

In particular:

- single-seed wins are unreliable;
- MTRL has not shown a reproducible significant improvement;
- pooling choice can create larger effects than architecture choice;
- ER's historical single split contains speaker leakage;
- use LOSO evaluation for serious ER claims;
- learned Ω is considerably more stable for KS↔SI than relations involving ER.

## Scientific workflow

Every architecture change must begin with a falsifiable hypothesis.

Each study must have a Study ID.

One Study may contain many runs/seeds.

Do not promote an architecture from a single run.

Screen cheaply first.

Use multi-seed confirmation for promising candidates.

Use LOSO for ER claims.

Measure task behavior, not just final aggregate accuracy.

When a method fails, determine why before proposing another method.

When local ideas are exhausted, search the literature based on the
observed failure mode rather than searching generically for "better MTL".

## Formal Task Relation Learning Scope

The formal research progression begins with classical MTRL.

### Active starting method

`improvements/taskrelation/01-mtrl/`

Classical MTRL is the primary existing Task Relation Learning baseline.
Future mechanisms should be motivated by experimentally observed limitations
of MTRL and supported by legitimate literature where the contribution is
claimed as literature-derived.

### Diagnostic-only historical work

`improvements/taskrelation/02-lnp/`

LNP is not an active Task Relation Learning architecture.

Its purpose is methodological: it demonstrated that pooling/representation
choice can confound architecture comparisons and affect learned task relations.

Preserve and use this evidence when designing controlled experiments.

### Archived / out of current formal scope

`improvements/taskrelation/03-gbc/`

GBC is an original project architecture for which no defensible matching
published Task Relation Learning method has currently been established.

Do not use GBC as a literature-grounded Task Relation Learning method.

Do not spend autonomous research cycles optimizing GBC unless the human
explicitly reactivates it.

The implementation and historical results should remain preserved.

### Research progression

The default progression is:

wavCSE baseline
→ MTRL
→ MTRL behavioural diagnostics
→ identify modelling limitation
→ targeted literature search
→ literature-supported Task Relation Learning extension
→ screening
→ multi-seed confirmation
→ LOSO where required
→ ablation
→ framework synthesis

Do not skip directly from MTRL to arbitrary new architectures.
