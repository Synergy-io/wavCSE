# Task Relation Learning Literature Registry

Primary-source literature cards for the formal Task Relation Learning programme. Cards record both eligible mechanisms and explicit taxonomy/assumption rejections; a paper’s presence here does not mean it is approved for implementation.

Paper identity is declared in [`catalog.jsonl`](catalog.jsonl): each existing
card filename stem is its immutable `paper_id`, and the catalog maps that ID and
recorded external identifiers to the canonical card. The catalog contains no
screening verdicts; those live in [`assessments.jsonl`](assessments.jsonl), the
canonical structured authority, and the `LT-*` tables below remain as
transitional human witnesses. Validate it with
`python -m improvements.taskrelation.research.literature_catalog`.

## Structured query interface

Use `python -m improvements.taskrelation.research.literature_query` for
read-only identity, metadata and `LT-*` Study relationships before opening
cards:

```bash
python -m improvements.taskrelation.research.literature_query list
python -m improvements.taskrelation.research.literature_query resolve <paper-id-or-alias>
python -m improvements.taskrelation.research.literature_query identify --doi 10.1145/3580305.3599261
python -m improvements.taskrelation.research.literature_query paper-studies <paper-id-or-alias>
python -m improvements.taskrelation.research.literature_query study-papers LT-0001
```

Results contain compact paper metadata, canonical card paths, Study metadata
and Study-scoped assessment records. A structured per-paper assessment is
returned for every `(investigation, paper)` pair; the assessment is canonical and
never a global paper status.

`identify` is the deduplication front door for a candidate paper: it reports
`known` (with the existing `paper_id` and card), `new`, or `ambiguous`. It uses
only the catalog's existing identity vocabulary — immutable slug, normalized
title, recorded DOI/arXiv, recorded source URL — with exact matching and no
fuzzy search, so it cannot invent a second identity system. Ambiguity, when
supplied fields point at different papers, is a returned verdict rather than a
silent pick.

### Operation authority

| Operation | READ/WRITE | Source of truth | Notes |
| --- | --- | --- | --- |
| `list_papers` | READ | `catalog.jsonl` | exact metadata filters only |
| `resolve_paper` | READ | `catalog.jsonl` | raises on unknown identity |
| `identify_candidate` | READ | `catalog.jsonl` | returns `known` / `new` / `ambiguous` |
| `studies_for_paper` | READ | `catalog.jsonl` × `assessments.jsonl` | each record investigation-scoped |
| `papers_for_study` | READ | `assessments.jsonl` | raises on unknown Study |
| `get_assessment` | READ | `assessments.jsonl` | exact `(investigation_id, paper_id)` |
| `get_study` | READ | `STUDIES.jsonl` + Study folder | artifact presence only |

Every operation above is read-only, idempotent and deterministic; none mutates
research state. Mutation of literature state remains outside this interface and
belongs to the future increments.

## Canonical paper assessments

[`assessments.jsonl`](assessments.jsonl) is the one structured machine-readable
authority for how an `LT-*` investigation assessed a paper. Its identity is
exactly `(investigation_id, paper_id)` and at most one current record exists for
that pair; `investigation_id` stays in the shared Study-ID namespace (`LT-0001`
and `LT-0002` are registered Studies) and `paper_id` is always a canonical
catalog identity. There is no global paper verdict: the catalog carries no
screening status and a card carries no global verdict.

Each record carries the paper's investigation-scoped `role`, the investigation's
`verdict`, the `gates` it applied (with normalised gate outcomes where the
investigation was gate-structured), a verbatim `reason_summary`, an
`assessment_anchor` (`<LT analysis>#<section>`) that points at the investigation's
own reasoning — never at a card — and the `assessed_at` boundary. Vocabularies are
investigation-scoped and derived from the existing records.

Validate the registry with
`python -m improvements.taskrelation.research.literature_assessment validate`, and
prove it is the *sole active authority* — no card verdict section, no
`STUDIES.jsonl` `cards`, no INDEX drift — with
`... literature_assessment check`. The table below is a derived view generated
from the registry; regenerate it with
`... literature_assessment index-assessments`.

Historical prose elsewhere — the completed LT-0001 `result.json`, the
`studies/LT-*/analysis.md` narrative — is retained evidence, not an assessment
authority.

<!-- BEGIN GENERATED: assessments -->
| Investigation | Paper | Role | Verdict |
| --- | --- | --- | --- |
| `LT-0001` | [chang-et-al-2024-informative-relations](chang-et-al-2024-informative-relations.md) | `recent_explicit_relation_method` | `exclude_decomposition_boundary` |
| `LT-0001` | [chen-et-al-2018-gradnorm](chen-et-al-2018-gradnorm.md) | `direct_gradient_scale_balancing` | `exclude_taxonomy` |
| `LT-0001` | [feldman-et-al-2014-mta](feldman-et-al-2014-mta.md) | `sample_variance_aware_relation` | `retain_framework_evidence_reject_implementation` |
| `LT-0001` | [kendall-et-al-2018-uncertainty-weighting](kendall-et-al-2018-uncertainty-weighting.md) | `task_reliability_loss_scale` | `exclude_taxonomy` |
| `LT-0001` | [rakitsch-et-al-2013-structured-residuals](rakitsch-et-al-2013-structured-residuals.md) | `signal_noise_relation_separation` | `retain_diagnostic_principle_reject_implementation` |
| `LT-0001` | [zhang-yang-2017-spats](zhang-yang-2017-spats.md) | `sparse_task_covariance` | `reject_for_F9` |
| `LT-0001` | [zhang-yang-2021-mtl-survey](zhang-yang-2021-mtl-survey.md) | `taxonomy_anchor` | `retain` |
| `LT-0001` | [zhang-yeung-2010-mtgtp](zhang-yeung-2010-mtgtp.md) | `bayesian_relation_uncertainty` | `reject_implementation` |
| `LT-0002` | [bonilla-2007-mtgp](bonilla-2007-mtgp.md) | `family_b_estimator` | `pass_with_documented_deviation` |
| `LT-0002` | [fifty-2021-tag](fifty-2021-tag.md) | `family_b_estimator` | `fail` |
| `LT-0002` | [goncalves-2016-mssl](goncalves-2016-mssl.md) | `family_b_estimator` | `pass` |
| `LT-0002` | [graffeuille-2024-self-auxiliaries](graffeuille-2024-self-auxiliaries.md) | `family_a_directed` | `fail` |
| `LT-0002` | [lee-2016-asymmetric-mtl](lee-2016-asymmetric-mtl.md) | `family_a_directed` | `pass_with_documented_deviation` |
| `LT-0002` | [lee-2018-deep-asymmetric-mtfl](lee-2018-deep-asymmetric-mtfl.md) | `family_a_directed` | `fail` |
| `LT-0002` | [liu-2017-trace-lasso-gamtl](liu-2017-trace-lasso-gamtl.md) | `family_a_directed` | `fail` |
| `LT-0002` | [nguyen-2021-tp-amtl](nguyen-2021-tp-amtl.md) | `family_a_directed` | `fail` |
| `LT-0002` | [oliveira-2019-group-lasso-asymmetric](oliveira-2019-group-lasso-asymmetric.md) | `family_a_directed` | `pass_with_documented_deviation` |
| `LT-0002` | [yu-2007-t-processes](yu-2007-t-processes.md) | `family_b_estimator` | `fail` |
| `LT-0002` | [yu-2020-graph-adjacency-gamtl](yu-2020-graph-adjacency-gamtl.md) | `family_a_directed` | `fail` |
| `LT-0002` | [zhang-schneider-2010-sparse-matrix-normal](zhang-schneider-2010-sparse-matrix-normal.md) | `family_b_estimator` | `fail` |
| `LT-0002` | [zhang-yeung-2014-mtrl-asymmetric](zhang-yeung-2014-mtrl-asymmetric.md) | `family_a_directed` | `fail` |
| `LT-0002` | [zhao-2020-fetr](zhao-2020-fetr.md) | `family_b_estimator` | `fail` |
| `LT-0002` | [zhou-2023-autotr](zhou-2023-autotr.md) | `family_a_directed` | `pass_with_documented_deviation` |
<!-- END GENERATED: assessments -->

## Primary artifacts

Retained original papers are addressed by `paper_id`, never by bucket, key, path
or credential:

```bash
python -m improvements.taskrelation.research.literature_primary validate
python -m improvements.taskrelation.research.literature_primary status <paper-id>
python -m improvements.taskrelation.research.literature_primary get <paper-id>
```

Authority split:

| Store | Owns |
| --- | --- |
| Git (`primary_manifest.jsonl`) | which `paper_id` has a retained artifact, its SHA-256, size, media type, provenance URL and object key |
| S3 | the canonical durable artifact bytes; publication, credentials, transfer and eviction live in the `wavcse-infra` checkout |
| local cache | disposable performance copies, never authoritative |

The manifest contains storage metadata only — never title, authors, year or any
other catalog identity. `object_key` is validated to equal the deterministic
derivation `[prefix]papers/<paper_id>/<role>.pdf`, so keys are derived from
`paper_id` and never authored by hand; changing a title cannot move an artifact.

The cache lives outside the repository (default `$XDG_CACHE_HOME/wavcse/literature-primary`,
else `~/.cache/wavcse/literature-primary`); a cache root inside the repository is
refused. A cached copy is trusted only when its SHA-256 matches the manifest, and
a corrupt entry is reported and removed rather than returned or silently
repaired. A manifest row declares retention under explicit infrastructure
authority — this repository never fabricates checksums, and an empty manifest
means no primary artifact is retained yet, so every paper reports
`PRIMARY_NOT_AVAILABLE` precisely.

Deterministic failure kinds: `UNKNOWN_PAPER`, `PRIMARY_NOT_AVAILABLE`,
`STORAGE_NOT_CONFIGURED`, `CREDENTIALS_UNAVAILABLE`, `REMOTE_RETRIEVAL_FAILED`,
`INTEGRITY_MISMATCH`. Credentials are never read, returned or logged here.

Retrieving a paper establishes trustworthy access to primary evidence only; it
does not summarize, extract claims from, parse, or modify any card. Claim-level
work is a later increment.

## Literature Agent (read-only V1)

A bounded Literature Review specialist exists for delegating one literature
question to a fresh, isolated context:

- `.omp/agents/literature-reviewer.md` — the agent definition (responsibility,
  authority, evidence discipline, output contract). Model role: `@slow`.
- `.agents/skills/wavcse-literature-review/SKILL.md` — the investigation
  methodology, including progressive disclosure and the four-way separation of
  paper claim / reported evidence / agent interpretation / research implication.
- `.omp/tools/literature.ts` — the semantic model-facing capabilities.

Exposed capabilities (all read-only over canonical research state):

| Tool | Purpose | Deterministic backing |
| --- | --- | --- |
| `literature_resolve` | identity resolution and candidate dedup (`known`/`new`/`ambiguous`) | `literature_query.identify_candidate` |
| `literature_query` | enumerate retained papers; Study-scoped literature state | `list_papers`, `papers_for_study`, `studies_for_paper`, `get_study` |
| `literature_read` | bounded text of one card or one registered `LT-*` artifact | `literature_read.LiteratureReader` |
| `literature_primary` | primary-artifact status / verified local copy | `literature_primary.LiteraturePrimary` |

The agent is addressed by `paper_id` or a registered Study artifact kind; it is
never given a bucket, object key, filesystem path, credential, or shell. Its only
write is the disposable primary cache. It cannot modify any research record,
authorize an experiment, or provision compute.

Verify the adapter with `make literature-tools-check` (needs `bun`); its static
contract is additionally enforced inside `make research-check`.

Two open gaps observed from real use are recorded in the roadmap: primary
artifacts are not yet retained (`PRIMARY_NOT_AVAILABLE` for every paper, so
primary verification is unavailable and answers rest on cards), and no wiring
exists to `infra/` (INC-004B).

This interface does not answer semantic claim questions such as which papers
support asymmetric transfer, contradict a mechanism, or learn a particular
relation. Those require claim-level evidence that is not represented yet; do
not infer it from titles or screening outcomes.

## Current literature Study

`LT-0002 — Published relation-learning variants for two selected families` (DEC-0013)

**Decision:** `COMPLETE — candidates found`. Family **B** (relation estimator / task-parameter representation) yields one clean pass: **Gonçalves et al. 2016 p-MSSL**, a sparse task *precision* learned by graphical lasso. Family **A** (asymmetric / directed relations) yields three methods whose relation object and update rule are published but which require a declared project-level deviation (the relation acts on class-mean head summaries rather than per-task model parameters): **Lee et al. 2016 AMTL**, **Zhou & Yang 2023 AutoTR**, **Oliveira et al. 2019 GAMTL**. Whether deviation-class arms are admissible is a human call, because LT-0002's falsification rule counted a documented deviation as a faithfulness fail.

`LT-0001 — Scale- and reliability-aware task relation literature gate` (superseded premise)

**Decision:** `REJECTED` candidate-method hypothesis. No reviewed published method simultaneously satisfies the explicit-relation, direct-F9, heterogeneous-classification, faithful-implementation, and category-boundary gates. Its rejections were scoped to the F9 gradient-scale rationale, which DEC-0010 later withdrew — see the closure note at the end of this file.

## LT-0001 synthesis

The literature separates the two properties F9 connects:

1. **Explicit Task Relation Learning** methods learn parameter/function covariance, sparse relations, or signal/noise covariance. The reviewed faithful formulations do not regulate shared-gradient magnitude in heterogeneous deep classifiers.
2. **Scale/reliability methods** such as uncertainty weighting and GradNorm directly address loss or gradient dominance, but learn per-task scalar weights rather than task relations and therefore belong to optimization/loss-balancing, not the binding Task Relation Learning category.

The closest explicit-relation paper, Rakitsch et al. (2013), learns separate signal and residual task covariance matrices. Its fully observed aligned Gaussian multi-output regression assumptions do not hold for disjoint KS/SI/ER datasets and heterogeneous multiclass heads. Porting only the idea would require a novel deep-classification hybrid, not a faithful published implementation.

No mechanism Study is authorized by LT-0001. The programme was left in `NEEDS-HUMAN-REVIEW`; that has since resolved as follows (2026-09-22):

* **DEC-0009 (human):** strict Task Relation Learning scope retained; broadening to optimization-aware MTL declined; a project-original hybrid deferred behind explicit authorization.
* **DG-0005 (CONFIRMED, F10):** the ER data-regime control was run. It showed the gradient-scale signal is a **training-mixture property**, so the F9-era rationale for a scale- or reliability-aware relation mechanism is withdrawn (DEC-0010).
* **DEC-0011:** this literature mandate is closed. Do not re-run this search against the same evidence.

Consequence for the assessed papers: the direct-scale methods (GradNorm, uncertainty weighting) remain excluded by taxonomy; the explicit-relation methods remain ineligible as published mechanisms; and Rakitsch et al.'s retained "diagnostic principle" is now less well supported than when LT-0001 closed it, because the measured signal turned out to be mixture-driven gradient-estimate scale rather than demonstrated residual/output noise. The open decisions are listed in `../FRAMEWORK.md` §6.
