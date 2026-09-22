# LT-0001 Analysis — Scale- and reliability-aware task relation literature gate

Status: **REJECTED — no eligible published mechanism found; NEEDS-HUMAN-REVIEW**

Completed: 2026-09-22T09:15:21+00:00

## Question and evidence gate

F9 establishes reproducible ER shared-gradient norm dominance and MTRL non-mitigation. LT-0001 tested whether a published method simultaneously:

1. learns explicit quantitative task relations;
2. directly models unequal scale/reliability/noise relevant to F9;
3. remains Task Relation Learning under Zhang and Yang’s taxonomy;
4. applies faithfully to disjoint heterogeneous multiclass tasks;
5. preserves the fixed representation and avoids low-rank, clustering, decomposition, generic loss weighting, and gradient-surgery drift.

The candidate hypothesis required one paper to satisfy all five. None did.

## Search path

Queries were constructed from F9 rather than generic performance terms: task relationship/covariance combined with gradient magnitude, uncertainty, heteroscedasticity, residual/noise covariance, reliability, unequal sample size, and recent informative relations. Primary sources were read for the formal taxonomy, foundational Bayesian covariance learning, sparse relation learning, signal/noise covariance separation, sample-aware similarity, direct uncertainty/gradient balancing, and a 2024 relation method.

Paper cards and primary URLs are indexed in `literature/INDEX.md`.

## Eligibility matrix

| Method | Explicit learned relation | Directly addresses scale/reliability | Heterogeneous deep classification fit | Category clean | Decision |
| --- | --- | --- | --- | --- | --- |
| MTGTP | yes: Bayesian task covariance | robust likelihood/task noise, not gradient scale | no; GP regression/classification approximation required | yes | reject |
| GP-kronsum structured residuals | yes: signal and residual task covariances | yes conceptually: separates relation from noise | no; aligned fully observed Gaussian outputs required | yes | closest, but reject now |
| Multi-task averaging | yes: task similarity graph | yes: sample variance and count enter shrinkage | no; mean estimation only | yes | framework evidence only |
| SPATS | yes: sparse covariance | no; pair sparsity only | classification possible but three-task/F9 assumption mismatch | yes | reject |
| uncertainty weighting | no: per-task scalar only | yes: task likelihood scale | yes | no; loss weighting | exclude |
| GradNorm | no: per-task scalar only | yes: direct gradient magnitude balancing | yes | no; optimization/loss balancing | exclude |
| informative relation MTL | yes: covariance over shared component | no direct scale mechanism | substantial adaptation | no; defining `W=H+P` decomposition crosses project boundary | exclude |

## Strongest positive knowledge

The search does produce a useful modelling distinction:

- **relation strength** describes signal/parameter association;
- **reliability or scale** describes observation noise, sample confidence, or optimization influence.

Rakitsch et al. show in an explicit relation model that conflating signal covariance and residual covariance can falsely attribute hidden/noisy processes to useful task sharing. Feldman et al. show that task variance and sample count should alter the amount of shrinkage even when task similarity is held fixed. These principles align with F6/F9 and strengthen the emerging framework: relation magnitude is not relation confidence or optimization weight.

They do not authorize an implementation. Their mathematical settings do not match heterogeneous KS/SI/ER classification, and the project has not established whether ER’s smaller effective batch/label noise causes the scale signal.

## Direct symptom match is outside scope

GradNorm is the cleanest direct response to F9: it identifies dominant gradient magnitude as the root symptom and learns task weights to equalize training rates. Kendall et al. similarly learn task uncertainty scales. Neither learns pairwise relations. Under the binding taxonomy, implementing either would change the independent variable from Task Relation Learning to optimization/loss weighting. Combining either with Ω would be an original hybrid; no reviewed primary paper specifies that mechanism.

## Alternative explanations and missing evidence

F9’s scale imbalance may reflect:

- ER’s much smaller effective batch;
- label or measurement noise;
- task difficulty or class structure;
- cross-entropy calibration differences;
- stochastic gradient-estimate variance.

These causes imply different mechanisms. The closest literature candidate assumes residual/noise structure, but DG-0002 did not identify which cause dominates. A causal data-regime/noise diagnostic therefore remains higher-integrity than inventing a relation-plus-weighting hybrid.

## Study decision

**REJECTED.** The hypothesis that an immediately implementable published explicit Task Relation Learning method directly addresses F9 is falsified by the reviewed literature.

This is a negative literature result, not evidence that no such paper can ever exist. It is sufficient to block implementation now because every identified candidate violates at least one pre-registered eligibility gate.

## Programme consequence

No `TR-xxxx` or mechanism-screening run is authorized. Set the programme to `NEEDS-HUMAN-REVIEW`.

Human decision required:

1. **Retain strict Task Relation Learning scope:** return to diagnostics, with the ER data-regime/gradient-noise control as the highest-value causal question; or
2. **Broaden scope explicitly:** allow optimization-aware MTL such as GradNorm as a separate branch/control; or
3. **Authorize a novel hybrid contribution:** combine explicit task covariance with reliability/scale, accepting that it is project-original rather than a faithful published method.

Option 1 preserves the current contribution claim and is the conservative default, but beginning a new Study is deferred because this iteration has already created LT-0001 and the required programme decision is material.
