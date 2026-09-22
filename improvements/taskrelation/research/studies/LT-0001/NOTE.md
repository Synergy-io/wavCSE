# LT-0001 Research Note

## Hypothesis

A published explicit Task Relation Learning method jointly represents task relations and unequal task reliability or optimization scale, maps directly to F9, and can be implemented faithfully for heterogeneous KS/SI/ER classification without category drift.

## Motivation

F9 confirms ER shared-gradient norm dominance across matched seeds and classical MTRL non-mitigation. Persistent pairwise conflict is absent. DEC-0007 therefore requires a targeted literature gate before any new mechanism.

## Method

Primary-source review of the formal task-relation taxonomy, covariance-based relation models, reliability/noise-aware relation models, direct gradient-scale methods, and recent relation-learning methods. Each paper is screened for explicit relation representation, direct F9 assumption match, heterogeneous classification applicability, faithful implementability, and category boundary.

## Execution ledger

This is a literature-only Study. It launches no training or MLflow run; the repository paper cards and Study analysis are the execution and interpretation record.

## Decision criterion

A candidate advances only if one published mechanism satisfies every eligibility condition in `PLAN.md`. An original combination of an explicit relation method with unrelated loss weighting or gradient surgery does not count as a published method.

## Result

Eight primary sources were screened. Explicit relation methods either lacked a direct optimization-scale/reliability mechanism or required aligned Gaussian/mean-estimation settings that do not admit a faithful KS/SI/ER classifier implementation. The methods that directly address F9—GradNorm and homoscedastic uncertainty weighting—learn per-task loss weights, not task relations. The 2024 informative-relation method relies on parameter decomposition, which crosses the project boundary.

## Interpretation

The literature supports a distinction among relation magnitude, relation confidence/noise, and optimization weight. Rakitsch et al.’s separate signal/noise task covariances are the closest relation-level principle, but F9 does not establish residual-noise causation and the paper’s aligned Gaussian regression assumptions do not hold here. Combining that principle with MTRL or GradNorm would be a project-original hybrid, not a faithful published method.

## Study decision

`REJECTED`. No mechanism is authorized. The candidate-method hypothesis failed every-paper eligibility checking; this is a negative literature result, not an architecture failure.

## Next step

Set the programme to `NEEDS-HUMAN-REVIEW`. The conservative option is to retain strict Task Relation Learning scope and run the ER data-regime/gradient-noise diagnostic next. Alternatives—broadening into optimization-aware MTL or authorizing an original relation-plus-reliability hybrid—require an explicit human scope decision.
