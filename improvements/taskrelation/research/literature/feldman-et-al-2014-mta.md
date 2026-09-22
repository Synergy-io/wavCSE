# Feldman, Gupta, and Frigyik (2014) — Multi-Task Averaging

## Citation

Sergey Feldman, Maya R. Gupta, and Bela A. Frigyik. “Revisiting Stein’s Paradox: Multi-Task Averaging.” *JMLR* 15:3621–3662, 2014.

Primary source: https://www.jmlr.org/papers/volume15/feldman14a/feldman14a.pdf

## Problem

Jointly estimate means for several tasks while accounting for task similarity, per-task variance, and unequal sample counts.

## Mathematical assumption

Each task supplies samples from a distribution with its own mean and variance. Empirical loss is normalized by task variance. A graph-Laplacian penalty ties task estimates according to a nonnegative task-similarity matrix. The benefit depends on mean separation relative to sample-mean variance.

## Relation representation

A symmetric nonnegative task-similarity matrix `A`; its graph Laplacian regularizes pairwise differences. Closed-form practical estimators infer similarity from observed task means. The solution explicitly includes sample-mean covariance `σ_t² / N_t`.

## Optimization method

Convex quadratic objective with a closed-form shrinkage estimator. The paper derives optimal two-task regularization and practical constant/minimax estimators for arbitrary task counts.

## Evidence

Theory, simulation, and four mean-estimation applications. Reported reductions in total squared error relative to sample means and James–Stein estimators when tasks are statistically close enough.

## Assumptions

Constant-function/mean estimation, scalar or vector observations, squared error, and nonnegative symmetric similarity. It is not a general classifier or deep parameter-relation model.

## Differences from our setting

The method is directly sample-size and variance aware, making it useful evidence that confidence and relation magnitude should be distinct. But its learned objects are task means—not heterogeneous classifier heads—and it does not control shared-gradient norms. Generalizing it to neural classification would require inventing a new estimator and relation target.

## Implementation difficulty

Low for mean estimation; high and scientifically undefined for KS/SI/ER deep classification.

## Candidate Study ID

`LT-0001` — adjacent evidence.

## LT-0001 decision

**REJECT AS MECHANISM CANDIDATE; RETAIN AS FRAMEWORK EVIDENCE.** It validates sample/variance-aware relation strength, but a faithful implementation does not answer the current classification problem.
