# Adaptive Dynamic-\(\tau\) Algorithm for RAMUSA-Based Multi-Task Learning

## 1. Objective

Modify the existing RAMUSA-based lower-rank Multi-Task Learning implementation so that the singular-value threshold

\[
\tau
\]

is **not fixed for the entire training process**.

Instead, determine an appropriate threshold dynamically at the end of every RAMUSA training stage using:

1. the current singular-value spectrum of the task-parameter matrix;
2. the separation or spectral gap between consecutive singular values;
3. validation performance of the candidate low-rank structures;
4. temporal smoothing to prevent unstable changes in \(\tau\).

The architecture contains exactly three speech tasks:

- Keyword Spotting — KS
- Speaker Identification — SID
- Emotion Recognition — ER

Intent Classification is not used.

The goal of the adaptive algorithm is to determine a suitable \(\tau^{(l)}\) for every RAMUSA stage \(l\), rather than requiring one manually selected global threshold.

---

# 2. Existing RAMUSA Architecture

The existing architecture should remain unchanged.

The common network produces a shared representation \(h\), after which three task-specific adapters are used:

\[
P_{KS},\qquad P_{SID},\qquad P_{ER}.
\]

Their weight matrices are vectorized and stacked to construct

\[
W_{\text{tasks}}
=
\left[
\operatorname{vec}(P_{KS}),
\operatorname{vec}(P_{SID}),
\operatorname{vec}(P_{ER})
\right].
\]

If each adapter contains \(D\) parameters,

\[
W_{\text{tasks}}\in\mathbb{R}^{D\times3}.
\]

Therefore,

\[
R=\min(D,3)=3.
\]

The matrix has at most three singular values:

\[
\sigma_1\geq\sigma_2\geq\sigma_3\geq0.
\]

The RAMUSA regularization used during a training stage remains

\[
\mathcal{R}_{RAMUSA}
=
\|W_{\text{tasks}}\|_{r-}
=
\sum_{i=r+1}^{R}\sigma_i(W_{\text{tasks}}).
\]

The complete training objective remains

\[
\mathcal{L}_{total}
=
\mathcal{L}_{MTL}
+
\lambda\mathcal{R}_{RAMUSA}.
\]

The adaptive-\(\tau\) algorithm must therefore **not change the model architecture or the RAMUSA loss formulation**.

It changes only the method used to determine \(r\) between stages.

---

# 3. Important Relationship Between \(\tau\) and \(r\)

In RAMUSA,

\[
r
=
\sum_{i=1}^{R}
I(\sigma_i\geq\tau).
\]

This means \(\tau\) itself is not directly present in the stage-specific loss.

Instead,

\[
\tau
\rightarrow
r
\rightarrow
\mathcal{R}_{RAMUSA}.
\]

For three tasks:

\[
r\in\{0,1,2,3\}.
\]

The interpretation is:

| \(r\) | Regularization | Meaning |
|---:|---|---|
| 0 | \(\sigma_1+\sigma_2+\sigma_3\) | Penalize all singular directions |
| 1 | \(\sigma_2+\sigma_3\) | Preserve strongest direction |
| 2 | \(\sigma_3\) | Preserve two strongest directions |
| 3 | \(0\) | No low-rank penalty |

Therefore, many different numerical values of \(\tau\) can result in exactly the same \(r\).

For example, if

\[
[\sigma_1,\sigma_2,\sigma_3]
=
[8.0,3.0,0.3],
\]

then every threshold satisfying

\[
3.0\geq\tau>0.3
\]

produces

\[
r=2.
\]

This is extremely important for the implementation.

Do **not** waste computation testing many threshold values that produce the same \(r\).

The dynamic algorithm should instead identify meaningful boundaries in the singular-value spectrum and select a \(\tau\) from those boundaries.

---

# 4. High-Level Adaptive Algorithm

Training should follow:

```text
Stage 1
    ↓
Train with r = 0
    ↓
Construct W_tasks
    ↓
Calculate singular values
    ↓
Find spectral gaps
    ↓
Generate candidate tau values
    ↓
Candidate tau values imply candidate r values
    ↓
Validate candidate low-rank structures
    ↓
Select best candidate
    ↓
Smooth tau
    ↓
Set tau and r for Stage 2
    ↓
Train Stage 2
    ↓
Repeat
```

The threshold must only change **between stages**.

Do not change \(\tau\) or \(r\):

- per batch;
- per optimizer step;
- per forward pass;
- in the middle of a stage.

---

# 5. Stage 1

Maintain the original RAMUSA behavior for the first stage.

Initialize

\[
r^{(0)}=0.
\]

Therefore Stage 1 uses

\[
\mathcal{R}^{(1)}
=
\sigma_1+\sigma_2+\sigma_3.
\]

The first stage therefore behaves like conventional trace-norm regularization.

There is no need to determine a dynamic \(\tau\) before Stage 1.

The first adaptive threshold is calculated **after Stage 1 has finished**.

---

# 6. Step 1 — Calculate the Singular-Value Spectrum

At the end of stage \(l\), construct the task-parameter matrix using the newly trained adapter parameters:

\[
W_{\text{tasks}}^{(l)}
=
[
\operatorname{vec}(P_{KS}^{(l)}),
\operatorname{vec}(P_{SID}^{(l)}),
\operatorname{vec}(P_{ER}^{(l)})
].
\]

Calculate:

\[
[\sigma_1^{(l)},\sigma_2^{(l)},\sigma_3^{(l)}]
=
\operatorname{SVDValues}
(W_{\text{tasks}}^{(l)}).
\]

Use:

```python
with torch.no_grad():
    W_tasks = model.build_ramusa_parameter_matrix()
    singular_values = torch.linalg.svdvals(W_tasks.float())
```

The returned singular values should already be ordered:

```text
sigma_1 >= sigma_2 >= sigma_3
```

Assert this or sort defensively if necessary.

---

# 7. Step 2 — Calculate Relative Spectral Gaps

The goal is to identify whether the singular values naturally separate into:

```text
dominant directions | weak directions
```

For three singular values calculate two relative gaps:

\[
g_1
=
\frac{\sigma_1-\sigma_2}
{\sigma_1+\epsilon}
\]

and

\[
g_2
=
\frac{\sigma_2-\sigma_3}
{\sigma_2+\epsilon}.
\]

Use a very small numerical constant such as

\[
\epsilon=10^{-8}.
\]

In code:

```python
eps = 1e-8

g1 = (s1 - s2) / (s1 + eps)
g2 = (s2 - s3) / (s2 + eps)
```

The use of a relative gap rather than

```text
sigma_i - sigma_i+1
```

is intentional.

Absolute singular-value magnitudes may change during training.

For example:

```text
Stage A: [10.0, 5.0, 1.0]
Stage B: [2.0, 1.0, 0.2]
```

The absolute magnitudes differ significantly, but their relative structures are similar.

Relative gaps make the threshold-selection mechanism less sensitive to this scaling.

---

# 8. Step 3 — Interpret the Spectral Gaps

Suppose

\[
\sigma=[8.0,3.0,0.3].
\]

Then

\[
g_1
=
\frac{8-3}{8}
=
0.625
\]

and

\[
g_2
=
\frac{3-0.3}{3}
=
0.9.
\]

Because

\[
g_2>g_1,
\]

the strongest natural separation occurs between

\[
\sigma_2
\]

and

\[
\sigma_3.
\]

This suggests:

\[
\sigma_1,\sigma_2
\]

represent dominant parameter-sharing directions, while

\[
\sigma_3
\]

is comparatively weak.

Therefore the natural effective rank is approximately

\[
r=2.
\]

---

# 9. Step 4 — Generate Candidate Thresholds

For every meaningful gap, generate a threshold located between the two singular values.

Use the geometric mean:

\[
\tau_k
=
\sqrt{
\sigma_k\sigma_{k+1}
}.
\]

For the boundary between \(\sigma_1\) and \(\sigma_2\):

\[
\tau_1
=
\sqrt{\sigma_1\sigma_2}.
\]

This produces

\[
r=1.
\]

For the boundary between \(\sigma_2\) and \(\sigma_3\):

\[
\tau_2
=
\sqrt{\sigma_2\sigma_3}.
\]

This produces

\[
r=2.
\]

## Why use the geometric mean?

Singular values can differ by orders of magnitude.

Suppose

\[
\sigma_2=10,\qquad\sigma_3=0.1.
\]

The arithmetic midpoint would be:

\[
5.05.
\]

The geometric midpoint is:

\[
\sqrt{10\times0.1}=1.
\]

The geometric mean is a more natural midpoint when dealing with multiplicative scale differences and low-rank spectra.

---

# 10. Minimum Spectral-Gap Safeguard

Do not assume every spectrum contains a meaningful low-rank separation.

For example:

\[
[4.8,4.3,3.9]
\]

contains relatively similar singular values.

There is no strong evidence that one direction should suddenly be classified as weak.

Introduce:

```yaml
min_spectral_gap: 0.20
```

or another configurable value.

Denote this by

\[
\gamma.
\]

A candidate boundary should normally only be considered if

\[
g_k\geq\gamma.
\]

For example:

```python
if gap >= min_spectral_gap:
    candidate_is_valid = True
```

Do not hard-code \(0.20\). Make it configurable.

A reasonable initial value is:

```text
0.15 – 0.25
```

---

# 11. Candidate Set for Three Tasks

Because the model has exactly three tasks, do not perform an unrestricted continuous search over \(\tau\).

The meaningful candidates are normally:

### Candidate A — effective rank 1

\[
\tau_1
=
\sqrt{\sigma_1\sigma_2}
\]

which gives:

\[
r=1.
\]

### Candidate B — effective rank 2

\[
\tau_2
=
\sqrt{\sigma_2\sigma_3}
\]

which gives:

\[
r=2.
\]

### Candidate C — previous configuration

Always retain the previous stage's

\[
(\tau^{(l-1)},r^{(l-1)})
\]

as a fallback candidate.

This prevents the model from being forced into a different rank when the new spectral evidence is weak.

---

# 12. Treatment of \(r=0\) and \(r=3\)

By default, do not actively generate new candidate thresholds for \(r=0\) or \(r=3\).

### \(r=0\)

This means

\[
\tau>\sigma_1
\]

and causes all singular values to remain penalized.

Stage 1 already uses \(r=0\).

Allow later stages to remain at \(r=0\) only if:

- it was the previous selected configuration; or
- configuration explicitly allows it.

### \(r=3\)

This means

\[
\tau\leq\sigma_3
\]

and gives

\[
\mathcal{R}_{RAMUSA}=0.
\]

RAMUSA would effectively stop providing low-rank regularization.

Do not automatically select \(r=3\) merely because all singular values become large.

By default:

```yaml
allow_zero_penalty_rank: false
```

Only allow \(r=3\) if explicitly enabled for experimentation.

---

# 13. Why Validation Is Required

The largest spectral gap tells us:

> Which separation looks strongest mathematically?

It does **not** necessarily tell us:

> Which separation produces the best KS, SID and ER performance?

For example:

```text
sigma = [8.0, 3.0, 0.3]
```

suggests \(r=2\).

However, preserving two parameter directions may improve KS and SID while hurting ER.

Therefore the final candidate must also be checked against validation performance.

The dynamic algorithm should combine:

```text
spectral evidence
+
validation evidence
```

rather than relying on singular values alone.

---

# 14. Candidate Validation Probe

At the end of a normal stage, save a temporary stage-boundary state containing:

```text
model weights
optimizer state
scheduler state
current stage
current r
current tau
random states if possible
```

Then evaluate each unique candidate rank.

For each candidate:

1. restore exactly the same stage-boundary model;
2. set the candidate \(r\);
3. run a very short training probe;
4. evaluate KS, SID and ER on the validation set;
5. record the results;
6. discard the probe model;
7. restore the original stage-boundary checkpoint before testing the next candidate.

Example:

```text
End of Stage 2
      ↓
Save temporary base state
      ↓
Candidate r = 1
      ↓
Probe for 1 epoch
      ↓
Validation KS/SID/ER
      ↓
Restore base state
      ↓
Candidate r = 2
      ↓
Probe for 1 epoch
      ↓
Validation KS/SID/ER
      ↓
Compare
```

The candidate probes must not permanently alter the actual model.

After selecting the best candidate:

```text
restore base state
→ set selected tau
→ set selected r
→ begin real next stage
```

---

# 15. Candidate Probe Configuration

Add configuration similar to:

```yaml
ramusa:
  adaptive_tau:
    enabled: true

    min_spectral_gap: 0.20

    smoothing_beta: 0.70

    validation_probe:
      enabled: true
      epochs: 1

    allow_rank_zero_after_stage1: true
    allow_zero_penalty_rank: false

    min_validation_gain: 0.0

    epsilon: 1.0e-8
```

Do not hard-code these values.

---

# 16. Avoid Duplicate Candidate Evaluation

Different numerical threshold values can produce exactly the same \(r\).

If two candidates produce

\[
r=2,
\]

their next-stage RAMUSA regularizer is identical:

\[
\mathcal{R}
=
\sigma_3.
\]

Therefore they should not both be probed.

Generate candidates and then deduplicate them by effective rank.

For example:

```python
candidates_by_rank = {}

for tau_candidate in tau_candidates:
    r_candidate = compute_r(singular_values, tau_candidate)

    if r_candidate not in candidates_by_rank:
        candidates_by_rank[r_candidate] = tau_candidate
```

This is especially important because with three tasks there are only a few possible structures.

---

# 17. Validation Scoring

Prefer to reuse an existing project-level aggregate validation metric if the codebase already provides one.

Do not unnecessarily introduce a new metric if KS, SID and ER are already combined consistently elsewhere.

If no suitable aggregate score exists, create a normalized validation improvement score.

Let the stage-boundary baseline metrics be

\[
M_t^{base}
\]

and the candidate probe metrics be

\[
M_t^{candidate}.
\]

For a higher-is-better metric:

\[
\Delta_t
=
\frac{
M_t^{candidate}-M_t^{base}
}{
|M_t^{base}|+\epsilon
}.
\]

For a lower-is-better metric such as error:

\[
\Delta_t
=
\frac{
M_t^{base}-M_t^{candidate}
}{
|M_t^{base}|+\epsilon
}.
\]

Then calculate

\[
S_{avg}
=
\frac{
\sum_t\omega_t\Delta_t
}{
\sum_t\omega_t
}.
\]

Here

\[
t\in\{KS,SID,ER\}.
\]

---

# 18. Negative-Transfer Safeguard

A candidate should not be selected simply because two tasks improve strongly while the third collapses.

Calculate:

\[
\Delta_{\min}
=
\min(
\Delta_{KS},
\Delta_{SID},
\Delta_{ER}
).
\]

Optionally calculate:

\[
S
=
S_{avg}
+
\eta\min(0,\Delta_{\min}),
\]

where

\[
\eta\geq0
\]

controls how strongly negative transfer is penalized.

For example:

```yaml
negative_transfer_penalty: 0.5
```

Again, reuse the project's existing multi-task model-selection criterion if one already exists.

Do not create an incompatible scoring method unnecessarily.

---

# 19. Candidate Selection

Select

\[
c^*
=
\arg\max_c S(c).
\]

The chosen candidate determines:

\[
r^*
\]

and a raw threshold

\[
\tau_{\text{raw}}.
\]

If two candidates have nearly identical validation scores, use these tie-breakers in order:

1. candidate with stronger spectral gap;
2. candidate closer to the previous \(r\);
3. candidate requiring the smaller change in \(\tau\).

This reduces unnecessary oscillations.

---

# 20. Minimum Validation-Gain Safeguard

A dynamically selected rank should not replace the previous stage configuration when it produces no meaningful improvement.

Introduce:

\[
\delta_{val}
\]

as a configurable minimum validation gain.

Example:

```yaml
min_validation_gain: 0.001
```

If

\[
S(c^*)-S(previous)<\delta_{val},
\]

keep:

\[
\tau^{(l)}
=
\tau^{(l-1)}
\]

and

\[
r^{(l)}
=
r^{(l-1)}.
\]

This provides hysteresis and reduces stage-to-stage instability.

---

# 21. Step 5 — Smooth the Selected Threshold

Do not immediately replace the old threshold with the raw new threshold.

Apply exponential moving average smoothing:

\[
\tilde{\tau}^{(l)}
=
\beta\tau^{(l-1)}
+
(1-\beta)\tau_{\text{raw}}^{(l)}.
\]

Recommended initial value:

\[
\beta=0.7.
\]

Interpretation:

- large \(\beta\) → more stable but slower adaptation;
- small \(\beta\) → faster but potentially unstable adaptation.

Make \(\beta\) configurable.

---

# 22. Critical Safeguard: Smoothing Must Preserve the Selected \(r\)

This is extremely important.

Suppose validation selected:

\[
r=2.
\]

For \(r=2\) to remain valid, the threshold must satisfy:

\[
\sigma_2\geq\tau>\sigma_3.
\]

EMA smoothing could accidentally move \(\tau\) outside that interval.

For example:

```text
selected raw tau = 0.8
previous tau = 4.0
beta = 0.7
```

The EMA could produce a threshold larger than \(\sigma_2\), accidentally changing \(r\) from 2 to 1.

This must not happen.

After smoothing, constrain the threshold to the interval corresponding to the selected rank.

For \(r=1\):

\[
\sigma_1\geq\tau>\sigma_2.
\]

For \(r=2\):

\[
\sigma_2\geq\tau>\sigma_3.
\]

Therefore:

```python
tau_smoothed = (
    beta * previous_tau
    + (1.0 - beta) * raw_tau
)

tau_final = clamp_tau_to_rank_interval(
    tau=tau_smoothed,
    singular_values=singular_values,
    target_r=selected_r,
)
```

After clamping, verify:

```python
assert compute_r(singular_values, tau_final) == selected_r
```

If this assertion fails because the singular values are almost identical, use the raw geometric-mean threshold instead.

---

# 23. Rank-Preserving Threshold Interval

For target rank \(r\), the valid interval is approximately:

\[
\sigma_r
\geq
\tau
>
\sigma_{r+1}.
\]

Use a very small safety margin so \(\tau\) does not exactly sit on an unstable boundary.

For example:

```python
width = upper - lower
margin = max(eps, 0.01 * width)

safe_lower = lower + margin
safe_upper = upper - margin
```

Then:

```python
tau = min(max(tau, safe_lower), safe_upper)
```

If:

```text
safe_lower >= safe_upper
```

because two singular values are almost equal, do not force a new threshold across that boundary.

Use the previous configuration instead.

---

# 24. Complete Stage Algorithm

The full training procedure should conceptually be:

```python
r = 0
tau = None

for stage in range(num_stages):

    # --------------------------------------
    # NORMAL RAMUSA TRAINING FOR THIS STAGE
    # --------------------------------------

    # r remains fixed throughout the stage.

    train_stage(
        model=model,
        r=r,
        ...
    )

    # Last stage does not require selecting
    # parameters for another stage.
    if stage == num_stages - 1:
        break

    # --------------------------------------
    # COMPUTE CURRENT SINGULAR SPECTRUM
    # --------------------------------------

    with torch.no_grad():
        W_tasks = model.build_ramusa_parameter_matrix()
        singular_values = torch.linalg.svdvals(
            W_tasks.float()
        )

    # --------------------------------------
    # COMPUTE SPECTRAL GAPS
    # --------------------------------------

    gaps = compute_relative_spectral_gaps(
        singular_values
    )

    # --------------------------------------
    # GENERATE UNIQUE CANDIDATE RANKS
    # --------------------------------------

    candidates = generate_dynamic_tau_candidates(
        singular_values=singular_values,
        gaps=gaps,
        previous_tau=tau,
        previous_r=r,
        min_gap=min_spectral_gap,
    )

    # --------------------------------------
    # VALIDATION PROBES
    # --------------------------------------

    if validation_probe_enabled:

        results = probe_candidates(
            base_state=current_stage_state,
            candidates=candidates,
        )

        selected = select_best_candidate(
            results
        )

    else:

        selected = select_largest_gap_candidate(
            candidates
        )

    # --------------------------------------
    # FALLBACK CHECK
    # --------------------------------------

    if selected.improvement < min_validation_gain:
        selected_r = r
        selected_tau = tau

    else:
        selected_r = selected.r
        selected_tau = selected.tau

    # --------------------------------------
    # SMOOTH TAU
    # --------------------------------------

    if tau is not None:
        smoothed_tau = (
            beta * tau
            + (1.0 - beta) * selected_tau
        )
    else:
        smoothed_tau = selected_tau

    # --------------------------------------
    # PRESERVE SELECTED R
    # --------------------------------------

    final_tau = clamp_tau_to_rank_interval(
        tau=smoothed_tau,
        singular_values=singular_values,
        target_r=selected_r,
    )

    final_r = compute_r(
        singular_values,
        final_tau,
    )

    assert final_r == selected_r

    tau = final_tau
    r = final_r
```

Adapt this structure to the existing trainer rather than replacing the project's trainer architecture.

---

# 25. Pure Spectral Fallback

The validation probe may be computationally expensive.

Provide a configurable fallback mode.

If:

```yaml
validation_probe:
  enabled: false
```

then use:

\[
k^*
=
\arg\max_k g_k.
\]

If

\[
g_{k^*}\geq\gamma,
\]

set:

\[
\tau_{\text{raw}}
=
\sqrt{
\sigma_{k^*}
\sigma_{k^*+1}
}
\]

and

\[
r=k^*.
\]

Then perform EMA smoothing and rank-preserving clamping.

If

\[
g_{k^*}<\gamma,
\]

keep the previous threshold and rank.

This produces a lightweight algorithm:

```text
SVD
 ↓
Largest meaningful spectral gap
 ↓
Geometric-mean threshold
 ↓
EMA smoothing
 ↓
Rank-preserving clamp
 ↓
Next stage
```

The validation-guided version should be preferred when computationally feasible.

---

# 26. Example 1 — Clear Rank-2 Structure

Suppose after Stage 1:

\[
\sigma
=
[8.0,3.0,0.3].
\]

Calculate:

\[
g_1=0.625
\]

and

\[
g_2=0.90.
\]

Candidate 1:

\[
\tau_1
=
\sqrt{8\times3}
\approx4.90
\]

giving

\[
r=1.
\]

Candidate 2:

\[
\tau_2
=
\sqrt{3\times0.3}
\approx0.95
\]

giving

\[
r=2.
\]

Suppose the validation probes produce:

| Candidate | \(r\) | KS | SID | ER | Overall |
|---|---:|---:|---:|---:|---:|
| A | 1 | 91.2 | 86.1 | 67.4 | 81.6 |
| B | 2 | 92.0 | 86.5 | 70.1 | 82.9 |

Select:

\[
r=2.
\]

Raw threshold:

\[
\tau_{\text{raw}}\approx0.95.
\]

After smoothing and rank-preserving clipping, use the resulting threshold for deciding the next stage's \(r\).

Stage 2 then penalizes:

\[
\mathcal{R}^{(2)}
=
\sigma_3.
\]

---

# 27. Example 2 — No Clear Spectral Structure

Suppose:

\[
\sigma
=
[4.8,4.3,3.9].
\]

Then approximately:

\[
g_1=0.104
\]

and

\[
g_2=0.093.
\]

If:

\[
\gamma=0.20,
\]

neither boundary is considered strong.

Therefore do not dynamically invent a new low-rank structure.

Prefer:

\[
\tau^{(l)}
=
\tau^{(l-1)}
\]

and

\[
r^{(l)}
=
r^{(l-1)}.
\]

This prevents unstable rank decisions.

---

# 28. Example 3 — Spectrum Changes Across Stages

Suppose:

```text
Stage 1:
[7.5, 3.2, 1.8]

Stage 2:
[8.1, 3.0, 0.7]

Stage 3:
[8.5, 2.9, 0.15]
```

The separation between

\[
\sigma_2
\]

and

\[
\sigma_3
\]

becomes progressively stronger.

The adaptive method can therefore gradually become more confident that:

\[
r=2
\]

is the appropriate effective rank.

This is the intended purpose of dynamic \(\tau\): the rank decision adapts to the parameter structure that actually emerges during training.

---

# 29. Required Helper Functions

Keep the implementation modular.

Prefer utilities similar to:

```python
def compute_relative_spectral_gaps(
    singular_values,
    eps=1e-8,
):
    ...
```

```python
def threshold_between(
    upper_sigma,
    lower_sigma,
):
    ...
```

```python
def compute_effective_rank(
    singular_values,
    tau,
):
    ...
```

```python
def generate_adaptive_tau_candidates(
    singular_values,
    previous_tau,
    previous_r,
    min_gap,
):
    ...
```

```python
def clamp_tau_to_rank_interval(
    tau,
    singular_values,
    target_r,
    eps=1e-8,
):
    ...
```

```python
def select_best_tau_candidate(
    candidate_results,
):
    ...
```

Do not place all adaptive-\(\tau\) logic directly inside the primary training loop.

---

# 30. Candidate Data Structure

Use a small structured object such as a dataclass.

Example:

```python
@dataclass
class TauCandidate:
    tau: float
    effective_rank: int
    spectral_gap: float
    source: str

    validation_metrics: dict | None = None
    validation_score: float | None = None
```

Possible `source` values:

```text
"spectral_gap_1"
"spectral_gap_2"
"previous_stage"
```

This will make debugging and experiment analysis easier.

---

# 31. State Management During Probe Training

Candidate validation must be fair.

Every candidate must begin from exactly the same model state.

Before candidate probing, preserve:

```text
model.state_dict()
optimizer.state_dict()
scheduler.state_dict()
AMP scaler state
Python random state
NumPy random state
PyTorch random state
CUDA RNG state
```

where applicable.

Do not run:

```text
candidate A
→ candidate B continuing from A
```

because Candidate B would then receive additional training and the comparison would be invalid.

Instead:

```text
BASE
├── Candidate A
├── Candidate B
└── Previous Configuration
```

Every branch starts from `BASE`.

---

# 32. Scheduler Considerations

If a learning-rate scheduler is used, restore its state for every validation probe.

Candidate probes must not permanently advance the real training scheduler.

After selecting a candidate:

1. restore the original end-of-stage state;
2. set the selected \(r\) and \(\tau\);
3. begin the actual next stage.

---

# 33. Logging Requirements

Log the following after every stage:

```text
stage
lambda
previous_tau
previous_r

sigma_1
sigma_2
sigma_3

gap_1
gap_2

candidate tau values
candidate effective ranks
candidate spectral gaps

candidate KS validation metric
candidate SID validation metric
candidate ER validation metric
candidate aggregate validation score

selected raw tau
selected r

smoothed tau
final tau
final r
```

Example:

```text
================================================
Adaptive RAMUSA Stage 2 Boundary
================================================

Singular values:
[8.032, 2.961, 0.354]

Relative gaps:
g1 = 0.631
g2 = 0.880

Candidates:

r=1
tau=4.876
gap=0.631
validation score=0.814

r=2
tau=1.024
gap=0.880
validation score=0.837

Previous:
r=1
tau=4.502
validation score=0.818

Selected:
r=2
raw tau=1.024

EMA tau:
2.067

Rank-preserving corrected tau:
1.024

Final:
tau=1.024
r=2

Stage 3 will penalize:
sigma_3
================================================
```

---

# 34. Store History for Later Analysis

Maintain stage-level history:

```python
adaptive_tau_history = [
    {
        "stage": 1,
        "singular_values": [...],
        "gaps": [...],
        "selected_tau": ...,
        "selected_r": ...,
        "validation_score": ...,
    },
]
```

Save this history with the experiment results or checkpoint.

This is important because later the project should be able to plot:

\[
\tau^{(l)}
\]

against stage,

\[
r^{(l)}
\]

against stage,

and each singular value:

\[
\sigma_1^{(l)},\sigma_2^{(l)},\sigma_3^{(l)}.
\]

These plots can help demonstrate whether the adaptive threshold is producing meaningful low-rank behavior.

---

# 35. Checkpointing

The RAMUSA checkpoint must additionally store:

```text
current stage
current r
current tau

adaptive tau enabled/disabled

tau history
r history
singular value history
spectral gap history
```

When resuming training, do not recompute the threshold from scratch.

Restore:

\[
\tau
\]

and

\[
r
\]

from the checkpoint.

---

# 36. Numerical Stability

Use:

```python
torch.linalg.svdvals()
```

for singular values.

For stage-boundary analysis gradients are unnecessary:

```python
with torch.no_grad():
```

may be used.

If mixed precision causes unstable SVD behavior, perform:

```python
W_for_svd = W_tasks.float()
```

before calculating the stage-boundary spectrum.

The training RAMUSA regularizer itself must remain differentiable and must not detach the adapter parameters.

---

# 37. Unit Tests

Implement tests for at least the following cases.

### Test 1 — clear \(r=2\) spectrum

Input:

```text
[8.0, 3.0, 0.3]
```

Expected:

```text
gap_2 > gap_1
```

and the strongest spectral candidate should correspond to:

\[
r=2.
\]

### Test 2 — clear \(r=1\) spectrum

Input:

```text
[10.0, 1.0, 0.8]
```

Expected:

```text
gap_1 > gap_2
```

and the strongest candidate should correspond to:

\[
r=1.
\]

### Test 3 — no meaningful gap

Input:

```text
[4.8, 4.3, 3.9]
```

with:

```text
min_gap = 0.20
```

Expected:

```text
no new spectral candidate
```

and the previous configuration should be retained.

### Test 4 — geometric threshold

For:

```text
sigma_2 = 3.0
sigma_3 = 0.3
```

verify:

\[
\tau
=
\sqrt{0.9}.
\]

### Test 5 — rank calculation

For:

```text
singular_values = [8.0, 3.0, 0.3]
tau = 1.0
```

verify:

```text
r == 2
```

### Test 6 — smoothing preserves selected rank

Construct a case where normal EMA would cross a singular-value boundary.

Verify that rank-preserving clamping keeps the selected \(r\).

### Test 7 — candidate deduplication

Two threshold candidates producing the same \(r\) must result in only one validation probe.

### Test 8 — candidate probe state restoration

Verify that evaluating Candidate A does not modify the starting state used for Candidate B.

### Test 9 — no test-set usage

Threshold selection must depend only on:

```text
training data
validation data
current model parameters
```

and never on test performance.

---

# 38. Ablation Compatibility

The code must support both:

```yaml
adaptive_tau:
  enabled: false
```

and

```yaml
adaptive_tau:
  enabled: true
```

When disabled, reproduce the existing fixed-\(\tau\) RAMUSA behavior exactly.

This is necessary for comparing:

```text
Original fixed-tau RAMUSA adaptation
vs
Dynamic-tau RAMUSA extension
```

Do not remove the existing fixed-\(\tau\) implementation.

---

# 39. Recommended Experimental Comparison

The final system should allow experiments such as:

```text
Experiment A:
RAMUSA with fixed tau

Experiment B:
RAMUSA with adaptive spectral tau

Experiment C:
RAMUSA with adaptive spectral + validation tau
```

This allows us to determine whether dynamic threshold selection actually improves:

- KS;
- SID;
- ER;
- average MTL performance;
- stability;
- effective rank;
- convergence.

---

# 40. Important Research Constraint

The original RAMUSA method uses a fixed threshold \(\tau\).

This adaptive threshold mechanism is a new extension for this project.

Therefore comments, documentation and experiment names should use wording such as:

```text
Adaptive-tau RAMUSA
Dynamic-threshold RAMUSA
RAMUSA-based adaptive reduced-rank MTL
```

Do not describe the dynamic-\(\tau\) mechanism as being directly proposed by Han and Zhang.

Also do not claim that the original RAMUSA theoretical convergence or estimation-error guarantees automatically apply to this modified algorithm.

---

# 41. Final Algorithm Summary

The final adaptive algorithm is:

\[
\boxed{
\text{Train Stage }l
}
\]

\[
\downarrow
\]

\[
W_{\text{tasks}}^{(l)}
\]

\[
\downarrow
\]

\[
\sigma_1^{(l)}
\geq
\sigma_2^{(l)}
\geq
\sigma_3^{(l)}
\]

\[
\downarrow
\]

Calculate

\[
g_k
=
\frac{
\sigma_k-\sigma_{k+1}
}{
\sigma_k+\epsilon
}
\]

\[
\downarrow
\]

Generate meaningful candidate boundaries

\[
\tau_k
=
\sqrt{
\sigma_k\sigma_{k+1}
}
\]

\[
\downarrow
\]

Each candidate determines an effective rank

\[
r_k
=
\sum_iI(\sigma_i\geq\tau_k)
\]

\[
\downarrow
\]

Probe unique candidate ranks using validation data

\[
\downarrow
\]

Choose best validation-supported low-rank structure

\[
\downarrow
\]

Smooth threshold

\[
\tilde\tau^{(l)}
=
\beta\tau^{(l-1)}
+
(1-\beta)\tau_{\text{raw}}^{(l)}
\]

\[
\downarrow
\]

Clamp the threshold so that the selected \(r\) is preserved

\[
\downarrow
\]

Set

\[
\tau^{(l)},r^{(l)}
\]

for the next stage

\[
\downarrow
\]

Continue training.

---

# 42. Core Principle

The algorithm should not simply ask:

> What numerical value of \(\tau\) looks good?

Instead, it should ask:

> How many dominant parameter-sharing directions are currently supported by the learned task-parameter matrix, and which of those candidate structures produces the best validation behavior across KS, SID and ER?

The singular-value spectrum identifies the plausible low-rank structures.

The validation set determines which plausible structure actually benefits the speech tasks.

EMA smoothing and rank-preserving constraints prevent unstable threshold changes.

This combination should form the complete adaptive-\(\tau\) mechanism for the RAMUSA-based lower-rank architecture.
