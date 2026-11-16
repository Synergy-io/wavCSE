# Codex Implementation Prompt — RAMUSA / Reduced-Rank Multi-Task Learning

## Goal

Implement the proposed **lower-rank multi-task learning architecture** for the speech project using **Python and PyTorch**, based on the RAMUSA concept from:

> L. Han and Y. Zhang, “Multi-Stage Multi-Task Learning with Reduced Rank,” AAAI, 2016.

The implementation must support exactly these three tasks:

- Keyword Spotting (KS)
- Speaker Identification (SID)
- Emotion Recognition (ER)

**Intent Classification (IC) is not part of this implementation and must not be added.**

A completed implementation of the project’s **task-clustering approach** will be provided in this repository. Before writing code, inspect that implementation carefully and use it as the structural reference. Follow the same project organization, module boundaries, configuration style, dataset interfaces, trainer structure, logging, checkpointing, evaluation utilities, naming conventions, and command-line entry points wherever possible. Do not redesign the repository unnecessarily.

---

## 1. First inspect the existing task-clustering implementation

Before modifying anything:

1. Identify the files/modules used by the task-clustering implementation for:
   - model definition;
   - shared upstream/backbone;
   - FC1, layer-wise pooling, and FC2;
   - task-specific branches/heads;
   - multi-task loss calculation;
   - training loop;
   - configuration/hyperparameters;
   - dataset loaders for KS, SID, and ER;
   - validation/testing;
   - logging and checkpointing.
2. Reuse the same interfaces and overall execution flow.
3. Add a separate lower-rank/RAMUSA implementation rather than breaking the task-clustering implementation.
4. Keep shared/common code shared. Do not duplicate dataset or upstream code unless the existing repository structure requires it.
5. If the current module names differ from the names used in this prompt, map the concepts to the existing modules instead of renaming the whole project.

Before implementing, briefly report which existing files will be reused, which files will be modified, and which new RAMUSA-specific files/modules will be added.

---

## 2. Architecture to implement

Preserve the project’s existing common speech-processing pipeline up to FC2. Conceptually, the forward path is:

```text
Audio
  -> Existing shared upstream/backbone
  -> Fully Connected 1 (FC1)
  -> Existing layer-wise pooling
  -> Fully Connected 2 (FC2)
  -> shared representation h
```

Do not change the upstream/backbone behavior used by the task-clustering implementation. Preserve its frozen/trainable status exactly unless the existing configuration explicitly allows otherwise.

From the shared FC2 representation `h`, create three task-specific branches:

```text
h
 ├─> KS adapter  P_KS  -> KS head  -> KS logits
 ├─> SID adapter P_SID -> SID head -> SID logits
 └─> ER adapter  P_ER  -> ER head  -> ER logits
```

The adapters are the parameters on which RAMUSA reduced-rank sharing is applied.

### Task-specific adapters

Create one identically structured trainable adapter for each task:

- `P_KS`
- `P_SID`
- `P_ER`

Each adapter takes the FC2 shared representation and produces a task-specialized representation for the corresponding prediction head.

All three adapter **weight matrices must have exactly the same shape**. This is required because their flattened weights will become columns of a single task-parameter matrix.

The adapter dimension must be configurable and should be derived from the existing FC2 output dimension and task-head input requirements. Do not hard-code dimensions that already exist in project configuration.

Use the same initialization rule for all three adapters, using the project’s standard PyTorch initialization where possible.

Unless the existing project design requires otherwise, apply RAMUSA regularization to the **adapter weight matrices only**, not to:

- adapter biases;
- FC1/FC2 shared weights;
- task classifier/head weights;
- upstream parameters.

Keep task heads independent because KS, SID, and ER have different output label spaces.

---

## 3. RAMUSA task-parameter matrix

At training time, flatten each adapter weight matrix into a vector:

\[
p_t = \operatorname{vec}(P_t), \qquad t \in \{KS,SID,ER\}.
\]

Construct the task-parameter matrix by stacking the three vectors as columns:

\[
W_{\text{tasks}}
=
[\operatorname{vec}(P_{KS}),
 \operatorname{vec}(P_{SID}),
 \operatorname{vec}(P_{ER})].
\]

In PyTorch, if each flattened adapter contains `D` parameters, the required shape is:

```text
W_tasks.shape == [D, 3]
```

Do not detach `W_tasks` when it is used to compute the training regularizer. Gradients from the regularization term must flow back to the three adapter weights.

Because there are three tasks:

\[
R = \min(D,3) = 3
\]

for the normal case where `D >= 3`. Therefore there are at most three singular values to monitor.

---

## 4. Singular-value computation

Compute the singular values of `W_tasks` using a differentiable PyTorch operation such as:

```python
singular_values = torch.linalg.svdvals(W_tasks)
```

PyTorch returns the singular values in descending order.

Do not manually set singular values to zero and reconstruct the adapters during the normal forward path. The reduced-rank effect must be introduced through the regularization objective and parameter optimization.

The singular values should be logged during training so that the evolution of the learned rank structure can be inspected.

---

## 5. RAMUSA objective

The conceptual capped trace-norm objective from RAMUSA is:

\[
\mathcal{R}_{\text{capped}}
=
\sum_{i=1}^{R}
\min(\sigma_i(W_{\text{tasks}}),\tau),
\]

where:

- `sigma_i` is the i-th singular value of `W_tasks`;
- `tau` is a fixed threshold selected before a training run;
- `R = min(D, number_of_tasks)`.

However, the architecture must follow the **multi-stage RAMUSA procedure** rather than simply applying the capped expression as an ordinary per-batch penalty for the whole run.

---

## 6. Multi-stage training logic

Training is organized into configurable stages.

Required configuration parameters should include at least:

```text
ramusa.enabled
ramusa.lambda
ramusa.tau
ramusa.num_stages
ramusa.epochs_per_stage
```

Use naming consistent with the existing configuration system.

### Initial state

Before Stage 1:

```text
r = 0
```

`tau` is fixed for the entire training run. Do **not** update or learn `tau` during training unless a future optional experiment explicitly adds that behavior.

### Stage 1

With `r = 0`, the stage-specific truncated trace-norm penalty becomes the ordinary trace/nuclear norm:

\[
\|W_{\text{tasks}}\|_{0-}
=
\sum_{i=1}^{R}\sigma_i(W_{\text{tasks}}).
\]

Therefore Stage 1 penalizes all singular values.

### End of each stage

After completing the training epochs for a stage:

1. Rebuild `W_tasks` from the **current updated adapter weights**.
2. Compute its singular values.
3. Update the number of dominant singular values using the fixed threshold `tau`:

\[
r
=
\sum_{i=1}^{R}
I(\sigma_i(\hat W_{\text{tasks}}) \ge \tau).
\]

4. Store/log the new `r`.
5. Freeze this `r` value for the next training stage.

### Stage 2 and later

For a stage whose fixed value is `r`, use the truncated trace-norm penalty:

\[
\|W_{\text{tasks}}\|_{r-}
=
\sum_{i=r+1}^{R}\sigma_i(W_{\text{tasks}}).
\]

Since the singular values are ordered from largest to smallest, this means:

- the first `r` dominant singular directions are not included in the stage penalty;
- only the remaining smaller singular values are penalized.

In code, conceptually:

```python
singular_values = torch.linalg.svdvals(W_tasks)
rank_penalty = singular_values[r:].sum()
```

Special cases must be handled safely:

- `r == 0`: penalize all singular values;
- `r == R`: penalty is zero;
- never allow `r` outside `[0, R]`.

Do not recompute `r` after every mini-batch. The RAMUSA stage logic requires `r` to be determined from the previous stage and then held fixed during the next stage.

---

## 7. Total multi-task loss

Use the existing project’s task-loss functions for KS, SID, and ER.

Let:

\[
\mathcal{L}_{KS},\quad
\mathcal{L}_{SID},\quad
\mathcal{L}_{ER}
\]

be the task losses and let the existing task weights be:

\[
\omega_{KS},\omega_{SID},\omega_{ER}.
\]

The common multi-task loss is:

\[
\mathcal{L}_{MTL}
=
\omega_{KS}\mathcal{L}_{KS}
+
\omega_{SID}\mathcal{L}_{SID}
+
\omega_{ER}\mathcal{L}_{ER}.
\]

For each stage, optimize:

\[
\mathcal{L}_{total}
=
\mathcal{L}_{MTL}
+
\lambda\|W_{\text{tasks}}\|_{r-}.
\]

`lambda` controls the strength of reduced-rank parameter sharing.

Reuse the task weighting and missing-task/batch handling behavior from the task-clustering implementation. Do not invent a new batching strategy if one is already implemented.

---

## 8. Important distinction from the original RAMUSA solver

The original Han and Zhang RAMUSA paper formulates a linear multi-task problem and solves its stage subproblems using an MM formulation, alternating optimization, and FISTA.

This project uses a neural speech architecture. Implement the **neural architecture adaptation described in this prompt**:

- keep the RAMUSA task-parameter matrix idea;
- keep the capped/truncated singular-value regularization principle;
- keep the stage-wise update of `r` using the fixed threshold `tau`;
- use PyTorch autograd and the project’s normal optimizer to train the neural network.

Do **not** attempt to replace the entire neural trainer with the original paper’s linear FISTA solver unless explicitly requested later.

Document this clearly in comments/README as a RAMUSA-based neural adaptation.

---

## 9. Parameter-update behavior

The implementation must follow this logic:

```text
Current adapter weights
    -> construct W_tasks
    -> compute singular values
    -> compute current stage rank penalty
    -> combine with KS/SID/ER losses
    -> backward()
    -> optimizer.step()
    -> adapter weights change
    -> next batch rebuilds W_tasks from updated adapter weights
```

Do not interpret RAMUSA as manually deleting singular values.

The regularizer creates gradients that encourage the smaller singular values to shrink. `W_tasks` changes because the underlying adapter parameters change during optimization.

---

## 10. Stage-level training flow

Implement the overall process approximately as follows, while matching the existing trainer structure:

```python
r = 0

for stage in range(num_stages):
    # r is fixed throughout this stage
    for epoch in range(epochs_per_stage):
        for batch in train_loader_or_multitask_iterator:
            optimizer.zero_grad()

            outputs = model(batch)

            task_loss = compute_existing_multitask_loss(outputs, batch)

            W_tasks = model.build_ramusa_parameter_matrix()
            singular_values = torch.linalg.svdvals(W_tasks)
            rank_penalty = singular_values[r:].sum()

            total_loss = task_loss + ramusa_lambda * rank_penalty
            total_loss.backward()
            optimizer.step()

        run_existing_validation(...)
        log_stage_metrics(...)

    # End-of-stage RAMUSA update
    with torch.no_grad():
        W_tasks = model.build_ramusa_parameter_matrix()
        singular_values = torch.linalg.svdvals(W_tasks)
        r = int((singular_values >= tau).sum().item())

    save_stage_checkpoint(...)
```

This is structural pseudocode only. Adapt it cleanly to the existing repository instead of forcing this exact code layout.

---

## 11. Model API requirements

Prefer a clean model API similar to the following, adapted to the repository’s current style:

```python
model.forward(...)
model.build_ramusa_parameter_matrix()
model.get_ramusa_singular_values()
```

A RAMUSA-specific helper/loss module may also expose something like:

```python
compute_truncated_trace_penalty(W_tasks, r)
update_effective_rank(W_tasks, tau)
```

Avoid scattering SVD/rank logic across many unrelated files.

---

## 12. Configuration

Follow the configuration format already used for task clustering.

Add configurable values for at least:

- RAMUSA enabled/disabled;
- `lambda` regularization strength;
- fixed `tau` threshold;
- number of stages;
- epochs per stage;
- adapter input/output dimensions if not already inferable;
- optional task-loss weights if they already exist in common config;
- optional RAMUSA logging frequency.

Example conceptual configuration only:

```yaml
ramusa:
  enabled: true
  lambda: 0.01
  tau: 1.0
  num_stages: 5
  epochs_per_stage: 5
```

Do not assume these example values are final research hyperparameters. Keep them configurable.

---

## 13. Training logs and experiment outputs

For every epoch, log the existing metrics plus:

- stage index;
- current fixed `r`;
- `L_MTL`;
- RAMUSA rank penalty;
- total loss;
- singular values of `W_tasks`;
- effective rank estimate based on `tau`.

At the end of each stage, clearly log something similar to:

```text
RAMUSA stage 2 completed
singular values: [...]
fixed tau: ...
previous r: ...
updated r for next stage: ...
```

Preserve the repository’s existing logging backend and output directory format.

---

## 14. Checkpointing and resume support

A RAMUSA checkpoint must save everything already saved by the existing trainer plus:

- current stage index;
- current `r`;
- `tau`;
- RAMUSA `lambda`;
- optimizer state;
- scheduler state if present.

When resuming training, restore `r` and the stage index correctly. Do not restart from `r = 0` unless a completely new run is requested.

---

## 15. Validation and inference

RAMUSA regularization is a **training-time parameter-sharing mechanism**.

During validation/test inference, the forward path should simply be:

```text
Audio
 -> shared speech network
 -> FC2 representation h
 -> corresponding task adapter
 -> corresponding task head
 -> prediction
```

Do not run SVD, thresholding, stage updates, or rank-penalty computation during normal inference unless diagnostic analysis is explicitly requested.

The learned adapter weights already contain the effect of RAMUSA training.

---

## 16. Tests and correctness checks

Add tests or assertions consistent with the existing project test structure.

At minimum verify:

1. Exactly three task adapters exist: KS, SID, ER.
2. IC is absent.
3. All three adapter weight matrices have identical shape.
4. `W_tasks` has shape `[D, 3]`.
5. Singular-value computation returns at most three values.
6. Stage 1 starts with `r = 0`.
7. With `r = 0`, the penalty equals the sum of all singular values.
8. With `r = 1`, only singular values from index 1 onward are included.
9. With `r = R`, the rank penalty is zero without device/dtype errors.
10. `tau` remains fixed throughout a training run.
11. `r` changes only at stage boundaries.
12. Gradients from the RAMUSA penalty reach all three adapter weight matrices.
13. Task heads continue receiving valid adapter outputs.
14. Normal inference does not require RAMUSA SVD calculations.
15. Checkpoint resume restores the correct stage and `r`.

Also run a small synthetic/unit test where the adapter parameters are manually controlled and verify that the computed singular values and truncated penalty match direct PyTorch calculations.

---

## 17. Numerical/device considerations

- Keep `W_tasks` on the same device and dtype as the adapter parameters.
- Avoid unnecessary CPU/GPU transfers during the training loop.
- Use `torch.linalg.svdvals` rather than converting to NumPy.
- Do not detach the matrix when computing the training regularizer.
- Use `torch.no_grad()` only for the end-of-stage `r` update and diagnostic logging where gradients are unnecessary.
- Handle mixed precision according to the existing trainer. If SVD becomes unstable under low precision, compute the SVD/regularizer in float32 while preserving gradient flow back to the original parameters.
- Avoid explicit matrix reconstruction unless required for diagnostics.

---

## 18. Preserve experimental comparability

The purpose of this architecture is to compare a lower-rank parameter-sharing method with the project’s other MTL approaches. Therefore:

- keep the same upstream/backbone;
- keep FC1, pooling, and FC2 consistent;
- keep the same task datasets and splits;
- keep the same task heads/output classes;
- keep the same evaluation metrics;
- keep training settings comparable where possible;
- change only the architecture-specific parameter-sharing mechanism and parameters required by RAMUSA.

Do not add unrelated architectural improvements that would make the comparison unfair.

---

## 19. Expected conceptual behavior

Suppose the three singular values are initially:

```text
[8.0, 2.5, 0.4]
```

and `tau = 1.0`.

After Stage 1, the end-of-stage update gives:

```text
r = 2
```

because two singular values are at least `tau`.

During Stage 2, the penalty should therefore contain only the third singular value:

```text
rank_penalty = 0.4
```

The method should encourage the weaker direction to shrink while avoiding direct penalization of the two dominant directions during that stage.

Do not force the singular values to become exactly zero. Their values should emerge from optimization.

---

## 20. Deliverables

After implementation, provide:

1. A short summary of the existing task-clustering project structure that was followed.
2. A list of new files created.
3. A list of existing files modified.
4. A concise explanation of each modification.
5. The implemented RAMUSA architecture flow.
6. The exact formula used for `W_tasks`.
7. The exact stage-specific regularization formula used in code.
8. Where and when `r` is updated.
9. Confirmation that `tau` is fixed throughout a run.
10. Confirmation that only KS, SID, and ER are included.
11. Confirmation that RAMUSA operations are training-only during normal inference.
12. Instructions/commands for starting a training run.
13. Instructions for resuming a checkpoint.
14. A short sanity-test command or procedure.

Do not leave TODO placeholders for the core RAMUSA logic.

---

## 21. Architecture summary

The final implemented architecture should correspond to:

```text
                         ┌─ KS Adapter (P_KS)   -> KS Head  -> KS output
Audio -> Shared Network -> FC2 h
                         ├─ SID Adapter (P_SID) -> SID Head -> SID output
                         └─ ER Adapter (P_ER)   -> ER Head  -> ER output

Training-only RAMUSA path:

P_KS, P_SID, P_ER
      -> flatten each adapter weight
      -> W_tasks = [vec(P_KS), vec(P_SID), vec(P_ER)]
      -> singular values
      -> stage-specific truncated trace-norm penalty
      -> add to common MTL loss
      -> backpropagation updates adapters

At each stage boundary:

updated W_tasks
      -> singular values
      -> compare with fixed tau
      -> update r
      -> use new r in the next stage
```

The key property that differentiates this architecture is that **KS, SID, and ER retain separate task-specific adapters, while RAMUSA encourages their adapter parameters to share a compact low-rank structure by progressively penalizing weaker singular directions across training stages.**
