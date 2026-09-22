# Nguyen, Jeong, Yang and Hwang (2021) — Temporal Probabilistic Asymmetric Multi-Task Learning

## Citation

A. Tuan Nguyen, Hyewon Jeong, Eunho Yang, and Sung Ju Hwang. “Clinical Risk Prediction with Temporal Probabilistic Asymmetric Multi-Task Learning.” *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-21)*, 2021.

Primary source: https://ojs.aaai.org/index.php/AAAI/article/view/17097 (PDF: https://ojs.aaai.org/index.php/AAAI/article/download/17097/16904)

Verified against that PDF (abstract; §Approach, Eqs. (1)–(12); Tables 1–3; Figs. 1–3). Code: https://github.com/mandiehyewon/TPAMTL. Same research line as AMTL (Lee et al. 2016) and Deep-AMTFL (Lee et al. 2018); the paper cites the latter as “Lee, Yang, and Hwang (2017)”.

## Problem

Loss-based asymmetric transfer (AMTL) uses task loss as a reliability proxy, but a task trained on few instances can have *low* loss while being overfitted and unreliable, and in time-series data there is no per-timestep loss. The paper substitutes feature-level *uncertainty* (epistemic + aleatoric) for loss and generalises the transfer direction to the time axis.

## Mathematical assumption

Each task has its own probabilistic encoder producing latent features `z_d ~ N(z̄_d, diag(σ_d²))` (`z̄_d = h_d W_d + b_d`, `σ_d = softplus(h_d W_d^σ + b_d^σ)`), with epistemic uncertainty from MC dropout and aleatoric from the learned variance. The transfer weight from task `j` to task `d` is computed by a small network

`α_{j,d} = F_{j,d}(z_{j,d}, z_d, σ²_{j,d}, σ²_d)`,

and features are combined as `C_d = z_d + G²_d(Σ_{j≠d} α_{j,d} G¹_j(z_{j,d}))`, with per-task transforms `G¹_j`/`G²_d` into and out of a shared latent space. In the temporal model, transfer is restricted to past→future timesteps, `C_d^(t) = z_d^(t) + G²_d(Σ_j Σ_{i≤t} α^{(i,t)}_{j,d} G¹_j(z_j^(i)))`, which is the paper’s source of structural asymmetry. Training maximises a per-task negative log-likelihood over the tasks whose labels are available for each instance, plus weight decay.

## Relation representation

A family of learned, per-task-pair, per-sample transfer coefficients `α_{j,d}` (thus a directed relation in expectation); no `T × T` relation matrix is stored, and there is no covariance, precision or similarity parameter. There is also no global relation object to compare against classical MTRL’s `Ω`. Reliability enters through feature uncertainty, and the paper’s non-temporal ablation `P-AMTL` shows the uncertainty-based variant beats the loss-based `AMTL-Loss` variant (0.6778 vs 0.6345 mean AUROC).

## Optimization method

Standard end-to-end SGD with MC-dropout sampling for epistemic variance; two additional networks per task pair (`F_{j,d}`) and two transform networks per task (`G¹`, `G²`); attention pooling over timesteps before a sigmoid output. The datasets are small (≈2000–4000 patients), so training is cheap; the model explicitly tolerates instances with labels for only some tasks.

## Evidence

MNIST-Variation ablation: 5 binary tasks with 5000/5000/1000/1000/500 samples; mean AUROC — P-AMTL 0.6778 > MTL 0.6565 > STL 0.6432 > AMTL-Loss 0.6345, i.e. loss-based transfer *fails* under imbalance while uncertainty-based transfer does not. MIMIC-III Infection (Fever/Infection/Mortality): TP-AMTL 0.7102 vs best baseline SAnD 0.6953 and AMTL-LSTM 0.6798. PhysioNet (Stay<3/Cardiac/Recovery/Mortality): TP-AMTL 0.8743 vs SAnD 0.8607 and AMTL-LSTM 0.8022. Uncertainty-loss weighting (RETAIN-Kendall) is the weakest multi-task baseline, and the paper reports no negative transfer for TP-AMTL. All tasks are binary clinical risk predictions on a shared EHR input space.

## Assumptions

An encoder–decoder network per task with a shared low-level embedding and RNN preprocessing; each task’s latent features live in a common dimension `k`; tasks are binary (or real-valued) single-output problems; per-pair transfer networks `F_{j,d}` and transforms `G¹/G²` are trained; in the temporal form, a timestep ordering exists so that past→future transfer is meaningful; instances may be disjoint across tasks but must share a feature space.

## Differences from our setting

* **Taxonomy.** This is a deep probabilistic feature-sharing model whose asymmetry comes from uncertainty-attenuated attention, not from a task-relation object. The project’s taxonomy excludes uncertainty weighting, and the relation here is a per-sample attention weight rather than a covariance/precision/directed relation matrix carried by the model.
* **Where the asymmetry lives.** The structural asymmetry is the past→future constraint on the time axis. Our three tasks (KS/SI/ER) have no temporal order, so that mechanism is unavailable; without it, the remaining asymmetry is only whatever the attention MLPs happen to learn, which is not a published relation object we could match to a control.
* **Architecture.** It requires per-task probabilistic encoders plus per-pair transfer and transform networks. Our trunk is shared across all three tasks, so task-specific latent features `z_d` do not exist; and the per-pair MLPs would have to be built on the shared trunk activation, i.e. a new architecture rather than this method.
* **Value retained.** Its negative result on loss-based transfer under class imbalance (MNIST-Variation) is informative for any loss-scaled relation arm, and its uncertainty-vs-loss experiment is the cleanest published evidence that task loss is a poor reliability proxy — directly relevant to AMTL’s `(1 + μ‖b_t^o‖₁)L(w_t)` design, though not usable as a family-A mechanism.

## Implementation difficulty

Not implementable as published here: it would require replacing the shared-trunk/three-head model with per-task probabilistic encoders, per-pair attention networks and per-task feature transforms, and its asymmetry mechanism (temporal direction) does not exist for our task set.

## Candidate Study ID

`LT-0002` — screened candidate (family A), recorded as a taxonomy and architecture failure.

## LT-0002 assessment

* **Gate 1 — explicit relation object: PARTIAL.** Directed transfer coefficients are learned from data (per task pair, per sample), but they are an attention/uncertainty mechanism, not an explicit relation structure (covariance, precision, similarity matrix or global directed graph) held by the model.
* **Gate 2 — taxonomy: FAIL.** Uncertainty-attenuated deep feature sharing with learned attention; the project excludes uncertainty weighting, and feature-attention transfer is not a §2.4 relation-learning object. (The paper itself benchmarks against, and improves on, Kendall-style uncertainty weighting.)
* **Gate 3 — heterogeneous heads: FAIL.** Needs per-task probabilistic encoders and per-pair transfer/transform networks; a shared trunk makes `z_d` meaningless. Its own asymmetry is a temporal past→future constraint our non-temporal tasks do not have.
* **Gate 4 — fixed representation: PASS.** No upstream retraining; it trains downstream representations from a fixed input embedding.
* **Gate 5 — faithful implementability: FAIL.** The published model cannot be instantiated on a shared-trunk three-head architecture without replacing it; doing so would be a new hybrid.
* **Gate 6 — source verified: PASS.** Authors, venue, equations (1)–(12) and the reported numbers checked against the official AAAI PDF.

**Verdict: `FAIL — uncertainty-attenuated deep feature-attention transfer outside the Task Relation Learning taxonomy, whose structural asymmetry is a temporal past→future constraint unavailable to three non-temporal tasks.`**
