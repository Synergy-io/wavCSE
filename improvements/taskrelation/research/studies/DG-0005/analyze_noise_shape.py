"""Post-hoc bound on how much of DG-0005's ER norm drop is estimator-size scaling.

PRE-REGISTRATION — written and committed BEFORE any statistic was computed.

Motivation. F10 established that ER's gradient-norm dominance is
training-mixture controlled: raising ER from ~47 to ~435 sampled examples per
2,048-example batch took the late max/min task-norm ratio from 8.962 to 2.352,
with ER's own late mean shared-gradient norm falling 6.972 -> 1.608. Two
explanations survived: (i) the ER training-set gradient estimate is less
noise-inflated at 435 samples (`E||g|| >= ||E g||`), or (ii) the ER head carries
a smaller *mean* gradient because it receives ~9x more updates per epoch and
saturates harder.

Why a naive shape test does not identify the mechanism. Writing a per-step
gradient as `g = m + n` with fixed `m` and zero-mean isotropic noise `n` of
per-coordinate scale `sigma` gives

    ||g||^2 = m^2 + 2 m.(sigma z) + sigma^2 * chi2_d,      d = 1,550,800,

and the relative dispersion of `||g||` is then `sqrt(4k^2 + 2k^4) /
(2 sqrt(d) (1 + k^2))` with `k = sigma sqrt(d)/||m||`. Over the whole range of
`k` this CV never exceeds `0.707/sqrt(d) ~ 0.00057`.

The observed within-phase dispersion of ER's per-step norm is of order 0.3
(CV), three orders of magnitude larger. Under an isotropic-noise model that is
impossible: the dispersion must instead be dominated by step-to-step variation
of `m` itself (batch composition and learning), which exists in both arms. So
within-phase *shape* statistics (mean/median, CV, skew) are **not** usable as a
noise-variance measurement with this instrumentation, and a shape-based
discriminator would be over-claiming. The script reports them descriptively and
declines to use them as a test.

What IS identifiable, without knowing sigma or materialising the noise. The
mean-inflation identity `E||g|| >= ||E g||` gives a ceiling on the contribution
of estimator size. If the noise term dominates the norm (the hypothesis under
test), then `E||g|| ~ sigma sqrt(d)`, and the standard estimator scaling
`sigma ∝ 1/sqrt(n)` predicts a mean-norm ratio between arms of exactly
`sqrt(n_A1/n_A0)`. Any observed drop larger than that cannot be explained by
estimator size alone. Therefore

    fraction explained by estimator-size scaling  <=  ln(sqrt(n1/n0))
                                                     / ln(mean_A0/mean_A1)

and the complementary share `1 - fraction` must come from a change in `m` (task
state / convergence) or from noise growing faster than `1/sqrt(n)`.

Predictions, fixed before computation.

  P1 (bound, primary). For ER in the late phase, `sqrt(n1/n0) ~ 3.0`, while the
     observed mean drop is larger, so the estimator-size share is bounded
     strictly below 1 and the complement must be positive. The reported number
     for each seed is that bound; a mean bound materially below 1 across seeds
     is the result. For KS and SI the counts move by only ~1.2x, so the same
     bound must be near 1 (their norms barely move) — a sanity check on the
     machinery, not a finding.
  P2 (descriptive, not a test). ER's shape statistics are similar across arms;
     under the isotropic-noise argument above, neither a null nor a non-null
     shape difference can be read as evidence about estimator variance. The
     script reports them so the claim is documented, not so it can be argued
     from.
  P3 (replication). DG-0002's matched baseline confirmation (same standard
     composition, different commit) supplies five independent A0 runs; ER's
     late shape statistics there must be consistent with DG-0005's A0 arm.

Decision rule. This analysis yields a *bound*, not a verdict on which mechanism
dominates. The claim it can carry is: "estimator-size scaling explains at most
X% of the observed ER late norm drop; at least (1-X)% requires a change in the
task's mean gradient or faster-than-1/sqrt(n) noise growth." No CONFIRMED
finding may be built on it.

Status and limits. Post-hoc exploratory analysis of an already-confirmed Study;
DG-0005's pre-registration said nothing about this. Additional limits:

  * one-knob design: A1 differs from A0 in estimator size AND optimizer path, so
    the complement is not attributable to a single named cause;
  * the bound assumes noise-dominated norms; if the norm is signal-dominated the
    noise share is 0 and the same data are equally consistent with "m fell
    entirely", so the bound is an upper bound and the true share is an interval
    [0, X];
  * `sigma ∝ 1/sqrt(n)` assumes independent per-sample noise; any composition
    effect that scales more slowly would shrink the bound further;
  * ReduceLROnPlateau means the arms need not share a learning rate within a
    phase;
  * n = 5 seeds, so intervals are wide by construction.
"""

import json
import math
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from scipy import stats

STUDY_ID = "DG-0005"
REPLICATION_STUDY_ID = "DG-0002"
EXPERIMENT_NAME = "taskrelation-diagnostics"
TRACKING_URI = "https://dagshub.com/Ke-vin-S/wavCSE.mlflow"
STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[4]
SEEDS = (0, 1, 2, 3, 4)
PHASES = ("early", "middle", "late")
TASKS = ("ks", "si", "er")
ARMS = {
    "a0": "wavcse-standard-composition",
    "a1": "wavcse-er-weighted-composition",
}
T_CRITICAL_95 = {4: 2.776}


def _summary(values):
    values = [float(value) for value in values]
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    critical = T_CRITICAL_95.get(len(values) - 1, 1.96)
    half_width = critical * sample_std / math.sqrt(len(values))
    return {
        "n": len(values),
        "values": values,
        "mean": mean,
        "sample_std": sample_std,
        "t_95_interval": [mean - half_width, mean + half_width],
        "min": min(values),
        "max": max(values),
    }


def _distribution_stats(norms):
    array = np.asarray(norms, dtype=float)
    mean = float(array.mean())
    median = float(np.median(array))
    std = float(array.std(ddof=1))
    q1, q3 = (float(value) for value in np.percentile(array, [25, 75]))
    return {
        "n": int(array.size),
        "mean": mean,
        "median": median,
        "std": std,
        "cv": std / mean,
        "mean_over_median": mean / median,
        "iqr_over_median": (q3 - q1) / median,
        "skew_g1": float(stats.skew(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _phase_norms(payload, phase, task):
    return [
        record["gradient_norms"][task]
        for record in payload["records"]
        if record["phase"] == phase
    ]


def _phase_counts(payload, phase, task):
    return [
        record["valid_examples"][task]
        for record in payload["records"]
        if record["phase"] == phase
    ]


def _download_artifact(client, run_id, destination):
    target = Path(destination) / run_id
    target.mkdir(parents=True, exist_ok=True)
    path = client.download_artifacts(
        run_id, "results/gradient_diagnostics.json", dst_path=str(target)
    )
    with open(path) as handle:
        return json.load(handle)


def _collect(client, experiment_id, study_id, stage, methods_by_arm):
    runs = client.search_runs(
        [experiment_id],
        f"tags.study_id = '{study_id}' and tags.stage = '{stage}'",
        max_results=30,
    )
    selected = {}
    for run in runs:
        method = run.data.tags.get("method")
        for arm, expected in methods_by_arm.items():
            if method == expected and run.info.status == "FINISHED":
                key = (arm, int(run.data.tags["seed"]))
                if key in selected:
                    raise RuntimeError(f"Duplicate run for {key}")
                selected[key] = run.info.run_id
    return selected


def _isotropic_cv_ceiling(shared_parameter_count):
    """Max relative dispersion of ||g|| an isotropic-noise model can produce."""
    return math.sqrt(2.0) / 2.0 * 1.0 / math.sqrt(shared_parameter_count)


def _estimator_share_bound(mean_a0, mean_a1, n_a0, n_a1):
    """Upper bound on the estimator-size share of the A0->A1 log change.

    `predicted_ratio` is the mean-norm ratio the noise-scaling model predicts,
    `mean_A0/mean_A1 = sqrt(n_A1/n_A0)`, written so that a value above 1 means
    "the model predicts A1's mean norm is smaller". Both directions are handled:
    reducing a task's sampled count is predicted to *raise* its norm.
    """
    if mean_a1 <= 0 or mean_a0 <= 0 or n_a0 <= 0 or n_a1 <= 0:
        return None
    predicted_ratio = math.sqrt(n_a1 / n_a0)
    observed_ratio = mean_a0 / mean_a1
    entry = {
        "predicted_ratio": predicted_ratio,
        "observed_ratio": observed_ratio,
    }
    if observed_ratio == 1.0 or predicted_ratio == 1.0:
        entry["share_bound"] = None
        entry["note"] = "degenerate ratio"
        return entry
    if (predicted_ratio - 1.0) * (observed_ratio - 1.0) <= 0.0:
        entry["share_bound"] = 0.0
        entry["note"] = (
            "model predicts the opposite direction; estimator-size scaling "
            "cannot account for the observed change at all"
        )
        return entry
    entry["predicted_log_change"] = math.log(predicted_ratio)
    entry["observed_log_change"] = math.log(observed_ratio)
    share = math.log(predicted_ratio) / math.log(observed_ratio)
    entry["share_bound"] = share
    entry["note"] = (
        "share > 1 means the model predicts a larger change than observed, so "
        "the observed norm is consistent with estimator scaling alone"
        if share > 1.0 else
        "share < 1 means estimator scaling cannot explain the whole change"
    )
    return entry


def main():
    load_dotenv(REPO_ROOT / ".env")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Missing MLflow experiment {EXPERIMENT_NAME}")

    selected = _collect(client, experiment.experiment_id, STUDY_ID, "confirm", ARMS)
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(selected) != expected:
        raise RuntimeError(f"Expected {sorted(expected)}, found {sorted(selected)}")
    replication = _collect(
        client,
        experiment.experiment_id,
        REPLICATION_STUDY_ID,
        "confirm",
        {"a0": "wavcse-baseline"},
    )

    with tempfile.TemporaryDirectory(dir=STUDY_DIR) as temporary_directory:
        payloads = {
            key: _download_artifact(client, run_id, temporary_directory)
            for key, run_id in sorted(selected.items())
        }
        replication_payloads = {
            run_id: _download_artifact(client, run_id, temporary_directory)
            for _, run_id in sorted(replication.items())
        }

        shared_parameter_count = payloads[("a0", 0)]["shared_parameter_count"]
        cv_ceiling = _isotropic_cv_ceiling(shared_parameter_count)

        stats_by_key = {
            (arm, seed): {
                phase: {
                    task: _distribution_stats(_phase_norms(payload, phase, task))
                    for task in TASKS
                }
                for phase in PHASES
            }
            for (arm, seed), payload in payloads.items()
        }
        counts_by_key = {
            (arm, seed): {
                phase: {
                    task: statistics.fmean(_phase_counts(payload, phase, task))
                    for task in TASKS
                }
                for phase in PHASES
            }
            for (arm, seed), payload in payloads.items()
        }
        replication_stats = {
            run_id: {
                phase: {
                    task: _distribution_stats(_phase_norms(payload, phase, task))
                    for task in TASKS
                }
                for phase in PHASES
            }
            for run_id, payload in replication_payloads.items()
        }

    bounds = {}
    for phase in ("middle", "late"):
        for task in TASKS:
            per_seed = []
            for seed in SEEDS:
                a0 = stats_by_key[("a0", seed)][phase][task]
                a1 = stats_by_key[("a1", seed)][phase][task]
                bound = _estimator_share_bound(
                    a0["mean"],
                    a1["mean"],
                    counts_by_key[("a0", seed)][phase][task],
                    counts_by_key[("a1", seed)][phase][task],
                )
                per_seed.append({
                    "seed": seed,
                    "a0_counts_mean": counts_by_key[("a0", seed)][phase][task],
                    "a1_counts_mean": counts_by_key[("a1", seed)][phase][task],
                    "a0_mean_norm": a0["mean"],
                    "a1_mean_norm": a1["mean"],
                    **(bound or {}),
                })
            shares = [
                entry["share_bound"] for entry in per_seed
                if entry.get("share_bound") is not None
            ]
            bounds[f"{phase}.{task}"] = {
                "per_seed": per_seed,
                "share_bound_summary": _summary(shares) if shares else None,
                "complement_share_summary": (
                    _summary([1.0 - value for value in shares]) if shares else None
                ),
            }

    shape_differences = {
        f"{phase}.{task}.{statistic}": _summary([
            stats_by_key[("a1", seed)][phase][task][statistic]
            - stats_by_key[("a0", seed)][phase][task][statistic]
            for seed in SEEDS
        ])
        for phase in PHASES
        for task in TASKS
        for statistic in ("mean_over_median", "cv", "skew_g1")
    }

    observed_cv_late_er = _summary([
        stats_by_key[("a0", seed)]["late"]["er"]["cv"] for seed in SEEDS
    ] + [
        stats_by_key[("a1", seed)]["late"]["er"]["cv"] for seed in SEEDS
    ])

    er_late = bounds["late.er"]["share_bound_summary"]
    er_middle = bounds["middle.er"]["share_bound_summary"]
    verdict = (
        "bounded: estimator-size scaling explains at most "
        f"{er_late['mean']:.2f} of the ER late log-drop "
        f"(95% CI [{er_late['t_95_interval'][0]:.2f}, {er_late['t_95_interval'][1]:.2f}])"
    )

    result = {
        "study_id": STUDY_ID,
        "type": "post_hoc_exploratory_analysis",
        "stage": "confirm_estimator_share_bound",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shared_parameter_count": shared_parameter_count,
        "seeds": list(SEEDS),
        "hypothesis_under_test": (
            "Share of DG-0005's ER mean-norm drop attributable to estimator-size "
            "scaling (sigma ~ 1/sqrt(n)) versus a change in the underlying mean "
            "gradient"
        ),
        "isotropic_noise_cv_ceiling": cv_ceiling,
        "observed_late_er_cv": observed_cv_late_er,
        "distribution_stats": {
            f"{arm}.s{seed}": stats_by_key[(arm, seed)]
            for arm, seed in sorted(stats_by_key)
        },
        "paired_shape_differences_a1_minus_a0": shape_differences,
        "estimator_share_bounds": bounds,
        "replication_a0_standard_composition": {
            "runs": len(replication_stats),
            "late_er_mean_over_median": _summary([
                record["late"]["er"]["mean_over_median"]
                for record in replication_stats.values()
            ]),
            "late_er_cv": _summary([
                record["late"]["er"]["cv"] for record in replication_stats.values()
            ]),
            "late_er_mean": _summary([
                record["late"]["er"]["mean"] for record in replication_stats.values()
            ]),
        },
        "verdict": verdict,
        "verdict_basis": {
            "er_late_share_bound": er_late,
            "er_middle_share_bound": er_middle,
            "shape_statistics_used_as_test": False,
            "shape_statistics_reason": (
                "isotropic-noise relative dispersion ceiling is ~"
                f"{cv_ceiling:.5f}, while observed late ER CV is ~"
                f"{observed_cv_late_er['mean']:.3f}; within-phase dispersion is "
                "therefore dominated by step-to-step mean-gradient variation and "
                "cannot proxy estimator variance"
            ),
            "note": (
                "Post-hoc; carries no CONFIRMED claim. The bound is an upper "
                "bound: a signal-dominated norm implies a 0 share, so the true "
                "estimator share lies in [0, bound]."
            ),
        },
    }
    output_path = STUDY_DIR / "noise_shape_result.json"
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)

    print(json.dumps({
        "result": str(output_path),
        "verdict": verdict,
        "isotropic_noise_cv_ceiling": cv_ceiling,
        "observed_late_er_cv": observed_cv_late_er,
        "er_late_share_bound": er_late,
        "er_middle_share_bound": er_middle,
        "ks_late_share_bound": bounds["late.ks"]["share_bound_summary"],
        "si_late_share_bound": bounds["late.si"]["share_bound_summary"],
        "late_er_shape_a0": [
            round(stats_by_key[("a0", seed)]["late"]["er"]["mean_over_median"], 3)
            for seed in SEEDS
        ],
        "late_er_shape_a1": [
            round(stats_by_key[("a1", seed)]["late"]["er"]["mean_over_median"], 3)
            for seed in SEEDS
        ],
        "replication": result["replication_a0_standard_composition"],
    }, indent=2))


if __name__ == "__main__":
    main()
