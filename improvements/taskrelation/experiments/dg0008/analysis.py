"""Frozen seed-level analysis for DG-0008 (proposal "Success and stop conditions").

The classification regions, the ``delta_DG = 0.010`` SESOI and the strict
boundary rule are fixed before any endpoint is read.  Folds are nested
description only and are never treated as 50 independent observations.
"""

import math

from . import SCHEMA_ANALYSIS
from .manifest import ManifestError

DELTA_DG = 0.010
SEEDS = 5


def t_critical_975_df4():
    """Two-sided 0.975 Student-t quantile with 4 degrees of freedom."""

    try:
        from scipy.stats import t as student_t
    except ImportError as exc:  # pragma: no cover - scipy is a declared dependency
        raise ManifestError("scipy is required for the frozen t interval") from exc
    return float(student_t.ppf(0.975, SEEDS - 1))


def summarize(values):
    """Mean, sample SD and two-sided 95% t interval over the five seed values."""

    values = [float(value) for value in values]
    if len(values) != SEEDS:
        raise ManifestError("summary requires exactly {} seed values".format(SEEDS))
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    sd = math.sqrt(variance)
    half_width = t_critical_975_df4() * sd / math.sqrt(len(values))
    return {
        "mean": mean,
        "sd": sd,
        "half_width": half_width,
        "low": mean - half_width,
        "high": mean + half_width,
        "values": values,
    }


def classify(t_si, t_ks, d, delta=DELTA_DG):
    """Exhaustive, mutually exclusive screening classification (strict inequalities)."""

    si_harmful = t_si["high"] < -delta
    ks_harmful = t_ks["high"] < -delta
    ks_equivalent = t_ks["low"] > -delta and t_ks["high"] < delta
    d_negative = d["high"] < -delta
    d_equivalent = d["low"] > -delta and d["high"] < delta
    si_equivalent = t_si["low"] > -delta and t_si["high"] < delta

    if si_harmful and ks_equivalent and d_negative:
        return "H1"
    if si_harmful and ks_harmful and d_equivalent:
        return "H2"
    if si_equivalent and ks_equivalent:
        return "H3"
    return "INCONCLUSIVE"


def nested_description(records):
    """Fold-level residuals, sign counts and held-out-speaker ranges (description only)."""

    by_cell = {"SI": {}, "KS": {}}
    for record in records:
        label = "SI" if record["cell"] == "si_er" else "KS"
        by_cell[label][(int(record["seed"]), int(record["fold"]))] = {
            "pair_accuracy": record["pair_er_accuracy"],
            "control_accuracy": record["control_er_accuracy"],
            "residual": record["pair_er_accuracy"] - record["control_er_accuracy"],
            "test_speaker": record.get("test_speaker"),
        }
    description = {}
    for label, cells in by_cell.items():
        residuals = [entry["residual"] for entry in cells.values()]
        description[label] = {
            "fold_residuals": {
                "{}|{}".format(seed, fold): entry["residual"]
                for (seed, fold), entry in sorted(cells.items())
            },
            "positive_folds": sum(1 for value in residuals if value > 0),
            "negative_folds": sum(1 for value in residuals if value < 0),
            "zero_folds": sum(1 for value in residuals if value == 0),
            "held_out_speakers": sorted(
                {entry["test_speaker"] for entry in cells.values() if entry["test_speaker"]}
            ),
        }
    return description


def analyse(per_seed_t_si, per_seed_t_ks, delta=DELTA_DG, nested=None):
    """The decision object: seed-level intervals and one screening class."""

    t_si = summarize(per_seed_t_si)
    t_ks = summarize(per_seed_t_ks)
    d_values = [si - ks for si, ks in zip(per_seed_t_si, per_seed_t_ks)]
    d = summarize(d_values)
    return {
        "schema": SCHEMA_ANALYSIS,
        "delta": float(delta),
        "t_critical_975_df4": t_critical_975_df4(),
        "T_SI": t_si,
        "T_KS": t_ks,
        "D": d,
        "classification": classify(t_si, t_ks, d, delta),
        "nested": nested or {},
    }
