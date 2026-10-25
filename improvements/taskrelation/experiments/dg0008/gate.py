"""Fail-closed Stage-1 validity gate for DG-0008 (proposal "Stage-1 validity gate").

Every required-zero count is computed from stored run records, never from a
transcript.  A single non-zero count invalidates the execution and is recorded
as ``stage1_valid: false``; it is not evidence for or against H1-H3.  Held-out
endpoint evaluation is admitted only when this gate reports valid.
"""

from . import SCHEMA_GATE
from .manifest import ManifestError

PAIR_SHARED_FIELDS = (
    "identity_digest",
    "run_manifest_digest",
    "epoch_digests",
    "consumed_n_er",
    "step_counts",
    "target_coefficient",
    "lr_sequence",
    "initialization_sha256",
    "rng_sequence_sha256",
)

REPEAT_FIELDS = (
    "loss_trace",
    "final_validation_er_accuracy",
    "fixed_final_checkpoint_sha256",
)


def _member_key(record):
    return (record["cell"], int(record["fold"]), int(record["seed"]), record["arm"])


def pair_mismatches(pair_record, control_record):
    """Every pair-equality field that differs (proposal discriminating measurements)."""

    mismatches = []
    for field in PAIR_SHARED_FIELDS:
        if pair_record.get(field) != control_record.get(field):
            mismatches.append(field)
    return mismatches


def repeat_mismatches(first_record, repeat_record):
    mismatches = []
    for field in REPEAT_FIELDS:
        if first_record.get(field) != repeat_record.get(field):
            mismatches.append(field)
    return mismatches


def manifest_digest_mismatches(records, expected_index):
    """``expected_index`` maps ``"cell|fold|seed|epoch"`` to the frozen digest."""

    count = 0
    for record in records:
        if record.get("run_manifest_digest") != expected_index["runs"].get(
            "{}|{}|{}".format(record["cell"], record["fold"], record["seed"])
        ):
            count += 1
        for epoch, digest in enumerate(record.get("epoch_digests", [])):
            key = "{}|{}|{}|{}".format(record["cell"], record["fold"], record["seed"], epoch)
            if digest != expected_index["epochs"].get(key):
                count += 1
    return count


def embedding_digest_mismatches(expected_digests, observed_digests):
    """Condition (a) of dual identity admission: the declared embedding SHA-256 set."""

    mismatches = 0
    for name, expected in expected_digests.items():
        if observed_digests.get(name) != expected:
            mismatches += 1
    for name in observed_digests:
        if name not in expected_digests:
            mismatches += 1
    return mismatches


def s_envelope_violations(records, minimum, maximum):
    """The coarse DG-0001 sanity envelope is sanity-only, never a reproduction."""

    violations = []
    for record in records:
        for steps in record.get("step_counts", []):
            if not (int(minimum) <= int(steps) <= int(maximum)):
                violations.append(
                    {"cell": record["cell"], "fold": record["fold"], "steps": steps}
                )
    return violations


def stage1_gate(
    records,
    *,
    expected_index,
    expected_identity_digest,
    observed_identity_digest,
    expected_embeddings,
    observed_embeddings,
    s_envelope,
    repeat_first,
    repeat_second,
    held_out_reads,
    expected_matrix_members,
):
    """Assemble the Stage-1 gate from stored records.

    ``expected_matrix_members`` is the set of ``(cell, fold, seed, arm)`` tuples
    that must all be present and valid before any held-out endpoint is read.
    """

    problems = []
    present = {_member_key(record): record for record in records}

    missing = sorted(expected_matrix_members - set(present))
    if missing:
        problems.append({"class": "MISSING_MATRIX_MEMBERS", "count": len(missing)})

    pair_mismatch_count = 0
    for cell in ("ks_er", "si_er"):
        for fold in range(10):
            for seed in range(5):
                pair = present.get((cell, fold, seed, "pair"))
                control = present.get((cell, fold, seed, "control"))
                if pair is None or control is None:
                    continue
                if pair_mismatches(pair, control):
                    pair_mismatch_count += 1

    repeat_count = 0
    if repeat_first is None or repeat_second is None:
        problems.append({"class": "MISSING_REPEAT", "count": 1})
    else:
        repeat_count = len(repeat_mismatches(repeat_first, repeat_second))

    counts = {
        "embedding_digest_mismatches": embedding_digest_mismatches(
            expected_embeddings, observed_embeddings
        ),
        "identity_manifest_mismatches": int(
            observed_identity_digest != expected_identity_digest
        ) + sum(
            1 for record in records
            if record.get("identity_digest") != expected_identity_digest
        ),
        "opportunity_or_aggregate_digest_mismatches": manifest_digest_mismatches(
            records, expected_index
        ),
        "pair_equality_mismatch_members": pair_mismatch_count,
        "er_count_or_order_mismatch_members": sum(
            1 for record in records if not record.get("er_counts_match", False)
        ),
        "repeat_bitwise_mismatches": repeat_count,
        "held_out_reads_before_gate": int(held_out_reads),
        "checkpoint_readback_failures": sum(
            1 for record in records if not record.get("checkpoint_readback_ok", False)
        ),
    }
    envelope_violations = s_envelope_violations(records, s_envelope[0], s_envelope[1])
    if envelope_violations:
        problems.append({"class": "S_ENVELOPE", "count": len(envelope_violations)})

    valid = (
        not problems
        and all(value == 0 for value in counts.values())
        and bool(records)
    )
    return {
        "schema": SCHEMA_GATE,
        "counts": counts,
        "s_envelope": {"minimum": int(s_envelope[0]), "maximum": int(s_envelope[1]),
                       "violations": len(envelope_violations)},
        "problems": problems,
        "stage1_valid": bool(valid),
    }


def assert_held_out_admitted(gate_document):
    """The delayed-evaluation guard: no endpoint before a valid gate."""

    if not isinstance(gate_document, dict) or gate_document.get("schema") != SCHEMA_GATE:
        raise ManifestError("held-out evaluation requires a dg0008 validity-gate document")
    if not gate_document.get("stage1_valid"):
        raise ManifestError(
            "held-out evaluation refused: the Stage-1 validity gate is not valid"
        )
    if int(gate_document["counts"].get("held_out_reads_before_gate", 1)) != 0:
        raise ManifestError("held-out evaluation refused: a premature read is recorded")
    return True
