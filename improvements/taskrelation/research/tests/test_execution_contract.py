"""Behavioral tests for the bounded Execution Plane V1 contract."""

import copy
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from improvements.taskrelation.research import execution_contract as contract


REPO_ROOT = Path(__file__).resolve().parents[4]
PROPOSAL_PATH = (
    "improvements/taskrelation/research/proposals/"
    "DG-0008_exact_matched_directed_transfer.md"
)
COMMIT = "a" * 40


def blocker(identifier="B-001", klass="PREREQUISITE", detail="not ready", owner="MAIN_OMP"):
    return {"id": identifier, "class": klass, "detail": detail, "owner": owner}


def workload(proposal=None, *, choice="MATERIALIZE", revision=1):
    proposal = proposal or contract.proposal_reference(PROPOSAL_PATH, repo_root=REPO_ROOT)
    document = {
        "schema_version": 1,
        "kind": "execution_workload",
        "workload_id": "DP-0008-W01",
        "revision": revision,
        "proposal": proposal,
        "scientific_invariants": [
            {
                "id": "SI-ARMS",
                "source": PROPOSAL_PATH + "#Proposed study",
                "name": "matrix",
                "value": {"cells": ["ks_er", "si_er"], "arms": ["pair", "control"]},
            },
            {
                "id": "SI-SEEDS",
                "source": PROPOSAL_PATH + "#Proposed study",
                "name": "seeds_and_folds",
                "value": {"seeds": [0, 1, 2, 3, 4], "folds": list(range(10))},
            },
        ],
        "implementation_flexible": [
            {
                "id": "IF-MANIFEST-IO",
                "name": "opportunity manifest construction",
                "value": choice,
                "allowed_values": ["MATERIALIZE", "STREAM_CANONICAL_BYTES"],
                "guard": "Canonical JSON bytes, tuple association and SHA-256 remain identical.",
            }
        ],
        "implementation": {
            "status": "PLANNED",
            "commit": None,
            "entrypoint": None,
            "compute_plan_path": None,
            "validation_commands": [
                {
                    "argv": ["python", "-m", "unittest", "test_dg0008"],
                    "status": "PENDING",
                    "evidence": "Not run because the Study is unregistered.",
                }
            ],
            "environment": ["Python 3.9", "PyTorch deterministic algorithms"],
            "checkpointing": {
                "mode": "one fixed-final checkpoint per matrix member",
                "resume": "Retry only infrastructure-failed jobs with identical identity.",
            },
        },
        "io": {
            "inputs": [
                {
                    "id": "INPUT-RAW",
                    "requirement": "lawfully restored raw corpora",
                    "status": "MISSING",
                    "integrity": "byte-identical example/split identity manifest",
                }
            ],
            "required_outputs": [
                {
                    "id": "OUTPUT-MANIFEST",
                    "requirement": "per-epoch canonical opportunity manifests",
                    "status": "PLANNED",
                    "integrity": "proposal-defined SHA-256 tuple association",
                }
            ],
        },
        "resource_profile": {
            "constraints": [
                {
                    "name": "gpu_compatibility",
                    "value": "CUDA architecture no newer than sm_90",
                    "source": PROPOSAL_PATH + "#Expected compute",
                }
            ],
            "estimated_runtime": {
                "value": None,
                "unit": "gpu_hours",
                "basis": "Benchmark required by the approved proposal.",
            },
            "estimated_cost": {
                "value": None,
                "currency": "USD",
                "basis": "Runtime, storage lifetime and current price are unknown.",
            },
            "telemetry_required": ["job state", "heartbeat", "disk", "cost"],
        },
        "existing_contracts": {
            "study_registered": False,
            "study_plan_path": None,
            "compute_plan_path": None,
            "authorization_path": None,
            "compute_interface": "python -m improvements.compute",
        },
        "readiness": {
            "state": "BLOCKED",
            "blockers": [blocker()],
        },
        "supersedes": None,
        "change_request_ref": None,
        "workload_digest": "",
    }
    return contract.seal(document, "workload_digest")


def assessment(
    current,
    *,
    round_number=1,
    decision="REQUEST_IMPLEMENTATION_CHANGE",
    requested="STREAM_CANONICAL_BYTES",
    mode="OPERATE",
):
    changes = []
    blockers = []
    feasible = False
    if decision == "REQUEST_IMPLEMENTATION_CHANGE":
        changes = [
            {
                "id": "CR-{:03d}".format(round_number),
                "classification": "IMPLEMENTATION_FLEXIBLE",
                "target_id": "IF-MANIFEST-IO",
                "requested_value": requested,
                "reason": "Bound peak controller/worker memory without changing canonical bytes.",
            }
        ]
    elif decision == "SCIENTIFIC_CHANGE_REQUIRED":
        changes = [
            {
                "id": "CR-{:03d}".format(round_number),
                "classification": "SCIENTIFIC_INVARIANT",
                "target_id": "SI-SEEDS",
                "requested_value": {"seeds": [0]},
                "reason": "Reducing seeds would change the approved estimand.",
            }
        ]
    elif decision == "BLOCKED":
        blockers = [blocker("B-INFRA", "CAPABILITY", "job interface unavailable", "INFRA")]

    document = {
        "schema_version": 1,
        "kind": "execution_assessment",
        "assessment_id": "DP-0008-A{:02d}".format(round_number),
        "round": round_number,
        "workload_ref": contract.workload_ref(current),
        "mode": mode,
        "infra_version": {
            "repository": "infra",
            "commit": COMMIT,
            "tree_state": "DIRTY",
            "version": "0.1.0",
        },
        "decision": decision,
        "feasible": feasible,
        "selected_strategy": None,
        "resource_recommendation": [],
        "estimated_runtime": {
            "value": None,
            "unit": "gpu_hours",
            "basis": "No compatible benchmark exists.",
        },
        "estimated_cost": {
            "value": None,
            "currency": "USD",
            "basis": "Unknown runtime and no authorization.",
        },
        "expected_bottlenecks": ["opportunity-manifest memory and disk volume"],
        "observed_bottlenecks": [],
        "telemetry": {
            "deterministic": ["worker state", "job state", "artifact state", "derived cost"],
            "reasoning_triggers": ["job heartbeat stale", "disk threshold exceeded"],
        },
        "change_requests": changes,
        "failure_class": None,
        "recommended_action": "Revise safely." if changes else "Resolve blocker.",
        "policy": {
            "authorization_path": None,
            "within_budget": None,
            "reason": "DG-0008 has no authorization envelope.",
        },
        "blockers": blockers,
        "maintenance": {
            "status": "NOT_REQUIRED" if mode == "OPERATE" else "PAUSED",
            "failure_ref": None if mode == "OPERATE" else "failure://infra-command-surface",
            "fix_commit": None,
            "re_evaluation_required": mode == "MAINTAIN",
        },
        "assessment_digest": "",
    }
    return contract.seal(document, "assessment_digest")


def revise(previous, request, *, choice):
    result = copy.deepcopy(previous)
    result["revision"] += 1
    result["implementation_flexible"][0]["value"] = choice
    result["supersedes"] = contract.workload_ref(previous)
    result["change_request_ref"] = contract.assessment_ref(request)
    result["workload_digest"] = ""
    return contract.seal(result, "workload_digest")


class ApprovedProposalBindingTests(unittest.TestCase):
    def test_reference_identifies_the_exact_human_approved_proposal(self):
        reference = contract.proposal_reference(PROPOSAL_PATH, repo_root=REPO_ROOT)
        committed = subprocess.check_output(
            ["git", "show", "{}:{}".format(reference["commit"], PROPOSAL_PATH)],
            cwd=str(REPO_ROOT),
        )

        self.assertEqual(reference["proposal_id"], "DP-0008")
        self.assertEqual(reference["allocated_study_id"], "DG-0008")
        self.assertEqual(reference["status"], "APPROVED")
        self.assertRegex(reference["commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(reference["sha256"], hashlib.sha256(committed).hexdigest())

    def test_workload_fails_when_the_approved_proposal_digest_drifts(self):
        document = workload()
        document["proposal"]["sha256"] = "0" * 64
        document = contract.seal(document, "workload_digest")

        with self.assertRaises(contract.ContractError) as caught:
            contract.validate_workload(document, repo_root=REPO_ROOT)

        self.assertEqual(caught.exception.kind, "PROPOSAL_DRIFT")

    def test_workload_digest_binds_revision_content(self):
        document = workload()
        document["readiness"]["blockers"][0]["detail"] = "silently changed"

        with self.assertRaisesRegex(contract.ContractError, "does not match canonical content"):
            contract.validate_workload(document, repo_root=REPO_ROOT)


class ScientificBoundaryTests(unittest.TestCase):
    def test_implementation_flexible_change_produces_a_valid_bound_revision(self):
        first = workload()
        request = assessment(first)
        second = revise(first, request, choice="STREAM_CANONICAL_BYTES")

        result = contract.validate_revision(first, second, request, repo_root=REPO_ROOT)

        self.assertEqual(result["revision"], 2)
        self.assertEqual(
            result["implementation_flexible"][0]["value"], "STREAM_CANONICAL_BYTES"
        )
        self.assertEqual(result["scientific_invariants"], first["scientific_invariants"])

    def test_executor_revision_cannot_mutate_a_scientific_invariant(self):
        first = workload()
        request = assessment(first)
        second = revise(first, request, choice="STREAM_CANONICAL_BYTES")
        second["scientific_invariants"][1]["value"]["seeds"] = [0]
        second = contract.seal(second, "workload_digest")

        with self.assertRaises(contract.ContractError) as caught:
            contract.validate_revision(first, second, request, repo_root=REPO_ROOT)

        self.assertEqual(caught.exception.kind, "SCIENTIFIC_CHANGE_REQUIRED")

    def test_infra_scientific_request_escalates_without_a_revision_path(self):
        current = workload()
        review = assessment(current, decision="SCIENTIFIC_CHANGE_REQUIRED")
        state = contract.new_negotiation("DP-0008-N01", current["proposal"], current)

        updated = contract.apply_assessment(state, current, review)

        self.assertEqual(updated["status"], "ESCALATED_SCIENTIFIC_CHANGE")
        self.assertEqual(updated["escalation"]["class"], "SCIENTIFIC_CHANGE_REQUIRED")

    def test_flexible_request_cannot_target_an_undeclared_choice(self):
        current = workload()
        review = assessment(current)
        review["change_requests"][0]["target_id"] = "IF-UNDECLARED"
        review = contract.seal(review, "assessment_digest")

        with self.assertRaisesRegex(contract.ContractError, "implementation-flexible"):
            contract.validate_assessment(review, current)


class NegotiationTests(unittest.TestCase):
    def test_executor_to_infra_handoff_needs_only_sealed_references(self):
        current = workload()
        state = contract.new_negotiation("DP-0008-N01", current["proposal"], current)
        review = assessment(current)

        changed = contract.apply_assessment(state, current, review)

        self.assertEqual(changed["status"], "CHANGES_REQUESTED")
        self.assertFalse(changed["transcript_required"])
        self.assertEqual(changed["rounds"][0]["workload_ref"], contract.workload_ref(current))
        self.assertEqual(changed["rounds"][0]["assessment_ref"], contract.assessment_ref(review))

    def test_blocked_assessment_closes_the_preflight_without_execution(self):
        current = workload()
        state = contract.new_negotiation("DP-0008-N01", current["proposal"], current)
        review = assessment(current, decision="BLOCKED")

        changed = contract.apply_assessment(state, current, review)

        self.assertEqual(changed["status"], "BLOCKED")
        self.assertEqual(changed["rounds"][0]["decision"], "BLOCKED")

    def test_third_unresolved_change_request_escalates_without_a_fourth_round(self):
        current = workload()
        state = contract.new_negotiation("DP-0008-N01", current["proposal"], current)
        choices = ["STREAM_CANONICAL_BYTES", "MATERIALIZE", "STREAM_CANONICAL_BYTES"]

        for round_number, choice in enumerate(choices, 1):
            review = assessment(
                current,
                round_number=round_number,
                requested=choice,
            )
            state = contract.apply_assessment(state, current, review)
            if round_number < 3:
                revised = revise(current, review, choice=choice)
                state = contract.apply_revision(
                    state, current, revised, review, repo_root=REPO_ROOT
                )
                current = revised

        self.assertEqual(state["status"], "ESCALATED_CONVERGENCE_LIMIT")
        self.assertEqual(len(state["rounds"]), 3)
        self.assertEqual(state["escalation"]["class"], "CONVERGENCE_LIMIT")

    def test_transcript_dependency_is_rejected(self):
        current = workload()
        state = contract.new_negotiation("DP-0008-N01", current["proposal"], current)
        state["transcript_required"] = True
        state = contract.seal(state, "negotiation_digest")

        with self.assertRaisesRegex(contract.ContractError, "transcript_required=false"):
            contract.validate_negotiation(state)


class PolicyAndMaintenanceTests(unittest.TestCase):
    def test_accept_cannot_bypass_registration_authorization_or_budget(self):
        current = workload()
        review = assessment(current)
        review["decision"] = "ACCEPT"
        review["feasible"] = True
        review["change_requests"] = []
        review = contract.seal(review, "assessment_digest")

        with self.assertRaisesRegex(contract.ContractError, "VALIDATED workload"):
            contract.validate_assessment(review, current)

    def test_workload_cannot_replace_the_deterministic_compute_interface(self):
        current = workload()
        current["existing_contracts"]["compute_interface"] = "infra job submit"
        current = contract.seal(current, "workload_digest")

        with self.assertRaisesRegex(contract.ContractError, "improvements.compute"):
            contract.validate_workload(current, repo_root=REPO_ROOT)

    def test_maintain_mode_requires_a_paused_block_and_re_evaluation(self):
        current = workload()
        review = assessment(current, decision="BLOCKED", mode="MAINTAIN")

        validated = contract.validate_assessment(review, current)

        self.assertEqual(validated["maintenance"]["status"], "PAUSED")
        self.assertTrue(validated["maintenance"]["re_evaluation_required"])

    def test_maintain_fix_cannot_continue_from_an_uncommitted_infra_tree(self):
        current = workload()
        review = assessment(current, decision="BLOCKED", mode="MAINTAIN")
        review["maintenance"].update({"status": "FIX_COMMITTED", "fix_commit": COMMIT})
        review = contract.seal(review, "assessment_digest")

        with self.assertRaisesRegex(contract.ContractError, "clean infrastructure tree"):
            contract.validate_assessment(review, current)


class DirectoryContractTests(unittest.TestCase):
    def test_directory_validation_reconstructs_state_without_transcripts(self):
        first = workload()
        request = assessment(first)
        second = revise(first, request, choice="STREAM_CANONICAL_BYTES")
        state = contract.new_negotiation("DP-0008-N01", first["proposal"], first)
        state = contract.apply_assessment(state, first, request)
        state = contract.apply_revision(state, first, second, request, repo_root=REPO_ROOT)
        final = assessment(second, round_number=2, decision="BLOCKED")
        state = contract.apply_assessment(state, second, final)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract.write(root / "workload.v1.json", first)
            contract.write(root / "assessment.r1.json", request)
            contract.write(root / "workload.v2.json", second)
            contract.write(root / "assessment.r2.json", final)
            contract.write(root / "negotiation.json", state)

            result = contract.check_directory(root, repo_root=REPO_ROOT)

        self.assertEqual(
            result,
            {
                "valid": True,
                "workloads": 2,
                "assessments": 2,
                "negotiation_status": "BLOCKED",
                "transcript_required": False,
            },
        )

    def test_real_dp0008_slice_is_a_complete_transcript_independent_preflight(self):
        directory = (
            REPO_ROOT
            / "improvements"
            / "taskrelation"
            / "research"
            / "execution"
            / "preflights"
            / "DP-0008"
        )

        result = contract.check_directory(directory, repo_root=REPO_ROOT)

        self.assertEqual(
            result,
            {
                "valid": True,
                "workloads": 2,
                "assessments": 2,
                "negotiation_status": "BLOCKED",
                "transcript_required": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
