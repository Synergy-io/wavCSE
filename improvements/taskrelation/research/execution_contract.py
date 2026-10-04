"""Typed durable contract for the wavCSE Execution Plane V1 (INC-021).

The contract sits above the existing study compute plan and authorization envelope.
It does not replace either: a WorkloadSpecification binds an approved proposal and
records implementation readiness; an ExecutionAssessment records infrastructure
reasoning; a NegotiationState bounds their exchange.  Paid transitions remain owned
by ``improvements.compute`` and the deterministic ``infra`` CLI.
"""

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from improvements.taskrelation.research import proposal_check


SCHEMA_VERSION = 1
MAX_NEGOTIATION_ROUNDS = 3

WORKLOAD_KIND = "execution_workload"
ASSESSMENT_KIND = "execution_assessment"
NEGOTIATION_KIND = "execution_negotiation"

PLANNED = "PLANNED"
IMPLEMENTED = "IMPLEMENTED"
VALIDATED = "VALIDATED"
IMPLEMENTATION_STATUSES = (PLANNED, IMPLEMENTED, VALIDATED)

ACCEPT = "ACCEPT"
REQUEST_IMPLEMENTATION_CHANGE = "REQUEST_IMPLEMENTATION_CHANGE"
BLOCKED = "BLOCKED"
SCIENTIFIC_CHANGE_REQUIRED = "SCIENTIFIC_CHANGE_REQUIRED"
ASSESSMENT_DECISIONS = (
    ACCEPT,
    REQUEST_IMPLEMENTATION_CHANGE,
    BLOCKED,
    SCIENTIFIC_CHANGE_REQUIRED,
)

OPERATE = "OPERATE"
MAINTAIN = "MAINTAIN"
ASSESSMENT_MODES = (OPERATE, MAINTAIN)

FLEXIBLE = "IMPLEMENTATION_FLEXIBLE"
SCIENTIFIC_INVARIANT = "SCIENTIFIC_INVARIANT"
CHANGE_CLASSES = (FLEXIBLE, SCIENTIFIC_INVARIANT)

PENDING_INFRA = "PENDING_INFRA"
CHANGES_REQUESTED = "CHANGES_REQUESTED"
PREFLIGHT_ACCEPTED = "PREFLIGHT_ACCEPTED"
NEGOTIATION_BLOCKED = "BLOCKED"
ESCALATED_SCIENTIFIC = "ESCALATED_SCIENTIFIC_CHANGE"
ESCALATED_CONVERGENCE = "ESCALATED_CONVERGENCE_LIMIT"
NEGOTIATION_STATES = (
    PENDING_INFRA,
    CHANGES_REQUESTED,
    PREFLIGHT_ACCEPTED,
    NEGOTIATION_BLOCKED,
    ESCALATED_SCIENTIFIC,
    ESCALATED_CONVERGENCE,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Z0-9][A-Z0-9._-]*$")

_WORKLOAD_KEYS = {
    "schema_version",
    "kind",
    "workload_id",
    "revision",
    "proposal",
    "scientific_invariants",
    "implementation_flexible",
    "implementation",
    "io",
    "resource_profile",
    "existing_contracts",
    "readiness",
    "supersedes",
    "change_request_ref",
    "workload_digest",
}
_ASSESSMENT_KEYS = {
    "schema_version",
    "kind",
    "assessment_id",
    "round",
    "workload_ref",
    "mode",
    "infra_version",
    "decision",
    "feasible",
    "selected_strategy",
    "resource_recommendation",
    "estimated_runtime",
    "estimated_cost",
    "expected_bottlenecks",
    "observed_bottlenecks",
    "telemetry",
    "change_requests",
    "failure_class",
    "recommended_action",
    "policy",
    "blockers",
    "maintenance",
    "assessment_digest",
}
_NEGOTIATION_KEYS = {
    "schema_version",
    "kind",
    "negotiation_id",
    "proposal_ref",
    "max_rounds",
    "status",
    "current_workload_ref",
    "rounds",
    "escalation",
    "transcript_required",
    "negotiation_digest",
}


class ContractError(ValueError):
    """One fail-closed execution-contract violation."""

    def __init__(self, message, kind="CONTRACT_INVALID"):
        super().__init__(message)
        self.kind = kind

    def as_dict(self):
        return {"kind": self.kind, "error": str(self)}


def canonical_bytes(document, *, omit=()):
    """Canonical bytes used for content identity, independent of file formatting."""

    payload = copy.deepcopy(document)
    for key in omit:
        payload.pop(key, None)
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError("document is not canonical JSON: {}".format(exc)) from exc
    return text.encode("utf-8")


def digest(document, *, omit=()):
    return hashlib.sha256(canonical_bytes(document, omit=omit)).hexdigest()


def render(document):
    """Stable, reviewable file representation; identity uses :func:`canonical_bytes`."""

    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def seal(document, field):
    result = copy.deepcopy(document)
    result[field] = digest(result, omit=(field,))
    return result


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError("cannot read JSON contract {}: {}".format(path, exc)) from exc


def write(path, document):
    """Write a validated contract artifact; callers still own repository policy."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render(document), encoding="utf-8")
    return destination


def proposal_reference(path, repo_root=None, commit=None):
    """Return an immutable Git reference to one valid, human-approved proposal."""

    root = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()
    proposal_path = Path(path)
    if not proposal_path.is_absolute():
        proposal_path = root / proposal_path
    proposal_path = proposal_path.resolve()
    proposals_root = root / "improvements" / "taskrelation" / "research" / "proposals"
    try:
        relative = proposal_path.relative_to(root)
        proposal_path.relative_to(proposals_root)
    except ValueError as exc:
        raise ContractError("proposal path must be inside the repository proposals directory") from exc

    revision = commit
    if revision is None:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            universal_newlines=True,
        )
        if completed.returncode != 0:
            raise ContractError("cannot resolve the commit containing the approved proposal")
        revision = completed.stdout.strip()
    if not _COMMIT.match(str(revision)):
        raise ContractError("approved proposal reference requires a full Git commit")

    completed = subprocess.run(
        ["git", "show", "{}:{}".format(revision, relative.as_posix())],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ContractError(
            "approved proposal {} is not present at commit {}".format(relative, revision)
        )
    payload = completed.stdout
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError("approved proposal is not UTF-8 at commit {}".format(revision)) from exc
    entries, parse_problems = proposal_check.parse_frontmatter(text)
    if entries is None:
        raise ContractError("approved proposal has no frontmatter")
    fields = dict(entries)
    registered = proposal_check.load_registered_ids(proposals_root.parent)
    kind, problems = proposal_check.check_text(text, proposals_root.parent, registered)
    if kind != "proposal" or parse_problems or problems:
        messages = list(parse_problems) + list(problems)
        raise ContractError(
            "approved proposal does not satisfy proposal_schema: {}".format("; ".join(messages))
        )
    if fields.get("status") != "APPROVED" or fields.get("review") != "PASS":
        raise ContractError("execution requires an APPROVED proposal with reviewer PASS")
    if not fields.get("approved_by") or not fields.get("approved_at"):
        raise ContractError("approved proposal lacks human approval identity or timestamp")
    return {
        "proposal_id": fields["proposal_id"],
        "allocated_study_id": fields["allocated_study_id"],
        "path": relative.as_posix(),
        "commit": revision,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "status": fields["status"],
        "approved_by": fields["approved_by"],
        "approved_at": fields["approved_at"],
    }


def workload_ref(workload):
    return {
        "workload_id": workload["workload_id"],
        "revision": workload["revision"],
        "digest": workload["workload_digest"],
    }


def assessment_ref(assessment):
    return {
        "assessment_id": assessment["assessment_id"],
        "round": assessment["round"],
        "digest": assessment["assessment_digest"],
    }


def validate_workload(document, repo_root=None):
    _mapping(document, "workload")
    _exact_keys(document, _WORKLOAD_KEYS, "workload")
    _header(document, WORKLOAD_KIND, "workload")
    _identifier(document["workload_id"], "workload.workload_id")
    _positive_int(document["revision"], "workload.revision")
    _validate_proposal(document["proposal"], repo_root=repo_root)

    invariants = _nonempty_list(document["scientific_invariants"], "scientific_invariants")
    invariant_ids = set()
    for index, invariant in enumerate(invariants):
        where = "scientific_invariants[{}]".format(index)
        _exact_keys(invariant, {"id", "source", "name", "value"}, where)
        identifier = _identifier(invariant["id"], where + ".id")
        if identifier in invariant_ids:
            raise ContractError("{} duplicates scientific invariant {}".format(where, identifier))
        invariant_ids.add(identifier)
        _text(invariant["source"], where + ".source")
        _text(invariant["name"], where + ".name")
        canonical_bytes(invariant["value"])

    choices = _list(document["implementation_flexible"], "implementation_flexible")
    choice_ids = set()
    for index, choice in enumerate(choices):
        where = "implementation_flexible[{}]".format(index)
        _exact_keys(choice, {"id", "name", "value", "allowed_values", "guard"}, where)
        identifier = _identifier(choice["id"], where + ".id")
        if identifier in choice_ids or identifier in invariant_ids:
            raise ContractError("{} duplicates contract identifier {}".format(where, identifier))
        choice_ids.add(identifier)
        _text(choice["name"], where + ".name")
        allowed = _nonempty_list(choice["allowed_values"], where + ".allowed_values")
        if choice["value"] not in allowed:
            raise ContractError("{}.value is not one of its allowed_values".format(where))
        _text(choice["guard"], where + ".guard")

    _validate_implementation(document["implementation"])
    _validate_io(document["io"])
    _validate_resource_profile(document["resource_profile"])
    contracts = _validate_existing_contracts(document["existing_contracts"])
    readiness = _validate_readiness(document["readiness"])
    _optional_ref(document["supersedes"], "supersedes", workload=True)
    _optional_ref(document["change_request_ref"], "change_request_ref", assessment=True)

    if document["revision"] == 1:
        if document["supersedes"] is not None or document["change_request_ref"] is not None:
            raise ContractError("workload revision 1 cannot supersede or answer another artifact")
    elif document["supersedes"] is None or document["change_request_ref"] is None:
        raise ContractError("revised workload requires supersedes and change_request_ref")

    if document["implementation"]["status"] == VALIDATED:
        if not contracts["study_registered"]:
            raise ContractError("VALIDATED workload requires a registered study")
        for key in ("study_plan_path", "compute_plan_path"):
            if not contracts[key]:
                raise ContractError("VALIDATED workload requires existing_contracts.{}".format(key))
        if readiness["state"] != "READY":
            raise ContractError("VALIDATED workload must have readiness.state READY")

    _sealed(document, "workload_digest", "workload")
    return document


def validate_assessment(document, workload):
    validate_workload(workload)
    _mapping(document, "assessment")
    _exact_keys(document, _ASSESSMENT_KEYS, "assessment")
    _header(document, ASSESSMENT_KIND, "assessment")
    _identifier(document["assessment_id"], "assessment.assessment_id")
    _positive_int(document["round"], "assessment.round")
    _same_ref(document["workload_ref"], workload_ref(workload), "assessment.workload_ref")
    _one_of(document["mode"], ASSESSMENT_MODES, "assessment.mode")
    _validate_infra_version(document["infra_version"])
    decision = _one_of(document["decision"], ASSESSMENT_DECISIONS, "assessment.decision")
    if not isinstance(document["feasible"], bool):
        raise ContractError("assessment.feasible must be boolean")
    if document["selected_strategy"] is not None:
        _text(document["selected_strategy"], "assessment.selected_strategy")
    _facts(document["resource_recommendation"], "assessment.resource_recommendation")
    _estimate(document["estimated_runtime"], "assessment.estimated_runtime")
    _cost(document["estimated_cost"], "assessment.estimated_cost")
    _strings(document["expected_bottlenecks"], "assessment.expected_bottlenecks")
    _strings(document["observed_bottlenecks"], "assessment.observed_bottlenecks")
    _validate_telemetry(document["telemetry"])
    requests = _validate_change_requests(document["change_requests"], workload)
    if document["failure_class"] is not None:
        _text(document["failure_class"], "assessment.failure_class")
    _text(document["recommended_action"], "assessment.recommended_action")
    policy = _validate_policy(document["policy"])
    blockers = _blockers(document["blockers"], "assessment.blockers")
    maintenance = _validate_maintenance(document["maintenance"], document["mode"])

    classes = {request["classification"] for request in requests}
    if decision == REQUEST_IMPLEMENTATION_CHANGE:
        if not requests or classes != {FLEXIBLE}:
            raise ContractError(
                "REQUEST_IMPLEMENTATION_CHANGE requires only implementation-flexible requests"
            )
    elif decision == SCIENTIFIC_CHANGE_REQUIRED:
        if SCIENTIFIC_INVARIANT not in classes:
            raise ContractError(
                "SCIENTIFIC_CHANGE_REQUIRED must name a scientific invariant"
            )
    elif requests:
        raise ContractError("{} assessment cannot carry change_requests".format(decision))

    if decision == BLOCKED and not blockers:
        raise ContractError("BLOCKED assessment requires at least one blocker")
    if decision == ACCEPT:
        if not document["feasible"] or blockers:
            raise ContractError("ACCEPT requires feasible=true and no blockers")
        if workload["implementation"]["status"] != VALIDATED:
            raise ContractError("ACCEPT requires a VALIDATED workload")
        if not policy["within_budget"] or not policy["authorization_path"]:
            raise ContractError("ACCEPT requires a bounded committed authorization")
        if document["infra_version"]["tree_state"] != "CLEAN":
            raise ContractError("ACCEPT requires a committed clean infrastructure version")
    if document["mode"] == MAINTAIN and decision != BLOCKED:
        raise ContractError("MAINTAIN mode always pauses as BLOCKED before re-evaluation")
    if maintenance["status"] == "FIX_COMMITTED":
        if not maintenance["fix_commit"]:
            raise ContractError("FIX_COMMITTED maintenance requires fix_commit")
        if document["infra_version"]["tree_state"] != "CLEAN":
            raise ContractError("FIX_COMMITTED maintenance requires a clean infrastructure tree")
        if maintenance["fix_commit"] != document["infra_version"]["commit"]:
            raise ContractError("maintenance fix_commit must identify the assessed infra version")

    _sealed(document, "assessment_digest", "assessment")
    return document


def validate_revision(previous, revised, assessment, repo_root=None):
    """Prove that an Executor revision answers only a permitted flexible request."""

    validate_workload(previous, repo_root=repo_root)
    validate_assessment(assessment, previous)
    validate_workload(revised, repo_root=repo_root)
    if assessment["decision"] != REQUEST_IMPLEMENTATION_CHANGE:
        raise ContractError("workload revision requires REQUEST_IMPLEMENTATION_CHANGE")
    if revised["revision"] != previous["revision"] + 1:
        raise ContractError("workload revision must increase by exactly one")
    if revised["workload_id"] != previous["workload_id"]:
        raise ContractError("workload identity cannot change during negotiation")
    if revised["proposal"] != previous["proposal"]:
        raise ContractError("proposal binding cannot change during negotiation")
    if revised["scientific_invariants"] != previous["scientific_invariants"]:
        raise ContractError(
            "scientific invariants are immutable; escalate instead of revising",
            kind="SCIENTIFIC_CHANGE_REQUIRED",
        )
    _same_ref(revised["supersedes"], workload_ref(previous), "revised.supersedes")
    _same_ref(
        revised["change_request_ref"], assessment_ref(assessment), "revised.change_request_ref"
    )

    old_choices = {item["id"]: item for item in previous["implementation_flexible"]}
    new_choices = {item["id"]: item for item in revised["implementation_flexible"]}
    if set(old_choices) != set(new_choices):
        raise ContractError("implementation-flexible choice identities cannot change")
    requested = {item["target_id"]: item["requested_value"] for item in assessment["change_requests"]}
    for identifier, old in old_choices.items():
        new = new_choices[identifier]
        if {key: value for key, value in new.items() if key != "value"} != {
            key: value for key, value in old.items() if key != "value"
        }:
            raise ContractError("revision may change only flexible choice values")
        expected = requested.get(identifier, old["value"])
        if new["value"] != expected:
            raise ContractError(
                "revision choice {} must equal requested value {!r}".format(identifier, expected)
            )
    return revised


def new_negotiation(negotiation_id, proposal_ref_value, workload, max_rounds=None):
    validate_workload(workload)
    if proposal_ref_value != workload["proposal"]:
        raise ContractError("negotiation proposal_ref must equal the workload proposal binding")
    maximum = MAX_NEGOTIATION_ROUNDS if max_rounds is None else max_rounds
    _positive_int(maximum, "max_rounds")
    if maximum > MAX_NEGOTIATION_ROUNDS:
        raise ContractError(
            "negotiation max_rounds cannot exceed {}".format(MAX_NEGOTIATION_ROUNDS)
        )
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": NEGOTIATION_KIND,
        "negotiation_id": negotiation_id,
        "proposal_ref": copy.deepcopy(proposal_ref_value),
        "max_rounds": maximum,
        "status": PENDING_INFRA,
        "current_workload_ref": workload_ref(workload),
        "rounds": [],
        "escalation": None,
        "transcript_required": False,
        "negotiation_digest": "",
    }
    return seal(document, "negotiation_digest")


def apply_assessment(negotiation, workload, assessment):
    """Record one Infra assessment and choose the deterministic next state."""

    validate_negotiation(negotiation)
    validate_assessment(assessment, workload)
    _same_ref(
        negotiation["current_workload_ref"], workload_ref(workload), "current_workload_ref"
    )
    if negotiation["status"] != PENDING_INFRA:
        raise ContractError(
            "assessment requires negotiation status PENDING_INFRA, got {}".format(
                negotiation["status"]
            )
        )
    if assessment["round"] != len(negotiation["rounds"]) + 1:
        raise ContractError("assessment round must be the next negotiation round")

    updated = copy.deepcopy(negotiation)
    updated["rounds"].append(
        {
            "round": assessment["round"],
            "workload_ref": workload_ref(workload),
            "assessment_ref": assessment_ref(assessment),
            "decision": assessment["decision"],
        }
    )
    decision = assessment["decision"]
    if decision == ACCEPT:
        updated["status"] = PREFLIGHT_ACCEPTED
    elif decision == BLOCKED:
        updated["status"] = NEGOTIATION_BLOCKED
    elif decision == SCIENTIFIC_CHANGE_REQUIRED:
        updated["status"] = ESCALATED_SCIENTIFIC
        updated["escalation"] = {
            "class": "SCIENTIFIC_CHANGE_REQUIRED",
            "reason": assessment["recommended_action"],
        }
    elif len(updated["rounds"]) >= updated["max_rounds"]:
        updated["status"] = ESCALATED_CONVERGENCE
        updated["escalation"] = {
            "class": "CONVERGENCE_LIMIT",
            "reason": "bounded Executor/Infra negotiation did not converge",
        }
    else:
        updated["status"] = CHANGES_REQUESTED
    return seal(updated, "negotiation_digest")


def apply_revision(negotiation, previous, revised, assessment, repo_root=None):
    validate_negotiation(negotiation)
    if negotiation["status"] != CHANGES_REQUESTED:
        raise ContractError("workload revision requires CHANGES_REQUESTED negotiation state")
    validate_revision(previous, revised, assessment, repo_root=repo_root)
    updated = copy.deepcopy(negotiation)
    updated["current_workload_ref"] = workload_ref(revised)
    updated["status"] = PENDING_INFRA
    return seal(updated, "negotiation_digest")


def validate_negotiation(document, workloads=None, assessments=None):
    _mapping(document, "negotiation")
    _exact_keys(document, _NEGOTIATION_KEYS, "negotiation")
    _header(document, NEGOTIATION_KIND, "negotiation")
    _identifier(document["negotiation_id"], "negotiation.negotiation_id")
    _validate_proposal_shape(document["proposal_ref"])
    maximum = _positive_int(document["max_rounds"], "negotiation.max_rounds")
    if maximum > MAX_NEGOTIATION_ROUNDS:
        raise ContractError("negotiation exceeds the global convergence limit")
    _one_of(document["status"], NEGOTIATION_STATES, "negotiation.status")
    _workload_ref(document["current_workload_ref"], "negotiation.current_workload_ref")
    rounds = _list(document["rounds"], "negotiation.rounds")
    if len(rounds) > maximum:
        raise ContractError("negotiation records more rounds than max_rounds")
    for index, item in enumerate(rounds):
        where = "negotiation.rounds[{}]".format(index)
        _exact_keys(item, {"round", "workload_ref", "assessment_ref", "decision"}, where)
        if _positive_int(item["round"], where + ".round") != index + 1:
            raise ContractError("{} is not sequential".format(where))
        _workload_ref(item["workload_ref"], where + ".workload_ref")
        _assessment_ref(item["assessment_ref"], where + ".assessment_ref")
        _one_of(item["decision"], ASSESSMENT_DECISIONS, where + ".decision")
    if document["escalation"] is not None:
        _exact_keys(document["escalation"], {"class", "reason"}, "negotiation.escalation")
        _text(document["escalation"]["class"], "negotiation.escalation.class")
        _text(document["escalation"]["reason"], "negotiation.escalation.reason")
    if document["transcript_required"] is not False:
        raise ContractError("durable negotiation must set transcript_required=false")
    if document["status"].startswith("ESCALATED") and document["escalation"] is None:
        raise ContractError("escalated negotiation requires escalation detail")
    _sealed(document, "negotiation_digest", "negotiation")

    if workloads is not None and assessments is not None:
        workload_by_digest = {item["workload_digest"]: item for item in workloads}
        assessment_by_digest = {item["assessment_digest"]: item for item in assessments}
        current = workload_by_digest.get(document["current_workload_ref"]["digest"])
        if current is None:
            raise ContractError("negotiation current_workload_ref is absent")
        if document["proposal_ref"] != current["proposal"]:
            raise ContractError("negotiation proposal_ref differs from the current workload")
        for item in rounds:
            workload = workload_by_digest.get(item["workload_ref"]["digest"])
            assessment = assessment_by_digest.get(item["assessment_ref"]["digest"])
            if workload is None or assessment is None:
                raise ContractError("negotiation references an absent durable artifact")
            validate_assessment(assessment, workload)
    return document


def check_directory(path, repo_root=None):
    """Validate every durable artifact in one preflight directory."""

    directory = Path(path)
    workloads = []
    assessments = []
    negotiations = []
    for artifact in sorted(directory.glob("*.json")):
        document = load(artifact)
        kind = document.get("kind") if isinstance(document, dict) else None
        if kind == WORKLOAD_KIND:
            validate_workload(document, repo_root=repo_root)
            workloads.append(document)
        elif kind == ASSESSMENT_KIND:
            assessments.append(document)
        elif kind == NEGOTIATION_KIND:
            negotiations.append(document)
        else:
            raise ContractError("{} has unknown execution contract kind {!r}".format(artifact, kind))
    if not workloads or not assessments or len(negotiations) != 1:
        raise ContractError(
            "preflight directory requires workloads, assessments, and one negotiation"
        )
    workload_by_digest = {item["workload_digest"]: item for item in workloads}
    for assessment in assessments:
        workload = workload_by_digest.get(assessment.get("workload_ref", {}).get("digest"))
        if workload is None:
            raise ContractError("assessment references an absent workload digest")
        validate_assessment(assessment, workload)
    assessment_by_digest = {item["assessment_digest"]: item for item in assessments}
    for revised in workloads:
        if revised["revision"] == 1:
            continue
        previous = workload_by_digest.get(revised["supersedes"]["digest"])
        request = assessment_by_digest.get(revised["change_request_ref"]["digest"])
        if previous is None or request is None:
            raise ContractError("revised workload references an absent prior artifact")
        validate_revision(previous, revised, request, repo_root=repo_root)
    validate_negotiation(negotiations[0], workloads=workloads, assessments=assessments)
    return {
        "valid": True,
        "workloads": len(workloads),
        "assessments": len(assessments),
        "negotiation_status": negotiations[0]["status"],
        "transcript_required": negotiations[0]["transcript_required"],
    }


def _validate_proposal(value, repo_root=None):
    _validate_proposal_shape(value)
    current = proposal_reference(value["path"], repo_root=repo_root, commit=value["commit"])
    if value != current:
        raise ContractError(
            "proposal reference no longer matches the approved proposal bytes",
            kind="PROPOSAL_DRIFT",
        )
    return value


def _validate_proposal_shape(value):
    _exact_keys(
        value,
        {
            "proposal_id",
            "allocated_study_id",
            "path",
            "sha256",
            "commit",
            "status",
            "approved_by",
            "approved_at",
        },
        "proposal_ref",
    )
    _identifier(value["proposal_id"], "proposal_ref.proposal_id")
    _identifier(value["allocated_study_id"], "proposal_ref.allocated_study_id")
    _relative(value["path"], "proposal_ref.path")
    if not _COMMIT.match(str(value["commit"])):
        raise ContractError("proposal_ref.commit must be a full Git commit")
    if not _SHA256.match(str(value["sha256"])):
        raise ContractError("proposal_ref.sha256 must be a full SHA-256")
    if value["status"] != "APPROVED":
        raise ContractError("proposal_ref.status must be APPROVED")
    _text(value["approved_by"], "proposal_ref.approved_by")
    _text(value["approved_at"], "proposal_ref.approved_at")
    return value


def _validate_implementation(value):
    _exact_keys(
        value,
        {
            "status",
            "commit",
            "entrypoint",
            "compute_plan_path",
            "validation_commands",
            "environment",
            "checkpointing",
        },
        "implementation",
    )
    status = _one_of(value["status"], IMPLEMENTATION_STATUSES, "implementation.status")
    if value["commit"] is not None and not _COMMIT.match(str(value["commit"])):
        raise ContractError("implementation.commit must be a full Git commit")
    if status in (IMPLEMENTED, VALIDATED) and value["commit"] is None:
        raise ContractError("{} implementation requires an exact commit".format(status))
    if value["entrypoint"] is not None:
        _exact_keys(value["entrypoint"], {"argv", "working_directory"}, "implementation.entrypoint")
        _argv(value["entrypoint"]["argv"], "implementation.entrypoint.argv")
        _relative(value["entrypoint"]["working_directory"], "implementation.entrypoint.working_directory")
    if value["compute_plan_path"] is not None:
        _relative(value["compute_plan_path"], "implementation.compute_plan_path")
    commands = _list(value["validation_commands"], "implementation.validation_commands")
    for index, command in enumerate(commands):
        where = "implementation.validation_commands[{}]".format(index)
        _exact_keys(command, {"argv", "status", "evidence"}, where)
        _argv(command["argv"], where + ".argv")
        _one_of(command["status"], ("PENDING", "PASS", "FAIL"), where + ".status")
        _text(command["evidence"], where + ".evidence", allow_empty=True)
    _strings(value["environment"], "implementation.environment")
    _exact_keys(value["checkpointing"], {"mode", "resume"}, "implementation.checkpointing")
    _text(value["checkpointing"]["mode"], "implementation.checkpointing.mode")
    _text(value["checkpointing"]["resume"], "implementation.checkpointing.resume")
    return value


def _validate_io(value):
    _exact_keys(value, {"inputs", "required_outputs"}, "io")
    for name in ("inputs", "required_outputs"):
        entries = _list(value[name], "io." + name)
        identifiers = set()
        for index, entry in enumerate(entries):
            where = "io.{}[{}]".format(name, index)
            _exact_keys(entry, {"id", "requirement", "status", "integrity"}, where)
            identifier = _identifier(entry["id"], where + ".id")
            if identifier in identifiers:
                raise ContractError("{} duplicates {}".format(where, identifier))
            identifiers.add(identifier)
            _text(entry["requirement"], where + ".requirement")
            _text(entry["status"], where + ".status")
            _text(entry["integrity"], where + ".integrity")
    return value


def _validate_resource_profile(value):
    _exact_keys(
        value,
        {"constraints", "estimated_runtime", "estimated_cost", "telemetry_required"},
        "resource_profile",
    )
    _facts(value["constraints"], "resource_profile.constraints")
    _estimate(value["estimated_runtime"], "resource_profile.estimated_runtime")
    _cost(value["estimated_cost"], "resource_profile.estimated_cost")
    _strings(value["telemetry_required"], "resource_profile.telemetry_required")
    return value


def _validate_existing_contracts(value):
    _exact_keys(
        value,
        {
            "study_registered",
            "study_plan_path",
            "compute_plan_path",
            "authorization_path",
            "compute_interface",
        },
        "existing_contracts",
    )
    if not isinstance(value["study_registered"], bool):
        raise ContractError("existing_contracts.study_registered must be boolean")
    for name in ("study_plan_path", "compute_plan_path", "authorization_path"):
        if value[name] is not None:
            _relative(value[name], "existing_contracts." + name)
    if not value["study_registered"] and any(
        value[name] is not None for name in ("study_plan_path", "compute_plan_path", "authorization_path")
    ):
        raise ContractError("unregistered workload cannot claim study or authorization artifacts")
    if value["compute_interface"] != "python -m improvements.compute":
        raise ContractError("paid execution must remain authoritative in improvements.compute")
    return value


def _validate_readiness(value):
    _exact_keys(value, {"state", "blockers"}, "readiness")
    _one_of(value["state"], ("PLANNING", "BLOCKED", "READY"), "readiness.state")
    blockers = _blockers(value["blockers"], "readiness.blockers")
    if value["state"] == "BLOCKED" and not blockers:
        raise ContractError("BLOCKED workload requires readiness blockers")
    if value["state"] == "READY" and blockers:
        raise ContractError("READY workload cannot retain blockers")
    return value


def _validate_infra_version(value):
    _exact_keys(value, {"repository", "commit", "tree_state", "version"}, "infra_version")
    _text(value["repository"], "infra_version.repository")
    if not _COMMIT.match(str(value["commit"])):
        raise ContractError("infra_version.commit must be a full Git commit")
    _one_of(value["tree_state"], ("CLEAN", "DIRTY"), "infra_version.tree_state")
    _text(value["version"], "infra_version.version")
    return value


def _validate_change_requests(requests, workload):
    entries = _list(requests, "assessment.change_requests")
    choices = {item["id"]: item for item in workload["implementation_flexible"]}
    invariants = {item["id"]: item for item in workload["scientific_invariants"]}
    identifiers = set()
    for index, request in enumerate(entries):
        where = "assessment.change_requests[{}]".format(index)
        _exact_keys(
            request,
            {"id", "classification", "target_id", "requested_value", "reason"},
            where,
        )
        identifier = _identifier(request["id"], where + ".id")
        if identifier in identifiers:
            raise ContractError("{} duplicates change request {}".format(where, identifier))
        identifiers.add(identifier)
        classification = _one_of(request["classification"], CHANGE_CLASSES, where + ".classification")
        target = _identifier(request["target_id"], where + ".target_id")
        _text(request["reason"], where + ".reason")
        if classification == FLEXIBLE:
            if target not in choices:
                raise ContractError("{} does not target an implementation-flexible choice".format(where))
            if request["requested_value"] not in choices[target]["allowed_values"]:
                raise ContractError("{} requests a value outside the approved flexible set".format(where))
        elif target not in invariants:
            raise ContractError("{} does not target a scientific invariant".format(where))
    return entries


def _validate_policy(value):
    _exact_keys(value, {"authorization_path", "within_budget", "reason"}, "assessment.policy")
    if value["authorization_path"] is not None:
        _relative(value["authorization_path"], "assessment.policy.authorization_path")
    if value["within_budget"] not in (True, False, None):
        raise ContractError("assessment.policy.within_budget must be true, false, or null")
    _text(value["reason"], "assessment.policy.reason")
    return value


def _validate_maintenance(value, mode):
    _exact_keys(
        value,
        {"status", "failure_ref", "fix_commit", "re_evaluation_required"},
        "assessment.maintenance",
    )
    status = _one_of(
        value["status"],
        ("NOT_REQUIRED", "PAUSED", "FIX_REQUIRED", "FIX_COMMITTED"),
        "assessment.maintenance.status",
    )
    if value["failure_ref"] is not None:
        _text(value["failure_ref"], "assessment.maintenance.failure_ref")
    if value["fix_commit"] is not None and not _COMMIT.match(str(value["fix_commit"])):
        raise ContractError("assessment.maintenance.fix_commit must be a full commit")
    if not isinstance(value["re_evaluation_required"], bool):
        raise ContractError("assessment.maintenance.re_evaluation_required must be boolean")
    if mode == OPERATE and status != "NOT_REQUIRED":
        raise ContractError("OPERATE assessment cannot claim an infra maintenance transition")
    if mode == MAINTAIN:
        if status == "NOT_REQUIRED" or not value["re_evaluation_required"]:
            raise ContractError("MAINTAIN assessment must pause and require re-evaluation")
    return value


def _validate_telemetry(value):
    _exact_keys(value, {"deterministic", "reasoning_triggers"}, "assessment.telemetry")
    _strings(value["deterministic"], "assessment.telemetry.deterministic")
    _strings(value["reasoning_triggers"], "assessment.telemetry.reasoning_triggers")
    return value


def _facts(value, where):
    entries = _list(value, where)
    for index, item in enumerate(entries):
        item_where = "{}[{}]".format(where, index)
        _exact_keys(item, {"name", "value", "source"}, item_where)
        _text(item["name"], item_where + ".name")
        canonical_bytes(item["value"])
        _text(item["source"], item_where + ".source")
    return entries


def _estimate(value, where):
    _exact_keys(value, {"value", "unit", "basis"}, where)
    _nonnegative_or_none(value["value"], where + ".value")
    _text(value["unit"], where + ".unit")
    _text(value["basis"], where + ".basis")
    return value


def _cost(value, where):
    _exact_keys(value, {"value", "currency", "basis"}, where)
    _nonnegative_or_none(value["value"], where + ".value")
    _text(value["currency"], where + ".currency")
    _text(value["basis"], where + ".basis")
    return value


def _blockers(value, where):
    entries = _list(value, where)
    identifiers = set()
    for index, blocker in enumerate(entries):
        item_where = "{}[{}]".format(where, index)
        _exact_keys(blocker, {"id", "class", "detail", "owner"}, item_where)
        identifier = _identifier(blocker["id"], item_where + ".id")
        if identifier in identifiers:
            raise ContractError("{} duplicates blocker {}".format(item_where, identifier))
        identifiers.add(identifier)
        _text(blocker["class"], item_where + ".class")
        _text(blocker["detail"], item_where + ".detail")
        _text(blocker["owner"], item_where + ".owner")
    return entries


def _header(value, kind, where):
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("{}.schema_version must be {}".format(where, SCHEMA_VERSION))
    if value.get("kind") != kind:
        raise ContractError("{}.kind must be {!r}".format(where, kind))


def _sealed(value, field, where):
    actual = value.get(field)
    if not _SHA256.match(str(actual or "")):
        raise ContractError("{}.{} must be a full SHA-256".format(where, field))
    expected = digest(value, omit=(field,))
    if actual != expected:
        raise ContractError("{}.{} does not match canonical content".format(where, field))


def _optional_ref(value, where, workload=False, assessment=False):
    if value is None:
        return None
    if workload:
        return _workload_ref(value, where)
    if assessment:
        return _assessment_ref(value, where)
    raise AssertionError("unknown reference type")


def _workload_ref(value, where):
    _exact_keys(value, {"workload_id", "revision", "digest"}, where)
    _identifier(value["workload_id"], where + ".workload_id")
    _positive_int(value["revision"], where + ".revision")
    if not _SHA256.match(str(value["digest"])):
        raise ContractError("{}.digest must be a full SHA-256".format(where))
    return value


def _assessment_ref(value, where):
    _exact_keys(value, {"assessment_id", "round", "digest"}, where)
    _identifier(value["assessment_id"], where + ".assessment_id")
    _positive_int(value["round"], where + ".round")
    if not _SHA256.match(str(value["digest"])):
        raise ContractError("{}.digest must be a full SHA-256".format(where))
    return value


def _same_ref(actual, expected, where):
    if actual != expected:
        raise ContractError("{} does not identify the required durable artifact".format(where))
    return actual


def _mapping(value, where):
    if not isinstance(value, dict):
        raise ContractError("{} must be an object".format(where))
    return value


def _exact_keys(value, expected, where):
    _mapping(value, where)
    actual = set(value)
    unknown = sorted(actual - set(expected))
    missing = sorted(set(expected) - actual)
    if unknown or missing:
        detail = []
        if unknown:
            detail.append("unknown: {}".format(", ".join(unknown)))
        if missing:
            detail.append("missing: {}".format(", ".join(missing)))
        raise ContractError("{} keys invalid ({})".format(where, "; ".join(detail)))
    return value


def _identifier(value, where):
    text = _text(value, where)
    if not _IDENTIFIER.match(text):
        raise ContractError("{} is not a stable identifier".format(where))
    return text


def _text(value, where, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ContractError("{} must be {}string".format(where, "a " if allow_empty else "a non-empty "))
    return value


def _relative(value, where):
    text = _text(value, where)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ContractError("{} must be repository-relative without traversal".format(where))
    return text


def _one_of(value, allowed, where):
    if value not in allowed:
        raise ContractError("{} must be one of {}, got {!r}".format(where, allowed, value))
    return value


def _list(value, where):
    if not isinstance(value, list):
        raise ContractError("{} must be an array".format(where))
    return value


def _nonempty_list(value, where):
    entries = _list(value, where)
    if not entries:
        raise ContractError("{} must not be empty".format(where))
    return entries


def _strings(value, where):
    entries = _list(value, where)
    for index, item in enumerate(entries):
        _text(item, "{}[{}]".format(where, index))
    return entries


def _argv(value, where):
    entries = _nonempty_list(value, where)
    for index, item in enumerate(entries):
        _text(item, "{}[{}]".format(where, index))
    return entries


def _positive_int(value, where):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError("{} must be a positive integer".format(where))
    return value


def _nonnegative_or_none(value, where):
    if value is None:
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("{} must be a non-negative finite number or null".format(where))
    if not math.isfinite(float(value)) or value < 0:
        raise ContractError("{} must be a non-negative finite number or null".format(where))
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m improvements.taskrelation.research.execution_contract"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="validate one durable preflight directory")
    check.add_argument("path")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "check":
            result = check_directory(arguments.path)
            print(json.dumps(result, sort_keys=True))
            return 0
        raise AssertionError("unhandled command")
    except ContractError as exc:
        print(json.dumps(exc.as_dict(), sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
