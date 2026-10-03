"""Deterministic literature-investigation lifecycle (INC-018).

An ``LT-*`` Study is the bounded literature investigation a research question is
delegated under. Before INC-018 registration was hand-editing ``STUDIES.jsonl``,
so a delegated Literature Agent could investigate but had no registered scope to
persist investigation-scoped output against. This module owns that lifecycle
deterministically, operator-side:

    Main OMP: open(question, scope) -> LT-XXXX (status active)
              delegate(LT-XXXX)         -> the one writable delegated scope
    Main OMP: complete(LT-XXXX)  -> durable outputs verified, status complete
              abandon(LT-XXXX, reason)

Authority boundary:

* Opening an LT investigation is research bookkeeping, not a scientific or
  human decision, so it needs no approval gate.
* Only this module writes ``STUDIES.jsonl``; the Literature Agent never does.
* The lifecycle vocabulary is reused from the existing Study model
  (``active`` / ``complete`` / ``abandoned``); no second status system is added.
* Exactly one investigation may be *delegated* at a time, which is what makes
  "the agent may only write under the investigation it was delegated" a
  deterministic property rather than a prompt convention.

Usage::

    python -m improvements.taskrelation.research.literature_investigation open \
        --question "..." --scope "..." [--title T] [--parent P] \
        [--role R ...] [--verdict V ...] [--request-id ID]
    python -m improvements.taskrelation.research.literature_investigation delegate LT-0003
    python -m improvements.taskrelation.research.literature_investigation complete LT-0003 --decision ...
    python -m improvements.taskrelation.research.literature_investigation abandon LT-0003 --reason ...
    python -m improvements.taskrelation.research.literature_investigation get LT-0003
    python -m improvements.taskrelation.research.literature_investigation list
"""

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from improvements.taskrelation.research import literature_assessment
from improvements.taskrelation.research import literature_synthesis


SCHEMA_VERSION = 1

STATUS_ACTIVE = "active"
STATUS_COMPLETE = "complete"
STATUS_ABANDONED = "abandoned"

# Failure taxonomy. Every case is deterministic and branchable.
INVESTIGATION_NOT_FOUND = "INVESTIGATION_NOT_FOUND"
INVESTIGATION_NOT_LITERATURE = "INVESTIGATION_NOT_LITERATURE"
INVESTIGATION_NOT_ACTIVE = "INVESTIGATION_NOT_ACTIVE"
INVESTIGATION_ALREADY_DELEGATED = "INVESTIGATION_ALREADY_DELEGATED"
OUTSIDE_DELEGATED_SCOPE = "OUTSIDE_DELEGATED_SCOPE"
INVALID_STATE_TRANSITION = "INVALID_STATE_TRANSITION"
INVALID_REQUEST = "INVALID_REQUEST"
INVESTIGATION_CONFLICT = "INVESTIGATION_CONFLICT"
STUDIES_UNREADABLE = "STUDIES_UNREADABLE"

_LT_ID = re.compile(r"^LT-(\d{4})$")
# Assessment roles/verdicts in the retained corpus are identifier tokens
# (e.g. ``family_b_estimator``, ``reject_for_F9``), so the scope vocabulary is a
# bounded identifier, not the hyphenated slug used for identity.
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

_RESEARCH_DIR = Path(__file__).resolve().parent
_DEFAULT_REPO_ROOT = _RESEARCH_DIR.parents[2]
_DEFAULT_STUDIES = _RESEARCH_DIR / "STUDIES.jsonl"


class InvestigationError(ValueError):
    """A literature-investigation request is invalid or out of lifecycle."""

    def __init__(self, message, kind=INVALID_REQUEST, detail=None):
        super(InvestigationError, self).__init__(message)
        self.kind = kind
        self.detail = dict(detail or {})

    def as_dict(self):
        return {"error": str(self), "kind": self.kind, "detail": self.detail}


# -- registry primitives -----------------------------------------------------


def _studies_path(studies_path=None):
    return Path(studies_path) if studies_path is not None else _DEFAULT_STUDIES


def _read_lines(path):
    path = Path(path)
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InvestigationError(
            "cannot read Study registry {}: {}".format(path, exc),
            kind=STUDIES_UNREADABLE,
        ) from exc


def _parse_rows(lines):
    """Return ``[(line_number, raw_text, row_dict), ...]`` for non-blank lines."""

    rows = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except ValueError as exc:
            raise InvestigationError(
                "Study registry line {} is not valid JSON: {}".format(line_number, exc),
                kind=STUDIES_UNREADABLE,
            ) from exc
        if not isinstance(raw, dict):
            raise InvestigationError(
                "Study registry line {} must be an object".format(line_number),
                kind=STUDIES_UNREADABLE,
            )
        rows.append((line_number, line, raw))
    return rows


def load_studies(studies_path=None):
    """Every registered Study row, keyed by ``study_id`` (last write wins)."""

    rows = {}
    for _number, _line, raw in _parse_rows(_read_lines(_studies_path(studies_path))):
        study_id = raw.get("study_id")
        if not isinstance(study_id, str) or not study_id.strip():
            raise InvestigationError(
                "Study registry has a row with no usable study_id",
                kind=STUDIES_UNREADABLE,
            )
        rows[study_id] = raw
    return rows


def _serialize(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _atomic_write(path, text):
    path = Path(path)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp"
    )
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, str(path))


def _append_row(studies_path, row):
    path = _studies_path(studies_path)
    lines = _read_lines(path)
    lines.append(_serialize(row))
    _atomic_write(path, "\n".join(lines) + "\n")


def _replace_row(studies_path, study_id, row):
    path = _studies_path(studies_path)
    lines = _read_lines(path)
    replaced = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.get("study_id") == study_id:
            lines[index] = _serialize(row)
            replaced = True
            break
    if not replaced:
        raise InvestigationError(
            "Study {!r} disappeared while it was being updated".format(study_id),
            kind=INVESTIGATION_NOT_FOUND,
        )
    _atomic_write(path, "\n".join(lines) + "\n")


# -- helpers -----------------------------------------------------------------


def _require_text(value, field):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise InvestigationError(
            "{} must be a non-empty trimmed string".format(field),
            kind=INVALID_REQUEST,
        )
    return value


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _literature_row(rows, investigation_id):
    _require_text(investigation_id, "investigation_id")
    row = rows.get(investigation_id)
    if row is None:
        raise InvestigationError(
            "no registered literature investigation {!r}".format(investigation_id),
            kind=INVESTIGATION_NOT_FOUND,
            detail={"investigation_id": investigation_id},
        )
    if row.get("type") != "literature" or not _LT_ID.match(investigation_id):
        raise InvestigationError(
            "Study {!r} is not a literature investigation".format(investigation_id),
            kind=INVESTIGATION_NOT_LITERATURE,
            detail={"investigation_id": investigation_id},
        )
    return row


def next_investigation_id(rows):
    """The next ``LT-%04d`` id after every registered Study id."""

    highest = 0
    for study_id in rows:
        match = _LT_ID.match(study_id)
        if match:
            highest = max(highest, int(match.group(1)))
    return "LT-{:04d}".format(highest + 1)


def _delegated_ids(rows):
    return [
        study_id
        for study_id, row in rows.items()
        if row.get("type") == "literature" and row.get("delegated") is True
    ]


def delegated_investigation(rows=None, studies_path=None):
    """The single delegated investigation id, or ``None`` when none is open."""

    rows = rows if rows is not None else load_studies(studies_path)
    delegated = sorted(_delegated_ids(rows))
    if len(delegated) > 1:
        raise InvestigationError(
            "more than one investigation is delegated: {}".format(
                ", ".join(delegated)
            ),
            kind=INVESTIGATION_CONFLICT,
            detail={"delegated": delegated},
        )
    return delegated[0] if delegated else None


def _plan_text(investigation_id, title, question, scope, parent, stopping):
    lines = [
        "# {} — {}".format(investigation_id, title),
        "",
        "Status: OPEN — registered by the Research Computer (no human approval "
        "required)",
        "Type: literature",
        "Created: {}".format(_now()),
        "",
        "## Question",
        "",
        question,
        "",
        "## Scope",
        "",
        scope,
    ]
    if parent:
        lines += ["", "## Parent context", "", parent]
    if stopping:
        lines += ["", "## Stopping criteria", "", stopping]
    return "\n".join(lines) + "\n"


def _analysis_text(investigation_id, title):
    return (
        "# {} — {}\n\n"
        "Investigation-scoped analysis. Sections below are upserted by the "
        "Literature Agent through `literature_record note`; every PaperAssessment "
        "anchor points at a heading in this artifact.\n\n"
        "## Assessment\n"
    ).format(investigation_id, title)


def _write_analysis(study_dir, investigation_id, title):
    study_dir.mkdir(parents=True, exist_ok=True)
    analysis = study_dir / "analysis.md"
    if not analysis.is_file():
        analysis.write_text(_analysis_text(investigation_id, title), encoding="utf-8")


# -- operations --------------------------------------------------------------


def open_investigation(
    question,
    scope,
    *,
    title=None,
    parent=None,
    stopping=None,
    assessment_roles=None,
    assessment_verdicts=None,
    request_id=None,
    studies_path=None,
    repo_root=None,
):
    """Register a bounded ``LT-*`` literature investigation (status ``active``).

    Idempotent on ``request_id``: a retried registration returns the existing
    investigation instead of allocating a second one. Returns a structured
    record with ``created`` naming whether a new Study was written.
    """

    question = _require_text(question, "question")
    scope = _require_text(scope, "scope")
    if parent is not None:
        parent = _require_text(parent, "parent")
    if stopping is not None:
        stopping = _require_text(stopping, "stopping")
    if request_id is not None:
        request_id = _require_text(request_id, "request_id")

    repo_root = Path(repo_root).resolve() if repo_root else _DEFAULT_REPO_ROOT
    rows = load_studies(studies_path)

    if request_id is not None:
        for study_id, row in rows.items():
            if row.get("request_id") == request_id:
                return _record(study_id, row, created=False)

    roles = _vocabulary(assessment_roles, "assessment_roles")
    verdicts = _vocabulary(assessment_verdicts, "assessment_verdicts")

    investigation_id = next_investigation_id(rows)
    title = title.strip() if isinstance(title, str) and title.strip() else (
        question[:72].rstrip()
    )
    studies_dir = _studies_path(studies_path).parent / "studies"
    study_dir = studies_dir / investigation_id
    try:
        study_path = study_dir.relative_to(repo_root).as_posix()
    except ValueError:
        study_path = study_dir.as_posix()
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "PLAN.md").write_text(
        _plan_text(investigation_id, title, question, scope, parent, stopping),
        encoding="utf-8",
    )
    _write_analysis(study_dir, investigation_id, title)

    now = _now()
    row = {
        "study_id": investigation_id,
        "type": "literature",
        "status": STATUS_ACTIVE,
        "stage": "investigation_open",
        "title": title,
        "question": question,
        "scope": scope,
        "path": study_path,
        "created_at": now,
        "started_at": now,
        "delegated": False,
    }
    if parent:
        row["parent"] = parent
    if request_id:
        row["request_id"] = request_id
    if roles or verdicts:
        row["assessment_scope"] = {}
        if roles:
            row["assessment_scope"]["roles"] = sorted(roles)
        if verdicts:
            row["assessment_scope"]["verdicts"] = sorted(verdicts)

    _append_row(studies_path, row)
    return _record(investigation_id, row, created=True)


def _vocabulary(values, field):
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        raise InvestigationError(
            "{} must be a list of lowercase-hyphenated values".format(field),
            kind=INVALID_REQUEST,
        )
    seen = []
    for value in values:
        if not isinstance(value, str) or not _TOKEN.match(value):
            raise InvestigationError(
                "{}.{} must be a bounded identifier".format(field, value),
                kind=INVALID_REQUEST,
            )
        if value not in seen:
            seen.append(value)
    return seen


def delegate_investigation(investigation_id, *, studies_path=None):
    """Mark one active investigation as the delegated, writable scope.

    At most one investigation may be delegated at a time; delegating an
    already-delegated investigation is idempotent.
    """

    rows = load_studies(studies_path)
    row = _literature_row(rows, investigation_id)
    if row.get("status") != STATUS_ACTIVE:
        raise InvestigationError(
            "investigation {!r} is not active (status {!r})".format(
                investigation_id, row.get("status")
            ),
            kind=INVESTIGATION_NOT_ACTIVE,
            detail={"investigation_id": investigation_id, "status": row.get("status")},
        )
    if row.get("delegated") is True:
        return _record(investigation_id, row, created=False)
    others = [study_id for study_id in _delegated_ids(rows)]
    if others:
        raise InvestigationError(
            "investigation {!r} is already delegated; complete it before "
            "delegating another".format(others[0]),
            kind=INVESTIGATION_ALREADY_DELEGATED,
            detail={"delegated": others[0]},
        )
    updated = dict(row)
    updated["delegated"] = True
    updated["delegated_at"] = _now()
    _replace_row(studies_path, investigation_id, updated)
    return _record(investigation_id, updated, created=False)


def complete_investigation(
    investigation_id,
    *,
    decision=None,
    summary=None,
    uncertainties=None,
    coverage_limitations=None,
    blockers=None,
    studies_path=None,
    assessments_path=None,
    repo_root=None,
    literature_dir=None,
):
    """Close an active investigation, verifying its durable outputs from state.

    Idempotent: completing an already-complete investigation returns its
    recorded completion instead of transitioning twice. It creates no
    ``FINDINGS.md`` / ``DECISIONS.md`` entry — promotion is a later, separate act.
    """

    rows = load_studies(studies_path)
    row = _literature_row(rows, investigation_id)
    if row.get("status") == STATUS_COMPLETE and row.get("completed_at"):
        return _completion_record(investigation_id, row, already_completed=True)
    if row.get("status") != STATUS_ACTIVE:
        raise InvestigationError(
            "investigation {!r} cannot complete from status {!r}".format(
                investigation_id, row.get("status")
            ),
            kind=INVALID_STATE_TRANSITION,
            detail={"investigation_id": investigation_id, "status": row.get("status")},
        )

    assessment_refs = _verified_assessments(
        investigation_id, studies_path, assessments_path, repo_root, literature_dir
    )
    synthesis_ids = _verified_syntheses(
        investigation_id, studies_path, repo_root, literature_dir
    )
    now = _now()
    completed = dict(row)
    completed["status"] = STATUS_COMPLETE
    completed["stage"] = "investigation_complete"
    completed["completed_at"] = now
    completed["delegated"] = False
    if decision is not None:
        completed["decision"] = _require_text(decision, "decision")

    result = {
        "schema_version": SCHEMA_VERSION,
        "investigation_id": investigation_id,
        "status": STATUS_COMPLETE,
        "question": row.get("question"),
        "scope": row.get("scope"),
        "created_at": row.get("created_at"),
        "completed_at": now,
        "decision": completed.get("decision"),
        "papers_assessed": sorted({ref["paper_id"] for ref in assessment_refs}),
        "assessment_count": len(assessment_refs),
        "assessments": assessment_refs,
        "synthesis_ids": synthesis_ids,
        "summary": summary,
        "uncertainties": _string_list(uncertainties, "uncertainties"),
        "coverage_limitations": _string_list(
            coverage_limitations, "coverage_limitations"
        ),
        "blockers": _string_list(blockers, "blockers"),
    }
    repo_root = Path(repo_root).resolve() if repo_root else _DEFAULT_REPO_ROOT
    study_dir = repo_root / completed["path"]
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _replace_row(studies_path, investigation_id, completed)
    return _completion_record(investigation_id, completed, already_completed=False)


def abandon_investigation(investigation_id, *, reason, studies_path=None):
    """Close an active investigation as abandoned, recording why."""

    reason = _require_text(reason, "reason")
    rows = load_studies(studies_path)
    row = _literature_row(rows, investigation_id)
    if row.get("status") != STATUS_ACTIVE:
        raise InvestigationError(
            "investigation {!r} is not active (status {!r})".format(
                investigation_id, row.get("status")
            ),
            kind=INVESTIGATION_NOT_ACTIVE,
            detail={"investigation_id": investigation_id, "status": row.get("status")},
        )
    updated = dict(row)
    updated["status"] = STATUS_ABANDONED
    updated["stage"] = "investigation_abandoned"
    updated["completed_at"] = _now()
    updated["decision"] = "ABANDONED"
    updated["abandon_reason"] = reason
    updated["delegated"] = False
    _replace_row(studies_path, investigation_id, updated)
    return _record(investigation_id, updated, created=False)


def get_investigation(investigation_id, *, studies_path=None):
    rows = load_studies(studies_path)
    row = _literature_row(rows, investigation_id)
    return _record(investigation_id, row, created=False)


def list_investigations(*, studies_path=None):
    rows = load_studies(studies_path)
    return [
        _record(study_id, row, created=False)
        for study_id, row in sorted(rows.items())
        if row.get("type") == "literature"
    ]


# -- verification ------------------------------------------------------------


def _verified_assessments(investigation_id, studies_path, assessments_path, repo_root,
                          literature_dir):
    """The investigation's assessment rows, from the validated registry."""

    kwargs = {"studies_path": studies_path}
    if assessments_path is None and literature_dir is not None:
        assessments_path = Path(literature_dir) / "assessments.jsonl"
    if assessments_path is not None:
        kwargs["assessments_path"] = assessments_path
    if repo_root is not None:
        kwargs["repo_root"] = repo_root
    if literature_dir is not None:
        kwargs["literature_dir"] = literature_dir
    try:
        registry = literature_assessment.load_assessments(**kwargs)
    except literature_assessment.AssessmentError as exc:
        raise InvestigationError(
            "cannot verify assessments: {}".format(exc),
            kind="INVESTIGATION_OUTPUT_INVALID",
            detail=exc.detail,
        ) from exc
    return [
        {
            "paper_id": record.paper_id,
            "role": record.role,
            "verdict": record.verdict,
        }
        for record in registry.records
        if record.investigation_id == investigation_id
    ]


def _verified_syntheses(investigation_id, studies_path, repo_root, literature_dir):
    """Registered synthesis ids whose provenance names this investigation."""

    kwargs = {"validate_references": False}
    if studies_path is not None:
        kwargs["studies_path"] = studies_path
    if repo_root is not None:
        kwargs["repo_root"] = repo_root
    if literature_dir is not None:
        kwargs["literature_dir"] = literature_dir
    try:
        registry = literature_synthesis.load_syntheses(**kwargs)
    except literature_synthesis.SynthesisError as exc:
        raise InvestigationError(
            "cannot verify syntheses: {}".format(exc),
            kind="INVESTIGATION_OUTPUT_INVALID",
            detail=exc.detail,
        ) from exc
    reference = "investigation:{}".format(investigation_id)
    return sorted(
        record.synthesis_id
        for record in registry.records
        if reference in record.derives_from
    )


def _string_list(values, field):
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)) or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise InvestigationError(
            "{} must be a list of non-empty strings".format(field),
            kind=INVALID_REQUEST,
        )
    return [value for value in values]


def _record(study_id, row, *, created):
    return {
        "investigation_id": study_id,
        "type": row.get("type"),
        "status": row.get("status"),
        "stage": row.get("stage"),
        "title": row.get("title"),
        "question": row.get("question"),
        "scope": row.get("scope"),
        "path": row.get("path"),
        "created_at": row.get("created_at"),
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "decision": row.get("decision"),
        "delegated": row.get("delegated", False),
        "assessment_scope": row.get("assessment_scope"),
        "created": created,
    }


def _completion_record(study_id, row, *, already_completed):
    result_path = Path(row.get("path", "")) / "result.json"
    completion = {
        "investigation_id": study_id,
        "status": row.get("status"),
        "completed_at": row.get("completed_at"),
        "decision": row.get("decision"),
        "result_path": result_path.as_posix(),
        "already_completed": already_completed,
    }
    return completion


# -- CLI ---------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    open_cmd = commands.add_parser("open", help="register a new LT investigation")
    open_cmd.add_argument("--question", required=True)
    open_cmd.add_argument("--scope", required=True)
    open_cmd.add_argument("--title")
    open_cmd.add_argument("--parent")
    open_cmd.add_argument("--stopping")
    open_cmd.add_argument("--role", action="append", dest="roles")
    open_cmd.add_argument("--verdict", action="append", dest="verdicts")
    open_cmd.add_argument("--request-id")

    delegate = commands.add_parser("delegate", help="mark the writable scope")
    delegate.add_argument("investigation_id")

    complete = commands.add_parser("complete", help="close and verify an investigation")
    complete.add_argument("investigation_id")
    complete.add_argument("--decision")
    complete.add_argument("--summary")
    complete.add_argument("--uncertainty", action="append", dest="uncertainties")
    complete.add_argument("--coverage-limitation", action="append", dest="coverage")
    complete.add_argument("--blocker", action="append", dest="blockers")

    abandon = commands.add_parser("abandon", help="close an investigation as abandoned")
    abandon.add_argument("investigation_id")
    abandon.add_argument("--reason", required=True)

    get = commands.add_parser("get", help="read one investigation")
    get.add_argument("investigation_id")

    commands.add_parser("list", help="list every literature investigation")
    commands.add_parser("validate", help="validate the Study registry")

    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "open":
            document = open_investigation(
                args.question,
                args.scope,
                title=args.title,
                parent=args.parent,
                stopping=args.stopping,
                assessment_roles=args.roles,
                assessment_verdicts=args.verdicts,
                request_id=args.request_id,
            )
        elif args.command == "delegate":
            document = delegate_investigation(args.investigation_id)
        elif args.command == "complete":
            document = complete_investigation(
                args.investigation_id,
                decision=args.decision,
                summary=args.summary,
                uncertainties=args.uncertainties,
                coverage_limitations=args.coverage,
                blockers=args.blockers,
            )
        elif args.command == "abandon":
            document = abandon_investigation(args.investigation_id, reason=args.reason)
        elif args.command == "get":
            document = get_investigation(args.investigation_id)
        elif args.command == "list":
            document = {"investigations": list_investigations()}
        else:
            rows = load_studies()
            document = {
                "studies": len(rows),
                "literature_investigations": len(
                    [row for row in rows.values() if row.get("type") == "literature"]
                ),
                "delegated": delegated_investigation(rows),
            }
        print(json.dumps(document, ensure_ascii=False, sort_keys=True))
        return 0
    except InvestigationError as exc:
        print(json.dumps(exc.as_dict(), sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
