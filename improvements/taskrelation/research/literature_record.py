"""Bounded, investigation-scoped literature-write surface (INC-018).

This is the *only* writer the Literature Agent reaches. It is deliberately not a
generic file or JSONL editor: every operation is semantically validated and its
authority is bounded to one ACTIVE, DELEGATED ``LT-*`` investigation.

    note put        investigation-scoped reasoning sections in studies/<LT>/analysis.md
    assessment put  one (investigation_id, paper_id) PaperAssessment row
    claim put       one paper-attributed, evidence-validated Claim row
    synthesis put   one registered cross-source Synthesis document

What it cannot do: create or mutate a Study, touch a research record outside its
delegated investigation (findings, decisions, failures, backlog, proposal or
authorization state), edit a card, the catalog, the primary manifest or an
admission ledger, admit a paper, acquire an artifact, or write outside the one
delegated investigation.

Authority model:

* a write is accepted only when its ``investigation_id`` names the single
  investigation currently *delegated* (see :mod:`literature_investigation`), so
  "write under LT-Y while delegated to LT-X" is structurally impossible;
* a Claim stays paper-global — it carries no ``investigation_id``. The
  delegated investigation only supplies authority to record it; evidence is
  validated through the existing claim loader;
* a Synthesis is investigation-scoped by *provenance*: its ``derives_from``
  must name the investigation, and it is registered so the durable record can
  later answer which investigation produced it and what supports it;
* every write is validated by the owning module's loader against the whole
  registry (identity, references, quotes, sort order) before it is committed;
  a rejected write changes nothing;
* re-writing the same identity updates in place (idempotent), never duplicates.

Usage::

    python -m improvements.taskrelation.research.literature_record note \
        --investigation LT-0003 --heading "Assessment" --body "..."
    python -m improvements.taskrelation.research.literature_record assessment \
        --investigation LT-0003 --paper-id goncalves-2016-mssl \
        --role family_b_estimator --verdict pass --reason "..." --anchor assessment
    python -m improvements.taskrelation.research.literature_record claim \
        --paper-id goncalves-2016-mssl --claim-id my-claim \
        --claim-type method-objective --assertion-kind paraphrase \
        --assertion "..." --evidence '[...]'
    python -m improvements.taskrelation.research.literature_record synthesis \
        --investigation LT-0003 --synthesis-id lt-0003-finding --kind synthesis \
        --status active --document LT_0003_FINDING.md \
        --derives-from '["investigation:LT-0003"]' --body "..."
    python -m improvements.taskrelation.research.literature_record validate
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from improvements.taskrelation.research import literature_assessment
from improvements.taskrelation.research import literature_claims
from improvements.taskrelation.research import literature_catalog
from improvements.taskrelation.research import literature_investigation
from improvements.taskrelation.research import literature_synthesis


# Failure taxonomy. Investigation-scope kinds are reused from the lifecycle
# module so the vocabulary stays one.
INVESTIGATION_NOT_FOUND = literature_investigation.INVESTIGATION_NOT_FOUND
INVESTIGATION_NOT_LITERATURE = literature_investigation.INVESTIGATION_NOT_LITERATURE
INVESTIGATION_NOT_ACTIVE = literature_investigation.INVESTIGATION_NOT_ACTIVE
OUTSIDE_DELEGATED_SCOPE = literature_investigation.OUTSIDE_DELEGATED_SCOPE
PAPER_NOT_FOUND = "PAPER_NOT_FOUND"
EVIDENCE_REFERENCE_INVALID = "EVIDENCE_REFERENCE_INVALID"
ASSESSMENT_CONFLICT = "ASSESSMENT_CONFLICT"
CLAIM_CONFLICT = "CLAIM_CONFLICT"
SYNTHESIS_INVALID = "SYNTHESIS_INVALID"
INVALID_REQUEST = "INVALID_REQUEST"
INVALID_STATE_TRANSITION = "INVALID_STATE_TRANSITION"

_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SURVEY_DOCUMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.md$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")

_RESEARCH_DIR = Path(__file__).resolve().parent
_DEFAULT_REPO_ROOT = _RESEARCH_DIR.parents[2]


class RecordError(ValueError):
    """A scoped literature write is invalid or out of authority."""

    def __init__(self, message, kind=INVALID_REQUEST, detail=None):
        super(RecordError, self).__init__(message)
        self.kind = kind
        self.detail = dict(detail or {})

    def as_dict(self):
        return {"error": str(self), "kind": self.kind, "detail": self.detail}


# -- path resolution ---------------------------------------------------------


def _paths(repo_root=None, studies_path=None, literature_dir=None):
    repo_root = Path(repo_root).resolve() if repo_root else _DEFAULT_REPO_ROOT
    research = repo_root / "improvements" / "taskrelation" / "research"
    literature_dir = Path(literature_dir) if literature_dir else research / "literature"
    studies_path = Path(studies_path) if studies_path else research / "STUDIES.jsonl"
    return repo_root, literature_dir, studies_path


def _require_text(value, field):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RecordError(
            "{} must be a non-empty trimmed string".format(field), kind=INVALID_REQUEST
        )
    return value


def _require_block(value, field):
    """A non-empty multi-line body (leading/trailing whitespace is allowed)."""

    if not isinstance(value, str) or not value.strip():
        raise RecordError(
            "{} must be a non-empty string".format(field), kind=INVALID_REQUEST
        )
    return value


# -- scope authority ---------------------------------------------------------


def _require_delegated(investigation_id, studies_path):
    """Prove the write is under the one currently delegated investigation."""

    investigation_id = _require_text(investigation_id, "investigation_id")
    rows = literature_investigation.load_studies(studies_path)
    row = rows.get(investigation_id)
    if row is None:
        raise RecordError(
            "no registered literature investigation {!r}".format(investigation_id),
            kind=INVESTIGATION_NOT_FOUND,
            detail={"investigation_id": investigation_id},
        )
    if row.get("type") != "literature":
        raise RecordError(
            "Study {!r} is not a literature investigation".format(investigation_id),
            kind=INVESTIGATION_NOT_LITERATURE,
            detail={"investigation_id": investigation_id},
        )
    if row.get("status") != literature_investigation.STATUS_ACTIVE:
        raise RecordError(
            "investigation {!r} is not active (status {!r})".format(
                investigation_id, row.get("status")
            ),
            kind=INVESTIGATION_NOT_ACTIVE,
            detail={"investigation_id": investigation_id, "status": row.get("status")},
        )
    delegated = literature_investigation.delegated_investigation(rows)
    if delegated != investigation_id:
        raise RecordError(
            "investigation {!r} is not the delegated scope (delegated: {!r})".format(
                investigation_id, delegated
            ),
            kind=OUTSIDE_DELEGATED_SCOPE,
            detail={"investigation_id": investigation_id, "delegated": delegated},
        )
    return row


def _require_paper(paper_id, repo_root, literature_dir):
    paper_id = _require_text(paper_id, "paper_id")
    try:
        catalog = literature_catalog.load_catalog(
            literature_dir / "catalog.jsonl",
            repo_root=repo_root,
            validate_references=False,
        )
    except literature_catalog.CatalogError as exc:
        raise RecordError(
            "cannot read the catalog: {}".format(exc), kind=PAPER_NOT_FOUND
        ) from exc
    known = {entry.paper_id for entry in catalog.entries}
    if paper_id not in known:
        raise RecordError(
            "unknown paper_id {!r}".format(paper_id),
            kind=PAPER_NOT_FOUND,
            detail={"paper_id": paper_id},
        )
    return paper_id


# -- atomic registry mechanics ----------------------------------------------


def _read_jsonl_lines(path):
    try:
        return Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise RecordError(
            "cannot read registry {}: {}".format(path, exc), kind=INVALID_REQUEST
        ) from exc


def _serialize(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _atomic_replace(path, text):
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


def _merged_jsonl(path, key, row, sort_key):
    """Return the full new JSONL text after upserting ``row`` at ``key``.

    Existing lines are preserved byte-for-byte; only the changed row is
    reserialized, and entries are re-emitted in ``sort_key`` order.
    """

    entries = []
    replaced = False
    for line in _read_jsonl_lines(path):
        if not line.strip():
            continue
        raw = json.loads(line)
        if key(raw) == key(row):
            entries.append((sort_key(row), _serialize(row)))
            replaced = True
        else:
            entries.append((sort_key(raw), line))
    if not replaced:
        entries.append((sort_key(row), _serialize(row)))
    entries.sort(key=lambda item: item[0])
    return "\n".join(text for _key, text in entries) + "\n"


def _validate_then_commit(path, text, loader, **loader_kwargs):
    """Write ``text`` to a sibling temp, validate it with ``loader``, then commit.

    A rejected write leaves the real registry untouched.
    """

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
    try:
        result = loader(Path(handle.name), **loader_kwargs)
    except Exception:
        os.unlink(handle.name)
        raise
    os.replace(handle.name, str(path))
    return result


# -- operations --------------------------------------------------------------


def put_note(investigation_id, heading, body, *, studies_path=None, repo_root=None,
             literature_dir=None):
    """Upsert one ``## heading`` reasoning section in the investigation artifact."""

    repo_root, literature_dir, studies_path = _paths(
        repo_root, studies_path, literature_dir
    )
    row = _require_delegated(investigation_id, studies_path)
    heading = _require_text(heading, "heading")
    if heading.startswith("#") or "\n" in heading:
        raise RecordError(
            "heading must be a bare heading text, without '#' or newlines",
            kind=INVALID_REQUEST,
        )
    body = _require_block(body, "body")
    slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
    if not _SLUG.match(slug):
        raise RecordError(
            "heading does not yield a lowercase-hyphenated slug",
            kind=INVALID_REQUEST,
        )

    study_dir = repo_root / row["path"]
    study_dir.mkdir(parents=True, exist_ok=True)
    analysis = study_dir / "analysis.md"
    text = analysis.read_text(encoding="utf-8") if analysis.is_file() else ""
    new_text = _upsert_section(text, heading, body)
    _atomic_replace(analysis, new_text)
    return {
        "investigation_id": investigation_id,
        "artifact": analysis.relative_to(repo_root).as_posix(),
        "heading": heading,
        "anchor": slug,
    }


def _upsert_section(text, heading, body):
    """Replace ``## heading``'s body, or append the section when absent."""

    lines = text.splitlines()
    start = None
    level = 2
    for index, line in enumerate(lines):
        match = _HEADING.match(line)
        if match and len(match.group(1)) == 2 and match.group(2).strip() == heading:
            start = index
            break
    section = ["## {}".format(heading), ""] + body.rstrip("\n").splitlines()
    if start is None:
        base = "\n".join(lines).rstrip("\n")
        joined = base + ("\n\n" if base else "") + "\n".join(section) + "\n"
        return joined
    end = len(lines)
    for index in range(start + 1, len(lines)):
        match = _HEADING.match(lines[index])
        if match and len(match.group(1)) <= level:
            end = index
            break
    rebuilt = lines[:start] + section + [""] + lines[end:]
    return "\n".join(rebuilt).rstrip("\n") + "\n"


def put_assessment(investigation_id, paper_id, role, verdict, reason_summary, anchor,
                   *, gates=None, studies_path=None, repo_root=None,
                   literature_dir=None, index_path=None):
    """Create or update one ``(investigation_id, paper_id)`` PaperAssessment."""

    repo_root, literature_dir, studies_path = _paths(
        repo_root, studies_path, literature_dir
    )
    row = _require_delegated(investigation_id, studies_path)
    paper_id = _require_paper(paper_id, repo_root, literature_dir)
    role = _require_text(role, "role")
    verdict = _require_text(verdict, "verdict")
    reason_summary = _require_text(reason_summary, "reason_summary")
    anchor = _require_text(anchor, "anchor")
    if not _SLUG.match(anchor):
        raise RecordError(
            "anchor must be the lowercase-hyphenated slug of a heading in the "
            "investigation's analysis.md",
            kind=INVALID_REQUEST,
        )
    gates = _validate_gates_input(gates)

    record = {
        "schema_version": literature_assessment.SCHEMA_VERSION,
        "investigation_id": investigation_id,
        "paper_id": paper_id,
        "role": role,
        "verdict": verdict,
        "gates": gates,
        "reason_summary": reason_summary,
        "assessment_anchor": "{}/analysis.md#{}".format(row["path"], anchor),
        "assessed_at": literature_investigation._now(),
    }
    assessments = literature_dir / "assessments.jsonl"
    text = _merged_jsonl(
        assessments,
        key=lambda raw: (raw.get("investigation_id"), raw.get("paper_id")),
        row=record,
        sort_key=lambda raw: (raw.get("investigation_id"), raw.get("paper_id")),
    )
    try:
        registry = _validate_then_commit(
            assessments,
            text,
            literature_assessment.load_assessments,
            repo_root=repo_root,
            literature_dir=literature_dir,
            studies_path=studies_path,
        )
    except (
        literature_assessment.AssessmentError,
        literature_catalog.CatalogError,
    ) as exc:
        raise RecordError(
            "assessment rejected: {}".format(exc),
            kind=ASSESSMENT_CONFLICT,
            detail=dict(getattr(exc, "detail", {}) or {}),
        ) from exc
    _sync_index(literature_dir, index_path, registry)
    return {"assessment": record, "path": _relative(assessments, repo_root)}


def _validate_gates_input(gates):
    if gates is None:
        return []
    if isinstance(gates, str):
        try:
            gates = json.loads(gates)
        except ValueError as exc:
            raise RecordError(
                "gates must be a JSON list", kind=INVALID_REQUEST
            ) from exc
    if not isinstance(gates, list):
        raise RecordError("gates must be a list", kind=INVALID_REQUEST)
    out = []
    for index, raw in enumerate(gates):
        if not isinstance(raw, dict):
            raise RecordError(
                "gates[{}] must be an object".format(index), kind=INVALID_REQUEST
            )
        gate_id = _require_text(raw.get("gate_id"), "gates[{}].gate_id".format(index))
        gate_verdict = _require_text(
            raw.get("verdict"), "gates[{}].verdict".format(index)
        )
        out.append({"gate_id": gate_id, "verdict": gate_verdict})
    return out


def put_claim(investigation_id, paper_id, claim_id, claim_type, assertion_kind,
              assertion, evidence, *, qualification=None, status="active",
              studies_path=None, repo_root=None, literature_dir=None):
    """Create or update one paper-attributed, evidence-validated Claim.

    The Claim stays paper-global (no ``investigation_id`` is stored); the
    delegated investigation supplies the authority to record it and ensures the
    write happens within a bounded, active investigation.
    """

    repo_root, literature_dir, studies_path = _paths(
        repo_root, studies_path, literature_dir
    )
    _require_delegated(investigation_id, studies_path)
    paper_id = _require_paper(paper_id, repo_root, literature_dir)
    claim_id = _require_text(claim_id, "claim_id")
    claim_type = _require_text(claim_type, "claim_type")
    assertion_kind = _require_text(assertion_kind, "assertion_kind")
    assertion = _require_text(assertion, "assertion")
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except ValueError as exc:
            raise RecordError(
                "evidence must be a JSON list", kind=INVALID_REQUEST
            ) from exc
    if not isinstance(evidence, list) or not evidence:
        raise RecordError(
            "evidence must be a non-empty list", kind=EVIDENCE_REFERENCE_INVALID
        )

    record = {
        "schema_version": literature_claims.SCHEMA_VERSION,
        "paper_id": paper_id,
        "claim_id": claim_id,
        "assertion_kind": assertion_kind,
        "assertion": assertion,
        "claim_type": claim_type,
        "evidence": evidence,
        "status": status,
    }
    if qualification is not None:
        record["qualification"] = _require_text(qualification, "qualification")

    claims = literature_dir / "claims.jsonl"
    text = _merged_jsonl(
        claims,
        key=lambda raw: (raw.get("paper_id"), raw.get("claim_id")),
        row=record,
        sort_key=lambda raw: (raw.get("paper_id"), raw.get("claim_id")),
    )
    try:
        registry = _validate_then_commit(
            claims,
            text,
            literature_claims.load_claims,
            repo_root=repo_root,
            literature_dir=literature_dir,
        )
    except (
        literature_claims.ClaimError,
        literature_catalog.CatalogError,
    ) as exc:
        raise RecordError(
            "claim rejected: {}".format(exc),
            kind=CLAIM_CONFLICT
            if getattr(exc, "kind", None) == literature_claims.DUPLICATE_EVIDENCE
            else EVIDENCE_REFERENCE_INVALID,
            detail=dict(getattr(exc, "detail", {}) or {}),
        ) from exc
    return {
        "claim_ref": "{}#{}".format(paper_id, claim_id),
        "path": _relative(claims, repo_root),
        "claims": len(registry.records),
    }


def put_synthesis(investigation_id, synthesis_id, kind, status, document,
                  derives_from, body, *, studies_path=None, repo_root=None,
                  literature_dir=None):
    """Register one Synthesis whose provenance names the investigation."""

    repo_root, literature_dir, studies_path = _paths(
        repo_root, studies_path, literature_dir
    )
    row = _require_delegated(investigation_id, studies_path)
    synthesis_id = _require_text(synthesis_id, "synthesis_id")
    kind = _require_text(kind, "kind")
    status = _require_text(status, "status")
    document = _require_text(document, "document")
    body = _require_block(body, "body")
    if not _SURVEY_DOCUMENT.match(document):
        raise RecordError(
            "document must be a direct literature_survey/*.md filename",
            kind=INVALID_REQUEST,
        )
    if isinstance(derives_from, str):
        try:
            derives_from = json.loads(derives_from)
        except ValueError as exc:
            raise RecordError(
                "derives_from must be a JSON list", kind=INVALID_REQUEST
            ) from exc
    if not isinstance(derives_from, list) or not derives_from:
        raise RecordError(
            "derives_from must be a non-empty list", kind=INVALID_REQUEST
        )
    required = "investigation:{}".format(investigation_id)
    if required not in derives_from:
        raise RecordError(
            "a synthesis produced by {!r} must record provenance {!r} in "
            "derives_from".format(investigation_id, required),
            kind=SYNTHESIS_INVALID,
            detail={"required": required},
        )

    survey_dir = literature_dir.parent / "literature_survey"
    survey_dir.mkdir(parents=True, exist_ok=True)
    doc_path = survey_dir / document
    path_field = doc_path.relative_to(repo_root).as_posix()
    registry_row = {
        "schema_version": literature_synthesis.SCHEMA_VERSION,
        "synthesis_id": synthesis_id,
        "kind": kind,
        "status": status,
        "path": path_field,
        "derives_from": derives_from,
    }

    previous = doc_path.read_text(encoding="utf-8") if doc_path.is_file() else None
    doc_path.write_text(body.rstrip("\n") + "\n", encoding="utf-8")
    registry = survey_dir / "registry.jsonl"
    text = _merged_jsonl(
        registry,
        key=lambda raw: raw.get("synthesis_id"),
        row=registry_row,
        sort_key=lambda raw: raw.get("synthesis_id"),
    )
    try:
        loaded = _validate_then_commit(
            registry,
            text,
            literature_synthesis.load_syntheses,
            repo_root=repo_root,
            studies_path=studies_path,
            literature_dir=literature_dir,
            validate_references=True,
        )
    except (
        literature_synthesis.SynthesisError,
        literature_claims.ClaimError,
        literature_catalog.CatalogError,
    ) as exc:
        if previous is None:
            doc_path.unlink(missing_ok=True)
        else:
            doc_path.write_text(previous, encoding="utf-8")
        raise RecordError(
            "synthesis rejected: {}".format(exc),
            kind=SYNTHESIS_INVALID,
            detail=dict(getattr(exc, "detail", {}) or {}),
        ) from exc
    return {
        "synthesis_id": synthesis_id,
        "path": path_field,
        "registered": len(loaded.records),
    }


# -- index view --------------------------------------------------------------


def _sync_index(literature_dir, index_path, registry):
    path = Path(index_path) if index_path else Path(literature_dir) / "INDEX.md"
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    begin = text.find(literature_assessment.INDEX_ASSESSMENTS_BEGIN)
    end = text.find(literature_assessment.INDEX_ASSESSMENTS_END)
    if begin == -1 or end == -1 or end < begin:
        return False
    table = literature_assessment.render_assessment_tables(registry).rstrip("\n")
    new_text = (
        text[: begin + len(literature_assessment.INDEX_ASSESSMENTS_BEGIN)]
        + "\n"
        + table
        + "\n"
        + text[end:]
    )
    if new_text != text:
        _atomic_replace(path, new_text)
    return True


def _relative(path, repo_root):
    try:
        return Path(path).relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def validate(*, studies_path=None, repo_root=None, literature_dir=None):
    """Load and validate every literature registry this surface can write."""

    repo_root, literature_dir, studies_path = _paths(
        repo_root, studies_path, literature_dir
    )
    studies = literature_investigation.load_studies(studies_path)
    assessments = literature_assessment.load_assessments(
        literature_dir / "assessments.jsonl",
        repo_root=repo_root,
        literature_dir=literature_dir,
        studies_path=studies_path,
    )
    claims = literature_claims.load_claims(
        literature_dir / "claims.jsonl",
        repo_root=repo_root,
        literature_dir=literature_dir,
    )
    syntheses = literature_synthesis.load_syntheses(
        literature_dir.parent / "literature_survey" / "registry.jsonl",
        repo_root=repo_root,
        studies_path=studies_path,
        literature_dir=literature_dir,
    )
    return {
        "studies": len(studies),
        "assessments": len(assessments.records),
        "claims": len(claims.records),
        "syntheses": len(syntheses.records),
    }


# -- CLI ---------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    note = commands.add_parser("note", help="upsert an investigation analysis section")
    note.add_argument("--investigation", required=True)
    note.add_argument("--heading", required=True)
    note.add_argument("--body", required=True)

    assessment = commands.add_parser("assessment", help="put one PaperAssessment")
    assessment.add_argument("--investigation", required=True)
    assessment.add_argument("--paper-id", required=True)
    assessment.add_argument("--role", required=True)
    assessment.add_argument("--verdict", required=True)
    assessment.add_argument("--reason", required=True)
    assessment.add_argument("--anchor", required=True)
    assessment.add_argument("--gates")

    claim = commands.add_parser("claim", help="put one evidence-validated Claim")
    claim.add_argument("--investigation", required=True)
    claim.add_argument("--paper-id", required=True)
    claim.add_argument("--claim-id", required=True)
    claim.add_argument("--claim-type", required=True)
    claim.add_argument("--assertion-kind", required=True)
    claim.add_argument("--assertion", required=True)
    claim.add_argument("--evidence", required=True)
    claim.add_argument("--qualification")
    claim.add_argument("--status", default="active")

    synthesis = commands.add_parser("synthesis", help="put one registered Synthesis")
    synthesis.add_argument("--investigation", required=True)
    synthesis.add_argument("--synthesis-id", required=True)
    synthesis.add_argument("--kind", required=True)
    synthesis.add_argument("--status", required=True)
    synthesis.add_argument("--document", required=True)
    synthesis.add_argument("--derives-from", required=True)
    synthesis.add_argument("--body", required=True)

    commands.add_parser("validate", help="validate every literature registry")

    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "note":
            document = put_note(
                args.investigation, args.heading, args.body
            )
        elif args.command == "assessment":
            document = put_assessment(
                args.investigation,
                args.paper_id,
                args.role,
                args.verdict,
                args.reason,
                args.anchor,
                gates=args.gates,
            )
        elif args.command == "claim":
            document = put_claim(
                args.investigation,
                args.paper_id,
                args.claim_id,
                args.claim_type,
                args.assertion_kind,
                args.assertion,
                args.evidence,
                qualification=args.qualification,
                status=args.status,
            )
        elif args.command == "synthesis":
            document = put_synthesis(
                args.investigation,
                args.synthesis_id,
                args.kind,
                args.status,
                args.document,
                args.derives_from,
                args.body,
            )
        else:
            document = validate()
        print(json.dumps(document, ensure_ascii=False, sort_keys=True))
        return 0
    except (RecordError, literature_investigation.InvestigationError) as exc:
        print(json.dumps(exc.as_dict(), sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
