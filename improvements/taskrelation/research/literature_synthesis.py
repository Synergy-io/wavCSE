"""Explicit, provenance-bounded registry over the literature-survey corpus.

``literature_survey/`` holds the retained cross-source and theoretical *narrative
reasoning* of the Task Relation Learning programme: theory notes, taxonomies,
candidate comparisons, pre-registered predictions, open questions, roadmaps and
post-experiment reconciliations. Those documents are canonical prose and stay
prose — this module never copies their reasoning into JSON. It adds only the
small envelope the rest of the system needs to address them deterministically:

    synthesis_id     stable identity, independent of the physical path
    kind             what sort of document it is
    status           active | historical (a frozen narrative snapshot)
    path             where the Markdown lives
    derives_from[]   bounded provenance to stable repository identities

The registry enumerates every ``literature_survey/*.md`` exactly once, so the
surface has one machine-readable map and no document is silently untracked. It is
*not* a Claim record, an assessment, a research decision, an authorization or a
knowledge graph: the only reference vocabulary is the small set of identities the
repository already validates.

Reference form (``derives_from``):

    paper:<paper_id>                 -> literature/catalog.jsonl
    claim:<paper_id>#<claim_id>      -> literature/claims.jsonl
    investigation:<study_id>         -> STUDIES.jsonl (a registered Study)
    finding:<F-id>                   -> FINDINGS.md heading
    synthesis:<synthesis_id>         -> this registry

A reference is recorded only where the document names a stable identity; a
narrative citation that has no structured identity stays prose. ``synthesis:``
dependencies must form a DAG; a self-reference or cycle is refused.

Usage::

    python -m improvements.taskrelation.research.literature_synthesis validate
    python -m improvements.taskrelation.research.literature_synthesis check
    python -m improvements.taskrelation.research.literature_synthesis list
    python -m improvements.taskrelation.research.literature_synthesis get <synthesis_id>
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from improvements.taskrelation.research import literature_catalog


SCHEMA_VERSION = 1

# The document classes the audit actually found. Small on purpose: extend only
# when a new document genuinely does not fit one of these.
KINDS = (
    "taxonomy",
    "theory",
    "comparison",
    "prediction",
    "open_questions",
    "roadmap",
    "synthesis",
    "index",
    "ledger",
)

# Lifecycle. ``historical`` marks a frozen narrative snapshot whose temporal
# meaning must be preserved (an imported note or a pre-registered prediction);
# ``historical`` never means invalid.
STATUSES = ("active", "historical")

# The reference kinds this registry may structurally encode.
REFERENCE_KINDS = ("paper", "claim", "investigation", "finding", "synthesis")

ALLOWED_KEYS = frozenset(
    {"schema_version", "synthesis_id", "kind", "status", "path", "derives_from"}
)

_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FINDING_HEADING = re.compile(r"^##\s+(F[0-9]+)\b", re.MULTILINE)

_RESEARCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _RESEARCH_DIR.parents[2]
_SURVEY_DIRNAME = "literature_survey"
_DEFAULT_REGISTRY = _RESEARCH_DIR / _SURVEY_DIRNAME / "registry.jsonl"


class SynthesisError(ValueError):
    """The synthesis registry is unusable or does not resolve."""

    def __init__(self, message, kind="SYNTHESIS_INVALID", detail=None):
        super(SynthesisError, self).__init__(message)
        self.kind = kind
        self.detail = dict(detail or {})

    def as_dict(self):
        return {"error": str(self), "kind": self.kind, "detail": self.detail}


@dataclass(frozen=True)
class SynthesisRecord:
    """One registered survey document; prose stays in the Markdown at ``path``."""

    schema_version: int
    synthesis_id: str
    kind: str
    status: str
    path: str
    derives_from: Tuple[str, ...]

    def as_dict(self):
        return {
            "synthesis_id": self.synthesis_id,
            "kind": self.kind,
            "status": self.status,
            "path": self.path,
            "derives_from": list(self.derives_from),
        }


class SynthesisRegistry:
    """Validated synthesis records with deterministic lookup and filtering."""

    def __init__(self, records):
        self.records = tuple(records)
        self._by_id = {record.synthesis_id: record for record in records}

    @property
    def synthesis_ids(self):
        return tuple(record.synthesis_id for record in self.records)

    def get(self, synthesis_id):
        if not isinstance(synthesis_id, str) or not synthesis_id.strip():
            raise SynthesisError("synthesis_id must be a non-empty string")
        synthesis_id = synthesis_id.strip()
        try:
            return self._by_id[synthesis_id]
        except KeyError as exc:
            raise SynthesisError(
                "unknown synthesis {!r}".format(synthesis_id),
                kind="UNKNOWN_SYNTHESIS",
                detail={"synthesis_id": synthesis_id},
            ) from exc

    def list(self, *, kind=None, status=None):
        """Return records in synthesis_id order, filtered by exact kind/status."""

        for field, value in (("kind", kind), ("status", status)):
            if value is not None and value not in (KINDS if field == "kind" else STATUSES):
                raise SynthesisError(
                    "{} must be one of: {}".format(
                        field, ", ".join(KINDS if field == "kind" else STATUSES)
                    ),
                    kind="INVALID_FILTER",
                    detail={field: value},
                )
        selected = []
        for record in self.records:
            if kind is not None and record.kind != kind:
                continue
            if status is not None and record.status != status:
                continue
            selected.append(record)
        return tuple(selected)


def load_syntheses(
    registry_path=_DEFAULT_REGISTRY,
    *,
    repo_root=_REPO_ROOT,
    studies_path=None,
    findings_path=None,
    literature_dir=None,
    validate_references=True,
    verify_claim_survey_evidence=None,
):
    """Load the synthesis registry; optionally validate every reference.

    ``validate_references`` resolves ``derives_from`` against the catalog, the
    claim registry, the Study registry and ``FINDINGS.md``, proves the
    ``synthesis:`` dependency graph is acyclic, and checks that every
    ``literature_survey/*.md`` is registered exactly once. Claim-backed survey
    evidence is additionally validated through the full claim loader (which
    re-reads the survey anchor) when ``verify_claim_survey_evidence`` is left
    true for the real repository; test fixtures disable it.
    """

    registry_path = Path(registry_path)
    repo_root = Path(repo_root).resolve()
    literature_dir = (
        Path(literature_dir)
        if literature_dir is not None
        else repo_root / "improvements/taskrelation/research/literature"
    )
    studies_path = (
        Path(studies_path)
        if studies_path is not None
        else repo_root / "improvements/taskrelation/research/STUDIES.jsonl"
    )
    findings_path = (
        Path(findings_path)
        if findings_path is not None
        else repo_root / "improvements/taskrelation/research/FINDINGS.md"
    )

    records = []
    seen_ids = {}
    seen_paths = {}
    for line_number, line in enumerate(_read_lines(registry_path), start=1):
        if not line.strip():
            raise SynthesisError(
                "registry line {} is blank".format(line_number),
                kind="BLANK_LINE",
                detail={"line": line_number},
            )
        try:
            raw = json.loads(line)
        except ValueError as exc:
            raise SynthesisError(
                "registry line {} is not valid JSON: {}".format(line_number, exc),
                kind="MALFORMED_JSON",
                detail={"line": line_number},
            ) from exc
        record = _validate_record(raw, line_number)
        if record.synthesis_id in seen_ids:
            raise SynthesisError(
                "duplicate synthesis_id {!r}".format(record.synthesis_id),
                kind="DUPLICATE_SYNTHESIS_ID",
                detail={"synthesis_id": record.synthesis_id},
            )
        if record.path in seen_paths:
            raise SynthesisError(
                "path {!r} is registered under two identities ({!r}, {!r})".format(
                    record.path, seen_paths[record.path], record.synthesis_id
                ),
                kind="DUPLICATE_SYNTHESIS_PATH",
                detail={"path": record.path},
            )
        seen_ids[record.synthesis_id] = record
        seen_paths[record.path] = record.synthesis_id
        records.append(record)

    if [record.synthesis_id for record in records] != sorted(
        record.synthesis_id for record in records
    ):
        raise SynthesisError(
            "registry entries must be sorted by synthesis_id",
            kind="UNSORTED_REGISTRY",
        )

    registry = SynthesisRegistry(records)
    if not validate_references:
        return registry

    _validate_surface(registry, repo_root)
    _validate_derives_from(registry, repo_root, literature_dir, studies_path, findings_path)
    if verify_claim_survey_evidence is None:
        verify_claim_survey_evidence = Path(literature_dir) == (
            repo_root / "improvements/taskrelation/research/literature"
        )
    if verify_claim_survey_evidence:
        _validate_claim_survey_evidence(registry, repo_root, literature_dir)
    return registry


def _read_lines(path):
    try:
        return Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SynthesisError(
            "cannot read synthesis registry {}: {}".format(path, exc),
            kind="REGISTRY_UNREADABLE",
        ) from exc


def _validate_record(raw, line_number):
    where = "registry line {}".format(line_number)
    if not isinstance(raw, dict):
        raise SynthesisError(
            "{} must be a JSON object".format(where), kind="MALFORMED_RECORD"
        )
    keys = set(raw)
    if keys != ALLOWED_KEYS:
        missing = sorted(ALLOWED_KEYS - keys)
        extra = sorted(keys - ALLOWED_KEYS)
        raise SynthesisError(
            "{} keys must be exactly the defined envelope".format(where),
            kind="UNEXPECTED_KEYS",
            detail={"missing": missing, "extra": extra},
        )
    if raw["schema_version"] != SCHEMA_VERSION or isinstance(raw["schema_version"], bool):
        raise SynthesisError(
            "{} schema_version must be {}".format(where, SCHEMA_VERSION),
            kind="UNSUPPORTED_SCHEMA_VERSION",
            detail={"schema_version": raw["schema_version"]},
        )
    synthesis_id = _require_string(raw, "synthesis_id", where)
    if not _ID_PATTERN.match(synthesis_id):
        raise SynthesisError(
            "{} synthesis_id {!r} must be lowercase-hyphenated".format(where, synthesis_id),
            kind="INVALID_SYNTHESIS_ID",
            detail={"synthesis_id": synthesis_id},
        )
    kind = _require_string(raw, "kind", where)
    if kind not in KINDS:
        raise SynthesisError(
            "{} kind {!r} must be one of: {}".format(where, kind, ", ".join(KINDS)),
            kind="INVALID_KIND",
            detail={"kind": kind},
        )
    status = _require_string(raw, "status", where)
    if status not in STATUSES:
        raise SynthesisError(
            "{} status {!r} must be one of: {}".format(where, status, ", ".join(STATUSES)),
            kind="INVALID_STATUS",
            detail={"status": status},
        )
    path = _require_string(raw, "path", where)
    derives_from = _validate_derives_from_field(raw, where)
    return SynthesisRecord(
        schema_version=SCHEMA_VERSION,
        synthesis_id=synthesis_id,
        kind=kind,
        status=status,
        path=path,
        derives_from=derives_from,
    )


def _require_string(raw, key, where):
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SynthesisError(
            "{} {} must be a non-empty trimmed string".format(where, key),
            kind="INVALID_FIELD",
            detail={"field": key},
        )
    return value


def _validate_derives_from_field(raw, where):
    value = raw["derives_from"]
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise SynthesisError(
            "{} derives_from must be a list of non-empty strings".format(where),
            kind="INVALID_DERIVES_FROM",
        )
    seen = set()
    for ref in value:
        if ref in seen:
            raise SynthesisError(
                "{} duplicate derives_from reference {!r}".format(where, ref),
                kind="DUPLICATE_REFERENCE",
                detail={"reference": ref},
            )
        seen.add(ref)
        _parse_reference(ref, where)
    return tuple(value)


def _parse_reference(ref, where):
    prefix, _, value = ref.partition(":")
    if prefix not in REFERENCE_KINDS or not value:
        raise SynthesisError(
            "{} reference {!r} must be <{}>:<value>".format(
                where, ref, "|".join(REFERENCE_KINDS)
            ),
            kind="INVALID_REFERENCE",
            detail={"reference": ref},
        )
    return prefix, value


def _validate_surface(registry, repo_root):
    survey_dir = (repo_root / "improvements/taskrelation/research" / _SURVEY_DIRNAME).resolve()
    for record in registry.records:
        path = _resolve_path(record, repo_root, survey_dir)
        if not path.is_file():
            raise SynthesisError(
                "registered synthesis {!r} path does not exist: {}".format(
                    record.synthesis_id, record.path
                ),
                kind="SYNTHESIS_PATH_MISSING",
                detail={"synthesis_id": record.synthesis_id, "path": record.path},
            )
    if not survey_dir.is_dir():
        return
    present = {entry.name for entry in survey_dir.glob("*.md")}
    registered = {Path(record.path).name for record in registry.records}
    unregistered = sorted(present - registered)
    if unregistered:
        raise SynthesisError(
            "literature_survey documents are not registered: {}".format(
                ", ".join(unregistered)
            ),
            kind="UNREGISTERED_SYNTHESIS",
            detail={"unregistered": unregistered},
        )
    missing = sorted(registered - present)
    if missing:
        raise SynthesisError(
            "registered synthesis documents are absent: {}".format(", ".join(missing)),
            kind="SYNTHESIS_PATH_MISSING",
            detail={"missing": missing},
        )


def _resolve_path(record, repo_root, survey_dir):
    candidate = Path(record.path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SynthesisError(
            "synthesis {!r} path must be repository-relative".format(record.synthesis_id),
            kind="INVALID_PATH",
            detail={"path": record.path},
        )
    resolved = (repo_root / candidate).resolve()
    if resolved != survey_dir and survey_dir not in resolved.parents:
        raise SynthesisError(
            "synthesis {!r} path escapes the literature_survey surface".format(
                record.synthesis_id
            ),
            kind="PATH_OUTSIDE_SURFACE",
            detail={"path": record.path},
        )
    return resolved


def _validate_derives_from(registry, repo_root, literature_dir, studies_path, findings_path):
    catalog = literature_catalog.load_catalog(
        Path(literature_dir) / "catalog.jsonl",
        repo_root=repo_root,
        validate_references=False,
    )
    paper_ids = {entry.paper_id for entry in catalog.entries}
    study_ids = _registered_study_ids(studies_path)
    finding_ids = _finding_ids(findings_path)
    claim_refs = _claim_refs(literature_dir)

    edges = {}
    for record in registry.records:
        local = []
        for ref in record.derives_from:
            prefix, value = _parse_reference(ref, record.synthesis_id)
            if prefix == "synthesis":
                if value not in registry._by_id:
                    raise SynthesisError(
                        "synthesis {!r} derives from unknown synthesis {!r}".format(
                            record.synthesis_id, value
                        ),
                        kind="UNRESOLVED_REFERENCE",
                        detail={"reference": ref},
                    )
                if value == record.synthesis_id:
                    raise SynthesisError(
                        "synthesis {!r} cannot derive from itself".format(record.synthesis_id),
                        kind="SELF_DEPENDENCY",
                        detail={"reference": ref},
                    )
                local.append(value)
            elif prefix == "paper":
                if value not in paper_ids:
                    raise SynthesisError(
                        "synthesis {!r} derives from unknown paper {!r}".format(
                            record.synthesis_id, value
                        ),
                        kind="UNRESOLVED_REFERENCE",
                        detail={"reference": ref},
                    )
            elif prefix == "claim":
                if value not in claim_refs:
                    raise SynthesisError(
                        "synthesis {!r} derives from unknown claim {!r}".format(
                            record.synthesis_id, value
                        ),
                        kind="UNRESOLVED_REFERENCE",
                        detail={"reference": ref},
                    )
            elif prefix == "investigation":
                if value not in study_ids:
                    raise SynthesisError(
                        "synthesis {!r} derives from unknown investigation {!r}".format(
                            record.synthesis_id, value
                        ),
                        kind="UNRESOLVED_REFERENCE",
                        detail={"reference": ref},
                    )
            else:  # finding
                if value not in finding_ids:
                    raise SynthesisError(
                        "synthesis {!r} derives from unknown finding {!r}".format(
                            record.synthesis_id, value
                        ),
                        kind="UNRESOLVED_REFERENCE",
                        detail={"reference": ref},
                    )
        edges[record.synthesis_id] = local
    _reject_cycles(edges)


def _reject_cycles(edges):
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {node: WHITE for node in edges}

    def visit(node, stack):
        colour[node] = GREY
        for target in edges[node]:
            if colour[target] == GREY:
                raise SynthesisError(
                    "synthesis dependency cycle: {}".format(
                        " -> ".join(stack + [node, target])
                    ),
                    kind="SYNTHESIS_CYCLE",
                    detail={"cycle": stack + [node, target]},
                )
            if colour[target] == WHITE:
                visit(target, stack + [node])
        colour[node] = BLACK

    for node in sorted(edges):
        if colour[node] == WHITE:
            visit(node, [])


def _validate_claim_survey_evidence(registry, repo_root, literature_dir):
    """Prove every survey-backed Claim's evidence document is a registered synthesis."""

    claims = _load_claims(literature_dir, repo_root)
    registered_paths = {record.path for record in registry.records}
    prefix = "improvements/taskrelation/research/literature_survey/"
    for record in claims.records:
        for reference in record.evidence:
            if reference.kind != "survey":
                continue
            document = reference.fields.get("document")
            expected = "{}{}".format(prefix, document)
            if expected not in registered_paths:
                raise SynthesisError(
                    "claim {!r} has survey evidence {!r} that is not a registered "
                    "synthesis".format(record.claim_ref, document),
                    kind="UNREGISTERED_SURVEY_EVIDENCE",
                    detail={"claim_ref": record.claim_ref, "document": document},
                )


def _registered_study_ids(studies_path):
    ids = set()
    try:
        lines = Path(studies_path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SynthesisError(
            "cannot read Study registry {}: {}".format(studies_path, exc),
            kind="STUDIES_UNREADABLE",
        ) from exc
    for line in lines:
        if not line.strip():
            continue
        raw = json.loads(line)
        study_id = raw.get("study_id") if isinstance(raw, dict) else None
        if isinstance(study_id, str):
            ids.add(study_id)
    return ids


def _finding_ids(findings_path):
    try:
        text = Path(findings_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SynthesisError(
            "cannot read FINDINGS.md {}: {}".format(findings_path, exc),
            kind="FINDINGS_UNREADABLE",
        ) from exc
    return {match for match in _FINDING_HEADING.findall(text)}


def _claim_refs(literature_dir):
    """Return the ``paper_id#claim_id`` identities in the claim registry.

    A minimal projection: resolving a ``claim:`` reference needs only the
    identity set, not full claim validation (which re-reads every located
    artifact). The deep check that survey-backed claim evidence resolves lives in
    :func:`_validate_claim_survey_evidence`.
    """

    path = Path(literature_dir) / "claims.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SynthesisError(
            "cannot read claim registry {}: {}".format(path, exc),
            kind="CLAIMS_UNREADABLE",
        ) from exc
    refs = set()
    for line in lines:
        if not line.strip():
            continue
        raw = json.loads(line)
        paper_id = raw.get("paper_id")
        claim_id = raw.get("claim_id")
        if isinstance(paper_id, str) and isinstance(claim_id, str):
            refs.add("{}#{}".format(paper_id, claim_id))
    return refs


def _load_claims(literature_dir, repo_root):
    # Imported lazily: literature_claims pulls in literature_read, which imports
    # the query layer; a module-level import here would close that cycle.
    from improvements.taskrelation.research import literature_claims

    literature_dir = Path(literature_dir)
    return literature_claims.load_claims(
        literature_dir / "claims.jsonl",
        repo_root=repo_root,
        literature_dir=literature_dir,
    )


def check_authority(
    registry_path=_DEFAULT_REGISTRY,
    *,
    repo_root=_REPO_ROOT,
    literature_dir=None,
    studies_path=None,
    findings_path=None,
):
    """Prove the synthesis layer is explicit and authority-bounded.

    Beyond ``load_syntheses`` validation (which resolves every path and
    reference), this proves the boundaries INC-V2-4 exists to protect:

    * PaperAssessment stays untouched — the canonical assessment registry still
      loads and validates;
    * every survey-backed Claim resolves, and its evidence document is a
      registered synthesis (never silently upgraded to card/primary);
    * synthesis metadata carries no assessment, decision, or authorization
      fields — the envelope is exactly the six defined keys;
    * a prediction snapshot is never presented as current synthesis — every
      ``prediction`` entry is ``historical``.
    """

    repo_root = Path(repo_root).resolve()
    literature_dir = (
        Path(literature_dir)
        if literature_dir is not None
        else repo_root / "improvements/taskrelation/research/literature"
    )
    registry = load_syntheses(
        registry_path,
        repo_root=repo_root,
        literature_dir=literature_dir,
        studies_path=studies_path,
        findings_path=findings_path,
    )
    for record in registry.records:
        if record.kind == "prediction" and record.status != "historical":
            raise SynthesisError(
                "prediction synthesis {!r} must stay a historical snapshot".format(
                    record.synthesis_id
                ),
                kind="PREDICTION_NOT_HISTORICAL",
                detail={"synthesis_id": record.synthesis_id},
            )
    _load_assessments(literature_dir, studies_path, repo_root)
    return registry


def _load_assessments(literature_dir, studies_path, repo_root):
    # Lazy for the same reason as the claim import above.
    from improvements.taskrelation.research import literature_assessment

    studies = (
        Path(studies_path)
        if studies_path is not None
        else repo_root / "improvements/taskrelation/research/STUDIES.jsonl"
    )
    return literature_assessment.load_assessments(
        Path(literature_dir) / "assessments.jsonl",
        repo_root=repo_root,
        literature_dir=Path(literature_dir),
        studies_path=studies,
    )


def _record_dict(record):
    return record.as_dict()


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate the synthesis registry and its references")
    commands.add_parser("check", help="prove the synthesis layer is authority-bounded")
    list_command = commands.add_parser("list", help="list registered syntheses")
    list_command.add_argument("--kind", choices=KINDS)
    list_command.add_argument("--status", choices=STATUSES)
    get_command = commands.add_parser("get", help="return one registered synthesis")
    get_command.add_argument("synthesis_id")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "check":
            registry = check_authority()
        else:
            registry = load_syntheses()
        if args.command == "list":
            document = {
                "syntheses": [
                    _record_dict(record)
                    for record in registry.list(kind=args.kind, status=args.status)
                ]
            }
        elif args.command == "get":
            document = {"synthesis": _record_dict(registry.get(args.synthesis_id))}
        else:
            document = {
                "syntheses": len(registry.records),
                "kinds": sorted({record.kind for record in registry.records}),
                "statuses": sorted({record.status for record in registry.records}),
            }
        print(json.dumps(document, ensure_ascii=False, sort_keys=True))
        return 0
    except SynthesisError as exc:
        print(json.dumps(exc.as_dict(), sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
