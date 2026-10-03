"""Canonical question-scoped PaperAssessment authority.

A paper has no global screening status. An assessment is always scoped to the
investigation that asked a question about the paper::

    assessment(investigation_id, paper_id)

This module owns the one structured machine-readable authority for that entity,
``literature/assessments.jsonl``. It replaced the several V1 shapes that carried
the same meaning (an ``LT-* result.json`` list, the ``STUDIES.jsonl`` ``cards``
list, per-card verdict sections and the ``INDEX.md`` tables); those are now
historical evidence only, or removed, and are never a second authority.

Two boundaries are deliberate:

* the identity of an assessment is exactly ``(investigation_id, paper_id)`` and
  at most one current record exists for that pair;
* ``investigation_id`` stays in the shared Study-ID namespace — ``LT-0001`` and
  ``LT-0002`` remain registered ``LT-*`` Studies — and ``paper_id`` is always a
  canonical ``catalog.jsonl`` identity, never a private/shadow key.

Vocabularies are investigation-scoped and derived from the existing records
rather than generalised for hypothetical future studies. ``role`` names the
paper's function inside that investigation; ``verdict`` is that investigation's
classification; ``gates`` records the eligibility gates where the investigation
was gate-structured. ``reason_summary`` is a verbatim clause migrated from the
legacy record and ``assessment_anchor`` points at the *investigation's own*
analysis artifact — the surviving prose that carries the reasoning — because a
PaperCard is paper-scoped and no longer holds any investigation verdict.

Usage::

    python -m improvements.taskrelation.research.literature_assessment validate
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

from improvements.taskrelation.research import literature_catalog


SCHEMA_VERSION = 1

# LT-0002's six eligibility gates, in the card/analysis order. The order is the
# gate identity: every LT-0002 card numbers its gates 1..6 with these meanings.
GATE_EXPLICIT_RELATION_OBJECT = "explicit_relation_object"
GATE_TAXONOMY = "taxonomy"
GATE_HETEROGENEOUS_HEADS = "heterogeneous_heads"
GATE_FIXED_REPRESENTATION = "fixed_representation"
GATE_FAITHFUL_IMPLEMENTABILITY = "faithful_implementability"
GATE_SOURCE_VERIFICATION = "source_verification"
GATE_IDS = (
    GATE_EXPLICIT_RELATION_OBJECT,
    GATE_TAXONOMY,
    GATE_HETEROGENEOUS_HEADS,
    GATE_FIXED_REPRESENTATION,
    GATE_FAITHFUL_IMPLEMENTABILITY,
    GATE_SOURCE_VERIFICATION,
)

# Normalised gate outcomes. ``pass_with_deviation`` is a gate satisfied only
# through a stated project-level change; ``partial`` is an ambiguous or partial
# satisfaction; ``not_applicable`` is a gate the investigation did not apply to
# the paper (e.g. LT-0002's gate 5 for the matched control).
GATE_VERDICTS = ("pass", "pass_with_deviation", "partial", "fail", "not_applicable")

# Investigation-scoped vocabularies, migrated from the records those
# investigations actually wrote. They are intentionally not a shared ontology.
ROLE_VOCABULARY = {
    "LT-0001": frozenset(
        (
            "taxonomy_anchor",
            "bayesian_relation_uncertainty",
            "signal_noise_relation_separation",
            "sample_variance_aware_relation",
            "sparse_task_covariance",
            "task_reliability_loss_scale",
            "direct_gradient_scale_balancing",
            "recent_explicit_relation_method",
        )
    ),
    "LT-0002": frozenset(("family_a_directed", "family_b_estimator")),
}

VERDICT_VOCABULARY = {
    "LT-0001": frozenset(
        (
            "retain",
            "reject_implementation",
            "retain_diagnostic_principle_reject_implementation",
            "retain_framework_evidence_reject_implementation",
            "reject_for_F9",
            "exclude_taxonomy",
            "exclude_decomposition_boundary",
        )
    ),
    "LT-0002": frozenset(("pass", "pass_with_documented_deviation", "fail")),
}

_ALLOWED_KEYS = frozenset(
    (
        "schema_version",
        "investigation_id",
        "paper_id",
        "role",
        "gates",
        "verdict",
        "reason_summary",
        "assessment_anchor",
        "assessed_at",
    )
)
_REQUIRED_KEYS = _ALLOWED_KEYS
_GATE_KEYS = frozenset(("gate_id", "verdict"))

_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_LT_STUDY = re.compile(r"^LT-\d{4}$")

# An anchored assessment points at a heading inside the investigation's analysis
# artifact; the heading's slug is its identity. ``_slugify`` matches the plain
# lowercase-hyphenated slugs the registry stores.
_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.M)
_NON_SLUG = re.compile(r"[^a-z0-9]+")

# The PaperCard sections this increment de-authorizes. Their presence on a card
# means investigation verdict state is leaking back into paper-scoped prose.
CARD_VERDICT_HEADING = re.compile(
    r"^## (?:Candidate Study ID|LT-\d{4} (?:decision|assessment))\s*$", re.M
)

# A generated block in ``INDEX.md``; the renderer and the consistency check share
# these markers so the view can be proven to derive from the registry.
INDEX_ASSESSMENTS_BEGIN = "<!-- BEGIN GENERATED: assessments -->"
INDEX_ASSESSMENTS_END = "<!-- END GENERATED: assessments -->"

_RESEARCH_DIR = Path(__file__).resolve().parent
_DEFAULT_REPO_ROOT = _RESEARCH_DIR.parents[2]
_DEFAULT_LITERATURE_DIR = _RESEARCH_DIR / "literature"
_DEFAULT_ASSESSMENTS = _DEFAULT_LITERATURE_DIR / "assessments.jsonl"
_DEFAULT_STUDIES = _RESEARCH_DIR / "STUDIES.jsonl"


class AssessmentError(ValueError):
    """A PaperAssessment record is unusable or disagrees with literature state."""

    def __init__(self, message, *, kind=None, detail=None):
        super().__init__(message)
        self.kind = kind
        self.detail = dict(detail) if detail else {}

    def as_dict(self):
        return {"error": str(self), "kind": self.kind, "detail": self.detail}


@dataclass(frozen=True)
class GateAssessment:
    """One eligibility gate's recorded outcome inside one assessment."""

    gate_id: str
    verdict: str

    def as_dict(self):
        return {"gate_id": self.gate_id, "verdict": self.verdict}


@dataclass(frozen=True)
class PaperAssessment:
    """How one investigation assessed one paper; never a global paper verdict."""

    investigation_id: str
    paper_id: str
    role: str
    verdict: str
    gates: Tuple[GateAssessment, ...]
    reason_summary: str
    assessment_anchor: str
    assessed_at: str

    @property
    def assessment_ref(self):
        """Stable reference: ``<investigation_id>#<paper_id>``."""

        return "{}#{}".format(self.investigation_id, self.paper_id)

    @property
    def detail_anchor(self):
        """The investigation analysis section slug the anchor points at."""

        return self.assessment_anchor.split("#", 1)[1]

    @property
    def detail_path(self):
        """The repository-relative investigation analysis artifact."""

        return self.assessment_anchor.split("#", 1)[0]

    def gate(self, gate_id):
        for gate in self.gates:
            if gate.gate_id == gate_id:
                return gate
        return None

    def as_dict(self):
        return {
            "assessment_ref": self.assessment_ref,
            "schema_version": SCHEMA_VERSION,
            "investigation_id": self.investigation_id,
            "paper_id": self.paper_id,
            "role": self.role,
            "gates": [gate.as_dict() for gate in self.gates],
            "verdict": self.verdict,
            "reason_summary": self.reason_summary,
            "assessment_anchor": self.assessment_anchor,
            "assessed_at": self.assessed_at,
        }


class AssessmentRegistry:
    """Validated PaperAssessment records with exact, deterministic lookup."""

    def __init__(self, records, paper_ids=(), investigation_ids=()):
        self.records = tuple(records)
        self._papers = frozenset(paper_ids)
        self._investigations = frozenset(investigation_ids)
        self._by_ref = {
            (record.investigation_id, record.paper_id): record
            for record in records
        }
        self._by_paper = {}
        self._by_investigation = {}
        for record in records:
            self._by_paper.setdefault(record.paper_id, []).append(record)
            self._by_investigation.setdefault(record.investigation_id, []).append(record)

    def assessment_refs(self):
        return tuple(record.assessment_ref for record in self.records)

    def investigations(self):
        """Registered LT investigations that carry at least one assessment."""

        return tuple(sorted(self._by_investigation))

    def assessments_for_paper(self, paper_id):
        """Every investigation-scoped assessment of one paper."""

        if not isinstance(paper_id, str) or not paper_id.strip():
            raise AssessmentError("paper_id must be a non-empty string")
        paper_id = paper_id.strip()
        if paper_id not in self._papers:
            raise AssessmentError(
                "unknown paper_id {!r}".format(paper_id),
                kind="UNKNOWN_PAPER",
                detail={"paper_id": paper_id},
            )
        return tuple(self._by_paper.get(paper_id, ()))

    def assessments_for_investigation(self, investigation_id):
        """Every paper assessment made by one registered LT investigation."""

        if not isinstance(investigation_id, str) or not investigation_id.strip():
            raise AssessmentError("investigation_id must be a non-empty string")
        investigation_id = investigation_id.strip()
        if investigation_id not in self._investigations:
            raise AssessmentError(
                "unknown literature investigation {!r}".format(investigation_id),
                kind="UNKNOWN_INVESTIGATION",
                detail={"investigation_id": investigation_id},
            )
        return tuple(self._by_investigation.get(investigation_id, ()))

    def get_assessment(self, investigation_id, paper_id):
        """Exact lookup by ``(investigation_id, paper_id)``; no fuzzy matching."""

        key = (str(investigation_id).strip(), str(paper_id).strip())
        record = self._by_ref.get(key)
        if record is None:
            raise AssessmentError(
                "no recorded assessment {!r}#{!r}".format(key[0], key[1]),
                kind="UNKNOWN_ASSESSMENT",
                detail={"investigation_id": key[0], "paper_id": key[1]},
            )
        return record


def load_assessments(
    assessments_path=_DEFAULT_ASSESSMENTS,
    *,
    repo_root=_DEFAULT_REPO_ROOT,
    literature_dir=None,
    studies_path=None,
):
    """Load and validate the assessment registry against the literature state.

    ``assessments_path`` may live outside the literature directory (tests use a
    fixture); ``literature_dir`` then names the catalog to check against.
    """

    assessments_path = Path(assessments_path)
    repo_root = Path(repo_root).resolve()
    literature_dir = (
        Path(literature_dir) if literature_dir is not None else assessments_path.parent
    )
    studies_path = (
        Path(studies_path)
        if studies_path is not None
        else _DEFAULT_STUDIES
        if repo_root == _DEFAULT_REPO_ROOT
        else literature_dir.parent / "STUDIES.jsonl"
    )
    catalog = literature_catalog.load_catalog(
        literature_dir / "catalog.jsonl",
        repo_root=repo_root,
        validate_references=False,
    )
    papers = {entry.paper_id: entry for entry in catalog.entries}
    investigations = _registered_lt_studies(studies_path)

    records = []
    seen = {}
    for line_number, line in enumerate(_read_lines(assessments_path), start=1):
        if not line.strip():
            raise AssessmentError(
                "assessments line {} is blank".format(line_number)
            )
        try:
            raw = json.loads(line)
        except ValueError as exc:
            raise AssessmentError(
                "assessments line {} is not valid JSON: {}".format(line_number, exc)
            ) from exc
        record = _validate_record(
            raw,
            line_number,
            repo_root=repo_root,
            papers=papers,
            investigations=investigations,
        )
        key = (record.investigation_id, record.paper_id)
        if key in seen:
            raise AssessmentError(
                "duplicate paper assessment {!r}".format(
                    "{0[0]}#{0[1]}".format(key)
                ),
                kind="DUPLICATE_ASSESSMENT",
                detail={"assessment_ref": "{0[0]}#{0[1]}".format(key)},
            )
        seen[key] = True
        records.append(record)

    if [record.assessment_ref for record in records] != sorted(
        record.assessment_ref for record in records
    ):
        raise AssessmentError(
            "assessments must be sorted by investigation_id then paper_id"
        )

    return AssessmentRegistry(
        records, papers.keys(), investigations.keys()
    )


def render_assessment_tables(
    registry=None,
    *,
    repo_root=_DEFAULT_REPO_ROOT,
    literature_dir=None,
    studies_path=None,
    assessments_path=_DEFAULT_ASSESSMENTS,
):
    """Render the deterministic, human-readable assessment table for ``INDEX.md``.

    This is a *derived view*: every cell comes from the canonical registry, so a
    hand edit cannot make INDEX a second assessment authority.
    """

    if registry is None:
        registry = load_assessments(
            assessments_path,
            repo_root=repo_root,
            literature_dir=literature_dir,
            studies_path=studies_path,
        )
    lines = [
        "| Investigation | Paper | Role | Verdict |",
        "| --- | --- | --- | --- |",
    ]
    for record in registry.records:
        lines.append(
            "| `{}` | [{}]({}.md) | `{}` | `{}` |".format(
                record.investigation_id,
                record.paper_id,
                record.paper_id,
                record.role,
                record.verdict,
            )
        )
    return "\n".join(lines) + "\n"


def _index_assessment_block(index_path):
    """The body between the generated-assessment markers, or ``None``."""

    try:
        text = Path(index_path).read_text(encoding="utf-8")
    except OSError:
        return None
    begin = text.find(INDEX_ASSESSMENTS_BEGIN)
    end = text.find(INDEX_ASSESSMENTS_END)
    if begin == -1 or end == -1 or end < begin:
        return None
    return text[begin + len(INDEX_ASSESSMENTS_BEGIN) : end].strip("\n")


def check_authority(
    assessments_path=_DEFAULT_ASSESSMENTS,
    *,
    repo_root=_DEFAULT_REPO_ROOT,
    literature_dir=None,
    studies_path=None,
    index_path=None,
):
    """Prove PaperAssessment has exactly one ACTIVE structured authority.

    Loads and validates the canonical registry, then proves no *other* artifact
    is still consumed as an assessment authority:

    * every PaperCard is free of investigation verdict sections;
    * no ``STUDIES.jsonl`` row duplicates PaperAssessment membership via ``cards``;
    * ``INDEX.md``'s generated block, when present, matches the registry exactly.

    Historical prose (a completed ``result.json``, the LT analysis narrative) is
    allowed to mention old outcomes; it is not an authority unless code consumes
    it as one, and nothing here does. Returns the loaded registry.
    """

    repo_root = Path(repo_root).resolve()
    assessments_path = Path(assessments_path)
    literature_dir = (
        Path(literature_dir) if literature_dir is not None else assessments_path.parent
    )
    studies_path = (
        Path(studies_path)
        if studies_path is not None
        else _DEFAULT_STUDIES
        if repo_root == _DEFAULT_REPO_ROOT
        else literature_dir.parent / "STUDIES.jsonl"
    )
    index = Path(index_path) if index_path is not None else literature_dir / "INDEX.md"
    registry = load_assessments(
        assessments_path,
        repo_root=repo_root,
        literature_dir=literature_dir,
        studies_path=studies_path,
    )

    catalog = literature_catalog.load_catalog(
        literature_dir / "catalog.jsonl", repo_root=repo_root, validate_references=False
    )
    for entry in catalog.entries:
        card = repo_root / entry.card_path
        if not card.is_file():
            continue
        match = CARD_VERDICT_HEADING.search(card.read_text(encoding="utf-8"))
        if match:
            raise AssessmentError(
                "PaperCard {} still carries an investigation verdict section "
                "{!r}".format(entry.card_path, match.group(0).strip()),
                kind="CARD_STILL_A_VERDICT_AUTHORITY",
                detail={"card": entry.card_path},
            )

    for line_number, line in enumerate(_read_lines(studies_path), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if isinstance(raw, dict) and "cards" in raw:
            raise AssessmentError(
                "Study registry line {} still carries a legacy 'cards' "
                "PaperAssessment membership".format(line_number),
                kind="STUDIES_ROW_STILL_A_MEMBERSHIP_AUTHORITY",
                detail={"study_id": raw.get("study_id")},
            )

    block = _index_assessment_block(index)
    if block is not None and block != render_assessment_tables(registry).strip("\n"):
        raise AssessmentError(
            "INDEX.md's generated assessment block disagrees with the registry; "
            "regenerate it",
            kind="INDEX_ASSESSMENT_DRIFT",
            detail={"index": index.as_posix()},
        )
    return registry


def _read_lines(path):
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AssessmentError(
            "cannot read assessment registry {}: {}".format(path, exc)
        ) from exc


def _registered_lt_studies(studies_path):
    """The registered ``LT-*`` literature Studies, keyed by Study ID.

    LT investigations stay in the shared Study-ID namespace; this reads the same
    registry every other component reads and never creates a second one.
    """

    try:
        lines = Path(studies_path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AssessmentError(
            "cannot read Study registry {}: {}".format(studies_path, exc)
        ) from exc
    studies = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except ValueError as exc:
            raise AssessmentError(
                "Study registry line {} is not valid JSON: {}".format(line_number, exc)
            ) from exc
        if not isinstance(raw, dict):
            raise AssessmentError(
                "Study registry line {} must be an object".format(line_number)
            )
        study_id = raw.get("study_id")
        if raw.get("type") != "literature":
            continue
        if not isinstance(study_id, str) or not _LT_STUDY.match(study_id):
            raise AssessmentError(
                "literature Study line {} has an invalid LT-* study_id {!r}".format(
                    line_number, study_id
                )
            )
        studies[study_id] = raw
    return studies


def _require_string(record, key, where):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise AssessmentError(
            "{} .{} must be a non-empty trimmed string".format(where, key)
        )
    return value


def _validate_record(record, line_number, *, repo_root, papers, investigations):
    where = "assessments line {}".format(line_number)
    if not isinstance(record, dict):
        raise AssessmentError("{} must be a JSON object".format(where))
    unknown = sorted(set(record) - _ALLOWED_KEYS)
    missing = sorted(_REQUIRED_KEYS - set(record))
    if unknown:
        raise AssessmentError(
            "{} has unknown key(s): {} (a global paper verdict does not belong "
            "in a PaperAssessment record)".format(where, ", ".join(unknown))
        )
    if missing:
        raise AssessmentError(
            "{} is missing key(s): {}".format(where, ", ".join(missing))
        )
    if record["schema_version"] != SCHEMA_VERSION:
        raise AssessmentError(
            "{}.schema_version must be {}".format(where, SCHEMA_VERSION)
        )

    investigation_id = _require_string(record, "investigation_id", where)
    if investigation_id not in investigations:
        raise AssessmentError(
            "{} references {!r}, which is not a registered LT-* literature "
            "Study".format(where, investigation_id)
        )

    paper_id = _require_string(record, "paper_id", where)
    if paper_id not in papers:
        raise AssessmentError(
            "{} references unknown paper_id {!r}".format(where, paper_id)
        )

    role = _require_string(record, "role", where)
    role_vocabulary = ROLE_VOCABULARY.get(investigation_id, frozenset())
    if role not in role_vocabulary:
        raise AssessmentError(
            "{}.role {!r} is not a known role for {}".format(
                where, role, investigation_id
            )
        )

    verdict = _require_string(record, "verdict", where)
    verdict_vocabulary = VERDICT_VOCABULARY.get(investigation_id, frozenset())
    if verdict not in verdict_vocabulary:
        raise AssessmentError(
            "{}.verdict {!r} is not a known verdict for {}".format(
                where, verdict, investigation_id
            )
        )

    gates = _validate_gates(record["gates"], where)
    reason_summary = _require_string(record, "reason_summary", where)

    anchor = _require_string(record, "assessment_anchor", where)
    _validate_anchor(anchor, where, investigation_id, repo_root, investigations)

    assessed_at = _require_string(record, "assessed_at", where)
    _validate_timestamp(assessed_at, where)

    return PaperAssessment(
        investigation_id=investigation_id,
        paper_id=paper_id,
        role=role,
        verdict=verdict,
        gates=gates,
        reason_summary=reason_summary,
        assessment_anchor=anchor,
        assessed_at=assessed_at,
    )


def _validate_gates(gates, where):
    if not isinstance(gates, list):
        raise AssessmentError("{}.gates must be a list".format(where))
    seen = set()
    parsed = []
    for index, raw in enumerate(gates):
        gate_where = "{}.gates[{}]".format(where, index)
        if not isinstance(raw, dict):
            raise AssessmentError("{} must be an object".format(gate_where))
        unknown = sorted(set(raw) - _GATE_KEYS)
        missing = sorted(_GATE_KEYS - set(raw))
        if unknown:
            raise AssessmentError(
                "{} has unknown key(s): {}".format(gate_where, ", ".join(unknown))
            )
        if missing:
            raise AssessmentError(
                "{} is missing key(s): {}".format(gate_where, ", ".join(missing))
            )
        gate_id = _require_string(raw, "gate_id", gate_where)
        if gate_id not in GATE_IDS:
            raise AssessmentError(
                "{}.gate_id {!r} is not a known gate".format(gate_where, gate_id)
            )
        if gate_id in seen:
            raise AssessmentError(
                "{} repeats gate_id {!r}".format(gate_where, gate_id)
            )
        seen.add(gate_id)
        gate_verdict = _require_string(raw, "verdict", gate_where)
        if gate_verdict not in GATE_VERDICTS:
            raise AssessmentError(
                "{}.verdict {!r} is not a known gate verdict".format(
                    gate_where, gate_verdict
                )
            )
        parsed.append(GateAssessment(gate_id=gate_id, verdict=gate_verdict))
    return tuple(parsed)


def _slugify(heading):
    """The plain lowercase-hyphenated slug an analysis heading is addressed by."""

    return _NON_SLUG.sub("-", heading.lower()).strip("-")


def _heading_slugs(text):
    """Every heading slug in a Markdown artifact."""

    return {_slugify(match.group(1)) for match in _HEADING.finditer(text)}


def _investigation_analysis_path(investigations, investigation_id):
    """The repository-relative analysis artifact of one registered LT Study."""

    study = investigations[investigation_id]
    path = study.get("path")
    if not isinstance(path, str) or not path.strip():
        raise AssessmentError(
            "literature Study {!r} has no usable path".format(investigation_id)
        )
    return (Path(path) / "analysis.md").as_posix()


def _validate_anchor(anchor, where, investigation_id, repo_root, investigations):
    """An assessment anchor names the investigation's own analysis reasoning.

    A PaperCard is paper-scoped and carries no investigation verdict, so the
    anchor must resolve to a heading in the investigating Study's ``analysis.md``;
    it can never point at a card section this increment removes.
    """

    if "#" not in anchor:
        raise AssessmentError(
            "{}.assessment_anchor must be '<analysis path>#<section slug>'".format(
                where
            )
        )
    path_part, _, slug = anchor.partition("#")
    if not _SLUG.match(slug):
        raise AssessmentError(
            "{}.assessment_anchor slug must be a lowercase hyphenated slug".format(
                where
            )
        )
    candidate = Path(path_part)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AssessmentError(
            "{}.assessment_anchor path must be repository-relative".format(where)
        )
    expected = _investigation_analysis_path(investigations, investigation_id)
    if path_part != expected:
        raise AssessmentError(
            "{}.assessment_anchor must point at {!r} (the investigation's own "
            "analysis artifact)".format(where, expected)
        )
    artifact = repo_root / candidate
    if not artifact.is_file():
        raise AssessmentError(
            "{}.assessment_anchor does not resolve to an existing artifact: "
            "{}".format(where, path_part)
        )
    if slug not in _heading_slugs(artifact.read_text(encoding="utf-8")):
        raise AssessmentError(
            "{}.assessment_anchor slug {!r} is not a heading in {}".format(
                where, slug, path_part
            )
        )


def _validate_timestamp(value, where):
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise AssessmentError(
            "{}.assessed_at must be an ISO 8601 timestamp".format(where)
        ) from exc
    if parsed.tzinfo is None:
        raise AssessmentError(
            "{}.assessed_at must carry a timezone offset".format(where)
        )


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("validate", help="validate the assessment registry")
    commands.add_parser(
        "check", help="prove assessments.jsonl is the sole active authority"
    )
    commands.add_parser(
        "index-assessments", help="render the derived INDEX assessment table"
    )
    commands.add_parser("list", help="list every assessment reference")
    paper = commands.add_parser("paper", help="list one paper's assessments")
    paper.add_argument("paper_id")
    investigation = commands.add_parser(
        "investigation", help="list one LT investigation's assessments"
    )
    investigation.add_argument("investigation_id")
    get = commands.add_parser("get", help="exact assessment lookup")
    get.add_argument("investigation_id")
    get.add_argument("paper_id")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    command = args.command or "validate"
    try:
        registry = load_assessments()
        if command == "validate":
            print(
                "literature assessments: OK ({} record(s))".format(
                    len(registry.records)
                )
            )
        elif command == "check":
            checked = check_authority()
            print(
                "literature assessments: sole active authority OK ({} "
                "record(s))".format(len(checked.records))
            )
        elif command == "index-assessments":
            sys.stdout.write(render_assessment_tables(registry))
        elif command == "list":
            print(
                json.dumps(
                    {"assessment_refs": list(registry.assessment_refs())},
                    sort_keys=True,
                )
            )
        elif command == "paper":
            records = registry.assessments_for_paper(args.paper_id)
            print(
                json.dumps(
                    {"assessments": [record.as_dict() for record in records]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        elif command == "investigation":
            records = registry.assessments_for_investigation(args.investigation_id)
            print(
                json.dumps(
                    {"assessments": [record.as_dict() for record in records]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            print(
                json.dumps(
                    registry.get_assessment(
                        args.investigation_id, args.paper_id
                    ).as_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return 0
    except (AssessmentError, literature_catalog.CatalogError) as exc:
        payload = (
            exc.as_dict()
            if isinstance(exc, AssessmentError)
            else {"error": str(exc), "kind": "CATALOG_INVALID", "detail": {}}
        )
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
