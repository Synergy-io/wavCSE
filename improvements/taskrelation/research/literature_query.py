"""Read-only structured queries over known papers and literature Studies.

Sources remain authoritative where they already live:

* ``literature/catalog.jsonl`` owns paper identity and metadata;
* ``literature/assessments.jsonl`` is the canonical machine-readable authority
  for how an ``LT-*`` investigation assessed a paper;
* ``STUDIES.jsonl`` owns Study lifecycle;
* the canonical card path points to per-paper derived knowledge.

The registry is the sole structured assessment authority: this module reads
``assessments.jsonl`` directly and holds no fallback to any legacy shape (a
``STUDIES.jsonl`` ``cards`` list, an ``LT-*`` ``result.json``
``primary_sources_reviewed`` list, or a per-card verdict section) — if the
canonical registry is invalid the query fails deterministically instead of
reconstructing from a stale witness. It never promotes an investigation-scoped
assessment to a global paper status, and it deliberately cannot answer
claim/topic questions.

Usage::

    python -m improvements.taskrelation.research.literature_query list
    python -m improvements.taskrelation.research.literature_query resolve <identity>
    python -m improvements.taskrelation.research.literature_query identify --doi <doi>
    python -m improvements.taskrelation.research.literature_query paper-studies <identity>
    python -m improvements.taskrelation.research.literature_query study-papers LT-0001
    python -m improvements.taskrelation.research.literature_query assessment LT-0002 <paper_id>

Every operation here is read-only, idempotent and deterministic. ``identify``
deduplicates a candidate paper against existing identity and returns ``known``,
``new`` or ``ambiguous``; ambiguity across disagreeing fields is a returned
verdict, never a silent pick. Mutation of literature state is outside this
interface.
"""

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from improvements.taskrelation.research import literature_assessment
from improvements.taskrelation.research import literature_catalog


_RESEARCH_DIR = Path(__file__).resolve().parent
_DEFAULT_REPO_ROOT = _RESEARCH_DIR.parents[2]
_DEFAULT_CATALOG = _RESEARCH_DIR / "literature" / "catalog.jsonl"
_DEFAULT_ASSESSMENTS = _RESEARCH_DIR / "literature" / "assessments.jsonl"
_DEFAULT_STUDIES = _RESEARCH_DIR / "STUDIES.jsonl"
_ARTIFACT_NAMES = ("PLAN.md", "NOTE.md", "analysis.md", "result.json")
_CANDIDATE_FIELDS = ("paper_id", "title", "doi", "arxiv", "source_url")


class LiteratureQueryError(ValueError):
    """A literature query is malformed or cannot resolve authoritative state."""


@dataclass(frozen=True)
class PaperRecord:
    """Paper identity and metadata; intentionally contains no screening status."""

    paper_id: str
    title: str
    year: int
    authors: Tuple[str, ...]
    venue: str
    external_ids: Mapping[str, Tuple[str, ...]]
    source_urls: Tuple[str, ...]
    aliases: Tuple[str, ...]
    card_path: str


@dataclass(frozen=True)
class StudyRecord:
    """One registered literature Study; decision is Study-level only."""

    study_id: str
    title: str
    status: str
    stage: str
    decision: Optional[str]
    path: str
    plan_path: Optional[str]
    note_path: Optional[str]
    analysis_path: Optional[str]
    result_path: Optional[str]


@dataclass(frozen=True)
class GateReference:
    """One eligibility gate outcome carried by a canonical assessment."""

    gate_id: str
    verdict: str


@dataclass(frozen=True)
class AssessmentReference:
    """One canonical investigation-scoped assessment, never a global verdict.

    Compact projection of a ``literature/assessments.jsonl`` record: enough to
    answer relationship questions without opening a card.
    """

    investigation_id: str
    paper_id: str
    role: str
    verdict: str
    gates: Tuple[GateReference, ...]
    reason_summary: str
    source_path: str
    detail_path: str
    detail_anchor: str


@dataclass(frozen=True)
class PaperStudyRecord:
    """Progressive-disclosure join: identity, Study, and scoped assessment pointer."""

    paper: PaperRecord
    study: StudyRecord
    assessment: AssessmentReference


@dataclass(frozen=True)
class CandidateMatch:
    """Deduplication verdict for a candidate paper; never a screening decision."""

    status: str
    matched_field: Optional[str]
    paper: Optional[PaperRecord]
    conflicts: Tuple[PaperRecord, ...]


class LiteratureQuery:
    """Validated, reusable read-only view over literature identity and LT Studies."""

    def __init__(
        self,
        *,
        repo_root=_DEFAULT_REPO_ROOT,
        catalog_path=None,
        studies_path=None,
        assessments_path=None
    ):
        self._repo_root = Path(repo_root).resolve()
        self._catalog_path = self._input_path(
            catalog_path if catalog_path is not None else _DEFAULT_CATALOG
        )
        self._studies_path = self._input_path(
            studies_path if studies_path is not None else _DEFAULT_STUDIES
        )
        self._assessments_path = self._input_path(
            assessments_path if assessments_path is not None else _DEFAULT_ASSESSMENTS
        )
        self._catalog = literature_catalog.load_catalog(
            self._catalog_path,
            repo_root=self._repo_root,
            validate_references=False,
        )
        self._papers = tuple(self._paper_record(entry) for entry in self._catalog.entries)
        self._papers_by_id = {paper.paper_id: paper for paper in self._papers}
        self._all_studies, self._literature_studies = self._load_studies()
        try:
            self._assessments = literature_assessment.load_assessments(
                self._assessments_path,
                repo_root=self._repo_root,
                literature_dir=self._catalog_path.parent,
                studies_path=self._studies_path,
            )
        except literature_assessment.AssessmentError as exc:
            raise LiteratureQueryError(str(exc)) from exc
        relationships = self._load_relationships()
        self._relationships_by_paper = self._group_relationships(
            relationships, lambda relationship: relationship.paper.paper_id
        )
        self._relationships_by_study = self._group_relationships(
            relationships, lambda relationship: relationship.study.study_id
        )

    def _input_path(self, path):
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self._repo_root / candidate

    def list_papers(self, *, year=None, author=None, venue=None):
        """Return paper metadata in paper_id order, filtered by exact metadata."""

        if year is not None and (not isinstance(year, int) or isinstance(year, bool)):
            raise LiteratureQueryError("year must be an integer")
        author_key = self._validate_text_filter(author, "author")
        venue_key = self._validate_text_filter(venue, "venue")
        selected = []
        for paper in self._papers:
            if year is not None and paper.year != year:
                continue
            if author_key is not None and author_key not in {
                self._normalize_text(value) for value in paper.authors
            }:
                continue
            if venue_key is not None and venue_key != self._normalize_text(paper.venue):
                continue
            selected.append(paper)
        return tuple(selected)

    def resolve_paper(self, identity):
        """Resolve a catalog identity/alias into compact paper metadata."""

        try:
            entry = self._catalog.lookup(identity)
        except literature_catalog.CatalogError as exc:
            raise LiteratureQueryError(str(exc)) from exc
        return self._papers_by_id[entry.paper_id]

    def identify_candidate(self, *, paper_id=None, title=None, doi=None,
                           arxiv=None, source_url=None):
        """Deduplicate a candidate paper against the catalog's existing identity.

        Answers "is this already a paper we retain?" using only the catalog's
        existing identity vocabulary — no fuzzy matching, no new identity system.
        Returns ``known``, ``new``, or ``ambiguous``; ambiguity is reported as a
        structured verdict, not raised, because it is a real query outcome.
        """

        provided = self._validate_candidate_fields(
            paper_id, title, doi, arxiv, source_url
        )
        hits = {}
        for field, value in provided:
            try:
                hits[field] = self._papers_by_id[self._catalog.lookup(value).paper_id]
            except literature_catalog.CatalogError:
                continue
        distinct = hits
        papers = {paper.paper_id: paper for paper in distinct.values()}
        if not papers:
            return CandidateMatch(
                status="new", matched_field=None, paper=None, conflicts=()
            )
        if len(papers) == 1:
            paper = next(iter(papers.values()))
            matched_field = next(
                field for field in _CANDIDATE_FIELDS if hits.get(field) is paper
            )
            return CandidateMatch(
                status="known",
                matched_field=matched_field,
                paper=paper,
                conflicts=(),
            )
        return CandidateMatch(
            status="ambiguous",
            matched_field=None,
            paper=None,
            conflicts=tuple(papers[key] for key in sorted(papers)),
        )

    @staticmethod
    def _validate_candidate_fields(paper_id, title, doi, arxiv, source_url):
        values = {
            "paper_id": paper_id,
            "title": title,
            "doi": doi,
            "arxiv": arxiv,
            "source_url": source_url,
        }
        provided = []
        for field in _CANDIDATE_FIELDS:
            value = values[field]
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise LiteratureQueryError(
                    "{} must be a non-empty string".format(field)
                )
            provided.append((field, value.strip()))
        if not provided:
            raise LiteratureQueryError(
                "at least one identity field is required "
                "(paper_id, title, doi, arxiv or source_url)"
            )
        return provided

    def studies_for_paper(self, identity):
        """Return every LT Study relationship for one resolved paper."""

        paper = self.resolve_paper(identity)
        return self._relationships_by_paper.get(paper.paper_id, ())

    def get_study(self, study_id):
        """Return one registered literature Study's metadata and artifact pointers."""

        if not isinstance(study_id, str) or not study_id.strip():
            raise LiteratureQueryError("study_id must be a non-empty string")
        study_id = study_id.strip()
        if study_id not in self._all_studies:
            raise LiteratureQueryError("unknown Study {!r}".format(study_id))
        if study_id not in self._literature_studies:
            raise LiteratureQueryError(
                "Study {!r} is not a literature Study".format(study_id)
            )
        return self._literature_studies[study_id]

    def papers_for_study(self, study_id):
        """Return every paper relationship for one registered LT Study."""

        study = self.get_study(study_id)
        return self._relationships_by_study.get(study.study_id, ())

    def get_assessment(self, investigation_id, paper_id):
        """Return the canonical ``(investigation_id, paper_id)`` assessment record.

        This is the detailed read: the full canonical ``PaperAssessment`` rather
        than the compact relationship projection. Identity and semantics are the
        registry's; the Study is resolved only to fail clearly on an unknown or
        non-literature investigation id.
        """

        self.get_study(investigation_id)
        try:
            return self._assessments.get_assessment(investigation_id, paper_id)
        except literature_assessment.AssessmentError as exc:
            raise LiteratureQueryError(str(exc)) from exc

    @staticmethod
    def _normalize_text(value):
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    @classmethod
    def _validate_text_filter(cls, value, field):
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise LiteratureQueryError("{} must be a non-empty string".format(field))
        return cls._normalize_text(value)

    @staticmethod
    def _paper_record(entry):
        external_ids = MappingProxyType(
            {
                key: tuple(values)
                for key, values in sorted(entry.external_ids.items())
            }
        )
        return PaperRecord(
            paper_id=entry.paper_id,
            title=entry.title,
            year=entry.year,
            authors=tuple(entry.authors),
            venue=entry.venue,
            external_ids=external_ids,
            source_urls=tuple(entry.source_urls),
            aliases=tuple(entry.aliases),
            card_path=entry.card_path,
        )

    def _load_studies(self):
        records = {}
        literature_records = {}
        try:
            lines = self._studies_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise LiteratureQueryError(
                "cannot read Study registry {}: {}".format(self._studies_path, exc)
            ) from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except ValueError as exc:
                raise LiteratureQueryError(
                    "Study registry line {} is not valid JSON: {}".format(
                        line_number, exc
                    )
                ) from exc
            if not isinstance(raw, dict):
                raise LiteratureQueryError(
                    "Study registry line {} must be an object".format(line_number)
                )
            study_id = self._required_text(raw, "study_id", line_number)
            if study_id in records:
                raise LiteratureQueryError(
                    "duplicate Study {!r} in registry".format(study_id)
                )
            records[study_id] = raw
            if raw.get("type") == "literature":
                literature_records[study_id] = self._study_record(raw, line_number)
        return records, literature_records

    @staticmethod
    def _required_text(raw, field, line_number):
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise LiteratureQueryError(
                "Study registry line {} {} must be a non-empty trimmed string".format(
                    line_number, field
                )
            )
        return value

    def _study_record(self, raw, line_number):
        study_id = self._required_text(raw, "study_id", line_number)
        title = self._required_text(raw, "title", line_number)
        status = self._required_text(raw, "status", line_number)
        stage = self._required_text(raw, "stage", line_number)
        path = self._required_text(raw, "path", line_number)
        decision = raw.get("decision")
        if decision is not None and not isinstance(decision, str):
            raise LiteratureQueryError(
                "Study {!r} decision must be a string or null".format(study_id)
            )
        folder = self._resolve_study_folder(study_id, path)
        artifacts = {
            name: self._relative_path(folder / name)
            if (folder / name).is_file()
            else None
            for name in _ARTIFACT_NAMES
        }
        return StudyRecord(
            study_id=study_id,
            title=title,
            status=status,
            stage=stage,
            decision=decision,
            path=path,
            plan_path=artifacts["PLAN.md"],
            note_path=artifacts["NOTE.md"],
            analysis_path=artifacts["analysis.md"],
            result_path=artifacts["result.json"],
        )

    def _resolve_study_folder(self, study_id, path):
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise LiteratureQueryError(
                "Study {!r} path must be repository-relative".format(study_id)
            )
        folder = self._repo_root / candidate
        if not folder.is_dir():
            raise LiteratureQueryError(
                "Study {!r} folder does not exist: {}".format(study_id, path)
            )
        return folder

    def _relative_path(self, path):
        return path.relative_to(self._repo_root).as_posix()

    def _load_relationships(self):
        """Project the canonical assessment registry into paper/Study joins."""

        registry_source = self._relative_path(self._assessments_path)
        relationships = []
        for record in self._assessments.records:
            paper = self._papers_by_id[record.paper_id]
            study = self._literature_studies[record.investigation_id]
            assessment = AssessmentReference(
                investigation_id=record.investigation_id,
                paper_id=record.paper_id,
                role=record.role,
                verdict=record.verdict,
                gates=tuple(
                    GateReference(gate.gate_id, gate.verdict)
                    for gate in record.gates
                ),
                reason_summary=record.reason_summary,
                source_path=registry_source,
                detail_path=record.detail_path,
                detail_anchor=record.detail_anchor,
            )
            relationships.append(
                PaperStudyRecord(paper=paper, study=study, assessment=assessment)
            )
        return tuple(relationships)

    @staticmethod
    def _group_relationships(relationships, key):
        grouped = {}
        for relationship in relationships:
            grouped.setdefault(key(relationship), []).append(relationship)
        return {
            group_key: tuple(
                sorted(
                    values,
                    key=lambda relationship: (
                        relationship.study.study_id,
                        relationship.paper.paper_id,
                    ),
                )
            )
            for group_key, values in grouped.items()
        }


def _paper_dict(paper):
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "year": paper.year,
        "authors": list(paper.authors),
        "venue": paper.venue,
        "external_ids": {
            key: list(values) for key, values in paper.external_ids.items()
        },
        "source_urls": list(paper.source_urls),
        "aliases": list(paper.aliases),
        "card_path": paper.card_path,
    }


def _study_dict(study):
    return {
        "study_id": study.study_id,
        "title": study.title,
        "status": study.status,
        "stage": study.stage,
        "decision": study.decision,
        "path": study.path,
        "plan_path": study.plan_path,
        "note_path": study.note_path,
        "analysis_path": study.analysis_path,
        "result_path": study.result_path,
    }


def _assessment_dict(assessment):
    return {
        "investigation_id": assessment.investigation_id,
        "paper_id": assessment.paper_id,
        "role": assessment.role,
        "verdict": assessment.verdict,
        "gates": [
            {"gate_id": gate.gate_id, "verdict": gate.verdict}
            for gate in assessment.gates
        ],
        "reason_summary": assessment.reason_summary,
        "source_path": assessment.source_path,
        "detail_path": assessment.detail_path,
        "detail_anchor": assessment.detail_anchor,
    }


def _relationship_dict(relationship):
    return {
        "paper": _paper_dict(relationship.paper),
        "study": _study_dict(relationship.study),
        "assessment": _assessment_dict(relationship.assessment),
    }


def _candidate_dict(match):
    return {
        "status": match.status,
        "matched_field": match.matched_field,
        "paper": _paper_dict(match.paper) if match.paper is not None else None,
        "conflicts": [_paper_dict(paper) for paper in match.conflicts],
    }


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    list_command = commands.add_parser("list", help="list/filter known papers")
    list_command.add_argument("--year", type=int)
    list_command.add_argument("--author")
    list_command.add_argument("--venue")
    resolve = commands.add_parser("resolve", help="resolve one paper identity")
    resolve.add_argument("identity")
    paper_studies = commands.add_parser(
        "paper-studies", help="list LT Study relationships for one paper"
    )
    paper_studies.add_argument("identity")
    study_papers = commands.add_parser(
        "study-papers", help="list paper relationships for one LT Study"
    )
    study_papers.add_argument("study_id")
    assessment = commands.add_parser(
        "assessment", help="return one canonical (investigation, paper) assessment"
    )
    assessment.add_argument("investigation_id")
    assessment.add_argument("paper_id")
    identify = commands.add_parser(
        "identify", help="deduplicate a candidate paper against known identity"
    )
    identify.add_argument("--paper-id")
    identify.add_argument("--title")
    identify.add_argument("--doi")
    identify.add_argument("--arxiv")
    identify.add_argument("--source-url")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        query = LiteratureQuery()
        if args.command == "list":
            document = {
                "papers": [
                    _paper_dict(paper)
                    for paper in query.list_papers(
                        year=args.year, author=args.author, venue=args.venue
                    )
                ]
            }
        elif args.command == "resolve":
            document = {"paper": _paper_dict(query.resolve_paper(args.identity))}
        elif args.command == "identify":
            document = {
                "candidate": _candidate_dict(
                    query.identify_candidate(
                        paper_id=args.paper_id,
                        title=args.title,
                        doi=args.doi,
                        arxiv=args.arxiv,
                        source_url=args.source_url,
                    )
                )
            }
        elif args.command == "paper-studies":
            paper = query.resolve_paper(args.identity)
            document = {
                "paper": _paper_dict(paper),
                "relationships": [
                    _relationship_dict(relationship)
                    for relationship in query.studies_for_paper(args.identity)
                ],
            }
        elif args.command == "assessment":
            document = {
                "assessment": query.get_assessment(
                    args.investigation_id, args.paper_id
                ).as_dict()
            }
        else:
            relationships = query.papers_for_study(args.study_id)
            study = query.get_study(args.study_id)
            document = {
                "study": _study_dict(study),
                "relationships": [
                    _relationship_dict(relationship)
                    for relationship in relationships
                ],
            }
        print(json.dumps(document, ensure_ascii=False, sort_keys=True))
        return 0
    except LiteratureQueryError as exc:
        print("literature query: ERROR: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
