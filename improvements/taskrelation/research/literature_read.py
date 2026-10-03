"""Bounded, reference-only reads of literature artifacts.

The literature agent may inspect a *bounded* set of evidence artifacts, but
never arbitrary filesystem paths. Every read is addressed by a resolved
reference the query layer already returned — a ``paper_id`` or a registered
``LT-*`` Study artifact kind — and is resolved through the catalog/registry
before any bytes are touched.

Authority split:

* cards and Study artifacts are READ-ONLY canonical literature knowledge;
* the primary PDF is READ-ONLY canonical evidence, retrieved through
  :mod:`literature_primary` (which may write the disposable local cache);
* nothing here mutates research state.

Usage::

    python -m improvements.taskrelation.research.literature_read card <paper_id>
    python -m improvements.taskrelation.research.literature_read study LT-0001 analysis
    python -m improvements.taskrelation.research.literature_read primary <paper_id>
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from improvements.taskrelation.research import literature_catalog
from improvements.taskrelation.research import literature_primary
from improvements.taskrelation.research import literature_primary_text
from improvements.taskrelation.research import literature_query


INVALID_REFERENCE = "INVALID_REFERENCE"
UNKNOWN_REFERENCE = "UNKNOWN_REFERENCE"
ARTIFACT_NOT_AVAILABLE = "ARTIFACT_NOT_AVAILABLE"

DEFAULT_MAX_CHARS = 20000
MAX_CHARS_CEILING = 100000

_STUDY_ARTIFACT_ATTRS = {
    "plan": "plan_path",
    "note": "note_path",
    "analysis": "analysis_path",
    "result": "result_path",
}
_STUDY_ARTIFACT_KINDS = ("analysis", "note", "plan", "result")
SURVEY_ARTIFACT = "survey"
SYNTHESIS_ARTIFACT = "synthesis"
_SURVEY_DOCUMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.md$")

_RESEARCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _RESEARCH_DIR.parents[2]
_SURVEY_DIR = _RESEARCH_DIR / "literature_survey"


class LiteratureReadError(Exception):
    """A bounded literature read failed with a deterministic kind."""

    def __init__(self, message, kind, detail=None):
        super(LiteratureReadError, self).__init__(message)
        self.kind = kind
        self.detail = dict(detail or {})

    def as_dict(self):
        return {"error": str(self), "kind": self.kind, "detail": self.detail}


@dataclass(frozen=True)
class ArtifactContent:
    """Bounded text plus the evidence provenance a synthesis must cite.

    The ``artifact_role``/``sha256``/``source_url``/``locator`` fields are populated
    only for primary evidence: a card, Study artifact or survey document has no
    artifact digest, while a primary read is bound to the exact retained version
    it was extracted from.
    """

    source_kind: str
    evidence_level: str
    reference: str
    path: str
    characters: int
    truncated: bool
    text: str
    paper_id: str = ""
    study_id: str = ""
    artifact_role: str = ""
    sha256: str = ""
    source_url: str = ""
    locator: str = ""
    warnings: tuple = ()
    page: int = 0
    page_end: int = 0
    page_count: int = 0

    def as_dict(self):
        return {
            "source_kind": self.source_kind,
            "evidence_level": self.evidence_level,
            "reference": self.reference,
            "path": self.path,
            "characters": self.characters,
            "truncated": self.truncated,
            "paper_id": self.paper_id,
            "study_id": self.study_id,
            "artifact_role": self.artifact_role,
            "sha256": self.sha256,
            "source_url": self.source_url,
            "locator": self.locator,
            "warnings": list(self.warnings),
            "page": self.page,
            "page_end": self.page_end,
            "page_count": self.page_count,
            "text": self.text,
        }


def _validate_max_chars(max_chars):
    if (
        not isinstance(max_chars, int)
        or isinstance(max_chars, bool)
        or max_chars <= 0
        or max_chars > MAX_CHARS_CEILING
    ):
        raise LiteratureReadError(
            "max_chars must be an integer in 1..{}".format(MAX_CHARS_CEILING),
            kind=INVALID_REFERENCE,
        )
    return max_chars


class LiteratureReader:
    """Resolve approved references to bounded text; refuse anything else."""

    def __init__(self, repo_root=None, *, query=None, primary=None):
        self.repo_root = Path(repo_root).resolve() if repo_root else _REPO_ROOT
        self._query = query if query is not None else literature_query.LiteratureQuery(
            repo_root=self.repo_root
        )
        self._primary = primary if primary is not None else literature_primary.LiteraturePrimary(
            repo_root=self.repo_root
        )

    # -- canonical literature knowledge -------------------------------------

    def read_card(self, paper_id, max_chars=DEFAULT_MAX_CHARS):
        """Return bounded text from one paper's canonical derived-knowledge card."""

        max_chars = _validate_max_chars(max_chars)
        paper = self._resolve_paper(paper_id)
        path = self._resolve_inside_repo(paper.card_path, "card_path", paper.paper_id)
        text, truncated = _read_bounded(path, max_chars, paper.paper_id)
        return ArtifactContent(
            source_kind="card",
            evidence_level="card-derived",
            reference=paper.paper_id,
            path=self._relative(path),
            characters=len(text),
            truncated=truncated,
            text=text,
            paper_id=paper.paper_id,
        )

    def read_study(self, study_id, artifact, max_chars=DEFAULT_MAX_CHARS):
        """Return bounded text from one registered LT Study artifact kind."""

        max_chars = _validate_max_chars(max_chars)
        if artifact not in _STUDY_ARTIFACT_KINDS:
            raise LiteratureReadError(
                "artifact must be one of: {}".format(
                    ", ".join(_STUDY_ARTIFACT_KINDS)
                ),
                kind=INVALID_REFERENCE,
                detail={"artifact": artifact},
            )
        study = self._resolve_study(study_id)
        attribute = _STUDY_ARTIFACT_ATTRS[artifact]
        relative = getattr(study, attribute)
        if not relative:
            raise LiteratureReadError(
                "Study {!r} has no {!r} artifact".format(study.study_id, artifact),
                kind=ARTIFACT_NOT_AVAILABLE,
                detail={"study_id": study.study_id, "artifact": artifact},
            )
        path = self._resolve_inside_repo(relative, "Study artifact", study.study_id)
        text, truncated = _read_bounded(path, max_chars, study.study_id)
        return ArtifactContent(
            source_kind="study",
            evidence_level="Study-derived",
            reference="{}/{}".format(study.study_id, artifact),
            path=self._relative(path),
            characters=len(text),
            truncated=truncated,
            text=text,
            study_id=study.study_id,
        )

    # -- canonical primary evidence -----------------------------------------

    def read_survey(self, document, max_chars=DEFAULT_MAX_CHARS):
        """Return bounded text of one literature-survey document.

        Survey documents are derived literature/theory knowledge, not code and
        not research state; they are the artifact class a claim locator may cite
        when the per-paper card does not carry an assertion.
        """

        max_chars = _validate_max_chars(max_chars)
        if not isinstance(document, str) or not _SURVEY_DOCUMENT.match(document):
            raise LiteratureReadError(
                "document must be a direct literature_survey/*.md filename",
                kind=INVALID_REFERENCE,
                detail={"document": document},
            )
        path = (_SURVEY_DIR / document).resolve()
        if path != _SURVEY_DIR.resolve() and _SURVEY_DIR.resolve() not in path.parents:
            raise LiteratureReadError(
                "document escapes the literature survey directory",
                kind=INVALID_REFERENCE,
                detail={"document": document},
            )
        if not path.is_file():
            raise LiteratureReadError(
                "survey document not found: {}".format(document),
                kind=ARTIFACT_NOT_AVAILABLE,
                detail={"document": document},
            )
        text, truncated = _read_bounded(path, max_chars, document)
        return ArtifactContent(
            source_kind=SURVEY_ARTIFACT,
            evidence_level="survey-derived",
            reference=document,
            path=self._relative(path),
            characters=len(text),
            truncated=truncated,
            text=text,
        )

    def read_synthesis(self, synthesis_id, max_chars=DEFAULT_MAX_CHARS):
        """Return bounded text of one registered synthesis, addressed by identity.

        Unlike :meth:`read_survey` (which takes a literal filename), this resolves
        a stable ``synthesis_id`` through the registry before touching bytes, so
        the caller never needs the filesystem layout. The evidence level is
        ``survey-derived``: a synthesis document is a derived cross-source narrative,
        never primary evidence.
        """

        max_chars = _validate_max_chars(max_chars)
        try:
            record = self._query.get_synthesis(synthesis_id)
        except Exception as exc:
            raise LiteratureReadError(
                str(exc),
                kind=UNKNOWN_REFERENCE,
                detail={"synthesis_id": synthesis_id},
            ) from exc
        path = self._resolve_inside_repo(record.path, "synthesis path", record.synthesis_id)
        text, truncated = _read_bounded(path, max_chars, record.synthesis_id)
        return ArtifactContent(
            source_kind=SYNTHESIS_ARTIFACT,
            evidence_level="survey-derived",
            reference=record.synthesis_id,
            path=self._relative(path),
            characters=len(text),
            truncated=truncated,
            text=text,
        )

    def read_primary(self, paper_id, max_chars=DEFAULT_MAX_CHARS, *, role=None,
                     page=None, page_end=None):
        """Return extracted text from the checksum-verified primary artifact.

        ``role`` names the artifact version (required when the paper retains more
        than one); ``page``/``page_end`` are 1-based physical PDF page indices, and
        omitting them reads from the first page. Extraction is delegated to the
        same ``literature_primary`` read the agent uses, so the returned provenance
        (``sha256``, ``source_url``, ``artifact_role``, ``locator``) names the exact
        artifact the text came from and no filesystem path is exposed.

        Retrieval failures propagate as :class:`literature_primary.PrimaryError`
        so the caller branches on the same deterministic kinds; a corrupt or
        unavailable artifact is never partially extracted.
        """

        max_chars = _validate_max_chars(max_chars)
        view = self._primary.read(
            paper_id, role, page=page, page_end=page_end, max_chars=max_chars
        )
        text = "\n\n".join(text for _page, text in view.pages)
        return ArtifactContent(
            source_kind="primary",
            evidence_level="primary",
            reference=view.paper_id,
            path="",
            characters=len(text),
            truncated=view.truncated,
            text=text,
            paper_id=view.paper_id,
            artifact_role=view.role,
            sha256=view.sha256,
            source_url=view.source_url,
            locator=view.locator,
            warnings=tuple(view.warnings),
            page=view.page,
            page_end=view.page_end,
            page_count=view.page_count,
        )

    # -- resolution helpers --------------------------------------------------

    def _resolve_paper(self, paper_id):
        try:
            return self._query.resolve_paper(paper_id)
        except Exception as exc:
            raise LiteratureReadError(
                str(exc), kind=UNKNOWN_REFERENCE, detail={"paper_id": paper_id}
            ) from exc

    def _resolve_study(self, study_id):
        try:
            return self._query.get_study(study_id)
        except Exception as exc:
            raise LiteratureReadError(
                str(exc), kind=UNKNOWN_REFERENCE, detail={"study_id": study_id}
            ) from exc

    def _resolve_inside_repo(self, relative, field, reference):
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise LiteratureReadError(
                "{} must be a repository-relative path".format(field),
                kind=INVALID_REFERENCE,
                detail={"reference": reference},
            )
        path = (self.repo_root / candidate).resolve()
        if path != self.repo_root and self.repo_root not in path.parents:
            raise LiteratureReadError(
                "{} escapes the repository".format(field),
                kind=INVALID_REFERENCE,
                detail={"reference": reference, "path": relative},
            )
        if not path.is_file():
            raise LiteratureReadError(
                "resolved artifact is missing: {}".format(relative),
                kind=ARTIFACT_NOT_AVAILABLE,
                detail={"reference": reference, "path": relative},
            )
        return path

    def _relative(self, path):
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:  # primary cache lives outside the repository
            return str(path)


def _read_bounded(path, max_chars, reference):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise LiteratureReadError(
            "cannot read artifact for {!r}".format(reference),
            kind=ARTIFACT_NOT_AVAILABLE,
            detail={"error_type": type(exc).__name__},
        ) from exc
    if len(text) > max_chars:
        return text[:max_chars], True
    return text, False


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    card = commands.add_parser("card", help="read one paper's canonical card")
    card.add_argument("paper_id")
    card.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    study = commands.add_parser("study", help="read one registered LT Study artifact")
    study.add_argument("study_id")
    study.add_argument("artifact", choices=_STUDY_ARTIFACT_KINDS)
    study.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    primary = commands.add_parser("primary", help="read the verified primary artifact")
    primary.add_argument("paper_id")
    primary.add_argument("--role", default=None)
    primary.add_argument("--page", type=int, default=None)
    primary.add_argument("--page-end", type=int, default=None)
    primary.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    survey = commands.add_parser("survey", help="read one literature-survey document")
    survey.add_argument("document")
    survey.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    synthesis = commands.add_parser(
        "synthesis", help="read one registered synthesis by stable identity"
    )
    synthesis.add_argument("synthesis_id")
    synthesis.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    reader = LiteratureReader()
    try:
        if args.command == "card":
            result = reader.read_card(args.paper_id, args.max_chars)
        elif args.command == "study":
            result = reader.read_study(args.study_id, args.artifact, args.max_chars)
        elif args.command == "survey":
            result = reader.read_survey(args.document, args.max_chars)
        elif args.command == "synthesis":
            result = reader.read_synthesis(args.synthesis_id, args.max_chars)
        else:
            result = reader.read_primary(
                args.paper_id,
                args.max_chars,
                role=args.role,
                page=args.page,
                page_end=args.page_end,
            )
        print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    except LiteratureReadError as exc:
        print(json.dumps(exc.as_dict(), sort_keys=True), file=sys.stderr)
        return 1
    except literature_primary.PrimaryError as exc:
        print(json.dumps(exc.as_dict(), sort_keys=True), file=sys.stderr)
        return 1
    except literature_primary_text.PrimaryTextError as exc:
        print(json.dumps(exc.as_dict(), sort_keys=True), file=sys.stderr)
        return 1
    except literature_catalog.CatalogError as exc:
        print(
            json.dumps(
                {"error": str(exc), "kind": "CATALOG_INVALID", "detail": {}},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
