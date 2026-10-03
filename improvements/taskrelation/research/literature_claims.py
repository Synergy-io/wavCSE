"""Source-bound literature claim records.

Purpose: attribution correctness, not semantic search. A record answers
"what exact literature assertion have we recorded, where did it come from, and
what evidence supports that attribution?"

The motivating failure (LIT-AGENT-V1 evaluation run 2): an agent attributed the
published ``λ₂`` grid to the per-paper card, which does not state any grid — the
grid is recorded in a survey document citing JMLR §4.1, while the card says only
that the paper selects ``λ₂`` on data. The value existed; the *provenance* was
invented. A claim record therefore names the artifact class and anchor it was
read from, and a verbatim quote must be provably present in that artifact.

Scope rules:

* every record is bound to one ``paper_id`` — no free-floating literature prose;
* the registry holds **paper-attributed** assertions only. Our own experimental
  configuration (a restricted grid, a frozen λ, a declared deviation) is research
  state and stays in ``DECISIONS.md`` / Study artifacts, owned by the main
  research agent;
* ``primary_verified`` is only legal when the paper actually has a retained
  primary artifact, so today nothing can claim it.

Authority: this module is read-only. Claim records are created under human/operator
review and validated here; the Literature Agent never writes them.

Usage::

    python -m improvements.taskrelation.research.literature_claims validate
    python -m improvements.taskrelation.research.literature_claims paper <paper_id>
    python -m improvements.taskrelation.research.literature_claims get <paper_id> <claim_id>
    python -m improvements.taskrelation.research.literature_claims list
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from improvements.taskrelation.research import literature_catalog
from improvements.taskrelation.research import literature_primary
from improvements.taskrelation.research import literature_read


SCHEMA_VERSION = 1

# Where the recorded assertion lives. `primary` means the paper's own artifact.
SOURCE_LEVELS = ("primary", "card", "survey", "study")

# Whether the proposition has been checked against the paper itself.
PRIMARY_VERIFIED = "primary_verified"
DERIVED_EXISTING_RECORD = "derived_existing_record"
UNVERIFIED_PRIMARY = "unverified_primary"
VERIFICATION_LEVELS = (PRIMARY_VERIFIED, DERIVED_EXISTING_RECORD, UNVERIFIED_PRIMARY)

CLAIM_TYPES = (
    "relation-object",
    "method-objective",
    "method-procedure",
    "hyperparameter-selection",
    "implementation-requirement",
    "taxonomy",
)

ASSERTION_KINDS = ("paraphrase", "quote")
CLAIM_STATUSES = ("active", "superseded", "retracted")

LOCATOR_CARD = "card"
LOCATOR_STUDY = "study"
LOCATOR_SURVEY = "survey"
LOCATOR_PRIMARY = "primary"
LOCATOR_UNAVAILABLE = "unavailable"
LOCATOR_KINDS = (
    LOCATOR_CARD,
    LOCATOR_STUDY,
    LOCATOR_SURVEY,
    LOCATOR_PRIMARY,
    LOCATOR_UNAVAILABLE,
)

_ALLOWED_KEYS = {
    "schema_version",
    "paper_id",
    "claim_id",
    "assertion_kind",
    "assertion",
    "quote",
    "claim_type",
    "source_level",
    "verification",
    "locator",
    "qualification",
    "status",
}
_REQUIRED_KEYS = _ALLOWED_KEYS - {"quote", "qualification"}

_CLAIM_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DOCUMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.md$")

_RESEARCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _RESEARCH_DIR.parents[2]
_DEFAULT_CLAIMS = _RESEARCH_DIR / "literature" / "claims.jsonl"


class ClaimError(ValueError):
    """A claim record is unusable or disagrees with the literature state."""

    def __init__(self, message, kind="CLAIM_INVALID", detail=None):
        super(ClaimError, self).__init__(message)
        self.kind = kind
        self.detail = dict(detail or {})

    def as_dict(self):
        return {"error": str(self), "kind": self.kind, "detail": self.detail}


@dataclass(frozen=True)
class ClaimRecord:
    """One paper-bound assertion with its provenance."""

    paper_id: str
    claim_id: str
    assertion_kind: str
    assertion: str
    claim_type: str
    source_level: str
    verification: str
    locator: dict
    status: str
    quote: str = ""
    qualification: str = ""

    @property
    def claim_ref(self):
        """Stable reference used by every query result and by future tooling."""

        return "{}#{}".format(self.paper_id, self.claim_id)

    def as_dict(self):
        return {
            "claim_ref": self.claim_ref,
            "paper_id": self.paper_id,
            "claim_id": self.claim_id,
            "assertion_kind": self.assertion_kind,
            "assertion": self.assertion,
            "quote": self.quote,
            "claim_type": self.claim_type,
            "source_level": self.source_level,
            "verification": self.verification,
            "locator": dict(self.locator),
            "qualification": self.qualification,
            "status": self.status,
        }


def normalize_whitespace(text):
    """Collapse whitespace so a Markdown-wrapped verbatim quote stays checkable."""

    return " ".join(text.split())


class ClaimRegistry:
    """Validated claim records with exact, deterministic lookup."""

    def __init__(self, records, paper_ids=()):
        self.records = tuple(records)
        self._papers = set(paper_ids)
        self._by_ref = {
            (record.paper_id, record.claim_id): record for record in records
        }
        self._by_paper = {}
        for record in records:
            self._by_paper.setdefault(record.paper_id, []).append(record)

    def claim_ids(self):
        return tuple(record.claim_ref for record in self.records)

    def claims_for_paper(self, paper_id, claim_type=None):
        """Return one paper's claims in deterministic order, optionally filtered."""

        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ClaimError("paper_id must be a non-empty string")
        paper_id = paper_id.strip()
        if paper_id not in self._papers:
            raise ClaimError(
                "unknown paper_id {!r} in the claim registry".format(paper_id),
                kind="UNKNOWN_PAPER",
                detail={"paper_id": paper_id},
            )
        records = self._by_paper.get(paper_id, ())
        if claim_type is not None:
            if claim_type not in CLAIM_TYPES:
                raise ClaimError(
                    "claim_type must be one of: {}".format(", ".join(CLAIM_TYPES)),
                    kind="INVALID_REFERENCE",
                    detail={"claim_type": claim_type},
                )
            records = tuple(
                record for record in records if record.claim_type == claim_type
            )
        return records

    def get_claim(self, paper_id, claim_id):
        """Exact lookup by ``(paper_id, claim_id)``; no fuzzy or text matching."""

        key = (str(paper_id).strip(), str(claim_id).strip())
        record = self._by_ref.get(key)
        if record is None:
            raise ClaimError(
                "no recorded claim {!r}#{!r}".format(key[0], key[1]),
                kind="UNKNOWN_CLAIM",
                detail={
                    "paper_id": key[0],
                    "claim_id": key[1],
                    "available": sorted(
                        ref[1] for ref in self._by_ref if ref[0] == key[0]
                    ),
                },
            )
        return record


def load_claims(
    claims_path=_DEFAULT_CLAIMS,
    *,
    repo_root=_REPO_ROOT,
    literature_dir=None,
    reader=None,
):
    """Load and validate the claim registry against the literature state.

    `claims_path` may live outside the literature directory (tests validate
    fixtures); `literature_dir` then names the catalog/manifest to check against.
    """

    claims_path = Path(claims_path)
    repo_root = Path(repo_root).resolve()
    literature_dir = Path(literature_dir) if literature_dir is not None else claims_path.parent
    catalog = literature_catalog.load_catalog(
        literature_dir / "catalog.jsonl",
        repo_root=repo_root,
        validate_references=False,
    )
    paper_ids = {entry.paper_id for entry in catalog.entries}
    retained = _retained_paper_ids(repo_root, literature_dir, paper_ids)
    reader = reader if reader is not None else literature_read.LiteratureReader(
        repo_root=repo_root,
        primary=literature_primary.LiteraturePrimary(
            repo_root=repo_root,
            catalog_path=literature_dir / "catalog.jsonl",
            manifest_path=literature_dir / "primary_manifest.jsonl",
        ),
    )

    records = []
    seen = {}
    for line_number, line in enumerate(_read_lines(claims_path), start=1):
        if not line.strip():
            raise ClaimError("claims line {} is blank".format(line_number))
        try:
            raw = json.loads(line)
        except ValueError as exc:
            raise ClaimError(
                "claims line {} is not valid JSON: {}".format(line_number, exc)
            ) from exc
        record = _validate_record(
            raw, line_number, paper_ids=paper_ids, retained=retained, reader=reader
        )
        key = (record.paper_id, record.claim_id)
        if key in seen:
            raise ClaimError(
                "duplicate claim identity {!r}#{!r}".format(key[0], key[1]),
                kind="DUPLICATE_CLAIM",
                detail={"claim_ref": "{0[0]}#{0[1]}".format(key)},
            )
        seen[key] = True
        records.append(record)

    if [record.claim_ref for record in records] != sorted(
        record.claim_ref for record in records
    ):
        raise ClaimError("claims must be sorted by paper_id then claim_id")

    return ClaimRegistry(records, paper_ids)


def _read_lines(path):
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ClaimError(
            "cannot read claims registry {}: {}".format(path, exc)
        ) from exc


def _retained_paper_ids(repo_root, literature_dir, paper_ids):
    """Papers whose primary artifact this repository actually retains."""

    try:
        primary = literature_primary.LiteraturePrimary(
            repo_root=repo_root,
            catalog_path=literature_dir / "catalog.jsonl",
            manifest_path=literature_dir / "primary_manifest.jsonl",
        )
    except literature_primary.PrimaryManifestError as exc:
        raise ClaimError(
            "cannot read the primary manifest: {}".format(exc)
        ) from exc
    return {
        row.paper_id for row in primary.manifest_rows() if row.paper_id in paper_ids
    }


def _require_string(record, key, where):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ClaimError("{} .{} must be a non-empty trimmed string".format(where, key))
    return value


def _validate_record(record, line_number, *, paper_ids, retained, reader):
    where = "claims line {}".format(line_number)
    if not isinstance(record, dict):
        raise ClaimError("{} must be a JSON object".format(where))
    unknown = sorted(set(record) - _ALLOWED_KEYS)
    missing = sorted(_REQUIRED_KEYS - set(record))
    if unknown:
        raise ClaimError(
            "{} has unknown key(s): {} (screening verdicts, research decisions and "
            "authorizations do not belong in a claim record)".format(
                where, ", ".join(unknown)
            )
        )
    if missing:
        raise ClaimError(
            "{} is missing key(s): {}".format(where, ", ".join(missing))
        )
    if record["schema_version"] != SCHEMA_VERSION:
        raise ClaimError(
            "{}.schema_version must be {}".format(where, SCHEMA_VERSION)
        )

    paper_id = _require_string(record, "paper_id", where)
    if paper_id not in paper_ids:
        raise ClaimError(
            "{} references unknown paper_id {!r}".format(where, paper_id)
        )
    claim_id = _require_string(record, "claim_id", where)
    if not _CLAIM_ID.match(claim_id):
        raise ClaimError(
            "{}.claim_id must be a lowercase hyphenated slug (stable across "
            "wording edits)".format(where)
        )

    assertion_kind = record["assertion_kind"]
    if assertion_kind not in ASSERTION_KINDS:
        raise ClaimError(
            "{}.assertion_kind must be one of: {}".format(
                where, ", ".join(ASSERTION_KINDS)
            )
        )
    assertion = _require_string(record, "assertion", where)

    claim_type = record["claim_type"]
    if claim_type not in CLAIM_TYPES:
        raise ClaimError(
            "{}.claim_type must be one of: {}".format(where, ", ".join(CLAIM_TYPES))
        )

    source_level = record["source_level"]
    if source_level not in SOURCE_LEVELS:
        raise ClaimError(
            "{}.source_level must be one of: {}".format(
                where, ", ".join(SOURCE_LEVELS)
            )
        )
    verification = record["verification"]
    if verification not in VERIFICATION_LEVELS:
        raise ClaimError(
            "{}.verification must be one of: {}".format(
                where, ", ".join(VERIFICATION_LEVELS)
            )
        )
    if verification == PRIMARY_VERIFIED:
        if source_level != "primary":
            raise ClaimError(
                "{}.verification is primary_verified but source_level is {!r}; only "
                "a primary source can be primary-verified".format(where, source_level)
            )
        if paper_id not in retained:
            raise ClaimError(
                "{} claims primary_verified for {!r}, but this repository retains no "
                "primary artifact for that paper".format(where, paper_id),
                kind="UNSUPPORTED_VERIFICATION",
                detail={"paper_id": paper_id},
            )
    if source_level == "primary" and paper_id not in retained:
        raise ClaimError(
            "{} uses source_level 'primary' for {!r}, but no primary artifact is "
            "retained; record the assertion against the card that carries it".format(
                where, paper_id
            ),
            kind="UNSUPPORTED_SOURCE",
            detail={"paper_id": paper_id},
        )

    section_text, locator = _validate_locator(
        record["locator"], where, paper_id=paper_id, source_level=source_level,
        reader=reader,
    )

    quote = record.get("quote", "")
    if assertion_kind == "quote":
        quote = _require_string(record, "quote", where)
        if source_level == "primary" and paper_id not in retained:
            raise ClaimError(
                "{} quotes the primary artifact, which is not retained".format(where)
            )
    elif quote:
        raise ClaimError(
            "{}.quote is only allowed when assertion_kind is 'quote'; a paraphrase "
            "must never carry verbatim text".format(where)
        )

    qualification = record.get("qualification", "")
    if qualification and (
        not isinstance(qualification, str) or not qualification.strip()
    ):
        raise ClaimError("{}.qualification must be a non-empty string".format(where))

    status = record["status"]
    if status not in CLAIM_STATUSES:
        raise ClaimError(
            "{}.status must be one of: {}".format(where, ", ".join(CLAIM_STATUSES))
        )

    if quote:
        if locator["kind"] == LOCATOR_UNAVAILABLE:
            raise ClaimError(
                "{}.quote is not allowed with an unavailable locator".format(where)
            )
        if normalize_whitespace(quote) not in normalize_whitespace(section_text):
            raise ClaimError(
                "{} quote is not present verbatim inside the located section".format(
                    where
                ),
                kind="QUOTE_NOT_VERBATIM",
                detail={"locator": locator},
            )

    return ClaimRecord(
        paper_id=paper_id,
        claim_id=claim_id,
        assertion_kind=assertion_kind,
        assertion=assertion,
        claim_type=claim_type,
        source_level=source_level,
        verification=verification,
        locator=locator,
        status=status,
        quote=quote,
        qualification=qualification,
    )


def _validate_locator(locator, where, *, paper_id, source_level, reader):
    if not isinstance(locator, dict):
        raise ClaimError("{}.locator must be an object".format(where))
    kind = locator.get("kind")
    if kind not in LOCATOR_KINDS:
        raise ClaimError(
            "{}.locator.kind must be one of: {}".format(where, ", ".join(LOCATOR_KINDS))
        )
    if kind == LOCATOR_UNAVAILABLE:
        _require_string(locator, "reason", where + ".locator")
        if set(locator) - {"kind", "reason"}:
            raise ClaimError(
                "{}.locator has unexpected keys for an unavailable locator".format(where)
            )
        return "", {"kind": kind, "reason": locator["reason"]}

    if kind != source_level:
        raise ClaimError(
            "{}.locator.kind {!r} must match source_level {!r}".format(
                where, kind, source_level
            )
        )

    if kind == LOCATOR_CARD:
        allowed = {"kind", "anchor"}
        _require_string(locator, "anchor", where + ".locator")
    elif kind == LOCATOR_STUDY:
        allowed = {"kind", "study_id", "artifact", "anchor"}
        _require_string(locator, "study_id", where + ".locator")
        _require_string(locator, "artifact", where + ".locator")
        _require_string(locator, "anchor", where + ".locator")
        if locator["artifact"] not in literature_read._STUDY_ARTIFACT_KINDS:
            raise ClaimError(
                "{}.locator.artifact must be one of: {}".format(
                    where, ", ".join(literature_read._STUDY_ARTIFACT_KINDS)
                )
            )
    elif kind == LOCATOR_SURVEY:
        allowed = {"kind", "document", "anchor"}
        document = _require_string(locator, "document", where + ".locator")
        if not _DOCUMENT.match(document):
            raise ClaimError(
                "{}.locator.document must be a direct literature_survey/*.md "
                "filename".format(where)
            )
        _require_string(locator, "anchor", where + ".locator")
    else:  # primary
        allowed = {"kind", "page", "section", "equation", "figure", "table"}
        present = sorted(set(locator) - {"kind"})
        if not present:
            raise ClaimError(
                "{}.locator needs at least one of page/section/equation/figure/"
                "table".format(where)
            )
        for key in present:
            if key not in allowed:
                raise ClaimError(
                    "{}.locator has unexpected key {!r} for a primary locator".format(
                        where, key
                    )
                )
            _require_string(locator, key, where + ".locator")

    unexpected = sorted(set(locator) - allowed)
    if unexpected:
        raise ClaimError(
            "{}.locator has unexpected key(s) for kind {!r}: {}".format(
                where, kind, ", ".join(unexpected)
            )
        )

    text = _locator_text(dict(locator), reader, where, paper_id)
    anchor = locator["anchor"]
    section = _section_text(text, anchor)
    if section is None:
        raise ClaimError(
            "{}.locator.anchor {!r} is not a heading in the located artifact".format(
                where, anchor
            ),
            kind="LOCATOR_NOT_FOUND",
            detail={"anchor": anchor},
        )
    return section, {key: locator[key] for key in sorted(locator)}


def _section_text(text, anchor):
    """Return the body of the section introduced by `anchor`, or None.

    A located assertion must sit *inside* the named section, not merely in the
    same file: this is what keeps a claim from borrowing a neighbouring
    section's authority.
    """

    lines = text.splitlines()
    wanted = normalize_whitespace(anchor)
    level = None
    start = None
    for index, line in enumerate(lines):
        if not line.startswith("#"):
            continue
        if normalize_whitespace(line.lstrip("#").strip()) == wanted:
            level = len(line) - len(line.lstrip("#"))
            start = index
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.startswith("#"):
            depth = len(line) - len(line.lstrip("#"))
            if depth <= level:
                end = index
                break
    return "\n".join(lines[start:end])


def _locator_text(locator, reader, where, paper_id):
    """Read the located artifact through the same bounded reader the agent uses.

    A claim can therefore only point where the Literature Agent can actually read,
    and a quote is checked against exactly those bytes.
    """

    if locator["kind"] == LOCATOR_UNAVAILABLE:
        raise ClaimError(
            "{} cannot resolve text for an unavailable locator".format(where)
        )
    if locator["kind"] == LOCATOR_PRIMARY:
        raise ClaimError(
            "{} cannot read a primary locator: no primary artifact is retained".format(
                where
            ),
            kind="LOCATOR_NOT_READABLE",
        )
    try:
        if locator["kind"] == LOCATOR_CARD:
            return reader.read_card(
                paper_id, literature_read.MAX_CHARS_CEILING
            ).text
        if locator["kind"] == LOCATOR_STUDY:
            return reader.read_study(
                locator["study_id"],
                locator["artifact"],
                literature_read.MAX_CHARS_CEILING,
            ).text
        return reader.read_survey(
            locator["document"], literature_read.MAX_CHARS_CEILING
        ).text
    except literature_read.LiteratureReadError as exc:
        raise ClaimError(
            "{} locator cannot be read: {}".format(where, exc),
            kind="LOCATOR_NOT_READABLE",
            detail=exc.detail,
        ) from exc


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("validate", help="validate the claim registry")
    commands.add_parser("list", help="list every recorded claim reference")
    paper = commands.add_parser("paper", help="list one paper's recorded claims")
    paper.add_argument("paper_id")
    paper.add_argument("--claim-type", choices=CLAIM_TYPES)
    get = commands.add_parser("get", help="exact claim lookup")
    get.add_argument("paper_id")
    get.add_argument("claim_id")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    command = args.command or "validate"
    try:
        registry = load_claims()
        if command == "validate":
            print("literature claims: OK ({} record(s))".format(len(registry.records)))
        elif command == "list":
            print(
                json.dumps(
                    {"claim_refs": list(registry.claim_ids())}, sort_keys=True
                )
            )
        elif command == "paper":
            records = registry.claims_for_paper(args.paper_id, args.claim_type)
            print(
                json.dumps(
                    {"claims": [record.as_dict() for record in records]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            print(
                json.dumps(
                    registry.get_claim(args.paper_id, args.claim_id).as_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return 0
    except (ClaimError, literature_catalog.CatalogError) as exc:
        payload = (
            exc.as_dict()
            if isinstance(exc, ClaimError)
            else {"error": str(exc), "kind": "CATALOG_INVALID", "detail": {}}
        )
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
