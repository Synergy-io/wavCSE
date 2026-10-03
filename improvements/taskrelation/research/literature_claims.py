"""Source-bound literature claim records.

Purpose: attribution correctness, not semantic search. A record answers
"what exact literature assertion have we recorded, where did it come from, and
what evidence supports that attribution?"

A claim owns the **proposition** — a paper-bound assertion. Where that assertion
was observed or supported is owned by one or more embedded ``EvidenceReference``
objects. Separating the two is what lets a proposition accumulate evidence
without deleting the evidence it was first recorded from: a claim first recorded
from a card can later gain a primary reference, and its *derived* source level
rises to ``primary`` while the card provenance is preserved.

The motivating failure (LIT-AGENT-V1 evaluation run 2): an agent attributed the
published ``λ₂`` grid to the per-paper card, which does not state any grid — the
grid is recorded in a survey document citing JMLR §4.1, while the card says only
that the paper selects ``λ₂`` on data. The value existed; the *provenance* was
invented. An evidence reference therefore names the artifact class and anchor it
was read from, and a verbatim quote must be provably present in that artifact.

Scope rules:

* every record is bound to one ``paper_id`` — no free-floating literature prose;
* the registry holds **paper-attributed** assertions only. Our own experimental
  configuration (a restricted grid, a frozen λ, a declared deviation) is research
  state and stays in ``DECISIONS.md`` / Study artifacts, owned by the main
  research agent;
* ``source_level`` and ``verification`` are **derived** from ``evidence`` and are
  never stored: a primary reference binds an explicit artifact version
  (``preprint``/``published``) **and** that artifact's SHA-256, so a claim can
  never drift onto different bytes. An evidence reference that quotes a primary
  artifact is read through the same bounded reader the Literature Agent uses and
  checked verbatim against the extracted page; a paraphrase is validated against
  the manifest binding alone, so the registry stays deterministic without the
  disposable local cache.

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
from types import MappingProxyType

from improvements.taskrelation.research import literature_catalog
from improvements.taskrelation.research import literature_primary
from improvements.taskrelation.research import literature_primary_text
from improvements.taskrelation.research import literature_read


# V2 splits the proposition from its evidence: a claim carries ``evidence`` (a
# non-empty list of references) instead of a fused locator + stored source level
# + stored verification.
SCHEMA_VERSION = 2

# The artifact class an evidence reference names. ``primary`` is the paper's own
# artifact; the others are derived records at one or more removes from it. These
# are the kinds V1 represented and no others.
EVIDENCE_PRIMARY = "primary"
EVIDENCE_CARD = "card"
EVIDENCE_SURVEY = "survey"
EVIDENCE_STUDY = "study"
EVIDENCE_KINDS = (EVIDENCE_PRIMARY, EVIDENCE_CARD, EVIDENCE_SURVEY, EVIDENCE_STUDY)

# Whether the proposition has been checked against the paper's own artifact.
# Derived from ``evidence`` and never stored on a record.
PRIMARY_VERIFIED = "primary_verified"
DERIVED_EXISTING_RECORD = "derived_existing_record"
VERIFICATION_LEVELS = (PRIMARY_VERIFIED, DERIVED_EXISTING_RECORD)

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

# A primary evidence reference names a retained artifact version and its digest.
# The text of a quoted reference is read through the bounded reader; when the
# disposable local copy is not present the quote cannot be verified and is refused
# rather than accepted unchecked.
PRIMARY_ARTIFACT_NOT_LOCAL = "PRIMARY_ARTIFACT_NOT_LOCAL"
ARTIFACT_MISMATCH = "ARTIFACT_MISMATCH"
DUPLICATE_EVIDENCE = "DUPLICATE_EVIDENCE"

# Required keys per evidence kind; the allowed set adds optional keys. Every
# present value must be a non-empty trimmed string.
_EVIDENCE_REQUIRED = {
    EVIDENCE_PRIMARY: ("role", "sha256", "page"),
    EVIDENCE_CARD: ("anchor",),
    EVIDENCE_SURVEY: ("document", "anchor"),
    EVIDENCE_STUDY: ("study_id", "artifact", "anchor"),
}
_EVIDENCE_OPTIONAL = {
    EVIDENCE_PRIMARY: ("page_end", "section", "equation", "figure", "table", "quote"),
    EVIDENCE_CARD: ("quote",),
    EVIDENCE_SURVEY: ("quote",),
    EVIDENCE_STUDY: ("quote",),
}
_EVIDENCE_ALLOWED = {
    kind: tuple(_EVIDENCE_REQUIRED[kind] + _EVIDENCE_OPTIONAL[kind])
    for kind in EVIDENCE_KINDS
}

_ALLOWED_KEYS = {
    "schema_version",
    "paper_id",
    "claim_id",
    "assertion_kind",
    "assertion",
    "claim_type",
    "evidence",
    "qualification",
    "status",
}
_REQUIRED_KEYS = _ALLOWED_KEYS - {"qualification"}

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
class EvidenceReference:
    """Where one recorded assertion was observed or supported.

    ``fields`` holds exactly the kind-specific keys (never the ``kind`` itself),
    normalized and sorted, so two references can be compared deterministically.
    """

    kind: str
    fields: MappingProxyType

    @property
    def quote(self):
        return self.fields.get("quote", "")

    def as_dict(self):
        payload = {"kind": self.kind}
        payload.update(self.fields)
        return payload


@dataclass(frozen=True)
class ClaimRecord:
    """One paper-bound proposition with its evidence references.

    ``source_level`` and ``verification`` are derived views over ``evidence``,
    never stored authority.
    """

    paper_id: str
    claim_id: str
    assertion_kind: str
    assertion: str
    claim_type: str
    evidence: tuple
    status: str
    qualification: str = ""

    @property
    def claim_ref(self):
        """Stable reference used by every query result and by future tooling."""

        return "{}#{}".format(self.paper_id, self.claim_id)

    @property
    def evidence_kinds(self):
        """The evidence kinds present, in canonical enumeration order."""

        return tuple(
            kind for kind in EVIDENCE_KINDS if any(ref.kind == kind for ref in self.evidence)
        )

    @property
    def source_level(self):
        """The derived strongest evidence level.

        ``primary`` when any primary reference is present. Otherwise the present
        non-primary kinds joined with ``+`` in canonical enumeration order; the
        order is enumeration order, **not** a scientific ranking of card, survey
        and study, which V1 never ranked.
        """

        kinds = self.evidence_kinds
        if EVIDENCE_PRIMARY in kinds:
            return EVIDENCE_PRIMARY
        return "+".join(kinds)

    @property
    def verification(self):
        """The derived verification level.

        ``primary_verified`` only when a validated primary reference is present —
        the reference is version- and digest-bound, and a quote is already
        validated verbatim against the artifact. Any other evidence is a derived
        existing record.
        """

        if any(ref.kind == EVIDENCE_PRIMARY for ref in self.evidence):
            return PRIMARY_VERIFIED
        return DERIVED_EXISTING_RECORD

    def as_dict(self):
        return {
            "claim_ref": self.claim_ref,
            "paper_id": self.paper_id,
            "claim_id": self.claim_id,
            "assertion_kind": self.assertion_kind,
            "assertion": self.assertion,
            "claim_type": self.claim_type,
            "evidence": [ref.as_dict() for ref in self.evidence],
            "source_level": self.source_level,
            "verification": self.verification,
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
    manifest = _manifest_index(repo_root, literature_dir, paper_ids)
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
            raw,
            line_number,
            paper_ids=paper_ids,
            manifest=manifest,
            reader=reader,
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


def _manifest_index(repo_root, literature_dir, paper_ids):
    """Map ``(paper_id, role)`` to the retained-artifact manifest row.

    Retention is a Git-declared fact: the row names the artifact version and its
    digest, which is what a primary evidence reference binds to. The bytes
    themselves live in the disposable local cache and are verified when the
    artifact is read.
    """

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
        (row.paper_id, row.role): row
        for row in primary.manifest_rows()
        if row.paper_id in paper_ids
    }


def _require_string(record, key, where):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ClaimError("{} .{} must be a non-empty trimmed string".format(where, key))
    return value


def _validate_record(record, line_number, *, paper_ids, manifest, reader):
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

    evidence = _validate_evidence(
        record["evidence"],
        where,
        paper_id=paper_id,
        assertion_kind=assertion_kind,
        manifest=manifest,
        reader=reader,
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

    return ClaimRecord(
        paper_id=paper_id,
        claim_id=claim_id,
        assertion_kind=assertion_kind,
        assertion=assertion,
        claim_type=claim_type,
        evidence=evidence,
        status=status,
        qualification=qualification,
    )


def _validate_evidence(evidence, where, *, paper_id, assertion_kind, manifest, reader):
    """Validate the non-empty evidence list and return the references.

    A proposition is never recorded without at least one real location, and a
    quote is validated against the exact reference that carries it.
    """

    if not isinstance(evidence, list) or not evidence:
        raise ClaimError(
            "{}.evidence must be a non-empty list of evidence references".format(
                where
            )
        )
    references = []
    seen = {}
    quoted = False
    for index, raw in enumerate(evidence):
        ref_where = "{}.evidence[{}]".format(where, index)
        reference, has_quote = _validate_evidence_reference(
            raw,
            ref_where,
            paper_id=paper_id,
            assertion_kind=assertion_kind,
            manifest=manifest,
            reader=reader,
        )
        key = (
            reference.kind,
            tuple(sorted(reference.fields.items(), key=lambda item: item[0])),
        )
        if key in seen:
            raise ClaimError(
                "{} duplicates the evidence reference at {}".format(ref_where, seen[key]),
                kind=DUPLICATE_EVIDENCE,
                detail={"kind": reference.kind},
            )
        seen[key] = ref_where
        quoted = quoted or has_quote
        references.append(reference)

    if assertion_kind == "quote" and not quoted:
        raise ClaimError(
            "{} has assertion_kind 'quote' but no evidence reference carries the "
            "verbatim text".format(where)
        )
    return tuple(references)


def _validate_evidence_reference(raw, where, *, paper_id, assertion_kind, manifest,
                                reader):
    if not isinstance(raw, dict):
        raise ClaimError("{} must be an object".format(where))
    kind = raw.get("kind")
    if kind not in EVIDENCE_KINDS:
        raise ClaimError(
            "{} .kind must be one of: {}".format(where, ", ".join(EVIDENCE_KINDS))
        )
    allowed = _EVIDENCE_ALLOWED[kind]
    unknown = sorted(set(raw) - set(allowed) - {"kind"})
    missing = sorted(set(_EVIDENCE_REQUIRED[kind]) - set(raw))
    if unknown:
        raise ClaimError(
            "{} has unexpected key(s) for kind {!r}: {}".format(
                where, kind, ", ".join(unknown)
            )
        )
    if missing:
        raise ClaimError(
            "{} is missing key(s) for kind {!r}: {}".format(
                where, kind, ", ".join(missing)
            )
        )
    for key in sorted(set(raw) - {"kind"}):
        _require_string(raw, key, where)

    has_quote = "quote" in raw
    if has_quote and assertion_kind != "quote":
        raise ClaimError(
            "{}.quote is only allowed when assertion_kind is 'quote'; a paraphrase "
            "must never carry verbatim text".format(where)
        )

    section_text = _validate_evidence_location(
        kind, raw, where, paper_id=paper_id, manifest=manifest, reader=reader,
        quoted=has_quote,
    )
    if has_quote and normalize_whitespace(raw["quote"]) not in normalize_whitespace(
        section_text
    ):
        raise ClaimError(
            "{} quote is not present verbatim inside the located section".format(where),
            kind="QUOTE_NOT_VERBATIM",
            detail={"evidence": where, "kind": kind},
        )

    fields = {key: raw[key] for key in sorted(set(raw) - {"kind"})}
    return EvidenceReference(kind=kind, fields=MappingProxyType(fields)), has_quote


def _validate_evidence_location(kind, raw, where, *, paper_id, manifest, reader,
                                quoted):
    """Validate one reference's location, returning its located section text.

    The returned text is what a carried quote is checked against. For a primary
    reference it is read only when a quote must be verified, which keeps a
    paraphrase claim deterministically validatable without the local cache.
    """

    if kind == EVIDENCE_PRIMARY:
        return _primary_evidence_text(
            raw, where, paper_id=paper_id, manifest=manifest, reader=reader,
            quoted=quoted,
        )

    if kind == EVIDENCE_CARD:
        text = _read_evidence_text(
            lambda: reader.read_card(paper_id, literature_read.MAX_CHARS_CEILING).text,
            where,
        )
    elif kind == EVIDENCE_STUDY:
        if raw["artifact"] not in literature_read._STUDY_ARTIFACT_KINDS:
            raise ClaimError(
                "{}.artifact must be one of: {}".format(
                    where, ", ".join(literature_read._STUDY_ARTIFACT_KINDS)
                )
            )
        text = _read_evidence_text(
            lambda: reader.read_study(
                raw["study_id"], raw["artifact"], literature_read.MAX_CHARS_CEILING
            ).text,
            where,
        )
    else:  # survey
        if not _DOCUMENT.match(raw["document"]):
            raise ClaimError(
                "{}.document must be a direct literature_survey/*.md filename".format(
                    where
                )
            )
        text = _read_evidence_text(
            lambda: reader.read_survey(
                raw["document"], literature_read.MAX_CHARS_CEILING
            ).text,
            where,
        )

    section = _section_text(text, raw["anchor"])
    if section is None:
        raise ClaimError(
            "{}.anchor {!r} is not a heading in the located artifact".format(
                where, raw["anchor"]
            ),
            kind="LOCATOR_NOT_FOUND",
            detail={"anchor": raw["anchor"]},
        )
    return section


def _read_evidence_text(read, where):
    """Read a located artifact through the same bounded reader the agent uses."""

    try:
        return read()
    except literature_read.LiteratureReadError as exc:
        raise ClaimError(
            "{} cannot be read: {}".format(where, exc),
            kind="LOCATOR_NOT_READABLE",
            detail=exc.detail,
        ) from exc


def _primary_evidence_text(raw, where, *, paper_id, manifest, reader, quoted):
    """Validate a primary reference against the manifest, reading it if quoted.

    The binding is (version role, digest): the reference must name a version role
    this repository retains for the paper, and the exact digest of that retained
    artifact, so a claim can never drift onto different bytes or a different
    version. A verbatim quote is additionally read through the bounded reader the
    Literature Agent uses and checked against the extracted page; a paraphrase is
    validated against the binding alone, which keeps the registry deterministic
    without the disposable local cache.
    """

    role = raw["role"]
    if role not in literature_primary.VERSION_ROLES:
        raise ClaimError(
            "{}.role must name a concrete version ({})".format(
                where, ", ".join(literature_primary.VERSION_ROLES)
            ),
            kind="CLAIM_INVALID",
            detail={"role": role},
        )
    entry = manifest.get((paper_id, role))
    if entry is None:
        raise ClaimError(
            "{} names {!r} {!r}, which this repository does not retain".format(
                where, paper_id, role
            ),
            kind="LOCATOR_NOT_READABLE",
            detail={"paper_id": paper_id, "role": role},
        )
    if raw["sha256"] != entry.sha256:
        raise ClaimError(
            "{}.sha256 is not the digest of the retained {!r} artifact for "
            "{!r}".format(where, role, paper_id),
            kind=ARTIFACT_MISMATCH,
            detail={
                "paper_id": paper_id,
                "role": role,
                "retained_sha256": entry.sha256,
            },
        )
    page = _positive_page(raw["page"], where, "page")
    page_end = None
    if "page_end" in raw:
        page_end = _positive_page(raw["page_end"], where, "page_end")
        if page_end < page:
            raise ClaimError(
                "{}.page_end must not precede page".format(where),
                kind="CLAIM_INVALID",
            )
    if not quoted:
        return ""
    try:
        content = reader.read_primary(
            paper_id,
            literature_read.MAX_CHARS_CEILING,
            role=role,
            page=page,
            page_end=page_end,
        )
    except (
        literature_primary.PrimaryError,
        literature_primary_text.PrimaryTextError,
        literature_read.LiteratureReadError,
    ) as exc:
        detail = dict(getattr(exc, "detail", {}) or {})
        if getattr(exc, "kind", None) == literature_primary.STORAGE_NOT_CONFIGURED:
            raise ClaimError(
                "{} names {!r} {!r}, whose bytes are not in the local primary "
                "cache; a quoted primary reference cannot be verified without "
                "them".format(where, paper_id, role),
                kind=PRIMARY_ARTIFACT_NOT_LOCAL,
                detail=detail,
            ) from exc
        raise ClaimError(
            "{} cannot be read: {}".format(where, exc),
            kind="LOCATOR_NOT_READABLE",
            detail=detail,
        ) from exc
    return content.text


def _positive_page(value, where, key):
    if not isinstance(value, str) or not value.isdigit() or int(value) < 1:
        raise ClaimError(
            "{}.{} must be a 1-based page number as a string".format(where, key),
            kind="CLAIM_INVALID",
        )
    return int(value)


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
