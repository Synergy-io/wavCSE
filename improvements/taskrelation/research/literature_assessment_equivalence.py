"""Transitional equivalence check between the canonical assessment registry and
the legacy assessment representations it replaces.

INC-V2-2 makes ``literature/assessments.jsonl`` canonical but deliberately leaves
the V1 representations in place (LT-0001 ``result.json`` /
``primary_sources_reviewed``, ``STUDIES.jsonl`` ``cards``, per-card verdict
sections, ``INDEX.md`` tables) as migration witnesses. This module re-derives the
canonical records from those legacy shapes and compares them with the registry,
so a change to a card verdict, a registry card list or an LT result relationship
cannot silently drift from the canonical authority.

The comparison is semantic, not byte equality: legacy shapes differ per
investigation and are parsed explicitly.

* LT-0001's structured authority is ``result.json.primary_sources_reviewed``
  (private ``key`` + ``url``, ``role``, ``decision``); the URL resolves to the
  canonical ``paper_id`` and the card ``## LT-0001 decision`` body supplies the
  migrated ``reason_summary``. LT-0001 wrote no per-paper gate structure, so its
  records carry no gates.
* LT-0002's paper set is ``STUDIES.jsonl`` ``cards``; each card's
  ``## LT-0002 assessment`` section holds the six numbered gates, the verdict and
  the reason clause; the ``INDEX.md`` LT-0002 table supplies the family role.

This checker is transitional: INC-V2-3 removes the legacy representations and can
drop this module with them.

Usage::

    python -m improvements.taskrelation.research.literature_assessment_equivalence check
"""

import argparse
import json
import re
import sys
from pathlib import Path

from improvements.taskrelation.research import literature_assessment
from improvements.taskrelation.research import literature_catalog


_RESEARCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _RESEARCH_DIR.parents[2]
_LITERATURE_DIR = _RESEARCH_DIR / "literature"
_STUDIES_PATH = _RESEARCH_DIR / "STUDIES.jsonl"


class AssessmentEquivalenceError(ValueError):
    """The canonical registry and the legacy representations disagree."""

    def __init__(self, message, *, differences=None):
        super().__init__(message)
        self.differences = list(differences or ())


# --- legacy parsing -------------------------------------------------------

_NUMBERED_GATE = re.compile(
    r"^\d+\.\s+\*\*(?P<label>[^*]+?)\s+—\s+(?P<verdict>[^*]+?)\.\*\*", re.M
)
_BULLET_GATE = re.compile(
    r"^\*\s+\*\*Gate\s+\d+\s+—\s+(?P<label>[^*]+?):\s+(?P<verdict>[^*]+?)\.\*\*",
    re.M,
)
_INDEX_ROW = re.compile(r"\|\s*\[[^\]]+\]\(([a-z0-9-]+)\.md\)\s*\|\s*([AB])\b")
_SECTION = r"^## {}(.*?)(?=^## |\Z)"
_LT0001_ASSESSED_AT = "2026-09-22T09:15:21+00:00"
_LT0002_ASSESSED_AT = "2026-09-22T18:45:00+00:00"
_VERDICTS = {
    "PASS": "pass",
    "PASS WITH DOCUMENTED DEVIATION": "pass_with_documented_deviation",
    "FAIL": "fail",
}


def _collapse_markdown(text):
    return " ".join(text.replace("**", "").replace("`", "").split())


def _normalize_gate_verdict(label):
    """Map an LT-0002 card's verbatim gate label to the normalised vocabulary."""

    if label.startswith("N/A"):
        return "not_applicable"
    if "FAIL" in label:
        return "fail"
    if "PARTIAL" in label:
        return "partial"
    if "STATED CHANGE" in label:
        return "pass_with_deviation"
    if label.startswith("PASS"):
        return "pass"
    raise AssessmentEquivalenceError(
        "unmapped LT-0002 gate verdict label {!r}".format(label)
    )


def _section_body(text, heading):
    match = re.search(_SECTION.format(re.escape(heading)), text, re.M | re.S)
    if match is None:
        raise AssessmentEquivalenceError(
            "missing '## {}' section".format(heading)
        )
    return match.group(1)


def _lt0001_result_records(literature_dir, repo_root):
    """Read LT-0001's structured source list and its private-key identity map."""

    result_path = literature_dir.parent / "studies" / "LT-0001" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    catalog = literature_catalog.load_catalog(
        literature_dir / "catalog.jsonl", repo_root=repo_root, validate_references=False
    )
    records = []
    key_map = {}
    for entry in result["primary_sources_reviewed"]:
        key = entry["key"]
        try:
            paper_id = catalog.lookup(entry["url"]).paper_id
        except literature_catalog.CatalogError as exc:
            raise AssessmentEquivalenceError(
                "LT-0001 private key {!r} url does not resolve to a canonical "
                "paper_id: {}".format(key, exc)
            ) from exc
        if paper_id in key_map.values():
            raise AssessmentEquivalenceError(
                "two LT-0001 private keys resolve to the same paper_id {!r}".format(
                    paper_id
                )
            )
        key_map[key] = paper_id
        card = (literature_dir / (paper_id + ".md")).read_text(encoding="utf-8")
        body = _section_body(card, "LT-0001 decision").strip()
        marker = re.match(r"\*\*(.+?)\.\*\*\s*(.*)", body, re.S)
        if marker is None:
            raise AssessmentEquivalenceError(
                "LT-0001 card {!r} decision section has no bold verdict marker".format(
                    paper_id
                )
            )
        records.append(
            {
                "paper_id": paper_id,
                "role": entry["role"],
                "verdict": entry["decision"],
                "gates": [],
                "reason_summary": _collapse_markdown(marker.group(2)),
                "assessment_anchor": "{}#lt-0001-decision".format(
                    _relative_card(repo_root, literature_dir, paper_id)
                ),
                "assessed_at": _LT0001_ASSESSED_AT,
            }
        )
    return records, key_map


def _relative_card(repo_root, literature_dir, paper_id):
    path = literature_dir / (paper_id + ".md")
    return path.relative_to(repo_root).as_posix()


def _lt0002_card_gates(card_text, paper_id):
    body = _section_body(card_text, "LT-0002 assessment")
    located = []
    for match in _NUMBERED_GATE.finditer(body):
        located.append((match.start(), match.group("verdict")))
    for match in _BULLET_GATE.finditer(body):
        located.append((match.start(), match.group("verdict")))
    located.sort()
    if len(located) != len(literature_assessment.GATE_IDS):
        raise AssessmentEquivalenceError(
            "LT-0002 card {!r} has {} numbered gates, expected {}".format(
                paper_id, len(located), len(literature_assessment.GATE_IDS)
            )
        )
    gates = [
        {
            "gate_id": gate_id,
            "verdict": _normalize_gate_verdict(verdict),
        }
        for gate_id, (_position, verdict) in zip(
            literature_assessment.GATE_IDS, located
        )
    ]
    verdict_match = re.search(r"\*\*Verdict:\s*(.+?)\*\*", body)
    if verdict_match is None:
        raise AssessmentEquivalenceError(
            "LT-0002 card {!r} has no Verdict clause".format(paper_id)
        )
    reason_summary = verdict_match.group(1).replace("`", "").strip()
    base = re.split(r"\s+—\s+|\.\s*$", reason_summary)[0].strip().rstrip(".")
    if base not in _VERDICTS:
        raise AssessmentEquivalenceError(
            "LT-0002 card {!r} verdict {!r} is not a known verdict".format(
                paper_id, base
            )
        )
    return gates, _VERDICTS[base], reason_summary


def _lt0002_records(literature_dir, repo_root, studies_path):
    studies = [
        json.loads(line)
        for line in Path(studies_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lt0002 = next(study for study in studies if study["study_id"] == "LT-0002")
    cards = lt0002.get("cards", [])
    index_text = (literature_dir / "INDEX.md").read_text(encoding="utf-8")
    families = {
        paper_id: "family_a_directed" if family == "A" else "family_b_estimator"
        for paper_id, family in _INDEX_ROW.findall(index_text)
    }
    records = []
    for paper_id in cards:
        if paper_id not in families:
            raise AssessmentEquivalenceError(
                "LT-0002 registry card {!r} has no family row in INDEX.md".format(
                    paper_id
                )
            )
        card = (literature_dir / (paper_id + ".md")).read_text(encoding="utf-8")
        gates, verdict, reason_summary = _lt0002_card_gates(card, paper_id)
        records.append(
            {
                "paper_id": paper_id,
                "role": families[paper_id],
                "verdict": verdict,
                "gates": gates,
                "reason_summary": reason_summary,
                "assessment_anchor": "{}#lt-0002-assessment".format(
                    _relative_card(repo_root, literature_dir, paper_id)
                ),
                "assessed_at": _LT0002_ASSESSED_AT,
            }
        )
    return records


def expected_records(
    *, repo_root=_REPO_ROOT, literature_dir=None, studies_path=None
):
    """Re-derive every canonical assessment row from the legacy representations."""

    repo_root = Path(repo_root).resolve()
    literature_dir = (
        Path(literature_dir) if literature_dir is not None else _LITERATURE_DIR
    )
    studies_path = (
        Path(studies_path) if studies_path is not None else _STUDIES_PATH
    )
    rows = []
    for investigation_id, records in (
        ("LT-0001", _lt0001_result_records(literature_dir, repo_root)[0]),
        ("LT-0002", _lt0002_records(literature_dir, repo_root, studies_path)),
    ):
        for record in records:
            row: dict = {"schema_version": literature_assessment.SCHEMA_VERSION}
            row["investigation_id"] = investigation_id
            row.update(record)
            rows.append(row)
    rows.sort(key=lambda row: (row["investigation_id"], row["paper_id"]))
    return rows


def private_key_map(*, repo_root=_REPO_ROOT, literature_dir=None):
    """The LT-0001 private source key -> canonical paper_id mapping."""

    repo_root = Path(repo_root).resolve()
    literature_dir = (
        Path(literature_dir) if literature_dir is not None else _LITERATURE_DIR
    )
    return _lt0001_result_records(literature_dir, repo_root)[1]


def check_equivalence(
    *,
    repo_root=_REPO_ROOT,
    literature_dir=None,
    studies_path=None,
    assessments_path=None,
):
    """Prove the canonical registry agrees with every legacy representation.

    Returns the legacy-derived rows on success and raises
    :class:`AssessmentEquivalenceError` naming each disagreement otherwise.
    """

    repo_root = Path(repo_root).resolve()
    literature_dir = (
        Path(literature_dir) if literature_dir is not None else _LITERATURE_DIR
    )
    studies_path = (
        Path(studies_path) if studies_path is not None else _STUDIES_PATH
    )
    assessments_path = (
        Path(assessments_path)
        if assessments_path is not None
        else literature_dir / "assessments.jsonl"
    )
    expected = expected_records(
        repo_root=repo_root,
        literature_dir=literature_dir,
        studies_path=studies_path,
    )
    registry = literature_assessment.load_assessments(
        assessments_path,
        repo_root=repo_root,
        literature_dir=literature_dir,
        studies_path=studies_path,
    )
    actual = {
        (record.investigation_id, record.paper_id): record.as_dict()
        for record in registry.records
    }
    expected_by_ref = {
        (row["investigation_id"], row["paper_id"]): row for row in expected
    }

    differences = []
    for ref in sorted(set(expected_by_ref) - set(actual)):
        differences.append("missing from registry: {}#{}".format(*ref))
    for ref in sorted(set(actual) - set(expected_by_ref)):
        differences.append("invented in registry: {}#{}".format(*ref))
    for ref in sorted(set(actual) & set(expected_by_ref)):
        row, record = expected_by_ref[ref], actual[ref]
        for field in (
            "role",
            "verdict",
            "gates",
            "reason_summary",
            "assessment_anchor",
            "assessed_at",
        ):
            if row[field] != record[field]:
                differences.append(
                    "{}#{} field {!r} differs: legacy={!r} registry={!r}".format(
                        ref[0], ref[1], field, row[field], record[field]
                    )
                )
    if differences:
        raise AssessmentEquivalenceError(
            "canonical assessments.jsonl disagrees with the legacy "
            "representations ({} difference(s))".format(len(differences)),
            differences=differences,
        )
    return expected


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="check", choices=("check",))
    return parser


def main(argv=None):
    _build_parser().parse_args(argv)
    try:
        rows = check_equivalence()
        print(
            "literature assessments: equivalent to legacy ({})".format(len(rows))
        )
        return 0
    except (AssessmentEquivalenceError, literature_assessment.AssessmentError) as exc:
        payload: dict = {"error": str(exc), "kind": "ASSESSMENT_EQUIVALENCE"}
        if isinstance(exc, AssessmentEquivalenceError):
            payload["differences"] = exc.differences
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
