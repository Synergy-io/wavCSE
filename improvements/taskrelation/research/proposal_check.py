#!/usr/bin/env python3
"""Deterministic validator for research-computer proposals.

A proposal is the research-computer's product between a designed study and the
human's decision. It is not a study and not an authorization, and this module
enforces that boundary without a model in the loop:

  * a strict frontmatter schema (unknown keys rejected, because a misspelled
    limit would silently not bind);
  * the proposal lifecycle vocabulary and its review precondition;
  * the event separation - while a proposal is pre-decision, its
    ``allocated_study_id`` must not appear in ``STUDIES.jsonl`` and must have no
    ``studies/<id>/`` directory, so "a proposal exists" can never silently become
    "a study is registered".

Legacy proposals with no ``proposal_schema`` frontmatter (the historical
``TR-0012`` / ``TR-0013`` pre-registration drafts) are reported as ``legacy`` and
are read-only to this validator: they are not rewritten and not failed.

CLI:
    python3 proposal_check.py check [--research-dir DIR]
    python3 proposal_check.py template
"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parent
PROPOSALS_DIR = RESEARCH_DIR / "proposals"
STUDIES_DIR = RESEARCH_DIR / "studies"
REGISTRY_PATH = RESEARCH_DIR / "STUDIES.jsonl"

SCHEMA_VERSION = "1"

STATUSES = (
    "DRAFT",
    "REVIEW_REQUIRED",
    "CHANGES_REQUESTED",
    "READY_FOR_HUMAN",
    "APPROVED",
    "REJECTED",
    "SUPERSEDED",
)
REVIEW_VALUES = ("", "PASS", "CHANGES_REQUIRED")
# A proposal in one of these states is pre-decision: it must not be registered.
PRE_DECISION_STATUSES = ("DRAFT", "REVIEW_REQUIRED", "CHANGES_REQUESTED", "READY_FOR_HUMAN")
REVIEW_REQUIRED_STATUSES = ("READY_FOR_HUMAN", "APPROVED")

PROPOSAL_KEYS = (
    "proposal_schema",
    "proposal_id",
    "allocated_study_id",
    "status",
    "created_at",
)
OPTIONAL_KEYS = (
    "review",
    "reviewed_by",
    "approved_by",
    "approved_at",
    "superseded_by",
)
ALLOWED_KEYS = PROPOSAL_KEYS + OPTIONAL_KEYS

REQUIRED_SECTIONS = (
    "Question",
    "Evidence basis",
    "Hypotheses",
    "Proposed study",
    "Discriminating measurements",
    "Success and stop conditions",
    "Expected compute",
    "Risks and confounds",
    "Repository effects",
    "Human decisions required",
    "Review",
)

STUDY_ID_PATTERN = re.compile(r"^[A-Z]{2}-\d{4}$")


def read_text(path):
    with open(str(path), "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def unquote(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_frontmatter(text):
    """Return (entries, problems).

    entries is an ordered list of (key, value); it is None when the file does not
    open with a '---' block. A duplicate key is a problem.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, []
    entries, problems, seen = [], [], set()
    closed = False
    for line in lines[1:]:
        if line.strip() == "---":
            closed = True
            break
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            problems.append("frontmatter line is not a 'key: value' pair: %s" % line.strip())
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in seen:
            problems.append("duplicate frontmatter key '%s'" % key)
            continue
        seen.add(key)
        entries.append((key, unquote(value)))
    if not closed:
        problems.append("frontmatter opened with '---' is never closed")
        return [], problems
    return entries, problems


def is_iso8601(value):
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        datetime.datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


def load_registered_ids(research_dir):
    registry = research_dir / "STUDIES.jsonl"
    ids = set()
    if not registry.is_file():
        return ids
    for line in read_text(registry).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("study_id"), str):
            ids.add(entry["study_id"])
    return ids


def check_text(text, research_dir, registered_ids):
    """Validate one proposal text. Returns (kind, problems).

    kind is "legacy" when the file has no schema frontmatter, "proposal"
    otherwise.
    """
    entries, problems = parse_frontmatter(text)
    if entries is None:
        return "legacy", []
    if problems:
        return "proposal", problems
    data = dict(entries)
    if data.get("proposal_schema") != SCHEMA_VERSION:
        return "legacy", []

    for key in data:
        if key not in ALLOWED_KEYS:
            problems.append("unknown frontmatter key '%s' (allowed: %s)" % (key, ", ".join(ALLOWED_KEYS)))
    for key in PROPOSAL_KEYS:
        if not data.get(key):
            problems.append("missing or empty required key '%s'" % key)

    status = data.get("status", "")
    if status and status not in STATUSES:
        problems.append("unknown status '%s' (allowed: %s)" % (status, ", ".join(STATUSES)))

    study_id = data.get("allocated_study_id", "")
    if study_id and not STUDY_ID_PATTERN.match(study_id):
        problems.append("allocated_study_id '%s' is not of the form XX-0000" % study_id)

    created_at = data.get("created_at", "")
    if created_at and not is_iso8601(created_at):
        problems.append("created_at '%s' is not ISO 8601" % created_at)

    review = data.get("review", "")
    if review not in REVIEW_VALUES:
        problems.append("review '%s' is not one of %s" % (review, REVIEW_VALUES))
    if status in REVIEW_REQUIRED_STATUSES and review != "PASS":
        problems.append("status %s requires review: PASS (found '%s')" % (status, review))

    if status == "APPROVED":
        if not data.get("approved_by"):
            problems.append("APPROVED requires a non-empty 'approved_by' (the human's act)")
        approved_at = data.get("approved_at", "")
        if not approved_at:
            problems.append("APPROVED requires a non-empty 'approved_at'")
        elif not is_iso8601(approved_at):
            problems.append("approved_at '%s' is not ISO 8601" % approved_at)

    if status in PRE_DECISION_STATUSES and study_id:
        if study_id in registered_ids:
            problems.append(
                "pre-decision proposal allocates '%s' but it is already registered in STUDIES.jsonl "
                "(a proposal exists is not a study is registered)" % study_id
            )
        if (research_dir / "studies" / study_id).exists():
            problems.append(
                "pre-decision proposal allocates '%s' but studies/%s/ already exists" % (study_id, study_id)
            )

    headings = set()
    for line in text.splitlines():
        if line.startswith("## "):
            headings.add(line[3:].strip())
    for section in REQUIRED_SECTIONS:
        if section not in headings:
            problems.append("missing required '## %s' section" % section)

    return "proposal", problems


def iter_proposals(proposals_dir):
    if not proposals_dir.is_dir():
        return []
    return sorted(
        (path for path in proposals_dir.glob("*.md") if path.name != "README.md"),
        key=lambda path: path.name,
    )


def cmd_check(research_dir):
    proposals_dir = research_dir / "proposals"
    registered_ids = load_registered_ids(research_dir)
    exit_code = 0
    for path in iter_proposals(proposals_dir):
        display = "proposals/%s" % path.name
        kind, problems = check_text(read_text(path), research_dir, registered_ids)
        if kind == "legacy":
            print("LEGACY %s: pre-schema proposal, not validated" % display)
            continue
        if problems:
            exit_code = 1
            for problem in problems:
                print("%s: %s" % (display, problem))
        else:
            print("OK     %s" % display)
    return exit_code


TEMPLATE = """---
proposal_schema: 1
proposal_id: DP-0000
allocated_study_id: XX-0000
status: DRAFT
created_at: 2000-01-01T00:00:00+00:00
review: ""
reviewed_by: ""
approved_by: ""
approved_at: ""
superseded_by: ""
---

# XX-0000 - <title>

**Written:** <date>, against canonical `<commit>`.
**Predecessor:** <study / finding / decision>.

## Question

One falsifiable question.

## Evidence basis

The F-ids, decision ids, study paths and `paper_id#claim_id` references this acts on.

## Hypotheses

H1 (primary), the competing explanation, and the pre-declared third outcome.

## Proposed study

Regime, independent variable, matched controls, and the staged plan (screening
then confirmation, as separate pre-registered stages).

## Discriminating measurements

What separates the hypotheses, and what explicitly does not.

## Success and stop conditions

Numeric, pre-registered, per stage.

## Expected compute

Per stage: runs, GPU-hours, concurrency, and the basis for the estimate.

## Risks and confounds

What the design cannot hold constant; the residual assumption.

## Repository effects

The files and research state a registered study would touch.

## Human decisions required

Numbered decisions the human must make.

## Review

Reviewer verdict and findings; revision history.
"""


def cmd_template():
    sys.stdout.write(TEMPLATE)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="proposal_check.py", description=(__doc__ or "").splitlines()[0])
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    check_parser = subparsers.add_parser("check", help="validate every proposal under proposals/")
    check_parser.add_argument("--research-dir", default=str(RESEARCH_DIR),
                              help="research directory holding proposals/, STUDIES.jsonl and studies/")
    subparsers.add_parser("template", help="print a conforming proposal skeleton")
    args = parser.parse_args(argv)
    if args.command == "template":
        return cmd_template()
    return cmd_check(Path(args.research_dir))


if __name__ == "__main__":
    sys.exit(main())
