#!/usr/bin/env python3
"""Deterministic candidate-change gate.

Main OMP runs this on a writing specialist's candidate — an OMP isolation patch
artifact, or a disposable workspace — *before* that candidate is integrated into
the canonical checkout. It answers exactly one question: may this change set
reach the canonical tree?

The gate is not a sandbox and not a security framework. It is the deterministic
boundary between a candidate and integration:

    candidate (patch artifact or workspace)
              |
              v
    candidate_gate.check      <- this file, in the trusted baseline
              |
         +----+----+
      accept     reject (deterministic reason codes)

Trusted-baseline policy. The authority for what is protected is the constants in
THIS file; the gate never loads a rule from the candidate it is judging. This
file is itself on the validation-authority list, so a candidate that edits the
gate is rejected rather than heard. The same holds for the checks the candidate
hopes to satisfy, the test discovery that finds them, and the agent definitions
that define the writing role.

What a rejected candidate must never be able to do:

    * make itself acceptable by editing the gate, its policy, the validation
      command, test discovery, an agent definition, or a tool definition;
    * make broken implementation pass by changing the judge — editing, deleting,
      or renaming an existing test, or weakening its assertion;
    * reach outside the write scope its role was granted.

Reasons (deterministic, exhaustive):

    CONTROL_PLANE_MODIFIED        changed a rule, tool, agent or policy that
                                  constrains agents
    VALIDATION_AUTHORITY_MODIFIED changed a gate, its configuration, or its
                                  discovery
    PROTECTED_PATH_MODIFIED       changed a frozen baseline, a lifecycle record,
                                  or the paid-execution enforcement seam
    EXISTING_TEST_MODIFIED        changed or weakened an existing test
    EXISTING_TEST_DELETED         deleted an existing test
    EXISTING_TEST_RENAMED         renamed an existing test (which also hides it)
    WRITE_SCOPE_VIOLATION         changed a path outside the role's scope
    ACCEPT                        no reason found

Existing tests are baseline-owned evidence. Adding a NEW test is allowed within
the role's scope; changing an existing one is not an ordinary implementation
change and must be escalated as separate TEST/CONTRACT MAINTENANCE.

Usage:

    candidate_gate.py check --repo DIR --baseline REV
                            (--patch FILE | --workspace DIR | --rev REV)
                            [--role ROLE] [--json]
    candidate_gate.py selftest [--json]

Standard library only, Python 3.9+, no network, no writes.
"""

import argparse
import json
import os
import re
import subprocess
import sys

# --------------------------------------------------------------------------
# Trusted-baseline policy. Every path below is resolved against the actual
# project-local layout; none of it is supplied by the candidate.
# --------------------------------------------------------------------------

RESEARCH = "improvements/taskrelation/research/"

# 1. Control plane: the rules, agents, tools and policies that constrain agents.
#    A writing agent must not rewrite its own authority.
CONTROL_PLANE_PREFIXES = (".omp/", ".agents/")
CONTROL_PLANE_FILES = ("AGENTS.md",)

# 2. Validation authority: what judges the change, and how it is discovered.
VALIDATION_PREFIXES = ("scripts/", ".github/")
VALIDATION_FILES = ("Makefile", "pyproject.toml", "uv.lock", "pytest.ini",
                    "setup.cfg", "tox.ini", "mypy.ini", "ruff.toml", ".flake8")

# 3. Frozen baselines, lifecycle records and the paid-execution enforcement seam.
PROTECTED_PREFIXES = (
    "downstream/",
    "upstream/",
    "improvements/compute/",
    RESEARCH + "proposals/",
    RESEARCH + "authorizations/",
    RESEARCH + "execution/",
    RESEARCH + "audits/",
    RESEARCH + "literature/",
    RESEARCH + "literature_discovery/",
    RESEARCH + "literature_survey/",
    RESEARCH + "task_relations/",
)
# `improvements/compute/tests/` is NOT protected here: it holds tests, and the
# test rules below already forbid changing the existing ones while permitting a
# new, feature-specific one.
PROTECTED_EXEMPT_PREFIXES = ("improvements/compute/tests/",)
PROTECTED_FILES = (
    ".dvc/",
    "embedding.tar.gz.dvc",
    RESEARCH + "OBJECTIVE.md",
    RESEARCH + "STATE.md",
    RESEARCH + "STUDIES.jsonl",
    RESEARCH + "FINDINGS.md",
    RESEARCH + "FAILURES.md",
    RESEARCH + "DECISIONS.md",
    RESEARCH + "BACKLOG.md",
    RESEARCH + "FRAMEWORK.md",
    RESEARCH + "VARIANT_BENCHMARK_PROTOCOL.md",
    RESEARCH + "WORKER_ENVIRONMENT.md",
    RESEARCH + "LITERATURE.md",
    RESEARCH + "proposal_check.py",
    RESEARCH + "execution_contract.py",
    RESEARCH + "convergence.py",
    RESEARCH + "literature_acquire.py",
    RESEARCH + "literature_admit.py",
    RESEARCH + "literature_assessment.py",
    RESEARCH + "literature_catalog.py",
    RESEARCH + "literature_claims.py",
    RESEARCH + "literature_ingest.py",
    RESEARCH + "literature_investigation.py",
    RESEARCH + "literature_io.py",
    RESEARCH + "literature_primary.py",
    RESEARCH + "literature_primary_text.py",
    RESEARCH + "literature_query.py",
    RESEARCH + "literature_read.py",
    RESEARCH + "literature_record.py",
    RESEARCH + "literature_synthesis.py",
)
# Lifecycle records that live beside a Study's implementation, which IS writable.
PROTECTED_BASENAMES = (
    (RESEARCH + "studies/", "PLAN.md"),
    (RESEARCH + "studies/", "NOTE.md"),
)

# Narrow write scopes. A role may only change paths under its own prefixes; the
# cross-repository maintainer never silently edits the research repository and
# vice versa. A change to `improvements/compute/` is a separately invoked
# trusted maintenance action, not an ordinary candidate.
ROLE_SCOPES = {
    "research-executor": ("improvements/",),
    "infrastructure-engineer": ("infra/",),
}

TEST_DIR_NAMES = ("test", "tests")


def is_test_path(path):
    """Whether a repository-relative path is a test the candidate must not edit."""

    parts = path.split("/")
    if any(part in TEST_DIR_NAMES for part in parts[:-1]):
        return True
    name = parts[-1]
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def classify(path, *, baseline_paths, role):
    """The single reason a changed path is rejected, or None to accept it.

    Order matters: control plane before validation before frozen baselines, then
    test integrity, then the role's write scope. The first rule that matches wins.
    """

    if _under(path, CONTROL_PLANE_PREFIXES) or path in CONTROL_PLANE_FILES:
        return "CONTROL_PLANE_MODIFIED"
    if _under(path, VALIDATION_PREFIXES) or path in VALIDATION_FILES:
        return "VALIDATION_AUTHORITY_MODIFIED"
    if _under(path, PROTECTED_PREFIXES) and not _under(path, PROTECTED_EXEMPT_PREFIXES):
        return "PROTECTED_PATH_MODIFIED"
    if path in PROTECTED_FILES or _under(path, PROTECTED_FILES):
        return "PROTECTED_PATH_MODIFIED"
    for prefix, basename in PROTECTED_BASENAMES:
        if path.startswith(prefix) and path.endswith("/" + basename):
            return "PROTECTED_PATH_MODIFIED"
    if is_test_path(path) and path in baseline_paths:
        return "EXISTING_TEST_MODIFIED"
    if role is not None:
        scope = ROLE_SCOPES.get(role)
        if scope is None:
            raise GateError("unknown role {!r}; roles are {}".format(
                role, ", ".join(sorted(ROLE_SCOPES))))
        if not _under(path, scope):
            return "WRITE_SCOPE_VIOLATION"
    return None


def _under(path, prefixes):
    return any(path == prefix.rstrip("/") or path.startswith(prefix)
               for prefix in prefixes)


class GateError(Exception):
    pass


# --------------------------------------------------------------------------
# Change extraction
# --------------------------------------------------------------------------

_DIFF_GIT = re.compile(r"^diff --git (?P<a>\S+) (?P<b>\S+)$")
_RENAME_FROM = "rename from "
_RENAME_TO = "rename to "
_COPY_FROM = "copy from "
_COPY_TO = "copy to "


def _strip_prefix(token):
    if token.startswith('"') and token.endswith('"') and len(token) >= 2:
        token = token[1:-1]
        token = token.replace("\\\"", "\"").replace("\\\\", "\\")
    for prefix in ("a/", "b/"):
        if token.startswith(prefix):
            return token[len(prefix):]
    return token


def parse_patch(text):
    """Every file change a unified git patch describes, in file order.

    Returns a list of ``{"path", "old_path", "kind"}`` with kind in
    ``added``/``deleted``/``modified``/``renamed``/``copied``.
    """

    changes = []
    current = None

    def flush():
        if current is not None:
            changes.append(current)

    for line in text.splitlines():
        match = _DIFF_GIT.match(line)
        if match:
            flush()
            current = {"path": _strip_prefix(match.group("b")),
                       "old_path": _strip_prefix(match.group("a")),
                       "kind": "modified"}
            continue
        if current is None:
            continue
        stripped = line.rstrip()
        if stripped.startswith("new file mode"):
            current["kind"] = "added"
        elif stripped.startswith("deleted file mode"):
            current["kind"] = "deleted"
        elif stripped.startswith(_RENAME_FROM):
            current["old_path"] = _strip_prefix(stripped[len(_RENAME_FROM):].strip())
            current["kind"] = "renamed"
        elif stripped.startswith(_RENAME_TO):
            current["path"] = _strip_prefix(stripped[len(_RENAME_TO):].strip())
        elif stripped.startswith(_COPY_FROM):
            current["old_path"] = _strip_prefix(stripped[len(_COPY_FROM):].strip())
            current["kind"] = "copied"
        elif stripped.startswith(_COPY_TO):
            current["path"] = _strip_prefix(stripped[len(_COPY_TO):].strip())
    flush()

    for change in changes:
        if change["kind"] == "modified" and change["old_path"] != change["path"]:
            # `diff --git a/X b/Y` with no rename header: a hybrid; treat as a
            # rename so the old path's protection still applies.
            change["kind"] = "renamed"
    return changes


def _parse_name_status(text):
    changes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") or status.startswith("C"):
            old, new = parts[1], parts[2]
            kind = "renamed" if status.startswith("R") else "copied"
            changes.append({"path": new, "old_path": old, "kind": kind})
        elif status == "A":
            changes.append({"path": parts[1], "old_path": None, "kind": "added"})
        elif status == "D":
            changes.append({"path": parts[1], "old_path": None, "kind": "deleted"})
        else:
            changes.append({"path": parts[1], "old_path": None, "kind": "modified"})
    return changes


def changes_from_rev(repo, baseline, rev):
    """Every change a committed candidate revision carries against its baseline.

    OMP's isolated patch mode commits the candidate to its own branch
    (``omp/task/<id>``), which is the artifact the gate normally judges.
    """

    return _parse_name_status(
        _git(repo, "diff", "--name-status", "-M", "--no-color", baseline, rev))


def changes_from_workspace(workspace, baseline):
    """Every change a working tree carries against its baseline commit.

    An unstaged rename in the working tree appears as a deletion plus an
    untracked addition (``git diff`` does not pair a tracked deletion with an
    untracked file), so it is rejected as a deleted test plus a new file rather
    than reported as a rename. The committed-revision mode sees real renames.
    """

    changes = _parse_name_status(
        _git(workspace, "diff", "--name-status", "-M", "--no-color", baseline))
    untracked = _git(workspace, "ls-files", "--others", "--exclude-standard")
    seen = {change["path"] for change in changes}
    for line in untracked.splitlines():
        path = line.strip()
        if path and path not in seen:
            changes.append({"path": path, "old_path": None, "kind": "added"})
            seen.add(path)
    return changes


def _git(cwd, *args):
    result = subprocess.run(
        ["git", "-C", cwd] + list(args),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        universal_newlines=True,
    )
    if result.returncode != 0:
        raise GateError("git {} failed in {}: {}".format(
            " ".join(args), cwd, result.stderr.strip()))
    return result.stdout


def baseline_paths(repo, baseline):
    """The tracked paths of the baseline commit, as a set."""

    out = _git(repo, "ls-tree", "-r", "--name-only", baseline)
    return {line.strip() for line in out.splitlines() if line.strip()}


def resolve_rev(repo, rev):
    return _git(repo, "rev-parse", "--verify", "{}^{{commit}}".format(rev)).strip()


# --------------------------------------------------------------------------
# Checking
# --------------------------------------------------------------------------

def evaluate(changes, *, baseline_paths, role):
    """The reasons a change set is rejected, sorted and deduplicated.

    A rename/deletion of an existing test is reported as its specific reason
    (EXISTING_TEST_DELETED / EXISTING_TEST_RENAMED) rather than the generic
    EXISTING_TEST_MODIFIED, because the reason code is also the escalation cue.
    """

    reasons = []
    for change in changes:
        path = change["path"]
        old_path = change.get("old_path")
        if change["kind"] == "deleted" and is_test_path(path) and path in baseline_paths:
            reasons.append(("EXISTING_TEST_DELETED", path))
            continue
        if change["kind"] == "renamed" and old_path and is_test_path(old_path) \
                and old_path in baseline_paths:
            reasons.append(("EXISTING_TEST_RENAMED", old_path))
            continue
        reason = classify(path, baseline_paths=baseline_paths, role=role)
        if reason is not None:
            reasons.append((reason, path))
        if old_path and old_path != path:
            old_reason = classify(old_path, baseline_paths=baseline_paths, role=role)
            if old_reason is not None:
                reasons.append((old_reason, old_path))
    return sorted(set(reasons))


def check(*, repo, baseline, patch=None, workspace=None, rev=None, role=None):
    """Return ``(verdict, reasons, changes, resolved_baseline)``."""

    sources = [source for source in (patch, workspace, rev) if source is not None]
    if len(sources) != 1:
        raise GateError("give exactly one of --patch, --workspace or --rev")
    resolved = resolve_rev(repo, baseline)
    if workspace is not None:
        if not os.path.isdir(workspace):
            raise GateError("workspace {} is not a directory".format(workspace))
        changes = changes_from_workspace(workspace, resolved)
    elif rev is not None:
        changes = changes_from_rev(repo, resolved, resolve_rev(repo, rev))
    else:
        assert patch is not None  # the exactly-one-of guard above guarantees it
        with open(patch, "r", encoding="utf-8", errors="replace") as handle:
            changes = parse_patch(handle.read())
    known = baseline_paths(repo, resolved)
    reasons = evaluate(changes, baseline_paths=known, role=role)
    return ("REJECT" if reasons else "ACCEPT"), reasons, changes, resolved


# --------------------------------------------------------------------------
# Self-test: the deterministic adversarial cases, runnable without a repo.
# --------------------------------------------------------------------------

_DIFF = "diff --git a/{a} b/{b}\nindex 1111111..2222222 100644\n--- a/{a}\n+++ b/{b}\n@@ -1,3 +1,3 @@\n context\n-old\n+new\n context\n"
_NEW = "diff --git a/{a} b/{a}\nnew file mode 100644\nindex 0000000..1111111\n--- /dev/null\n+++ b/{a}\n@@ -0,0 +1 @@\n+line\n"
_DELETED = "diff --git a/{a} b/{a}\ndeleted file mode 100644\nindex 1111111..0000000\n--- a/{a}\n+++ /dev/null\n@@ -1 +0,0 @@\n-line\n"
_RENAMED = "diff --git a/{a} b/{b}\nsimilarity index 100%\nrename from {a}\nrename to {b}\n"


def _patch(kind, path, new_path=None):
    if kind == "added":
        return _NEW.format(a=path)
    if kind == "deleted":
        return _DELETED.format(a=path)
    if kind == "renamed":
        return _RENAMED.format(a=path, b=new_path)
    return _DIFF.format(a=path, b=path)


# A synthetic baseline containing the paths the "existing" cases refer to.
SELFTEST_BASELINE = {
    "improvements/compute/tests/test_jobspec.py",
    RESEARCH + "tests/test_proposal_check.py",
    RESEARCH + "tests/test_execution_contract.py",
    "improvements/taskrelation/01-mtrl/mtrl_model.py",
    "improvements/compute/jobspec.py",
    "AGENTS.md",
    "Makefile",
    ".omp/agents/research-executor.md",
    ".agents/skills/wavcse-research-computer/SKILL.md",
    "scripts/agents/agent_assets.py",
    RESEARCH + "proposal_check.py",
    RESEARCH + "STUDIES.jsonl",
    "downstream/model/downstream_model.py",
}

# (name, patch, role, expected reason set). Expected sets are exact.
SELFTEST_CASES = (
    ("legitimate implementation change",
     _patch("modified", "improvements/taskrelation/01-mtrl/mtrl_model.py"),
     "research-executor", set()),
    ("legitimate new feature test",
     _patch("added", RESEARCH + "tests/test_new_feature.py"),
     "research-executor", set()),
    ("agent definition modified",
     _patch("modified", ".omp/agents/research-executor.md"),
     "research-executor", {"CONTROL_PLANE_MODIFIED"}),
    ("own skill modified",
     _patch("modified", ".agents/skills/wavcse-research-computer/SKILL.md"),
     "research-executor", {"CONTROL_PLANE_MODIFIED"}),
    ("capability policy modified",
     _patch("modified", ".omp/config.yml"),
     "research-executor", {"CONTROL_PLANE_MODIFIED"}),
    ("gate policy modified",
     _patch("modified", "scripts/agents/candidate_gate.py"),
     "research-executor", {"VALIDATION_AUTHORITY_MODIFIED"}),
    ("gate implementation modified",
     _patch("modified", "scripts/agents/agent_assets.py"),
     "research-executor", {"VALIDATION_AUTHORITY_MODIFIED"}),
    ("existing test modified (assertion weakened)",
     _patch("modified", "improvements/compute/tests/test_jobspec.py"),
     "research-executor", {"EXISTING_TEST_MODIFIED"}),
    ("existing test deleted",
     _patch("deleted", RESEARCH + "tests/test_proposal_check.py"),
     "research-executor", {"EXISTING_TEST_DELETED"}),
    ("existing test renamed to hide it",
     _patch("renamed", RESEARCH + "tests/test_execution_contract.py",
            RESEARCH + "tests/test_execution_contract_v2.py"),
     "research-executor", {"EXISTING_TEST_RENAMED"}),
    ("existing test gains skip",
     _patch("modified", RESEARCH + "tests/test_execution_contract.py"),
     "research-executor", {"EXISTING_TEST_MODIFIED"}),
    ("Makefile changed",
     _patch("modified", "Makefile"),
     "research-executor", {"VALIDATION_AUTHORITY_MODIFIED"}),
    ("ci workflow changed",
     _patch("modified", ".github/workflows/infra-ci.yml"),
     "research-executor", {"VALIDATION_AUTHORITY_MODIFIED"}),
    ("proposal checker changed",
     _patch("modified", RESEARCH + "proposal_check.py"),
     "research-executor", {"PROTECTED_PATH_MODIFIED"}),
    ("study registry changed",
     _patch("modified", RESEARCH + "STUDIES.jsonl"),
     "research-executor", {"PROTECTED_PATH_MODIFIED"}),
    ("frozen downstream baseline changed",
     _patch("modified", "downstream/model/downstream_model.py"),
     "research-executor", {"PROTECTED_PATH_MODIFIED"}),
    ("paid-execution seam changed",
     _patch("modified", "improvements/compute/jobspec.py"),
     "research-executor", {"PROTECTED_PATH_MODIFIED"}),
    ("project instructions changed",
     _patch("modified", "AGENTS.md"),
     "research-executor", {"CONTROL_PLANE_MODIFIED"}),
    ("write outside the role scope",
     _patch("modified", "infra/src/wavcse_infra/cli.py"),
     "research-executor", {"WRITE_SCOPE_VIOLATION"}),
    ("infra engineer inside its own scope",
     _patch("modified", "infra/src/wavcse_infra/cli.py"),
     "infrastructure-engineer", set()),
)


def selftest():
    """Run the adversarial cases; return a list of (name, ok, detail)."""

    results = []
    for name, patch_text, role, expected in SELFTEST_CASES:
        changes = parse_patch(patch_text)
        reasons = evaluate(changes, baseline_paths=SELFTEST_BASELINE, role=role)
        found = {reason for reason, _ in reasons}
        results.append((name, found == expected,
                        "expected {} got {}".format(sorted(expected), sorted(found))))
    # A candidate that both legitimately implements and illegitimately edits its
    # judge must be rejected for the judge edit, not accepted for the rest.
    mixed = parse_patch(_patch("modified", RESEARCH + "tests/test_proposal_check.py")
                        + _patch("modified", RESEARCH + "tests/test_new_thing.py"))
    found = {reason for reason, _ in evaluate(
        mixed, baseline_paths=SELFTEST_BASELINE, role="research-executor")}
    results.append(("mixed legitimate+illegitimate is rejected",
                    found == {"EXISTING_TEST_MODIFIED"},
                    "got {}".format(sorted(found))))
    return results


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_check(args):
    verdict, reasons, changes, resolved = check(
        repo=args.repo, baseline=args.baseline, patch=args.patch,
        workspace=args.workspace, rev=args.rev, role=args.role,
    )
    if args.json:
        print(json.dumps({
            "verdict": verdict,
            "baseline": resolved,
            "changes": changes,
            "reasons": [{"reason": reason, "path": path} for reason, path in reasons],
        }, indent=2, sort_keys=True))
    else:
        for reason, path in reasons:
            print("{} {}".format(reason, path))
        print(verdict)
    return 0 if verdict == "ACCEPT" else 1


def cmd_selftest(args):
    results = selftest()
    failures = [entry for entry in results if not entry[1]]
    if args.json:
        print(json.dumps({
            "cases": [{"name": name, "ok": ok, "detail": detail}
                      for name, ok, detail in results],
            "passed": len(results) - len(failures),
            "total": len(results),
        }, indent=2, sort_keys=True))
    else:
        for name, ok, detail in results:
            print("{} {}".format("PASS" if ok else "FAIL", name))
            if not ok:
                print("     {}".format(detail))
        print("{} of {} adversarial cases passed".format(
            len(results) - len(failures), len(results)))
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="candidate_gate.py",
        description="Deterministic gate between a candidate change set and the canonical checkout.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    check_parser = subparsers.add_parser("check", help="judge one candidate")
    check_parser.add_argument("--repo", required=True,
                              help="canonical checkout that holds the trusted baseline")
    check_parser.add_argument("--baseline", required=True,
                              help="immutable baseline revision the candidate was made against")
    source = check_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--patch", help="path to a unified git patch artifact")
    source.add_argument("--workspace", help="path to a candidate working tree")
    source.add_argument("--rev", help="a committed candidate revision (e.g. omp/task/<id>)")
    check_parser.add_argument("--role", default=None,
                              help="writing role whose write scope applies")
    check_parser.add_argument("--json", action="store_true")

    selftest_parser = subparsers.add_parser(
        "selftest", help="run the deterministic adversarial cases (no repository needed)")
    selftest_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            return cmd_check(args)
        return cmd_selftest(args)
    except GateError as exc:
        print("candidate_gate: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
