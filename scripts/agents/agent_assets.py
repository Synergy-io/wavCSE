#!/usr/bin/env python3
"""Deterministic sync/check tooling for the agent knowledge system.

Canonical layout (identical in every repository that vendors this tool):
  AGENTS.md                        repository-root agent instructions
  .agents/skills/<name>/SKILL.md   canonical skills
  .agents/commands/<name>.md       canonical commands
  .omp/AGENTS.md                   relative symlink -> ../AGENTS.md

Usage: python3 scripts/agents/agent_assets.py (sync [--force] | check)
Standard library only, Python 3.9+, no network, no git.
"""
import argparse
import os
import re
import sys
from pathlib import Path

OMP_DIR = ".omp"
OMP_AGENTS = ".omp/AGENTS.md"
OMP_AGENTS_TARGET = "../AGENTS.md"
OMP_AGENT_DIRS = (".omp/agents", ".omp/tools")
ROOT_AGENTS = "AGENTS.md"
SKILLS_DIR = ".agents/skills"
COMMANDS_DIR = ".agents/commands"
SCRIPTS_DIR = "scripts/agents"
MAKEFILE = "Makefile"
SKILL_KEYS = ("name", "description")
MIN_SKILL_LINES = 20
MAX_DESCRIPTION_CHARS = 200

BOUNDARY_FLAGS = ("read-only", "no-commit", "no-paid-compute", "writes-reports",
                  "mutates-research-state", "may-provision-compute", "may-commit")
# Must not exist: every agent host reads .agents/** and .omp/AGENTS.md instead.
RETIRED_PATHS = (".omp/skills", ".omp/commands", ".claude/commands",
                 ".codex/commands", ".claude/skills", ".codex/skills")
# Present-but-unread locations: reported as INERT, never as a failure.
INERT_PATHS = (".claude/rules", "CLAUDE.md")
# Paths scanned for forbidden content; the Makefile is deliberately in both sets,
# and the native agent/tool roots are scanned so project agent assets get the
# same hygiene check as skills and commands.
CONTENT_TARGETS = (".agents", ROOT_AGENTS, "CLAUDE.md", MAKEFILE) + OMP_AGENT_DIRS
TOOLING_TARGETS = (SCRIPTS_DIR, MAKEFILE)

# Every needle is assembled from fragments on purpose: this file lives under
# scripts/agents/, one of the scanned paths, so it must not contain what it forbids.
HOME_SIGN = "~" + "/"
STATE_HOME = ".local" + "/state"
FORBIDDEN_LITERALS = (
    "../" + "wavCSE",
    "../" + "wavcse-infra",
    "$HOME" + "/projects",
    HOME_SIGN + "projects",
    "-----" + "BEGIN",
    "X-" + "Amz-",
    "aws_" + "secret",
    "ssh." + "runpod.io",
    "api." + "runpod.io",
)
ABSOLUTE_HOME_PATTERN = re.compile(r"/home/[A-Za-z0-9._-]+")
ACCESS_KEY_PATTERN = re.compile(r"AKIA[0-9A-Z]{16}")
IPV4_PATTERN = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
# A home path is machine-specific, with exactly one documented exemption: the
# controller state location, which documentation refers to by name, not by a real path.
STATE_HOME_EXEMPTION = re.escape(STATE_HOME) + r"(?![A-Za-z0-9._-])"
TILDE_PATH_PATTERN = re.compile(HOME_SIGN + "(?!" + STATE_HOME_EXEMPTION + ")")


def read_text(path):
    with open(str(path), "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def rel(root, path):
    try:
        return os.path.relpath(str(path), str(root)).replace(os.sep, "/")
    except ValueError:
        return str(path)


def iter_files(root, target):
    """Every file under target (recursive, sorted); target may itself be a file."""
    path = Path(target)
    if not os.path.lexists(str(path)):
        return []
    if not path.is_dir() or path.is_symlink():
        return [path]
    found = []
    for dirpath, dirnames, filenames in os.walk(str(path)):
        dirnames[:] = sorted(name for name in dirnames if name != "__pycache__")
        for name in sorted(filenames):
            found.append(Path(dirpath) / name)
    return sorted(found, key=lambda p: rel(root, p))


class Check(object):
    """One numbered check: an expected summary line plus any violations."""

    def __init__(self, number, title):
        self.number, self.title, self.detail, self.violations = number, title, "no items", []

    def add(self, path, line, reason):
        self.violations.append((str(path), int(line), reason))

    def summary(self):
        return "check %d %s: OK (%s)" % (self.number, self.title, self.detail)


def parse_document(lines):
    """Parse a leading '---' block into (entries, body start index).

    entries is a list of (line number, key, value); a line without a colon gets an
    empty key. Returns None when the file does not open with a '---' block, and a
    body start of None when that block is never closed.
    """
    if not lines or lines[0].strip() != "---":
        return None
    entries = []
    for index in range(1, len(lines)):
        line = lines[index]
        if line.strip() == "---":
            return entries, index + 1
        if ":" in line:
            key, value = line.split(":", 1)
            entries.append((index + 1, key.strip(), value.strip()))
        else:
            entries.append((index + 1, "", line.strip()))
    return entries, None


def forbidden_reasons(line):
    """Every reason this line breaks the forbidden-content rule (possibly none)."""
    reasons = []
    for literal in FORBIDDEN_LITERALS:
        if literal in line:
            reasons.append("forbidden content: %s" % literal)
    match = ABSOLUTE_HOME_PATTERN.search(line)
    if match:
        reasons.append("forbidden content: absolute home directory path %s" % match.group(0))
    match = TILDE_PATH_PATTERN.search(line)
    if match:
        reasons.append("forbidden content: home path %s (only the documented state location is exempt)" % match.group(0))
    if ACCESS_KEY_PATTERN.search(line):
        reasons.append("forbidden content: AKIA access key id")
    if IPV4_PATTERN.search(line):
        reasons.append("forbidden content: bare IPv4 literal")
    return reasons


def scan_forbidden(root, targets, check):
    """Scan every file reachable from the given paths; returns the number scanned."""
    files = []
    for target in targets:
        files.extend(iter_files(root, root / target))
    files = sorted(set(files), key=lambda p: rel(root, p))
    for path in files:
        display = rel(root, path)
        for lineno, line in enumerate(read_text(path).splitlines(), start=1):
            for reason in forbidden_reasons(line):
                check.add(display, lineno, reason)
    return len(files)


def remove_path(path):
    """Remove a file, symlink, or directory tree (stdlib only)."""
    if os.path.islink(str(path)) or os.path.isfile(str(path)):
        os.unlink(str(path))
        return
    for dirpath, dirnames, filenames in os.walk(str(path), topdown=False):
        for name in sorted(dirnames) + sorted(filenames):
            child = os.path.join(dirpath, name)
            if os.path.islink(child) or os.path.isfile(child):
                os.unlink(child)
            elif os.path.isdir(child):
                os.rmdir(child)
    os.rmdir(str(path))


def cmd_sync(root, force):
    """Materialize tool-specific state: ensure .omp/AGENTS.md is ../AGENTS.md."""
    actions, changed = [], False
    omp_dir = root / OMP_DIR
    if os.path.lexists(str(omp_dir)) and not omp_dir.is_dir():
        print("ERROR: %s exists and is not a directory; remove it and re-run." % OMP_DIR)
        return 1
    if omp_dir.is_dir():
        actions.append("directory %s/ already exists" % OMP_DIR)
    else:
        omp_dir.mkdir(parents=True)
        changed = True
        actions.append("created directory %s/" % OMP_DIR)
    link, existing = root / OMP_AGENTS, None
    if os.path.islink(str(link)):
        existing = "symlink -> %s" % os.readlink(str(link))
    elif os.path.isdir(str(link)):
        existing = "directory"
    elif os.path.lexists(str(link)):
        existing = "regular file"
    if existing == "symlink -> %s" % OMP_AGENTS_TARGET:
        actions.append("%s already correct: relative symlink -> %s" % (OMP_AGENTS, OMP_AGENTS_TARGET))
    elif existing is not None and not force:
        print("REFUSED: %s exists as a %s, not the required relative symlink -> %s." % (OMP_AGENTS, existing, OMP_AGENTS_TARGET))
        print("Refusing to overwrite it (no --force given).")
        print("Fix: remove %s and re-run, or re-run with --force to replace it." % OMP_AGENTS)
        return 1
    else:
        if existing is not None:
            remove_path(link)
            actions.append("replaced %s (%s) with relative symlink -> %s" % (OMP_AGENTS, existing, OMP_AGENTS_TARGET))
        else:
            actions.append("created relative symlink %s -> %s" % (OMP_AGENTS, OMP_AGENTS_TARGET))
        os.symlink(OMP_AGENTS_TARGET, str(link))
        changed = True
    for action in actions:
        print(action)
    print("sync: %s" % ("applied changes" if changed else "already up to date"))
    return 0


def check_omp_symlink(root, check):
    link = root / OMP_AGENTS
    if not os.path.lexists(str(link)):
        check.add(OMP_AGENTS, 0, "missing: expected a relative symlink -> %s" % OMP_AGENTS_TARGET)
    elif not os.path.islink(str(link)):
        kind = "directory" if os.path.isdir(str(link)) else "regular file"
        check.add(OMP_AGENTS, 0, "not a symlink (found a %s); run 'sync --force' to replace it with a symlink -> %s" % (kind, OMP_AGENTS_TARGET))
    elif os.readlink(str(link)) != OMP_AGENTS_TARGET:
        check.add(OMP_AGENTS, 0, "wrong symlink target %s; expected exactly %s" % (os.readlink(str(link)), OMP_AGENTS_TARGET))
    elif os.path.realpath(str(link)) != os.path.realpath(str(root / ROOT_AGENTS)):
        check.add(OMP_AGENTS, 0, "symlink does not resolve to the repository-root %s" % ROOT_AGENTS)
    else:
        check.detail = "relative symlink -> %s" % OMP_AGENTS_TARGET


def check_skills(root, check, unique_check):
    skills_root = root / SKILLS_DIR
    dirs = sorted((p for p in skills_root.iterdir() if p.is_dir()), key=lambda p: p.name) if skills_root.is_dir() else []
    declared = []
    for skill_dir in dirs:
        path = "%s/%s/SKILL.md" % (SKILLS_DIR, skill_dir.name)
        if not (skill_dir / "SKILL.md").is_file():
            check.add(path, 0, "missing SKILL.md")
            continue
        lines = read_text(skill_dir / "SKILL.md").splitlines()
        nonblank = len([line for line in lines if line.strip()])
        if nonblank < MIN_SKILL_LINES:
            check.add(path, 0, "only %d non-blank lines (minimum %d)" % (nonblank, MIN_SKILL_LINES))
        parsed = parse_document(lines)
        if parsed is None:
            check.add(path, 1, "does not start with a '---' frontmatter block")
            continue
        entries, body_start = parsed
        if body_start is None:
            check.add(path, 1, "frontmatter opened with '---' is never closed")
        keys = {}
        for lineno, key, value in entries:
            if not key:
                check.add(path, lineno, "frontmatter line is not a 'key: value' pair: %s" % value)
            elif key in keys:
                check.add(path, lineno, "duplicate frontmatter key '%s'" % key)
            else:
                keys[key] = (lineno, value)
        for key in sorted(set(keys) - set(SKILL_KEYS)):
            check.add(path, keys[key][0], "unexpected frontmatter key '%s' (only 'name' and 'description' are allowed)" % key)
        name = keys.get("name")
        if name is None:
            check.add(path, 1, "frontmatter is missing 'name:'")
        elif not name[1]:
            check.add(path, name[0], "frontmatter 'name:' is empty")
        elif name[1] != skill_dir.name:
            check.add(path, name[0], "name '%s' does not match directory name '%s'" % (name[1], skill_dir.name))
        if name and name[1]:
            declared.append((name[1], path))
        description = keys.get("description")
        if description is None:
            check.add(path, 1, "frontmatter is missing a non-empty 'description:'")
        elif not description[1]:
            check.add(path, description[0], "frontmatter 'description:' is empty")
        elif len(description[1]) > MAX_DESCRIPTION_CHARS:
            check.add(path, description[0], "description is %d characters (maximum %d)" % (len(description[1]), MAX_DESCRIPTION_CHARS))
    check.detail = "%d skill(s) validated" % len(dirs)
    unique_check.detail = "%d unique name(s)" % len(declared)
    seen = {}
    for name, path in declared:
        if name in seen:
            unique_check.add(path, 0, "duplicate skill name '%s' (also used by %s)" % (name, seen[name]))
        else:
            seen[name] = path


def check_commands(root, check):
    commands_root = root / COMMANDS_DIR
    files = sorted(commands_root.glob("*.md")) if commands_root.is_dir() else []
    for command_file in files:
        path = "%s/%s" % (COMMANDS_DIR, command_file.name)
        lines = read_text(command_file).splitlines()
        parsed, body_start = parse_document(lines), 0
        if parsed is None:
            check.add(path, 1, "does not start with a '---' frontmatter block")
        else:
            entries, body_start = parsed
            if body_start is None:
                check.add(path, 1, "frontmatter opened with '---' is never closed")
                continue
            if not any(key == "description" and value for _, key, value in entries):
                check.add(path, 1, "frontmatter is missing a non-empty 'description:'")
        body = lines[body_start:]
        found = {}
        for prefix in ("Skills:", "Boundaries:"):
            hits = [(body_start + index + 1, line[len(prefix):].strip()) for index, line in enumerate(body) if line.startswith(prefix)]
            found[prefix] = hits
            if len(hits) != 1:
                check.add(path, hits[0][0] if hits else 0, "%d lines start with '%s' (expected exactly 1)" % (len(hits), prefix))
        for lineno, value in found["Skills:"]:
            for name in value.split(","):
                name = name.strip()
                if not name:
                    check.add(path, lineno, "'Skills:' lists an empty skill name")
                elif not (root / SKILLS_DIR / name).is_dir():
                    check.add(path, lineno, "unknown skill '%s' (no %s/%s/ directory)" % (name, SKILLS_DIR, name))
        for lineno, value in found["Boundaries:"]:
            flags = [flag.strip() for flag in value.split(",")]
            if not any(flags):
                check.add(path, lineno, "'Boundaries:' lists no boundary flags")
            for flag in flags:
                if not flag:
                    check.add(path, lineno, "'Boundaries:' lists an empty boundary flag")
                elif flag not in BOUNDARY_FLAGS:
                    check.add(path, lineno, "unknown boundary flag '%s'" % flag)
    check.detail = "%d command(s) validated" % len(files)


def check_retired(root, check, inert_lines):
    present = 0
    for path in RETIRED_PATHS:
        if os.path.lexists(str(root / path)):
            present += 1
            check.add(path, 0, "retired location exists and must be removed")
    check.detail = "%d present" % present
    for path in INERT_PATHS:
        if os.path.lexists(str(root / path)):
            inert_lines.append("INERT: %s is present but inert (not read by OMP; Claude Code is not installed)" % path)


def cmd_check(root):
    titles = ("AGENTS.md", OMP_AGENTS, "skills", "skill names", "commands",
              "retired locations", "forbidden content", "cross-repo paths")
    checks = [Check(number + 1, title) for number, title in enumerate(titles)]
    inert_lines = []

    agents_md = root / ROOT_AGENTS
    if not agents_md.is_file():
        checks[0].add(ROOT_AGENTS, 0, "missing: no AGENTS.md at the repository root")
    elif not read_text(agents_md).strip():
        checks[0].add(ROOT_AGENTS, 0, "empty: AGENTS.md has no non-blank content")
    else:
        checks[0].detail = "1 file, non-empty"
    check_omp_symlink(root, checks[1])
    check_skills(root, checks[2], checks[3])
    check_commands(root, checks[4])
    check_retired(root, checks[5], inert_lines)

    count = scan_forbidden(root, CONTENT_TARGETS, checks[6])
    checks[6].detail = "%d file(s) scanned, %d hit(s)" % (count, len(checks[6].violations))
    count = scan_forbidden(root, TOOLING_TARGETS, checks[7])
    checks[7].detail = "%d file(s) scanned, %d hit(s)" % (count, len(checks[7].violations))

    violations = []
    for check in checks:
        violations.extend(check.violations)
    if violations:
        for path, lineno, reason in sorted(set(violations)):
            print("%s:%d: %s" % (path, lineno, reason))
    else:
        for check in checks:
            print(check.summary())
    for line in inert_lines:
        print(line)
    return 1 if violations else 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="agent_assets.py",
                                     description="Deterministic sync/check tooling for agent knowledge assets.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    sync_parser = subparsers.add_parser("sync", help="materialize the .omp/AGENTS.md relative symlink")
    sync_parser.add_argument("--force", action="store_true",
                             help="replace a regular file, directory, or wrong symlink instead of refusing")
    subparsers.add_parser("check", help="validate agent assets without mutating anything")
    args = parser.parse_args(argv)
    if args.command == "sync":
        return cmd_sync(Path.cwd(), args.force)
    return cmd_check(Path.cwd())


if __name__ == "__main__":
    sys.exit(main())
