"""The Research Computer V1 assets keep their role and authority boundaries.

The literature agent's contract is tested by ``test_literature_agent_assets``;
this module holds the equivalent deterministic contract for the research-computer
loop: the designer designs and the reviewer critiques, and neither can write,
execute, authorize or provision anything. The tool allowlists are the
enforcement, so a widened list is a test failure, not a prompt review.
"""

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
AGENTS_DIR = REPO_ROOT / ".omp" / "agents"
SKILLS_DIR = REPO_ROOT / ".agents" / "skills"
COMMANDS_DIR = REPO_ROOT / ".agents" / "commands"
RESEARCH_DIR = REPO_ROOT / "improvements" / "taskrelation" / "research"

DESIGNER = AGENTS_DIR / "research-designer.md"
REVIEWER = AGENTS_DIR / "research-reviewer.md"
SKILL = SKILLS_DIR / "wavcse-research-computer" / "SKILL.md"
COMMAND = COMMANDS_DIR / "wav-propose.md"
VALIDATOR = RESEARCH_DIR / "proposal_check.py"

# The approved surfaces. Anything outside these is a boundary breach.
DESIGNER_TOOLS = {
    "read", "grep", "glob", "find",
    "literature_resolve", "literature_query", "literature_read", "literature_primary",
}
REVIEWER_TOOLS = {"read", "grep", "glob", "find"}

# No agent in the loop may hold one of these.
FORBIDDEN_TOOLS = {
    "write", "edit", "bash", "task", "eval", "ask", "web_search", "todo", "wait",
}

SKILL_KEYS = {"name", "description"}
VALID_BOUNDARY_FLAGS = {
    "read-only", "no-commit", "no-paid-compute", "writes-reports",
    "mutates-research-state", "may-provision-compute", "may-commit",
}


def parse_frontmatter(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise AssertionError("{} does not start with frontmatter".format(path.name))
    fields = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return fields
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip("\"'")
    raise AssertionError("{} frontmatter is never closed".format(path.name))


def tools_of(fields):
    raw = fields.get("tools", "")
    return {name.strip() for name in raw.split(",") if name.strip()}


class AgentBoundaryTests(unittest.TestCase):
    def test_designer_tool_surface(self):
        fields = parse_frontmatter(DESIGNER)
        self.assertEqual(tools_of(fields), DESIGNER_TOOLS)
        self.assertFalse(tools_of(fields) & FORBIDDEN_TOOLS)
        self.assertNotIn("spawns", fields)

    def test_reviewer_tool_surface(self):
        fields = parse_frontmatter(REVIEWER)
        self.assertEqual(tools_of(fields), REVIEWER_TOOLS)
        self.assertFalse(tools_of(fields) & FORBIDDEN_TOOLS)
        self.assertNotIn("spawns", fields)

    def test_designer_autoloads_the_experiment_operator_and_loop_skills(self):
        skills = parse_frontmatter(DESIGNER).get("autoloadSkills", "")
        self.assertIn("wavcse-experiment-operator", skills)
        self.assertIn("wavcse-research-computer", skills)
        self.assertTrue((SKILLS_DIR / "wavcse-experiment-operator").is_dir())

    def test_reviewer_autoloads_the_loop_skill(self):
        skills = parse_frontmatter(REVIEWER).get("autoloadSkills", "")
        self.assertIn("wavcse-research-computer", skills)

    def test_agents_state_their_authority_limits(self):
        for path, needles in (
            (DESIGNER, ("no write", "authorization", "cannot")),
            (REVIEWER, ("no write", "authorize", "cannot")),
        ):
            body = path.read_text(encoding="utf-8").lower()
            for needle in needles:
                self.assertIn(needle, body, "{} lacks {!r}".format(path.name, needle))

    def test_agents_are_named_by_existing_convention(self):
        for path, name in ((DESIGNER, "research-designer"), (REVIEWER, "research-reviewer")):
            self.assertEqual(parse_frontmatter(path)["name"], name)


class SkillAndCommandTests(unittest.TestCase):
    def test_skill_frontmatter(self):
        fields = parse_frontmatter(SKILL)
        self.assertEqual(set(fields) & SKILL_KEYS, SKILL_KEYS)
        self.assertEqual(fields["name"], "wavcse-research-computer")
        self.assertTrue(fields["description"])
        self.assertLessEqual(len(fields["description"]), 200)
        nonblank = [line for line in SKILL.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(nonblank), 20)

    def test_skill_names_every_role_and_the_validator(self):
        body = SKILL.read_text(encoding="utf-8")
        for needle in ("literature-reviewer", "research-designer", "research-reviewer",
                       "proposal_check.py", "approval request"):
            self.assertIn(needle, body)

    def test_command_declares_skills_and_boundaries(self):
        text = COMMAND.read_text(encoding="utf-8")
        skills = [line for line in text.splitlines() if line.startswith("Skills:")]
        boundaries = [line for line in text.splitlines() if line.startswith("Boundaries:")]
        self.assertEqual(len(skills), 1)
        self.assertEqual(len(boundaries), 1)
        for name in skills[0][len("Skills:"):].split(","):
            self.assertTrue((SKILLS_DIR / name.strip()).is_dir(), name)
        flags = {flag.strip() for flag in boundaries[0][len("Boundaries:"):].split(",") if flag.strip()}
        self.assertNotIn("may-provision-compute", flags)
        self.assertNotIn("may-commit", flags)
        self.assertTrue(flags <= VALID_BOUNDARY_FLAGS)

    def test_command_body_never_grants_execution(self):
        body = COMMAND.read_text(encoding="utf-8").lower()
        self.assertIn("never registers a study", body)
        self.assertIn("stop", body)
        self.assertNotIn("provision workers", body)

    def test_validator_exists_and_is_importable(self):
        self.assertTrue(VALIDATOR.is_file())
        import improvements.taskrelation.research.proposal_check as module
        self.assertIsNotNone(module)


class LifecycleVocabularyTests(unittest.TestCase):
    def test_skill_distinguishes_the_four_events(self):
        body = SKILL.read_text(encoding="utf-8")
        self.assertIn("registered", body)
        self.assertIn("authorization", body)
        self.assertRegex(body, re.compile(r"APPROVED"))

    def test_validator_status_set_covers_the_skill_states(self):
        from improvements.taskrelation.research import proposal_check
        for state in ("DRAFT", "REVIEW_REQUIRED", "CHANGES_REQUESTED",
                      "READY_FOR_HUMAN", "APPROVED", "REJECTED"):
            self.assertIn(state, proposal_check.STATUSES)


if __name__ == "__main__":
    unittest.main()
