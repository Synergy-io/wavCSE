"""Static authority and discoverability contract for Execution Plane V1 agents."""

import unittest
from pathlib import Path

from improvements.taskrelation.research import proposal_check


REPO_ROOT = Path(__file__).resolve().parents[4]
AGENT_DIR = REPO_ROOT / ".omp" / "agents"
SKILL_DIR = REPO_ROOT / ".agents" / "skills"
COMMAND_DIR = REPO_ROOT / ".agents" / "commands"

EXECUTOR = AGENT_DIR / "research-executor.md"
INFRA = AGENT_DIR / "infrastructure-engineer.md"
DESIGNER = AGENT_DIR / "research-designer.md"
REVIEWER = AGENT_DIR / "research-reviewer.md"
LITERATURE = AGENT_DIR / "literature-reviewer.md"
PREFLIGHT = COMMAND_DIR / "wav-execution-preflight.md"
EXPERIMENT = COMMAND_DIR / "wav-experiment.md"
NORMATIVE = (
    REPO_ROOT / "improvements" / "taskrelation" / "research" / "execution" / "README.md"
)

IMPLEMENTATION_TOOLS = {"read", "grep", "glob", "find", "bash", "edit", "write"}
READ_ONLY_TOOLS = {"read", "grep", "glob", "find"}
DESIGNER_TOOLS = READ_ONLY_TOOLS | {
    "literature_resolve",
    "literature_query",
    "literature_read",
    "literature_primary",
}
LITERATURE_TOOLS = {
    "literature_resolve",
    "literature_query",
    "literature_read",
    "literature_primary",
    "literature_discover",
    "literature_record",
}
UNRELATED_OR_DELEGATING_TOOLS = {
    "task",
    "ask",
    "web_search",
    "literature_record",
    "literature_discover",
    "literature_admit",
    "literature_acquire",
}


def frontmatter(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        raise AssertionError("{} has no frontmatter".format(path))
    result = {}
    for line in lines[1:]:
        if line == "---":
            return result
        key, separator, value = line.partition(":")
        if separator:
            result[key.strip()] = value.strip().strip('"')
    raise AssertionError("{} frontmatter is not closed".format(path))


def tools(path):
    return {
        item.strip()
        for item in frontmatter(path).get("tools", "").split(",")
        if item.strip()
    }


def boundaries(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Boundaries:"):
            return {
                item.strip()
                for item in line[len("Boundaries:") :].split(",")
                if item.strip()
            }
    return set()


class SpecialistGrantTests(unittest.TestCase):
    def test_research_executor_has_local_implementation_tools_but_no_execution_or_literature_tools(self):
        self.assertEqual(tools(EXECUTOR), IMPLEMENTATION_TOOLS)
        self.assertFalse(tools(EXECUTOR) & UNRELATED_OR_DELEGATING_TOOLS)
        self.assertEqual(frontmatter(EXECUTOR)["name"], "research-executor")
        self.assertIn("wavcse-research-executor", frontmatter(EXECUTOR)["autoloadSkills"])

    def test_infrastructure_engineer_has_maintenance_tools_but_no_provider_or_literature_tools(self):
        self.assertEqual(tools(INFRA), IMPLEMENTATION_TOOLS)
        self.assertFalse(tools(INFRA) & UNRELATED_OR_DELEGATING_TOOLS)
        self.assertEqual(frontmatter(INFRA)["name"], "infrastructure-engineer")
        self.assertIn("wavcse-infrastructure-engineer", frontmatter(INFRA)["autoloadSkills"])

    def test_specialists_cannot_delegate_or_mutate_literature_through_their_grants(self):
        for agent in (EXECUTOR, INFRA):
            self.assertNotIn("spawns", frontmatter(agent))
            self.assertFalse(tools(agent) & {"task", "literature_record", "literature_discover"})


class ExistingAgentCompatibilityTests(unittest.TestCase):
    def test_designer_and_reviewer_grants_are_unchanged(self):
        self.assertEqual(tools(DESIGNER), DESIGNER_TOOLS)
        self.assertEqual(tools(REVIEWER), READ_ONLY_TOOLS)

    def test_literature_agent_grant_is_unchanged(self):
        self.assertEqual(tools(LITERATURE), LITERATURE_TOOLS)

    def test_proposal_lifecycle_vocabulary_is_unchanged(self):
        self.assertEqual(
            proposal_check.STATUSES,
            (
                "DRAFT",
                "REVIEW_REQUIRED",
                "CHANGES_REQUESTED",
                "READY_FOR_HUMAN",
                "APPROVED",
                "REJECTED",
                "SUPERSEDED",
            ),
        )


class EntryPointTests(unittest.TestCase):
    def test_preflight_command_is_explicitly_zero_cost(self):
        self.assertEqual(
            boundaries(PREFLIGHT), {"writes-reports", "may-commit", "no-paid-compute"}
        )
        self.assertIn("wavcse-execution-plane", PREFLIGHT.read_text(encoding="utf-8"))

    def test_live_experiment_command_requires_the_execution_plane(self):
        text = EXPERIMENT.read_text(encoding="utf-8")
        self.assertIn("wavcse-execution-plane", text)
        self.assertIn("PREFLIGHT_ACCEPTED", text)

    def test_one_normative_document_owns_the_execution_architecture(self):
        self.assertTrue(NORMATIVE.is_file())
        text = NORMATIVE.read_text(encoding="utf-8")
        self.assertIn("APPROVED PROPOSAL", text)
        self.assertIn("RESEARCH EXECUTOR", text)
        self.assertIn("INFRASTRUCTURE ENGINEER", text)
        self.assertIn("DETERMINISTIC WAVCSE-INFRA", text)


if __name__ == "__main__":
    unittest.main()
