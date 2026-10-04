"""Static contract for the capability / mutation boundary.

The boundary is enforced in two places, and both are pinned here:

* the project OMP config, which must keep every writing spawn's changes as a
  candidate patch (`task.isolation.enabled: true`, `task.isolation.apply:
  false`) instead of writing them back to the canonical checkout;
* the writing-role grants and the deterministic candidate-change gate that
  judges their candidates before integration.

A change that silently flips `task.isolation.apply` back on, widens a writing
grant to include delegation, or removes the gate's self-test from ``make check``
must fail here rather than in review.
"""

import os
import re
import unittest

import yaml


REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
CONFIG = os.path.join(REPO_ROOT, ".omp", "config.yml")
AGENT_DIR = os.path.join(REPO_ROOT, ".omp", "agents")
MAKEFILE = os.path.join(REPO_ROOT, "Makefile")
GATE = os.path.join(REPO_ROOT, "scripts", "agents", "candidate_gate.py")

WRITING_AGENTS = ("research-executor", "infrastructure-engineer")
READ_ONLY_AGENTS = ("research-designer", "research-reviewer")

# Tools that would let a writing specialist escape its candidate workspace by
# delegating to another agent, running ad-hoc code with a different authority,
# or reaching a provider. `bash` alone is necessary for local implementation and
# is the residual escape route documented in AGENTS.md.
DELEGATING_TOOLS = {"task", "ask", "eval", "web_search", "computer", "browser",
                    "launch", "hub"}
FORBIDDEN_PATTERNS = ("mcp__", "xd://mcp__")


def frontmatter(path):
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise AssertionError("{} has no frontmatter".format(path))
    entries = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        entries[key.strip()] = value.strip()
    return entries, text


def tools_of(path):
    entries, _ = frontmatter(path)
    return {name.strip() for name in entries.get("tools", "").split(",") if name.strip()}


class ProjectIsolationConfigTests(unittest.TestCase):
    def setUp(self):
        with open(CONFIG, "r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

    def test_isolation_is_enabled_and_never_auto_applied(self):
        isolation = self.config["task"]["isolation"]
        self.assertIs(True, isolation["enabled"],
                      "task.isolation.enabled must stay true so a writing spawn "
                      "can be given a candidate workspace")
        self.assertIs(False, isolation["apply"],
                      "task.isolation.apply must stay false: a successful "
                      "candidate must never be written back to the canonical "
                      "checkout automatically")

    def test_candidate_changes_are_never_auto_applied(self):
        # The invariant is `apply: false`. The merge strategy (patch or branch)
        # only chooses how the retained candidate is represented.
        isolation = self.config["task"]["isolation"]
        self.assertIs(False, isolation["apply"])
        self.assertIn(isolation["merge"], ("patch", "branch"))

    def test_no_project_setting_re_enables_a_bypass(self):
        # `disabledExtensions` is the only intended global toggle; nothing may
        # disable the isolation mechanism or the agent-asset gate.
        self.assertNotIn("disabledExtensions", self.config.get("tools", {}))
        self.assertNotIn("disabledAgents", self.config.get("task", {}))


class WritingGrantTests(unittest.TestCase):
    def test_writing_agents_cannot_delegate_or_reach_a_provider(self):
        for name in WRITING_AGENTS:
            granted = tools_of(os.path.join(AGENT_DIR, name + ".md"))
            with self.subTest(agent=name):
                self.assertFalse(granted & DELEGATING_TOOLS,
                                 "{} may not hold {}".format(name, sorted(granted & DELEGATING_TOOLS)))
                self.assertTrue({"edit", "write"} <= granted,
                                "{} is the writing specialist and must hold edit+write".format(name))
                self.assertIn("bash", granted)

    def test_writing_agent_prompts_state_the_candidate_workspace_rule(self):
        for name in WRITING_AGENTS:
            _, text = frontmatter(os.path.join(AGENT_DIR, name + ".md"))
            with self.subTest(agent=name):
                self.assertRegex(text, r"isolated candidate workspace|candidate workspace")

    def test_read_only_specialists_have_no_write_tools(self):
        for name in READ_ONLY_AGENTS:
            granted = tools_of(os.path.join(AGENT_DIR, name + ".md"))
            with self.subTest(agent=name):
                self.assertFalse({"edit", "write", "bash"} & granted)
                self.assertFalse(granted & DELEGATING_TOOLS)

    def test_no_agent_grant_uses_an_mcp_proxy(self):
        for name in os.listdir(AGENT_DIR):
            if not name.endswith(".md"):
                continue
            with open(os.path.join(AGENT_DIR, name), "r", encoding="utf-8") as handle:
                text = handle.read()
            for pattern in FORBIDDEN_PATTERNS:
                with self.subTest(agent=name, pattern=pattern):
                    self.assertNotIn(pattern, text)


class CandidateGateTests(unittest.TestCase):
    def test_the_gate_exists_and_is_stdlib_only(self):
        self.assertTrue(os.path.isfile(GATE))
        with open(GATE, "r", encoding="utf-8") as handle:
            text = handle.read()
        for banned in ("import requests", "import boto3", "urllib.request"):
            self.assertNotIn(banned, text)

    def test_make_check_runs_the_gate_self_test(self):
        with open(MAKEFILE, "r", encoding="utf-8") as handle:
            makefile = handle.read()
        self.assertIn("candidate-gate-check:", makefile)
        check_line = next(line for line in makefile.splitlines()
                          if line.startswith("check:"))
        self.assertIn("candidate-gate-check", check_line,
                      "`make check` must include the candidate-gate self-test")

    def test_gate_protects_the_authority_paths_the_boundary_names(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("candidate_gate", GATE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertIn("scripts/", module.VALIDATION_PREFIXES)
        self.assertIn(".omp/", module.CONTROL_PLANE_PREFIXES)
        self.assertIn(".agents/", module.CONTROL_PLANE_PREFIXES)
        self.assertIn("Makefile", module.VALIDATION_FILES)
        self.assertIn("AGENTS.md", module.CONTROL_PLANE_FILES)
        self.assertEqual(
            {"research-executor": ("improvements/",),
             "infrastructure-engineer": ("infra/",)},
            module.ROLE_SCOPES)

        # The research lifecycle authorities the boundary names must be protected.
        for protected in (
            module.RESEARCH + "proposals/",
            module.RESEARCH + "authorizations/",
            module.RESEARCH + "execution/",
            module.RESEARCH + "STUDIES.jsonl",
            module.RESEARCH + "proposal_check.py",
            module.RESEARCH + "execution_contract.py",
            "improvements/compute/",
            "downstream/",
        ):
            with self.subTest(path=protected):
                reason = module.classify(
                    protected + "x" if protected.endswith("/") else protected,
                    baseline_paths=set(), role=None)
                self.assertIsNotNone(reason, "{} must be protected".format(protected))


class BoundaryDocumentationTests(unittest.TestCase):
    def test_agents_md_records_the_boundary_and_the_residual_trust(self):
        with open(os.path.join(REPO_ROOT, "AGENTS.md"), "r", encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("Capability boundary", text)
        self.assertIn("candidate-gate-check", text)
        # The boundary is honest about what is not sandboxed.
        self.assertRegex(text, r"[Mm]ain OMP")
        self.assertRegex(text, r"[Rr]esidual")


if __name__ == "__main__":
    unittest.main()
