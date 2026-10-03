"""The Research Computer's bounded design-review convergence loop.

The main OMP session owns ordinary convergence between the designer and the
reviewer: a ``CHANGES_REQUIRED`` verdict is not by itself a reason to return
control to the human. ``improvements/taskrelation/research/convergence.py`` is
the normative rule, and the skill and command must bind to it. These tests prove
each branch of the loop and that the loop can never register a study, create an
authorization, provision compute or execute anything.
"""

import unittest
from pathlib import Path

from improvements.taskrelation.research import convergence as conv


REPO_ROOT = Path(__file__).resolve().parents[4]
SKILL = REPO_ROOT / ".agents" / "skills" / "wavcse-research-computer" / "SKILL.md"
COMMAND = REPO_ROOT / ".agents" / "commands" / "wav-propose.md"


class ConvergenceDecisionTests(unittest.TestCase):
    """1-5: the five non-approval branches and the approval branch."""

    def test_changes_required_without_human_decision_loops(self):
        """1. CHANGES_REQUIRED + no human decision -> another revision/review cycle."""
        for cycle in range(conv.MAX_REVISION_CYCLES):
            self.assertEqual(
                conv.classify_review_outcome(conv.CHANGES_REQUIRED, cycle),
                conv.REVISE_AND_REREVIEW,
                "cycle {} should loop".format(cycle),
            )
        self.assertIn(conv.REVISE_AND_REREVIEW, conv.LOOP_ACTIONS)
        self.assertNotIn(conv.REVISE_AND_REREVIEW, conv.YIELD_ACTIONS)

    def test_changes_required_with_human_decision_yields(self):
        """2. CHANGES_REQUIRED + human decision -> yield, regardless of cycle."""
        self.assertEqual(
            conv.classify_review_outcome(
                conv.CHANGES_REQUIRED, 0, human_decision_required=True
            ),
            conv.YIELD_HUMAN_DECISION,
        )
        # A human decision yields even at/after the bound.
        self.assertEqual(
            conv.classify_review_outcome(
                conv.CHANGES_REQUIRED, conv.MAX_REVISION_CYCLES,
                human_decision_required=True,
            ),
            conv.YIELD_HUMAN_DECISION,
        )

    def test_pass_yields_ready_for_human(self):
        """3. PASS -> READY_FOR_HUMAN -> yield."""
        self.assertEqual(conv.classify_review_outcome(conv.PASS, 0), conv.READY_FOR_HUMAN)
        # Flags cannot degrade a PASS; a PASS is never bypassed.
        self.assertEqual(
            conv.classify_review_outcome(
                conv.PASS, conv.MAX_REVISION_CYCLES + 5,
                human_decision_required=True, blocked=True,
            ),
            conv.READY_FOR_HUMAN,
        )

    def test_blocked_evidence_or_capability_yields_blocker(self):
        """4. blocked evidence/capability -> yield."""
        self.assertEqual(
            conv.classify_review_outcome(conv.CHANGES_REQUIRED, 0, blocked=True),
            conv.YIELD_BLOCKED,
        )
        self.assertEqual(
            conv.classify_review_outcome(
                conv.CHANGES_REQUIRED, conv.MAX_REVISION_CYCLES, blocked=True
            ),
            conv.YIELD_BLOCKED,
        )

    def test_repeated_non_convergence_yields_at_the_bound(self):
        """5. repeated non-convergence -> yield, and never a fourth cycle."""
        self.assertGreater(conv.MAX_REVISION_CYCLES, 0)
        self.assertEqual(
            conv.classify_review_outcome(
                conv.CHANGES_REQUIRED, conv.MAX_REVISION_CYCLES - 1
            ),
            conv.REVISE_AND_REREVIEW,
        )
        for cycle in range(conv.MAX_REVISION_CYCLES, conv.MAX_REVISION_CYCLES + 3):
            self.assertEqual(
                conv.classify_review_outcome(conv.CHANGES_REQUIRED, cycle),
                conv.YIELD_DISAGREEMENT,
            )

    def test_vocabulary_and_guards(self):
        self.assertEqual(conv.ACTIONS, conv.LOOP_ACTIONS + conv.YIELD_ACTIONS)
        with self.assertRaises(ValueError):
            conv.classify_review_outcome("APPROVED", 0)
        with self.assertRaises(ValueError):
            conv.classify_review_outcome(conv.CHANGES_REQUIRED, -1)


class ConvergenceAuthorityTests(unittest.TestCase):
    """6. No registration/authorization/compute can occur inside this loop."""

    FORBIDDEN = (
        "register", "registration", "authorize", "authorization",
        "provision", "submit", "compute", "execute", "spend",
    )

    def test_convergence_module_exposes_no_side_effecting_capability(self):
        public = set(conv.__all__)
        # Only the classifier is callable.
        callables = {
            name for name in public
            if callable(getattr(conv, name)) and not isinstance(getattr(conv, name), type)
        }
        self.assertEqual(callables, {"classify_review_outcome"})
        # No action or public name is a registration/authorization/compute act.
        for name in public:
            lowered = name.lower()
            self.assertFalse(
                any(bad in lowered for bad in self.FORBIDDEN),
                "loop exposes forbidden capability {!r}".format(name),
            )

    def test_skill_and_command_forbid_side_effects_inside_the_loop(self):
        skill = " ".join(SKILL.read_text(encoding="utf-8").split())
        command = " ".join(COMMAND.read_text(encoding="utf-8").split())
        for text, label in ((skill, "skill"), (command, "command")):
            lowered = text.lower()
            for needle in ("register", "authorization", "compute"):
                self.assertIn(needle, lowered, "{} lacks {!r}".format(label, needle))
        # The skill enumerates the loop's hard prohibitions, and names the one
        # registration the orchestrator owns (a bounded LT literature investigation).
        self.assertIn("register a DG/TR/AB study", skill)
        self.assertIn("the one registration the orchestrator owns", skill)
        self.assertIn("create or widen an authorization", skill)
        self.assertIn("bypass a reviewer `PASS`", skill)


class ConvergenceBindingTests(unittest.TestCase):
    """The prose contract binds to the deterministic policy module."""

    def test_skill_binds_to_the_module_and_bound(self):
        skill = SKILL.read_text(encoding="utf-8")
        self.assertIn("convergence.py", skill)
        self.assertIn("classify_review_outcome", skill)
        self.assertIn("MAX_REVISION_CYCLES = {}".format(conv.MAX_REVISION_CYCLES), skill)
        self.assertIn("Bounded convergence loop", skill)
        self.assertIn("is not by itself a reason to return", skill)
        self.assertIn("owns ordinary convergence", skill)

    def test_skill_lists_the_five_yield_reasons(self):
        skill = " ".join(SKILL.read_text(encoding="utf-8").split())
        for needle in (
            "correctable within the current scope and no human decision required",
            "genuine human decision required",
            "unavailable evidence or capability blocks progress",
            "Never start a fourth cycle",
            "READY_FOR_HUMAN",
        ):
            self.assertIn(needle, skill, "skill lacks {!r}".format(needle))

    def test_command_binds_to_the_bounded_loop(self):
        command = COMMAND.read_text(encoding="utf-8")
        self.assertIn("convergence.py", command)
        self.assertIn("MAX_REVISION_CYCLES", command)
        self.assertIn("correctable", command)
        self.assertIn("yield on a correctable", command)


if __name__ == "__main__":
    unittest.main()
