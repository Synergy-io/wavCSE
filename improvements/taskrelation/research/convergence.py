"""Bounded design-review convergence policy for the wavCSE Research Computer.

The main OMP session — not the human — owns ordinary convergence between the
designer and the reviewer. This module is the normative, deterministic rule the
skill `wavcse-research-computer` and the `/wav-propose` command apply after
every `research-reviewer` verdict. It decides only **what the orchestrator does
next**; it cannot and does not register a study, create or widen an
authorization, provision or submit compute, execute anything, or mutate any
research record.

Vocabulary (one action per reviewer verdict):

* ``PASS`` → ``READY_FOR_HUMAN`` — the proposal may be yielded as an approval
  request; a reviewer ``PASS`` is never bypassed.
* ``CHANGES_REQUIRED`` that is correctable within the current scope and needs no
  human decision → ``REVISE_AND_REREVIEW`` — the orchestrator revises with the
  designer and re-reviews; it does **not** return control to the human.
* ``CHANGES_REQUIRED`` requiring a genuine human decision (research direction, a
  human-fixed decision, or a protocol/resource choice classified as human by
  ``.agents/policies/autonomy.md``) → ``YIELD_HUMAN_DECISION``.
* ``CHANGES_REQUIRED`` blocked by unavailable evidence or capability → ``YIELD_BLOCKED``.
* ``CHANGES_REQUIRED`` that has not converged after ``MAX_REVISION_CYCLES``
  revision cycles → ``YIELD_DISAGREEMENT`` (unresolved specialist disagreement /
  non-convergence); a fourth cycle is never started.

``revision_cycle`` is the number of revision cycles already completed for the
proposal while handling ``CHANGES_REQUIRED`` (0 for the first verdict).
"""

MAX_REVISION_CYCLES = 3

PASS = "PASS"
CHANGES_REQUIRED = "CHANGES_REQUIRED"

# Actions the orchestrator may take.
READY_FOR_HUMAN = "READY_FOR_HUMAN"
REVISE_AND_REREVIEW = "REVISE_AND_REREVIEW"
YIELD_HUMAN_DECISION = "YIELD_HUMAN_DECISION"
YIELD_BLOCKED = "YIELD_BLOCKED"
YIELD_DISAGREEMENT = "YIELD_DISAGREEMENT"

LOOP_ACTIONS = (REVISE_AND_REREVIEW,)
YIELD_ACTIONS = (
    READY_FOR_HUMAN,
    YIELD_HUMAN_DECISION,
    YIELD_BLOCKED,
    YIELD_DISAGREEMENT,
)
ACTIONS = LOOP_ACTIONS + YIELD_ACTIONS

__all__ = [
    "MAX_REVISION_CYCLES",
    "PASS",
    "CHANGES_REQUIRED",
    "READY_FOR_HUMAN",
    "REVISE_AND_REREVIEW",
    "YIELD_HUMAN_DECISION",
    "YIELD_BLOCKED",
    "YIELD_DISAGREEMENT",
    "LOOP_ACTIONS",
    "YIELD_ACTIONS",
    "ACTIONS",
    "classify_review_outcome",
]


def classify_review_outcome(
    verdict,
    revision_cycle,
    human_decision_required=False,
    blocked=False,
):
    """Return the next orchestrator action for one reviewer verdict.

    ``verdict`` is ``PASS`` or ``CHANGES_REQUIRED``. ``revision_cycle`` is the
    number of revision cycles already completed (>= 0). ``human_decision_required``
    and ``blocked`` are the orchestrator's classification of the reviewer's
    blocking findings. A human decision or a blocker always yields immediately;
    otherwise a ``CHANGES_REQUIRED`` verdict either loops (below the bound) or
    yields as an unresolved disagreement (at or above the bound).
    """
    if verdict == PASS:
        return READY_FOR_HUMAN
    if verdict != CHANGES_REQUIRED:
        raise ValueError("unknown review verdict: {!r}".format(verdict))
    if revision_cycle < 0:
        raise ValueError("revision_cycle must be >= 0")
    if human_decision_required:
        return YIELD_HUMAN_DECISION
    if blocked:
        return YIELD_BLOCKED
    if revision_cycle >= MAX_REVISION_CYCLES:
        return YIELD_DISAGREEMENT
    return REVISE_AND_REREVIEW
