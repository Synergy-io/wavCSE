---
paths:
  - "improvements/**"
---

# Compatibility stub — not a source of truth

This file exists only so Claude Code has something to load for `improvements/**`; it is not
authoritative and must never become a second copy of a rule.

- Canonical instructions: `AGENTS.md` (its **Engineering conventions** section covers this
  directory); domain competence: `.agents/skills/`; workflow intent: `.agents/commands/`.
- The load-bearing content that used to live here (MLflow/DagsHub tracking and experiment naming,
  numbered architecture folders, CPU embedding loading, ER leave-one-speaker-out, dataset paths,
  disk-space preflight) now lives in `AGENTS.md`, `wavcse-experiment-operator/SKILL.md` and the READMEs.
