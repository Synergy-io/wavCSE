# CLAUDE.md

Compatibility pointer for Claude Code. This repository's agent tooling is **not** Claude Code, and
nothing durable belongs in this file: the canonical entry point is `AGENTS.md`, which carries the
project instructions, the hard invariants, the validation gate and the engineering conventions.

- **Domain competence:** `.agents/skills/<name>/SKILL.md` — `wavcse-research-runner` (whole-system
  entry point), `wavcse-experiment-operator` (one Task Relation Learning study),
  `wavcse-embedding-generation` (embedding readiness); infrastructure mechanics are the bounded
  `infra/` subsystem, whose rules live in `infra/AGENTS.md`.
- **Workflow intent:** `.agents/commands/` — `wav-cycle`, `wav-experiment`, `wav-embeddings`,
  `wav-analyze`, `wav-literature`, `wav-status`, `wav-weekly`, `wav-propose`,
  `wav-execution-preflight`.
- **What the repository is:** the two-stage wavCSE framework (frozen WavLM embeddings → shared
  backbone with per-task heads) plus `improvements/`, the parameter-based MTL architectures — see
  `AGENTS.md` and `improvements/README.md`.
- `.claude/rules/` holds compatibility stubs only; a rule duplicated there would be a second source
  of truth. Edit `AGENTS.md` or a skill instead.
- Runtime error messages in `improvements/` say "see CLAUDE.md gotchas": those gotchas (disk-space
  preflight, module invocation, CPU embedding loading) are now in `AGENTS.md` under *Validation* and
  *Engineering conventions*.
