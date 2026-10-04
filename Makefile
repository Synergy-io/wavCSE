# Verification entrypoints.
#
# `make check` is the gate a change must pass: the deterministic agent-asset
# drift check, the compute-backend tests, and the research suite.
# Nothing here needs a GPU, a network, or cloud credentials.
#
# The MSSL solver module `improvements/taskrelation/research/tests/test_mssl_omega_solver.py`
# used to be excluded here as known red: it asserted properties of the
# unregistered `04-mssl` draft, and its three failures were real defects (a
# transposed analytic gradient, bit-exact-zero assertions on the wrong
# variable, and an ADMM that did not converge at the summary matrix's scale).
# TR-0007's Option-A decision (DEC-0015) fixed both sides, so the module now
# runs inside the gate and no exclusion remains.

UV := uv run --locked
RESEARCH_TESTS := improvements/taskrelation/research/tests
COMPUTE_TESTS := improvements/compute/tests

.PHONY: agents-sync agents-check check compute-check infra-check literature-tools-check research-check research-check-all help

help:
	@echo "agents-sync               materialize .omp/AGENTS.md"
	@echo "agents-check              validate agent assets (no mutation)"
	@echo "compute-check             compute-backend tests (offline, fixtures)"
	@echo "infra-check               infra subsystem checks (run with infra's Python 3.12 environment)"
	@echo "literature-tools-check    exercise the model-facing literature tool adapter (needs bun; not part of 'check')"
	@echo "research-check            research tests"
	@echo "research-check-all        alias of research-check (kept for callers)"
	@echo "check                     agents-check + compute-check + research-check"

agents-sync:
	python3 scripts/agents/agent_assets.py sync

agents-check:
	python3 scripts/agents/agent_assets.py check

# Not part of `check`: this gate needs bun (OMP's runtime), and the research suite
# must stay runnable without a JS toolchain. The agent-asset tests inside
# research-check enforce the same surface's static contract.
literature-tools-check:
	bun run scripts/agents/literature_tools_check.ts

compute-check:
	$(UV) python -m unittest discover -s $(COMPUTE_TESTS) -t .

infra-check:
	$(MAKE) -C infra check

research-check:
	$(UV) python -m unittest discover -s $(RESEARCH_TESTS) -t $(RESEARCH_TESTS)

research-check-all: research-check

check: agents-check compute-check research-check
