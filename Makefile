# Verification entrypoints for the agent knowledge system.
# `make check` runs the deterministic agent-asset drift check (CI-suitable);
# `make agents-sync` materializes the tool-specific state it validates.
# Pure Python standard library: no pytest, no uv, no network.

.PHONY: agents-sync agents-check check

agents-sync:
	python3 scripts/agents/agent_assets.py sync

agents-check:
	python3 scripts/agents/agent_assets.py check

check: agents-check
