# uv Guide

This repository uses [uv](https://docs.astral.sh/uv/) instead of conda.
Dependencies live in `pyproject.toml`; `uv.lock` pins every package to an exact
version so all four clones on the shared machine run identical environments.

## Install uv (once per machine)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Daily commands

| Task | Command |
| --- | --- |
| Create / update the venv | `uv sync` |
| Run anything inside the venv | `uv run python main.py ...` |
| Activate the venv manually | `source .venv/bin/activate` |
| Add a dependency | `uv add <package>` |
| Remove a dependency | `uv remove <package>` |
| Upgrade all pins | `uv lock --upgrade && uv sync` |
| See what is installed | `uv pip list` |

Notes:

- `uv sync` uses `uv.lock` — every clone ends up with byte-for-byte the same
  package set. Always run it after pulling, and **commit `uv.lock`** whenever
  `pyproject.toml` changes.
- Python 3.9 is managed by uv itself (`pyproject.toml` pins
  `requires-python = "==3.9.*"`); the first `uv sync` downloads it
  automatically. Do not use the system/conda Python for this project.
- `uv add` / `uv remove` edit `pyproject.toml` **and** re-lock automatically,
  so prefer them over hand-editing the file.
- Old conda workflow (`conda activate opencv`) is gone — the conda
  environments were removed. Use `uv run` or `source .venv/bin/activate`
  instead.
- DVC is installed inside the venv. Run `uv run dvc pull` (or activate first)
  to fetch `embedding.tar.gz`.

## How it behaves on the shared machine

- Each clone has its **own `.venv/`** (git-ignored), so different branches can
  have different dependencies without clashing.
- uv keeps a global cache (`~/.cache/uv`) and hardlinks packages into each
  venv, so after the first `uv sync`, additional clones cost almost no disk
  space. If `df -h` shows the root disk near full, see the repo CLAUDE.md
  gotchas before starting long training runs.

## Troubleshooting

- **`uv: command not found`** — the installer adds uv to `~/.local/bin`;
  start a new shell or run `source ~/.bashrc`.
- **Disk full during `uv sync`** — free space (old results/checkpoints, or
  `uv cache clean`) and retry.
- **CUDA** — the `torch` wheels from PyPI bundle CUDA 12.6 runtime libraries,
  so no system CUDA install is needed; the GPU driver must support CUDA 12
  (`nvidia-smi` driver ≥ 525).
