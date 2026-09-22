"""Run matched DG-0005 confirmation arms sequentially on one GPU."""

import argparse
import subprocess
import sys
from pathlib import Path


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[4]
RUN_BASE = REPO_ROOT / "improvements" / "base" / "run_base.py"
CONFIGS = (
    STUDY_DIR / "configs" / "confirm_a0.yml",
    STUDY_DIR / "configs" / "confirm_a1.yml",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()

    for seed in args.seeds:
        for config in CONFIGS:
            command = [
                sys.executable,
                str(RUN_BASE),
                "--task_type",
                "ks_si_er",
                "--config",
                str(config),
                "--device_index",
                str(args.device_index),
                "--seed",
                str(seed),
            ]
            print(
                f"DG-0005 confirmation | gpu={args.device_index} | "
                f"seed={seed} | config={config.name}",
                flush=True,
            )
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
