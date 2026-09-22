#!/usr/bin/env python3
"""Run DG-0002 matched confirmation arms sequentially on one GPU."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[4]
ARMS = (
    ("baseline", "original", STUDY_DIR / "configs" / "confirm_baseline.yml"),
    ("mtrl", "mtrl", STUDY_DIR / "configs" / "confirm_mtrl.yml"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = STUDY_DIR / "logs" / "confirmation"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    for seed in args.seeds:
        for arm, model, config_path in ARMS:
            command = [
                sys.executable,
                "improvements/run_improvements.py",
                "--model",
                model,
                "--task_type",
                "ks_si_er",
                "--config",
                str(config_path.relative_to(REPO_ROOT)),
                "--device_index",
                str(args.device),
                "--seed",
                str(seed),
            ]
            log_path = log_dir / f"confirm_{arm}_s{seed:02d}.log"
            print(f"Starting {arm} seed {seed} on cuda:{args.device}; log={log_path}", flush=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"\nCOMMAND: {' '.join(command)}\n")
                log_file.flush()
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                print(
                    f"FAILED {arm} seed {seed} with exit code {completed.returncode}",
                    file=sys.stderr,
                    flush=True,
                )
                return completed.returncode
            print(f"Completed {arm} seed {seed}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
