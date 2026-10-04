"""Worker-side payload for an infrastructure-validation job.

Runs inside the checkout at the exact commit the job names, on the worker the
control plane prepared, and does exactly one thing: writes a small,
deterministic JSON record of the run it is part of.

It is deliberately not a research command. It reads no dataset, loads no
embeddings, trains nothing, reaches no tracker and calls no provider. It exists
so an infrastructure-validation scope can exercise the real path — exact
commit, allocation, readiness, job execution, S3 persistence and read-back,
terminal release — without borrowing a Study's scientific authority.

Invoked as::

    python3 -m improvements.compute.infra_probe --seed 0 \
        --output outputs/smoke_s00/smoke_probe.txt

``--output`` is relative to the job directory the control plane provides, never
to the checkout the command happens to run in. Only the standard library is
imported, so the job needs no research environment installed.
"""

import argparse
import json
import os
import pathlib
import sys

PROTOCOL = "in0001-smoke-v1"
JOB_DIRECTORY_ENV = "WAVCSE_JOB_DIRECTORY"


def job_directory():
    configured = os.environ.get(JOB_DIRECTORY_ENV, "").strip()
    if not configured:
        raise SystemExit(
            "{} is not set; this probe runs only inside a prepared job".format(
                JOB_DIRECTORY_ENV
            )
        )
    return pathlib.Path(configured)


def build_payload(seed, environ):
    """The deterministic record of one probe run.

    Nothing here is derived from a clock or a random source, so two runs of the
    same job on the same worker produce the same bytes; the only variation is
    the provenance the control plane injected.
    """

    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "kind": "infrastructure_validation_probe",
        "seed": int(seed),
        "job_id": environ.get("WAVCSE_JOB_ID", ""),
        "commit": environ.get("WAVCSE_JOB_COMMIT", ""),
        "provider": environ.get("INFRA_PROVIDER", ""),
        "worker_id": environ.get("INFRA_WORKER_ID", ""),
        "gpu": environ.get("INFRA_GPU", ""),
    }


def resolve_output(job_directory_path, relative):
    """Resolve ``--output`` inside the job directory, and never outside it."""

    root = os.path.realpath(str(job_directory_path))
    target = os.path.realpath(os.path.join(root, relative))
    if target != root and not target.startswith(root + os.sep):
        raise SystemExit("--output must stay inside the job directory")
    return target, root


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Write one infrastructure-validation probe artifact"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True,
                        help="job-directory-relative output file")
    args = parser.parse_args(argv)

    directory = job_directory()
    target, root = resolve_output(directory, args.output)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    text = json.dumps(build_payload(args.seed, os.environ), sort_keys=True, indent=2)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(
        "infrastructure probe wrote {} ({} bytes)".format(
            os.path.relpath(target, root), os.path.getsize(target)
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
