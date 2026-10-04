"""Worker-side job entry point: run one study arm, then stage its evidence.

The control plane runs this module (from the checkout at the exact commit the
job names), never a hand-written shell command. It does three things and
nothing else:

1. expands and runs the arm's argv from the committed plan, with the run
   identity redirected to a file so the produced run is known exactly rather
   than inferred;
2. stages the plan's declared outputs to deterministic workspace paths — the
   same paths the job spec declared as required;
3. writes a sidecar ``MANIFEST.json`` recording each staged file's SHA-256 and
   size, so the evidence carries its own identity.

It exits non-zero if the training argv fails or if a required output cannot be
staged; a run that produced a *negative result* but wrote its outputs is a
success here, and the interpretation belongs to the science.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

from improvements import embedding_root
from improvements.compute import jobspec
from improvements.compute.embedding_layout import EmbeddingLayoutError
from improvements.run_identity import ENV_FILE, read_identity_file

JOB_DIRECTORY_ENV = "WAVCSE_JOB_DIRECTORY"
JOB_ID_ENV = "WAVCSE_JOB_ID"
JOB_COMMIT_ENV = "WAVCSE_JOB_COMMIT"

_EPOCH = re.compile(r"_epoch(\d+)\.pth$")


def _sha256_of(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _job_directory():
    configured = os.environ.get(JOB_DIRECTORY_ENV, "").strip()
    return os.path.abspath(configured) if configured else os.getcwd()


def _select_checkpoint(checkpoints_dir, tag):
    """Pick the checkpoint for one tag inside a single, explicit run directory.

    The selection rule mirrors the evaluator's own: the tag is matched against
    the file name inside this run's directory. For the rolling ``epoch``
    checkpoints the highest epoch number wins, which is deterministic and is
    the same file the evaluator evaluated last.
    """

    if not os.path.isdir(checkpoints_dir):
        raise RuntimeError("checkpoint directory not found: {}".format(checkpoints_dir))
    candidates = [
        name for name in sorted(os.listdir(checkpoints_dir))
        if name.endswith(".pth") and tag in name.lower()
    ]
    if not candidates:
        raise RuntimeError(
            "no checkpoint matching {!r} in {}".format(tag, checkpoints_dir)
        )
    if tag == "epoch":
        numbered = [
            (int(match.group(1)), name)
            for match in (_EPOCH.search(name) for name in candidates) if match
        ]
        if numbered:
            numbered.sort()
            return os.path.join(checkpoints_dir, numbered[-1][1])
    if len(candidates) > 1:
        raise RuntimeError(
            "ambiguous checkpoint for tag {!r} in {}: {}".format(
                tag, checkpoints_dir, ", ".join(candidates)
            )
        )
    return os.path.join(checkpoints_dir, candidates[0])


def _source_for(plan, arm, seed, output, identity):
    results_dir = identity.get("results_dir")
    checkpoints_dir = identity.get("checkpoints_dir")
    kind = output["kind"]
    if kind == "checkpoint":
        return _select_checkpoint(checkpoints_dir, output["tag"])
    if kind == "gradient_diagnostics":
        path = os.path.join(results_dir, "gradient_diagnostics.json")
        if not os.path.exists(path):
            raise RuntimeError(
                "the plan requires gradient_diagnostics.json but the run did not "
                "write one (is training.gradient_diagnostics.enabled set?)"
            )
        return path
    name = output["name"]
    relative = name if name.endswith(".txt") else name + ".txt"
    path = os.path.join(results_dir, relative)
    if not os.path.exists(path):
        raise RuntimeError(
            "the run did not write the declared results file {!r}".format(relative)
        )
    return path


def _target_name(output):
    if output["kind"] == "checkpoint":
        return "checkpoint_{}.pth".format(output["tag"])
    if output["kind"] == "gradient_diagnostics":
        return "gradient_diagnostics.json"
    return output["name"] + ".txt"


def stage_outputs(plan, arm, seed, identity, outputs_dir):
    """Copy each declared output to its deterministic name and hash it."""

    os.makedirs(outputs_dir, exist_ok=True)
    staged = []
    missing = []
    for output in plan["outputs"]:
        required = bool(output.get("required", True))
        try:
            source = _source_for(plan, arm, seed, output, identity)
        except RuntimeError as exc:
            if required:
                missing.append("{}: {}".format(output["name"], exc))
            continue
        target = os.path.join(outputs_dir, _target_name(output))
        if required and os.path.getsize(source) == 0:
            missing.append("{}: required output is empty".format(output["name"]))
            continue
        with open(source, "rb") as reader, open(target, "wb") as writer:
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                writer.write(chunk)
        staged.append({
            "name": output["name"],
            "kind": output["kind"],
            "required": required,
            "target": os.path.basename(target),
            "source": source,
            "sha256": _sha256_of(target),
            "size_bytes": os.path.getsize(target),
        })
    return staged, missing


def write_manifest(plan, *, stage, arm, seed, identity, staged, outputs_dir,
                   training_exit_code, embedding_layout=None):
    manifest = {
        "schema_version": 1,
        "study": jobspec.plan_scope(plan),
        "stage": stage,
        "arm": arm["arm"],
        "method": arm.get("method", arm["arm"]),
        "seed": seed,
        "task_type": plan["task_type"],
        "job_id": os.environ.get(JOB_ID_ENV, ""),
        "commit": os.environ.get(JOB_COMMIT_ENV, ""),
        "run_id": identity.get("run_id"),
        "checkpoint_run_id": identity.get("checkpoint_run_id"),
        "training_exit_code": training_exit_code,
        "staged_at": datetime.now(timezone.utc).isoformat(),
        "files": sorted(staged, key=lambda entry: entry["name"]),
    }
    if embedding_layout is not None:
        # The evidence carries the mapping it ran under, so a reader can see which
        # verified artifacts the loader-visible root was built from.
        manifest["embedding_layout"] = embedding_layout
    path = os.path.join(outputs_dir, "MANIFEST.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one staged study arm and collect its evidence")
    parser.add_argument("--plan", required=True, help="Path of the compute plan in the checkout")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outputs-root", required=True,
                        help="Job-directory-relative outputs directory")
    args = parser.parse_args(argv)

    plan = jobspec.load_plan(args.plan)
    arm = jobspec.arm_by_name(plan, args.arm)
    if args.seed not in jobspec.stage_seeds(plan, args.stage):
        print(
            "seed {} is not pre-registered for stage {} of {}".format(
                args.seed, args.stage, jobspec.plan_scope(plan)
            ),
            file=sys.stderr,
        )
        return 2
    if args.arm not in jobspec.stage_arm_names(plan, args.stage):
        print(
            "arm {} is not selected by stage {} of {} (stage runs: {})".format(
                args.arm, args.stage, jobspec.plan_scope(plan),
                ", ".join(jobspec.stage_arm_names(plan, args.stage)),
            ),
            file=sys.stderr,
        )
        return 2

    job_directory = _job_directory()
    outputs_dir = os.path.join(job_directory, args.outputs_root)
    identity_file = os.path.join(outputs_dir, ".run_identity.jsonl")
    os.makedirs(os.path.dirname(identity_file), exist_ok=True)

    layout_summary = None
    if plan.get("embedding_layout"):
        # The verified inputs become the tree the loader reads, and the root is named
        # explicitly. Failure here is fatal on purpose: a run that cannot prove which
        # embeddings it is about to read must not read any.
        from improvements.compute import embedding_layout

        try:
            marker = embedding_layout.prepare(
                plan, stage=args.stage, arm=args.arm, seed=args.seed,
                checkout_root=os.getcwd(), job_directory=job_directory,
                commit=os.environ.get(JOB_COMMIT_ENV, ""), job_id=os.environ.get(JOB_ID_ENV, ""),
                log=lambda line: print(line, flush=True),
            )
        except EmbeddingLayoutError as exc:
            print("embedding layout failed: {}".format(exc), file=sys.stderr)
            return 4
        layout_summary = embedding_layout.describe(marker)
        environment_override = marker["loader_root"]
    else:
        environment_override = None

    command = jobspec.expand_argv(plan, arm, args.seed)
    environment = dict(os.environ)
    environment[ENV_FILE] = identity_file
    if environment_override:
        environment[embedding_root.ENV_ROOT] = environment_override
    environment.setdefault("PYTHONUNBUFFERED", "1")

    print("ARC_COMMAND " + json.dumps(command), flush=True)
    completed = subprocess.run(command, cwd=os.getcwd(), env=environment, check=False)
    training_exit_code = completed.returncode

    records = read_identity_file(identity_file)
    if not records:
        print(
            "the training command exited {} without reporting a run identity, so "
            "no evidence can be staged".format(training_exit_code),
            file=sys.stderr,
        )
        return training_exit_code or 1
    identity = records[-1]

    if training_exit_code != 0:
        write_manifest(
            plan, stage=args.stage, arm=arm, seed=args.seed, identity=identity,
            staged=[], outputs_dir=outputs_dir,
            training_exit_code=training_exit_code,
            embedding_layout=layout_summary,
        )
        print(
            "training failed with exit {}; no outputs were staged".format(
                training_exit_code
            ),
            file=sys.stderr,
        )
        return training_exit_code

    staged, missing = stage_outputs(plan, arm, args.seed, identity, outputs_dir)
    write_manifest(
        plan, stage=args.stage, arm=arm, seed=args.seed, identity=identity,
        staged=staged, outputs_dir=outputs_dir, training_exit_code=0,
        embedding_layout=layout_summary,
    )
    if missing:
        for item in missing:
            print("required output not staged: {}".format(item), file=sys.stderr)
        return 3
    print(
        "staged {} output(s) for {} {} seed {}".format(
            len(staged), jobspec.plan_scope(plan), arm["arm"], args.seed
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
