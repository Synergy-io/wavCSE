"""Command line for layer-combination sweeps.

    python -m improvements.sweep plan    --spec <study>/sweep.json
    python -m improvements.sweep configs --spec <study>/sweep.json
    python -m improvements.sweep push    --spec <study>/sweep.json
    python -m improvements.sweep start   --spec <study>/sweep.json --stage screen
    python -m improvements.sweep status  --spec <study>/sweep.json
    python -m improvements.sweep report  --spec <study>/sweep.json

``plan`` is dry-run and read-only: it is what an operator (or an agent) reads
before spending anything. ``configs`` writes the committed per-combo configs and
refuses a dirty checkout, because a sweep's configs are the record of what ran.
``start`` launches the supervisor -- locally, or on the hiring VM's pod when the
spec names a ``remote.host`` -- and every other verb reads that work back.
"""

import argparse
import json
import os
import shlex
import sys

from improvements.sweep import (config_gen, ledger, manifest, report, resources,
                                supervisor)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3


def _load(args, *, check_combos=True):
    return manifest.load_spec(args.spec, check_combos=check_combos)


def _remote_host(spec, args):
    override = getattr(args, "host", None)
    if override:
        return override
    return manifest.validate_remote_host(spec)


def _remote_checkout(spec):
    remote = spec.get("remote") or {}
    checkout = remote.get("checkout")
    if not checkout:
        raise manifest.SpecError(
            "remote.checkout is not set in the spec; a remote run has no working "
            "directory without it")
    return checkout


def _study_relpath(spec):
    return manifest.study_relpath(spec)


def cmd_plan(args):
    spec = _load(args)
    stages = [args.stage] if args.stage else sorted(spec["stages"])
    policy = spec["policy"]
    per_run = float(args.per_run_minutes) * 60.0
    if args.json:
        print(json.dumps({
            "study_id": spec["study_id"], "sweep_id": spec["sweep_id"],
            "experiment_name": spec["experiment_name"],
            "combos": [
                dict(combo, layer_count=len(manifest.parse_layers(
                    combo["layers"], spec["upstream_model_type"])))
                for combo in spec["combos"]
            ],
            "stages": {stage: len(manifest.stage_jobs(spec, stage)) for stage in stages},
            "policy": policy,
            "jobs_total": sum(len(manifest.stage_jobs(spec, stage)) for stage in stages),
        }, sort_keys=True, indent=2))
        return EXIT_OK
    print("study        : {}".format(spec["study_id"]))
    print("sweep        : {}".format(spec["sweep_id"]))
    print("experiment   : {}".format(spec["experiment_name"]))
    print("model        : {}".format(spec["run_model"]))
    print("template     : {}".format(os.path.relpath(spec["config_template"],
                                                     manifest.repo_root(spec))))
    print("combos       : {} declared".format(len(spec["combos"])))
    for combo in spec["combos"]:
        try:
            count = len(manifest.parse_layers(combo["layers"],
                                              spec["upstream_model_type"]))
            count_text = "{} layers".format(count)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            count_text = "INVALID ({})".format(exc)
        print("   {:<12}{:<28}k={:<3}{:<10}{}".format(
            manifest.combo_id(combo)[:12], str(combo.get("group") or "-")[:28],
            str(combo.get("k") if combo.get("k") is not None else "-"),
            count_text, str(combo["layers"])[:60]))
    total = 0
    for stage in stages:
        jobs = manifest.stage_jobs(spec, stage)
        total += len(jobs)
        print("stage {:<10}: {} jobs ({} combos x {} seeds), serial {:.1f} h, "
              "at {:.0f}x concurrency {:.1f} h".format(
                  stage, len(jobs), len(manifest.stage_combos(spec, stage)),
                  len(spec["stages"][stage]["seeds"]),
                  len(jobs) * per_run / 3600.0, args.assumed_concurrency,
                  len(jobs) * per_run / 3600.0 / args.assumed_concurrency))
    print("policy       : max_concurrent={} ceiling={} vram/run={}GB ram/run={}GB "
          "ram_reserve={}GB cores/run={} disk_min={}GB workers/run={}".format(
              policy["max_concurrent"], policy["ceiling"], policy["vram_per_run_gb"],
              policy["ram_per_run_gb"], policy["ram_reserve_gb"],
              policy["cpu_cores_per_run"], policy["min_free_disk_gb"],
              policy["workers_per_run"]))
    if policy.get("concurrency_basis"):
        print("override     : {}".format(policy["concurrency_basis"]))
    print("wall budget  : {:.1f} h per supervisor run".format(
        policy["max_wall_seconds"] / 3600.0))
    print("jobs total   : {}".format(total))
    return EXIT_OK


def cmd_configs(args):
    spec = _load(args)
    repo = manifest.repo_root(spec)
    state = manifest.git_state(repo)
    if state["dirty"] and not args.allow_dirty:
        print("refusing to generate configs from a dirty checkout: the generated "
              "configs are the committed record of what ran. Commit or stash "
              "first, or pass --allow-dirty.", file=sys.stderr)
        for line in state["dirty_entries"][:10]:
            print("  {}".format(line), file=sys.stderr)
        return EXIT_USAGE
    stages = args.stage or sorted(spec["stages"])
    written = []
    for stage in stages:
        if stage not in spec["stages"]:
            print("unknown stage {!r}".format(stage), file=sys.stderr)
            return EXIT_USAGE
        for path in config_gen.write_stage(spec, stage, force=args.force):
            written.append(path)
            print("wrote {}".format(os.path.relpath(path, repo)))
    print("{} config(s) at commit {}".format(len(written), state["head"][:12]))
    return EXIT_OK


def cmd_push(args):
    from improvements.sweep import remote
    spec = _load(args)
    host = _remote_host(spec, args)
    if not host:
        print("no remote host (set remote.host in the spec or pass --host)",
              file=sys.stderr)
        return EXIT_USAGE
    checkout = _remote_checkout(spec)
    study_rel = _study_relpath(spec)
    remote.rsync(host, os.path.join(spec["_study_dir"], "") ,
                 "{}/{}/".format(checkout, study_rel))
    print("pushed {} to {}:{}".format(spec["sweep_id"], host,
                                      "{}/{}".format(checkout, study_rel)))
    return EXIT_OK


def _state_read(spec, host, state_dir):
    """Read the sweep summary and ledger, locally or from the pod."""
    if host:
        from improvements.sweep import remote
        summary_path = "{}/SWEEP.json".format(state_dir)
        ledger_path = "{}/runs.jsonl".format(state_dir)
        summary = remote.read_json(host, summary_path) if remote.exists(
            host, summary_path) else None
        records = []
        if remote.exists(host, ledger_path):
            completed = remote.ssh(host, "cat {}".format(shlex.quote(ledger_path)))
            for line in completed.stdout.decode("utf-8", "replace").splitlines():
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except ValueError:
                        continue
        return summary, records
    summary_path = os.path.join(state_dir, "SWEEP.json")
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
    return summary, ledger.read(ledger.path_for(state_dir))


def cmd_status(args):
    spec = _load(args)
    host = _remote_host(spec, args)
    state_dir = args.state_dir or manifest.state_dir(spec)
    summary, records = _state_read(spec, host, state_dir)
    latest = ledger.index(records)
    counts = ledger.counts(records)
    running = sorted(key for key, record in latest.items()
                     if record.get("state") in (ledger.RUNNING, ledger.LAUNCHING))
    payload = {
        "study_id": spec["study_id"], "sweep_id": spec["sweep_id"],
        "host": host, "state_dir": state_dir, "counts": counts,
        "running": running, "records": len(latest),
        "summary": summary, "drain": os.path.exists(os.path.join(state_dir, "DRAIN")),
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True, indent=2))
        return EXIT_OK
    print("sweep {} on {}".format(spec["sweep_id"], host or "this host"))
    print("state        : {}".format(state_dir))
    print("jobs recorded: {}".format(len(latest)))
    for state_name in sorted(counts):
        print("   {:<10} {}".format(state_name, counts[state_name]))
    if running:
        print("in flight    : {}".format(", ".join(running)))
    if summary:
        print("last summary : {} stage {} at {} ({:.0f}s, {})".format(
            summary.get("written_at"), summary.get("stage"),
            str(summary.get("commit"))[:12], summary.get("elapsed_seconds") or 0,
            summary.get("counts")))
    return EXIT_OK


def cmd_logs(args):
    spec = _load(args)
    host = _remote_host(spec, args)
    state_dir = args.state_dir or manifest.state_dir(spec)
    path = os.path.join(state_dir, "logs", "{}.log".format(args.run))
    if host:
        from improvements.sweep import remote
        command = "tail -n {} {}".format(int(args.tail), shlex.quote(path))
        completed = remote.ssh(host, command, check=False)
        sys.stdout.write(completed.stdout.decode("utf-8", "replace"))
        return EXIT_OK if completed.returncode == 0 else EXIT_UNAVAILABLE
    if not os.path.exists(path):
        print("no log at {}".format(path), file=sys.stderr)
        return EXIT_UNAVAILABLE
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()
    for line in lines[-int(args.tail):]:
        print(line)
    return EXIT_OK


def cmd_stop(args):
    spec = _load(args)
    host = _remote_host(spec, args)
    state_dir = args.state_dir or manifest.state_dir(spec)
    content = "stop requested by operator\n"
    if host:
        from improvements.sweep import remote
        remote.write_remote_file(host, os.path.join(state_dir, "DRAIN"), content)
    else:
        os.makedirs(state_dir, exist_ok=True)
        with open(os.path.join(state_dir, "DRAIN"), "w", encoding="utf-8") as handle:
            handle.write(content)
    print("drain requested; running jobs are left to finish, nothing new starts")
    return EXIT_OK


def cmd_pull(args):
    from improvements.sweep import remote
    spec = _load(args)
    host = _remote_host(spec, args)
    if not host:
        print("nothing to pull: the sweep runs on this host", file=sys.stderr)
        return EXIT_USAGE
    checkout = _remote_checkout(spec)
    study_rel = _study_relpath(spec)
    local_study = spec["_study_dir"]
    for name in ("sweep_state", "outputs", "configs"):
        remote.rsync(host, os.path.join(checkout, study_rel, name) + "/",
                     os.path.join(local_study, name) + "/", source_is_remote=True)
        print("pulled {}".format(name))
    return EXIT_OK


def cmd_start(args):
    spec = _load(args)
    host = _remote_host(spec, args)
    stage = args.stage
    if stage not in spec["stages"]:
        print("unknown stage {!r}; declared: {}".format(
            stage, ", ".join(sorted(spec["stages"]))), file=sys.stderr)
        return EXIT_USAGE
    state_dir = args.state_dir or manifest.state_dir(spec)
    if host:
        from improvements.sweep import remote
        checkout = _remote_checkout(spec)
        python = shlex.split((spec.get("remote") or {}).get("python")
                             or "uv run --locked python")
        argv = list(python) + ["-m", "improvements.sweep.supervisor",
                               "--spec", _study_relpath(spec) + "/sweep.json",
                               "--stage", stage,
                               "--state-dir", state_dir,
                               "--checkout", checkout]
        if args.once:
            argv.append("--once")
        if args.dry_run:
            argv.append("--dry-run")
        if args.allow_dirty:
            argv.append("--allow-dirty")
        log_path = os.path.join(state_dir, "supervisor.log")
        remote.ssh(host, "mkdir -p {}".format(shlex.quote(state_dir)))
        remote.start_detached(host, checkout, argv, log_path)
        print("supervisor started on {} (log {})".format(host, log_path))
        return EXIT_OK
    argv = ["--spec", args.spec, "--stage", stage, "--state-dir", state_dir]
    if args.once:
        argv.append("--once")
    if args.dry_run:
        argv.append("--dry-run")
    if args.allow_dirty:
        argv.append("--allow-dirty")
    return supervisor.main(argv)


def cmd_report(args):
    spec = _load(args)
    try:
        rows = report.collect(spec, tracking_uri=args.tracking_uri)
    except report.ReportError as exc:
        print("cannot read the sweep from MLflow: {}".format(exc), file=sys.stderr)
        return EXIT_UNAVAILABLE
    if not rows:
        print("no runs with sweep_id={!r} in experiment {!r} yet".format(
            spec["sweep_id"], spec["experiment_name"]))
        return EXIT_OK
    if args.json:
        print(report.as_json(rows, metric=args.metric))
    else:
        print(report.render(rows, metric=args.metric, top=args.top))
        print()
        print("per-combo peak:")
        for row in report.by_combo(rows, metric=args.metric):
            print("   {:<52} {:.5f} (seed {})".format(
                str(row["combo"])[:52], row[args.metric], row["seed"]))
    return EXIT_OK


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m improvements.sweep",
        description="Declare and run a layer-combination sweep.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, help_text):
        child = sub.add_parser(name, help=help_text)
        child.add_argument("--spec", required=True, help="path of sweep.json")
        child.add_argument("--stage", default=None)
        child.add_argument("--state-dir", default=None)
        child.add_argument("--host", default=None,
                           help="override spec.remote.host")
        return child

    plan = add("plan", "dry-run: what the sweep would run, and the policy")
    plan.add_argument("--json", action="store_true")
    plan.add_argument("--per-run-minutes", type=float, default=19.0)
    plan.add_argument("--assumed-concurrency", type=float, default=4.0)
    plan.set_defaults(func=cmd_plan)

    configs = add("configs", "write the committed per-combo configs")
    configs.add_argument("--force", action="store_true")
    configs.add_argument("--allow-dirty", action="store_true")
    configs.set_defaults(func=cmd_configs)

    push = add("push", "copy the study directory to the remote checkout")
    push.set_defaults(func=cmd_push)

    start = add("start", "run the supervisor (locally, or on spec.remote.host)")
    start.add_argument("--once", action="store_true")
    start.add_argument("--dry-run", action="store_true")
    start.add_argument("--allow-dirty", action="store_true")
    start.set_defaults(func=cmd_start)

    status = add("status", "ledger and summary, from the pod or locally")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    logs = add("logs", "tail one run's log")
    logs.add_argument("--run", required=True, help="job key (see `status`)")
    logs.add_argument("--tail", type=int, default=200)
    logs.set_defaults(func=cmd_logs)

    stop = add("stop", "write the DRAIN flag: admit nothing new")
    stop.set_defaults(func=cmd_stop)

    pull = add("pull", "rsync outputs and sweep_state back from the pod")
    pull.set_defaults(func=cmd_pull)

    report_cmd = add("report", "aggregate the sweep from MLflow")
    report_cmd.add_argument("--metric", default="val_acc_all")
    report_cmd.add_argument("--top", type=int, default=None)
    report_cmd.add_argument("--json", action="store_true")
    report_cmd.add_argument("--tracking-uri", default=None)
    report_cmd.set_defaults(func=cmd_report)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "start" and not args.stage:
        print("start requires --stage", file=sys.stderr)
        return EXIT_USAGE
    try:
        return args.func(args)
    except manifest.SpecError as exc:
        print("sweep spec error: {}".format(exc), file=sys.stderr)
        return EXIT_USAGE
    except (config_gen.ConfigError, OSError) as exc:
        print("sweep error: {}".format(exc), file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
