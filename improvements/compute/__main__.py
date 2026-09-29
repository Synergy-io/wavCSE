"""Command line for the compute backend.

    python -m improvements.compute status          --scope TR-0007 [--json]
    python -m improvements.compute resolve         [--json]
    python -m improvements.compute envelope-check  --scope TR-0007 --action submit-job
    python -m improvements.compute sweep           --scope TR-0007 [--execute]
    python -m improvements.compute plan            --scope TR-0007 --plan P --stage screen
    python -m improvements.compute worker-ensure   --scope TR-0007 --plan P
    python -m improvements.compute advance         --scope TR-0007 --plan P --stage screen [--dry-run]
    python -m improvements.compute collect         --scope TR-0007 --plan P --stage screen
    python -m improvements.compute finish          --scope TR-0007 --plan P
    python -m improvements.compute stop            --scope TR-0007 [--destroy]

Exit codes: 0 proceed, 2 usage/configuration failure, 3 refused by policy.
Read-only verbs (`status`, `resolve`, `envelope-check`, `plan`, `--dry-run`)
never write runtime state.
"""

import argparse
import contextlib
import os
import shutil
import sys

from improvements.compute import cli_support, envelope as envelope_module, jobspec
from improvements.compute import ledger, run_study, state as state_module, status as status_module
from improvements.compute import sweep as sweep_module
from improvements.compute import worker as worker_module
from improvements.compute.errors import ComputeError, ConfigurationError, UsageError

EXIT_OK = cli_support.EXIT_OK
EXIT_ERROR = cli_support.EXIT_ERROR
EXIT_REFUSED = cli_support.EXIT_REFUSED


def _json_flag():
    """A shared ``--json`` flag that never overwrites an outer parse.

    ``default=SUPPRESS`` matters: without it, a subparser's default would
    overwrite a ``--json`` given before the verb.
    """

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--json", action="store_true", dest="as_json",
                        default=argparse.SUPPRESS)
    return parent


def _parser():
    parser = argparse.ArgumentParser(
        prog="python -m improvements.compute",
        description="Autonomous Research Control compute backend",
    )
    parser.add_argument("--json", action="store_true", dest="as_json",
                        default=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="verb", required=True)
    flag = _json_flag()

    status = subparsers.add_parser("status", help="read-only compute picture",
                                   parents=[flag])
    status.add_argument("--scope", required=True)

    subparsers.add_parser("resolve", help="locate the infrastructure control plane",
                          parents=[flag])

    envelope_check = subparsers.add_parser("envelope-check", help="evaluate one action",
                                           parents=[flag])
    envelope_check.add_argument("--scope", required=True)
    envelope_check.add_argument("--action", required=True,
                                choices=[envelope_module.ACTION_CREATE_WORKER,
                                         envelope_module.ACTION_SUBMIT_JOB,
                                         envelope_module.ACTION_ATTACH_VOLUME,
                                         envelope_module.ACTION_CREATE_VOLUME,
                                         envelope_module.ACTION_STOP_WORKER,
                                         envelope_module.ACTION_DESTROY_WORKER])

    sweep = subparsers.add_parser("sweep", parents=[flag],
                                  help="reconcile scope compute (dry-run by default)")
    sweep.add_argument("--scope", required=True)
    sweep.add_argument("--execute", action="store_true",
                       help="perform the reversible actions (stop); never destroys")

    plan_cmd = subparsers.add_parser("plan", help="show what a stage would submit",
                                     parents=[flag])
    plan_cmd.add_argument("--scope", required=True)
    plan_cmd.add_argument("--plan", required=True)
    plan_cmd.add_argument("--stage", required=True)

    worker_ensure = subparsers.add_parser("worker-ensure", help="ensure a ready worker",
                                          parents=[flag])
    worker_ensure.add_argument("--scope", required=True)
    worker_ensure.add_argument("--plan", required=True)

    advance = subparsers.add_parser("advance", help="one stage transition",
                                    parents=[flag])
    advance.add_argument("--scope", required=True)
    advance.add_argument("--plan", required=True)
    advance.add_argument("--stage", required=True)
    advance.add_argument("--dry-run", action="store_true")

    collect = subparsers.add_parser("collect", help="verify outputs of a stage",
                                    parents=[flag])
    collect.add_argument("--scope", required=True)
    collect.add_argument("--plan", required=True)
    collect.add_argument("--stage", required=True)

    finish = subparsers.add_parser("finish", help="stop or destroy scope compute",
                                   parents=[flag])
    finish.add_argument("--scope", required=True)
    finish.add_argument("--plan", required=True)

    stop = subparsers.add_parser("stop", help="stop scope compute", parents=[flag])
    stop.add_argument("--scope", required=True)
    stop.add_argument("--destroy", action="store_true")

    reap = subparsers.add_parser(
        "reap",
        parents=[flag],
        help="crash-independent reconciliation of every known scope (dry-run by default)",
    )
    reap.add_argument("--execute", action="store_true",
                      help="perform the reversible actions (stop); never destroys")
    reap.add_argument("--scope", action="append", default=None,
                      help="limit to one scope (repeatable); default is every known scope")

    reap_install = subparsers.add_parser(
        "reap-install",
        parents=[flag],
        help="render or install the controller timer that runs `reap` unattended",
    )
    reap_install.add_argument("--print", action="store_true",
                              help="render the unit files to stdout (the default)")
    reap_install.add_argument("--install", action="store_true",
                              help="write the unit files into the unit directory")
    reap_install.add_argument("--enable", action="store_true",
                              help="also enable and start the timer (writes to systemd)")
    reap_install.add_argument("--user", action="store_true",
                              help="install user units instead of system units")
    reap_install.add_argument("--dir", default=None,
                              help="explicit unit directory (defaults to the systemd one)")
    reap_install.add_argument("--interval-seconds", type=int, default=None)
    reap_install.add_argument("--dry-run", action="store_true",
                              help="report what --install would write without writing")

    preflight = subparsers.add_parser(
        "preflight", parents=[flag],
        help="read-only: prove this checkout's exact commit is publishable to workers",
    )
    preflight.add_argument("--plan", required=True)
    preflight.add_argument("--commit", default=None,
                           help="exact commit to check (defaults to this checkout's HEAD)")

    return parser


def _infra():
    from improvements.compute import infra_cli, resolve as resolve_module

    return infra_cli.InfraCli(resolve_module.resolve())


@contextlib.contextmanager
def _mutation(scope):
    """The guards around a mutating verb: one host controller, then one scope.

    Read-only verbs never take either, so a status check can always run. The
    host-level guard is what makes "one controller per host" a checked invariant rather
    than an assumption — it refuses a second mutating process even when it was handed a
    different state root, because nothing inside a state root can see a second one. The
    order (host, then scope) is the same everywhere, so the two locks cannot deadlock.
    """

    with state_module.controller_guard():
        with state_module.scope_lock(scope):
            yield


def _emit(args, payload, exit_code=EXIT_OK, lines=None):
    return cli_support.emit(payload, as_json=getattr(args, "as_json", False),
                            exit_code=exit_code, lines=lines)


def _run(args):
    if args.verb == "resolve":
        from improvements.compute import resolve as resolve_module

        return _emit(args, resolve_module.resolve().as_dict())

    if args.verb == "status":
        picture = status_module.build(args.scope)
        return _emit(args, picture)

    if args.verb == "envelope-check":
        view = envelope_module.load(args.scope)
        try:
            infra = _infra()
            facts = ledger.derive_spend(infra.worker_list(), args.scope).facts()
        except ComputeError as exc:
            return _emit(args, {"allowed": False, "class": "TRANSIENT_INFRA",
                                "reason": "provider facts unavailable: {}".format(exc),
                                "action": args.action}, EXIT_REFUSED)
        requested = None
        if args.action == envelope_module.ACTION_CREATE_WORKER:
            requested = {
                "hourly_usd": str(view.envelope["budget"]["max_gpu_hourly_usd"]),
                "projected_hours": "1",
                "container_disk_gb": view.envelope["resources"]["container_disk_gb_max"],
            }
        decision = envelope_module.check(view, args.action, facts, requested=requested)
        return _emit(args, decision.as_dict(),
                     EXIT_OK if decision.allowed else EXIT_REFUSED)

    if args.verb == "sweep":
        infra = _infra()
        if args.execute:
            with _mutation(args.scope):
                result = sweep_module.sweep(infra, args.scope, execute=True)
        else:
            # A dry run is observational, so it never contends for the lock.
            result = sweep_module.sweep(infra, args.scope, execute=False)
        return _emit(args, result)

    plan = jobspec.load_plan(args.plan) if getattr(args, "plan", None) else None

    if args.verb == "plan":
        record = run_study.load_record(args.scope)
        view = envelope_module.load(args.scope)
        result = run_study.submit_pending(
            args.scope, plan, args.stage, infra=_infra(), view=view, record=record,
            dry_run=True,
        )
        result["envelope"] = view.as_dict()
        return _emit(args, result)

    if args.verb == "worker-ensure":
        infra = _infra()
        view = envelope_module.load(args.scope)
        with _mutation(args.scope):
            worker, actions = worker_module.ensure_worker(plan, view, infra=infra)
        return _emit(args, {
            "worker": {key: worker.get(key) for key in
                       ("id", "name", "state", "hourly_cost", "gpu_type")},
            "actions": actions,
            "envelope_digest": view.digest,
        })

    if args.verb == "advance":
        infra = _infra()
        record = run_study.load_record(args.scope)
        if args.dry_run:
            result = run_study.advance(
                args.scope, plan, args.stage, infra=infra, record=record, dry_run=True,
            )
        else:
            with _mutation(args.scope):
                result = run_study.advance(
                    args.scope, plan, args.stage, infra=infra, record=record,
                )
        return _emit(args, result)

    if args.verb == "collect":
        infra = _infra()
        record = run_study.load_record(args.scope)
        with _mutation(args.scope):
            result = run_study.collect(args.scope, plan, args.stage, record=record,
                                       reader=infra)
        return _emit(args, result,
                     EXIT_OK if not result["unverified"] and not result["invalid"]
                     else EXIT_REFUSED)

    if args.verb == "finish":
        infra = _infra()
        view = envelope_module.load(args.scope)
        record = run_study.load_record(args.scope)
        with _mutation(args.scope):
            result = run_study.finish(args.scope, plan, infra=infra, view=view,
                                      record=record)
        return _emit(args, result)

    if args.verb == "preflight":
        from improvements.compute import remote_commit

        commit = args.commit or jobspec.git_state()["head"]
        try:
            availability = remote_commit.verify_available(
                plan.get("repository"), commit
            )
        except ComputeError as exc:
            return _emit(args, {"commit": commit, "available": False,
                                "class": type(exc).__name__, "reason": str(exc)},
                         exit_code=EXIT_REFUSED)
        payload = availability.as_dict()
        if not availability.available:
            payload["class"] = "REPOSITORY_CONFLICT"
            payload["reason"] = (
                "the commit is not on the remote a disposable worker clones; publishing "
                "it is a developer action (`git push`), and no substitute commit is "
                "chosen automatically"
            )
        return _emit(args, payload,
                     exit_code=EXIT_OK if availability.available else EXIT_REFUSED)

    if args.verb == "reap":
        from improvements.compute import reaper

        infra = _infra()
        report = reaper.reap(infra, execute=args.execute, scopes=args.scope)
        return _emit(args, report, exit_code=reaper.exit_code(report),
                     lines=reaper.summarize(report))

    if args.verb == "reap-install":
        from improvements.compute import reaper_units
        from improvements.compute import resolve as resolve_module

        uv = shutil.which("uv")
        if not uv:
            raise ConfigurationError(
                "`uv` is not on PATH, so the reaper timer cannot be rendered with a "
                "deterministic interpreter; run this from the controller bootstrap shell"
            )
        checkout = resolve_module.resolve().checkout
        home = os.path.expanduser("~")
        interval = args.interval_seconds or reaper_units.DEFAULT_INTERVAL_SECONDS
        units = reaper_units.build_units(
            repo_root=resolve_module.repo_root(),
            command_argv=[uv, "run", "--locked", "python", "-m",
                          "improvements.compute", "reap", "--execute"],
            state_root=state_module.state_root(),
            infra_checkout=checkout,
            interval_seconds=interval,
            run_as=None if args.user else os.environ.get("USER") or None,
            extra_environment=(("HOME", home),),
        )
        if not args.install:
            payload = {"dry_run": True, "unit_dir": None, "units": units,
                       "interval_seconds": interval}
            return _emit(args, payload)
        target = args.dir or reaper_units.default_unit_dir(user=args.user)
        result = reaper_units.install(units, target, dry_run=args.dry_run)
        payload = {"units": units, "interval_seconds": interval, **result}
        if args.enable:
            if args.dry_run:
                payload["enable"] = {"enabled": False, "reason": "--dry-run"}
            else:
                payload["enable"] = reaper_units.enable(user=args.user)
        return _emit(args, payload)

    if args.verb == "stop":
        infra = _infra()
        actions = []
        with _mutation(args.scope):
            leased_ids = {lease.get("worker_id") for lease in ledger.active_leases(args.scope)}
            open_jobs = [job for job in infra.job_list()
                         if (job.get("worker_id") in leased_ids or
                             ((job.get("spec") or {}).get("tracking") or {}).get("metadata", {}).get("scope") == args.scope)
                         and str(job.get("state") or "").upper() in run_study.OPEN_STATES]
            if open_jobs:
                raise UsageError("scope has open jobs; reconcile or cancel them before stopping compute")
            if args.destroy:
                entries = list(run_study.load_record(args.scope)["entries"].values())
                if not entries or any(entry.get("state") != run_study.COLLECTED for entry in entries):
                    raise UsageError("destruction requires every recorded run to have verified outputs")
            for lease in ledger.active_leases(args.scope):
                worker_id = lease.get("worker_id")
                if args.destroy:
                    result = worker_module.destroy_worker(infra, worker_id, reason="operator stop")
                    actions.append({"worker_id": worker_id, "action": "destroy"})
                else:
                    result = worker_module.stop_worker(infra, worker_id, reason="operator stop")
                    actions.append({"worker_id": worker_id, "action": "stop"})
                if result.returncode != 0:
                    raise UsageError("worker {} cleanup failed; reconcile provider state"
                                     .format(worker_id))
        return _emit(args, {"scope": args.scope, "actions": actions})

    raise UsageError("unhandled verb {!r}".format(args.verb))


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    as_json = getattr(args, "as_json", False)
    try:
        return _run(args)
    except (UsageError, ConfigurationError) as exc:
        return cli_support.emit(
            {"error": type(exc).__name__, "reason": str(exc)},
            as_json=as_json, exit_code=EXIT_ERROR,
        )
    except ComputeError as exc:
        return cli_support.emit(
            {"error": type(exc).__name__, "reason": str(exc)},
            as_json=as_json, exit_code=EXIT_REFUSED,
        )


if __name__ == "__main__":
    sys.exit(main())
