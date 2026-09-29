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

    return parser


def _infra():
    from improvements.compute import infra_cli, resolve as resolve_module

    return infra_cli.InfraCli(resolve_module.resolve())


def _scope_lock(scope):
    """Exclusive guard for one scope's mutating verbs.

    Read-only verbs never take it, so a status check can always run; two
    orchestrator instances contend and the loser refuses rather than acting.
    """

    return state_module.scope_lock(scope)


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
            with _scope_lock(args.scope):
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
        with _scope_lock(args.scope):
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
            with _scope_lock(args.scope):
                result = run_study.advance(
                    args.scope, plan, args.stage, infra=infra, record=record,
                )
        return _emit(args, result)

    if args.verb == "collect":
        record = run_study.load_record(args.scope)
        with _scope_lock(args.scope):
            result = run_study.collect(args.scope, plan, args.stage, record=record)
        return _emit(args, result,
                     EXIT_OK if not result["unverified"] else EXIT_REFUSED)

    if args.verb == "finish":
        infra = _infra()
        view = envelope_module.load(args.scope)
        record = run_study.load_record(args.scope)
        with _scope_lock(args.scope):
            result = run_study.finish(args.scope, plan, infra=infra, view=view,
                                      record=record)
        return _emit(args, result)

    if args.verb == "stop":
        infra = _infra()
        actions = []
        with _scope_lock(args.scope):
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
