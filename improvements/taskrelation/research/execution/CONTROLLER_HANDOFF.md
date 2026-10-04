# DEV → CONTROLLER handoff — DG-0008 / Execution Plane V1

Deterministic handoff from the development computer to the controller. It
records only identities and unresolved bindings; it copies no artifact, carries
no credential, and grants no authority. Written on the development machine on
2026-10-04, on top of wavCSE commit `91a8f2edbdbba8060bba7eac381735889a4df14b`;
the controller must check out the commit that contains this file (resolve it with
`git log -1 --format=%H -- improvements/taskrelation/research/execution/CONTROLLER_HANDOFF.md`).

Nothing here authorizes compute. A preflight acceptance is not an authorization,
and an authorization is not a submission.

## wavCSE side

| Item | Value |
| --- | --- |
| Repository | `https://github.com/Synergy-io/wavCSE.git` |
| Branch | `feature/mssl-task-relation-study` |
| Approved proposal | `DP-0008` — `research/proposals/DG-0008_exact_matched_directed_transfer.md`, commit `83694cd965a3045ddf2f0bc8c8c9949a5abfaba2`, SHA-256 `c3548d7f9bae2881dcd833c9ddb19e5abb0df174ae76692dac79ab6b566572bc`, review `PASS`, approved by Kevin Sanjula `<kevinxsanjula@gmail.com>` 2026-10-03T16:18:45+00:00 |
| Registered Study | `DG-0008` — `research/studies/DG-0008/PLAN.md`, `NOTE.md`; registration commit `fa326fc39ca1bc2977d30a0964b80c493571e02b` |
| Implementation identity | `research/dg0008/` plus `studies/DG-0008/{run_dg0008.py,generate_opportunity_manifest.py,evaluate_dg0008.py,check_validity_gate.py,analyze_dg0008.py,configs/dg0008.yaml}` — commit `d9675688cdb3759778214509d3f6e7fa3348f429` |
| Compute inputs | `research/studies/DG-0008/compute/inputs.json` — 15 digest-pinned requirements (5 speechcommand, 9 voxceleb, 1 iemocap), validated by `improvements.compute.artifacts` |
| Compute plan | **Not authored.** Every static field is known, but `worker.gpu_type`/`worker.cloud`/`worker.image` and a bounded `timeout_seconds` are controller-time bindings and must not be invented on the development machine. |
| JobSpec schema | `improvements/compute/jobspec.py`, `SCHEMA_VERSION = 1`; stages may select an arm subset, so the approved 200 + 1 topology is expressible (`improvements/compute/tests/test_stage_arms.py`) |
| Scientific invariants | The sealed workload `execution/preflights/DP-0008/workload.v2.json` (26 invariants, one flexible choice `IF-MANIFEST-IO=STREAM_CANONICAL_BYTES`). Not duplicated here. |
| Execution contract | `improvements/taskrelation/research/execution/README.md`; validator `python -m improvements.taskrelation.research.execution_contract check <dir>` |
| Authorization | **None.** `research/authorizations/DG-0008.yaml` does not exist and must not be created by an agent. |
| Capability boundary | `AGENTS.md` "Capability boundary"; `.omp/config.yml` `task.isolation.{enabled:true,apply:false}`; gate `scripts/agents/candidate_gate.py` |

### Compute-plan fields: intent vs binding

| Field | Kind |
| --- | --- |
| `study`, `arms[].{arm,method,argv,config}`, `stages[].{seeds,arms}`, `outputs`, `inputs_file`, `embedding_layout`, `repository`, `task_type`, `environment_secrets` | static scientific / execution intent |
| `worker.{gpu_type,cloud,image,template,gpu_count,container_disk_gb,volume_gb,data_centers,network_volume}` | controller-time provider binding (offer + cloud tier + volume are discovered at authorization time) |
| `timeout_seconds` | controller-time binding (schema default until a compatible-GPU benchmark bounds it) |
| `device_index` | execution intent with a safe default |

### Required before any DG-0008 job

- Lawful restoration of the Speech Commands, VoxCeleb1 and licence-gated IEMOCAP
  corpora, and the frozen `dg0008.example-identity.v1` manifest + its digest.
- Every identity/opportunity manifest, run-level digest and the matrix digest
  (§5–§14 of the approved proposal), generated after lawful restoration. None
  exists and none may be fabricated.
- A compatible-GPU five-epoch benchmark bounding runtime for all four arm
  classes and the repeat.
- A human-bounded authorization envelope, then a fresh accepted preflight.

## wavcse-infra side

| Item | Value |
| --- | --- |
| Canonical repository | `https://github.com/Ke-vin-S/wavcse-infra.git` |
| Branch / commit | `main` @ `63c61af1ed70cd283b03438ec9ec11cf4166541b` (implementation Phases 0–6.2) |
| Required compatibility | that commit or a later `main`; `improvements.compute` was reconciled against it |
| Provider capability | **RunPod** implemented (REST v2): worker lifecycle, network volumes, S3 artifact storage, exact-commit jobs. **Colab: not present in this commit** — see open decision below. |

### Interface assumptions wavCSE makes

`improvements/compute/infra_cli.py` shells out to the resolved `infra` binary and
parses `--json` output. It assumes:

- `doctor`;
- `worker list --read-only --json`, `worker show <id> --read-only --json`,
  `worker create … --yes`, `worker wait-ssh`, `worker bootstrap`, `worker health`,
  `worker start/stop/destroy`;
- `volume list --read-only --json`, `volume cache stats --worker <id> --json`;
- `job submit <spec> --worker <id> --json`, `job status <id> --json`,
  `job list [--state] [--worker] --json`, `job logs <id> --tail-bytes N`,
  `job cancel <id> --json`;
- `storage read <artifact> --expected-sha256 … --json`, `storage verify`,
  `storage download`, `storage list`;
- the version-1 JobSpec JSON: `source{repository,commit}`, `command{argv}`,
  `setup{argv}`, `runtime{timeout_seconds,environment,environment_secrets}`,
  `inputs[{artifact,destination,required,sha256,size_bytes}]`,
  `outputs[{path,artifact,required,overwrite}]`, `tracking{metadata}`.

Resolution order is `WAVCSE_INFRA_CLI` → `WAVCSE_INFRA_CHECKOUT` → `<repo>/infra`
→ `infra` on `PATH` → legacy sibling `wavcse-infra`. Two hazards:

- the `infra/` **committed inside wavCSE is a superseded fork** at v0.1.1 that
  implements only Phases 0–4 and cannot serve a job; it must not be the resolved
  control plane for DG-0008;
- the local standalone checkout on the development machine
  (`/home/kevin/projects/wavcse-infra` @ `df748a0`, v0.1.0) is **stale**; it also
  predates Phases 5–6.2. Refresh it, or bind explicitly to a verified checkout.

## Controller-only checks

None of these can be performed on the development machine.

1. Check out the exact wavCSE commit that contains this handoff, and confirm
   `make check` is green there.
2. Confirm the project-local Research Computer resolves from the repository —
   agents `.omp/agents/*`, skills `.agents/skills/*`, commands
   `.agents/commands/*`, tools `.omp/tools/*`, policy `.agents/policies/*` — and
   that `task.isolation.enabled` is in place **before** OMP starts (a mid-session
   change does not reach an already-built task schema).
3. Resolve and verify the canonical `wavcse-infra` checkout; confirm `infra
   doctor`, `infra worker list --read-only --json` and `infra job list --json`.
4. **Decide the Colab question.** The human's architecture is Colab primary,
   RunPod secondary, but no Colab provider exists in `wavcse-infra` `main`.
   Establish which repository/commit carries it, or that it is still to be built.
5. Private S3 availability and canonical embedded-object verification against
   `studies/DG-0008/compute/inputs.json`.
6. Lawful raw-corpus availability (Speech Commands, VoxCeleb1, licence-gated
   IEMOCAP) and the independent raw-metadata identity extraction.
7. MLflow/DagsHub credentials; Colab runtime/account state and current resource
   availability; current RunPod price/availability if used.
8. Authorization envelope, compatible-GPU benchmark, and a fresh accepted
   execution preflight before any submission.

## What the controller must not need to copy

The controller obtains from wavCSE Git alone: the Research Executor,
Infrastructure Engineer, Research Designer, Research Reviewer and Literature
Agent definitions, every project skill and command, the project tools, the
capability policy, the candidate gate, and the execution contract. Machine-local
state stays machine-local: credentials, `wavcse-infra` configuration,
authentication and provider/SSH state.

## Open decision

**Colab capability is unresolved.** The approved architecture makes Colab the
primary execution plane, and this handoff cannot verify it: the canonical
`wavcse-infra` `main` (`63c61af`) contains no Colab provider, and no other
reachable repository implements one. Until a human resolves whether Colab exists
elsewhere (private/unpushed) or is still to be implemented, any Colab-first plan
rests on an unverified premise. Carry out every other controller-side check
first; those are independent of this decision.
