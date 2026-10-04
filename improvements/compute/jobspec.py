"""Deterministic, exact-commit job specifications.

One job is one (scope, stage, arm, seed) on one exact commit — the scope being
a research Study or an infrastructure-validation scope, which differ only in
what the job runs and what it promises to produce. Everything about the
resulting document is derived from committed state — the plan file in the
checkout at that commit, the seed set, the arm, and the envelope digest — so
generating the same job twice produces byte-identical bytes. There is no
timestamp, no branch name, and no "latest" anywhere in the payload.

A Study job runs the committed wrapper ``improvements.compute.worker_stage`` so
that what executes is pinned by the SHA the spec names, and the spec names the
*wrapper* rather than a shell string: the training argv lives in the plan file,
where review can see it. An infrastructure-validation job has no research
wrapper to run, so the spec is the plan's own command.

Local validation mirrors the control plane's own rules (full 40/64-hex commit,
anonymous HTTPS remote, argv as a vector, relative paths only, no credentials).
It is a second line of defence, not a replacement: the control plane revalidates
every field when the spec is submitted.
"""

import json
import os
import re
import subprocess

from improvements.compute import providers
from improvements.compute import scopes
from improvements.compute import state as state_module
from improvements.compute.errors import (
    ArtifactIntegrityError,
    ConfigurationError,
    RepositoryConflictError,
)

SCHEMA_VERSION = 1

_COMMIT = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")
_NAME_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
_SECRET_MARKERS = ("X-Amz-", "?Signature=", "&Signature=", "Bearer ", "AKIA",
                   "-----BEGIN", "ssh://")
_PLACEHOLDERS = ("task_type", "config", "device_index", "seed")

_PLAN_KEYS = {"schema_version", "study", "scope", "scope_kind", "repository",
              "task_type", "timeout_seconds", "device_index", "arms", "stages",
              "outputs", "inputs_file", "environment", "environment_secrets",
              "setup_argv", "method", "worker", "embedding_layout", "provider"}
_LAYOUT_KEYS = {"root", "datasets"}
_LAYOUT_DATASET_KEYS = {"dataset", "input", "inputs"}
_ARM_KEYS = {"arm", "method", "argv", "config", "labels"}
# A stage always declares the seeds it runs. It may also select a subset of the
# plan's arms; a stage that declares no `arms` keeps the original behaviour and
# expands every declared arm, so a pre-extension plan expands exactly as before.
_STAGE_KEYS = {"seeds", "arms"}

# Worker intent is split by what it means, not by which provider accepts it:
#   neutral — the execution resource itself, meaningful to every provider;
#   runpod  — RunPod placement, image and storage choices;
#   colab   — nothing beyond the neutral intent: Colab owns its runtime, disk,
#             placement, session identity and CU policy in its own configuration,
#             so a Colab plan must not carry fields it cannot honour.
_WORKER_NEUTRAL_KEYS = {"gpu_type", "gpu_count", "container_disk_gb"}
_WORKER_PROVIDER_KEYS = {
    providers.RUNPOD: {"cloud", "image", "template", "volume_gb", "volume_mount_path",
                      "network_volume", "data_centers"},
    providers.COLAB: set(),
}
_WORKER_KEYS = _WORKER_NEUTRAL_KEYS | set().union(*_WORKER_PROVIDER_KEYS.values())
_VOLUME_KEYS = {"min_size_gb", "datacenter", "volume_type"}
_OUTPUT_KEYS = {"name", "kind", "tag", "required"}
_ALLOWED_OUTPUT_KINDS = ("checkpoint", "results_file", "gradient_diagnostics")

WRAPPER_MODULE = "improvements.compute.worker_stage"
DEFAULT_SETUP_ARGV = ("uv", "sync", "--locked")
DEFAULT_TIMEOUT_SECONDS = 86400
_ALLOWED_RUNTIME_SECRETS = {"MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"}


def _fail(field, message):
    raise ConfigurationError("plan {}: {}".format(field, message))


def _require(condition, field, message):
    if not condition:
        _fail(field, message)


def _check_secret_free(value, field):
    if isinstance(value, str):
        for marker in _SECRET_MARKERS:
            if marker in value:
                raise ConfigurationError(
                    "plan {} contains credential- or URL-shaped text ({!r}); specs "
                    "carry identifiers, never bearer material".format(field, marker)
                )
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_secret_free(item, "{}.{}".format(field, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_secret_free(item, "{}[{}]".format(field, index))


def plan_provider(plan):
    """The execution provider a plan declares, or the documented legacy default.

    A plan written before providers were explicit carries no ``provider`` and is
    a RunPod plan. The rule is deterministic and is never inferred from which
    provider-specific fields happen to be present, so a legacy plan and a plan
    that states ``provider: runpod`` behave identically.
    """

    try:
        return providers.normalize_provider(plan.get("provider"))
    except providers.ProviderValueError as exc:
        _fail("provider", str(exc))


def plan_scope_kind(plan):
    """The execution-scope kind a plan declares.

    A plan written before kinds existed carries none and is a Study plan, so
    every legacy Study plan keeps its exact meaning. An unrecognised kind is
    refused here rather than defaulted.
    """

    try:
        return scopes.normalize_kind(plan.get("scope_kind"))
    except ConfigurationError as exc:
        _fail("scope_kind", str(exc))


def plan_scope(plan):
    """The execution scope a plan is authorized and attributed by.

    A Study plan names its scope with ``study``; an infrastructure-validation
    plan names it with ``scope``. The two spellings are deliberately exclusive
    per kind, so a plan can never carry a scientific Study id *and* a
    non-Study scope id at once and leave a reader to guess which one owns the
    compute.
    """

    if plan_scope_kind(plan) == scopes.STUDY:
        return plan["study"]
    return plan["scope"]


def validate_plan(plan):
    """Validate one compute plan.

    A Study plan and an infrastructure-validation plan share one schema and
    differ only in the kind they declare and the field that carries the scope
    identity, so the shared validation stays shared and the difference stays
    explicit.
    """

    _require(isinstance(plan, dict), "root", "must be a mapping")
    # Keys starting with "_" are internal annotations added by load_plan (the
    # source path), never part of the document a human authors.
    unknown = sorted(key for key in set(plan) - _PLAN_KEYS if not str(key).startswith("_"))
    _require(not unknown, "root", "has unknown key(s): {}".format(", ".join(unknown)))
    _require(plan.get("schema_version") == SCHEMA_VERSION, "schema_version",
             "must be {}".format(SCHEMA_VERSION))
    kind = plan_scope_kind(plan)
    if kind == scopes.STUDY:
        # The scope identity of a Study plan *is* its Study id; `scope` is the
        # non-Study spelling and is never a second way to say the same thing.
        _require(isinstance(plan.get("study"), str) and plan["study"].strip(),
                 "study", "must be a study identifier")
        _require("scope" not in plan, "scope",
                 "is not used by a study plan; the study id is the execution scope")
    else:
        _require(isinstance(plan.get("scope"), str)
                 and scopes.SCOPE_ID.match(plan["scope"]), "scope",
                 "must be an infrastructure-validation scope identifier of the form IN-0001")
        _require("study" not in plan, "study",
                 "is not used by an infrastructure-validation plan; name the scope instead")
    _check_secret_free(plan, "root")
    secret_names = plan.get("environment_secrets", [])
    _require(isinstance(secret_names, list) and all(
        isinstance(name, str) and name in _ALLOWED_RUNTIME_SECRETS
        for name in secret_names), "environment_secrets",
        "may name only the MLflow credentials required by this research workload")
    _require(len(set(secret_names)) == len(secret_names), "environment_secrets",
             "must not repeat a name")
    if kind == scopes.STUDY:
        _require(set(secret_names) == _ALLOWED_RUNTIME_SECRETS, "environment_secrets",
                 "must declare both MLflow credentials for a recorded research job")
    else:
        # An infrastructure-validation job runs no training and reaches no
        # tracker, so it declares no research credential. The control plane only
        # requires the secrets a spec names, so demanding MLflow here would put
        # a research credential on a job that must not carry one.
        _require(not secret_names, "environment_secrets",
                 "an infrastructure-validation job declares no research credentials")

    repository = plan.get("repository", "")
    _require(repository.startswith("https://"), "repository",
             "must be an anonymous https:// URL so a worker needs no Git credential")
    _require("@" not in repository.split("/")[2], "repository",
             "must not embed credentials")
    _require(isinstance(plan.get("task_type"), str) and plan["task_type"],
             "task_type", "must be a non-empty task string")

    timeout = plan.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    _require(isinstance(timeout, int) and not isinstance(timeout, bool) and 0 < timeout <= 604800,
             "timeout_seconds", "must be an integer in 1..604800")

    device_index = plan.get("device_index", 0)
    _require(isinstance(device_index, int) and not isinstance(device_index, bool)
             and device_index >= 0, "device_index", "must be a non-negative integer")

    arms = plan.get("arms")
    _require(isinstance(arms, list) and arms, "arms", "must be a non-empty list")
    seen = set()
    for index, arm in enumerate(arms):
        field = "arms[{}]".format(index)
        _require(isinstance(arm, dict), field, "must be a mapping")
        unknown = sorted(set(arm) - _ARM_KEYS)
        _require(not unknown, field, "has unknown key(s): {}".format(", ".join(unknown)))
        _require(isinstance(arm.get("arm"), str) and _NAME_SAFE.match(arm["arm"]),
                 field + ".arm", "must be a simple name")
        _require(arm["arm"] not in seen, field + ".arm",
                 "duplicates arm name {!r}".format(arm.get("arm")))
        seen.add(arm["arm"])
        argv = arm.get("argv")
        _require(isinstance(argv, list) and argv, field + ".argv",
                 "must be a non-empty argument vector (a shell string is not representable)")
        for token in argv:
            _require(isinstance(token, str) and token and "\x00" not in token,
                     field + ".argv", "every token must be a non-empty string")
        _require(any("{seed}" in token for token in argv), field + ".argv",
                 "must consume {seed}; a job that ignores its seed is not reproducible")
        for token in argv:
            for placeholder in re.findall(r"\{([a-z_]+)\}", token):
                _require(placeholder in _PLACEHOLDERS, field + ".argv",
                         "uses unknown placeholder {{{}}}".format(placeholder))
        if "{config}" in " ".join(argv):
            _require(isinstance(arm.get("config"), str) and arm["config"],
                     field + ".config", "is required because argv uses {config}")

    # Stages are validated after arms so a stage's optional `arms` list can be
    # checked against the names the plan actually declares. A stage that omits
    # `arms` keeps the original behaviour: every declared arm under that stage's
    # seeds. A stage that selects a subset runs only those arms — the bounded
    # extension that lets a plan say "200 matrix jobs plus exactly one repeat"
    # without encoding experiment topology in worker concurrency.
    stages = plan.get("stages")
    _require(isinstance(stages, dict) and stages, "stages", "must be a non-empty mapping")
    for stage, spec in stages.items():
        field = "stages.{}".format(stage)
        _require(isinstance(spec, dict), field, "must be a mapping")
        unknown = sorted(set(spec) - _STAGE_KEYS)
        _require(not unknown, field, "has unknown key(s): {}".format(", ".join(unknown)))
        seeds = spec.get("seeds")
        _require(isinstance(seeds, list) and seeds, field + ".seeds", "must be a non-empty list")
        for seed in seeds:
            _require(isinstance(seed, int) and not isinstance(seed, bool),
                     field + ".seeds", "seeds must be integers")
        selected = spec.get("arms")
        if selected is not None:
            _require(isinstance(selected, list) and selected, field + ".arms",
                     "must be a non-empty list of arm names")
            for name in selected:
                _require(isinstance(name, str) and name in seen, field + ".arms",
                         "references unknown arm {!r}; arms are {}".format(
                             name, ", ".join(sorted(seen))))
            _require(len(set(selected)) == len(selected), field + ".arms",
                     "must not repeat an arm name")

    outputs = plan.get("outputs")
    _require(isinstance(outputs, list) and outputs, "outputs", "must be a non-empty list")
    for index, output in enumerate(outputs):
        field = "outputs[{}]".format(index)
        _require(isinstance(output, dict), field, "must be a mapping")
        unknown = sorted(set(output) - _OUTPUT_KEYS)
        _require(not unknown, field, "has unknown key(s): {}".format(", ".join(unknown)))
        _require(isinstance(output.get("name"), str) and _NAME_SAFE.match(output["name"]),
                 field + ".name", "must be a simple name")
        _require(output.get("kind") in _ALLOWED_OUTPUT_KINDS, field + ".kind",
                 "must be one of {}".format(", ".join(_ALLOWED_OUTPUT_KINDS)))
        if output["kind"] == "checkpoint":
            _require(output.get("tag") in ("best", "opt", "epoch"), field + ".tag",
                     "a checkpoint output must name its tag")

    layout = plan.get("embedding_layout")
    if layout is not None:
        _require(isinstance(layout, dict), "embedding_layout", "must be a mapping")
        unknown = sorted(set(layout) - _LAYOUT_KEYS)
        _require(not unknown, "embedding_layout",
                 "has unknown key(s): {}".format(", ".join(unknown)))
        missing = sorted(_LAYOUT_KEYS - set(layout))
        _require(not missing, "embedding_layout",
                 "is missing key(s): {}".format(", ".join(missing)))
        root = layout["root"]
        _require(isinstance(root, str) and _NAME_SAFE.match(root)
                 and "/" not in root and root not in (".", ".."),
                 "embedding_layout.root",
                 "must be a single safe directory name inside the job workspace")
        _require(isinstance(plan.get("inputs_file"), str) and plan["inputs_file"],
                 "inputs_file",
                 "is required when embedding_layout delegates the loader's root to a "
                 "declared artifact")
        datasets = layout["datasets"]
        _require(isinstance(datasets, list) and datasets, "embedding_layout.datasets",
                 "must be a non-empty list")
        seen_datasets = set()
        for index, entry in enumerate(datasets):
            field = "embedding_layout.datasets[{}]".format(index)
            _require(isinstance(entry, dict), field, "must be a mapping")
            unknown = sorted(set(entry) - _LAYOUT_DATASET_KEYS)
            _require(not unknown, field,
                     "has unknown key(s): {}".format(", ".join(unknown)))
            _require(isinstance(entry.get("dataset"), str) and entry["dataset"],
                     field + ".dataset", "must be a non-empty string")
            _require(_NAME_SAFE.match(entry["dataset"]), field + ".dataset",
                     "must be a simple dataset directory name")
            _require(entry["dataset"] not in seen_datasets, field + ".dataset",
                     "duplicates dataset {!r}".format(entry["dataset"]))
            seen_datasets.add(entry["dataset"])
            # One archive when the dataset fits one object, several when the artifact
            # pipeline had to shard it: a single stored object cannot exceed the
            # provider's single-PUT ceiling.
            declared = [key for key in ("input", "inputs") if key in entry]
            _require(len(declared) == 1, field,
                     "must declare exactly one of 'input' (one archive) or 'inputs' "
                     "(a sharded set), not both and not neither")
            if "input" in entry:
                shards = [entry["input"]]
                where = field + ".input"
            else:
                shards = entry["inputs"]
                where = field + ".inputs"
                _require(isinstance(shards, list) and shards, where,
                         "must be a non-empty list of artifact keys")
            for shard_index, shard in enumerate(shards):
                at = where if "input" in entry else "{}[{}]".format(where, shard_index)
                _require(isinstance(shard, str) and shard, at,
                         "must be a non-empty string")
                _require(not os.path.isabs(shard) and ".." not in shard.split("/"), at,
                         "must be an artifact key relative to the storage namespace")
            _require(len(set(shards)) == len(shards), where,
                     "names the same artifact twice")

    provider = plan_provider(plan)
    worker = plan.get("worker")
    _require(isinstance(worker, dict) and worker, "worker",
             "must declare the resource the arm needs")
    allowed = _WORKER_NEUTRAL_KEYS | _WORKER_PROVIDER_KEYS[provider]
    unknown = sorted(set(worker) - allowed)
    _require(not unknown, "worker",
             "has key(s) provider {} does not define: {}".format(
                 provider, ", ".join(unknown)))
    count = worker.get("gpu_count", 1)
    _require(isinstance(count, int) and not isinstance(count, bool) and count >= 1,
             "worker.gpu_count", "must be an integer >= 1")

    if provider == providers.COLAB:
        # Colab owns its runtime, scratch, placement and session identity. The
        # plan may only express the accelerator preference; everything else is
        # controller-side Colab configuration, not scientific intent.
        _require(count == 1, "worker.gpu_count",
                 "Colab allocations are single-accelerator")
        accelerator = worker.get("gpu_type")
        _require(accelerator is None or (isinstance(accelerator, str) and accelerator.strip()),
                 "worker.gpu_type",
                 "must be a non-empty accelerator name when present")
        return plan

    _require(isinstance(worker.get("gpu_type"), str) and worker["gpu_type"],
             "worker.gpu_type", "must be a non-empty string")
    _require(isinstance(worker.get("cloud"), str) and worker["cloud"],
             "worker.cloud", "must be a non-empty string")
    image = worker.get("image")
    template = worker.get("template")
    _require((image is None) != (template is None), "worker",
             "must declare exactly one of image or template")
    for field in ("image", "template"):
        value = worker.get(field)
        _require(value is None or (isinstance(value, str) and value),
                 "worker." + field, "must be a non-empty string when present")
    disk = worker.get("container_disk_gb")
    _require(isinstance(disk, int) and not isinstance(disk, bool) and disk >= 1,
             "worker.container_disk_gb", "must be an integer >= 1")
    volume_gb = worker.get("volume_gb")
    if volume_gb is not None:
        _require(isinstance(volume_gb, int) and not isinstance(volume_gb, bool)
                 and volume_gb >= 0, "worker.volume_gb",
                 "must be an integer >= 0")
    mount = worker.get("volume_mount_path")
    if mount is not None:
        _require(isinstance(mount, str) and mount, "worker.volume_mount_path",
                 "must be a non-empty string")
    data_centers = worker.get("data_centers")
    if data_centers is not None:
        _require(isinstance(data_centers, list) and data_centers, "worker.data_centers",
                 "must be a non-empty list of provider data-center identifiers")
        for center in data_centers:
            _require(isinstance(center, str) and center.strip(), "worker.data_centers",
                     "every data center must be a non-empty string")
    volume = worker.get("network_volume")
    if volume is not None:
        _require(isinstance(volume, dict), "worker.network_volume", "must be a mapping")
        unknown = sorted(set(volume) - _VOLUME_KEYS)
        _require(not unknown, "worker.network_volume",
                 "has unknown key(s): {}".format(", ".join(unknown)))
        minimum = volume.get("min_size_gb")
        _require(isinstance(minimum, int) and not isinstance(minimum, bool) and minimum > 0,
                 "worker.network_volume.min_size_gb", "must be a positive integer")
    return plan


def load_plan(path):
    with open(path, "r", encoding="utf-8") as handle:
        try:
            plan = json.load(handle)
        except ValueError as exc:
            raise ConfigurationError(
                "compute plan {} is not valid JSON: {}".format(path, exc)
            ) from exc
    validate_plan(plan)
    plan["_path"] = os.path.abspath(path)
    return plan


def stage_seeds(plan, stage):
    spec = plan["stages"].get(stage)
    if spec is None:
        raise ConfigurationError(
            "plan declares no stage {!r}; stages are {}".format(
                stage, ", ".join(sorted(plan["stages"]))
            )
        )
    return list(spec["seeds"])


def stage_arm_names(plan, stage):
    """The arm names one stage expands.

    A stage without an explicit `arms` selection runs every declared arm, which
    is exactly the behaviour every pre-extension plan relied on. A stage that
    declares `arms` runs only those, so a plan can say "200 matrix jobs plus
    exactly one same-arm repeat" without multiplying the repeat across the
    stage's seeds or inventing worker concurrency to express topology.
    """

    spec = plan["stages"].get(stage)
    if spec is None:
        raise ConfigurationError(
            "plan declares no stage {!r}; stages are {}".format(
                stage, ", ".join(sorted(plan["stages"]))
            )
        )
    selected = spec.get("arms")
    if selected is None:
        return arm_names(plan)
    return list(selected)


def stage_arms(plan, stage):
    """The arm specifications one stage expands, in declared order."""

    return [arm_by_name(plan, name) for name in stage_arm_names(plan, stage)]


def arm_names(plan):
    return [arm["arm"] for arm in plan["arms"]]


def arm_by_name(plan, name):
    for arm in plan["arms"]:
        if arm["arm"] == name:
            return arm
    raise ConfigurationError(
        "plan declares no arm {!r}; arms are {}".format(name, ", ".join(arm_names(plan)))
    )


def expansion_values(plan, arm, seed):
    return {
        "task_type": plan["task_type"],
        "config": arm.get("config", ""),
        "device_index": str(plan.get("device_index", 0)),
        "seed": str(seed),
    }


def expand_argv(plan, arm, seed):
    """Substitute the four plan placeholders. No shell parsing is involved."""

    values = expansion_values(plan, arm, seed)
    argv = []
    for token in arm["argv"]:
        expanded = token
        for name, value in values.items():
            expanded = expanded.replace("{" + name + "}", value)
        if "{" in expanded or "}" in expanded:
            raise ConfigurationError(
                "arm {!r} argv token {!r} still contains a placeholder after "
                "expansion".format(arm["arm"], token)
            )
        argv.append(expanded)
    return argv


def job_name(study, stage, arm, seed):
    return "{}__{}__{}__s{:02d}".format(study, stage, arm, seed)


def job_key(scope, study, stage, arm, seed, commit):
    """Deterministic identity used to detect a duplicate submission."""

    return "::".join((scope, job_name(study, stage, arm, seed), commit))


def outputs_root(arm, seed):
    return "outputs/{}_s{:02d}".format(arm, seed)


def declared_outputs(plan, arm, seed):
    """The workspace-relative files this job promises to produce.

    Paths are derived from the plan and the (arm, seed) pair, never from a
    timestamp, so they are known before the job runs and can be declared
    required in the spec.

    A Study job also declares the wrapper's sidecar ``MANIFEST.json``, which is
    what the evidence validator reads back. An infrastructure-validation job
    runs a plain command and stages no research evidence, so its declared
    outputs are exactly the artifacts that command was asked to write.
    """

    root = outputs_root(arm["arm"], seed)
    key_root = "{}/{}_s{:02d}".format(plan_scope(plan), arm["arm"], seed)
    entries = []
    for output in plan["outputs"]:
        kind = output["kind"]
        if kind == "checkpoint":
            filename = "checkpoint_{}.pth".format(output["tag"])
        elif kind == "gradient_diagnostics":
            filename = "gradient_diagnostics.json"
        else:
            filename = output["name"] + ".txt"
        entries.append({
            "path": "{}/{}".format(root, filename),
            "artifact": "{}/{}".format(key_root, filename),
            "required": bool(output.get("required", True)),
            "overwrite": False,
        })
    if plan_scope_kind(plan) == scopes.STUDY:
        entries.append({
            "path": "{}/MANIFEST.json".format(root),
            "artifact": "{}/MANIFEST.json".format(key_root),
            "required": True,
            "overwrite": False,
        })
    entries.sort(key=lambda entry: entry["path"])
    return entries


def build_spec(plan, *, stage, arm, seed, commit, scope, envelope_digest,
               inputs=()):
    """Render one version 1 job specification.

    The shape follows the plan's scope kind and nothing else: a Study job names
    the committed research wrapper and declares its evidence manifest, while an
    infrastructure-validation job names the plan's own probe command and
    declares only the artifacts that command writes.
    """

    validate_plan(plan)
    commit = str(commit).strip().lower()
    if not _COMMIT.match(commit):
        raise RepositoryConflictError(
            "an exact full commit is required, got {!r}; branches, tags, short "
            "prefixes and HEAD are not reproducible".format(commit)
        )
    arm_spec = arm_by_name(plan, arm) if isinstance(arm, str) else arm
    seed = int(seed)
    if seed not in stage_seeds(plan, stage):
        raise ConfigurationError(
            "seed {} is not in the pre-registered set for stage {} ({})".format(
                seed, stage, ", ".join(str(value) for value in stage_seeds(plan, stage))
            )
        )
    if arm_spec["arm"] not in stage_arm_names(plan, stage):
        raise ConfigurationError(
            "arm {!r} is not selected by stage {!r}; that stage runs {}".format(
                arm_spec["arm"], stage, ", ".join(stage_arm_names(plan, stage))
            )
        )

    kind = plan_scope_kind(plan)
    scope_id = plan_scope(plan)
    name = job_name(scope_id, stage, arm_spec["arm"], seed)
    root = outputs_root(arm_spec["arm"], seed)
    if kind == scopes.STUDY:
        # The wrapper is committed code: it runs the arm, stages the declared
        # outputs and writes the sidecar manifest the evidence validator reads.
        command_argv = [
            "uv", "run", "--locked", "python", "-m", WRAPPER_MODULE,
            "--plan", _plan_relative(plan),
            "--stage", stage,
            "--arm", arm_spec["arm"],
            "--seed", str(seed),
            "--outputs-root", root,
        ]
        setup_argv = list(plan.get("setup_argv") or DEFAULT_SETUP_ARGV)
    else:
        # An infrastructure-validation job *is* the command the plan names: no
        # research wrapper, no run identity, no staged evidence. An absent
        # setup means no setup, because such a job is a direct probe, not a
        # training run that needs the research environment installed.
        command_argv = expand_argv(plan, arm_spec, seed)
        setup_argv = [str(token) for token in (plan.get("setup_argv") or [])]

    metadata = {
        "stage": stage,
        "arm": arm_spec["arm"],
        "method": plan.get("method") or arm_spec.get("method", arm_spec["arm"]),
        "seed": str(seed),
        "task_type": plan["task_type"],
        "scope": scope,
        "envelope_digest": envelope_digest,
    }
    if kind == scopes.STUDY:
        metadata["study_id"] = scope_id
    else:
        metadata["scope_id"] = scope_id
        metadata["scope_kind"] = kind
    for key, value in (arm_spec.get("labels") or {}).items():
        metadata[key] = str(value)

    spec = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "source": {"repository": plan["repository"], "commit": commit},
        "command": {"argv": command_argv},
        "runtime": {
            "timeout_seconds": int(plan.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)),
            "environment": dict(plan.get("environment") or {}),
        },
        "inputs": list(inputs),
        "outputs": declared_outputs(plan, arm_spec, seed),
        "tracking": {"metadata": metadata},
    }
    if setup_argv:
        # The control plane's own model requires a non-empty setup argv, so "no
        # setup" is the absent key, never an empty vector.
        spec["setup"] = {"argv": list(setup_argv)}
    secrets = plan.get("environment_secrets")
    if secrets:
        spec["runtime"]["environment_secrets"] = list(secrets)
    validate_spec_locally(spec)
    return spec


def _plan_relative(plan):
    path = plan.get("_path")
    if not path:
        raise ConfigurationError("plan has no source path; load it with load_plan()")
    root = os.path.realpath(repo_root())
    real_path = os.path.realpath(path)
    if not real_path.startswith(root + os.sep):
        raise RepositoryConflictError("compute plan must be inside the research checkout")
    relative = os.path.relpath(real_path, root)
    tracked = subprocess.run(
        ["git", "-C", root, "ls-files", "--error-unmatch", "--", relative],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        universal_newlines=True,
    )
    if tracked.returncode != 0:
        raise RepositoryConflictError("compute plan {} is not tracked by Git".format(relative))
    return relative


def repo_root():
    from improvements.compute import resolve as resolve_module

    return resolve_module.repo_root()


def render(spec):
    """Canonical bytes for one spec: sorted keys, fixed indent, trailing newline."""

    return state_module.canonical_json(spec)


def job_key_of(spec):
    metadata = spec.get("tracking", {}).get("metadata", {})
    commit = spec["source"]["commit"]
    return "::".join((
        metadata.get("scope", ""),
        spec["name"],
        commit,
    ))


def validate_spec_locally(spec):
    """Mirror the control plane's own validation so nothing invalid is ever sent."""

    _require(isinstance(spec, dict), "spec", "must be a mapping")
    expected = {"schema_version", "name", "source", "command", "setup", "runtime",
                "inputs", "outputs", "tracking"}
    present = set(spec)
    _require(present <= expected, "spec", "has unexpected key(s): {}".format(
        ", ".join(sorted(present - expected))))
    for required in ("schema_version", "name", "source", "command", "runtime",
                     "inputs", "outputs", "tracking"):
        _require(required in spec, "spec", "is missing {}".format(required))
    _require(spec["schema_version"] == SCHEMA_VERSION, "schema_version",
             "must be {}".format(SCHEMA_VERSION))
    _require(1 <= len(spec["name"]) <= 81, "name", "must be 1..81 characters")
    _require(_COMMIT.match(spec["source"]["commit"]), "source.commit",
             "must be a full 40 or 64 character hex commit")
    _require(spec["source"]["repository"].startswith("https://"), "source.repository",
             "must be an anonymous https:// URL")
    argv = spec["command"]["argv"]
    _require(isinstance(argv, list) and argv, "command.argv", "must be a non-empty vector")
    for token in argv:
        _require(isinstance(token, str) and token, "command.argv",
                 "every token must be a non-empty string")
    for entry in spec["inputs"]:
        unknown = sorted(set(entry) - {"artifact", "destination", "required",
                                       "sha256", "size_bytes", "manifest"})
        _require(not unknown, "inputs", "entry has unknown key(s): {}".format(
            ", ".join(unknown)))
        _require(entry.get("sha256") and _COMMIT.match(entry["sha256"]),
                 "inputs", "every input must carry a 64-hex sha256")
        _require(_relative(entry["destination"]), "inputs",
                 "destination {} is not a safe relative path".format(entry["destination"]))
        _require(entry.get("manifest") is None, "inputs",
                 "declare either a manifest or an explicit digest, not both")
    for entry in spec["outputs"]:
        unknown = sorted(set(entry) - {"path", "artifact", "required", "overwrite"})
        _require(not unknown, "outputs", "entry has unknown key(s): {}".format(
            ", ".join(unknown)))
        _require(_relative(entry["path"]), "outputs",
                 "path {} is not a safe relative path".format(entry["path"]))
        _require(entry["path"].split("/")[0] not in ("source", "state"), "outputs",
                 "path {} targets a reserved job directory".format(entry["path"]))
    for key, value in spec["tracking"]["metadata"].items():
        _require(bool(value) and len(value) <= 1024, "tracking.metadata",
                 "{} must be non-empty and at most 1024 characters".format(key))
        _check_secret_free(value, "tracking.metadata.{}".format(key))
    return spec


def _relative(path):
    text = str(path)
    if not text or text.startswith("/") or text.startswith("~") or "\\" in text:
        return False
    return ".." not in text.split("/")


def git_state(root=None):
    """HEAD and whether tracked files have uncommitted changes."""

    root = root or repo_root()
    head = subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, universal_newlines=True,
    )
    if head.returncode != 0:
        raise RepositoryConflictError(
            "cannot read HEAD in {}: {}".format(root, head.stderr.strip())
        )
    status = subprocess.run(
        ["git", "-C", root, "status", "--porcelain", "--untracked-files=all"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, universal_newlines=True,
    )
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "head": head.stdout.strip(),
        "dirty": dirty,
        "clean": not dirty,
    }


def require_committed_experiment(root=None, expected_commit=None):
    """Refuse to generate a recorded job from a tree that is not commit-clean.

    A recorded run must correspond to a commit: the worker checks out that SHA,
    so anything uncommitted here would silently not be what ran.
    """

    state = git_state(root)
    if not state["clean"]:
        raise RepositoryConflictError(
            "the scientific working tree has uncommitted changes ({}); commit the "
            "exact implementation and configuration before generating a recorded "
            "job".format(", ".join(state["dirty"][:5]))
        )
    if expected_commit and state["head"] != expected_commit.lower():
        raise RepositoryConflictError(
            "working tree HEAD is {} but the plan/run names {}; check out the "
            "committed revision you intend to execute".format(
                state["head"], expected_commit
            )
        )
    return state


def write_spec(spec, path):
    """Write one spec deterministically."""

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render(spec))
    return path


def verification_digest(spec):
    return state_module.digest_of_text(render(spec))


def looks_like_duplicate(record, spec):
    """Whether an existing control-plane job record is this same job."""

    if not isinstance(record, dict):
        return False
    if record.get("name") != spec["name"]:
        return False
    existing = (record.get("spec") or {})
    source = existing.get("source") or {}
    if source.get("commit") != spec["source"]["commit"]:
        return False
    metadata = ((existing.get("tracking") or {}).get("metadata") or {})
    wanted = (spec.get("tracking") or {}).get("metadata") or {}
    for key in ("scope", "stage", "arm", "seed"):
        if metadata.get(key) != wanted.get(key):
            return False
    return True


def artifact_integrity_error(message):
    return ArtifactIntegrityError(message)
