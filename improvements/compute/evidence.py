"""Independent semantic validation of a completed run's stored evidence.

Stored bytes with a matching digest prove only that *some* bytes were stored and that
they are the bytes that were hashed. They do not prove the result belongs to this study,
this stage, this arm, this seed, this commit or this job — a truncated manifest, a
manifest from a neighbouring seed, a metrics file that names the wrong tasks, or a
plausible-looking `NaN` all survive a digest check. ARC must therefore read the evidence
back and validate its *content* before an entry becomes COLLECTED, before the research
state advances, and before cleanup becomes eligible.

What this module establishes is identity, completeness and structural validity — never
quality. A run that lost to the baseline has valid evidence and is exactly as collectable
as one that won: "valid accuracy below baseline" is a result, and deciding what it means
is the science's business, not this validator's. The checks are limited to facts the
registered protocol already fixes:

* the plan's own declared outputs (membership, kinds, required flags);
* the run identity a job is keyed by (study, stage, arm, seed, exact commit, task set);
* the manifest the job wrapper wrote (schema, identity, staged files and their digests);
* the metric vocabulary the protocol's evaluator writes (all-task and per-task lines);
* the diagnostics artifact's declared shape, when the plan requires one;
* the materialization record of every required input.

Everything is read from canonical storage through the control plane's read-only
read-back, and every read is bound to the digest the control plane already verified for
that object, so the bytes inspected here are the bytes that were verified.
"""

import json
import re

from improvements.compute import jobspec
from improvements.compute.errors import EvidenceError

SCHEMA_VERSION = 1
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INTEGER = re.compile(r"^[0-9]+$")

_ALL_TASKS = re.compile(r"^loss_all=(?P<loss>\S+)\s*\|\s*acc_all=(?P<acc>\S+)\s*$")
_ONE_TASK = re.compile(
    r"^(?P<task>[A-Za-z0-9_]+)\s*\|\s*loss=(?P<loss>\S+)\s*\|\s*"
    r"acc=(?P<acc>\S+)\s*\|\s*samples=(?P<samples>\S+)\s*$"
)

MANIFEST_NAME = "MANIFEST.json"
_MANIFEST_KEYS = (
    "schema_version", "study", "stage", "arm", "method", "seed", "task_type",
    "job_id", "commit", "run_id", "checkpoint_run_id", "training_exit_code",
    "staged_at", "files",
)
_MANIFEST_FILE_KEYS = ("name", "kind", "required", "target", "source", "sha256",
                       "size_bytes")


def _fail(where, message):
    raise EvidenceError("{}: {}".format(where, message))


def _require_mapping(value, where):
    if not isinstance(value, dict):
        _fail(where, "must be a JSON object, got {}".format(type(value).__name__))
    return value


def _require_text(value, where):
    if not isinstance(value, str) or not value.strip():
        _fail(where, "must be a non-empty string, got {!r}".format(value))
    return value


def _require_int(value, where, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(where, "must be an integer, got {!r}".format(value))
    if minimum is not None and value < minimum:
        _fail(where, "must be at least {}, got {}".format(minimum, value))
    return value


def _finite(value, where):
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(where, "must be a number, got {!r}".format(value))
    if number != number or number in (float("inf"), float("-inf")):
        _fail(where, "must be finite, got {!r}".format(value))
    return number


def expected_tasks(plan):
    """The task vocabulary the registered protocol fixes for this plan."""

    return [token for token in str(plan["task_type"]).split("_") if token.strip()]


def declared_kind_map(plan, declared):
    """Map each declared output path onto the kind the plan gave it."""

    kinds = {}
    for output in plan["outputs"]:
        filename = output_filename(output)
        for declaration in declared:
            if declaration["path"].rsplit("/", 1)[-1] == filename:
                kinds[declaration["path"]] = output["kind"]
    return kinds


def output_filename(output):
    if output["kind"] == "checkpoint":
        return "checkpoint_{}.pth".format(output["tag"])
    if output["kind"] == "gradient_diagnostics":
        return "gradient_diagnostics.json"
    return output["name"] + ".txt"


def read_evidence(reader, artifact, sha256, *, where):
    """Read one evidence object back, bound to the digest already verified for it."""

    if not _SHA256.match(str(sha256 or "")):
        _fail(where, "no verified sha256 is recorded for this object")
    payload = reader.storage_read(artifact, expected_sha256=sha256,
                                  max_bytes=MAX_EVIDENCE_BYTES)
    if not isinstance(payload, dict) or "text" not in payload:
        _fail(where, "the control plane returned no readable text for the artifact")
    if str(payload.get("sha256")) != str(sha256):
        _fail(where, "the read-back digest does not match the recorded digest")
    return payload["text"]


def check_source_belongs_to_run(source, entry, manifest, *, where):
    """Every staged file must come out of *this* run's own directories.

    The wrapper stages results files from ``results_dir`` and checkpoints from
    ``checkpoints_dir``, and the manifest names both by their basename. A file staged
    from anywhere else — a neighbouring run sharing the same worker, or a stale tree —
    is not this run's evidence.
    """

    segments = [part for part in str(source).split("/") if part]
    wanted = manifest.get("checkpoint_run_id") if entry.get("kind") == "checkpoint" \
        else manifest.get("run_id")
    if not wanted or str(wanted) not in segments:
        _fail(where, "the staged source for {!r} does not come from this run's own "
                     "directories ({!r}, run id {!r})".format(
                         entry.get("target"), source, wanted))


def validate_manifest(text, *, plan, stage, arm, seed, commit, job_id, declared, kinds,
                      outputs_by_path, where):
    """Validate the job wrapper's manifest against the job it claims to describe."""

    try:
        document = json.loads(text)
    except ValueError as exc:
        _fail(where, "the manifest is not valid JSON ({})".format(exc))
    _require_mapping(document, where)
    for key in _MANIFEST_KEYS:
        if key not in document:
            _fail(where, "the manifest is missing {!r}".format(key))
    if document["schema_version"] != SCHEMA_VERSION:
        _fail(where, "the manifest schema_version is {!r}, not {}".format(
            document["schema_version"], SCHEMA_VERSION))

    identity = {
        "study": plan["study"],
        "stage": stage,
        "arm": arm,
        "task_type": plan["task_type"],
        "commit": commit,
        "job_id": job_id,
    }
    for key, value in identity.items():
        if str(document[key]) != str(value):
            _fail(where, "the manifest names {}={!r} but this job is {!r}".format(
                key, document[key], value))
    _require_int(document["seed"], where + ".seed")
    if int(document["seed"]) != int(seed):
        _fail(where, "the manifest names seed {} but this job is seed {}".format(
            document["seed"], seed))
    for key in ("method", "run_id", "checkpoint_run_id", "staged_at"):
        _require_text(document[key], where + "." + key)
    _require_int(document["training_exit_code"], where + ".training_exit_code")
    if document["training_exit_code"] != 0:
        _fail(where, "the manifest records a training exit code of {}".format(
            document["training_exit_code"]))

    files = document["files"]
    if not isinstance(files, list) or not files:
        _fail(where, "the manifest lists no staged files")
    by_target = {}
    for index, entry in enumerate(files):
        _require_mapping(entry, "{}.files[{}]".format(where, index))
        for key in _MANIFEST_FILE_KEYS:
            if key not in entry:
                _fail(where, "manifest file {} is missing {!r}".format(index, key))
        target = _require_text(entry["target"], "{}.files[{}].target".format(where, index))
        if target in by_target:
            _fail(where, "the manifest stages {!r} twice".format(target))
        by_target[target] = entry
        _require_text(entry["source"], "{}.files[{}].source".format(where, index))
        if not _SHA256.match(str(entry["sha256"])):
            _fail(where, "manifest file {!r} has no usable sha256".format(target))
        check_source_belongs_to_run(entry["source"], entry, document, where=where)

    for declaration in declared:
        if not declaration["required"]:
            continue
        if declaration["path"].rsplit("/", 1)[-1] == MANIFEST_NAME:
            # The manifest cannot list itself among the files it stages: it *is* the
            # record, and its own identity is verified by its artifact digest.
            continue
        target = declaration["path"].rsplit("/", 1)[-1]
        entry = by_target.get(target)
        if entry is None:
            _fail(where, "the manifest does not stage the required output {!r}".format(
                declaration["path"]))
        record = outputs_by_path.get(declaration["path"])
        if record is None:
            _fail(where, "no verified object is recorded for {!r}".format(
                declaration["path"]))
        # The manifest must describe the bytes that were actually stored for this job:
        # a manifest from a neighbouring run names the same targets with other digests.
        if str(entry["sha256"]) != str(record.get("sha256")):
            _fail(where, "the manifest digest for {!r} is not the digest the control "
                         "plane verified for it".format(declaration["path"]))
        if int(entry.get("size_bytes") or -1) != \
                int(record.get("verified_size_bytes") or -2):
            _fail(where, "the manifest size for {!r} is not the size the control plane "
                         "verified for it".format(declaration["path"]))
        if entry["kind"] != kinds.get(declaration["path"]):
            _fail(where, "the manifest kind for {!r} is not the kind the plan declares"
                  .format(declaration["path"]))
    return document


def validate_embedding_layout(manifest, *, plan, where):
    """The run's loader root must have been built from the plan's declared inputs.

    The wrapper records which artifacts it turned into the loader's tree. That record is
    only evidence if it agrees with the plan: a manifest naming another artifact, another
    digest, or a dataset the protocol never loads cannot be collected, because the run
    that produced it did not read the inputs this study declared.
    """

    declared = {}
    for entry in (plan.get("embedding_layout") or {}).get("datasets") or ():
        shards = entry.get("inputs") if "inputs" in entry else [entry.get("input")]
        declared[entry["dataset"]] = sorted(shards)

    reported = manifest.get("embedding_layout")
    if not declared:
        if reported is not None:
            _fail(where, "the manifest reports an embedding layout, but the plan "
                         "declares none")
        return None
    if not isinstance(reported, dict):
        _fail(where, "the manifest does not report the embedding layout the run used")
    for key in ("root", "upstream_model_type", "frame_pool_id"):
        _require_text(reported.get(key), where + ".embedding_layout." + key)

    from improvements.compute import run_study

    digest_of = {item["artifact"]: str(item["sha256"])
                 for item in run_study.inputs_for(plan)}

    datasets = reported.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        _fail(where, "the manifest's embedding layout names no dataset")
    seen = {}
    for index, entry in enumerate(datasets):
        at = "{}.embedding_layout.datasets[{}]".format(where, index)
        _require_mapping(entry, at)
        name = _require_text(entry.get("dataset"), at + ".dataset")
        if name in seen:
            _fail(where, "the manifest's embedding layout names {!r} twice".format(name))
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            _fail(at, "names no verified archive")
        names = []
        for shard_index, shard in enumerate(artifacts):
            _require_mapping(shard, "{}[{}]".format(at + ".artifacts", shard_index))
            artifact = _require_text(shard.get("artifact"), at + ".artifacts")
            digest = str(shard.get("sha256") or "")
            if not _SHA256.match(digest):
                _fail(at, "reports {!r} without a usable digest".format(artifact))
            expected = digest_of.get(artifact)
            if expected is None:
                _fail(at, "reports {!r}, which the plan's inputs file does not declare"
                      .format(artifact))
            if expected != digest:
                _fail(at, "reports {!r} at digest {}, but the plan declares {}"
                      .format(artifact, digest, expected))
            names.append(artifact)
        seen[name] = sorted(names)

    missing = sorted(set(declared) - set(seen))
    if missing:
        _fail(where, "the manifest's embedding layout does not report dataset(s) {}"
              .format(", ".join(missing)))
    extra = sorted(set(seen) - set(declared))
    if extra:
        _fail(where, "the manifest's embedding layout reports dataset(s) the plan never "
                     "loads: {}".format(", ".join(extra)))
    for name, artifacts in sorted(declared.items()):
        if seen[name] != artifacts:
            _fail(where, "the manifest's embedding layout reports {} for dataset {!r}, "
                         "but the plan declares {}".format(
                             seen[name], name, artifacts))
    return {"datasets": {name: len(artifacts) for name, artifacts in sorted(seen.items())}}


def validate_metrics(text, *, plan, where):
    """Validate the evaluator's metrics vocabulary, task names and value domains."""

    tasks = expected_tasks(plan)
    if not tasks:
        _fail(where, "the plan declares no task vocabulary")
    all_tasks = None
    per_task = {}
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        whole = _ALL_TASKS.match(stripped)
        if whole:
            if all_tasks is not None:
                _fail(where, "the metrics file reports the all-task summary twice")
            accuracy = _finite(whole.group("acc"), where + " acc_all")
            if not 0.0 <= accuracy <= 1.0:
                _fail(where, "acc_all is {!r}, outside the range an accuracy can take"
                      .format(accuracy))
            all_tasks = {
                "loss": _finite(whole.group("loss"), where + " loss_all"),
                "accuracy": accuracy,
            }
            continue
        single = _ONE_TASK.match(stripped)
        if single:
            name = single.group("task")
            if name in per_task:
                _fail(where, "the metrics file reports {!r} twice".format(name))
            if not _INTEGER.match(single.group("samples")):
                _fail(where, "{}.samples must be a non-negative integer, got {!r}"
                      .format(name, single.group("samples")))
            accuracy = _finite(single.group("acc"), where + " " + name + ".acc")
            if not 0.0 <= accuracy <= 1.0:
                _fail(where, "{}.acc is {!r}, outside the range an accuracy can take"
                      .format(name, accuracy))
            per_task[name] = {
                "loss": _finite(single.group("loss"), where + " " + name + ".loss"),
                "accuracy": accuracy,
                "samples": int(single.group("samples")),
            }
            continue
        _fail(where, "unrecognised metrics line {!r}".format(stripped[:120]))

    if all_tasks is None:
        _fail(where, "the metrics file has no all-task summary line")
    missing = [task for task in tasks if task not in per_task]
    if missing:
        _fail(where, "the metrics file is missing per-task metrics for {}".format(
            ", ".join(missing)))
    unexpected = sorted(set(per_task) - set(tasks))
    if unexpected:
        _fail(where, "the metrics file reports tasks this protocol does not run: "
                     "{}".format(", ".join(unexpected)))
    return {"all": all_tasks, "tasks": per_task}


def validate_gradient_diagnostics(text, *, plan, where):
    """Validate the diagnostics artifact's declared shape."""

    try:
        document = json.loads(text)
    except ValueError as exc:
        _fail(where, "gradient diagnostics is not valid JSON ({})".format(exc))
    _require_mapping(document, where)
    tasks = expected_tasks(plan)
    for key in ("task_array", "sample_interval_steps", "total_training_steps",
                "sampled_steps", "shared_parameter_names", "shared_parameter_count",
                "records"):
        if key not in document:
            _fail(where, "gradient diagnostics is missing {!r}".format(key))
    if list(document["task_array"]) != tasks:
        _fail(where, "gradient diagnostics names tasks {} but this protocol runs {}"
              .format(document["task_array"], tasks))
    if _require_int(document["shared_parameter_count"], where + ".shared_parameter_count",
                    minimum=1) < 1:
        _fail(where, "gradient diagnostics reports no shared parameters")
    records = document["records"]
    if not isinstance(records, list):
        _fail(where, "gradient diagnostics records must be a list")
    sampled = _require_int(document["sampled_steps"], where + ".sampled_steps")
    if sampled != len(records) or not records:
        _fail(where, "gradient diagnostics claims {} sampled steps but carries {} "
                     "records".format(sampled, len(records)))
    for index, record in enumerate(records):
        _require_mapping(record, "{}.records[{}]".format(where, index))
        for key in ("step", "progress", "gradient_norms", "pairwise_cosines"):
            if key not in record:
                _fail(where, "gradient record {} is missing {!r}".format(index, key))
        _finite(record["progress"], "{}.records[{}].progress".format(where, index))
        norms = _require_mapping(record["gradient_norms"],
                                 "{}.records[{}].gradient_norms".format(where, index))
        for task in tasks:
            if task not in norms:
                _fail(where, "gradient record {} has no norm for {!r}".format(index, task))
            value = _finite(norms[task], "{}.records[{}].gradient_norms.{}".format(
                where, index, task))
            if value < 0:
                _fail(where, "gradient record {} reports a negative norm for {!r}".format(
                    index, task))
        cosines = _require_mapping(record["pairwise_cosines"],
                                   "{}.records[{}].pairwise_cosines".format(where, index))
        for pair, value in cosines.items():
            number = _finite(value, "{}.records[{}].pairwise_cosines.{}".format(
                where, index, pair))
            if not -1.0 <= number <= 1.0:
                _fail(where, "gradient record {} reports a cosine of {!r} for {!r}"
                      .format(index, number, pair))
    return {"sampled_steps": len(records)}


def validate_declared_inputs(entry, *, where):
    """Every required declared input must show as materialized and digest-verified."""

    inputs = entry.get("inputs")
    if inputs is None:
        return 0
    if not isinstance(inputs, list):
        _fail(where, "the job record's inputs must be a list")
    materialized = 0
    for index, item in enumerate(inputs):
        _require_mapping(item, "{}.inputs[{}]".format(where, index))
        if item.get("required") is False:
            continue
        if item.get("materialized") is not True:
            _fail(where, "required input {!r} was not materialized".format(
                item.get("destination")))
        if not _SHA256.match(str(item.get("sha256") or "")):
            _fail(where, "required input {!r} carries no verified digest".format(
                item.get("destination")))
        materialized += 1
    return materialized


def validate(plan, *, stage, arm, seed, commit, entry, reader, where):
    """Validate one completed entry's evidence; return what was proven about it."""

    declared = jobspec.declared_outputs(plan, jobspec.arm_by_name(plan, arm), seed)
    kinds = declared_kind_map(plan, declared)
    outputs_by_path = {}
    for output in entry.get("outputs") or ():
        path = output.get("path")
        if path:
            outputs_by_path.setdefault(path, output)

    manifests = [item for item in declared
                 if item["path"].rsplit("/", 1)[-1] == MANIFEST_NAME]
    if len(manifests) != 1:
        _fail(where, "the job declares no single manifest output")
    manifest_record = outputs_by_path.get(manifests[0]["path"])
    if manifest_record is None:
        _fail(where, "no verified manifest object is recorded")
    manifest = validate_manifest(
        read_evidence(reader, manifest_record["artifact"], manifest_record["sha256"],
                      where=where),
        plan=plan, stage=stage, arm=arm, seed=seed, commit=commit,
        job_id=entry.get("job_id"), declared=declared, kinds=kinds,
        outputs_by_path=outputs_by_path, where=where,
    )

    summary = {
        "manifest_digest": manifest_record["sha256"],
        "run_id": manifest["run_id"],
        "checkpoint_run_id": manifest["checkpoint_run_id"],
        "files": len(manifest["files"]),
        "inputs_materialized": validate_declared_inputs(entry, where=where),
        # Which verified artifacts the loader-visible root was built from, checked
        # against the plan's own declared inputs rather than trusted from the wrapper.
        "embedding_layout": validate_embedding_layout(manifest, plan=plan, where=where),
        "metrics": {},
        "gradient_diagnostics": None,
    }
    for output in plan["outputs"]:
        if output["kind"] not in ("results_file", "gradient_diagnostics"):
            continue
        filename = output_filename(output)
        declaration = next((item for item in declared
                            if item["path"].rsplit("/", 1)[-1] == filename), None)
        if declaration is None:
            continue
        record_output = outputs_by_path.get(declaration["path"])
        if record_output is None:
            _fail(where, "no verified object is recorded for {!r}".format(
                declaration["path"]))
        text = read_evidence(reader, record_output["artifact"], record_output["sha256"],
                             where=where)
        if output["kind"] == "results_file":
            summary["metrics"][declaration["path"]] = validate_metrics(
                text, plan=plan, where=where + " " + declaration["path"])
        else:
            summary["gradient_diagnostics"] = validate_gradient_diagnostics(
                text, plan=plan, where=where + " " + declaration["path"])
    return summary
