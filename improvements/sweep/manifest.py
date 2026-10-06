"""The ``sweep.json`` contract: what to run, and the identity of each run.

A sweep is declared, not discovered. The file names the study, the model, the
exact layer combinations, the stages and seeds, and the resource policy; the
generator turns it into one committed config per (stage, combo), and the
supervisor expands it into (combo, seed) jobs. Nothing about what runs is
decided at launch time, which is what makes a sweep reproducible from the
repository alone.

Layer combinations use the same grammar the pipeline already parses
(``downstream/utils/parse_transformer_layers.py``): ``all``, a comma list, or
an inclusive range. ``parse_layers`` re-implements that grammar locally so this
module stays importable without the frozen downstream tree on ``sys.path``;
``test_layer_sweep_manifest.py`` asserts parity with the downstream parser on a
table of inputs, so the two cannot drift silently.
"""

import hashlib
import json
import os
import re
import subprocess

from improvements.sweep.scheduler import DEFAULT_POLICY

SCHEMA_VERSION = 1

REQUIRED_KEYS = ("schema_version", "study_id", "sweep_id", "task_type",
                 "run_model", "config_template", "combos", "stages")

_NUMERIC_POLICY_KEYS = ("ceiling", "vram_per_run_gb", "ram_per_run_gb",
                        "ram_reserve_gb", "cpu_cores_per_run", "min_free_disk_gb",
                        "max_gpu_util_pct", "max_load_fraction", "workers_per_run",
                        "probe_interval_s", "max_wall_seconds", "device_index")
_SLUG_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class SpecError(Exception):
    """The sweep specification cannot be honoured as written."""


def parse_layers(combo, model_type):
    """Resolve a layer combination to an explicit index list.

    Mirrors ``downstream/utils/parse_transformer_layers.py``: ``all`` resolves
    against the model variant (25 layers for ``wavlm_large``), a bare range is
    inclusive, and a comma list is taken verbatim -- order included, because a
    position-weighted layer pooling would care about it.
    """
    if combo is None or not str(combo).strip():
        raise SpecError("empty layer combination")
    text = str(combo).strip().lower()
    if text == "all":
        variant = str(model_type).split("_")[-1].lower()
        return list(range((12 if variant == "base" else 24) + 1))
    if "-" in text and "," not in text:
        low, high = (int(part.strip()) for part in text.split("-", 1))
        if low > high:
            raise SpecError("inverted layer range: {}".format(combo))
        return list(range(low, high + 1))
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise SpecError("no layers in combination: {}".format(combo))
    return [int(part) for part in parts]


def combo_slug(combo):
    """A filesystem- and tag-safe name for a combo, still readable.

    ``6,1,0,3`` becomes ``6_1_0_3`` rather than a hash: a sweep directory full
    of digests is unreadable exactly when an operator is trying to see which
    combination failed. A combo that carries an explicit ``id`` uses it, which
    is how a sweep gets short, meaningful directory names for long layer lists.
    """
    if isinstance(combo, dict) and combo.get("id"):
        return _SLUG_SAFE.sub("_", str(combo["id"]))
    layers = combo["layers"] if isinstance(combo, dict) else combo
    slug = _SLUG_SAFE.sub("_", str(layers).replace(",", "_"))
    return slug or "combo"


def combo_id(combo):
    """The stable key of a declared combination (its ``id``, else its layers)."""
    if isinstance(combo, dict) and combo.get("id"):
        return str(combo["id"])
    return str(combo["layers"] if isinstance(combo, dict) else combo)


def normalize_combo(entry, index):
    """Accept a bare layer string or a record, and return a normalized record.

    A sweep's independent variable is the layer set, but a layer set is not
    self-describing: ``0-16`` says nothing about *why* those layers were chosen.
    The record form carries the grouping the experiment is actually about
    (``group``, ``k``) so it can travel into the config, the MLflow tags and the
    run note instead of living only in the operator's head.
    """
    field = "combos[{}]".format(index)
    if isinstance(entry, str):
        record = {"id": None, "layers": entry, "group": None, "k": None}
    elif isinstance(entry, dict):
        _require("layers" in entry, field,
                 "a combo object needs a `layers` string")
        record = {
            "id": entry.get("id"),
            "layers": entry["layers"],
            "group": entry.get("group"),
            "k": entry.get("k"),
        }
        for key in ("id", "group"):
            if record[key] is not None:
                _require(isinstance(record[key], str), "{}.{}".format(field, key),
                         "must be a string")
        if record["k"] is not None:
            _require(isinstance(record["k"], int) and not isinstance(record["k"], bool),
                     "{}.k".format(field), "must be an integer")
    else:
        raise SpecError("{}. must be a layer string or an object with `layers`"
                        .format(field))
    _require(isinstance(record["layers"], str) and record["layers"].strip(),
             "{}.layers".format(field), "must be a non-empty string")
    return record


def is_selected(combo, selection):
    """Whether a stage's ``combos`` selection names this combo (id or layers)."""
    return combo_id(combo) in selection or str(combo["layers"]) in selection


def job_key(study_id, combo, seed, commit):
    """Deterministic identity of one (study, combo, seed, commit).

    The same idea as ``improvements.compute.jobspec.job_key``: identity is
    derived from committed state, never from a timestamp, so restarting a
    supervisor cannot silently launch a second copy of the same run.
    """
    payload = "{}::{}::{}::{}".format(study_id, combo, int(seed),
                                      str(commit).strip().lower())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def dirty_split(tracked_text, untracked_text, study_rel):
    """Split ``git status`` output into what should block a sweep, and what should not.

    The question the check answers is narrow: *are the sweep's own inputs the
    committed ones?* A modified tracked file means the code or a committed
    config is not what the repository says; an untracked file **inside the study
    directory** means a config that was generated but never committed. Both
    block.

    An untracked file anywhere else does not. A colleague's scratch notes, an
    editor file or a weekly slide deck sitting in the working tree has no
    bearing on whether this sweep's configs are the committed record -- and
    letting it block every launch would teach an operator to reach for
    ``--allow-dirty``, which is how the check stops meaning anything.
    """
    tracked_entries = [line for line in (tracked_text or "").splitlines() if line.strip()]
    study_entries = []
    if study_rel and not study_rel.startswith(".."):
        prefix = study_rel.rstrip("/") + "/"
        for path in (untracked_text or "").splitlines():
            path = path.strip()
            if path and (path == study_rel or path.startswith(prefix)):
                study_entries.append("?? " + path)
    return tracked_entries, study_entries


def git_state(repo_root, study_dir=None):
    """The exact commit, and whether the sweep's own inputs are uncommitted.

    A sweep binds itself to a commit (the configs it generates are committed
    before they run), so the check is about *this study's* inputs rather than
    about the working tree being pristine -- see ``dirty_split``.
    """
    def run(args):
        completed = subprocess.run(
            ["git"] + args, cwd=repo_root, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        return completed.returncode, completed.stdout.decode("utf-8", "replace").strip()

    code, head = run(["rev-parse", "HEAD"])
    if code != 0:
        raise SpecError("not a git repository: {}".format(repo_root))
    code, tracked = run(["status", "--porcelain", "--untracked-files=no"])
    code, untracked = run(["ls-files", "--others", "--exclude-standard"])
    study_rel = None
    if study_dir:
        study_rel = os.path.relpath(os.path.abspath(study_dir),
                                    repo_root).replace(os.sep, "/")
    tracked_entries, study_entries = dirty_split(tracked, untracked, study_rel)
    entries = tracked_entries + study_entries
    return {"head": head, "dirty": bool(entries), "dirty_entries": entries,
            "tracked_dirty": tracked_entries, "study_untracked": study_entries}


def _require(condition, field, message):
    if not condition:
        raise SpecError("{}: {}".format(field, message))


def normalize_policy(policy):
    """Merge a spec's policy over the defaults and validate its types."""
    merged = dict(DEFAULT_POLICY)
    for key, value in (policy or {}).items():
        if key not in DEFAULT_POLICY:
            raise SpecError("policy.{}: unknown policy key".format(key))
        if key in _NUMERIC_POLICY_KEYS:
            try:
                merged[key] = int(value) if key in ("ceiling", "cpu_cores_per_run",
                                                    "workers_per_run", "probe_interval_s",
                                                    "max_wall_seconds", "device_index",
                                                    "max_gpu_util_pct") else float(value)
            except (TypeError, ValueError):
                raise SpecError("policy.{}: expected a number, got {!r}".format(key, value))
        elif key == "max_concurrent":
            if not (value == "auto" or isinstance(value, int)):
                raise SpecError("policy.max_concurrent: expected 'auto' or an integer, "
                                "got {!r}".format(value))
            merged[key] = value
        else:
            merged[key] = value
    for key in ("vram_per_run_gb", "ram_per_run_gb", "cpu_cores_per_run", "workers_per_run"):
        if merged[key] <= 0:
            raise SpecError("policy.{}: must be positive".format(key))
    if merged["ram_reserve_gb"] < 0:
        raise SpecError("policy.ram_reserve_gb: must not be negative")
    if not 0 < merged["max_load_fraction"]:
        raise SpecError("policy.max_load_fraction: must be positive")
    return merged


def load_spec(path, *, model_type=None, check_combos=True):
    """Read, validate and normalize a ``sweep.json``."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            spec = json.load(handle)
    except OSError as exc:
        raise SpecError("cannot read sweep spec {}: {}".format(path, exc))
    except ValueError as exc:
        raise SpecError("sweep spec {} is not valid JSON: {}".format(path, exc))

    _require(isinstance(spec, dict), "spec", "must be a JSON object")
    for key in REQUIRED_KEYS:
        _require(key in spec, key, "is required")
    _require(int(spec["schema_version"]) == SCHEMA_VERSION, "schema_version",
             "must be {}".format(SCHEMA_VERSION))

    combos = spec["combos"]
    _require(isinstance(combos, list) and combos, "combos",
             "must be a non-empty list of layer combinations (the sweep's "
             "independent variable); an empty list is refused rather than run")
    normalized = [normalize_combo(entry, index) for index, entry in enumerate(combos)]
    ids = [combo_id(combo) for combo in normalized]
    _require(len(set(ids)) == len(ids), "combos",
             "contains duplicate ids/layer lists")
    if check_combos:
        model_type = model_type or (spec.get("upstream_model_type") or "wavlm_large")
        for combo in normalized:
            try:
                parse_layers(combo["layers"], model_type)
            except ValueError as exc:
                raise SpecError("combo {!r}: {}".format(combo["layers"], exc))
    spec["combos"] = normalized

    stages = spec["stages"]
    _require(isinstance(stages, dict) and stages, "stages", "must be a non-empty object")
    for name, stage in stages.items():
        _require(isinstance(stage, dict), "stages.{}".format(name), "must be an object")
        seeds = stage.get("seeds")
        _require(isinstance(seeds, list) and seeds, "stages.{}.seeds".format(name),
                 "must be a non-empty list")
        for seed in seeds:
            _require(isinstance(seed, int), "stages.{}.seeds".format(name),
                     "every seed must be an integer, got {!r}".format(seed))
        selected = stage.get("combos")
        if selected is not None:
            _require(isinstance(selected, list), "stages.{}.combos".format(name),
                     "must be a list when present")
            unknown = [entry for entry in selected
                       if not any(is_selected(combo, [entry]) for combo in normalized)]
            _require(not unknown, "stages.{}.combos".format(name),
                     "names combinations absent from `combos` (use a combo's id or "
                     "its exact layer string): {}".format(unknown))

    template = spec["config_template"]
    if not os.path.isabs(template):
        spec_dir = os.path.dirname(os.path.abspath(path))
        candidates = [os.path.normpath(os.path.join(spec_dir, template))]
        checkout = _find_checkout(spec_dir)
        if checkout:
            candidates.append(os.path.normpath(os.path.join(checkout, template)))
        resolved = next((item for item in candidates if os.path.exists(item)), None)
        if resolved is None:
            raise SpecError(
                "config_template {!r} does not exist; tried {} (a path is taken "
                "relative to the spec's directory first, then to the repository "
                "root)".format(template, ", ".join(candidates)))
        spec["config_template"] = resolved
    spec["policy"] = normalize_policy(spec.get("policy"))
    spec["_spec_path"] = os.path.abspath(path)
    spec["_study_dir"] = os.path.dirname(os.path.abspath(path))
    spec.setdefault("upstream_model_type", "wavlm_large")
    spec.setdefault("method", spec["run_model"])
    spec.setdefault("experiment_name", "taskrelation-{}-layersweep".format(
        spec["run_model"]))
    spec.setdefault("remote", {})
    return spec


_PLACEHOLDER_HOSTS = ("change-me", "changeme", "todo", "tbd", "none", "example")


def validate_remote_host(spec):
    """Refuse a placeholder host, at the point a verb is about to ssh.

    Called by the verbs that need a host rather than by ``load_spec``: a spec
    copied from the template carries ``remote.host: "CHANGE-ME"``, and refusing
    it at load time would stop ``plan``/``configs`` -- which never touch the
    network -- from being the first thing an operator can usefully run.
    """
    remote = spec.get("remote") or {}
    host = remote.get("host")
    if host is None or str(host).strip() == "":
        return None
    if str(host).strip().lower() in _PLACEHOLDER_HOSTS:
        raise SpecError(
            "remote.host is still the placeholder {!r}; set it to the pod's ssh "
            "target, or remove the `remote` block to run on this machine".format(host))
    if remote.get("checkout") and not str(remote["checkout"]).startswith("/"):
        raise SpecError("remote.checkout must be an absolute path on the remote "
                        "host, got {!r}".format(remote["checkout"]))
    return str(host).strip()


def stage_combos(spec, stage):
    """The combo records one stage runs (all declared combos unless it selects a subset)."""
    _require(stage in spec["stages"], "stage", "unknown stage {!r}".format(stage))
    selected = spec["stages"][stage].get("combos")
    if not selected:
        return list(spec["combos"])
    return [combo for combo in spec["combos"] if is_selected(combo, selected)]


def stage_jobs(spec, stage):
    """Every (combo, seed) pair a stage must produce, in a stable order."""
    _require(stage in spec["stages"], "stage", "unknown stage {!r}".format(stage))
    seeds = spec["stages"][stage]["seeds"]
    return [(combo, int(seed)) for combo in stage_combos(spec, stage) for seed in seeds]


def configs_dir(spec, stage):
    return os.path.join(spec["_study_dir"], "configs", stage)


def state_dir(spec):
    return os.path.join(spec["_study_dir"], "sweep_state")


def _find_checkout(start):
    """Walk up from ``start`` until a git checkout is found (``.git`` may be a file)."""
    path = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(path, ".git")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


def repo_root(spec):
    """The checkout holding the study directory.

    Configs name their output roots relative to the *repository root*, because
    that is the working directory ``run_improvements`` is launched from (its
    ``from improvements....`` imports only resolve there). The pod's checkout
    may sit at a different absolute path than the controller's, so a relative
    path is the only form that is correct on both.
    """
    found = _find_checkout(spec["_study_dir"])
    if found is None:
        raise SpecError("no git checkout above {}".format(spec["_study_dir"]))
    return found


def study_relpath(spec, *parts):
    """A path inside the study directory, relative to the repository root."""
    absolute = os.path.join(spec["_study_dir"], *parts)
    relative = os.path.relpath(absolute, repo_root(spec))
    return relative.replace(os.sep, "/")


def remote_paths(spec, checkout=None):
    """Where the sweep's files live **on the pod**, not on this machine.

    Every path handed to a remote command must be derived from the pod's own
    checkout. Deriving them from the controller's study directory looks right
    on a machine where both happen to use the same absolute path and is wrong
    everywhere else -- the controller would pass its own `sweep_state`
    directory to a supervisor running under a different root, and read logs
    from a path that does not exist there.

    Repo-relative paths are what make this safe: the configs already name their
    output roots relative to the repository root, so the same relative layout
    resolves correctly under any checkout.
    """
    remote = spec.get("remote") or {}
    root = checkout or remote.get("checkout")
    if not root:
        raise SpecError("remote.checkout is not set in the spec; pass --checkout "
                        "or add a remote block")
    root = str(root)
    if not root.startswith("/"):
        raise SpecError("remote.checkout must be an absolute path on the pod, "
                        "got {!r}".format(root))
    study_rel = study_relpath(spec)
    study = os.path.join(root, study_rel)
    return {"checkout": root, "study_rel": study_rel, "study": study,
            "state": os.path.join(study, "sweep_state"),
            "spec": study_rel + "/sweep.json"}


def estimate_seconds(spec, stage, per_run_seconds):
    jobs = len(stage_jobs(spec, stage))
    return jobs * float(per_run_seconds)
