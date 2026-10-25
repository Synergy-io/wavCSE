"""One committed training config per (stage, combo).

A sweep's configs are evidence: they are what the run identity and the MLflow
tags are read from, so they are generated, committed, and then launched
unchanged. The generator therefore touches only the keys a layer sweep is
allowed to move -- the layer combination, the two output roots, the loader
worker count (a resource knob, not a scientific one), and the research/MLflow
identity block -- and leaves everything else exactly as the study template had
it. ``test_layer_sweep_config_gen.py`` asserts that: it renders two combos and
fails if any other key differs, which is how a stray edit to a shared template
gets caught before it becomes an unexplainable result.
"""

import os

import yaml

from improvements.sweep import manifest


class ConfigError(Exception):
    """A sweep config cannot be generated or written as requested."""


# Keys the sweep is allowed to differ on, by flattened path.
MUTABLE_KEYS = (
    "upstream.selected_transformer_layers",
    "paths.results_root",
    "paths.checkpoints_root",
    "training.num_workers",
    "evaluation.num_workers",
    "mlflow.experiment_name",
)

# Research-block keys the generator owns. Not all of them differ between two
# combos of the same stage (study_id and sweep_id do not), but every one of them
# is written by this module, so a diff that touches any of them is expected
# rather than suspicious.
RESEARCH_MUTABLE_KEYS = (
    "research.study_id",
    "research.stage",
    "research.method",
    "research.hypothesis_slug",
    "research.sweep_id",
    "research.combo",
    "research.group",
    "research.k",
    "research.layers",
    "research.layer_count",
    "research.representation",
    "research.task_set",
    "research.run_name",
    "research.run_note",
)


def _load_template(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError("cannot read config template {}: {}".format(path, exc))
    except yaml.YAMLError as exc:
        raise ConfigError("config template {} is not valid YAML: {}".format(path, exc))
    if not isinstance(document, dict):
        raise ConfigError("config template {} is not a mapping".format(path))
    for section in ("upstream", "paths", "training", "evaluation", "mlflow",
                    "research", "pooling"):
        if not isinstance(document.get(section), dict):
            raise ConfigError("config template {} has no usable `{}` block".format(
                path, section))
    return document


def render(spec, stage, combo, template=None):
    """Render one combo's config as a nested dict (no I/O beyond the template)."""
    document = _load_template(template or spec["config_template"])
    layers = manifest.parse_layers(combo["layers"], spec["upstream_model_type"])
    policy = spec["policy"]

    output_rel = manifest.study_relpath(spec, "outputs", manifest.combo_slug(combo))
    document["upstream"]["selected_transformer_layers"] = combo["layers"]
    document["paths"]["results_root"] = output_rel + "/results"
    document["paths"]["checkpoints_root"] = output_rel + "/checkpoints"
    document["training"]["num_workers"] = int(policy["workers_per_run"])
    document["evaluation"]["num_workers"] = int(policy["workers_per_run"])
    document["mlflow"]["experiment_name"] = spec["experiment_name"]
    document["research"].update({
        "study_id": spec["study_id"],
        "stage": stage,
        "method": spec["method"],
        "hypothesis_slug": spec["sweep_id"],
        "sweep_id": spec["sweep_id"],
        "combo": manifest.combo_id(combo),
        "group": combo.get("group"),
        "k": combo.get("k"),
        "layers": combo["layers"],
        "layer_count": len(layers),
        "representation": "{}{}L".format(
            document["pooling"]["layer_pooling_type"], len(layers)),
        "task_set": spec["task_type"],
        "agent_generated": True,
        # The sweep's own axis cannot be expressed by the fixed name fields, and
        # the two other options are worse: a run whose name is wrong until it
        # finishes, or a name that encodes the axis nowhere.
        "run_name": spec.get("run_name_template",
                             "{study_id}__{stage}__{combo}__{task_type}__"
                             "{representation}__s{seed:02d}"),
        # `research.run_note` is what mlflow_utils.resolve_run_note publishes as
        # `mlflow.note.content` -- the DagsHub run description. The exact layer
        # list belongs there: it is the detail a reader needs to reproduce the
        # combination, and a tag is the wrong place for 23 numbers.
        # `mlflow_utils.resolve_run_note` prefers `run_note_file` over
        # `run_note`, so an inherited note-file path would silently discard the
        # per-combo detail block below. Clearing it is what makes the inline
        # note authoritative for sweep runs.
        "run_note_file": None,
        "run_note": render_run_note(
            combo, layers,
            manifest.parse_layers("all", spec["upstream_model_type"])),
    })
    if spec.get("parent_study"):
        document["research"]["parent_study"] = spec["parent_study"]
    if spec.get("baseline_study"):
        document["research"]["baseline_study"] = spec["baseline_study"]
    return document


def render_run_note(combo, layers, all_layers):
    """The human-readable detail block published as the run's note."""
    kept = ",".join(str(index) for index in layers)
    dropped = ",".join(str(index) for index in all_layers if index not in layers)
    return (
        "Group: {group}\n"
        "k (layers dropped): {k}\n"
        "Combo: {combo}\n"
        "Layers kept ({count} of {total}): {kept}\n"
        "Layers dropped: {dropped}\n"
    ).format(group=combo.get("group") or "ungrouped", k=combo.get("k"),
             combo=manifest.combo_id(combo), count=len(layers),
             total=len(all_layers), kept=kept, dropped=dropped or "(none)")


def flatten(document, prefix=""):
    """Flatten nested mappings to ``a.b.c -> value`` for structural diffing."""
    flat = {}
    for key, value in (document or {}).items():
        path = "{}.{}".format(prefix, key) if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat


def differing_keys(left, right):
    """Flattened keys whose values differ between two rendered configs."""
    flat_left, flat_right = flatten(left), flatten(right)
    keys = set(flat_left) | set(flat_right)
    return sorted(key for key in keys if flat_left.get(key) != flat_right.get(key))


def write(spec, stage, combo, *, force=False):
    """Write one combo's config, refusing to clobber a committed one."""
    document = render(spec, stage, combo)
    directory = manifest.configs_dir(spec, stage)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, manifest.combo_slug(combo) + ".yml")
    if os.path.exists(path) and not force:
        raise ConfigError("{} already exists; pass force=True to overwrite an "
                          "existing committed config".format(path))
    header = (
        "# Generated by `python -m improvements.sweep configs` from {sweep}.\n"
        "# Sweep: {sweep_id} | stage: {stage} | combo: {combo}\n"
        "# Do not hand-edit: regenerate from the spec so the config stays the\n"
        "# committed record of what ran (see improvements/sweep/config_gen.py).\n"
    ).format(sweep=os.path.relpath(spec["_spec_path"], manifest.repo_root(spec)),
             sweep_id=spec["sweep_id"], stage=stage, combo=combo)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        yaml.safe_dump(document, handle, sort_keys=False, default_flow_style=False)
    return path


def write_stage(spec, stage, *, force=False):
    """Write every config one stage needs, in declaration order."""
    return [write(spec, stage, combo, force=force)
            for combo in manifest.stage_combos(spec, stage)]


def config_path(spec, stage, combo):
    """The path ``render``/``write`` uses for one (stage, combo)."""
    return os.path.join(manifest.configs_dir(spec, stage),
                        manifest.combo_slug(combo) + ".yml")


def relative_config_path(spec, stage, combo):
    """The repo-relative config path handed to ``run_improvements --config``."""
    absolute = config_path(spec, stage, combo)
    relative = os.path.relpath(absolute, manifest.repo_root(spec))
    return relative.replace(os.sep, "/")
