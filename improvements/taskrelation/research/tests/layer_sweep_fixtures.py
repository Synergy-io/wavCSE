"""Shared fixtures for the layer-sweep tests.

Kept out of ``improvements/sweep`` on purpose: the production package carries no
test scaffolding. Unittest's discovery pattern is ``test*.py``, so this module
is not collected as a test itself.
"""

import json
import os

TEMPLATE = """\
log_level: INFO
seed: 42
device:
  type: cuda
  index: 0
paths:
  root_data_path: ~/voice_dataset
  root_emb_path: ~/embedding
  results_root: results/placeholder
  checkpoints_root: checkpoints/placeholder
upstream:
  model_type: wavlm_large
  selected_transformer_layers: all
dataset:
  subset_percentage: 100
  ignore_index: -1
pooling:
  frame_pooling_type: mean
  frame_pooling_param: null
  layer_pooling_type: smp
  layer_pooling_param: 0.5
model:
  mtrl_lambda: 0.01
  normalize_w: false
training:
  num_epochs: 30
  batch_size: 2048
  num_workers: 4
  pin_memory: true
evaluation:
  batch_size: 2048
  num_workers: 4
  pin_memory: true
mlflow:
  tracking_uri: https://example.invalid/mlflow
  experiment_name: placeholder
research:
  study_id: placeholder
  stage: placeholder
  method: placeholder
  run_note_file: improvements/taskrelation/research/studies/DG-00XX/NOTE.md
"""

BASE_SPEC = {
    "schema_version": 1,
    "study_id": "TR-TEST",
    "sweep_id": "layersweep-test",
    "run_model": "mtrl",
    "method": "mtrl_norm_corrected",
    "task_type": "ks_si_er",
    "upstream_model_type": "wavlm_large",
    "config_template": "template.yml",
    "combos": ["all", "0-15", "6,1,0,3"],
    "stages": {"screen": {"seeds": [42]}},
    "policy": {"max_concurrent": "auto"},
}


def make_study(tmp, spec_overrides=None, template_text=TEMPLATE, git=True):
    """Build a throwaway checkout with a study dir, a template and a spec.

    ``.git`` is created as an empty directory: the manifest only needs to
    *find* a checkout, and fabricating a real repository here would test git
    rather than the sweep.
    """
    root = os.path.join(tmp, "repo")
    study = os.path.join(root, "improvements", "taskrelation", "research",
                         "studies", "TR-TEST")
    # exist_ok: a test may build several specs in one tmp dir, each overriding
    # only part of the base spec. Re-creating the study directory is the point.
    os.makedirs(study, exist_ok=True)
    if git:
        os.makedirs(os.path.join(root, ".git"), exist_ok=True)
    with open(os.path.join(root, "template.yml"), "w", encoding="utf-8") as handle:
        handle.write(template_text)
    spec = dict(BASE_SPEC)
    spec.update(spec_overrides or {})
    path = os.path.join(study, "sweep.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)
    return {"root": root, "study": study, "spec_path": path}
