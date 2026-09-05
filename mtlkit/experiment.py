"""mtlkit/experiment.py — config-driven experiment orchestration.

Eng Review decision 1B (2026-09-06): Hydra owns YAML composition, CLI/script
overrides, and structured-config validation. This module is a THIN adapter:
it resolves a Hydra config into the typed `ExperimentConfig` object the
trainer consumes, and owns the MLflow-wrapped telemetry (no off-the-shelf
equivalent tied to `improvements/`'s existing run-naming convention).

    conf/*.yaml  ----------+
                           v
    load_experiment_config(config_dir, config_name, overrides=[...])
                           |  Hydra: compose YAML + apply "key.path=value"
                           |  overrides (the Python-override escape hatch,
                           |  Open Question 3 -- resolved as Hydra overrides
                           |  rather than free-form Python mutation)
                           v
                  OmegaConf DictConfig (resolved)
                           |
                           v  validate required fields, reject unknown
                  ExperimentConfig (typed, what the trainer consumes)
                           |
                           v  start_mlflow_run(config, category, model)
                  MLflow run, named {category}_{model}_{task_type}_{ts}
                  (same pattern as improvements/mlflow_utils.build_run_name)
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Typed config object
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = ("task_type", "upstream_model_type")


@dataclass
class ExperimentConfig:
    """The typed config object the trainer consumes, resolved from a Hydra
    DictConfig by `load_experiment_config`. Fields not recognized here land
    in `extra` rather than being silently dropped."""

    task_type: str = ""
    upstream_model_type: str = ""
    embedding_dim_shared1: int = 256
    embedding_dim_shared2: int = 128
    layer_pooling_type: str = "mean"
    layer_pooling_param: Optional[float] = None
    combine_strategy: str = "uniform_average"
    num_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    mlflow_experiment_name: str = "default"
    mlflow_tracking_uri: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class ConfigError(ValueError):
    """A malformed config or an override that doesn't resolve to a real key
    (Eng Review decision 5A). Raised before training starts, with a message
    naming exactly what's wrong -- never a deep stack trace mid-run."""


def _known_field_names() -> set:
    return {f.name for f in fields(ExperimentConfig)} - {"extra"}


def _to_experiment_config(cfg: DictConfig) -> ExperimentConfig:
    try:
        container = OmegaConf.to_container(cfg, resolve=True)
    except Exception as exc:  # OmegaConf's own interpolation/resolution errors
        raise ConfigError(f"Failed to resolve config: {exc}") from exc

    if not isinstance(container, dict):
        raise ConfigError(
            f"Resolved config must be a mapping (a YAML document with top-level "
            f"keys), got {type(container).__name__}."
        )

    missing = [name for name in _REQUIRED_FIELDS if not container.get(name)]
    if missing:
        raise ConfigError(
            f"Config is missing required field(s): {', '.join(missing)}. "
            f"Provide them in the YAML spec or via an override "
            f"(e.g. task_type=ks_si_er)."
        )

    known = _known_field_names()
    kwargs = {k: v for k, v in container.items() if k in known}
    extra = {k: v for k, v in container.items() if k not in known}
    return ExperimentConfig(**kwargs, extra=extra)


# ---------------------------------------------------------------------------
# Hydra adapter — YAML + override composition
# ---------------------------------------------------------------------------


def load_experiment_config(
    config_dir: str,
    config_name: str,
    overrides: Optional[List[str]] = None,
) -> ExperimentConfig:
    """Compose ``config_name`` (without ``.yaml``) from ``config_dir`` via
    Hydra, apply ``overrides`` (Hydra ``key.path=value`` override strings —
    the Python-override escape hatch), and resolve into an
    :class:`ExperimentConfig`.

    Raises :class:`ConfigError` on a missing required field, an override
    key that doesn't resolve to anything in the composed config, or any
    other Hydra/OmegaConf composition failure — always before training
    starts, never mid-run.
    """
    from hydra import compose, initialize_config_dir

    abs_config_dir = os.path.abspath(config_dir)
    if not os.path.isdir(abs_config_dir):
        raise ConfigError(f"config_dir does not exist: {abs_config_dir}")

    try:
        with initialize_config_dir(version_base=None, config_dir=abs_config_dir):
            cfg = compose(config_name=config_name, overrides=overrides or [])
    except ConfigError:
        raise
    except Exception as exc:
        # Covers Hydra's own HydraException (missing config file, bad
        # override key) AND lower-level failures it doesn't wrap (a
        # malformed YAML document raises yaml.YAMLError, not HydraException)
        # -- either way this must surface as a clear ConfigError before
        # training starts, never a bare stack trace (Eng Review decision 5A).
        raise ConfigError(f"Failed to compose '{config_name}' from '{abs_config_dir}': {exc}") from exc

    return _to_experiment_config(cfg)


def apply_python_override(config: ExperimentConfig, override_fn) -> ExperimentConfig:
    """The Python-script escape hatch for anything a Hydra override string
    can't express: mutates ``config`` in place via an arbitrary callable,
    after Hydra has already composed and validated the YAML spec. Returns
    the same object so callers can chain."""
    override_fn(config)
    return config


# ---------------------------------------------------------------------------
# MLflow-wrapped telemetry
# ---------------------------------------------------------------------------


def build_run_name(category: str, model: str, task_type: str, suffix: Optional[str] = None) -> str:
    """Same run-name pattern as ``improvements/mlflow_utils.build_run_name``:
    ``{category}_{model}_{task_type}_{timestamp}[_{suffix}]``. Canonical
    copy now lives here (mtlkit); `improvements/mlflow_utils.py` keeps its
    own until its owner folders migrate onto mtlkit."""
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    name = f"{category}_{model}_{task_type}_{timestamp}"
    if suffix:
        name = f"{name}_{suffix}"
    return name


@contextmanager
def start_mlflow_run(config: ExperimentConfig, category: str, model: str, suffix: Optional[str] = None):
    """Context manager: sets up the MLflow experiment/tracking URI from
    ``config``, starts a run named per :func:`build_run_name`, logs every
    ``ExperimentConfig`` field as an MLflow param, and ends the run on exit
    (even on an exception)."""
    import mlflow

    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    run_name = build_run_name(category, model, config.task_type, suffix=suffix)
    with mlflow.start_run(run_name=run_name) as run:
        params = {
            f.name: getattr(config, f.name)
            for f in fields(ExperimentConfig)
            if f.name != "extra" and getattr(config, f.name) is not None
        }
        params.update({f"extra.{k}": v for k, v in config.extra.items() if v is not None})
        mlflow.log_params(params)
        yield run
