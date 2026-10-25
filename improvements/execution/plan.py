"""Decode a private, committed plan without importing private controller code."""

import base64
import binascii
import hashlib
import json
import os

from improvements.execution.errors import ConfigurationError


_ALLOWED_PLACEHOLDERS = ("task_type", "config", "device_index", "seed")


def decode_document(parts, sha256):
    """Read one SHA-addressed JSON document from the private controller."""
    if not parts or any(not isinstance(part, str) or not part for part in parts):
        raise ConfigurationError("the job carries no complete private JSON payload")
    try:
        payload = base64.b64decode("".join(parts), altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ConfigurationError("the private plan payload is not valid base64") from exc
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ConfigurationError("the private plan digest does not match its payload")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ConfigurationError("the private plan payload is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise ConfigurationError("the private JSON document must be a mapping")
    return document


def decode(parts, sha256):
    """Return only a Study plan after its transport digest verifies."""
    document = decode_document(parts, sha256)
    if document.get("scope_kind", "study") != "study":
        raise ConfigurationError("the worker requires one Study plan")
    if not isinstance(document.get("study"), str) or not document["study"]:
        raise ConfigurationError("the Study plan has no identity")
    return document


def plan_scope(plan):
    return plan["study"]


def arm_by_name(plan, name):
    for arm in plan.get("arms", ()):
        if arm.get("arm") == name:
            return arm
    raise ConfigurationError("plan declares no arm {!r}".format(name))


def stage_seeds(plan, stage):
    spec = plan.get("stages", {}).get(stage)
    if not isinstance(spec, dict) or not isinstance(spec.get("seeds"), list):
        raise ConfigurationError("plan declares no stage {!r}".format(stage))
    return spec["seeds"]


def stage_arm_names(plan, stage):
    spec = plan.get("stages", {}).get(stage)
    if not isinstance(spec, dict):
        raise ConfigurationError("plan declares no stage {!r}".format(stage))
    selected = spec.get("arms")
    if selected is None:
        return [arm["arm"] for arm in plan["arms"]]
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ConfigurationError("stage arms must be a non-empty distinct list")
    return selected


def expand_argv(plan, arm, seed):
    """Expand only the four declared placeholders, as a vector (never a shell)."""
    values = {
        "task_type": plan["task_type"],
        "config": arm.get("config", ""),
        "device_index": str(plan.get("device_index", 0)),
        "seed": str(seed),
    }
    config = values["config"]
    if config and (os.path.isabs(config) or ".." in config.split("/")):
        raise ConfigurationError("the public arm config must be a safe relative path")
    argv = []
    for token in arm["argv"]:
        expanded = token
        for name in _ALLOWED_PLACEHOLDERS:
            expanded = expanded.replace("{" + name + "}", values[name])
        if "{" in expanded or "}" in expanded:
            raise ConfigurationError("unresolved private plan placeholder in {!r}".format(token))
        argv.append(expanded)
    return argv
