"""Declared experiment inputs: what a job needs, never how to fetch it.

The research side describes required artifacts; the control plane materializes
them (presigned, digest-verified, cache-first). This module is the declaration:
a deterministic, content-addressed list that becomes the job spec's ``inputs``.

A declared input must carry a SHA-256 digest. The control plane refuses a
required input that declares only a size, because size does not verify content,
and the same rule applies here: an unidentified artifact cannot be trusted and
cannot be answered from a content-addressed cache.
"""

import json
import os
import re

from improvements.compute.errors import ArtifactIntegrityError, ConfigurationError

SCHEMA_VERSION = 1

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_SECRET_MARKERS = ("X-Amz-", "?Signature=", "&Signature=", "Bearer ", "AKIA",
                   "-----BEGIN", "https://", "ssh://")

_ALLOWED_KEYS = {
    "artifact", "destination", "sha256", "size_bytes", "required", "membership",
}
_DOCUMENT_KEYS = {"schema_version", "requirements"}


def _check_relative(value, field, *, allow_dotdot=False):
    text = str(value)
    if not text or os.path.isabs(text) or text.startswith("~"):
        raise ConfigurationError(
            "{} must be a relative path, got {!r}".format(field, value)
        )
    if not allow_dotdot and ".." in text.split("/"):
        raise ConfigurationError("{} must not contain '..': {!r}".format(field, value))
    if "\\" in text or text.startswith("/"):
        raise ConfigurationError("{} must use forward slashes: {!r}".format(field, value))
    return text


def _check_key(value, field):
    text = str(value)
    for marker in _SECRET_MARKERS:
        if marker in text:
            raise ConfigurationError(
                "{} must be a storage key, not a URL or credential: {!r}".format(
                    field, value
                )
            )
    if not _SAFE_KEY.match(text) or ".." in text.split("/"):
        raise ConfigurationError(
            "{} must be a relative, unambiguous storage key: {!r}".format(field, value)
        )
    return text


def validate(document):
    """Validate one input-requirement document."""

    if not isinstance(document, dict):
        raise ConfigurationError("the input document must be a mapping")
    unknown = sorted(set(document) - _DOCUMENT_KEYS)
    if unknown:
        raise ConfigurationError(
            "input document has unknown key(s): {}".format(", ".join(unknown))
        )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ConfigurationError(
            "input document schema_version must be {}".format(SCHEMA_VERSION)
        )
    requirements = document.get("requirements")
    if not isinstance(requirements, list):
        raise ConfigurationError("input document requires a 'requirements' list")

    seen_destinations = set()
    for index, requirement in enumerate(requirements):
        where = "requirements[{}]".format(index)
        if not isinstance(requirement, dict):
            raise ConfigurationError("{} must be a mapping".format(where))
        unknown = sorted(set(requirement) - _ALLOWED_KEYS)
        if unknown:
            raise ConfigurationError(
                "{} has unknown key(s): {}".format(where, ", ".join(unknown))
            )
        for field in ("artifact", "destination", "sha256"):
            if not requirement.get(field):
                raise ConfigurationError("{} is missing {}".format(where, field))
        _check_key(requirement["artifact"], where + ".artifact")
        _check_relative(requirement["destination"], where + ".destination")
        if not _SHA256.match(str(requirement["sha256"])):
            raise ArtifactIntegrityError(
                "{}: sha256 must be 64 lowercase hex characters; a size or a "
                "filename is not an identity".format(where)
            )
        size = requirement.get("size_bytes")
        if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
            raise ConfigurationError("{}: size_bytes must be a non-negative integer".format(where))
        if "required" in requirement and not isinstance(requirement["required"], bool):
            raise ConfigurationError("{}: required must be a boolean".format(where))
        destination = requirement["destination"]
        if destination in seen_destinations:
            raise ConfigurationError(
                "two requirements materialize to the same destination {!r}".format(
                    destination
                )
            )
        seen_destinations.add(destination)
        if requirement.get("required", True) and not requirement.get("membership"):
            # Not an error: many artifacts are verified structurally by their
            # loader. Recorded so a reviewer can see which oracle applies.
            requirement["membership"] = "verified structurally by its consumer"
    return document


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        try:
            document = json.load(handle)
        except ValueError as exc:
            raise ConfigurationError(
                "input requirement file {} is not valid JSON: {}".format(path, exc)
            ) from exc
    return validate(document)


def to_job_inputs(document):
    """Render the control plane's ``inputs[]`` shape.

    A required input always carries a digest, which is also what lets the
    control plane answer it from the rebuildable content-addressed cache.
    """

    validate(document)
    rendered = []
    for requirement in document["requirements"]:
        entry = {
            "artifact": requirement["artifact"],
            "destination": requirement["destination"],
            "required": bool(requirement.get("required", True)),
            "sha256": requirement["sha256"],
        }
        if requirement.get("size_bytes") is not None:
            entry["size_bytes"] = int(requirement["size_bytes"])
        rendered.append(entry)
    rendered.sort(key=lambda item: item["destination"])
    return rendered
