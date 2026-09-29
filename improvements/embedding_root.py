"""Where a run reads its embeddings from, and why that is not always the config.

Every wavCSE config names an embedding root (`paths.root_emb_path`, usually ``~/embedding``)
and the loader expands it against the process's ``$HOME``. That is fine on a workstation,
where the operator symlinks the shared store into place, and wrong on a disposable worker:
the worker has no such store, and the declared, digest-verified input the job actually
materialized is somewhere else entirely. A run that silently used ``~/embedding`` there
would be reading a stale or absent tree while the verified artifact sat unused — the exact
contract failure this module exists to prevent.

So a job may prepare its own root and name it explicitly, in one variable:

    WAVCSE_ROOT_EMB_PATH=<prepared root>

The variable is honoured only when the root carries the layout marker its preparer wrote
(:data:`LAYOUT_MARKER`), naming the artifacts it was built from and the directories it
created. Without that marker the variable is refused rather than trusted, and there is no
fallback: a root that cannot be proven is never silently replaced by ``~/embedding``,
because falling back is how an experiment reads the wrong bytes without saying so.

When the variable is absent — the controller, a local run, every existing config — the
configured value is returned unchanged, so nothing about current behaviour moves.
"""

import json
import os

ENV_ROOT = "WAVCSE_ROOT_EMB_PATH"
LAYOUT_MARKER = ".arc_embedding_layout.json"
SCHEMA_VERSION = 1


class EmbeddingRootError(RuntimeError):
    """The run's embedding root cannot be established from any trustworthy source."""


def layout_marker_path(root):
    return os.path.join(root, LAYOUT_MARKER)


def read_layout(root):
    """Read and validate the prepared-layout marker under ``root``."""

    path = layout_marker_path(root)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except OSError as exc:
        raise EmbeddingRootError(
            "the embedding root {!r} carries no prepared-layout marker ({}): {}".format(
                root, path, exc
            )
        ) from exc
    except ValueError as exc:
        raise EmbeddingRootError(
            "the prepared-layout marker at {} is not valid JSON: {}".format(path, exc)
        ) from exc
    if not isinstance(document, dict):
        raise EmbeddingRootError("the prepared-layout marker at {} is not an object"
                                 .format(path))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise EmbeddingRootError(
            "the prepared-layout marker at {} declares schema_version {!r}, not {}".format(
                path, document.get("schema_version"), SCHEMA_VERSION
            )
        )
    declared_root = os.path.abspath(str(document.get("root_emb_path") or ""))
    if declared_root != os.path.abspath(root):
        raise EmbeddingRootError(
            "the prepared-layout marker at {} names root_emb_path {!r}, not this root "
            "{!r}".format(path, document.get("root_emb_path"), root)
        )
    datasets = document.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise EmbeddingRootError(
            "the prepared-layout marker at {} names no datasets".format(path)
        )
    for entry in datasets:
        if not isinstance(entry, dict):
            raise EmbeddingRootError(
                "the prepared-layout marker at {} has a malformed dataset entry".format(path)
            )
        for field in ("dataset", "extracted_to"):
            if not entry.get(field):
                raise EmbeddingRootError(
                    "the prepared-layout marker at {} leaves {!r} unset for {!r}".format(
                        path, field, entry.get("dataset")
                    )
                )
        # A dataset is carried by one archive or by several shards; either way every
        # archive it was built from must be named with the digest it was verified at.
        shards = entry.get("artifacts")
        if shards is None and entry.get("artifact"):
            shards = [{"artifact": entry.get("artifact"), "sha256": entry.get("sha256")}]
        if not isinstance(shards, list) or not shards:
            raise EmbeddingRootError(
                "the prepared-layout marker at {} names no archive for dataset {!r}"
                .format(path, entry.get("dataset"))
            )
        for shard in shards:
            if not isinstance(shard, dict):
                raise EmbeddingRootError(
                    "the prepared-layout marker at {} has a malformed archive entry "
                    "for dataset {!r}".format(path, entry.get("dataset"))
                )
            for field in ("artifact", "sha256"):
                if not shard.get(field):
                    raise EmbeddingRootError(
                        "the prepared-layout marker at {} leaves {!r} unset for an "
                        "archive of dataset {!r}".format(
                            path, field, entry.get("dataset"))
                    )
        if not os.path.isdir(str(entry["extracted_to"])):
            raise EmbeddingRootError(
                "the prepared-layout marker claims {!r} for dataset {!r}, but that "
                "directory does not exist; the layout is not the one that was verified"
                .format(entry["extracted_to"], entry["dataset"])
            )
    return document


def resolve_root(config_value, *, environ=None):
    """The embedding root this run must use, and how it was decided."""

    environ = os.environ if environ is None else environ
    override = str(environ.get(ENV_ROOT) or "").strip()
    if not override:
        return os.path.abspath(os.path.expanduser(str(config_value)))
    root = os.path.abspath(os.path.expanduser(override))
    read_layout(root)
    return root


def mark_prepared(root, document):
    """Write the marker that makes ``root`` a prepared, explicitly configured root."""

    root = os.path.abspath(root)
    payload = dict(document)
    payload["schema_version"] = SCHEMA_VERSION
    payload["root_emb_path"] = root
    os.makedirs(root, exist_ok=True)
    path = layout_marker_path(root)
    staging = path + ".staging"
    with open(staging, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, path)
    return payload
