"""Compute backend for autonomous research control (ARC v1).

wavCSE owns the research orchestration. The infrastructure control plane
(``wavcse-infra``) owns provider, storage and job mechanics. This package is the
seam between them: it speaks to the control plane only through its CLI, inside
an authorization envelope, and it keeps the runtime facts (leases, spend, job
identifiers) that never belong in Git.

Nothing here imports ``wavcse_infra``, calls a provider, or moves artifact
bytes itself. See ``README.md`` in this directory and
``.agents/policies/autonomy.md`` in the repository root for the rules this code
implements.
"""

SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION"]
