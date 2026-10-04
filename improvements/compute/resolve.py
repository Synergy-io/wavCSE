"""Locate the bounded infrastructure control plane without hardcoded host paths.

wavCSE never imports infrastructure code and never talks to a provider. It
drives the control plane through its CLI, so the one thing it must answer is
"where is that CLI, and which subsystem does it belong to".

Resolution order (first hit wins, each candidate is validated):

1. ``WAVCSE_INFRA_CLI`` — an explicit executable, used by tests and unusual
   installs. Requires ``WAVCSE_INFRA_CHECKOUT`` for subsystem identity.
2. ``WAVCSE_INFRA_CHECKOUT`` — an explicit infra subsystem root.
3. ``<wavCSE>/infra`` — the canonical monorepo subsystem.
4. the ``infra`` executable on ``PATH``, resolved through its symlink.
5. a legacy sibling checkout, retained only for controller cutover.

Every candidate is validated by package markers belonging to the infra
subsystem (or the legacy operator-skill marker). Nothing is guessed and nothing
is disabled silently: when no candidate validates, resolution fails explicitly
so the caller reports a blocked compute step instead of proceeding without
infrastructure.
"""

import os
import shutil
import sys

from improvements.compute.errors import ConfigurationError

CHECKOUT_ENV = "WAVCSE_INFRA_CHECKOUT"
CLI_ENV = "WAVCSE_INFRA_CLI"

# A current subsystem is identified by both package markers. The operator-skill
# marker remains accepted for backwards compatibility with already-provisioned
# controllers during the one-repository cutover.
_PACKAGE_MARKERS = (
    "pyproject.toml",
    os.path.join("src", "wavcse_infra", "__init__.py"),
)
_SKILL_RELATIVE = os.path.join(
    ".agents", "skills", "wavcse-infra-operator", "SKILL.md"
)
_VENV_BIN = os.path.join(".venv", "bin")
_MONOREPO_DIR = "infra"
_SIBLING_NAME = "-".join(("wavcse", "infra"))

class InfraLocation(object):
    """Where the control plane lives and how to invoke it."""

    def __init__(self, checkout, cli, source):
        self.checkout = checkout
        self.cli = cli
        self.source = source

    def as_dict(self):
        return {
            "checkout": self.checkout,
            "cli": self.cli,
            "resolved_by": self.source,
        }

    def __repr__(self):  # pragma: no cover - debugging aid
        return "InfraLocation(checkout={!r}, cli={!r}, source={!r})".format(
            self.checkout, self.cli, self.source
        )


REPO_ROOT_ENV = "WAVCSE_REPO_ROOT"

# How a control plane counts as "explicitly bound" rather than merely found.
EXPLICIT_SOURCES = ("environment", "environment-cli")


def repo_root():
    """The wavCSE repository root, derived from this file's location.

    ``WAVCSE_REPO_ROOT`` overrides it so tests can point at a fixture checkout
    without a real one; production never sets it.
    """

    override = os.environ.get(REPO_ROOT_ENV, "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    here = os.path.dirname(os.path.abspath(__file__))          # improvements/compute
    return os.path.dirname(os.path.dirname(here))


def _valid_checkout(path):
    if not path:
        return False
    package = all(os.path.isfile(os.path.join(path, marker))
                  for marker in _PACKAGE_MARKERS)
    legacy = os.path.isfile(os.path.join(path, _SKILL_RELATIVE))
    return package or legacy


def _checkout_cli(checkout):
    candidate = os.path.join(checkout, _VENV_BIN, "infra")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def _checkout_from_executable(executable):
    """Recover a checkout root from an ``infra`` console script."""

    if not executable:
        return None
    resolved = os.path.realpath(executable)
    suffix = os.sep + os.path.join(_VENV_BIN, "infra")
    if not resolved.endswith(suffix):
        return None
    return resolved[: -len(suffix)]


def resolve(checkout=None, cli=None):
    """Resolve the control plane, or raise :class:`ConfigurationError`.

    Raises rather than returning ``None``: a caller that cannot reach
    infrastructure must stop its compute step, never continue as if the
    infrastructure layer were optional.
    """

    explicit_cli = cli or os.environ.get(CLI_ENV, "").strip() or None
    explicit_checkout = (
        checkout or os.environ.get(CHECKOUT_ENV, "").strip() or None
    )

    if explicit_checkout:
        expanded = os.path.abspath(os.path.expanduser(explicit_checkout))
        if not _valid_checkout(expanded):
            raise ConfigurationError(
                "{}={} is not an infra subsystem root (expected package markers "
                "{} or legacy marker {}); unset it or point it at this "
                "repository's infra/ directory.".format(
                    CHECKOUT_ENV, explicit_checkout, ", ".join(_PACKAGE_MARKERS),
                    _SKILL_RELATIVE
                )
            )
        resolved_cli = explicit_cli or _checkout_cli(expanded) or shutil.which("infra")
        if not resolved_cli:
            raise ConfigurationError(
                "found the infra subsystem at {} but no usable `infra` "
                "executable (looked for {} and PATH). Run `uv sync --locked "
                "--all-groups` there, or set {}.".format(
                    expanded, os.path.join(_VENV_BIN, "infra"), CLI_ENV
                )
            )
        if _checkout_from_executable(resolved_cli) != os.path.realpath(expanded):
            raise ConfigurationError(
                "the resolved infra executable does not belong to the selected "
                "subsystem; refusing a mixed-version control plane")
        return InfraLocation(expanded, resolved_cli, "environment")

    if explicit_cli:
        if not os.access(explicit_cli, os.X_OK):
            raise ConfigurationError(
                "{}={} is not executable.".format(CLI_ENV, explicit_cli)
            )
        recovered = _checkout_from_executable(explicit_cli)
        if not _valid_checkout(recovered):
            raise ConfigurationError(
                "the explicit infra executable does not resolve to a valid "
                "infra subsystem")
        return InfraLocation(recovered, explicit_cli, "environment-cli")

    embedded = os.path.join(repo_root(), _MONOREPO_DIR)
    if _valid_checkout(embedded):
        resolved_cli = _checkout_cli(embedded)
        if not resolved_cli:
            raise ConfigurationError(
                "found the canonical infra subsystem at {} but no usable `infra` "
                "executable at {}. Run `uv sync --locked --all-groups` from that "
                "directory and re-run.".format(
                    embedded, os.path.join(embedded, _VENV_BIN, "infra")
                )
            )
        return InfraLocation(embedded, resolved_cli, "monorepo")

    on_path = shutil.which("infra")
    if on_path:
        recovered = _checkout_from_executable(on_path)
        if _valid_checkout(recovered):
            return InfraLocation(recovered, on_path, "path")

    sibling = os.path.join(os.path.dirname(repo_root()), _SIBLING_NAME)
    if _valid_checkout(sibling):
        resolved_cli = _checkout_cli(sibling) or shutil.which("infra")
        if resolved_cli:
            return InfraLocation(sibling, resolved_cli, "sibling")

    marker = " and ".join(_PACKAGE_MARKERS)
    raise ConfigurationError(
        "cannot locate the infra control plane. Tried {cli_env}, {checkout_env}, "
        "the monorepo infra/ subsystem, `infra` on PATH, and a legacy sibling "
        "checkout. Set {checkout_env} to a subsystem root containing {marker}, "
        "or set {cli_env} to the executable, and re-run. "
        "Infrastructure-dependent steps stay blocked until this resolves; they "
        "are never silently skipped.".format(
            cli_env=CLI_ENV,
            checkout_env=CHECKOUT_ENV,
            marker=marker,
        )
    )


def require_explicit(location, *, operation):
    """Refuse a provider-mutating step without an explicitly bound control plane.

    The controller preflight established that ``WAVCSE_INFRA_CHECKOUT`` is not
    persisted, and that this repository's embedded ``infra/`` subsystem wins
    resolution ahead of ``PATH``. That subsystem predates the canonical
    implementation and cannot serve a job, so operating it silently is how a
    recorded run would validate the wrong software. Nothing here repairs or
    synchronizes the embedded copy: a recorded mutation simply stops until the
    canonical external checkout is named explicitly.
    """

    if location.source not in EXPLICIT_SOURCES:
        raise ConfigurationError(
            "{} requires an explicitly bound infrastructure control plane, but "
            "resolution picked {} ({}). Bind the canonical external wavcse-infra "
            "checkout with {} (or {}) and re-run; a merely discovered control plane "
            "is never used for a provider-mutating step.".format(
                operation, location.checkout, location.source, CHECKOUT_ENV, CLI_ENV
            )
        )
    return location
