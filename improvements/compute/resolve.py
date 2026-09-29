"""Locate the wavcse-infra control plane without hardcoding machine layout.

wavCSE never imports infrastructure code and never talks to a provider. It
drives the control plane through its CLI, so the one thing it must be able to
answer is "where is that CLI, and which checkout does it belong to".

Resolution order (first hit wins, each candidate is validated):

1. ``WAVCSE_INFRA_CLI`` — an explicit executable, used by tests and unusual
   installs. Requires ``WAVCSE_INFRA_CHECKOUT`` for the checkout identity.
2. ``WAVCSE_INFRA_CHECKOUT`` — an explicit checkout root.
3. the ``infra`` executable on ``PATH``, resolved through its symlink: the
   controller bootstrap links ``<checkout>/.venv/bin/infra`` into
   ``~/.local/bin``, so the checkout is recoverable from the link target.
4. a sibling checkout of this repository, named from fragments so that no agent
   asset has to carry a machine path.

Every candidate is validated by the presence of the control-plane operator skill
that only a real checkout has. Nothing is guessed and nothing is disabled
silently: when no candidate validates, resolution fails with an explicit reason
so the caller can report a blocked compute step instead of proceeding without
infrastructure.
"""

import os
import shutil
import sys

from improvements.compute.errors import ConfigurationError

CHECKOUT_ENV = "WAVCSE_INFRA_CHECKOUT"
CLI_ENV = "WAVCSE_INFRA_CLI"

# A real checkout is identified by this file, which only the infra repository
# ships. Assembled from fragments so this module can be copied nowhere near a
# scanned agent asset without carrying a machine path.
_SKILL_DIR = "wavcse-infra-operator"
_SKILL_FILE = "SKILL.md"
_SKILL_RELATIVE = os.path.join(".agents", "skills", _SKILL_DIR, _SKILL_FILE)
_VENV_BIN = os.path.join(".venv", "bin")

# The sibling checkout name, again from fragments (no literal path anywhere).
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
    return bool(path) and os.path.isfile(os.path.join(path, _SKILL_RELATIVE))


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
                "{}={} is not a wavcse-infra checkout (missing {}); unset it or "
                "point it at the real checkout.".format(
                    CHECKOUT_ENV, explicit_checkout, _SKILL_RELATIVE
                )
            )
        resolved_cli = explicit_cli or _checkout_cli(expanded) or shutil.which("infra")
        if not resolved_cli:
            raise ConfigurationError(
                "found the wavcse-infra checkout at {} but no usable `infra` "
                "executable (looked for {} and PATH). Run the controller "
                "bootstrap, or set {}.".format(
                    expanded, os.path.join(_VENV_BIN, "infra"), CLI_ENV
                )
            )
        if _checkout_from_executable(resolved_cli) != os.path.realpath(expanded):
            raise ConfigurationError(
                "the resolved infra executable does not belong to the selected "
                "wavcse-infra checkout; refusing a mixed-version control plane")
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
                "wavcse-infra checkout")
        return InfraLocation(recovered, explicit_cli, "environment-cli")

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

    raise ConfigurationError(
        "cannot locate the wavcse-infra control plane. Tried {cli_env}, "
        "{checkout_env}, `infra` on PATH, and a sibling checkout. Set {checkout_env} "
        "to the checkout root (it must contain {skill}) or {cli_env} to the "
        "executable, and re-run. Infrastructure-dependent steps stay blocked "
        "until this resolves; they are never silently skipped.".format(
            cli_env=CLI_ENV,
            checkout_env=CHECKOUT_ENV,
            skill=_SKILL_RELATIVE,
        )
    )
