"""Provider vocabulary shared with canonical ``wavcse-infra``.

Canonical interface: ``wavcse-infra`` commit
``2d7640c7c6454b662ab92c6744beff946bc111fa``. Keep these values byte-for-byte
compatible with its ``ProviderKind``, ``CostUnit`` and ``ExecutionTransport``
enums. This module owns vocabulary only — provider implementation remains in
``wavcse-infra``.

A missing provider in a pre-provider compute plan or authorization is the
explicit legacy rule ``runpod``. It is not inferred from whichever fields happen
to be present. New Colab-first plans state ``provider: colab``.
"""

RUNPOD = "runpod"
COLAB = "colab"
PROVIDERS = (RUNPOD, COLAB)
DEFAULT_PROVIDER = RUNPOD

USD_PER_HOUR = "USD/hour"
COMPUTE_UNITS = "CU"
COST_UNITS = (USD_PER_HOUR, COMPUTE_UNITS)

SSH = "ssh"
COLAB_EXEC = "colab_exec"
EXECUTION_TRANSPORTS = (SSH, COLAB_EXEC)


class ProviderValueError(ValueError):
    pass


def normalize_provider(value):
    """Return one canonical provider string; ``None`` is legacy RunPod."""

    provider = DEFAULT_PROVIDER if value is None else str(value).strip().lower()
    if provider not in PROVIDERS:
        raise ProviderValueError(
            "unknown execution provider {!r}; providers are {}".format(
                value, ", ".join(PROVIDERS)
            )
        )
    return provider


def provider_cost_unit(provider):
    """The provider's native cost unit — never a cross-unit conversion."""

    provider = normalize_provider(provider)
    return COMPUTE_UNITS if provider == COLAB else USD_PER_HOUR


def worker_provider(worker):
    """Provider carried by a normalized infra worker; legacy rows are RunPod."""

    return normalize_provider((worker or {}).get("provider"))


def worker_transport(worker):
    """Execution transport carried by the worker — never inferred from readiness."""

    value = (worker or {}).get("execution_transport")
    if value is None:
        # Pre-provider RunPod worker rows did not carry the field.
        return SSH
    transport = str(value).strip().lower()
    if transport not in EXECUTION_TRANSPORTS:
        raise ProviderValueError(
            "unknown execution transport {!r}; transports are {}".format(
                value, ", ".join(EXECUTION_TRANSPORTS)
            )
        )
    return transport
