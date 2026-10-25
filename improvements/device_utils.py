"""Device assertions for runs that declare a requirement on the training device.

`downstream/utils/setup_device.py` resolves a device and, when CUDA is missing,
prints "CUDA not available. Falling back to CPU." and returns a CUDA-less
device; it also clamps an out-of-range index to ``cuda:0`` with a print. That is
acceptable for interactive experimentation but unacceptable for a recorded run:
a job whose config declares ``device.type: cuda`` would train on CPU and still
produce plausible metrics, and the exit status would say nothing about it.

`downstream/` is frozen, so the assertion lives here and the entry points in
`improvements/` call it immediately after ``set_device``. Explicitly configured
CPU work is untouched: only a run that asked for CUDA is checked.

Usage::

    device = set_device(cfg["device"]["type"], cfg["device"]["index"])
    assert_training_device(cfg["device"]["type"], cfg["device"]["index"], device)
"""

import torch


def assert_training_device(configured_type, device_index, device, *, context=""):
    """Fail loudly when a run configured for CUDA cannot actually use it.

    Returns the device unchanged so it can be used inline. Raises ``RuntimeError``
    when the configured type is CUDA and any of the following holds:

    * CUDA is not available to this process;
    * the resolved device is not a CUDA device;
    * the resolved device index differs from the requested one (setup_device
      silently substitutes ``cuda:0``).

    A configured type other than CUDA (including ``cpu``) is returned unchanged.
    """

    configured = str(configured_type).strip().lower()
    if configured != "cuda":
        # Explicitly configured CPU work is legitimate and is not second-guessed.
        return device

    where = f" ({context})" if context else ""
    requested_index = 0 if device_index is None else int(device_index)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This run declares device.type: cuda but CUDA is not available to "
            f"this process{where}. Refusing to train on CPU: the metrics would "
            "look valid while the run used a different device than the recorded "
            "configuration. Fix the environment (driver, GPU visibility, or "
            "container device request) or run the job on a machine that has the "
            "requested GPU."
        )

    if device is None or device.type != "cuda":
        raise RuntimeError(
            "This run declares device.type: cuda but the resolved training "
            f"device is {device!r}{where}. Refusing to continue."
        )

    actual_index = 0 if device.index is None else int(device.index)
    if actual_index != requested_index:
        raise RuntimeError(
            "This run requested cuda:{} but the resolved training device is "
            "cuda:{}{}. setup_device substitutes cuda:0 when the requested index "
            "does not exist, which silently changes which GPU is used and can "
            "collide with a concurrent run. Refusing to continue.".format(
                requested_index, actual_index, where
            )
        )

    return device
