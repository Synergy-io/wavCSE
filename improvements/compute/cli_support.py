"""Shared output and process plumbing for the compute CLI.

Two conventions the whole backend follows:

* every machine-facing verb supports ``--json`` and prints exactly one JSON
  object, so an agent never has to parse human prose;
* exit codes are meaningful: ``0`` proceed, ``2`` usage/configuration error,
  ``3`` refused by policy (authorization, cost, or a hard stop) with the class
  in the payload.

Nothing here prints raw control-plane output. Values that can carry bearer
material (presigned URLs) are replaced before anything is echoed.
"""

import json
import re
import sys

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_REFUSED = 3

_URL_QUERY = re.compile(r"(https?://[^\s\"']+?)\?[^\s\"']*")
_BEARER = re.compile(r"(?i)\b(authorization|bearer|token|api[_-]?key)\b\s*[:=]\s*\S+")


def scrub(text):
    """Remove bearer-shaped material from text before it is printed."""

    if text is None:
        return None
    scrubbed = _URL_QUERY.sub(r"\1?<redacted>", str(text))
    scrubbed = _BEARER.sub(r"\1=<redacted>", scrubbed)
    return scrubbed


def scrub_deep(value):
    """Scrub every string inside a JSON-shaped value."""

    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {key: scrub_deep(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_deep(item) for item in value]
    return value


def emit(payload, *, as_json, stream=None, exit_code=EXIT_OK, lines=None):
    """Print a payload as JSON or as human lines and return the exit code."""

    stream = stream if stream is not None else sys.stdout
    if as_json:
        json.dump(scrub_deep(payload), stream, sort_keys=True, indent=2)
        stream.write("\n")
    else:
        if lines is None:
            lines = human_lines(payload)
        for line in lines:
            stream.write(scrub(line) + "\n")
    stream.flush()
    return exit_code


def human_lines(payload, prefix=""):
    """Render a nested payload as ``key: value`` lines."""

    lines = []
    if isinstance(payload, dict):
        for key in sorted(payload):
            value = payload[key]
            if isinstance(value, (dict, list)):
                nested = human_lines(value, prefix + "  ")
                if nested:
                    lines.append("{}{}:".format(prefix, key))
                    lines.extend(nested)
                else:
                    lines.append("{}{}: (none)".format(prefix, key))
            else:
                lines.append("{}{}: {}".format(prefix, key, value))
    elif isinstance(payload, list):
        if not payload:
            return lines
        for index, item in enumerate(payload):
            if isinstance(item, (dict, list)):
                lines.append("{}[{}]".format(prefix, index))
                lines.extend(human_lines(item, prefix + "  "))
            else:
                lines.append("{}- {}".format(prefix, item))
    else:
        lines.append("{}{}".format(prefix, payload))
    return lines
