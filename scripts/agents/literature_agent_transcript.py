#!/usr/bin/env python3
"""Prove a Literature Agent run's tool grant from its own transcript.

A Literature Agent evaluation is only meaningful if the run's session transcript
shows (a) the approved literature tools were **granted** and (b) they were
actually **used**. This checker reads the transcript the harness already writes
and decides that mechanically, so a result can never be accepted on the strength
of a configuration file or an agent's self-report.

Observed failure it exists for: an in-session run without the literature grant
answered a provenance question from a historical-recall MCP tool instead
(`PROVENANCE_BYPASS`), and nothing in the record distinguished it from a run that
had used the corpus.

Transcript sources
------------------
A child agent's session file in the OMP session store
(`<omp-config>/agent/sessions/<project>/<session>/<RUN-ID>.jsonl`), or an
evaluation's own `omp -p` transcript. `session_init.tools` holds the granted tool list;
`message.content[].toolCall.name` holds the calls.

Usage
-----
    python3 scripts/agents/literature_agent_transcript.py <transcript.jsonl> [...]
    python3 scripts/agents/literature_agent_transcript.py --require-call literature_query RUN.jsonl

Exit status is 0 only for `VALID`.
"""

import argparse
import json
import sys
from pathlib import Path

REQUIRED_TOOLS = (
    "literature_resolve",
    "literature_query",
    "literature_read",
    "literature_primary",
)

# The harness primitive that returns the result; permitted but not evidence.
HARNESS_TOOLS = ("yield",)

# Any MCP server is outside the approved authority: the literature surface is the
# four capabilities above, and a memory/recall server is exactly the bypass this
# checker detects.
FORBIDDEN_PREFIXES = ("mcp__", "xd://mcp__")

# Native recall/memory devices and any transcript-recall surface.
FORBIDDEN_SUBSTRINGS = ("recall", "retain", "reflect", "history", "memory")

# Evidence capabilities that must actually be called, not merely granted.
CALL_REQUIRED_ANY = (
    "literature_resolve",
    "literature_query",
    "literature_read",
    "literature_primary",
)


def read_records(path):
    records = []
    with open(str(path), "r", encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise SystemExit(
                    "%s:%d is not valid JSON (%s); refusing to guess at a "
                    "malformed transcript" % (path, number, exc)
                )
    return records


def granted_tools(records):
    """The tool list the session recorded for the agent, or None if absent."""

    for record in records:
        if record.get("type") != "session_init":
            continue
        tools = record.get("tools")
        if isinstance(tools, list):
            return [str(name) for name in tools]
    return None


def called_tools(records):
    calls = []
    for record in records:
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        for item in message.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "toolCall":
                name = item.get("name")
                if isinstance(name, str):
                    calls.append(name)
    return calls


def forbidden(names):
    hits = []
    for name in names:
        lowered = name.lower()
        if any(lowered.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            hits.append(name)
        elif any(needle in lowered for needle in FORBIDDEN_SUBSTRINGS):
            hits.append(name)
    return sorted(set(hits))


def check(path, required, require_call):
    records = read_records(path)
    granted = granted_tools(records)
    calls = called_tools(records)

    reasons = []
    if granted is None:
        reasons.append(
            "TRANSCRIPT_WITHOUT_TOOL_GRANT: no session_init.tools record, so the "
            "grant cannot be proven from this transcript"
        )
        granted = []

    missing = [name for name in required if name not in granted]
    if missing:
        reasons.append(
            "REQUIRED_TOOL_NOT_GRANTED: " + ", ".join(missing)
        )

    forbidden_granted = forbidden(granted)
    if forbidden_granted:
        reasons.append(
            "FORBIDDEN_TOOL_GRANTED: " + ", ".join(forbidden_granted)
        )

    forbidden_called = forbidden(calls)
    if forbidden_called:
        reasons.append(
            "FORBIDDEN_TOOL_CALLED: " + ", ".join(forbidden_called)
        )

    if require_call:
        wanted = require_call if isinstance(require_call, (list, tuple)) else [require_call]
        missing_calls = [name for name in wanted if name not in calls]
        if missing_calls:
            reasons.append(
                "REQUIRED_CALL_ABSENT: " + ", ".join(missing_calls)
            )
    elif not any(name in CALL_REQUIRED_ANY for name in calls):
        reasons.append(
            "NO_EVIDENCE_CALL: the run called none of "
            + ", ".join(CALL_REQUIRED_ANY)
            + " — a grant that was never exercised proves nothing"
        )

    return {
        "transcript": str(path),
        "verdict": "VALID" if not reasons else "INVALID",
        "ok": not reasons,
        "granted": granted,
        "called": calls,
        "required": list(required),
        "missing_required": missing,
        "forbidden_granted": forbidden_granted,
        "forbidden_called": forbidden_called,
        "reasons": reasons,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcripts", nargs="+", help="session transcript JSONL file(s)")
    parser.add_argument(
        "--require",
        default=",".join(REQUIRED_TOOLS),
        help="comma-separated tool names that must be granted, or 'none'",
    )
    parser.add_argument(
        "--require-call",
        default="",
        help="comma-separated tool names that must appear as calls, or 'none'",
    )
    parser.add_argument("--json", action="store_true", help="print one JSON object per transcript")
    args = parser.parse_args(argv)

    required = () if args.require == "none" else tuple(
        name.strip() for name in args.require.split(",") if name.strip()
    )
    require_call = None
    if args.require_call == "none":
        require_call = ()
    elif args.require_call:
        require_call = tuple(
            name.strip() for name in args.require_call.split(",") if name.strip()
        )

    results = [check(Path(name), required, require_call) for name in args.transcripts]

    if args.json:
        for result in results:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        for result in results:
            print("%s  %s" % (result["verdict"], result["transcript"]))
            print("  granted: %s" % ", ".join(result["granted"]) or "  granted: (none)")
            print("  called : %s" % (", ".join(result["called"]) or "(none)"))
            for reason in result["reasons"]:
                print("  ! %s" % reason)

    failed = [result for result in results if not result["ok"]]
    if failed:
        print(
            "\nliterature agent transcript check: %d/%d INVALID"
            % (len(failed), len(results)),
            file=sys.stderr,
        )
        return 1
    print("\nliterature agent transcript check: OK (%d transcript(s))" % len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
