"""CLI for structured scholarly discovery.

    python -m improvements.taskrelation.research.literature_discovery \
        discover --doi 10.1145/3580305.3599261
    python -m improvements.taskrelation.research.literature_discovery \
        discover --arxiv 2410.15875
    python -m improvements.taskrelation.research.literature_discovery \
        discover --title "Automatic Temporal Relation in Multi-Task Learning"
    python -m improvements.taskrelation.research.literature_discovery \
        search "multi-task relationship learning gradient interference"
    python -m improvements.taskrelation.research.literature_discovery references zhang-yeung-2014-mtrl-asymmetric
    python -m improvements.taskrelation.research.literature_discovery citations zhang-yeung-2014-mtrl-asymmetric
    python -m improvements.taskrelation.research.literature_discovery providers

Output is normalized JSON on stdout. Provider failures (rate limiting, a missing
record, a provider outage) are part of a well-formed result and do not change the
exit status; only a malformed request does.
"""

import argparse
import json
import sys

from .core import DiscoveryError, StructuredDiscovery
from .model import INVALID_QUERY


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover", help="resolve an identifier or near-exact title")
    discover.add_argument("--doi")
    discover.add_argument("--arxiv")
    discover.add_argument("--title")
    discover.add_argument("--author", action="append", default=[])
    discover.add_argument("--venue")
    discover.add_argument("--year", type=int)
    discover.add_argument("--limit", type=int)

    search = sub.add_parser("search", help="search providers by a scholarly query")
    search.add_argument("query")
    search.add_argument("--year", type=int)
    search.add_argument("--limit", type=int)

    references = sub.add_parser("references", help="expand a paper's references")
    references.add_argument("paper_id")
    references.add_argument("--limit", type=int)

    citations = sub.add_parser("citations", help="expand a paper's citations")
    citations.add_argument("paper_id")
    citations.add_argument("--limit", type=int)

    sub.add_parser("providers", help="report provider availability and capabilities")
    return parser


def _emit(payload, ok):
    stream = sys.stdout if ok else sys.stderr
    stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return 0 if ok else 1


def main(argv=None):
    args = _build_parser().parse_args(argv)

    if args.command == "providers":
        return _emit(
            {"kind": "PROVIDERS", "providers": list(StructuredDiscovery().provider_status())},
            True,
        )

    discovery = StructuredDiscovery()
    try:
        if args.command == "discover":
            result = discovery.discover(
                doi=args.doi, arxiv=args.arxiv, title=args.title,
                authors=args.author, venue=args.venue, year=args.year,
                limit=args.limit,
            )
        elif args.command == "search":
            result = discovery.search(args.query, limit=args.limit, year=args.year)
        elif args.command == "references":
            result = discovery.references(args.paper_id, limit=args.limit)
        else:
            result = discovery.citations(args.paper_id, limit=args.limit)
    except DiscoveryError as exc:
        return _emit({"kind": INVALID_QUERY, "message": str(exc)}, False)

    payload = result.as_dict()
    if result.ok:
        return _emit(payload, True)
    payload["kind"] = INVALID_QUERY
    return _emit(payload, False)


if __name__ == "__main__":
    sys.exit(main())
