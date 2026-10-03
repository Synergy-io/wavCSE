"""Deterministic identity catalog for retained literature cards.

The catalog answers what a paper is and where its canonical card lives. It does
not encode Study-specific screening decisions or research interpretation.

Usage:
    python -m improvements.taskrelation.research.literature_catalog
    python -m improvements.taskrelation.research.literature_catalog --lookup <identity>
"""

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


SCHEMA_VERSION = 1
_ALLOWED_KEYS = {
    "schema_version",
    "paper_id",
    "card_path",
    "title",
    "year",
    "authors",
    "venue",
    "external_ids",
    "source_urls",
    "aliases",
}
_REQUIRED_KEYS = _ALLOWED_KEYS
_EXTERNAL_ID_TYPES = {"arxiv", "doi"}
_PAPER_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DOI = re.compile(r"^10\.\d{4,9}/[-._;()/:a-z0-9]+$", re.IGNORECASE)
_ARXIV = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$", re.IGNORECASE)
_CITATION_TITLE = re.compile(r"“([^”]+)”")
_INDEX_LINK = re.compile(r"\]\(([^)#?]+\.md)(?:#[^)]+)?\)")
_RESEARCH_DIR = Path(__file__).resolve().parent
_DEFAULT_REPO_ROOT = _RESEARCH_DIR.parents[2]
_DEFAULT_LITERATURE_DIR = _RESEARCH_DIR / "literature"
_DEFAULT_CATALOG = _DEFAULT_LITERATURE_DIR / "catalog.jsonl"


class CatalogError(ValueError):
    """The literature catalog cannot establish a unique paper identity."""


@dataclass(frozen=True)
class PaperEntry:
    schema_version: int
    paper_id: str
    card_path: str
    title: str
    year: int
    authors: tuple
    venue: str
    external_ids: dict
    source_urls: tuple
    aliases: tuple


class LiteratureCatalog:
    """Validated paper identities with deterministic exact-alias lookup."""

    def __init__(self, entries):
        self.entries = tuple(entries)
        self._paper_ids = {entry.paper_id: entry for entry in entries}
        self._aliases = {}
        for entry in entries:
            self._register_alias(entry.paper_id, entry)
            self._register_alias(_normalize_title(entry.title), entry)
            for alias in entry.aliases:
                self._register_alias(_normalize_title(alias), entry)
            for doi in entry.external_ids.get("doi", ()):
                self._register_alias(_normalize_doi(doi), entry)
            for arxiv_id in entry.external_ids.get("arxiv", ()):
                self._register_alias(_normalize_arxiv(arxiv_id), entry)
            for source_url in entry.source_urls:
                self._register_alias(_normalize_url(source_url), entry)

    def _register_alias(self, alias, entry):
        existing = self._aliases.get(alias)
        if existing is not None and existing.paper_id != entry.paper_id:
            raise CatalogError(
                "identity alias {!r} resolves to both {!r} and {!r}".format(
                    alias, existing.paper_id, entry.paper_id
                )
            )
        self._aliases[alias] = entry

    def get(self, paper_id):
        try:
            return self._paper_ids[paper_id]
        except KeyError as exc:
            raise CatalogError("unknown paper_id {!r}".format(paper_id)) from exc

    def lookup(self, identity):
        """Resolve an exact paper ID, title/alias, DOI, arXiv ID, or source URL."""

        if not isinstance(identity, str) or not identity.strip():
            raise CatalogError("paper lookup identity must be a non-empty string")
        query = identity.strip()
        candidates = [query, _normalize_title(query)]
        if _looks_like_doi(query):
            candidates.insert(0, _normalize_doi(query))
        if _looks_like_arxiv(query):
            candidates.insert(0, _normalize_arxiv(query))
        if _looks_like_url(query):
            candidates.insert(0, _normalize_url(query))
        for candidate in candidates:
            entry = self._aliases.get(candidate)
            if entry is not None:
                return entry
        raise CatalogError("unknown paper identity {!r}".format(identity))


def _normalize_title(value):
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized).split())


def _normalize_doi(value):
    normalized = value.strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return normalized.rstrip(".")


def _normalize_arxiv(value):
    normalized = value.strip().casefold()
    for prefix in (
        "https://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/abs/",
        "http://arxiv.org/pdf/",
        "arxiv:",
    ):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    normalized = normalized.removesuffix(".pdf")
    return re.sub(r"v\d+$", "", normalized)


def _normalize_url(value):
    return value.strip().rstrip("/")


def _looks_like_doi(value):
    return bool(
        _DOI.match(_normalize_doi(value))
        or value.casefold().startswith(("doi:", "https://doi.org/", "http://doi.org/"))
    )


def _looks_like_arxiv(value):
    return bool(
        _ARXIV.match(
            value.strip()
            .casefold()
            .removeprefix("arxiv:")
            .removesuffix(".pdf")
        )
        or "arxiv.org/" in value.casefold()
    )


def _looks_like_url(value):
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _require_string(record, key, where):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise CatalogError("{}.{} must be a non-empty trimmed string".format(where, key))
    return value


def _require_string_list(record, key, where, allow_empty=False):
    values = record.get(key)
    if not isinstance(values, list):
        raise CatalogError("{}.{} must be a list".format(where, key))
    if not allow_empty and not values:
        raise CatalogError("{}.{} must not be empty".format(where, key))
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise CatalogError(
                "{}.{}[{}] must be a non-empty trimmed string".format(
                    where, key, index
                )
            )
    if len(values) != len(set(values)):
        raise CatalogError("{}.{} contains duplicate values".format(where, key))
    return values


def _validate_record(record, line_number):
    where = "catalog line {}".format(line_number)
    if not isinstance(record, dict):
        raise CatalogError("{} must be a JSON object".format(where))
    unknown = sorted(set(record) - _ALLOWED_KEYS)
    missing = sorted(_REQUIRED_KEYS - set(record))
    if unknown:
        raise CatalogError(
            "{} has unknown key(s): {}".format(where, ", ".join(unknown))
        )
    if missing:
        raise CatalogError(
            "{} is missing key(s): {}".format(where, ", ".join(missing))
        )
    if record["schema_version"] != SCHEMA_VERSION:
        raise CatalogError(
            "{}.schema_version must be {}".format(where, SCHEMA_VERSION)
        )
    paper_id = _require_string(record, "paper_id", where)
    if not _PAPER_ID.match(paper_id):
        raise CatalogError(
            "{}.paper_id must be a lowercase hyphenated slug".format(where)
        )
    card_path = _require_string(record, "card_path", where)
    title = _require_string(record, "title", where)
    venue = _require_string(record, "venue", where)
    year = record["year"]
    if not isinstance(year, int) or isinstance(year, bool):
        raise CatalogError("{}.year must be an integer".format(where))
    if year < 1800 or year > 2100:
        raise CatalogError("{}.year is outside the supported range".format(where))
    authors = _require_string_list(record, "authors", where)
    aliases = _require_string_list(record, "aliases", where, allow_empty=True)
    source_urls = _require_string_list(record, "source_urls", where)
    for index, source_url in enumerate(source_urls):
        if not _looks_like_url(source_url):
            raise CatalogError(
                "{}.source_urls[{}] must be an http(s) URL".format(where, index)
            )
    external_ids = record["external_ids"]
    if not isinstance(external_ids, dict):
        raise CatalogError("{}.external_ids must be an object".format(where))
    unknown_id_types = sorted(set(external_ids) - _EXTERNAL_ID_TYPES)
    if unknown_id_types:
        raise CatalogError(
            "{}.external_ids has unknown type(s): {}".format(
                where, ", ".join(unknown_id_types)
            )
        )
    normalized_ids = {}
    for id_type in sorted(external_ids):
        values = external_ids[id_type]
        if not isinstance(values, list) or not values:
            raise CatalogError(
                "{}.external_ids.{} must be a non-empty list".format(where, id_type)
            )
        normalized = []
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise CatalogError(
                    "{}.external_ids.{}[{}] must be a non-empty trimmed string".format(
                        where, id_type, index
                    )
                )
            if id_type == "doi":
                canonical = _normalize_doi(value)
                if value != canonical or not _DOI.match(canonical):
                    raise CatalogError(
                        "{}.external_ids.doi[{}] must be a canonical DOI".format(
                            where, index
                        )
                    )
            else:
                canonical = value.casefold()
                if value != canonical or not _ARXIV.match(canonical):
                    raise CatalogError(
                        "{}.external_ids.arxiv[{}] must be a canonical arXiv ID".format(
                            where, index
                        )
                    )
            normalized.append(canonical)
        if normalized != sorted(set(normalized)):
            raise CatalogError(
                "{}.external_ids.{} must be unique and sorted".format(where, id_type)
            )
        normalized_ids[id_type] = tuple(normalized)
    return PaperEntry(
        schema_version=SCHEMA_VERSION,
        paper_id=paper_id,
        card_path=card_path,
        title=title,
        year=year,
        authors=tuple(authors),
        venue=venue,
        external_ids=normalized_ids,
        source_urls=tuple(source_urls),
        aliases=tuple(aliases),
    )


def _read_jsonl(path):
    records = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CatalogError("cannot read catalog {}: {}".format(path, exc)) from exc
    if not lines:
        raise CatalogError("catalog {} is empty".format(path))
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise CatalogError("catalog line {} is blank".format(line_number))
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise CatalogError(
                "catalog line {} is not valid JSON: {}".format(line_number, exc)
            ) from exc
        records.append(_validate_record(record, line_number))
    return records


def discover_card_ids(literature_dir):
    """Return canonical card slugs; INDEX.md is the one non-card Markdown file."""

    literature_dir = Path(literature_dir)
    return {
        path.stem
        for path in literature_dir.glob("*.md")
        if path.name != "INDEX.md"
    }


def _extract_card_title(card_text, paper_id):
    lines = card_text.splitlines()
    try:
        citation_index = lines.index("## Citation")
    except ValueError as exc:
        raise CatalogError(
            "card {!r} has no '## Citation' section".format(paper_id)
        ) from exc
    for line in lines[citation_index + 1:]:
        if line.startswith("## "):
            break
        match = _CITATION_TITLE.search(line)
        if match:
            return match.group(1)
    raise CatalogError(
        "card {!r} has no quoted title in its Citation section".format(paper_id)
    )


def _validate_card(entry, repo_root, literature_dir):
    card_path = repo_root / entry.card_path
    expected_path = literature_dir / (entry.paper_id + ".md")
    expected_relative = expected_path.relative_to(repo_root).as_posix()
    if entry.card_path != expected_relative:
        raise CatalogError(
            "paper {!r} card_path must be exactly {}".format(
                entry.paper_id, expected_relative
            )
        )
    try:
        card_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CatalogError(
            "paper {!r} card_path does not exist: {}".format(
                entry.paper_id, entry.card_path
            )
        ) from exc
    card_text = card_path.read_text(encoding="utf-8")
    card_title = _extract_card_title(card_text, entry.paper_id)
    if _normalize_title(card_title) != _normalize_title(entry.title):
        raise CatalogError(
            "paper {!r} title differs from its card Citation title".format(
                entry.paper_id
            )
        )
    for source_url in entry.source_urls:
        if source_url not in card_text:
            raise CatalogError(
                "paper {!r} source URL is not recorded in its card: {}".format(
                    entry.paper_id, source_url
                )
            )
    folded_card = card_text.casefold()
    for id_type, values in entry.external_ids.items():
        for value in values:
            if value.casefold() not in folded_card:
                raise CatalogError(
                    "paper {!r} {} is not recorded in its card: {}".format(
                        entry.paper_id, id_type, value
                    )
                )


def _validate_unique_identities(entries):
    seen_papers = {}
    seen_titles = {}
    seen_external_ids = {"doi": {}, "arxiv": {}}
    seen_urls = {}
    for entry in entries:
        if entry.paper_id in seen_papers:
            raise CatalogError("duplicate paper_id {!r}".format(entry.paper_id))
        seen_papers[entry.paper_id] = entry.paper_id
        title_key = (_normalize_title(entry.title), entry.year)
        if title_key in seen_titles:
            raise CatalogError(
                "duplicate title/year identity {!r} ({}) for {!r} and {!r}".format(
                    entry.title,
                    entry.year,
                    seen_titles[title_key],
                    entry.paper_id,
                )
            )
        seen_titles[title_key] = entry.paper_id
        for id_type, values in entry.external_ids.items():
            for value in values:
                key = _normalize_doi(value) if id_type == "doi" else _normalize_arxiv(value)
                previous = seen_external_ids[id_type].get(key)
                if previous is not None:
                    raise CatalogError(
                        "duplicate {} {!r} for {!r} and {!r}".format(
                            id_type, key, previous, entry.paper_id
                        )
                    )
                seen_external_ids[id_type][key] = entry.paper_id
        for source_url in entry.source_urls:
            key = _normalize_url(source_url)
            previous = seen_urls.get(key)
            if previous is not None:
                raise CatalogError(
                    "duplicate source URL {!r} for {!r} and {!r}".format(
                        key, previous, entry.paper_id
                    )
                )
            seen_urls[key] = entry.paper_id


def _validate_card_coverage(entries, literature_dir):
    catalog_ids = {entry.paper_id for entry in entries}
    card_ids = discover_card_ids(literature_dir)
    missing = sorted(card_ids - catalog_ids)
    extra = sorted(catalog_ids - card_ids)
    if missing:
        raise CatalogError(
            "canonical card missing from catalog: {}".format(", ".join(missing))
        )
    if extra:
        raise CatalogError(
            "catalog paper has no canonical card: {}".format(", ".join(extra))
        )


def _validate_index(entries, index_path):
    if index_path is None:
        return
    index_path = Path(index_path)
    try:
        text = index_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError("cannot read literature index {}: {}".format(index_path, exc)) from exc
    known = {entry.paper_id for entry in entries}
    for target in _INDEX_LINK.findall(text):
        target_path = (index_path.parent / target).resolve()
        if target_path.parent != index_path.parent.resolve():
            continue
        paper_id = target_path.stem
        if paper_id == "INDEX":
            continue
        if paper_id not in known:
            raise CatalogError(
                "literature index references uncataloged card {!r}".format(paper_id)
            )
        if not target_path.is_file():
            raise CatalogError(
                "literature index references missing card {!r}".format(paper_id)
            )


def load_catalog(
    catalog_path=_DEFAULT_CATALOG,
    *,
    repo_root=_DEFAULT_REPO_ROOT,
    literature_dir=None,
    index_path=None,
    validate_references=True
):
    """Load the catalog; optionally validate its repository references."""

    catalog_path = Path(catalog_path)
    repo_root = Path(repo_root).resolve()
    literature_dir = (
        Path(literature_dir)
        if literature_dir is not None
        else catalog_path.parent
    )
    index_path = (
        Path(index_path)
        if index_path is not None
        else literature_dir / "INDEX.md"
    )
    entries = _read_jsonl(catalog_path)
    if [entry.paper_id for entry in entries] != sorted(entry.paper_id for entry in entries):
        raise CatalogError("catalog entries must be sorted by paper_id")
    _validate_unique_identities(entries)
    if validate_references:
        for entry in entries:
            _validate_card(entry, repo_root, literature_dir)
        _validate_card_coverage(entries, literature_dir)
        _validate_index(entries, index_path)
    return LiteratureCatalog(entries)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=_DEFAULT_CATALOG,
        help="catalog JSONL path (default: repository literature/catalog.jsonl)",
    )
    parser.add_argument(
        "--lookup",
        help="resolve an exact paper_id, title/alias, DOI, arXiv ID, or source URL",
    )
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        catalog = load_catalog(args.catalog)
        if args.lookup:
            entry = catalog.lookup(args.lookup)
            print(
                json.dumps(
                    {"paper_id": entry.paper_id, "card_path": entry.card_path},
                    sort_keys=True,
                )
            )
        else:
            external_count = sum(
                len(values)
                for entry in catalog.entries
                for values in entry.external_ids.values()
            )
            print(
                "literature catalog: OK ({} papers, {} external identifiers)".format(
                    len(catalog.entries), external_count
                )
            )
        return 0
    except CatalogError as exc:
        print("literature catalog: ERROR: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
