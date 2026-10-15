"""Reading a column filter request, matching it to columns, and applying it.

A column filter keeps only the columns a request names, in include mode, or
removes them, in exclude mode. The command line, the Python API, the build
form, and the CSV filter tool all go through this module. So they read, match,
and apply a request in the same way. A malformed request raises ValueError as
soon as it is read. The web app can then answer it with 400 instead of
starting a job that would fail later.

Names are matched to the columns of the table at hand, not to a fixed list.
The columns depend on the engines a run enables and on what each file holds.
So a name that matches no column is not an error. It is left out and named in
a warning, since the next dataset may have that column. A name that matches
several columns once case and punctuation are ignored is treated the same way.
Picking one of them would be a guess.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from typing import Literal, get_args

import pandas as pd

# Whether a column filter keeps only the requested columns or removes them.
type ColumnFilterMode = Literal["include", "exclude"]

COLUMN_FILTER_MODES: tuple[ColumnFilterMode, ...] = get_args(ColumnFilterMode.__value__)

_COLUMN_LIST_DELIMITERS = re.compile(r"[\r\n,;]+")
_NON_ALPHANUMERIC_RUN = re.compile(r"[^a-z0-9]+")


def parse_column_filter_mode(
    value: object, key: str = "column_filter_mode"
) -> ColumnFilterMode | None:
    """Return the column filter mode a request names, or None if it names none.

    Case and surrounding whitespace are ignored. None or a blank value names
    no mode. A value other than include or exclude raises ValueError, and the
    message calls the field ``key``.
    """
    text = str(value).strip().lower() if value is not None else ""
    if not text:
        return None
    if text not in COLUMN_FILTER_MODES:
        raise ValueError(
            f"{key} must be one of {', '.join(COLUMN_FILTER_MODES)}; got {value!r}."
        )
    return text


def parse_column_list(column_list: str | Iterable[str] | None) -> list[str]:
    """Split a requested column list into names, in order, dropping blanks.

    The list is one string or several, as a command line passes one per
    argument. Within a string, names are separated by commas, semicolons, or
    line breaks. Repeated names are kept. None requests no columns.
    """
    if column_list is None:
        return []
    texts = [column_list] if isinstance(column_list, str) else column_list
    parts = (
        part.strip() for text in texts for part in _COLUMN_LIST_DELIMITERS.split(text)
    )
    return [part for part in parts if part]


def parse_column_filter(
    mode: object,
    column_list: str | Iterable[str] | None,
    *,
    mode_key: str = "column_filter_mode",
    list_key: str = "column_filter_list",
) -> tuple[ColumnFilterMode | None, list[str]]:
    """Return the column filter a request names, as its mode and its columns.

    A filter needs both a mode and at least one column. A request that names
    neither gets None and no columns. A request that names only one of them
    raises ValueError, and so does an unknown mode. The error messages call
    the two fields mode_key and list_key, so each front end can use the names
    its users know, such as columnFilterMode in the web app.
    """
    parsed_mode = parse_column_filter_mode(mode, mode_key)
    columns = parse_column_list(column_list)
    if (parsed_mode is None) != (not columns):
        raise ValueError(
            f"{mode_key} and {list_key} must be given together; got mode "
            f"{parsed_mode!r} and columns {columns!r}."
        )
    return parsed_mode, columns


def matches_glob(column: str, glob: str) -> bool:
    """Return whether ``column`` fits ``glob``, ignoring case.

    Only * is a wildcard, and it stands for any run of characters. Every other
    character is literal. mmCIF names such as _refine.aniso_B[1][1] contain
    brackets, and fnmatch would read those as a character class.
    """
    pattern = ".*".join(re.escape(part) for part in glob.split("*"))
    return re.fullmatch(pattern, column, re.IGNORECASE) is not None


def _column_lookup_key(name: str) -> str:
    # The key keeps the name's letters and digits in lower case. Each run of
    # other characters becomes one underscore, and none is left at either end.
    # So "_cell.length_a" and "Cell Length A" get the same key.
    return _NON_ALPHANUMERIC_RUN.sub("_", name.lower()).strip("_")


def _folded_matches(available: Iterable[str]) -> Callable[[str], list[str]]:
    # The returned function gives the columns that a name matches when case
    # is ignored. If there are none, it gives the columns with the same
    # letters and digits. When several columns share a fold, it gives them
    # all, so the caller can refuse to guess. Each column appears once, in
    # available order.
    by_lowercase: dict[str, list[str]] = {}
    by_lookup_key: dict[str, list[str]] = {}
    for column in dict.fromkeys(available):
        by_lowercase.setdefault(column.lower(), []).append(column)
        by_lookup_key.setdefault(_column_lookup_key(column), []).append(column)

    def matches(name: str) -> list[str]:
        lowercase_matches = by_lowercase.get(name.lower())
        if lowercase_matches:
            return lowercase_matches
        key = _column_lookup_key(name)
        # A name with no letters or digits, such as "-", has an empty key. It
        # must not match a column such as "_" whose key is empty too.
        return by_lookup_key.get(key, []) if key else []

    return matches


def resolve_requested_columns(
    available: Sequence[str], requested: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Return the columns the requests match, and the requests that match none.

    A request that contains * is a glob. It matches every column it fits, as
    matches_glob decides. Any other request matches the column spelled the
    same. Failing that, it matches one spelled the same in another case, and
    failing that, one with the same letters and digits. So _cell_length_a
    finds _cell.length_a. A request that matches several columns in either
    fallback matches none of them. Each matched column appears once, in the
    order of the requests. The columns of one glob keep their available order.
    """
    exact_names = set(available)
    folded_matches = _folded_matches(available)
    # Dict keys hold each matched column once, in the order first matched.
    matched: dict[str, None] = {}
    missing: list[str] = []
    for request in requested:
        if "*" in request:
            columns = [column for column in available if matches_glob(column, request)]
        elif request in exact_names:
            columns = [request]
        else:
            # This runs for every entry of a build, so an ambiguous name is
            # not raised here. It is left unmatched like an unknown name, and
            # the run warns of both once it settles.
            columns = folded_matches(request)
            if len(columns) > 1:
                columns = []
        if columns:
            matched.update(dict.fromkeys(columns))
        else:
            missing.append(request)
    return list(matched), missing


def filter_dataframe_columns(
    dataframe: pd.DataFrame, mode: ColumnFilterMode, requested: Iterable[str]
) -> pd.DataFrame:
    """Return ``dataframe`` with the requested columns kept or removed.

    The requests are matched by resolve_requested_columns, and kept_columns
    decides which columns stay.
    """
    available = list(dataframe.columns)
    matched, _missing = resolve_requested_columns(available, requested)
    return dataframe[kept_columns(available, mode, matched)]


def kept_columns(
    available: Sequence[str], mode: ColumnFilterMode, matched: Iterable[str]
) -> list[str]:
    """Return the columns a filter keeps, given those its request matched.

    Include keeps the matched columns in the order given. Exclude keeps every
    other available column in available order, so it keeps them all when
    nothing matched. A filter must not leave a table without columns. That
    happens when include matches nothing or exclude matches everything. The
    filter then keeps only pdb_id, so each row still names its structure. If
    there is no pdb_id, it keeps every column. Both the build and the CSV
    filter tool choose their columns here.
    """
    if mode == "include":
        kept = list(matched)
    else:
        dropped = set(matched)
        kept = [column for column in available if column not in dropped]
    if kept:
        return kept
    return ["pdb_id"] if "pdb_id" in available else list(available)


def missing_columns_warning(missing: Iterable[str], available: Iterable[str]) -> str:
    """Return the warning naming the requested columns that matched nothing.

    Each name is listed once, in sorted order. Some names match several of
    the ``available`` columns once case and punctuation are ignored. Those
    are listed apart, each with the columns it could mean, so the user can
    spell out the one they want.
    """
    folded_matches = _folded_matches(available)
    not_found: list[str] = []
    ambiguous: list[str] = []
    for name in sorted(set(missing)):
        candidates = [] if "*" in name else folded_matches(name)
        if len(candidates) > 1:
            ambiguous.append(f"{name} ({', '.join(candidates)})")
        else:
            not_found.append(name)
    parts: list[str] = []
    if not_found:
        parts.append(
            "The following columns were not found and were ignored: "
            + ", ".join(not_found)
        )
    if ambiguous:
        parts.append(
            "The following names match several columns ignoring case or "
            "punctuation and were ignored: " + "; ".join(ambiguous)
        )
    return ". ".join(parts)


__all__ = [
    "COLUMN_FILTER_MODES",
    "ColumnFilterMode",
    "filter_dataframe_columns",
    "kept_columns",
    "matches_glob",
    "missing_columns_warning",
    "parse_column_filter",
    "parse_column_filter_mode",
    "parse_column_list",
    "resolve_requested_columns",
]
