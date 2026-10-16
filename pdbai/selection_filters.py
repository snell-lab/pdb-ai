"""The filters an archive search can add, each declared once.

A selection picks its cohort by technique and X-ray source. A range filter
narrows it by a number the archive records for each entry, such as the
data-collection temperature or the resolution. SEARCH_FILTERS declares every
one. The query, the command-line flag, the web form field and its JSON key,
the key in the selection record, and the text that describes the criteria
are all generated from a declaration. Adding a bound on a quantity declared
here is one line in SEARCH_FILTERS, and a new quantity is one declaration
more.

UNIPROT_ACCESSIONS is the one filter of another kind. It keeps the entries
whose polymer entities RCSB maps to any of a list of UniProt accessions,
which is how a cohort of the same proteins as another cohort is selected
without building the whole archive.

A range filter is strict. An entry may record a quantity more than once. It
holds a temperature for each data-collection record, and a joint
X-ray/neutron entry holds a resolution for each of its two data sets. Those
are measurements of different experiments, so none of them speaks for the
others, and the entry matches only when every value it records is in range.
An entry that mixes room-temperature and cryogenic data is therefore in
neither cohort. An entry that records no value never matches, since nothing
then shows what it holds.

A range filter narrows the search and nothing else. The technique and the
X-ray source are run options too, which intake checks again on each entry. A
range is not, so the columns of the dataset, such as ``temperature_K``,
remain the record of what each entry holds.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from .engines.core import camel_case
from .positive_numbers import parse_positive_number, positive_number

type Bound = Literal["min", "max"]

# What each bound reads as in a label and in the criteria text, and the
# RCSB Search API operator that matches a value on the wrong side of it.
_BOUND_LABELS: Mapping[Bound, str] = {"min": "Minimum", "max": "Maximum"}
_BOUND_PHRASES: Mapping[Bound, str] = {"min": "at least", "max": "at most"}
_OUT_OF_RANGE_OPERATORS: Mapping[Bound, str] = {"min": "less", "max": "greater"}


def _number_text(value: float) -> str:
    # A whole value is written without its ".0", so 273.0 reads "273".
    return str(int(value)) if value.is_integer() else repr(value)


def _terminal(attribute: str, operator: str, **parameters: object) -> dict[str, Any]:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, **parameters},
    }


@dataclass(frozen=True)
class SearchQuantity:
    """One number the archive records for an entry, which a search can bound."""

    # The name the quantity goes by in a filter's identifier and flag.
    id: str
    # The RCSB Search API attribute that holds the values.
    attribute: str
    # What the quantity is called in labels and in the criteria text.
    label: str
    # The unit as it is shown, as it ends an identifier, and as it is named
    # in a sentence. The identifier's suffix is ASCII and follows the
    # dataset's column names, such as ``temperature_K``.
    unit: str
    unit_suffix: str
    unit_name: str
    # What a user needs to know about the values, in whole sentences.
    note: str


@dataclass(frozen=True)
class RangeFilter:
    """One bound on a quantity, declared once for every place it appears."""

    quantity: SearchQuantity
    bound: Bound

    @property
    def id(self) -> str:
        """Return the identifier, which is also the selection record's key."""
        quantity = self.quantity
        return f"{self.bound}_{quantity.id}_{quantity.unit_suffix}"

    @property
    def json_key(self) -> str:
        """Return the camelCase key the web API uses for this filter."""
        return camel_case(self.id)

    @property
    def cli_flag(self) -> str:
        """Return the command-line spelling, which leaves the unit out."""
        return "--" + f"{self.bound}_{self.quantity.id}".replace("_", "-")

    @property
    def label(self) -> str:
        """Return the name a form shows beside this filter's field."""
        quantity = self.quantity
        return f"{_BOUND_LABELS[self.bound]} {quantity.label} ({quantity.unit})"

    @property
    def help(self) -> str:
        """Return what the filter keeps and leaves out, in whole sentences."""
        quantity = self.quantity
        return (
            f"Keep an entry only when every {quantity.label} it records is "
            f"{_BOUND_PHRASES[self.bound]} this many {quantity.unit_name}. An "
            f"entry that records none is left out. {quantity.note}"
        )

    def check(self, value: object, name: str | None = None) -> float:
        """Return ``value`` if this filter can hold it, or raise ValueError.

        The value must be a positive, finite number. No text is parsed, since
        the value comes from code or from storage. The error names ``name``,
        or the filter's identifier when none is given.
        """
        return positive_number(value, name or self.id)

    def parse(self, value: object, name: str | None = None) -> float:
        """Return the bound a person or a client supplied, or raise ValueError.

        The value may be a positive number or its plain decimal text. The
        error names ``name``, which is the flag or the key as the caller
        spelled it, or the filter's identifier when none is given.
        """
        return parse_positive_number(value, name or self.id)

    def describe(self, value: float) -> str:
        """Return the bound as criteria text, such as "temperature at least 273 K"."""
        quantity = self.quantity
        return (
            f"{quantity.label} {_BOUND_PHRASES[self.bound]} "
            f"{_number_text(value)} {quantity.unit}"
        )

    def out_of_range_node(self, value: float) -> dict[str, Any]:
        """Return the search node that refuses an entry with a value out of range.

        The node is a negated comparison. The archive matches a plain
        comparison when any one value of the entry meets it, which would let
        in an entry with another value out of range.
        """
        return _terminal(
            self.quantity.attribute,
            _OUT_OF_RANGE_OPERATORS[self.bound],
            value=value,
            negation=True,
        )


TEMPERATURE = SearchQuantity(
    id="temperature",
    attribute="diffrn.ambient_temp",
    label="temperature",
    unit="K",
    unit_suffix="K",
    unit_name="kelvin",
    note=(
        "The archive holds one data-collection temperature for each "
        "diffraction record of an entry."
    ),
)
RESOLUTION = SearchQuantity(
    id="resolution",
    attribute="rcsb_entry_info.resolution_combined",
    label="resolution",
    unit="Å",
    unit_suffix="A",
    unit_name="angstroms",
    note=(
        "A smaller resolution value is a finer resolution. The archive holds "
        "one resolution for each data set of an entry, so a joint "
        "X-ray/neutron entry has two and both must be in range."
    ),
)


# A UniProt accession, in the two forms UniProt defines, upper case. An
# isoform suffix such as -2 is read and dropped, since the archive maps an
# entity to the accession and names the isoform apart.
_UNIPROT_ACCESSION = re.compile(
    r"(?P<accession>[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
    r"(?:-[0-9]+)?"
)
# Whitespace, commas, and semicolons separate accessions in text.
_ACCESSION_SEPARATORS = re.compile(r"[\s,;]+")


@dataclass(frozen=True)
class AccessionFilter:
    """The filter that keeps the entries mapped to any of some UniProt accessions.

    It is declared once, like a range filter, and the flag, the key, the
    criteria text, and the query nodes come from it. Its value is a tuple of
    accessions, sorted and distinct, in upper case without isoform suffixes.
    """

    id: str = "uniprot_accessions"
    # The RCSB Search API attributes of a polymer entity's reference
    # sequences, which name the database beside each accession.
    attribute: str = (
        "rcsb_polymer_entity_container_identifiers."
        "reference_sequence_identifiers.database_accession"
    )
    database_attribute: str = (
        "rcsb_polymer_entity_container_identifiers."
        "reference_sequence_identifiers.database_name"
    )
    database_name: str = "UniProt"
    label: str = "UniProt accessions"
    note: str = (
        "An entry matches when RCSB maps any of its polymer entities to one of "
        "the accessions, as the dataset's uniprot_id column records. An "
        "isoform suffix is dropped, since the archive maps entities to the "
        "accession itself."
    )

    @property
    def json_key(self) -> str:
        """Return the camelCase key the web API uses for this filter."""
        return camel_case(self.id)

    @property
    def cli_flag(self) -> str:
        """Return the command-line spelling."""
        return "--" + self.id.replace("_", "-")

    @property
    def help(self) -> str:
        """Return what the filter keeps and leaves out, in whole sentences."""
        return (
            "Keep an entry only when one of its polymer entities is mapped to "
            f"one of these UniProt accessions. {self.note}"
        )

    def parse(self, value: object, name: str | None = None) -> tuple[str, ...]:
        """Return the accessions a person or a client supplied, or raise ValueError.

        The value may be text, with the accessions separated by whitespace,
        commas, or semicolons, or a list of such texts. Case is ignored, an
        isoform suffix is dropped, and the result is sorted and distinct. A
        value of another type, a token that is not a UniProt accession, and
        a value that names no accession at all raise ValueError naming
        ``name``, or the filter's identifier when none is given.
        """
        name = name or self.id
        texts = [value] if isinstance(value, str) else value
        if not isinstance(texts, (list, tuple)) or not all(
            isinstance(text, str) for text in cast(list[object], texts)
        ):
            raise ValueError(f"{name} must be text or a list of text; got {value!r}.")
        accessions: set[str] = set()
        for text in cast(list[str], texts):
            for token in _ACCESSION_SEPARATORS.split(text.strip()):
                if not token:
                    continue
                match = _UNIPROT_ACCESSION.fullmatch(token.upper())
                if match is None:
                    raise ValueError(
                        f"{name} must hold UniProt accessions; got {token!r}."
                    )
                accessions.add(match.group("accession"))
        if not accessions:
            raise ValueError(f"{name} names no UniProt accession; got {value!r}.")
        return tuple(sorted(accessions))

    def check(self, value: object, name: str | None = None) -> tuple[str, ...]:
        """Return ``value`` if this filter can hold it, or raise ValueError.

        The value must be a list or tuple of accessions as parse returns
        them, in upper case without suffixes, sorted and distinct, or empty
        for a filter not set. No text is parsed, since the value comes from
        code or from storage.
        """
        name = name or self.id
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, str) for item in cast(list[object], value)
        ):
            raise ValueError(
                f"{name} must be a list of UniProt accessions; got {value!r}."
            )
        accessions = tuple(cast(list[str], value))
        for accession in accessions:
            match = _UNIPROT_ACCESSION.fullmatch(accession)
            if match is None or match.group("accession") != accession:
                raise ValueError(
                    f"{name} must hold UniProt accessions in canonical form; "
                    f"got {accession!r}."
                )
        if list(accessions) != sorted(set(accessions)):
            raise ValueError(f"{name} must be sorted and distinct; got {value!r}.")
        return accessions

    def describe(self, accessions: tuple[str, ...]) -> str:
        """Return the filter as criteria text, naming one accession or counting them."""
        if len(accessions) == 1:
            return f"UniProt accession {accessions[0]}"
        return f"one of {len(accessions)} UniProt accessions"

    def nodes(self, accessions: tuple[str, ...]) -> list[dict[str, Any]]:
        """Return the search nodes that keep the entries mapped to the accessions.

        One node matches the accession among a reference sequence's
        identifiers and the other requires that the identifier is UniProt's,
        since the same attribute holds the accessions of other databases.
        """
        return [
            _terminal(self.attribute, "in", value=list(accessions)),
            _terminal(self.database_attribute, "exact_match", value=self.database_name),
        ]


UNIPROT_ACCESSIONS = AccessionFilter()


# Every range filter, in the order the flags, the form fields, and the
# criteria text list them. The bounds of one quantity stay together.
SEARCH_FILTERS: tuple[RangeFilter, ...] = (
    RangeFilter(TEMPERATURE, "min"),
    RangeFilter(TEMPERATURE, "max"),
    RangeFilter(RESOLUTION, "max"),
)
SEARCH_FILTER_IDS: tuple[str, ...] = tuple(
    search_filter.id for search_filter in SEARCH_FILTERS
)
# The quantities the filters bound, each once, in the order first declared.
SEARCH_QUANTITIES: tuple[SearchQuantity, ...] = tuple(
    {
        search_filter.quantity.id: search_filter.quantity
        for search_filter in SEARCH_FILTERS
    }.values()
)
_SEARCH_FILTERS_BY_ID: Mapping[str, RangeFilter] = {
    search_filter.id: search_filter for search_filter in SEARCH_FILTERS
}


def get_search_filter(filter_id: str) -> RangeFilter:
    """Return the range filter ``filter_id`` names.

    An identifier that names no filter raises ValueError.
    """
    try:
        return _SEARCH_FILTERS_BY_ID[filter_id]
    except KeyError:
        raise ValueError(
            f"A search filter must be one of {', '.join(SEARCH_FILTER_IDS)}; "
            f"got {filter_id!r}"
        ) from None


def _by_quantity(
    bounds: Mapping[str, float],
) -> list[tuple[SearchQuantity, list[tuple[RangeFilter, float]]]]:
    # Group the chosen bounds by quantity, both in declaration order.
    grouped: dict[str, tuple[SearchQuantity, list[tuple[RangeFilter, float]]]] = {}
    for search_filter in SEARCH_FILTERS:
        if search_filter.id in bounds:
            quantity = search_filter.quantity
            _, chosen = grouped.setdefault(quantity.id, (quantity, []))
            chosen.append((search_filter, bounds[search_filter.id]))
    return list(grouped.values())


def check_bounds(bounds: object) -> dict[str, float]:
    """Return the bounds a selection may hold, as a new dict, or raise ValueError.

    ``bounds`` maps filter identifiers to their values. The dict returned
    lists them in declaration order with each value as a float. A value that
    is not a mapping raises ValueError. So does an identifier that names no
    filter, and a value that is not a positive, finite number. So does a
    minimum above the maximum of the same quantity, which no entry could
    meet.
    """
    if not isinstance(bounds, Mapping):
        raise ValueError(
            f"bounds must map search filters to their values; got {bounds!r}"
        )
    given = cast(Mapping[object, object], bounds)
    for filter_id in given:
        # A key of another type names no filter either, and is refused with
        # the same message.
        get_search_filter(cast(str, filter_id))
    checked = {
        search_filter.id: search_filter.check(given[search_filter.id])
        for search_filter in SEARCH_FILTERS
        if search_filter.id in given
    }
    for quantity, chosen in _by_quantity(checked):
        values = {search_filter.bound: value for search_filter, value in chosen}
        if "min" in values and "max" in values and values["min"] > values["max"]:
            raise ValueError(
                f"The minimum {quantity.label} ({_number_text(values['min'])} "
                f"{quantity.unit}) is above the maximum "
                f"({_number_text(values['max'])} {quantity.unit}), so no entry "
                "can match."
            )
    return checked


def describe_bounds(bounds: Mapping[str, float]) -> tuple[str, ...]:
    """Return the bounds as criteria text, one phrase or more for each quantity.

    A quantity bounded on both sides reads as a range, such as "temperature
    273 to 310 K". ``bounds`` must be as check_bounds returns them.
    """
    phrases: list[str] = []
    for quantity, chosen in _by_quantity(bounds):
        values = {search_filter.bound: value for search_filter, value in chosen}
        if "min" in values and "max" in values:
            phrases.append(
                f"{quantity.label} {_number_text(values['min'])} to "
                f"{_number_text(values['max'])} {quantity.unit}"
            )
        else:
            phrases.extend(
                search_filter.describe(value) for search_filter, value in chosen
            )
    return tuple(phrases)


def range_filter_nodes(bounds: Mapping[str, float]) -> list[dict[str, Any]]:
    """Return the RCSB Search API nodes that express the bounds.

    Each bounded quantity gives one node requiring that the entry records a
    value, then one negated node for each bound, refusing an entry with any
    value beyond it. Together they match exactly the entries whose every
    recorded value is in range. ``bounds`` must be as check_bounds returns
    them.
    """
    nodes: list[dict[str, Any]] = []
    for quantity, chosen in _by_quantity(bounds):
        nodes.append(_terminal(quantity.attribute, "exists"))
        nodes.extend(
            search_filter.out_of_range_node(value) for search_filter, value in chosen
        )
    return nodes


__all__ = [
    "RESOLUTION",
    "UNIPROT_ACCESSIONS",
    "AccessionFilter",
    "SEARCH_FILTERS",
    "SEARCH_FILTER_IDS",
    "SEARCH_QUANTITIES",
    "TEMPERATURE",
    "Bound",
    "RangeFilter",
    "SearchQuantity",
    "check_bounds",
    "describe_bounds",
    "get_search_filter",
    "range_filter_nodes",
]
