from __future__ import annotations

from collections import Counter

import pytest

from pdbai import selection_filters
from pdbai.selection_filters import (
    RESOLUTION,
    SEARCH_FILTER_IDS,
    SEARCH_FILTERS,
    SEARCH_QUANTITIES,
    TEMPERATURE,
    UNIPROT_ACCESSIONS,
    RangeFilter,
    check_bounds,
    describe_bounds,
    get_search_filter,
    range_filter_nodes,
)


def _parameters(bounds: dict[str, float]) -> list[dict[str, object]]:
    return [node["parameters"] for node in range_filter_nodes(check_bounds(bounds))]


def test_the_first_filters_bound_temperature_and_resolution() -> None:
    assert [
        (search_filter.id, search_filter.cli_flag, search_filter.json_key)
        for search_filter in SEARCH_FILTERS
    ] == [
        ("min_temperature_K", "--min-temperature", "minTemperatureK"),
        ("max_temperature_K", "--max-temperature", "maxTemperatureK"),
        ("max_resolution_A", "--max-resolution", "maxResolutionA"),
    ]
    assert tuple(f.id for f in SEARCH_FILTERS) == SEARCH_FILTER_IDS
    assert SEARCH_QUANTITIES == (TEMPERATURE, RESOLUTION)


def test_every_filter_has_its_own_identifier_flag_and_key() -> None:
    for spelling in ("id", "cli_flag", "json_key", "label"):
        counts = Counter(getattr(f, spelling) for f in SEARCH_FILTERS)
        assert [name for name, count in counts.items() if count > 1] == []
    # An identifier ends in its unit, as the dataset's column names do.
    assert all(f.id.endswith(f"_{f.quantity.unit_suffix}") for f in SEARCH_FILTERS)
    assert all(f.quantity.unit_suffix.isascii() for f in SEARCH_FILTERS)


def test_a_filter_declared_twice_is_refused() -> None:
    twice = (RangeFilter(TEMPERATURE, "min"), RangeFilter(TEMPERATURE, "min"))

    with pytest.raises(ValueError, match="declared twice: min_temperature_K"):
        selection_filters._checked_filters(twice)  # pyright: ignore[reportPrivateUsage]


def test_a_filter_says_what_it_keeps_and_what_it_leaves_out() -> None:
    minimum = get_search_filter("min_temperature_K")
    maximum = get_search_filter("max_resolution_A")

    assert minimum.label == "Minimum temperature (K)"
    assert maximum.label == "Maximum resolution (Å)"
    assert minimum.help.startswith(
        "Keep an entry only when every temperature it records is at least this "
        "many kelvin. An entry that records none is left out."
    )
    assert maximum.help.startswith(
        "Keep an entry only when every resolution it records is at most this "
        "many angstroms. An entry that records none is left out."
    )
    assert maximum.help.endswith(RESOLUTION.note)


def test_get_search_filter_refuses_an_unknown_identifier() -> None:
    with pytest.raises(ValueError, match="must be one of min_temperature_K, .*'ph'"):
        get_search_filter("ph")


def test_a_filter_parses_input_and_checks_stored_values_by_name() -> None:
    search_filter = get_search_filter("max_resolution_A")

    assert search_filter.parse("2.5") == 2.5
    assert search_filter.check(2) == 2.0
    with pytest.raises(ValueError, match="^--max-resolution must be a positive"):
        search_filter.parse("fine", "--max-resolution")
    with pytest.raises(ValueError, match="^max_resolution_A must be a positive"):
        search_filter.check("2.5")


def test_check_bounds_returns_floats_in_declaration_order() -> None:
    given = {"max_resolution_A": 2, "min_temperature_K": 273}

    checked = check_bounds(given)

    assert list(checked.items()) == [
        ("min_temperature_K", 273.0),
        ("max_resolution_A", 2.0),
    ]
    assert all(type(value) is float for value in checked.values())
    assert checked is not given
    assert check_bounds({}) == {}


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ([("min_temperature_K", 273)], "bounds must map search filters"),
        ({"min_temp": 273}, "must be one of min_temperature_K"),
        ({3: 273}, "must be one of min_temperature_K"),
        ({"min_temperature_K": None}, "min_temperature_K must be a positive number"),
        ({"min_temperature_K": "273"}, "min_temperature_K must be a positive number"),
        ({"min_temperature_K": True}, "min_temperature_K must be a positive number"),
        ({"max_resolution_A": 0}, "max_resolution_A must be a positive number"),
        (
            {"min_temperature_K": 300, "max_temperature_K": 280.5},
            r"minimum temperature \(300 K\) is above the maximum \(280.5 K\)",
        ),
    ],
)
def test_check_bounds_refuses_what_no_search_could_use(
    bounds: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        check_bounds(bounds)


def test_a_minimum_may_equal_its_maximum() -> None:
    bounds = {"min_temperature_K": 100, "max_temperature_K": 100}

    assert describe_bounds(check_bounds(bounds)) == ("temperature 100 to 100 K",)


def test_bounds_are_described_one_quantity_at_a_time() -> None:
    assert describe_bounds({}) == ()
    assert describe_bounds(check_bounds({"min_temperature_K": 273})) == (
        "temperature at least 273 K",
    )
    assert describe_bounds(
        check_bounds({"max_resolution_A": 2.5, "max_temperature_K": 120})
    ) == ("temperature at most 120 K", "resolution at most 2.5 Å")
    assert describe_bounds(
        check_bounds(
            {
                "min_temperature_K": 273.15,
                "max_temperature_K": 310,
                "max_resolution_A": 2,
            }
        )
    ) == ("temperature 273.15 to 310 K", "resolution at most 2 Å")


def test_a_bound_is_sent_as_a_value_that_exists_and_is_never_out_of_range() -> None:
    # The archive matches a comparison when any one value of an entry meets
    # it. So each bound is sent negated, and no value may be out of range.
    assert _parameters({"min_temperature_K": 273}) == [
        {"attribute": "diffrn.ambient_temp", "operator": "exists"},
        {
            "attribute": "diffrn.ambient_temp",
            "operator": "less",
            "value": 273.0,
            "negation": True,
        },
    ]
    # A joint X-ray/neutron entry records two resolutions, and both must pass.
    assert _parameters({"max_resolution_A": 2.5}) == [
        {"attribute": "rcsb_entry_info.resolution_combined", "operator": "exists"},
        {
            "attribute": "rcsb_entry_info.resolution_combined",
            "operator": "greater",
            "value": 2.5,
            "negation": True,
        },
    ]


def test_two_bounds_on_one_quantity_share_its_exists_node() -> None:
    parameters = _parameters(
        {"max_temperature_K": 310, "min_temperature_K": 273, "max_resolution_A": 2}
    )

    assert [(p["attribute"], p["operator"]) for p in parameters] == [
        ("diffrn.ambient_temp", "exists"),
        ("diffrn.ambient_temp", "less"),
        ("diffrn.ambient_temp", "greater"),
        ("rcsb_entry_info.resolution_combined", "exists"),
        ("rcsb_entry_info.resolution_combined", "greater"),
    ]
    assert all(p.get("negation") is True for p in parameters if "value" in p)
    assert range_filter_nodes({}) == []


@pytest.mark.parametrize("search_filter", SEARCH_FILTERS, ids=SEARCH_FILTER_IDS)
def test_every_declared_filter_is_strict_and_reaches_the_criteria(
    search_filter: RangeFilter,
) -> None:
    bounds = check_bounds({search_filter.id: 5})

    exists, out_of_range = range_filter_nodes(bounds)

    # An entry that records no value never matches.
    assert exists["parameters"] == {
        "attribute": search_filter.quantity.attribute,
        "operator": "exists",
    }
    assert out_of_range == search_filter.out_of_range_node(5.0)
    assert out_of_range["type"] == "terminal" and out_of_range["service"] == "text"
    assert out_of_range["parameters"]["negation"] is True
    assert out_of_range["parameters"]["operator"] == (
        "less" if search_filter.bound == "min" else "greater"
    )
    assert describe_bounds(bounds) == (search_filter.describe(5.0),)
    assert search_filter.describe(5.0).endswith(f"5 {search_filter.quantity.unit}")


def test_the_accession_filter_is_declared_beside_the_range_filters() -> None:
    assert (
        UNIPROT_ACCESSIONS.id,
        UNIPROT_ACCESSIONS.json_key,
        UNIPROT_ACCESSIONS.cli_flag,
        UNIPROT_ACCESSIONS.label,
    ) == (
        "uniprot_accessions",
        "uniprotAccessions",
        "--uniprot-accessions",
        "UniProt accessions",
    )
    assert UNIPROT_ACCESSIONS.id not in SEARCH_FILTER_IDS
    assert UNIPROT_ACCESSIONS.help.startswith("Keep an entry only when one of its")
    assert UNIPROT_ACCESSIONS.help.endswith(UNIPROT_ACCESSIONS.note)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("P00698", ("P00698",)),
        (" p00698 ", ("P00698",)),
        (
            "P0DTD1 P00698\nP00698;A0A024R161,Q9Y6K9-3",
            ("A0A024R161", "P00698", "P0DTD1", "Q9Y6K9"),
        ),
        (["P00698", "P0DTD1, P00698"], ("P00698", "P0DTD1")),
    ],
)
def test_accessions_are_read_from_text_in_any_separation_and_case(
    value: object, expected: tuple[str, ...]
) -> None:
    assert UNIPROT_ACCESSIONS.parse(value) == expected


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "names no UniProt accession"),
        (" , ", "names no UniProt accession"),
        ([], "names no UniProt accession"),
        ("P00698 lysozyme", "must hold UniProt accessions; got 'lysozyme'"),
        ("1ABC", "must hold UniProt accessions; got '1ABC'"),
        ("P0069", "must hold UniProt accessions"),
        (5, "must be text or a list of text"),
        (["P00698", 5], "must be text or a list of text"),
    ],
)
def test_accessions_that_cannot_be_searched_are_refused_naming_the_key(
    value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=f"^uniprotAccessions {message}"):
        UNIPROT_ACCESSIONS.parse(value, "uniprotAccessions")


def test_stored_accessions_are_checked_not_parsed() -> None:
    assert UNIPROT_ACCESSIONS.check(["P00698", "P0DTD1"]) == ("P00698", "P0DTD1")
    assert UNIPROT_ACCESSIONS.check(()) == ()
    for bad, message in (
        ("P00698", "must be a list"),
        (["p00698"], "canonical form"),
        (["P00698-2"], "canonical form"),
        (["P0DTD1", "P00698"], "sorted and distinct"),
        (["P00698", "P00698"], "sorted and distinct"),
    ):
        with pytest.raises(ValueError, match=message):
            UNIPROT_ACCESSIONS.check(bad)


def test_the_accession_filter_names_one_accession_or_counts_them() -> None:
    assert UNIPROT_ACCESSIONS.describe(("P00698",)) == "UniProt accession P00698"
    assert UNIPROT_ACCESSIONS.describe(("P00698", "P0DTD1")) == (
        "one of 2 UniProt accessions"
    )


def test_the_accession_filter_matches_uniprot_identifiers_only() -> None:
    nodes = UNIPROT_ACCESSIONS.nodes(("P00698", "P0DTD1"))

    assert [
        (
            n["parameters"]["attribute"].rsplit(".", 1)[1],
            n["parameters"]["operator"],
            n["parameters"]["value"],
        )
        for n in nodes
    ] == [
        ("database_accession", "in", ["P00698", "P0DTD1"]),
        ("database_name", "exact_match", "UniProt"),
    ]
