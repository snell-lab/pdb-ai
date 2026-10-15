from __future__ import annotations

from typing import cast

import pytest

from pdbai.selection import Selection, SelectionResult, build_search_request


def test_selection_uses_the_archive_vocabulary_for_its_query() -> None:
    selection = Selection("x_ray_diffraction", "free_electron_laser")

    request = build_search_request(selection)

    assert selection.describe() == (
        "X-ray diffraction (XRD), X-ray free-electron laser (XFEL)"
    )
    assert request["return_type"] == "entry"
    assert request["request_options"] == {
        "return_all_hits": True,
        "results_content_type": ["experimental"],
    }
    query = request["query"]
    assert query["type"] == "group" and query["logical_operator"] == "and"
    assert [node["parameters"] for node in query["nodes"]] == [
        {
            "attribute": "exptl.method",
            "operator": "exact_match",
            "value": "X-RAY DIFFRACTION",
        },
        {
            "attribute": "diffrn_source.source",
            "operator": "exact_match",
            "value": "FREE ELECTRON LASER",
        },
    ]


def test_a_single_filter_is_sent_as_a_terminal_node() -> None:
    request = build_search_request(Selection("electron_microscopy"))

    assert request["query"]["type"] == "terminal"
    assert request["query"]["parameters"]["value"] == "ELECTRON MICROSCOPY"


def test_integrative_entries_are_selected_by_methodology_like_intake() -> None:
    request = build_search_request(Selection("integrative"))

    assert request["query"]["parameters"] == {
        "attribute": "rcsb_entry_info.structure_determination_methodology",
        "operator": "exact_match",
        "value": "integrative",
    }


def test_a_selection_is_described_by_its_technique_and_source_labels() -> None:
    assert Selection("x_ray_diffraction").describe() == "X-ray diffraction (XRD)"
    assert Selection("x_ray_diffraction", "synchrotron").describe() == (
        "X-ray diffraction (XRD), Synchrotron"
    )


def test_selection_rejects_what_the_run_options_reject() -> None:
    with pytest.raises(ValueError, match="applies only to technique"):
        Selection("solution_nmr", "synchrotron")
    with pytest.raises(ValueError, match="technique must be one of"):
        Selection("cryo")
    with pytest.raises(ValueError, match="xray_source must be one of"):
        Selection("x_ray_diffraction", "xfel")
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        Selection("x_ray_diffraction", limit=0)
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        Selection("x_ray_diffraction", limit="5")  # type: ignore[arg-type]


def test_selection_result_round_trips_and_validates_its_record() -> None:
    result = SelectionResult(
        filters=Selection("x_ray_diffraction", "free_electron_laser", limit=2),
        pdb_ids=("11YO", "13CH"),
        total_count=1041,
        service="rcsb_search",
        service_url="https://search.example/query",
        query={"query": {}},
        retrieved_at=1_700_000_000.0,
        truncated=True,
        warnings=("one warning",),
    )

    record = result.to_dict()
    rebuilt = SelectionResult.from_dict(record)

    assert result.truncated is True
    assert record["returned_count"] == 2
    assert record["retrieved_at_iso"] == "2023-11-14T22:13:20Z"
    assert record["description"] == result.filters.describe()
    assert rebuilt == result
    assert rebuilt.to_dict() == record


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ("nope", "must be a JSON object"),
        ({"filters": "x"}, "selection.filters must be a JSON object"),
        (
            {"filters": {"technique": "x_ray_diffraction"}, "pdb_ids": "11YO"},
            "pdb_ids must be a list",
        ),
        (
            {"filters": {"technique": "x_ray_diffraction"}, "pdb_ids": [None]},
            "pdb_ids must be a list of strings",
        ),
        (
            {
                "filters": {"technique": "x_ray_diffraction"},
                "pdb_ids": ["11YO"],
                "total_count": True,
            },
            "total_count must be a non-negative integer",
        ),
        (
            {
                "filters": {"technique": "x_ray_diffraction"},
                "pdb_ids": ["11YO"],
                "total_count": 1,
                "retrieved_at": "yesterday",
            },
            "retrieved_at must be a number",
        ),
        (
            {
                "filters": {"technique": "x_ray_diffraction"},
                "pdb_ids": ["11YO"],
                "total_count": 1,
                "retrieved_at": 1.0,
                "query": [],
            },
            "query must be a JSON object",
        ),
        (
            {
                "filters": {"technique": "x_ray_diffraction"},
                "pdb_ids": ["not-an-id"],
                "total_count": 1,
                "retrieved_at": 1.0,
                "query": {},
            },
            "Invalid PDB ID",
        ),
    ],
)
def test_selection_result_rejects_malformed_records(
    record: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SelectionResult.from_dict(record)


def _record(**changes: object) -> dict[str, object]:
    """Return a valid limited, truncated record with some fields changed."""
    record: dict[str, object] = {
        "filters": {"technique": "x_ray_diffraction", "limit": 2},
        "pdb_ids": ["11YO", "13CH"],
        "total_count": 5,
        "truncated": True,
        "service": "rcsb_search",
        "service_url": "https://search.example/query",
        "query": {},
        "retrieved_at": 1_700_000_000.0,
        "warnings": [],
    }
    record.update(changes)
    return record


def test_selection_result_accepts_the_valid_record() -> None:
    assert SelectionResult.from_dict(_record()).truncated is True


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"retrieved_at": float("nan")}, "not a time the platform can represent"),
        ({"retrieved_at": 1e300}, "not a time the platform can represent"),
        ({"retrieved_at": 10**400}, "not a time the platform can represent"),
        ({"retrieved_at": -1.0}, "must not be before 1970"),
        ({"pdb_ids": ["13CH", "11YO"]}, "sorted and distinct"),
        ({"pdb_ids": ["11YO", "11YO"]}, "sorted and distinct"),
        ({"total_count": 1}, "total_count is below the identifiers kept"),
        (
            {"pdb_ids": ["11YO", "13CH", "5MCC"]},
            "more identifiers than the limit",
        ),
        ({"truncated": "yes"}, "truncated must be true or false"),
        (
            {"filters": {"technique": "x_ray_diffraction"}},
            "the limit did not drop identifiers",
        ),
        ({"total_count": 2}, "the limit did not drop identifiers"),
        ({"pdb_ids": ["11YO"]}, "the limit did not drop identifiers"),
        (
            {"filters": {"technique": "x_ray_diffraction", "limit": "2"}},
            "limit must be a positive integer",
        ),
        (
            {"filters": {"technique": None, "limit": 2}},
            r"^selection\.filters\.technique must be a string",
        ),
        (
            {"filters": {"technique": "x_ray_diffraction", "xray_source": 3}},
            r"^selection\.filters\.xray_source must be a string",
        ),
        ({"service": None}, "service must be a string"),
        ({"service_url": 7}, "service_url must be a string"),
        ({"warnings": [None]}, "warnings must be a list of strings"),
        ({"warnings": "one"}, "warnings must be a list of strings"),
    ],
)
def test_selection_result_rejects_records_a_search_could_not_have_made(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SelectionResult.from_dict(_record(**changes))


def test_selection_result_takes_defaults_only_for_missing_fields() -> None:
    record = _record(truncated=False, total_count=2)
    for key in ("truncated", "service", "service_url", "warnings"):
        del record[key]
    record["filters"] = {"limit": 2}

    rebuilt = SelectionResult.from_dict(record)

    assert rebuilt.filters == Selection(limit=2)
    assert (rebuilt.truncated, rebuilt.service, rebuilt.warnings) == (False, "", ())


def test_range_filters_join_the_query_after_technique_and_source() -> None:
    selection = Selection(
        "x_ray_diffraction",
        "synchrotron",
        bounds={"max_resolution_A": 2.5, "min_temperature_K": 273},
    )

    query = build_search_request(selection)["query"]

    assert query["type"] == "group" and query["logical_operator"] == "and"
    assert [node["parameters"] for node in query["nodes"]] == [
        {
            "attribute": "exptl.method",
            "operator": "exact_match",
            "value": "X-RAY DIFFRACTION",
        },
        {
            "attribute": "diffrn_source.source",
            "operator": "exact_match",
            "value": "SYNCHROTRON",
        },
        {"attribute": "diffrn.ambient_temp", "operator": "exists"},
        {
            "attribute": "diffrn.ambient_temp",
            "operator": "less",
            "value": 273.0,
            "negation": True,
        },
        {"attribute": "rcsb_entry_info.resolution_combined", "operator": "exists"},
        {
            "attribute": "rcsb_entry_info.resolution_combined",
            "operator": "greater",
            "value": 2.5,
            "negation": True,
        },
    ]


def test_a_selection_without_bounds_asks_what_it_asked_before() -> None:
    assert build_search_request(Selection("neutron_diffraction")) == (
        build_search_request(Selection("neutron_diffraction", bounds={}))
    )
    assert build_search_request(Selection("neutron_diffraction"))["query"]["type"] == (
        "terminal"
    )


def test_a_selection_describes_its_bounds_after_technique_and_source() -> None:
    selection = Selection(
        "x_ray_diffraction",
        "synchrotron",
        bounds={"min_temperature_K": 273, "max_resolution_A": 2.5},
    )

    assert selection.describe() == (
        "X-ray diffraction (XRD), Synchrotron, temperature at least 273 K, "
        "resolution at most 2.5 Å"
    )
    assert Selection(
        "neutron_diffraction", bounds={"max_temperature_K": 120}
    ).describe() == ("Neutron diffraction, temperature at most 120 K")


def test_a_selection_copies_and_orders_its_bounds() -> None:
    given = {"max_resolution_A": 2, "min_temperature_K": 273}

    selection = Selection(bounds=given)
    given["max_temperature_K"] = 310

    assert list(selection.bounds.items()) == [
        ("min_temperature_K", 273.0),
        ("max_resolution_A", 2.0),
    ]
    assert selection == Selection(
        bounds={"min_temperature_K": 273.0, "max_resolution_A": 2.0}
    )
    assert selection != Selection()
    # A selection stays usable as a key, as it was before it held bounds.
    assert hash(selection) == hash(Selection())


def test_selection_rejects_bounds_no_search_could_use() -> None:
    with pytest.raises(ValueError, match="must be one of min_temperature_K"):
        Selection(bounds={"min_ph": 7})
    with pytest.raises(ValueError, match="max_resolution_A must be a positive number"):
        Selection(bounds={"max_resolution_A": -1})
    with pytest.raises(ValueError, match="is above the maximum"):
        Selection(bounds={"min_temperature_K": 300, "max_temperature_K": 100})
    with pytest.raises(ValueError, match="bounds must map search filters"):
        Selection(bounds=[273])  # type: ignore[arg-type]


def test_a_selection_record_holds_every_range_filter_set_or_not() -> None:
    assert Selection().to_dict() == {
        "technique": "x_ray_diffraction",
        "xray_source": "all",
        "limit": None,
        "min_temperature_K": None,
        "max_temperature_K": None,
        "max_resolution_A": None,
        "uniprot_accessions": None,
    }
    assert Selection(bounds={"max_temperature_K": 120}).to_dict() == {
        "technique": "x_ray_diffraction",
        "xray_source": "all",
        "limit": None,
        "min_temperature_K": None,
        "max_temperature_K": 120.0,
        "max_resolution_A": None,
        "uniprot_accessions": None,
    }
    assert Selection(uniprot_accessions=("P00698",)).to_dict()[
        "uniprot_accessions"
    ] == ["P00698"]


def test_the_accession_filter_joins_the_query_and_the_criteria() -> None:
    selection = Selection(
        "x_ray_diffraction",
        "synchrotron",
        bounds={"max_temperature_K": 120},
        uniprot_accessions=("P00698", "P0DTD1"),
    )

    nodes = build_search_request(selection)["query"]["nodes"]

    assert [node["parameters"] for node in nodes[-2:]] == [
        {
            "attribute": (
                "rcsb_polymer_entity_container_identifiers."
                "reference_sequence_identifiers.database_accession"
            ),
            "operator": "in",
            "value": ["P00698", "P0DTD1"],
        },
        {
            "attribute": (
                "rcsb_polymer_entity_container_identifiers."
                "reference_sequence_identifiers.database_name"
            ),
            "operator": "exact_match",
            "value": "UniProt",
        },
    ]
    assert selection.describe() == (
        "X-ray diffraction (XRD), Synchrotron, temperature at most 120 K, "
        "one of 2 UniProt accessions"
    )
    assert Selection(uniprot_accessions=("P00698",)).describe() == (
        "X-ray diffraction (XRD), UniProt accession P00698"
    )
    assert "in" not in [
        node["parameters"]["operator"]
        for node in build_search_request(
            Selection("neutron_diffraction", bounds={"max_temperature_K": 120})
        )["query"]["nodes"]
    ]


def test_a_selection_checks_its_accessions_as_stored_values() -> None:
    with pytest.raises(ValueError, match="canonical form; got 'p00698'"):
        Selection(uniprot_accessions=("p00698",))
    with pytest.raises(ValueError, match="sorted and distinct"):
        Selection(uniprot_accessions=("P0DTD1", "P00698"))
    with pytest.raises(ValueError, match="must be a list of UniProt accessions"):
        Selection(uniprot_accessions="P00698")  # type: ignore[arg-type]


def test_selection_result_round_trips_its_accessions_and_reads_old_records() -> None:
    result = SelectionResult(
        filters=Selection(uniprot_accessions=("P00698", "P0DTD1")),
        pdb_ids=("11YO",),
        total_count=1,
        service="rcsb_search",
        service_url="https://search.example/query",
        query={"query": {}},
        retrieved_at=1_700_000_000.0,
    )

    record = result.to_dict()
    rebuilt = SelectionResult.from_dict(record)

    assert record["filters"]["uniprot_accessions"] == ["P00698", "P0DTD1"]
    assert rebuilt == result
    old = _record(truncated=False, total_count=2)
    assert SelectionResult.from_dict(old).filters.uniprot_accessions == ()
    with pytest.raises(ValueError, match="uniprot_accessions must be a list"):
        SelectionResult.from_dict(
            _record(
                filters={
                    "technique": "x_ray_diffraction",
                    "limit": 2,
                    "uniprot_accessions": "P00698",
                }
            )
        )
    with pytest.raises(ValueError, match="canonical form"):
        SelectionResult.from_dict(
            _record(
                filters={
                    "technique": "x_ray_diffraction",
                    "limit": 2,
                    "uniprot_accessions": ["p00698"],
                }
            )
        )


def test_selection_result_round_trips_its_bounds() -> None:
    result = SelectionResult(
        filters=Selection(
            "x_ray_diffraction",
            "synchrotron",
            bounds={"min_temperature_K": 273, "max_resolution_A": 2.5},
        ),
        pdb_ids=("11YO", "13CH"),
        total_count=2,
        service="rcsb_search",
        service_url="https://search.example/query",
        query={"query": {}},
        retrieved_at=1_700_000_000.0,
    )

    record = result.to_dict()
    rebuilt = SelectionResult.from_dict(record)

    assert record["description"] == (
        "X-ray diffraction (XRD), Synchrotron, temperature at least 273 K, "
        "resolution at most 2.5 Å"
    )
    assert rebuilt == result
    assert rebuilt.to_dict() == record


def test_a_record_written_before_the_range_filters_reads_as_a_search_without_them() -> (
    None
):
    record = _record(truncated=False, total_count=2)
    assert "min_temperature_K" not in cast(dict[str, object], record["filters"])

    rebuilt = SelectionResult.from_dict(record)

    assert rebuilt.filters == Selection(limit=2)
    assert rebuilt.filters.bounds == {}


@pytest.mark.parametrize(
    ("stored", "message"),
    [
        ({"min_temperature_K": "273"}, "min_temperature_K must be a positive number"),
        ({"max_resolution_A": True}, "max_resolution_A must be a positive number"),
        ({"max_temperature_K": 0}, "max_temperature_K must be a positive number"),
        (
            {"min_temperature_K": 300, "max_temperature_K": 100},
            "is above the maximum",
        ),
    ],
)
def test_selection_result_rejects_stored_bounds_a_search_could_not_have_used(
    stored: dict[str, object], message: str
) -> None:
    filters = {"technique": "x_ray_diffraction", "limit": 2, **stored}

    with pytest.raises(ValueError, match=message):
        SelectionResult.from_dict(_record(filters=filters))
