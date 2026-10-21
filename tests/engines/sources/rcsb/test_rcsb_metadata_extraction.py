from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, cast

import pytest
from fake_http import FakeResponse, FakeSession

from pdbai.engines.sources.http_cache import CACHE_TTL_SECONDS
from pdbai.engines.sources.rcsb.extraction import (
    extract_beamline_and_wavelength,
    extract_citation_fields,
    extract_experiment_fields,
)
from pdbai.engines.sources.rcsb.metadata import fetch_rcsb_metadata

# This is the RCSB Data API entry record for 5MCC, trimmed to the fields the
# extractors read. The expected values below are the ones the entry publishes.
RECORDED_5MCC_ENTRY: dict[str, Any] = {
    "diffrn": [{"ambient_temp": 100.0}],
    "diffrn_detector": [{"detector": "PIXEL", "type": "PSI PILATUS 6M"}],
    "diffrn_source": [
        {
            "pdbx_synchrotron_beamline": "I02",
            "pdbx_synchrotron_site": "Diamond",
            "pdbx_wavelength_list": "0.980",
            "source": "SYNCHROTRON",
        }
    ],
    "exptl": [{"crystals_number": 1, "method": "X-RAY DIFFRACTION"}],
    "exptl_crystal_grow": [
        {"method": "VAPOR DIFFUSION, HANGING DROP", "pH": 4.0, "temp": 289.0}
    ],
    "rcsb_accession_info": {"deposit_date": "2016-11-09T00:00:00.000+00:00"},
    "rcsb_entry_info": {"structure_determination_methodology": "experimental"},
    "rcsb_primary_citation": {
        "journal_abbrev": "J Synchrotron Radiat",
        "pdbx_database_id_DOI": "10.1107/S1600577516016775",
        "pdbx_database_id_PubMed": 28009542,
        "title": "OH cleavage from tyrosine: debunking a myth.",
        "year": 2017,
    },
    "refine": [
        {
            "ls_R_factor_R_free": 0.1728,
            "ls_R_factor_R_work": 0.1513,
            "ls_d_res_high": 2.0,
        }
    ],
    "reflns": [
        {
            "pdbx_CC_half": 0.997,
            "pdbx_netI_over_sigmaI": 12.2,
            "pdbx_redundancy": 3.3,
            "percent_possible_obs": 99.8,
        }
    ],
    "struct": {
        "title": (
            "Radiation damage to GH7 Family Cellobiohydrolase from "
            "Daphnia pulex: Dose (DWD) 1.11 MGy"
        )
    },
    "symmetry": {"space_group_name_H_M": "C 1 2 1"},
}


def test_extraction_matches_the_published_values_for_a_real_entry() -> None:
    result = extract_experiment_fields(RECORDED_5MCC_ENTRY)

    assert result["r_work"] == 0.1513
    assert result["r_free"] == 0.1728
    assert result["space_group"] == "C 1 2 1"
    assert result["cc_half"] == 0.997
    assert result["i_over_sigma"] == 12.2
    assert result["reflns_completeness"] == 99.8
    assert result["multiplicity"] == 3.3
    assert result["resolution_rcsb"] == 2.0
    assert result["crystal_pH"] == 4.0
    assert result["num_crystals"] == 1
    assert result["deposition_year"] == 2016
    # The fixture carries no pdbx_collection_date, so the collection year is
    # unknown.
    assert result["collection_year"] is None
    assert result["first_radiation_source"] == "SYNCHROTRON"
    assert result["synchrotron_site"] == "Diamond"
    assert result["detector_type"] == "PIXEL"
    assert result["crystallization_method"] == "VAPOR DIFFUSION, HANGING DROP"


def test_every_experimental_method_is_extracted_in_deposition_order() -> None:
    single = extract_experiment_fields(RECORDED_5MCC_ENTRY)
    joint = extract_experiment_fields(
        {
            "exptl": [
                {"method": "X-RAY DIFFRACTION", "crystals_number": 1},
                {"method": "NEUTRON DIFFRACTION"},
            ],
            "rcsb_entry_info": {"structure_determination_methodology": "integrative"},
        }
    )
    absent = extract_experiment_fields({"exptl": [{"method": ""}]})

    assert single["experimental_methods"] == ["X-RAY DIFFRACTION"]
    assert single["structure_determination_methodology"] == "experimental"
    assert single["radiation_sources"] == ["SYNCHROTRON"]
    both_beams = extract_experiment_fields(
        {"diffrn_source": [{"source": "SYNCHROTRON"}, {"source": "NUCLEAR REACTOR"}]}
    )
    assert both_beams["radiation_sources"] == ["SYNCHROTRON", "NUCLEAR REACTOR"]
    assert both_beams["first_radiation_source"] == "SYNCHROTRON"
    assert absent["radiation_sources"] == []
    assert joint["experimental_methods"] == ["X-RAY DIFFRACTION", "NEUTRON DIFFRACTION"]
    assert joint["structure_determination_methodology"] == "integrative"
    assert absent["experimental_methods"] == []
    assert absent["structure_determination_methodology"] is None


def test_temperature_is_the_collection_temperature_not_the_growth_temperature() -> None:
    result = extract_experiment_fields(RECORDED_5MCC_ENTRY)

    assert result["temperature_K"] == 100.0

    growth_only = extract_experiment_fields({"exptl_crystal_grow": [{"temp": 289.0}]})
    assert growth_only["temperature_K"] is None


def test_dose_is_recovered_from_a_dose_series_title() -> None:
    result = extract_experiment_fields(RECORDED_5MCC_ENTRY)

    assert result["title_dose_MGy"] == 1.11


def test_wavelength_and_beamline_come_from_the_real_entry() -> None:
    result = extract_beamline_and_wavelength(RECORDED_5MCC_ENTRY)

    assert result["beamline"] == "I02"
    assert math.isclose(cast(float, result["wavelength"]), 0.980)


def test_citation_fields_come_from_the_real_entry() -> None:
    result = extract_citation_fields(RECORDED_5MCC_ENTRY)

    assert result["citation_doi"] == "10.1107/S1600577516016775"
    assert result["citation_year"] == 2017
    assert result["citation_pubmed_id"] == 28009542
    assert result["citation_journal_abbrev"] == "J Synchrotron Radiat"


def test_deposit_authors_order_numerically_beyond_nine() -> None:
    authors: list[dict[str, object]] = [
        {"name": f"Author{ordinal}", "pdbx_ordinal": ordinal}
        for ordinal in range(1, 13)
    ]

    result = extract_citation_fields({"audit_author": authors})

    assert result["deposit_authors"] == "; ".join(
        f"Author{ordinal}" for ordinal in range(1, 13)
    )


def test_deposit_authors_order_from_shuffled_string_ordinals() -> None:
    authors: list[dict[str, object]] = [
        {"name": "Third", "pdbx_ordinal": "12"},
        {"name": "Second", "pdbx_ordinal": "2"},
        {"name": "First", "pdbx_ordinal": 1},
    ]

    result = extract_citation_fields({"audit_author": authors})

    assert result["deposit_authors"] == "First; Second; Third"


def test_deposit_authors_keep_input_order_for_unusable_ordinals() -> None:
    authors: list[dict[str, object]] = [
        {"name": "NoOrdinal"},
        {"name": "Numbered", "pdbx_ordinal": 5},
        {"name": "BadOrdinal", "pdbx_ordinal": "n/a"},
    ]

    result = extract_citation_fields({"audit_author": authors})

    # An author with a usable ordinal comes first, and the rest follow in the
    # order the record lists them.
    assert result["deposit_authors"] == "Numbered; NoOrdinal; BadOrdinal"


@pytest.mark.parametrize(
    ("wavelength_list", "expected"),
    [
        ("0.9792 1.0000", 0.9792),
        ("0.9792, 1.0000", 0.9792),
        ("0.9792,1.0000", 0.9792),
        ("  1.5418  ", 1.5418),
        ("0.9792", 0.9792),
    ],
)
def test_wavelength_parses_comma_and_whitespace_separated_lists(
    wavelength_list: str, expected: float
) -> None:
    result = extract_beamline_and_wavelength(
        {"diffrn_source": [{"pdbx_wavelength_list": wavelength_list}]}
    )

    assert math.isclose(cast(float, result["wavelength"]), expected)


@pytest.mark.parametrize("wavelength_list", ["", "   ", "not-a-number", ",", "0"])
def test_wavelength_missing_for_unusable_values(wavelength_list: str) -> None:
    result = extract_beamline_and_wavelength(
        {"diffrn_source": [{"pdbx_wavelength_list": wavelength_list}]}
    )

    assert result["wavelength"] is None


@pytest.mark.parametrize("single", ["1.5418", 1.5418])
def test_wavelength_falls_back_to_the_single_wavelength(single: object) -> None:
    alone = extract_beamline_and_wavelength(
        {"diffrn_source": [{"pdbx_wavelength": single}]}
    )
    listed = extract_beamline_and_wavelength(
        {"diffrn_source": [{"pdbx_wavelength": single, "pdbx_wavelength_list": "0.98"}]}
    )

    assert alone["wavelength"] == 1.5418
    assert listed["wavelength"] == 0.98


def test_text_fields_are_stripped_and_refuse_values_that_are_not_text() -> None:
    record: dict[str, Any] = {
        "diffrn_source": [
            {"pdbx_synchrotron_beamline": "   ", "type": " RIGAKU ", "source": 7}
        ],
        "diffrn_detector": [{"detector": {"name": "PIXEL"}, "type": "CCD"}],
        "symmetry": {"space_group_name_H_M": ["P 1"]},
        "exptl": [{"method": " X-RAY DIFFRACTION "}, {"method": 5}],
        "rcsb_primary_citation": {"title": {"text": "x"}, "pdbx_database_id_DOI": " "},
    }

    beamline = extract_beamline_and_wavelength(record)
    info = extract_experiment_fields(record)
    citation = extract_citation_fields(record)

    assert beamline["beamline"] == "RIGAKU"
    assert info["first_radiation_source"] is None
    assert info["detector_type"] == "CCD"
    assert info["space_group"] is None
    assert info["experimental_methods"] == ["X-RAY DIFFRACTION"]
    assert citation["citation_title"] is None
    assert citation["citation_doi"] is None


def test_author_names_that_are_not_text_are_left_out() -> None:
    result = extract_citation_fields(
        {
            "rcsb_primary_citation": {"rcsb_authors": ["Smith, J.", None, " "]},
            "audit_author": [
                {"name": {"last": "Doe"}, "pdbx_ordinal": 1},
                {"name": "Roe, R.", "pdbx_ordinal": 2},
            ],
        }
    )
    unnamed: dict[str, Any] = {
        "rcsb_primary_citation": {"rcsb_authors": [3]},
        "audit_author": [{}],
    }
    nobody = extract_citation_fields(unnamed)

    assert result["citation_authors"] == "Smith, J."
    assert result["deposit_authors"] == "Roe, R."
    assert nobody["citation_authors"] is None
    assert nobody["deposit_authors"] is None


@pytest.mark.parametrize("pubmed_id", [-1, "-1", 0])
def test_a_pubmed_id_the_archive_writes_for_no_article_is_missing(
    pubmed_id: object,
) -> None:
    result = extract_citation_fields(
        {"rcsb_primary_citation": {"pdbx_database_id_PubMed": pubmed_id}}
    )

    assert result["citation_pubmed_id"] is None


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Thaumatin after 5.2 MGy", 5.2),
        ("THAUMATIN AFTER 12 MGY", 12.0),
        ("Dose series, 0.75MGy", 0.75),
        # A dose in kGy is given in MGy.
        ("Thaumatin after 520 kGy", 0.52),
        ("DTPB (56 KGY)", 0.056),
        # A dose in Gy, written out or in scientific notation, is given in MGy.
        ("Acetylcholinesterase with a cumulated dose of 9400000 Gy", 9.4),
        ("X-ray structure of LPMO at 7.88x10^3 Gy", 0.00788),
        ("Structure of an LPMO at 1.45 x 10^6 Gy", 1.45),
        # An exponent without its mantissa, or a word that begins with Gy,
        # is not a dose.
        ("Thaumatin after 10^6 Gy", None),
        ("Complex 2 Gyrase", None),
        # A dose rate is not a dose, and the dose after it is still read.
        ("Thaumatin at 1.2 MGy/s", None),
        ("Thaumatin at 40 kGy s-1", None),
        ("Thaumatin at 40 kGy/s, after 300 kGy", 0.3),
        # A number that is only the tail of a longer one is not a dose.
        ("Thaumatin after 1,5 MGy", None),
        ("Thaumatin after 1.2.5 MGy", None),
        ("Thaumatin", None),
    ],
)
def test_dose_is_read_only_from_a_whole_number_before_mgy_kgy_or_gy(
    title: str, expected: float | None
) -> None:
    result = extract_experiment_fields({"struct": {"title": title}})

    assert result["title_dose_MGy"] == expected


def _cache_envelope(entry: dict[str, Any], *, age_seconds: float = 0.0) -> str:
    return json.dumps({"cached_at": time.time() - age_seconds, "entry": entry})


def test_metadata_is_served_from_a_current_cache_without_a_request(
    tmp_path: Path, network_logs: pytest.LogCaptureFixture
) -> None:
    cache_path = tmp_path / "5mcc.json"
    cache_path.write_text(_cache_envelope(RECORDED_5MCC_ENTRY), encoding="utf-8")

    result = fetch_rcsb_metadata(
        "5MCC", session=cast(Any, None), timeout=1.0, metadata_cache_path=cache_path
    )

    assert result.ok
    assert result.metadata["r_work"] == 0.1513
    assert "Using cached RCSB metadata" in network_logs.text
    assert [path.name for path in tmp_path.iterdir()] == ["5mcc.json"]


def test_metadata_cache_write_is_atomic_and_leaves_no_scratch_file(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "5mcc.json"
    session = FakeSession(FakeResponse(RECORDED_5MCC_ENTRY))

    result = fetch_rcsb_metadata(
        "5MCC",
        session=session.as_session(),
        timeout=1.0,
        metadata_cache_path=cache_path,
    )

    assert result.ok
    assert [path.name for path in tmp_path.iterdir()] == ["5mcc.json"]


def test_cache_stores_the_raw_response_so_extractor_changes_take_effect(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "5mcc.json"
    session = FakeSession(FakeResponse(RECORDED_5MCC_ENTRY))
    fetch_rcsb_metadata(
        "5MCC",
        session=session.as_session(),
        timeout=1.0,
        metadata_cache_path=cache_path,
    )

    stored: dict[str, Any] = json.loads(cache_path.read_text(encoding="utf-8"))

    assert stored["entry"] == RECORDED_5MCC_ENTRY, "raw payload, not columns"

    reused = fetch_rcsb_metadata(
        "5MCC", session=cast(Any, None), timeout=1.0, metadata_cache_path=cache_path
    )
    assert "cached_at" not in reused.metadata
    assert reused.metadata["r_work"] == 0.1513


@pytest.mark.parametrize(
    ("description", "contents"),
    [
        ("file without a cache envelope", json.dumps({"citation_doi": "10.1/x"})),
        (
            "expired entry",
            _cache_envelope(RECORDED_5MCC_ENTRY, age_seconds=CACHE_TTL_SECONDS + 60),
        ),
        (
            "future-dated entry",
            _cache_envelope(RECORDED_5MCC_ENTRY, age_seconds=-3600),
        ),
        ("unparseable file", "{not json"),
    ],
)
def test_unusable_cache_entries_are_refetched(
    tmp_path: Path,
    network_logs: pytest.LogCaptureFixture,
    description: str,
    contents: str,
) -> None:
    cache_path = tmp_path / "5mcc.json"
    cache_path.write_text(contents, encoding="utf-8")
    session = FakeSession(FakeResponse(RECORDED_5MCC_ENTRY))

    result = fetch_rcsb_metadata(
        "5MCC",
        session=session.as_session(),
        timeout=1.0,
        metadata_cache_path=cache_path,
    )

    assert result.ok, description
    assert len(session.calls) == 1, f"{description} should have refetched"
    assert "Using cached RCSB metadata" not in network_logs.text
    assert result.metadata["r_work"] == 0.1513


def test_a_failed_fetch_is_not_cached(tmp_path: Path) -> None:
    cache_path = tmp_path / "1abc.json"
    session = FakeSession()

    result = fetch_rcsb_metadata(
        "1ABC",
        session=session.as_session(),
        timeout=1.0,
        metadata_cache_path=cache_path,
    )

    assert not result.ok
    assert not cache_path.exists()


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse({}),
        FakeResponse([RECORDED_5MCC_ENTRY]),
        FakeResponse("5MCC"),
        FakeResponse(text="null"),
    ],
)
def test_a_response_that_is_not_a_non_empty_object_fails_and_is_not_cached(
    tmp_path: Path, response: FakeResponse
) -> None:
    cache_path = tmp_path / "5mcc.json"

    result = fetch_rcsb_metadata(
        "5MCC",
        session=FakeSession(response).as_session(),
        timeout=1.0,
        metadata_cache_path=cache_path,
    )

    assert not result.ok
    assert result.metadata == {}
    assert result.message is not None
    assert "non-empty JSON object" in result.message
    assert not cache_path.exists()


def test_an_empty_cached_record_is_fetched_again(
    tmp_path: Path, network_logs: pytest.LogCaptureFixture
) -> None:
    cache_path = tmp_path / "5mcc.json"
    cache_path.write_text(_cache_envelope({}), encoding="utf-8")
    session = FakeSession(FakeResponse(RECORDED_5MCC_ENTRY))

    result = fetch_rcsb_metadata(
        "5MCC",
        session=session.as_session(),
        timeout=1.0,
        metadata_cache_path=cache_path,
    )

    assert result.ok
    assert not result.cached
    assert len(session.calls) == 1
    assert "Using cached RCSB metadata" not in network_logs.text
    assert result.metadata["r_work"] == 0.1513


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("", None),
        (None, None),
        ("VAPOR DIFFUSION, HANGING DROP", "VAPOR DIFFUSION, HANGING DROP"),
    ],
)
def test_an_empty_crystallization_method_is_missing_like_other_text_fields(
    method: object, expected: object
) -> None:
    result = extract_experiment_fields(
        {
            "exptl_crystal_grow": [{"method": method}],
            "diffrn_source": [{"source": method, "pdbx_synchrotron_site": method}],
        }
    )

    assert result["crystallization_method"] == expected
    assert result["first_radiation_source"] == expected
    assert result["synchrotron_site"] == expected


_NUMERIC_FIELD_CASES: list[tuple[str, str, str, object]] = [
    ("diffrn", "ambient_temp", "temperature_K", 100.0),
    ("exptl", "crystals_number", "num_crystals", 3),
    ("exptl_crystal_grow", "pH", "crystal_pH", 7.5),
    ("reflns", "pdbx_CC_half", "cc_half", 0.998),
    ("reflns", "pdbx_netI_over_sigmaI", "i_over_sigma", 12.2),
    ("reflns", "B_iso_Wilson_estimate", "wilson_b", 31.5),
    ("refine", "ls_R_factor_R_work", "r_work", 0.18),
    ("refine", "ls_R_factor_R_free", "r_free", 0.21),
]


@pytest.mark.parametrize(
    ("category", "key", "output_name", "expected"), _NUMERIC_FIELD_CASES
)
def test_numeric_fields_coerce_strings_from_the_api(
    category: str, key: str, output_name: str, expected: object
) -> None:
    as_string = extract_experiment_fields({category: [{key: str(expected)}]})
    as_number = extract_experiment_fields({category: [{key: expected}]})

    assert as_string[output_name] == expected
    assert as_number[output_name] == expected
    assert type(as_string[output_name]) is type(as_number[output_name])
    assert not isinstance(as_string[output_name], str)


@pytest.mark.parametrize(
    ("category", "key", "output_name", "_expected"), _NUMERIC_FIELD_CASES
)
@pytest.mark.parametrize("bad_value", ["", "  ", "n/a", "not-a-number", None, {}, []])
def test_numeric_fields_are_missing_rather_than_wrong(
    category: str, key: str, output_name: str, _expected: object, bad_value: object
) -> None:
    result = extract_experiment_fields({category: [{key: bad_value}]})

    assert result[output_name] is None


def test_numeric_fields_are_missing_when_the_category_is_absent() -> None:
    result = extract_experiment_fields({})

    for _category, _key, output_name, _expected in _NUMERIC_FIELD_CASES:
        assert result[output_name] is None, output_name


def test_detector_type_falls_back_to_the_secondary_field() -> None:
    both = extract_experiment_fields(
        {"diffrn_detector": [{"detector": "PIXEL", "type": "PSI PILATUS 6M"}]}
    )
    type_only = extract_experiment_fields(
        {"diffrn_detector": [{"type": "PSI PILATUS 6M"}]}
    )

    assert both["detector_type"] == "PIXEL"
    assert type_only["detector_type"] == "PSI PILATUS 6M"


def test_zero_is_kept_rather_than_treated_as_missing() -> None:
    result = extract_experiment_fields({"reflns": [{"pdbx_CC_half": 0}]})

    assert result["cc_half"] == 0.0


def test_non_positive_resolution_is_rejected_but_a_valid_one_is_kept() -> None:
    for value in (0, -1.5, "0"):
        rejected = extract_experiment_fields({"refine": [{"ls_d_res_high": value}]})
        assert rejected["resolution_rcsb"] is None, value

    accepted = extract_experiment_fields({"refine": [{"ls_d_res_high": "1.8"}]})
    assert accepted["resolution_rcsb"] == 1.8


def test_resolution_falls_back_to_the_entry_summary() -> None:
    result = extract_experiment_fields(
        {"rcsb_entry_info": {"resolution_combined": ["2.10"]}}
    )

    assert result["resolution_rcsb"] == 2.10


def test_deposition_year_comes_from_the_deposit_date() -> None:
    result = extract_experiment_fields(
        {"rcsb_accession_info": {"deposit_date": "2019-04-01T00:00:00Z"}}
    )

    assert result["deposition_year"] == 2019
    # No detector collection date was supplied, so collection year is unknown
    # rather than silently copied from the deposit date.
    assert result["collection_year"] is None


@pytest.mark.parametrize(
    "deposit_date", ["", "19", None, "abcd-01-01", "-201-01-01", "20190-01-01", 2019]
)
def test_deposition_year_missing_for_unusable_deposit_dates(
    deposit_date: object,
) -> None:
    result = extract_experiment_fields(
        {"rcsb_accession_info": {"deposit_date": deposit_date}}
    )

    assert result["deposition_year"] is None


def test_collection_year_comes_from_the_detector_collection_date() -> None:
    # These dates come from 7BCE. It was collected in February 2017 and
    # deposited in December 2020.
    result = extract_experiment_fields(
        {
            "diffrn_detector": [
                {
                    "detector": "PIXEL",
                    "pdbx_collection_date": "2017-02-02T00:00:00.000+00:00",
                }
            ],
            "rcsb_accession_info": {"deposit_date": "2020-12-19T00:00:00.000+00:00"},
        }
    )

    assert result["collection_year"] == 2017
    assert result["deposition_year"] == 2020


@pytest.mark.parametrize("collection_date", ["", "20", None, "xxxx-01-01"])
def test_collection_year_missing_for_unusable_collection_dates(
    collection_date: object,
) -> None:
    result = extract_experiment_fields(
        {
            "diffrn_detector": [
                {"detector": "PIXEL", "pdbx_collection_date": collection_date}
            ]
        }
    )

    assert result["collection_year"] is None


def test_citation_numbers_coerce_strings() -> None:
    result = extract_citation_fields(
        {
            "rcsb_primary_citation": {
                "year": "2021",
                "pdbx_database_id_PubMed": "34567890",
            }
        }
    )

    assert result["citation_year"] == 2021
    assert result["citation_pubmed_id"] == 34567890


def test_the_title_and_keywords_are_kept_and_a_time_resolved_mention_is_flagged() -> (
    None
):
    resting = extract_experiment_fields(
        {
            "struct": {"title": " Bacteriorhodopsin: resting state structure "},
            "struct_keywords": {"text": "XFEL, time-resolved SFX, PROTON TRANSPORT"},
        }
    )
    pump_probe = extract_experiment_fields(
        {"struct": {"title": "Pump probe series, 10 ms delay"}}
    )
    spaced = extract_experiment_fields({"struct_keywords": {"text": "TIME RESOLVED"}})
    joined = extract_experiment_fields({"struct": {"title": "a timeresolved study"}})
    plain = extract_experiment_fields(
        {
            "struct": {"title": "Lysozyme at 1.2 A"},
            "struct_keywords": {"text": "HYDROLASE"},
        }
    )
    absent = extract_experiment_fields(
        {"struct": {"title": " "}, "struct_keywords": {}}
    )

    assert resting["title"] == "Bacteriorhodopsin: resting state structure"
    assert resting["keywords"] == "XFEL, time-resolved SFX, PROTON TRANSPORT"
    # The match says the entry belongs to a time-resolved series, not which
    # state it shows.
    assert resting["mentions_time_resolved"] is True
    assert pump_probe["mentions_time_resolved"] is True
    assert spaced["mentions_time_resolved"] is True
    assert joined["mentions_time_resolved"] is True
    assert plain["mentions_time_resolved"] is False
    assert absent["title"] is None
    assert absent["keywords"] is None
    assert absent["mentions_time_resolved"] is None


def test_serial_crystallography_fields_come_from_their_first_records() -> None:
    serial = extract_experiment_fields(
        {
            "diffrn": [{"pdbx_serial_crystal_experiment": "Y"}],
            "pdbx_serial_crystallography_sample_delivery": [
                {"diffrn_id": "1", "method": "injection"},
                {"diffrn_id": "2", "method": "fixed target"},
            ],
            "pdbx_serial_crystallography_measurement": [
                {"diffrn_id": "1", "pulse_duration": 30.0}
            ],
        }
    )
    single = extract_experiment_fields(
        {"diffrn": [{"pdbx_serial_crystal_experiment": "n"}]}
    )
    unsaid = extract_experiment_fields(
        {
            "diffrn": [{"pdbx_serial_crystal_experiment": None}],
            "pdbx_serial_crystallography_measurement": [{"pulse_duration": 0}],
        }
    )

    assert serial["is_serial_crystallography"] is True
    assert serial["sample_delivery_method"] == "injection"
    assert serial["xfel_pulse_duration_fs"] == 30.0
    assert single["is_serial_crystallography"] is False
    assert single["sample_delivery_method"] is None
    # A pulse of no duration is not a measurement.
    assert unsaid["is_serial_crystallography"] is None
    assert unsaid["xfel_pulse_duration_fs"] is None
