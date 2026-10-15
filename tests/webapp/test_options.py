from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pytest import MonkeyPatch

from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.core import DEFAULT_DATASET_SHAPE, DatasetSource, EngineOptions
from pdbai.profiles import get_profile
from pdbai.webapp import options as options_module
from pdbai.webapp.options import (
    PDB_ID_DELIMITERS,
    engine_options_from_settings,
    optional_text,
    parse_flag,
    parse_limit,
    parse_pdb_ids,
    parse_search_accessions,
    parse_search_bounds,
    settings_from_request,
)
from webapp.conftest import _client, _options


def _settings(payload: dict[str, object]) -> object:
    return settings_from_request(payload, source="download")


def _default_settings() -> dict[str, object]:
    options = get_engine_registry().with_defaults(EngineOptions())
    return dict(options.as_settings_dict())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (" true ", True),
        ("FALSE", False),
        (None, False),
        ("  ", False),
    ],
)
def test_parse_flag_reads_a_bool_or_true_and_false_text(
    value: object, expected: bool
) -> None:
    assert parse_flag(value, "resume") is expected


@pytest.mark.parametrize("value", ["1", "yes", "flase", 1])
def test_parse_flag_refuses_anything_else_naming_the_key(value: object) -> None:
    with pytest.raises(ValueError, match=r"^resume must be true or false; got "):
        parse_flag(value, "resume")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (" \t\n", None),
        ("  Electron Microscopy ", "Electron Microscopy"),
    ],
)
def test_optional_text_strips_text_and_reads_blank_as_none(
    value: object, expected: str | None
) -> None:
    assert optional_text(value, "technique") == expected


@pytest.mark.parametrize("value", [0, True, ["x"]])
def test_optional_text_refuses_a_value_that_is_not_text(value: object) -> None:
    with pytest.raises(ValueError, match=r"^technique must be text; got "):
        optional_text(value, "technique")


def test_the_pdb_id_delimiters_are_the_ones_the_browser_mirrors() -> None:
    # PDB_ID_DELIMITERS in static/app.js repeats this pattern. A change here
    # must be made there too.
    assert PDB_ID_DELIMITERS.pattern == r"[\t\n\v\f\r ,;]+"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        (" ,; \n", []),
        (
            " 5MCC, 4hhb;\t1ABC\r\n\n2XYZ ;; 3DEF ",
            ["5MCC", "4HHB", "1ABC", "2XYZ", "3DEF"],
        ),
        ("5MCC 5MCC", ["5MCC", "5MCC"]),
        (["5mcc", " 4HHB ", ""], ["5MCC", "4HHB"]),
    ],
)
def test_parse_pdb_ids_splits_text_and_drops_blanks(
    value: object, expected: list[str]
) -> None:
    assert parse_pdb_ids(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        # A non-breaking space is not a delimiter, so the text either side of
        # it is one invalid identifier.
        "5MCC 4HHB",
        # A list item is one identifier and is never split.
        ["5MCC,4HHB"],
    ],
)
def test_parse_pdb_ids_refuses_text_joined_by_a_character_that_is_no_delimiter(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="Invalid PDB ID"):
        parse_pdb_ids(value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (5, "pdbIds must be text or a list of text; got 5."),
        ([None, "4HHB"], "pdbIds must hold only text; got None."),
    ],
)
def test_parse_pdb_ids_refuses_a_value_that_is_not_text_or_a_list_of_text(
    value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_pdb_ids(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), (" ", None), (" 5 ", 5), (5, 5)],
)
def test_parse_limit_reads_blank_as_no_limit(
    value: object, expected: int | None
) -> None:
    assert parse_limit(value) == expected


@pytest.mark.parametrize("value", [True, 0, "five", 2.5])
def test_parse_limit_refuses_anything_but_a_positive_whole_number(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="^limit must be a positive integer"):
        parse_limit(value)


def test_parse_search_bounds_reads_each_filter_under_its_camel_case_key() -> None:
    assert parse_search_bounds({}) == {}
    assert parse_search_bounds(
        {
            "minTemperatureK": " 273 ",
            "maxTemperatureK": 310.5,
            "maxResolutionA": "2.5",
            "technique": "neutron_diffraction",
        }
    ) == {
        "min_temperature_K": 273.0,
        "max_temperature_K": 310.5,
        "max_resolution_A": 2.5,
    }


def test_parse_search_bounds_reads_a_blank_filter_as_not_set() -> None:
    assert parse_search_bounds(
        {"minTemperatureK": None, "maxTemperatureK": "  ", "maxResolutionA": 2}
    ) == {"max_resolution_A": 2.0}
    # The snake_case identifier is the record's key, not the request's.
    assert parse_search_bounds({"min_temperature_K": 273}) == {}


@pytest.mark.parametrize("value", [True, 0, -5, "1e2", "warm", [273]])
def test_parse_search_bounds_refuses_anything_but_a_positive_number(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="^maxResolutionA must be a positive number"):
        parse_search_bounds({"maxResolutionA": value})


def test_parse_search_accessions_reads_text_or_a_list_and_blank_as_not_set() -> None:
    assert parse_search_accessions({}) == ()
    assert parse_search_accessions({"uniprotAccessions": "  "}) == ()
    assert parse_search_accessions({"uniprotAccessions": None}) == ()
    assert parse_search_accessions({"uniprotAccessions": "p0dtd1\nP00698-2"}) == (
        "P00698",
        "P0DTD1",
    )
    assert parse_search_accessions({"uniprotAccessions": ["P00698", "P0DTD1"]}) == (
        "P00698",
        "P0DTD1",
    )
    with pytest.raises(ValueError, match="^uniprotAccessions must hold UniProt"):
        parse_search_accessions({"uniprotAccessions": "P00698 lysozyme"})
    with pytest.raises(ValueError, match="^uniprotAccessions must be text"):
        parse_search_accessions({"uniprotAccessions": 5})


def test_profile_engines_the_technique_rules_out_are_left_off() -> None:
    options = _options(
        {"technique": "electron_microscopy", "profileId": "structure_quality"}
    )

    assert options.technique == "electron_microscopy"
    # The structure_quality profile asks for RABDAM, UniProt, and
    # validation. Electron microscopy rules out RABDAM, so only the other
    # two are switched on.
    assert options.flag("include_rabdam") is False
    assert options.flag("include_uniprot") is True
    assert options.flag("include_validation") is True


def test_a_profile_applies_its_column_filter_to_api_requests_too() -> None:
    # The CLI and the browser apply a profile's column filter. An API client
    # that sends only the profile gets the same filter from the server.
    profile = get_profile("ml_minimal")
    settings = settings_from_request({"profileId": "ml_minimal"}, source="download")
    explicit = settings_from_request(
        {
            "profileId": "ml_minimal",
            "columnFilterMode": "exclude",
            "columnFilterList": "x",
        },
        source="download",
    )

    assert settings.column_filter_mode == profile.filter_mode
    assert settings.column_filter_list == profile.columns
    # An explicit filter wins over the profile's.
    assert explicit.column_filter_mode == "exclude"
    assert explicit.column_filter_list == ("x",)


def test_column_filter_mode_ignores_case_and_a_blank_one_means_none() -> None:
    spelled = settings_from_request(
        {"columnFilterMode": " Exclude ", "columnFilterList": "x"}, source="download"
    )
    blank = settings_from_request({"columnFilterMode": ""}, source="download")

    assert spelled.column_filter_mode == "exclude"
    assert blank.column_filter_mode is None


def test_column_filter_list_may_be_a_list_of_text() -> None:
    settings = settings_from_request(
        {"columnFilterMode": "include", "columnFilterList": ["a", " b;c ", ""]},
        source="download",
    )

    assert settings.column_filter_list == ("a", "b", "c")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"pdb_id": True}, "columnFilterList must be text or a list of text"),
        (5, "columnFilterList must be text or a list of text; got 5."),
        (["a", None], "columnFilterList must hold only text; got None."),
    ],
)
def test_column_filter_list_refuses_anything_but_text_or_a_list_of_text(
    value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _settings({"columnFilterMode": "include", "columnFilterList": value})


@pytest.mark.parametrize(
    "payload",
    [
        {"columnFilterMode": "include"},
        {"columnFilterList": "x"},
        {"columnFilterMode": "include", "columnFilterList": " ;, "},
    ],
)
def test_column_filter_mode_and_list_must_be_given_together(
    payload: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError,
        match=r"^columnFilterMode and columnFilterList must be given together; "
        r"got mode .* and columns .*\.$",
    ):
        _settings(payload)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" flat ", "flat"),
        ("Star_Schema", "star_schema"),
        (None, DEFAULT_DATASET_SHAPE),
        (" ", DEFAULT_DATASET_SHAPE),
    ],
)
def test_dataset_shape_ignores_case_and_a_blank_one_means_the_default(
    value: object, expected: str
) -> None:
    assert _options({"datasetShape": value}).dataset_shape == expected


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (0, "datasetShape must be text; got 0."),
        ("star", "datasetShape must be one of flat, star_schema; got 'star'."),
    ],
)
def test_dataset_shape_refuses_a_value_that_names_no_shape(
    value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _settings({"datasetShape": value})


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (False, "profileId must be text; got False."),
        ("ml_maximal", "profileId must be one of .*; got 'ml_maximal'."),
    ],
)
def test_profile_id_refuses_a_value_that_names_no_profile(
    value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _settings({"profileId": value})


def test_profile_id_ignores_case_and_a_blank_one_means_no_profile() -> None:
    spelled = settings_from_request({"profileId": " ML_Minimal "}, source="download")
    blank = settings_from_request({"profileId": " "}, source="download")

    assert spelled.profile == get_profile("ml_minimal")
    assert blank.profile is None


def test_technique_and_xray_source_ignore_case_and_surrounding_space() -> None:
    options = _options(
        {"technique": " X_Ray_Diffraction ", "xraySource": "SYNCHROTRON"}
    )

    assert options.technique == "x_ray_diffraction"
    assert options.xray_source == "synchrotron"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"technique": 1}, "technique must be text; got 1."),
        ({"xraySource": True}, "xraySource must be text; got True."),
        ({"xraySource": "xfel"}, "xraySource must be one of .*; got 'xfel'."),
        ({"includeSasa": "on"}, "includeSasa must be true or false; got 'on'."),
        ({"rabdamSource": 5}, "rabdamSource must be text; got 5."),
        ({"rabdamSource": "PDBREDO"}, "rabdamSource must be one of .*'PDBREDO'."),
        ({"sasaNSpherePoints": True}, "sasaNSpherePoints must be a positive integer"),
        (
            {"columnFilterMode": True, "columnFilterList": "x"},
            "columnFilterMode must be text; got True.",
        ),
        (
            {"columnFilterMode": "keep", "columnFilterList": "x"},
            "columnFilterMode must be one of include, exclude; got 'keep'.",
        ),
    ],
)
def test_errors_name_the_request_key_the_client_sent(
    payload: dict[str, object], message: str
) -> None:
    # A TypeError would also fail pytest.raises here. Raised in a route, it
    # would be a server error rather than a 400.
    with pytest.raises(ValueError, match=message):
        _settings(payload)


def test_engine_settings_follow_the_same_blank_and_case_rule() -> None:
    options = _options(
        {
            "includeSasa": " TRUE ",
            "sasaAccuracyMode": " High ",
            "sasaNSpherePoints": " ",
            # A blank flag counts as not given, so it keeps its declared
            # default of true.
            "sasaIncludeUnknownNonpolymers": "",
        }
    )

    assert options.flag("include_sasa") is True
    assert options.choice("sasa_accuracy_mode") == "high"
    assert options.value("sasa_n_sphere_points") is None
    assert options.flag("sasa_include_unknown_nonpolymers") is True


def test_engine_options_from_settings_reads_an_absent_source_as_download() -> None:
    settings = _default_settings()
    settings.pop("source", None)

    assert engine_options_from_settings(settings) == EngineOptions.from_settings_dict(
        settings
    )


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [
        ({}, "download"),
        ({"source": "download"}, "download"),
        ({"source": "upload"}, "upload"),
    ],
)
def test_engine_options_from_settings_checks_options_for_the_recorded_source(
    monkeypatch: MonkeyPatch, recorded: dict[str, object], expected: str
) -> None:
    checked: list[DatasetSource] = []

    def record_source(_options: EngineOptions, *, source: DatasetSource) -> None:
        checked.append(source)

    monkeypatch.setattr(options_module, "require_compatible_options", record_source)

    engine_options_from_settings({**_default_settings(), **recorded})

    assert checked == [expected]


@pytest.mark.parametrize("source", ["UPLOAD", None, 5])
def test_engine_options_from_settings_refuses_an_unknown_source(
    source: object,
) -> None:
    with pytest.raises(ValueError, match="^source must be one of download, upload"):
        engine_options_from_settings({**_default_settings(), "source": source})


def test_engine_options_from_settings_raises_value_error_for_a_malformed_value() -> (
    None
):
    with pytest.raises(ValueError, match="engine_settings"):
        engine_options_from_settings({**_default_settings(), "engine_settings": []})


def test_json_toggles_reject_values_that_only_look_boolean(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/build",
        json={"pdbIds": ["1ABC"], "includeSasa": 1},
    )

    assert response.status_code == 400
    payload = cast(dict[str, object], response.get_json())
    assert payload["error"] == "includeSasa must be true or false; got 1."
