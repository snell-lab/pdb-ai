from __future__ import annotations

import pytest
from fake_http import FakeResponse, FakeSession

from pdbai.engines.sources.alphafold import model_fetch as model_fetch_module
from pdbai.engines.sources.alphafold.model_fetch import (
    fetch_afdb_model,
    prediction_range,
    prediction_sort_key,
    select_prediction_family,
)
from pdbai.engines.sources.alphafold.models import (
    AFDB_STATUS_AVAILABLE,
    AFDB_STATUS_DOWNLOAD_FAILED,
    AFDB_STATUS_LOOKUP_FAILED,
    AFDB_STATUS_MODEL_NOT_FOUND,
)

# parse_afdb_model_cif returns this pair of scores and residue names, each
# keyed by label_seq_id.
_ParsedModel = tuple[dict[str, float], dict[str, str]]


def _alanines(scores: dict[str, float]) -> _ParsedModel:
    """Return a parsed model whose every scored residue is an alanine."""
    return scores, dict.fromkeys(scores, "ALA")


def _prediction(
    entry_id: str,
    *,
    accession: str = "P12345",
    start: int | None = None,
    end: int | None = None,
    cif_url: str | None = None,
) -> dict[str, object]:
    prediction: dict[str, object] = {
        "uniprotAccession": accession,
        "entryId": entry_id,
        "cifUrl": f"https://example.test/{entry_id}.cif"
        if cif_url is None
        else cif_url,
    }
    if start is not None:
        prediction["uniprotStart"] = start
    if end is not None:
        prediction["uniprotEnd"] = end
    return prediction


def _selected(
    predictions: list[dict[str, object]], accession: str = "P12345"
) -> list[str]:
    return [
        str(prediction.get("entryId"))
        for prediction in select_prediction_family(predictions, accession)
    ]


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ({"uniprotStart": 201, "uniprotEnd": 400}, (201, 400)),
        ({"uniprotStart": 201}, (201, None)),
        ({"sequenceStart": "5", "sequenceEnd": "9"}, (5, 9)),
        ({}, (1, None)),
        ({"uniprotStart": -5, "sequenceStart": 201}, (201, None)),
        ({"uniprotStart": 0, "uniprotEnd": 0}, (1, None)),
    ],
)
def test_prediction_range_uses_documented_uniprot_coordinates(
    prediction: dict[str, object], expected: tuple[int, int | None]
) -> None:
    assert prediction_range(prediction) == expected


def test_fragments_of_one_family_are_returned_in_deterministic_order() -> None:

    predictions = [
        _prediction("AF-P12345-F3", start=401, end=1800),
        _prediction("AF-P12345-F1", start=1, end=1400),
        _prediction("AF-P12345-F2", start=201, end=1600),
    ]

    assert _selected(predictions) == [
        "AF-P12345-F1",
        "AF-P12345-F2",
        "AF-P12345-F3",
    ]


def test_a_canonical_family_excludes_unrelated_models() -> None:
    predictions = [
        _prediction("SOMEDB-XYZ"),
        _prediction("AF-P12345-F1", start=1, end=1400),
    ]

    assert _selected(predictions) == ["AF-P12345-F1"]


def test_an_isoform_family_is_not_merged_into_the_canonical_one() -> None:

    predictions = [
        _prediction("AF-P12345-2-F1"),
        _prediction("AF-P12345-F1", start=1, end=1400),
    ]

    assert _selected(predictions) == ["AF-P12345-F1"]


def test_a_family_for_another_accession_is_rejected_entirely() -> None:
    assert _selected([_prediction("AF-Q99999-F1", accession="Q99999")]) == []


@pytest.mark.parametrize(
    "predictions",
    [
        [],
        [{"uniprotAccession": "P12345", "entryId": "AF-P12345-F1"}],
        [{"uniprotAccession": "P12345", "entryId": "AF-P12345-F1", "cifUrl": ""}],
    ],
)
def test_predictions_without_a_downloadable_model_are_dropped(
    predictions: list[dict[str, object]],
) -> None:
    assert select_prediction_family(predictions, "P12345") == []


def test_without_a_canonical_family_the_widest_exact_match_is_chosen() -> None:

    predictions = [
        _prediction("COLLAB-A", start=1, end=100),
        _prediction("COLLAB-B", start=1, end=500),
        _prediction("COLLAB-C", start=1, end=200),
    ]

    assert _selected(predictions) == ["COLLAB-B"]


def test_the_fallback_requires_an_exactly_declared_accession() -> None:

    predictions = [_prediction("COLLAB-A", accession="P99999-2")]

    assert select_prediction_family(predictions, "P12345") == []


def test_sort_key_ranks_canonical_first_then_related_then_foreign() -> None:
    canonical = prediction_sort_key(_prediction("AF-P12345-F1"), "P12345")
    later_fragment = prediction_sort_key(_prediction("AF-P12345-F2"), "P12345")
    foreign = prediction_sort_key(_prediction("COLLAB-X"), "P12345")

    assert canonical[0] == 0
    assert later_fragment[0] == 1
    assert foreign[0] == 2
    assert canonical < later_fragment < foreign


def test_sort_key_is_case_insensitive_about_the_accession() -> None:
    assert (
        prediction_sort_key(_prediction("af-p12345-f1"), "P12345")[0]
        == prediction_sort_key(_prediction("AF-P12345-F1"), "p12345")[0]
        == 0
    )


# Each fragment gives all its positions the same score. A score therefore
# shows which fragment supplied it. Both fragments cover positions 201 to 400.
_OVERLAPPING_SCORES: dict[str, dict[str, float]] = {
    "https://example.test/AF-P12345-F1.cif": {
        str(position): 11.0 for position in range(1, 401)
    },
    "https://example.test/AF-P12345-F2.cif": {
        str(position): 99.0 for position in range(201, 601)
    },
}


def _fetch_with_response_order(
    monkeypatch: pytest.MonkeyPatch, predictions: list[dict[str, object]]
) -> dict[str, float]:
    session = FakeSession(
        FakeResponse(predictions),
        FakeResponse(text="data_fragment"),
        FakeResponse(text="data_fragment"),
    )

    def fake_parse(_text: str) -> _ParsedModel:
        requested_url, _timeout = session.calls[-1]
        return _alanines(dict(_OVERLAPPING_SCORES[requested_url]))

    monkeypatch.setattr(model_fetch_module, "parse_afdb_model_cif", fake_parse)
    result = fetch_afdb_model("P12345", session=session.as_session(), timeout=1.0)
    assert result.status == AFDB_STATUS_AVAILABLE
    return result.plddt_by_position


def test_overlapping_fragments_resolve_the_same_way_whatever_the_api_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    fragment_1 = _prediction("AF-P12345-F1", start=1, end=400)
    fragment_2 = _prediction("AF-P12345-F2", start=201, end=600)

    forwards = _fetch_with_response_order(monkeypatch, [fragment_1, fragment_2])
    backwards = _fetch_with_response_order(monkeypatch, [fragment_2, fragment_1])

    assert forwards == backwards
    assert forwards["300"] == 11.0, "the canonical F1 fragment wins the overlap"
    assert forwards["500"] == 99.0, "F2 still supplies the region F1 never covers"
    assert len(forwards) == 600


@pytest.mark.parametrize(
    ("predictions", "status", "reason"),
    [
        (
            [{"uniprotAccession": "P12345", "entryId": "AF-P12345-F1"}],
            AFDB_STATUS_LOOKUP_FAILED,
            "predictions without a CIF URL",
        ),
        (
            [_prediction("AF-Q99999-F1", accession="Q99999")],
            AFDB_STATUS_MODEL_NOT_FOUND,
            "no compatible accession model",
        ),
    ],
    ids=["no CIF URL", "another accession"],
)
def test_a_lookup_without_a_usable_model_downloads_nothing(
    predictions: list[dict[str, object]], status: str, reason: str
) -> None:
    session = FakeSession(FakeResponse(predictions))

    result = fetch_afdb_model("P12345", session=session.as_session(), timeout=1.0)

    assert result.status == status
    assert result.failure_reason is not None
    assert reason in result.failure_reason
    assert len(session.calls) == 1


def test_a_complex_never_joins_a_family() -> None:
    complex_prediction = _prediction("AF-P12345-F1", start=1, end=1400)
    complex_prediction["isComplex"] = True
    single_chain = _prediction("COLLAB-A", start=1, end=100)

    assert _selected([complex_prediction, single_chain]) == ["COLLAB-A"]
    assert _selected([complex_prediction]) == []


def test_a_family_member_that_scores_no_new_position_is_not_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # F2 covers only positions F1 already scored, so it supplies nothing.
    predictions = [
        _prediction("AF-P12345-F1", start=1, end=400),
        _prediction("AF-P12345-F2", start=201, end=300),
    ]
    session = FakeSession(
        FakeResponse(predictions),
        FakeResponse(text="data_fragment"),
        FakeResponse(text="data_fragment"),
    )

    def fake_parse(_text: str) -> _ParsedModel:
        requested_url, _timeout = session.calls[-1]
        if requested_url.endswith("F1.cif"):
            return _alanines({str(position): 50.0 for position in range(1, 401)})
        return _alanines({str(position): 90.0 for position in range(201, 301)})

    monkeypatch.setattr(model_fetch_module, "parse_afdb_model_cif", fake_parse)

    result = fetch_afdb_model("P12345", session=session.as_session(), timeout=1.0)

    assert result.status == AFDB_STATUS_AVAILABLE
    assert result.model_ids == ["AF-P12345-F1"]
    assert result.source_urls == ["https://example.test/AF-P12345-F1.cif"]
    assert set(result.model_id_by_position.values()) == {"AF-P12345-F1"}


def test_an_accession_is_quoted_as_one_segment_of_the_api_url() -> None:
    session = FakeSession(FakeResponse(None, status_code=404))

    fetch_afdb_model("P1/../x", session=session.as_session(), timeout=1.0)

    assert session.calls[0][0] == (
        "https://alphafold.ebi.ac.uk/api/prediction/P1%2F..%2Fx"
    )


def test_a_family_that_fails_only_to_download_reports_download_failed() -> None:
    session = FakeSession(
        FakeResponse([_prediction("AF-P12345-F1")]),
        FakeResponse(None, status_code=503),
    )

    result = fetch_afdb_model("P12345", session=session.as_session(), timeout=1.0)

    assert result.status == AFDB_STATUS_DOWNLOAD_FAILED
    assert result.failure_reason == "AF-P12345-F1: download failed (HTTP 503)"
