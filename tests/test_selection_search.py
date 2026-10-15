from __future__ import annotations

from typing import Any

import pytest
import requests
from fake_http import FakeResponse, FakeSession

from pdbai import selection as selection_module
from pdbai.selection import (
    RCSB_SEARCH_URL,
    ArchiveSearchError,
    Selection,
    build_search_request,
    search_archive,
)


def _hits(*identifiers: str) -> dict[str, Any]:
    return {
        "total_count": len(identifiers),
        "result_set": [{"identifier": item, "score": 1.0} for item in identifiers],
    }


def test_search_archive_sends_the_query_once_and_records_it() -> None:
    session = FakeSession(FakeResponse(_hits("5mcc", "11YO", "13CH")))
    selection = Selection("x_ray_diffraction", "free_electron_laser")

    result = search_archive(selection, session=session.as_session(), timeout=7.0)

    assert session.calls == [(RCSB_SEARCH_URL, 7.0)]
    assert session.json_bodies == [build_search_request(selection)]
    assert result.pdb_ids == ("11YO", "13CH", "5MCC")
    assert result.total_count == 3
    assert result.truncated is False
    assert result.service == "rcsb_search"
    assert result.service_url == RCSB_SEARCH_URL
    assert result.query == build_search_request(selection)
    assert result.warnings == ()
    assert result.retrieved_at > 0


def test_search_archive_applies_the_limit_in_sorted_order() -> None:
    session = FakeSession(FakeResponse(_hits("9ZZZ", "1AAA", "5MMM")))

    result = search_archive(
        Selection("x_ray_diffraction", limit=2),
        session=session.as_session(),
    )

    assert result.pdb_ids == ("1AAA", "5MMM")
    assert result.total_count == 3
    assert result.truncated is True


def test_search_archive_treats_no_content_as_an_empty_cohort() -> None:
    session = FakeSession(FakeResponse(status_code=204))

    result = search_archive(Selection("epr"), session=session.as_session())

    assert result.pdb_ids == ()
    assert result.total_count == 0
    assert result.truncated is False


def test_search_archive_flags_hits_it_cannot_use_once_and_is_not_truncated() -> None:
    payload = {
        "total_count": 5,
        "result_set": [
            {"identifier": "1AAA"},
            {"identifier": "AF_AFP12345F1"},
            "garbage",
            {"identifier": 1234},
            {"identifier": "1AAA"},
        ],
    }
    session = FakeSession(FakeResponse(payload))

    result = search_archive(
        Selection("x_ray_diffraction"), session=session.as_session()
    )

    assert result.pdb_ids == ("1AAA",)
    assert result.total_count == 5
    # Skipped hits leave fewer identifiers than matches, but no limit dropped
    # any of them.
    assert result.truncated is False
    assert result.warnings == (
        "3 search hit(s) were not valid PDB identifiers and were ignored.",
    )


def test_search_archive_flags_a_count_that_disagrees_with_the_hits() -> None:
    payload = {"total_count": 5, "result_set": [{"identifier": "1AAA"}]}
    session = FakeSession(FakeResponse(payload))

    result = search_archive(
        Selection("x_ray_diffraction"), session=session.as_session()
    )

    assert result.total_count == 5
    assert result.truncated is False
    assert result.warnings == (
        "The search reported 5 matching entries but returned 1 hits.",
    )


def test_search_archive_is_truncated_only_when_the_limit_drops_identifiers() -> None:
    payload = {
        "total_count": 3,
        "result_set": [
            {"identifier": "1AAA"},
            {"identifier": "AF_AFP12345F1"},
            {"identifier": "1BBB"},
        ],
    }

    exact = search_archive(
        Selection("x_ray_diffraction", limit=2),
        session=FakeSession(FakeResponse(payload)).as_session(),
    )
    short = search_archive(
        Selection("x_ray_diffraction", limit=1),
        session=FakeSession(FakeResponse(payload)).as_session(),
    )

    assert (exact.pdb_ids, exact.truncated) == (("1AAA", "1BBB"), False)
    assert (short.pdb_ids, short.truncated) == (("1AAA",), True)


class _OwnedSession:
    """A session that records whether search_archive closed it."""

    def __init__(self) -> None:
        self.closed = False

    def __enter__(self) -> _OwnedSession:
        return self

    def __exit__(self, *_exception: object) -> None:
        self.closed = True

    def post(self, url: str, *, json: object, timeout: float) -> FakeResponse:
        return FakeResponse(_hits("1ABC"))


def test_search_archive_closes_the_session_it_makes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = _OwnedSession()
    monkeypatch.setattr(selection_module, "_search_session", lambda: owned)

    result = search_archive(Selection("x_ray_diffraction"))

    assert result.pdb_ids == ("1ABC",)
    assert owned.closed is True


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (requests.ConnectionError("down"), "PDB search failed: down"),
        (FakeResponse(status_code=503), "answered HTTP 503"),
        (FakeResponse(text="not json"), "malformed JSON"),
        (FakeResponse([]), "unexpected payload"),
        (FakeResponse({"total_count": 1}), "no result set"),
    ],
)
def test_search_archive_reports_service_failures(
    response: FakeResponse | Exception, message: str
) -> None:
    session = FakeSession(response)

    with pytest.raises(ArchiveSearchError, match=message):
        search_archive(Selection("x_ray_diffraction"), session=session.as_session())
