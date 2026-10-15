from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import cast

from fake_http import FakeResponse, FakeSession

from pdbai.engines.sources.http_cache import CACHE_TTL_SECONDS
from pdbai.engines.sources.rcsb.metadata import fetch_rcsb_metadata
from pdbai.engines.sources.validation.fetch import (
    VALIDATION_STATUS_AVAILABLE,
    VALIDATION_STATUS_CACHED,
    VALIDATION_STATUS_FETCH_FAILED,
    VALIDATION_STATUS_NOT_AVAILABLE,
    VALIDATION_STATUS_PARSE_FAILED,
    fetch_validation_report,
)


def test_rcsb_metadata_is_cached_and_reused_without_a_second_request(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "metadata.json"
    first_session = FakeSession(FakeResponse(payload={"rcsb_id": "1ABC"}))

    first = fetch_rcsb_metadata(
        "1ABC",
        session=first_session.as_session(),
        timeout=3.0,
        metadata_cache_path=cache,
    )
    second_session = FakeSession()
    second = fetch_rcsb_metadata(
        "1ABC",
        session=second_session.as_session(),
        timeout=3.0,
        metadata_cache_path=cache,
    )

    assert first.ok
    assert not first.cached
    assert first.metadata["citation_doi"] is None
    assert cache.exists()
    assert second.ok
    assert second.cached
    assert second.metadata == first.metadata
    assert len(first_session.calls) == 1
    assert second_session.calls == []


def test_rcsb_metadata_stale_cache_is_refetched_and_not_reported_cached(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "metadata.json"
    cache.write_text(
        json.dumps(
            {
                "cached_at": time.time() - CACHE_TTL_SECONDS - 1,
                "entry": {
                    "rcsb_primary_citation": {"pdbx_database_id_DOI": "10.1/stale"}
                },
            }
        ),
        encoding="utf-8",
    )
    session = FakeSession(FakeResponse(payload={"rcsb_id": "1ABC"}))

    result = fetch_rcsb_metadata(
        "1ABC", session=session.as_session(), timeout=3.0, metadata_cache_path=cache
    )

    assert result.ok
    assert not result.cached
    assert result.metadata["citation_doi"] is None
    assert len(session.calls) == 1


def test_validation_xml_parses_residue_and_structure_quality_and_caches(
    tmp_path: Path,
) -> None:
    xml = b"""<wwPDB-validation-information>
      <Entry clashscore="4.2" percent-rama-outliers="1.1" percent-rota-outliers="2.2" />
      <ModelledSubgroup model="1" chain="A" resnum="10" icode="B" rscc="0.95"
          rsr="0.12" rama="Favored" rota="OUTLIER" />
      <ModelledSubgroup model="2" chain="A" resnum="10" icode="B" rscc="0.75"
          rsr="0.22" rama="OUTLIER" rota="Allowed" />
    </wwPDB-validation-information>"""
    cache = tmp_path / "validation.json"
    session = FakeSession(FakeResponse(content=gzip.compress(xml)))

    result = fetch_validation_report(
        "1ABC", session=session.as_session(), timeout=2.0, cache_path=cache
    )

    assert result.status == VALIDATION_STATUS_AVAILABLE
    assert result.ok
    assert result.report["clashscore"] == 4.2
    assert result.report["percent_rama_outliers"] == 1.1
    # The report keys each residue by model, chain, residue number, and
    # insertion code. The two models of residue 10B therefore stay apart.
    residues = cast(dict[str, dict[str, object]], result.report["residues"])
    assert residues["1:A:10B"] == {
        "rscc": 0.95,
        "rsr": 0.12,
        "rama_class": "Favored",
        "rotamer_class": "OUTLIER",
    }
    assert residues["2:A:10B"] == {
        "rscc": 0.75,
        "rsr": 0.22,
        "rama_class": "OUTLIER",
        "rotamer_class": "Allowed",
    }
    cached_json = json.loads(cache.read_text(encoding="utf-8"))
    assert isinstance(cached_json["cached_at"], float)
    assert cached_json["report"]["clashscore"] == 4.2
    assert set(cached_json) == {"cached_at", "report"}

    cached_session = FakeSession()
    cached = fetch_validation_report(
        "1ABC", session=cached_session.as_session(), timeout=2.0, cache_path=cache
    )
    assert cached.status == VALIDATION_STATUS_CACHED
    assert cached.ok
    assert cached.report == result.report
    assert cached_session.calls == []


def test_validation_failure_returns_stable_empty_shape(tmp_path: Path) -> None:
    result = fetch_validation_report(
        "1ABC",
        session=FakeSession().as_session(),
        timeout=1.0,
        cache_path=tmp_path / "validation.json",
    )

    assert result.status == VALIDATION_STATUS_FETCH_FAILED
    assert not result.ok
    assert result.message is not None
    assert "Unexpected network request" in result.message
    assert result.report == {
        "residues": {},
        "clashscore": None,
        "percent_rama_outliers": None,
        "percent_rotamer_outliers": None,
    }


def test_validation_cache_without_an_envelope_is_refetched(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "validation.json"
    cache.write_text(
        json.dumps({"residues": {"A:10": {"rscc": 0.5}}}),
        encoding="utf-8",
    )
    xml = b"""<wwPDB-validation-information>
      <Entry />
      <ModelledSubgroup model="1" chain="A" resnum="10" rscc="0.95" />
    </wwPDB-validation-information>"""
    session = FakeSession(FakeResponse(content=gzip.compress(xml)))

    result = fetch_validation_report(
        "1ABC", session=session.as_session(), timeout=2.0, cache_path=cache
    )

    residues = cast(dict[str, dict[str, object]], result.report["residues"])
    assert residues["1:A:10"]["rscc"] == 0.95
    assert len(session.calls) == 1


def test_validation_missing_report_is_distinct_from_a_fetch_failure(
    tmp_path: Path,
) -> None:
    result = fetch_validation_report(
        "1ABC",
        session=FakeSession(FakeResponse(status_code=404)).as_session(),
        timeout=1.0,
        cache_path=tmp_path / "validation.json",
    )

    assert result.status == VALIDATION_STATUS_NOT_AVAILABLE
    assert not result.ok
    assert result.message == (
        "wwPDB validation report is not available for 1ABC (HTTP 404)."
    )


def test_validation_invalid_gzip_is_reported_as_a_parse_failure(
    tmp_path: Path,
) -> None:
    result = fetch_validation_report(
        "1ABC",
        session=FakeSession(FakeResponse(content=b"not gzip")).as_session(),
        timeout=1.0,
        cache_path=tmp_path / "validation.json",
    )

    assert result.status == VALIDATION_STATUS_PARSE_FAILED
    assert not result.ok
    assert result.message is not None
    assert "invalid gzip payload" in result.message


def test_validation_stale_cache_is_refetched(tmp_path: Path) -> None:
    cache = tmp_path / "validation.json"
    cache.write_text(
        json.dumps(
            {
                "cached_at": time.time() - CACHE_TTL_SECONDS - 1,
                "report": {
                    "residues": {"1:A:10": {"rscc": 0.5}},
                    "clashscore": 9.9,
                },
            }
        ),
        encoding="utf-8",
    )
    xml = b"""<wwPDB-validation-information>
      <Entry clashscore="1.5" />
      <ModelledSubgroup model="1" chain="A" resnum="10" rscc="0.95" />
    </wwPDB-validation-information>"""
    session = FakeSession(FakeResponse(content=gzip.compress(xml)))

    result = fetch_validation_report(
        "1ABC", session=session.as_session(), timeout=2.0, cache_path=cache
    )

    assert result.status == VALIDATION_STATUS_AVAILABLE
    assert result.report["clashscore"] == 1.5
    assert len(session.calls) == 1
