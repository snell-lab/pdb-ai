from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import pandas as pd
import pytest
import requests
from fake_http import FakeResponse, FakeSession

from pdbai.engines.core import EngineContext, EngineOptions, EngineServices
from pdbai.engines.sources.http_cache import CACHE_TTL_SECONDS
from pdbai.engines.sources.validation import engine as engine_module
from pdbai.engines.sources.validation.artifacts import VALIDATION_REPORT
from pdbai.engines.sources.validation.fetch import (
    VALIDATION_STATUS_AVAILABLE,
    VALIDATION_STATUS_FETCH_FAILED,
    VALIDATION_STATUS_NOT_AVAILABLE,
    VALIDATION_STATUS_PARSE_FAILED,
    ValidationReportResult,
    fetch_validation_report,
    validation_report_url,
)


def _context(
    pdb_id: str,
    atom_dataframe: pd.DataFrame,
    options: EngineOptions,
    *,
    report: dict[str, object] | None = None,
) -> EngineContext:
    # The report stands in for the one the acquisition engine publishes. With
    # no report, or an empty one, nothing is published. That is how a
    # structure without a validation report looks.
    context = EngineContext(pdb_id, atom_dataframe, options)
    if report:
        context.artifacts.put(
            VALIDATION_REPORT,
            ValidationReportResult(report=report, status=VALIDATION_STATUS_AVAILABLE),
        )
    return context


def test_validation_acquisition_returns_fetched_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected: dict[str, object] = {"residues": {}, "clashscore": 3.2}
    expected_cache_path = tmp_path / "1abc_validation.json"
    session = requests.Session()

    def fake_fetch(
        pdb_id: str,
        *,
        session: requests.Session,
        timeout: float,
        cache_path: Path,
    ) -> ValidationReportResult:
        assert (pdb_id, timeout, cache_path) == (
            "1ABC",
            6.0,
            expected_cache_path,
        )
        assert isinstance(session, requests.Session)
        return ValidationReportResult(
            report=expected,
            status=VALIDATION_STATUS_AVAILABLE,
        )

    monkeypatch.setattr(engine_module, "fetch_validation_report", fake_fetch)
    context = EngineContext(
        "1ABC",
        pd.DataFrame(),
        EngineOptions(values={"include_validation": True}),
        services=EngineServices(session=session, timeout=6, cache_root=tmp_path),
    )

    result = engine_module.VALIDATION_ACQUISITION_ENGINE.run(context)

    assert result.artifacts == {
        VALIDATION_REPORT: ValidationReportResult(
            report=expected,
            status=VALIDATION_STATUS_AVAILABLE,
        )
    }
    assert result.warnings == ()
    assert [source.role for source in result.source_files] == [
        "wwpdb_validation_report"
    ]


def test_validation_annotation_reports_how_the_report_was_obtained() -> None:
    dataframe = pd.DataFrame({"chain_id": ["A"], "res_num": ["1"]})
    fetched = _context(
        "1ABC",
        dataframe,
        EngineOptions(values={"include_validation": True}),
        report={"residues": {}},
    )
    missing = EngineContext(
        "1ABC", dataframe, EngineOptions(values={"include_validation": True})
    )
    missing.failures["validation_source"] = "archive offline"

    assert engine_module.VALIDATION_ANNOTATION_ENGINE.run(fetched).report == {
        "validation_status": VALIDATION_STATUS_AVAILABLE,
        "validation_message": None,
    }
    assert engine_module.VALIDATION_ANNOTATION_ENGINE.run(missing).report == {
        "validation_status": VALIDATION_STATUS_FETCH_FAILED,
        "validation_message": "archive offline",
    }


def test_validation_annotation_maps_insertion_codes_and_structure_metrics() -> None:
    dataframe = pd.DataFrame(
        {
            "model_num": ["1", "1", "1"],
            "chain_id": ["A", "A", "A"],
            "res_num": ["10", "10", "11"],
            "_atom_site.pdbx_PDB_ins_code": ["B", "B", "?"],
        },
        index=[5, 6, 9],
    )
    context = _context(
        "1ABC",
        dataframe,
        EngineOptions(values={"include_validation": True}),
        report={
            "residues": {
                "1:A:10B": {
                    "rscc": 0.96,
                    "rsr": 0.11,
                    "rama_class": "Favored",
                    "rotamer_class": "Allowed",
                }
            },
            "clashscore": 4.2,
            "percent_rama_outliers": 1.1,
            "percent_rotamer_outliers": 2.2,
        },
    )

    result = engine_module.VALIDATION_ANNOTATION_ENGINE.run(context)

    assert result.columns is not None
    assert result.columns.loc[[5, 6], "eds_rscc"].tolist() == [0.96, 0.96]
    assert result.columns.loc[5, "rama_class"] == "Favored"
    assert pd.isna(result.columns.loc[9, "eds_rsr"])
    assert result.columns["clashscore"].tolist() == [4.2, 4.2, 4.2]


def test_validation_annotation_keeps_coordinate_models_distinct() -> None:
    dataframe = pd.DataFrame(
        {
            "model_num": ["1", "2"],
            "chain_id": ["A", "A"],
            "res_num": ["10", "10"],
            "_atom_site.pdbx_PDB_ins_code": ["?", "?"],
        }
    )
    context = _context(
        "1ABC",
        dataframe,
        EngineOptions(values={"include_validation": True}),
        report={
            "residues": {
                "1:A:10": {"rscc": 0.95, "rama_class": "Favored"},
                "2:A:10": {"rscc": 0.75, "rama_class": "OUTLIER"},
            }
        },
    )

    result = engine_module.VALIDATION_ANNOTATION_ENGINE.run(context)

    assert result.columns is not None
    assert result.columns["eds_rscc"].tolist() == [0.95, 0.75]
    assert result.columns["rama_class"].tolist() == ["Favored", "OUTLIER"]


_VALIDATION_XML = (
    b'<?xml version="1.0"?><wwPDB-validation-information>'
    b'<Entry clashscore="2.5" percent-rama-outliers="0.5"/>'
    b'<ModelledSubgroup model="1" chain="A" resnum="9" rscc="0.88" rsr="0.12"/>'
    b"</wwPDB-validation-information>"
)


def _report_session() -> FakeSession:
    return FakeSession(FakeResponse(content=gzip.compress(_VALIDATION_XML)))


def _write_report_cache(path: Path, *, age_seconds: float) -> None:
    path.write_text(
        json.dumps(
            {
                "cached_at": time.time() - age_seconds,
                "report": {"residues": {}, "clashscore": 9.9},
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("age_seconds", "requests_made", "cached_before"),
    [(60.0, 0, True), (CACHE_TTL_SECONDS + 60.0, 1, False)],
    ids=["fresh", "stale"],
)
def test_validation_acquisition_reports_whether_the_cache_was_used(
    tmp_path: Path, age_seconds: float, requests_made: int, cached_before: bool
) -> None:
    # A stale cache file is downloaded again, so it does not count as served
    # from the cache.
    _write_report_cache(tmp_path / "1abc_validation.json", age_seconds=age_seconds)
    session = _report_session()
    context = EngineContext(
        "1ABC",
        pd.DataFrame(),
        EngineOptions(values={"include_validation": True}),
        services=EngineServices(
            session=session.as_session(), timeout=6, cache_root=tmp_path
        ),
    )

    result = engine_module.VALIDATION_ACQUISITION_ENGINE.run(context)

    assert len(session.calls) == requests_made
    assert [source.cached_before for source in result.source_files] == [cached_before]


def test_validation_cache_survives_a_write_interrupted_partway(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_path = tmp_path / "validation.json"

    # The intact file is a bare report without a cache envelope, so it counts
    # as no cache. The report is fetched again, and fetch_validation_report
    # tries to replace the file.
    intact = '{"residues": {"A:7": {"rscc": 0.9}}, "clashscore": 1.0}'
    cache_path.write_text(intact, encoding="utf-8")

    real_write_text = Path.write_text

    def failing_write_text(
        self: Path, data: str, encoding: str | None = None, **kwargs: object
    ) -> int:
        real_write_text(self, data[: len(data) // 2], encoding=encoding)
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", failing_write_text)
    result = fetch_validation_report(
        "1ABC",
        session=_report_session().as_session(),
        timeout=1.0,
        cache_path=cache_path,
    )
    monkeypatch.undo()

    assert result.report["clashscore"] == 2.5
    assert cache_path.read_text(encoding="utf-8") == intact
    assert [path.name for path in tmp_path.iterdir()] == ["validation.json"]


def test_validation_report_is_cached_after_a_successful_write(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "validation.json"

    fetch_validation_report(
        "1ABC",
        session=_report_session().as_session(),
        timeout=1.0,
        cache_path=cache_path,
    )

    assert cache_path.exists()
    assert [path.name for path in tmp_path.iterdir()] == ["validation.json"]
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cached["report"]["residues"]["1:A:9"]["rscc"] == 0.88


def test_validation_report_url_files_a_report_under_the_middle_of_the_id() -> None:
    expected = (
        "https://files.rcsb.org/pub/pdb/validation_reports"
        "/mc/5mcc/5mcc_validation.xml.gz"
    )

    assert validation_report_url("5mcc") == expected
    assert validation_report_url(" 5MCC ") == expected
    with pytest.raises(ValueError, match="Invalid PDB ID"):
        validation_report_url("pdb_00005mcc")


def test_validation_fetch_names_the_normalized_entry_in_its_messages(
    tmp_path: Path,
) -> None:
    session = FakeSession(FakeResponse(status_code=500))

    result = fetch_validation_report(
        " 1abc ",
        session=session.as_session(),
        timeout=1,
        cache_path=tmp_path / "1abc_validation.json",
    )

    assert result.status == VALIDATION_STATUS_FETCH_FAILED
    assert result.message == "wwPDB validation report fetch failed for 1ABC: HTTP 500"
    assert session.calls == [(validation_report_url("1ABC"), 1)]


# files.rcsb.org answers 403 for a report it does not hold.
@pytest.mark.parametrize("status_code", [403, 404, 410])
def test_validation_report_missing_from_the_archive_is_not_available(
    tmp_path: Path, status_code: int
) -> None:
    session = FakeSession(FakeResponse(status_code=status_code))

    result = fetch_validation_report(
        "1ABC",
        session=session.as_session(),
        timeout=1,
        cache_path=tmp_path / "1abc_validation.json",
    )

    assert result.status == VALIDATION_STATUS_NOT_AVAILABLE
    assert result.report["residues"] == {}


def test_a_cached_report_without_residues_is_fetched_again(tmp_path: Path) -> None:
    cache_path = tmp_path / "1abc_validation.json"
    cache_path.write_text(
        json.dumps(
            {
                "cached_at": time.time(),
                "report": {"residues": [], "clashscore": 9.9},
            }
        ),
        encoding="utf-8",
    )
    session = _report_session()

    result = fetch_validation_report(
        "1ABC", session=session.as_session(), timeout=1.0, cache_path=cache_path
    )

    assert result.status == VALIDATION_STATUS_AVAILABLE
    assert result.report["clashscore"] == 2.5
    assert len(session.calls) == 1


def test_xml_that_is_not_a_report_is_a_parse_failure_and_is_not_cached(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "1abc_validation.json"
    session = FakeSession(
        FakeResponse(content=gzip.compress(b"<Error><Code>NoSuchKey</Code></Error>"))
    )

    result = fetch_validation_report(
        "1ABC", session=session.as_session(), timeout=1.0, cache_path=cache_path
    )

    assert result.status == VALIDATION_STATUS_PARSE_FAILED
    assert result.message is not None
    assert "is not a validation report" in result.message
    assert result.report["residues"] == {}
    assert not cache_path.exists()


def test_validation_annotation_warns_only_about_a_report_it_could_not_get() -> None:
    dataframe = pd.DataFrame({"chain_id": ["A"], "res_num": ["1"]})
    options = EngineOptions(values={"include_validation": True})
    missing_report = EngineContext("1ABC", dataframe, options)
    missing_report.artifacts.put(
        VALIDATION_REPORT,
        ValidationReportResult(
            report={"residues": {}},
            status=VALIDATION_STATUS_NOT_AVAILABLE,
            message="No report for 1ABC.",
        ),
    )
    source_failed = EngineContext("1ABC", dataframe, options)
    source_failed.failures["validation_source"] = "archive offline"
    fetched = _context("1ABC", dataframe, options, report={"residues": {}})

    engine = engine_module.VALIDATION_ANNOTATION_ENGINE

    assert engine.run(missing_report).warnings == ("No report for 1ABC.",)
    # The coordinator already warned about the source engine's failure.
    assert engine.run(source_failed).warnings == ()
    assert engine.run(fetched).warnings == ()
