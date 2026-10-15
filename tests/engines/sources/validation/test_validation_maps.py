from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fake_http import FakeResponse, FakeSession

from pdbai.engines.core import EngineContext, EngineOptions, EngineServices
from pdbai.engines.sources.validation.maps import (
    VALIDATION_2FOFC_MAP_PATH,
    VALIDATION_FOFC_MAP_PATH,
    VALIDATION_MAP_ACQUISITION_ENGINE,
    validation_map_url,
)

_GZIP = b"\x1f\x8b\x08\x00compressed"
_BASE = "https://files.rcsb.org/pub/pdb/validation_reports/ab/1abc/1abc_"


def _context(session: FakeSession, cache_root: Path) -> EngineContext:
    return EngineContext(
        "1ABC",
        pd.DataFrame(),
        EngineOptions(values={"include_density": True, "density_map_source": "wwpdb"}),
        services=EngineServices(
            session=session.as_session(), timeout=2.5, cache_root=cache_root
        ),
    )


def test_the_urls_follow_the_archives_layout() -> None:
    assert validation_map_url("1ABC", "validation_2fo-fc_map_coef.cif.gz") == (
        f"{_BASE}validation_2fo-fc_map_coef.cif.gz"
    )


def test_both_map_files_are_downloaded_into_the_cache(tmp_path: Path) -> None:
    session = FakeSession(FakeResponse(content=_GZIP), FakeResponse(content=_GZIP))

    result = VALIDATION_MAP_ACQUISITION_ENGINE.run(_context(session, tmp_path))

    two_fofc = tmp_path / "validation_maps" / "1abc_2fofc_map_coef.cif.gz"
    fofc = tmp_path / "validation_maps" / "1abc_fofc_map_coef.cif.gz"
    assert result.artifacts == {
        VALIDATION_2FOFC_MAP_PATH: two_fofc,
        VALIDATION_FOFC_MAP_PATH: fofc,
    }
    assert two_fofc.read_bytes() == _GZIP
    assert session.calls == [
        (f"{_BASE}validation_2fo-fc_map_coef.cif.gz", 2.5),
        (f"{_BASE}validation_fo-fc_map_coef.cif.gz", 2.5),
    ]
    assert [
        (source.role, source.path, source.cached_before)
        for source in result.source_files
    ] == [
        ("wwpdb_validation_2fofc_map", two_fofc, False),
        ("wwpdb_validation_fofc_map", fofc, False),
    ]

    again = VALIDATION_MAP_ACQUISITION_ENGINE.run(_context(FakeSession(), tmp_path))
    assert [source.cached_before for source in again.source_files] == [True, True]


@pytest.mark.parametrize("status_code", [403, 404])
def test_an_entry_without_maps_is_not_available_rather_than_failed(
    tmp_path: Path, status_code: int
) -> None:
    session = FakeSession(
        FakeResponse(status_code=status_code), FakeResponse(status_code=status_code)
    )

    result = VALIDATION_MAP_ACQUISITION_ENGINE.run(_context(session, tmp_path))

    assert result.artifacts == {}
    assert [source.cached_before for source in result.source_files] == [False, False]


def test_one_missing_file_publishes_neither(tmp_path: Path) -> None:
    session = FakeSession(FakeResponse(content=_GZIP), FakeResponse(status_code=403))

    result = VALIDATION_MAP_ACQUISITION_ENGINE.run(_context(session, tmp_path))

    assert result.artifacts == {}


def test_a_response_that_is_not_gzip_fails(tmp_path: Path) -> None:
    session = FakeSession(FakeResponse(content=b"<html>error</html>"))

    with pytest.raises(RuntimeError, match="not a gzip file"):
        VALIDATION_MAP_ACQUISITION_ENGINE.run(_context(session, tmp_path))
    assert not (tmp_path / "validation_maps").exists() or not any(
        (tmp_path / "validation_maps").iterdir()
    )
