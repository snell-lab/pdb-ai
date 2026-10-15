from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fake_http import FakeResponse, FakeSession

from pdbai.engines.core import (
    STRUCTURE_CIF_PATH,
    EngineContext,
    EngineOptions,
    EngineServices,
)
from pdbai.engines.sources.rcsb import engine as engine_module


def test_rcsb_acquisition_downloads_then_reuses_the_cached_file(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "rcsb" / "1abc.cif"
    session = FakeSession(FakeResponse(content=b"data_1abc\n"))
    context = EngineContext(
        pdb_id="1ABC",
        atom_dataframe=pd.DataFrame(),
        options=EngineOptions(),
        services=EngineServices(
            session=session.as_session(), timeout=4, cache_root=tmp_path
        ),
    )

    first = engine_module.RCSB_ACQUISITION_ENGINE.run(context)
    second = engine_module.RCSB_ACQUISITION_ENGINE.run(context)

    assert first.artifacts == {STRUCTURE_CIF_PATH: expected}
    assert second.artifacts == {STRUCTURE_CIF_PATH: expected}
    assert session.calls == [("https://files.rcsb.org/download/1abc.cif", 4)]
    assert [
        (source.role, source.path, source.cached_before)
        for result in (first, second)
        for source in result.source_files
    ] == [("rcsb_mmcif", expected, False), ("rcsb_mmcif", expected, True)]


def test_rcsb_acquisition_requires_network_services() -> None:
    context = EngineContext("1ABC", pd.DataFrame(), EngineOptions())

    with pytest.raises(RuntimeError, match="network services"):
        engine_module.RCSB_ACQUISITION_ENGINE.run(context)
