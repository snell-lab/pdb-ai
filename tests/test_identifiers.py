from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fake_http import FakeSession

from pdbai.engines.core import EngineContext, EngineOptions, EngineServices
from pdbai.engines.sources.rcsb.engine import RCSB_ACQUISITION_ENGINE
from pdbai.identifiers import normalize_pdb_id
from pdbai.orchestration import build_atom_site_dataset


@pytest.mark.parametrize("value", ["1abc", " 4HhB ", "9xyz"])
def test_normalize_pdb_id_accepts_public_four_character_form(value: str) -> None:
    assert normalize_pdb_id(value) == value.strip().upper()


# Å counts as alphanumeric to str.isalnum, and the dotless ı upper-cases to
# an ASCII I. A check made with str methods would let both through.
@pytest.mark.parametrize(
    "value",
    ["", "ABC", "ABCDE", "../A", "/TMP", "A/B1", "AB_1", "ÅBC1", "1ÅBC", "1ıbc"],
)
def test_normalize_pdb_id_rejects_malformed_or_path_like_values(value: str) -> None:
    with pytest.raises(ValueError, match="digit from 1 to 9 followed by three"):
        normalize_pdb_id(value)


# NONE and TRUE are words a careless caller might pass as text.
@pytest.mark.parametrize("value", ["0ABC", "ABCD", "NONE", "TRUE"])
def test_normalize_pdb_id_rejects_four_characters_that_do_not_start_with_1_to_9(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="Invalid PDB ID"):
        normalize_pdb_id(value)


def test_rcsb_download_rejects_path_before_cache_or_network_access(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.cif"
    outside.write_text("private", encoding="utf-8")
    session = FakeSession()
    context = EngineContext(
        str(outside.with_suffix("")),
        pd.DataFrame(),
        EngineOptions(),
        services=EngineServices(
            session=session.as_session(), timeout=1.0, cache_root=tmp_path / "cache"
        ),
    )

    with pytest.raises(ValueError, match="Invalid PDB ID"):
        RCSB_ACQUISITION_ENGINE.run(context)

    assert session.calls == []
    assert not (tmp_path / "cache").exists()


def test_dataset_api_rejects_invalid_pdb_id_before_creating_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"

    with pytest.raises(ValueError, match="Invalid PDB ID"):
        build_atom_site_dataset(["../secret"], workspace)

    assert not workspace.exists()
