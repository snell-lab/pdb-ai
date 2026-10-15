from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from pdbai.engines.core import ArtifactKey, EngineContext, EngineOptions
from pdbai.engines.sources.pdb_redo import PDB_REDO_DATAFRAME
from pdbai.engines.sources.pdb_redo.engine import PDB_REDO_COMPARISON_ENGINE

_BOOLEAN_SPEC_NAMES = ("pdb_redo_revised",)


def _context(
    pdb_id: str,
    atom_dataframe: pd.DataFrame,
    options: EngineOptions,
    *,
    artifacts: Mapping[ArtifactKey[Any], object] | None = None,
) -> EngineContext:
    context = EngineContext(pdb_id, atom_dataframe, options)
    context.artifacts.update(artifacts or {})
    return context


def _identity(res_num: str, atom_name: str) -> dict[str, str]:
    return {
        "pdb_id": "1abc",
        "model_num": "1",
        "chain_id": "A",
        "_atom_site.auth_asym_id": "A",
        "res_num": res_num,
        "_atom_site.auth_seq_id": res_num,
        "_atom_site.pdbx_PDB_ins_code": "",
        "res_name": "GLY",
        "atom_name": atom_name,
    }


def _comparison_columns() -> pd.DataFrame:
    rows = [_identity(str(index + 1), "CA") for index in range(3)]
    atom_dataframe = pd.DataFrame(rows)
    atom_dataframe["x"] = [1.0, 2.0, 3.0]
    pdb_redo_dataframe = pd.DataFrame(rows)
    pdb_redo_dataframe["x"] = [1.0, 2.5, 3.0]

    context = _context(
        "1ABC",
        atom_dataframe,
        EngineOptions(values={"include_pdb_redo": True}),
        artifacts={PDB_REDO_DATAFRAME: pdb_redo_dataframe},
    )
    columns = PDB_REDO_COMPARISON_ENGINE.run(context).columns
    assert columns is not None
    return columns


def test_declared_boolean_columns_are_emitted_as_nullable_booleans() -> None:
    columns = _comparison_columns()

    for name in _BOOLEAN_SPEC_NAMES:
        assert columns[name].dtype == "boolean", name

    assert columns["pdb_redo_revised"].tolist() == [False, True, False]


def test_boolean_spec_and_emitted_dtype_agree() -> None:
    declared = {
        spec.name
        for spec in PDB_REDO_COMPARISON_ENGINE.spec.columns
        if spec.dtype == "boolean"
    }

    assert declared == set(_BOOLEAN_SPEC_NAMES)
