from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pdbai.engines.analyses.rabdam import calculation, integration
from pdbai.engines.analyses.rabdam.calculation import (
    add_agreement_flags,
    place_scores_on_atom_table,
    score_structure_file,
)
from pdbai.engines.analyses.rabdam.engine import (
    RABDAM_ENGINE,
    RABDAM_STATUS_AVAILABLE,
    RABDAM_STATUS_FAILED,
    RABDAM_STATUS_PARTIAL,
    RabdamEngine,
)
from pdbai.engines.analyses.rabdam.integration import (
    RabdamCustomError,
    RabdamCustomNotInstalledError,
    RabdamCustomResult,
)
from pdbai.engines.core import (
    STRUCTURE_CIF_PATH,
    ColumnGranularity,
    EngineContext,
    EngineOptions,
    EngineResult,
)
from pdbai.engines.sources.pdb_redo import PDB_REDO_CIF_PATH, PDB_REDO_DATAFRAME


def _rabdam_result() -> RabdamCustomResult:
    return RabdamCustomResult(
        per_atom=pd.DataFrame(
            {
                "_atom_site.id": ["2", "1"],
                "bdamage": [1.2, 0.8],
                "packing_density": [8.0, 7.0],
            }
        ),
        bnet=0.25,
        bnet_percentile=72.0,
        resolution_angstrom=1.5,
        site_count=2,
    )


def _aligned_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Both DataFrames hold the same two atoms. They match by residue identity,
    # so RABDAM can map PDB-Redo scores onto the RCSB atoms.
    identity = {"chain_id": "A", "res_num": "7", "res_name": "GLY"}
    rows = [
        {**identity, "atom_name": "N", "_atom_site.id": "1"},
        {**identity, "atom_name": "CA", "_atom_site.id": "2"},
    ]
    return pd.DataFrame(rows), pd.DataFrame(rows)


# An upload has no named source, so its columns carry no _pdb_redo or _rcsb
# suffix.
_BARE_COLUMNS = {
    "bdamage",
    "packing_density",
    "bnet",
    "bnet_percentile",
    "bnet_site_count",
    "bdamage_scored_atom_count",
    "bnet_percentile_resolution_in_range",
}


# These files are never opened. The stub scorer uses the paths only as keys.
RCSB_CIF = Path("rcsb.cif")
PDB_REDO_CIF = Path("pdb_redo.cif")


def _run(
    dataframe: pd.DataFrame,
    options: EngineOptions,
    *,
    pdb_redo_result: RabdamCustomResult | None = None,
    rcsb_result: RabdamCustomResult | None = None,
    pdb_redo_dataframe: pd.DataFrame | None = None,
    upload: bool = False,
    pdb_id: str = "1ABC",
) -> EngineResult:
    # rabdam-custom is replaced by a scorer that returns the given result for
    # each structure file. A source without a result has no file, so the
    # engine never calls the scorer for it.
    scores = {PDB_REDO_CIF: pdb_redo_result, RCSB_CIF: rcsb_result}

    def scorer(
        structure_id: str,
        path: Path,
        structure: pd.DataFrame,
        warnings: list[str],
    ) -> RabdamCustomResult | None:
        return scores[path]

    context = EngineContext(
        pdb_id, dataframe, options, source="upload" if upload else "download"
    )
    if rcsb_result is not None:
        context.artifacts.put(STRUCTURE_CIF_PATH, RCSB_CIF)
    if pdb_redo_result is not None:
        context.artifacts.put(PDB_REDO_CIF_PATH, PDB_REDO_CIF)
    if pdb_redo_dataframe is not None:
        context.artifacts.put(PDB_REDO_DATAFRAME, pdb_redo_dataframe)
    return RabdamEngine(scorer=scorer).run(context)


def _outcome(result: EngineResult) -> dict[str, object]:
    return {
        name: result.report[name]
        for name in (
            "rabdam_status",
            "rabdam_message",
            "rabdam_delivered_pdb_redo",
            "rabdam_delivered_rcsb",
            "rabdam_alignment_match_rate",
        )
    }


def _run_for(source: str, *, source_distinction: bool = True) -> set[str]:
    dataframe = pd.DataFrame({"_atom_site.id": ["1", "2"]}, index=[5, 7])
    result = _run(
        dataframe,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": source}),
        pdb_redo_result=_rabdam_result() if source_distinction else None,
        rcsb_result=_rabdam_result(),
        upload=not source_distinction,
    )
    assert result.columns is not None
    return {str(column) for column in result.columns.columns}


def test_rabdam_schema_matches_what_each_source_mode_produces() -> None:
    for source in ("pdb_redo", "rcsb", "both"):
        options = EngineOptions(
            values={"include_rabdam": True, "rabdam_source": source}
        )
        enabled = {
            column.name
            for column in RABDAM_ENGINE.spec.columns
            if column.is_enabled(options, source="download")
        }

        assert enabled - _BARE_COLUMNS == _run_for(source)

    assert _run_for("pdb_redo", source_distinction=False) == _BARE_COLUMNS
    alignment_specs = [
        column
        for column in RABDAM_ENGINE.spec.columns
        if column.name.startswith(("rabdam_alignment_", "bdamage_coverage"))
    ]
    assert {column.name for column in alignment_specs} == {
        "bdamage_coverage_pdb_redo",
        "rabdam_alignment_rcsb_ambiguous_atom_count",
        "rabdam_alignment_pdb_redo_ambiguous_atom_count",
    }
    assert all(
        column.granularity is ColumnGranularity.STRUCTURE for column in alignment_specs
    )


def test_rabdam_engine_maps_rcsb_scores_by_atom_id() -> None:
    dataframe = pd.DataFrame({"_atom_site.id": ["1", "2", "3"]}, index=[5, 7, 9])

    result = _run(
        dataframe,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "rcsb"}),
        rcsb_result=_rabdam_result(),
    )

    assert result.columns is not None
    assert result.columns.index.equals(dataframe.index)
    assert result.columns["bdamage_rcsb"].iloc[:2].tolist() == [0.8, 1.2]
    assert math.isnan(result.columns["bdamage_rcsb"].iloc[2])
    assert result.columns["bnet_rcsb"].tolist() == [0.25, 0.25, 0.25]
    assert result.columns["bnet_percentile_rcsb"].tolist() == [72.0] * 3
    assert _outcome(result) == {
        "rabdam_status": RABDAM_STATUS_AVAILABLE,
        "rabdam_message": None,
        "rabdam_delivered_pdb_redo": None,
        "rabdam_delivered_rcsb": True,
        "rabdam_alignment_match_rate": None,
    }


def test_rabdam_pdb_redo_mode_reports_only_its_requested_source() -> None:
    rcsb, pdb_redo = _aligned_dataframes()
    result = _run(
        rcsb,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "pdb_redo"}),
        pdb_redo_result=_rabdam_result(),
        pdb_redo_dataframe=pdb_redo,
    )

    assert _outcome(result) == {
        "rabdam_status": RABDAM_STATUS_AVAILABLE,
        "rabdam_message": None,
        "rabdam_delivered_pdb_redo": True,
        "rabdam_delivered_rcsb": None,
        "rabdam_alignment_match_rate": 1.0,
    }


def test_rabdam_both_is_available_only_when_every_source_is_delivered() -> None:
    rcsb, pdb_redo = _aligned_dataframes()
    dataframe = pd.DataFrame({"_atom_site.id": ["1", "2"]})

    complete = _run(
        rcsb,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "both"}),
        pdb_redo_result=_rabdam_result(),
        rcsb_result=_rabdam_result(),
        pdb_redo_dataframe=pdb_redo,
    )
    pdb_redo_only = _run(
        rcsb,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "both"}),
        pdb_redo_result=_rabdam_result(),
        pdb_redo_dataframe=pdb_redo,
    )
    rcsb_only = _run(
        dataframe,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "both"}),
        rcsb_result=_rabdam_result(),
    )

    assert complete.report["rabdam_status"] == RABDAM_STATUS_AVAILABLE
    assert complete.report["rabdam_delivered_pdb_redo"] is True
    assert complete.report["rabdam_delivered_rcsb"] is True

    assert pdb_redo_only.report["rabdam_status"] == RABDAM_STATUS_PARTIAL
    assert pdb_redo_only.report["rabdam_delivered_pdb_redo"] is True
    assert pdb_redo_only.report["rabdam_delivered_rcsb"] is False

    assert _outcome(rcsb_only) == {
        "rabdam_status": RABDAM_STATUS_PARTIAL,
        # PDB-Redo had no file to score, and the message names that source.
        "rabdam_message": (
            "No structure file of the PDB-Redo model was available for 1ABC, so "
            "its RABDAM columns are empty."
        ),
        "rabdam_delivered_pdb_redo": False,
        "rabdam_delivered_rcsb": True,
        "rabdam_alignment_match_rate": None,
    }


def test_rabdam_both_reports_failed_when_neither_source_succeeds() -> None:
    dataframe = pd.DataFrame({"_atom_site.id": ["1", "2"]})
    result = _run(
        dataframe,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "both"}),
    )

    assert _outcome(result) == {
        "rabdam_status": RABDAM_STATUS_FAILED,
        # Each missing file is reported under its own source.
        "rabdam_message": (
            "No structure file of the PDB-Redo model was available for 1ABC, so "
            "its RABDAM columns are empty. | No structure file of the RCSB model "
            "was available for 1ABC, so its RABDAM columns are empty."
        ),
        "rabdam_delivered_pdb_redo": False,
        "rabdam_delivered_rcsb": False,
        "rabdam_alignment_match_rate": None,
    }


def test_upload_rabdam_does_not_claim_a_named_source() -> None:
    dataframe = pd.DataFrame({"_atom_site.id": ["1", "2"]})
    result = _run(
        dataframe,
        EngineOptions(values={"include_rabdam": True}),
        rcsb_result=_rabdam_result(),
        upload=True,
        pdb_id="upload",
    )

    assert _outcome(result) == {
        "rabdam_status": RABDAM_STATUS_AVAILABLE,
        "rabdam_message": None,
        "rabdam_delivered_pdb_redo": None,
        "rabdam_delivered_rcsb": None,
        "rabdam_alignment_match_rate": None,
    }


def test_rabdam_pdb_redo_bridge_keeps_models_distinct() -> None:
    identity = {
        "chain_id": "A",
        "res_num": "7",
        "res_name": "GLY",
        "atom_name": "CA",
    }
    rcsb = pd.DataFrame(
        [
            {**identity, "model_num": "1", "_atom_site.id": "rcsb-1"},
            {**identity, "model_num": "2", "_atom_site.id": "rcsb-2"},
        ],
        index=[10, 20],
    )
    pdb_redo = pd.DataFrame(
        [
            {**identity, "model_num": "2", "_atom_site.id": "2"},
            {**identity, "model_num": "1", "_atom_site.id": "1"},
        ]
    )

    result = place_scores_on_atom_table(
        rcsb,
        _rabdam_result(),
        pdb_redo,
        pdb_id="1ABC",
        suffix="_pdb_redo",
        requires_alignment=True,
    )

    assert result.columns.index.equals(rcsb.index)
    assert result.columns["bdamage_pdb_redo"].tolist() == [0.8, 1.2]
    assert result.columns["packing_density_pdb_redo"].tolist() == [7.0, 8.0]
    assert result.alignment is not None
    assert result.alignment.match_rate == 1.0
    assert result.alignment.rcsb_ambiguous_atom_count == 0
    assert result.alignment.pdb_redo_ambiguous_atom_count == 0

    engine_result = _run(
        rcsb,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "pdb_redo"}),
        pdb_redo_result=_rabdam_result(),
        pdb_redo_dataframe=pdb_redo,
    )
    assert engine_result.columns is not None
    assert engine_result.columns["bdamage_coverage_pdb_redo"].tolist() == [1.0, 1.0]
    assert engine_result.report["rabdam_alignment_match_rate"] == 1.0


def test_rabdam_never_maps_pdb_redo_scores_by_raw_atom_serial() -> None:
    # RABDAM scored the PDB-Redo model, but no PDB-Redo atom table was
    # published. That happens when the PDB-Redo file fails to parse. Atom
    # serials are renumbered between refinements, so a join on serials would
    # give each score to an unrelated atom.
    rcsb, _ = _aligned_dataframes()

    result = _run(
        rcsb,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "pdb_redo"}),
        pdb_redo_result=_rabdam_result(),
    )

    assert result.columns is not None
    assert result.columns["bdamage_pdb_redo"].isna().all()
    assert result.columns["packing_density_pdb_redo"].isna().all()
    # The structure-level scores do not depend on per-atom matching, so they
    # survive intact and the run is partial rather than failed.
    assert result.columns["bnet_pdb_redo"].tolist() == [0.25, 0.25]
    assert result.columns["bnet_percentile_pdb_redo"].tolist() == [72.0, 72.0]
    assert result.report["rabdam_delivered_pdb_redo"] is False
    assert result.report["rabdam_status"] == RABDAM_STATUS_PARTIAL
    assert result.warnings == (
        "rabdam-custom scored the PDB-Redo model for 1ABC, but no parsed PDB-Redo "
        "atom table was available to pair its atoms by identity. Its BDamage and "
        "packing-density columns are therefore empty for this structure.",
    )


def test_rabdam_upload_path_still_maps_scores_by_atom_serial() -> None:
    # An uploaded structure is scored on the same file it was parsed from, so
    # serials remain the only available join and stay correct.
    dataframe = pd.DataFrame({"_atom_site.id": ["1", "2"]})

    result = _run(
        dataframe,
        EngineOptions(values={"include_rabdam": True}),
        rcsb_result=_rabdam_result(),
        upload=True,
        pdb_id="upload",
    )

    assert result.columns is not None
    assert result.columns["bdamage"].tolist() == [0.8, 1.2]
    assert result.warnings == ()


def test_rabdam_agreement_distinguishes_mismatch_from_missing() -> None:
    # The first pair differs by less than the 1e-3 agreement tolerance, and the
    # second by more.
    dataframe = pd.DataFrame(
        {
            "bdamage_pdb_redo": [1.0, 1.0, None],
            "bdamage_rcsb": [1.0005, 1.01, 1.0],
        }
    )

    result = add_agreement_flags(dataframe)

    assert bool(result.loc[0, "bdamage_matches"])
    assert not bool(result.loc[1, "bdamage_matches"])
    assert pd.isna(result.loc[2, "bdamage_matches"])


def test_pdb_redo_bridge_rejects_mispaired_atoms() -> None:
    # The identity fields agree, but the PDB-Redo atoms sit 25 Å away. That is
    # far past the 1 Å limit on the median shift. The merge must not attach
    # BDamage to atoms that are not the same atoms.
    identity = {
        "pdb_id": "1ABC",
        "model_num": "1",
        "chain_id": "A",
        "res_num": "10",
        "res_name": "GLY",
        "y": 0.0,
        "z": 0.0,
    }
    rcsb = pd.DataFrame(
        [
            {**identity, "atom_name": "N", "x": 0.0, "_atom_site.id": "rcsb-1"},
            {**identity, "atom_name": "CA", "x": 1.0, "_atom_site.id": "rcsb-2"},
        ]
    )
    pdb_redo = pd.DataFrame(
        [
            {**identity, "atom_name": "N", "x": 25.0, "_atom_site.id": "1"},
            {**identity, "atom_name": "CA", "x": 26.0, "_atom_site.id": "2"},
        ]
    )

    result = place_scores_on_atom_table(
        rcsb,
        _rabdam_result(),
        pdb_redo,
        pdb_id="1ABC",
        suffix="_pdb_redo",
        requires_alignment=True,
    )

    assert result.unaligned_reason is not None
    assert "25.0 Å" in result.unaligned_reason
    assert result.columns["bdamage_pdb_redo"].isna().all()
    assert result.columns["packing_density_pdb_redo"].isna().all()
    assert result.alignment is not None
    assert result.alignment.match_rate == 0.0


def test_bdamage_coverage_counts_protein_heavy_atoms_only() -> None:
    # The RCSB model holds two protein heavy atoms, a hydrogen, and a water.
    # PDB-Redo scores only one of the heavy atoms. Coverage counts protein
    # heavy atoms only, so it is 1 of 2, not 1 of 4.
    base = {"pdb_id": "1ABC", "model_num": "1", "chain_id": "A", "y": 0.0, "z": 0.0}
    protein = {
        **base,
        "res_num": "10",
        "res_name": "GLY",
        "_pdbai_entity_type": "polymer",
        "_pdbai_polymer_type": "polypeptide(L)",
    }
    rcsb = pd.DataFrame(
        [
            {
                **protein,
                "atom_name": "N",
                "element": "N",
                "x": 0.0,
                "_atom_site.id": "r1",
            },
            {
                **protein,
                "atom_name": "CA",
                "element": "C",
                "x": 1.0,
                "_atom_site.id": "r2",
            },
            {
                **protein,
                "atom_name": "H",
                "element": "H",
                "x": 0.5,
                "_atom_site.id": "r3",
            },
            {
                **base,
                "res_num": "201",
                "res_name": "HOH",
                "_pdbai_entity_type": "water",
                "_pdbai_polymer_type": "",
                "atom_name": "O",
                "element": "O",
                "x": 9.0,
                "_atom_site.id": "r4",
            },
        ]
    )
    # The PDB-Redo N sits 0.05 Å from the RCSB N. That is a typical
    # re-refinement shift, well inside the 1 Å limit.
    pdb_redo = pd.DataFrame(
        [{**protein, "atom_name": "N", "element": "N", "x": 0.05, "_atom_site.id": "1"}]
    )
    scores = RabdamCustomResult(
        per_atom=pd.DataFrame(
            {"_atom_site.id": ["1"], "bdamage": [1.1], "packing_density": [5.0]}
        ),
        bnet=0.25,
        bnet_percentile=72.0,
        resolution_angstrom=1.5,
        site_count=1,
    )

    engine_result = _run(
        rcsb,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "pdb_redo"}),
        pdb_redo_result=scores,
        pdb_redo_dataframe=pdb_redo,
    )

    assert engine_result.columns is not None
    assert engine_result.columns["bdamage_coverage_pdb_redo"].tolist() == [0.5] * 4
    assert engine_result.columns["bdamage_pdb_redo"].notna().tolist() == [
        True,
        False,
        False,
        False,
    ]
    # Partial coverage describes the data rather than a fault, so it raises no
    # warning.
    assert engine_result.warnings == ()


def _result_with(**overrides: Any) -> RabdamCustomResult:
    return dataclasses.replace(_rabdam_result(), **overrides)


def _run_with_result(
    result: RabdamCustomResult, source: str = "pdb_redo"
) -> EngineResult:
    rcsb, pdb_redo = _aligned_dataframes()
    return _run(
        rcsb,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": source}),
        pdb_redo_result=result,
        rcsb_result=result,
        pdb_redo_dataframe=pdb_redo,
    )


def test_structure_score_bookkeeping_columns_are_emitted() -> None:
    result = _result_with(
        site_count=37,
        resolution_angstrom=2.0,
        percentile_resolution_min=1.98,
        percentile_resolution_max=2.02,
    )

    engine_result = _run_with_result(result)

    assert engine_result.columns is not None
    columns = engine_result.columns
    assert columns["bnet_site_count_pdb_redo"].tolist() == [37, 37]
    assert columns["bdamage_scored_atom_count_pdb_redo"].tolist() == [2, 2]
    assert columns["bnet_percentile_resolution_in_range_pdb_redo"].tolist() == [
        True,
        True,
    ]
    assert engine_result.report["rabdam_status"] == "available"


def test_percentile_out_of_reference_range_is_flagged_not_dropped() -> None:
    # A 4.5 Å structure lies past the end of the reference database. It is
    # ranked against the 3.3 to 3.5 Å tail, so its percentile is an
    # extrapolation.
    result = _result_with(
        resolution_angstrom=4.5,
        percentile_resolution_min=3.3,
        percentile_resolution_max=3.5,
    )

    engine_result = _run_with_result(result)

    assert engine_result.columns is not None
    columns = engine_result.columns
    assert columns["bnet_percentile_pdb_redo"].tolist() == [72.0, 72.0]
    assert columns["bnet_percentile_resolution_in_range_pdb_redo"].tolist() == [
        False,
        False,
    ]
    assert engine_result.report["rabdam_status"] == "available"


def test_in_range_flag_is_missing_without_a_percentile_window() -> None:
    engine_result = _run_with_result(_rabdam_result())

    assert engine_result.columns is not None
    flag = engine_result.columns["bnet_percentile_resolution_in_range_pdb_redo"]
    assert flag.isna().all()
    assert str(flag.dtype) == "boolean"


def test_missing_bnet_makes_the_source_partial() -> None:
    # BDamage was scored, but Bnet could not be computed from too few sites.
    result = _result_with(bnet=None, bnet_percentile=None, site_count=None)

    engine_result = _run_with_result(result)

    assert engine_result.columns is not None
    columns = engine_result.columns
    assert columns["bdamage_pdb_redo"].notna().all()
    assert columns["bnet_pdb_redo"].isna().all()
    assert columns["bnet_site_count_pdb_redo"].isna().all()
    assert engine_result.report["rabdam_status"] == "partial"
    assert engine_result.report["rabdam_delivered_pdb_redo"] is False


def test_missing_percentile_alone_makes_the_source_partial() -> None:
    result = _result_with(bnet_percentile=None)

    engine_result = _run_with_result(result)

    assert engine_result.columns is not None
    assert engine_result.columns["bnet_pdb_redo"].notna().all()
    assert engine_result.report["rabdam_status"] == "partial"


def test_upload_path_is_partial_without_bnet() -> None:
    result = _result_with(bnet=None, bnet_percentile=None, site_count=None)
    dataframe = pd.DataFrame({"_atom_site.id": ["1", "2"]}, index=[5, 7])

    engine_result = _run(
        dataframe,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "pdb_redo"}),
        rcsb_result=result,
        upload=True,
    )

    assert engine_result.columns is not None
    assert engine_result.columns["bdamage"].notna().all()
    assert engine_result.columns["bnet"].isna().all()
    assert engine_result.report["rabdam_status"] == "partial"


def test_both_mode_is_partial_when_only_one_source_has_bnet() -> None:
    rcsb, pdb_redo = _aligned_dataframes()
    engine_result = _run(
        rcsb,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "both"}),
        pdb_redo_result=_rabdam_result(),
        rcsb_result=_result_with(bnet=None, bnet_percentile=None),
        pdb_redo_dataframe=pdb_redo,
    )

    assert engine_result.report["rabdam_status"] == "partial"
    assert engine_result.report["rabdam_delivered_pdb_redo"] is True
    assert engine_result.report["rabdam_delivered_rcsb"] is False


def test_in_range_flag_tolerates_reference_database_rounding() -> None:
    # The reference database stores resolution to two decimals. A busy bin can
    # hold all 1000 nearest neighbours, so the window collapses to a single
    # value, 1.55 Å. A 1.553 Å structure was ranked against exactly the right
    # neighbours. It must not be flagged as extrapolated. 8ROS, 9C4W, and 9D8U
    # hit this case.
    result = _result_with(
        resolution_angstrom=1.553,
        percentile_resolution_min=1.55,
        percentile_resolution_max=1.55,
    )

    engine_result = _run_with_result(result)

    assert engine_result.columns is not None
    assert engine_result.columns[
        "bnet_percentile_resolution_in_range_pdb_redo"
    ].tolist() == [True, True]


def test_in_range_flag_still_catches_genuine_extrapolation_near_the_edge() -> None:
    # Rounded to two decimals, 3.51 Å is still 3.51 Å. That lies outside a
    # window ending at 3.50 Å.
    result = _result_with(
        resolution_angstrom=3.51,
        percentile_resolution_min=3.30,
        percentile_resolution_max=3.50,
    )

    engine_result = _run_with_result(result)

    assert engine_result.columns is not None
    assert engine_result.columns[
        "bnet_percentile_resolution_in_range_pdb_redo"
    ].tolist() == [False, False]


def test_bnet_failure_reason_is_carried_as_the_report_message() -> None:
    # rabdam-custom explains why Bnet is missing, for example zero variance or
    # too few sites. The engine reports that text so the manifest and run
    # report can keep it beside the partial status.
    reason = "rabdam-custom Bnet unavailable for 1ABC: zero variance"
    without_bnet = dataclasses.replace(
        _rabdam_result(),
        bnet=None,
        bnet_percentile=None,
        site_count=None,
        warnings=(reason,),
    )

    def scorer(
        structure_id: str,
        path: Path,
        structure: pd.DataFrame,
        warnings: list[str],
    ) -> RabdamCustomResult | None:
        warnings.extend(without_bnet.warnings)
        return without_bnet

    rcsb, pdb_redo = _aligned_dataframes()
    context = EngineContext(
        "1ABC",
        rcsb,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "pdb_redo"}),
    )
    context.artifacts.put(PDB_REDO_CIF_PATH, PDB_REDO_CIF)
    context.artifacts.put(PDB_REDO_DATAFRAME, pdb_redo)
    result = RabdamEngine(scorer=scorer).run(context)

    assert result.report["rabdam_status"] == RABDAM_STATUS_PARTIAL
    assert result.report["rabdam_message"] == f"The PDB-Redo model: {reason}"


def test_the_same_scorer_warning_from_both_sources_is_kept_for_each() -> None:
    reason = "rabdam-custom Bnet unavailable for 1ABC: zero variance"

    def scorer(
        structure_id: str,
        path: Path,
        structure: pd.DataFrame,
        warnings: list[str],
    ) -> RabdamCustomResult | None:
        warnings.append(reason)
        return dataclasses.replace(_rabdam_result(), bnet=None, bnet_percentile=None)

    rcsb, pdb_redo = _aligned_dataframes()
    context = EngineContext(
        "1ABC",
        rcsb,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "both"}),
    )
    context.artifacts.put(STRUCTURE_CIF_PATH, RCSB_CIF)
    context.artifacts.put(PDB_REDO_CIF_PATH, PDB_REDO_CIF)
    context.artifacts.put(PDB_REDO_DATAFRAME, pdb_redo)
    result = RabdamEngine(scorer=scorer).run(context)

    assert result.report["rabdam_message"] == (
        f"The PDB-Redo model: {reason} | The RCSB model: {reason}"
    )


def test_scored_atoms_without_a_row_are_counted_in_a_warning() -> None:
    # rabdam-custom scored serial 3, the conformer the atom table dropped.
    scored = dataclasses.replace(
        _rabdam_result(),
        per_atom=pd.DataFrame(
            {
                "_atom_site.id": ["1", "2", "3"],
                "bdamage": [0.8, 1.2, 1.1],
                "packing_density": [7.0, 8.0, 9.0],
            }
        ),
    )

    result = _run(
        pd.DataFrame({"_atom_site.id": ["1", "2"]}),
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "rcsb"}),
        rcsb_result=scored,
    )

    assert result.report["rabdam_status"] == RABDAM_STATUS_AVAILABLE
    assert len(result.warnings) == 1
    assert result.warnings[0].startswith(
        "1 of the atoms rabdam-custom scored on the RCSB model for 1ABC have no "
        "row in the atom table"
    )


def test_report_message_is_missing_without_warnings() -> None:
    assert _run_with_result(_rabdam_result()).report["rabdam_message"] is None


def test_rabdam_reports_its_outcome_under_its_declared_fields() -> None:
    rcsb, pdb_redo = _aligned_dataframes()
    result = _run(
        rcsb,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "pdb_redo"}),
        pdb_redo_result=_rabdam_result(),
        pdb_redo_dataframe=pdb_redo,
    )

    assert set(result.report) == set(RABDAM_ENGINE.spec.report_field_names)
    assert result.report["rabdam_status"] == RABDAM_STATUS_AVAILABLE
    assert result.report["rabdam_delivered_pdb_redo"] is True
    assert result.report["rabdam_delivered_rcsb"] is None
    assert result.report["rabdam_alignment_match_rate"] == 1.0


def test_rabdam_needs_pdb_redo_only_when_it_scores_pdb_redo() -> None:
    spec = RABDAM_ENGINE.spec
    for source, needs in (("pdb_redo", True), ("both", True), ("rcsb", False)):
        options = EngineOptions(
            values={"include_rabdam": True, "rabdam_source": source}
        )
        assert ("pdb_redo_structure" in spec.dependencies_under(options)) is needs


def test_bnet_reaches_every_row_even_when_no_atom_score_does() -> None:
    dataframe = pd.DataFrame({"_atom_site.id": ["1", "2"]})
    without_serials = _result_with(
        per_atom=pd.DataFrame(
            {"_atom_site.id": [], "bdamage": [], "packing_density": []}
        ),
    )

    result = place_scores_on_atom_table(
        dataframe, without_serials, suffix="_rcsb", requires_alignment=False
    )

    assert result.columns["bnet_rcsb"].tolist() == [0.25, 0.25]
    assert result.columns["bdamage_rcsb"].isna().all()
    assert result.unaligned_reason == "it gave no scored atom a serial number"


def test_a_serial_shared_by_two_scored_atoms_pairs_with_neither() -> None:
    # Two scored atoms share serial 1, so neither score can be given to atom
    # 1. That atom is left empty rather than the repeated key failing the
    # engine.
    dataframe = pd.DataFrame({"_atom_site.id": ["1", "2"]})
    shared = _result_with(
        per_atom=pd.DataFrame(
            {
                "_atom_site.id": ["1", "1", "2"],
                "bdamage": [0.5, 0.7, 1.2],
                "packing_density": [3.0, 4.0, 8.0],
            }
        )
    )

    result = place_scores_on_atom_table(
        dataframe, shared, suffix="_rcsb", requires_alignment=False
    )

    assert math.isnan(result.columns["bdamage_rcsb"].iloc[0])
    assert result.columns["bdamage_rcsb"].iloc[1] == 1.2
    assert result.unaligned_reason is None


def test_rcsb_scores_that_match_no_atom_serial_make_the_source_partial() -> None:
    dataframe = pd.DataFrame({"_atom_site.id": ["8", "9"]})

    result = _run(
        dataframe,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "rcsb"}),
        rcsb_result=_rabdam_result(),
    )

    assert result.columns is not None
    assert result.columns["bdamage_rcsb"].isna().all()
    assert result.columns["bnet_rcsb"].tolist() == [0.25, 0.25]
    assert result.report["rabdam_status"] == RABDAM_STATUS_PARTIAL
    assert result.report["rabdam_delivered_rcsb"] is False
    assert result.warnings == (
        "rabdam-custom scored the RCSB model for 1ABC, but none of its atom "
        "serial numbers is in the atom table. Its BDamage and packing-density "
        "columns are therefore empty for this structure.",
    )


def test_a_bridge_that_pairs_no_atom_makes_the_pdb_redo_source_partial() -> None:
    # PDB-Redo numbered the residue differently, so no identity matches. The
    # bridge then has no distance to reject, yet a source whose BDamage is
    # empty on every atom is not delivered.
    rcsb, pdb_redo = _aligned_dataframes()

    result = _run(
        rcsb,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "pdb_redo"}),
        pdb_redo_result=_rabdam_result(),
        pdb_redo_dataframe=pdb_redo.assign(res_num="8"),
    )

    assert result.columns is not None
    assert result.columns["bdamage_pdb_redo"].isna().all()
    assert result.columns["bdamage_coverage_pdb_redo"].tolist() == [0.0, 0.0]
    assert result.report["rabdam_status"] == RABDAM_STATUS_PARTIAL
    assert result.report["rabdam_delivered_pdb_redo"] is False
    assert result.warnings == (
        "rabdam-custom scored the PDB-Redo model for 1ABC, but none of its scored "
        "atoms paired with an RCSB atom by identity. Its BDamage and "
        "packing-density columns are therefore empty for this structure.",
    )


def test_coverage_is_missing_when_pdb_redo_was_not_scored() -> None:
    # Most entries PDB-Redo lacks would otherwise read as scored with no atom
    # covered.
    dataframe = pd.DataFrame({"_atom_site.id": ["1", "2"]})

    result = _run(
        dataframe,
        EngineOptions(values={"include_rabdam": True, "rabdam_source": "both"}),
        rcsb_result=_rabdam_result(),
    )

    assert result.columns is not None
    assert result.columns["bdamage_coverage_pdb_redo"].isna().all()
    assert result.columns["bdamage_rcsb"].tolist() == [0.8, 1.2]


def test_a_file_rabdam_custom_cannot_score_becomes_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(
        pdb_id: str, cif_path: Path, *, resolution_override: float | None = None
    ) -> RabdamCustomResult:
        raise RabdamCustomError("bad file")

    monkeypatch.setattr(calculation, "run_rabdam_custom_for_structure", fail)
    warnings: list[str] = []

    result = score_structure_file("1ABC", RCSB_CIF, pd.DataFrame(), warnings)

    assert result is None
    assert warnings == [
        "rabdam-custom failed for 1ABC: bad file. BDamage columns will be empty "
        "for this structure."
    ]


def test_a_missing_installation_is_raised_rather_than_warned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(
        pdb_id: str, cif_path: Path, *, resolution_override: float | None = None
    ) -> RabdamCustomResult:
        raise RabdamCustomNotInstalledError("No module named 'rabdam'")

    monkeypatch.setattr(calculation, "run_rabdam_custom_for_structure", missing)

    with pytest.raises(RabdamCustomNotInstalledError):
        score_structure_file("1ABC", RCSB_CIF, pd.DataFrame(), [])


def test_the_percentile_is_matched_on_the_atom_table_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, float | None] = {}

    def score(
        pdb_id: str, cif_path: Path, *, resolution_override: float | None = None
    ) -> RabdamCustomResult:
        seen["resolution"] = resolution_override
        return _result_with(warnings=("too few sites",))

    monkeypatch.setattr(calculation, "run_rabdam_custom_for_structure", score)
    warnings: list[str] = []
    dataframe = pd.DataFrame({"resolution": [None, "1.8", "1.8"]})

    result = score_structure_file("1ABC", PDB_REDO_CIF, dataframe, warnings)

    assert result is not None
    assert seen == {"resolution": 1.8}
    assert warnings == ["too few sites"]


def test_the_percentile_is_matched_on_the_xray_refinement_s_resolution_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[float | None] = []

    def score(
        pdb_id: str, cif_path: Path, *, resolution_override: float | None = None
    ) -> RabdamCustomResult:
        seen.append(resolution_override)
        return _result_with()

    monkeypatch.setattr(calculation, "run_rabdam_custom_for_structure", score)
    # A joint entry whose neutron refinement is listed first.
    joint = pd.DataFrame({"resolution": [2.2, 2.2], "resolution_xray": [1.6, 1.6]})
    # A neutron-only entry has no X-ray refinement, so the column is empty.
    neutron_only = pd.DataFrame({"resolution": [2.2], "resolution_xray": [None]})

    score_structure_file("1ABC", PDB_REDO_CIF, joint, [])
    score_structure_file("1ABC", PDB_REDO_CIF, neutron_only, [])

    assert seen == [1.6, 2.2]


def test_preflight_refuses_a_run_without_rabdam_custom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = RABDAM_ENGINE.spec.preflight
    assert preflight is not None
    preflight()

    def missing(module_name: str, attribute: str) -> Any:
        raise ImportError(f"No module named {module_name!r}")

    monkeypatch.setattr(integration, "_import_callable", missing)

    with pytest.raises(RabdamCustomNotInstalledError, match="pip install") as raised:
        preflight()
    assert "Underlying import error: No module named" in str(raised.value)
