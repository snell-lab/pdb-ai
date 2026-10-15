from __future__ import annotations

import math

import pandas as pd
import pytest

from pdbai.engines.analyses.sasa import compute as compute_module
from pdbai.engines.analyses.sasa import engine as engine_module
from pdbai.engines.analyses.sasa.columns import (
    ALL_OUTPUT_COLUMNS,
    BOOLEAN_OUTPUT_COLUMNS,
    NUMERIC_OUTPUT_COLUMNS,
    SASA_COLUMN_DESCRIPTIONS,
)
from pdbai.engines.analyses.sasa.compute import (
    SASAComputationResult,
    compute_structure_sasa,
)
from pdbai.engines.analyses.sasa.constants import DSSP_RADIUS_CA, SASA_PROBE_RADIUS
from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.core import EngineContext, EngineOptions


def test_sasa_empty_input_has_stable_failure_diagnostics() -> None:
    result = compute_structure_sasa(pd.DataFrame(), accuracy_mode="fast")

    assert result.status == "failed"
    assert not result.ok
    assert [warning.code for warning in result.warnings] == ["EMPTY_INPUT"]
    assert result.columns.empty
    assert "sasa_status" in result.columns


def test_sasa_calculates_identical_coordinate_models_independently() -> None:
    dataframe = pd.DataFrame(
        {
            "pdb_id": ["1ABC", "1ABC"],
            "model_num": ["1", "2"],
            "chain_id": ["A", "A"],
            "res_num": ["1", "1"],
            "res_name": ["ALA", "ALA"],
            "atom_name": ["CA", "CA"],
            "element": ["C", "C"],
            "is_hetatm": [False, False],
            "_atom_site.label_asym_id": ["A", "A"],
            "_atom_site.label_seq_id": ["1", "1"],
            "_pdbai_entity_type": ["polymer", "polymer"],
            "_pdbai_polymer_type": ["polypeptide(L)", "polypeptide(L)"],
            "occupancy": [1.0, 1.0],
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        },
        index=[3, 8],
    )

    result = compute_structure_sasa(dataframe, accuracy_mode="fast")

    # Both models put a CA at the origin. If the models were pooled, each atom
    # would bury the other. A full isolated-sphere area for each atom shows the
    # models were computed apart.
    areas = result.columns["atom_sasa_protein"]
    expected = 4.0 * math.pi * (DSSP_RADIUS_CA + SASA_PROBE_RADIUS) ** 2
    assert areas.loc[[3, 8]].tolist() == pytest.approx([expected, expected], rel=1e-12)


def _protein_atom_with_ligand(ligand_occupancy: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pdb_id": ["1ABC", "1ABC"],
            "model_num": ["1", "1"],
            "chain_id": ["A", "A"],
            "res_num": ["1", "2"],
            "res_name": ["ALA", "LIG"],
            "atom_name": ["CA", "C1"],
            "element": ["C", "C"],
            "is_hetatm": [False, True],
            "_atom_site.label_asym_id": ["A", "B"],
            "_atom_site.label_seq_id": ["1", ""],
            "_pdbai_entity_type": ["polymer", "non-polymer"],
            "_pdbai_polymer_type": ["polypeptide(L)", ""],
            "occupancy": [1.0, ligand_occupancy],
            "x": [0.0, 3.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        }
    )


def test_partial_occupancy_occluder_is_reported_without_degrading_status() -> None:
    # The protein-context areas are exact either way, so an ambiguous bound
    # context is surfaced through its own flag rather than through sasa_ok.
    result = compute_structure_sasa(
        _protein_atom_with_ligand(0.5), accuracy_mode="fast"
    )

    assert result.status == "ok"
    assert result.ok
    assert [warning.code for warning in result.warnings] == [
        "PARTIAL_OCCUPANCY_NONPROTEIN"
    ]
    assert [warning.degrades for warning in result.warnings] == [False]
    assert result.columns["bound_context_ambiguous"].iloc[0]
    # The warning code is still written on the atom's row even though it does
    # not count against the status.
    assert (
        result.columns["sasa_warning_codes"].iloc[0] == "PARTIAL_OCCUPANCY_NONPROTEIN"
    )


def test_fully_occupied_occluder_raises_no_warning_at_all() -> None:
    result = compute_structure_sasa(
        _protein_atom_with_ligand(1.0), accuracy_mode="fast"
    )

    assert result.status == "ok"
    assert result.warnings == ()
    assert not result.columns["bound_context_ambiguous"].iloc[0]


def test_degrading_warning_still_degrades_status() -> None:
    # A polymer atom with no canonical residue identity loses its
    # residue-level SASA. That is a real shortfall, not a note about the
    # surroundings.
    dataframe = pd.DataFrame(
        {
            "pdb_id": ["1ABC"],
            "model_num": ["1"],
            "chain_id": ["A"],
            "res_num": [""],
            "res_name": ["ALA"],
            "atom_name": ["CA"],
            "element": ["C"],
            "is_hetatm": [False],
            "_atom_site.label_asym_id": ["A"],
            "_atom_site.label_seq_id": [""],
            "_pdbai_entity_type": ["polymer"],
            "_pdbai_polymer_type": ["polypeptide(L)"],
            "occupancy": [1.0],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
        }
    )

    result = compute_structure_sasa(dataframe, accuracy_mode="fast")

    assert result.status == "degraded"
    assert not result.ok
    assert [warning.degrades for warning in result.warnings] == [True]
    # Nothing about the bound context failed, so only the warning degrades it.
    assert result.columns["bound_context_complete"].iloc[0]


def test_sasa_engine_passes_options_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = pd.DataFrame({"x": [0.0]}, index=[12])
    columns = pd.DataFrame({"sasa_status": ["ok"]}, index=dataframe.index)
    captured: dict[str, object] = {}

    def fake_compute(
        input_dataframe: pd.DataFrame, **kwargs: object
    ) -> SASAComputationResult:
        assert input_dataframe is dataframe
        captured.update(kwargs)
        return SASAComputationResult(columns, "ok")

    monkeypatch.setattr(engine_module, "compute_structure_sasa", fake_compute)
    options = EngineOptions(
        values={
            "include_sasa": True,
            "sasa_accuracy_mode": "high",
            "sasa_n_sphere_points": 123,
            "sasa_include_experimental_additives": True,
            "sasa_include_unknown_nonpolymers": False,
        }
    )
    context = EngineContext(
        "1ABC",
        dataframe,
        options,
        structure_context="biological_assembly",
    )

    result = engine_module.SASA_ENGINE.run(context)

    assert result.columns is columns
    assert result.report["sasa_status"] == "ok"
    assert captured == {
        "accuracy_mode": "high",
        "n_sphere_points": 123,
        "include_experimental_additives": True,
        "include_unknown_nonpolymers": False,
        "structure_context": "biological_assembly",
    }


def _chain_rows(residues: list[tuple[str, list[str]]]) -> pd.DataFrame:
    # Every atom of the chain lies on the x axis, 1.33 Å from the one before.
    # Each residue lists C last and the next residue lists N first, so every
    # peptide bond is confirmed and only the two chain ends are terminal.
    rows: list[dict[str, object]] = []
    for number, (residue_name, atom_names) in enumerate(residues, start=1):
        for atom_name in atom_names:
            rows.append(
                {
                    "pdb_id": "1ABC",
                    "model_num": "1",
                    "chain_id": "A",
                    "res_num": str(number),
                    "res_name": residue_name,
                    "atom_name": atom_name,
                    "element": atom_name[:2] if atom_name == "SE" else atom_name[0],
                    "is_hetatm": residue_name == "CSE",
                    "_atom_site.label_asym_id": "A",
                    "_atom_site.label_seq_id": str(number),
                    "_pdbai_entity_type": "polymer",
                    "_pdbai_polymer_type": "polypeptide(L)",
                    "occupancy": 1.0,
                    "x": 1.33 * len(rows),
                    "y": 0.0,
                    "z": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_complete_selenocysteine_residue_is_eligible_for_normalization() -> None:
    glycine = ["N", "CA", "O", "C"]
    dataframe = _chain_rows(
        [
            ("GLY", glycine),
            ("CSE", ["N", "CA", "CB", "SE", "O", "C"]),
            ("GLY", glycine),
        ]
    )

    result = compute_structure_sasa(dataframe, accuracy_mode="fast")

    selenocysteine = result.columns[dataframe["res_name"] == "CSE"]
    assert selenocysteine["residue_sasa_complete"].all()
    assert selenocysteine["rasa_eligible"].all()
    assert selenocysteine["rasa_ineligible_reason"].isna().all()
    assert selenocysteine["rasa_protein"].notna().all()


@pytest.mark.parametrize("include_unknown_nonpolymers", [True, False])
def test_bound_context_is_not_ambiguous_when_no_polymer_atom_is_valid(
    include_unknown_nonpolymers: bool,
) -> None:
    # The protein atom has zero occupancy, so no model has a target and no
    # bound context is built. The unknown non-polymer is never included in
    # one, whatever the option says.
    dataframe = _protein_atom_with_ligand(1.0)
    dataframe.loc[0, "occupancy"] = 0.0
    dataframe.loc[1, "res_name"] = "UNL"

    result = compute_structure_sasa(
        dataframe,
        accuracy_mode="fast",
        include_unknown_nonpolymers=include_unknown_nonpolymers,
    )

    assert result.status == "failed"
    assert [warning.code for warning in result.warnings] == ["NO_VALID_PROTEIN_ATOMS"]
    assert result.columns["sasa_component_category"].iloc[1] == "unknown_nonpolymer"
    assert result.columns["has_unknown_nonpolymer"].all()
    assert not result.columns["bound_context_ambiguous"].any()


def test_residue_is_not_complete_when_its_protein_area_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_sasa(*args: object, **kwargs: object) -> object:
        raise RuntimeError("grid exhausted")

    monkeypatch.setattr(compute_module, "compute_sasa", failing_sasa)
    dataframe = _chain_rows([("GLY", ["N", "CA", "O", "C"])])

    result = compute_structure_sasa(dataframe, accuracy_mode="fast")

    assert [warning.code for warning in result.warnings] == ["PROTEIN_MODEL_FAILED"]
    assert result.columns["residue_sasa_protein"].isna().all()
    # Every expected atom is present, but none has an area.
    assert not result.columns["residue_sasa_complete"].any()
    assert result.columns["residue_sasa_complete"].notna().all()


def test_every_declared_column_is_produced_with_its_declared_kind() -> None:
    glycine = ["N", "CA", "O", "C"]
    dataframe = _chain_rows([("GLY", glycine), ("ALA", glycine), ("GLY", glycine)])

    result = compute_structure_sasa(dataframe, accuracy_mode="fast")

    columns = result.columns
    assert list(columns.columns) == list(ALL_OUTPUT_COLUMNS)
    for name in BOOLEAN_OUTPUT_COLUMNS:
        assert columns[name].dtype == "boolean", name
    for name in NUMERIC_OUTPUT_COLUMNS:
        assert pd.api.types.is_numeric_dtype(columns[name].dtype), name
    assert set(SASA_COLUMN_DESCRIPTIONS) == set(ALL_OUTPUT_COLUMNS)
    for column in engine_module.SASA_ENGINE.spec.columns:
        assert column.description == SASA_COLUMN_DESCRIPTIONS[column.name]
    assert columns["sasa_n_sphere_points"].eq(92).all()
    assert columns["sasa_probe_radius"].eq(SASA_PROBE_RADIUS).all()
    # The alanine lacks its CB, so it is incomplete as well as inside the
    # chain, and the glycines are the two chain ends.
    assert columns["rasa_ineligible_reason"].tolist() == (
        ["terminal_residue"] * 4 + ["missing_atoms:CB"] * 4 + ["terminal_residue"] * 4
    )
    assert (
        columns["is_terminal_residue"].tolist() == [True] * 4 + [False] * 4 + [True] * 4
    )


def test_hydrogens_are_neither_targets_nor_occluders() -> None:
    dataframe = _protein_atom_with_ligand(1.0)
    dataframe.loc[1, "res_name"] = "HOH"
    hydrogens = pd.DataFrame(
        [
            # A protein hydrogen, a deuterium with no element, and a hydrogen
            # on the ligand, each close enough to bury part of the CA.
            {**dataframe.iloc[0].to_dict(), "atom_name": "HA", "element": "H"},
            {**dataframe.iloc[0].to_dict(), "atom_name": "D1", "element": ""},
            {**_protein_atom_with_ligand(1.0).iloc[1].to_dict(), "element": "H"},
        ]
    )
    hydrogens[["x", "y"]] = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]

    result = compute_structure_sasa(
        pd.concat([dataframe, hydrogens], ignore_index=True), accuracy_mode="fast"
    )

    areas = result.columns["atom_sasa_bound"]
    expected = 4.0 * math.pi * (DSSP_RADIUS_CA + SASA_PROBE_RADIUS) ** 2
    assert areas.iloc[0] == pytest.approx(expected, rel=1e-12)
    assert areas.iloc[1:].isna().all()


@pytest.mark.parametrize(
    ("residue_name", "include_additives", "occludes"),
    [
        ("LIG", False, True),
        ("HOH", True, False),
        ("GOL", False, False),
        ("GOL", True, True),
    ],
)
def test_the_bound_context_includes_the_partners_the_policy_names(
    residue_name: str, include_additives: bool, occludes: bool
) -> None:
    dataframe = _protein_atom_with_ligand(1.0)
    dataframe.loc[1, "res_name"] = residue_name

    result = compute_structure_sasa(
        dataframe,
        accuracy_mode="fast",
        include_experimental_additives=include_additives,
    )

    columns = result.columns.iloc[0]
    assert result.status == "ok"
    assert columns["atom_sasa_protein"] == pytest.approx(
        4.0 * math.pi * (DSSP_RADIUS_CA + SASA_PROBE_RADIUS) ** 2
    )
    assert bool(columns["atom_sasa_bound"] < columns["atom_sasa_protein"]) is occludes
    assert bool(columns["has_bound_nonprotein"]) is occludes


def test_the_web_form_describes_the_unknown_nonpolymer_setting_as_it_acts() -> None:
    (option,) = (
        option
        for option in engine_module.SASA_ENGINE.spec.options
        if option.name == "sasa_include_unknown_nonpolymers"
    )

    # The checkbox turns the setting on, while the command-line flag turns it
    # off, so the form needs a caption of its own.
    assert option.default is True
    assert option.cli_negated
    assert option.form_caption.startswith("Unclassified non-polymers occlude")
    assert option.help.startswith("Exclude")


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (None, "**sasa_accuracy_mode** (`fast`)"),
        (
            3000,
            "**sasa_accuracy_mode** (`custom`): the named mode `fast` was "
            "overridden by sasa_n_sphere_points.",
        ),
    ],
)
def test_the_summary_names_the_mode_the_rows_record(
    points: int | None, expected: str
) -> None:
    # The rows say custom whenever a point count is given, so the summary
    # must not report the named mode it overrode.
    accuracy = next(
        option
        for option in engine_module.SASA_ENGINE.spec.options
        if option.name == "sasa_accuracy_mode"
    )
    options = get_engine_registry().with_defaults(
        EngineOptions(
            values={
                "include_sasa": True,
                "sasa_accuracy_mode": "fast",
                "sasa_n_sphere_points": points,
            }
        )
    )

    assert accuracy.summary_line(options).startswith(expected)
