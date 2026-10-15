from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from pdbai.engines.analyses.disulfides import DISULFIDE_ENGINE
from pdbai.engines.analyses.disulfides.columns import (
    DISULFIDE_LINK_COLUMN,
    DISULFIDES_TABLE,
)
from pdbai.engines.analyses.disulfides.constants import (
    DETECTION_METHODS,
    PAIR_STATUSES,
)
from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.compatibility import require_compatible_options
from pdbai.engines.core import (
    PROFILE_ML_MINIMAL,
    PROFILE_RADIATION_SENSITIVITY,
    PROFILE_RESEARCH_READY,
    STRUCTURE_CIF_PATH,
    DatasetShapeName,
    EngineContext,
    EngineOptions,
)
from pdbai.orchestration import build_atom_site_dataset_from_mmcif_files
from pdbai.profiles import apply_profile, resolve_run_settings
from pdbai.structure import parse_mmcif_atom_rows

_ON = {"include_disulfides": True}

# The representative SG atoms of the fixture's rows, as its docstring lists
# them.
_LINKED_SITES = {
    "1DSF:1:A:101::SG:",
    "1DSF:1:A:140::SG:",
    "1DSF:1:A:105::SG:A",
    "1DSF:1:A:130::SG:A",
    "1DSF:1:A:120:A:SG:",
    "1DSF:1:A:160::SG:",
    "1DSF:1:A:125::SG:",
    "1DSF:1:B:301::SG:",
    "1DSF:1:A:155::SG:",
    "1DSF:1:A:176::SG:",
    "1DSF:1:A:194::SG:",
    "1DSF:1:A:210::SG:",
    "1DSF:1:A:213::SG:",
    "1DSF:2:A:101::SG:",
    "1DSF:2:A:140::SG:",
}


def _write_label_named_cif(path: Path, *, residue_numbers: bool) -> Path:
    """Write two cysteines named by their label fields alone, and return the path.

    Some integrative models are deposited this way. The file has no
    _atom_site.auth_comp_id, and no _atom_site.auth_seq_id either unless
    ``residue_numbers`` is set. Cys 18 and Cys 107 of author chain A, which
    is label chain L, have their SG atoms 2.03 Å apart on the x axis.
    """
    atoms = [
        ("CB", "18", "0", "-1.8"),
        ("SG", "18", "0", "0"),
        ("CB", "107", "2.03", "-1.8"),
        ("SG", "107", "2.03", "0"),
    ]
    rows = [
        {
            "group_PDB": "ATOM",
            "id": str(number),
            "type_symbol": name[0],
            "label_atom_id": name,
            "label_comp_id": "CYS",
            "label_asym_id": "L",
            "label_seq_id": residue,
            "Cartn_x": x,
            "Cartn_y": "0",
            "Cartn_z": z,
            "occupancy": "1.00",
            "B_iso_or_equiv": "20",
            "auth_asym_id": "A",
            "pdbx_PDB_model_num": "1",
            **({"auth_seq_id": residue} if residue_numbers else {}),
        }
        for number, (name, residue, x, z) in enumerate(atoms, start=1)
    ]
    lines = [
        "data_label",
        "loop_",
        *(f"_atom_site.{item}" for item in rows[0]),
        *(" ".join(row.values()) for row in rows),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _context(path: Path, *, with_file: bool = True) -> EngineContext:
    """Return the context of an upload of ``path``, its file published or not."""
    atoms = parse_mmcif_atom_rows(path, {}, structure_id=path.stem).dataframe
    context = EngineContext(
        path.stem, atoms, EngineOptions(values=_ON), source="upload"
    )
    if with_file:
        context.artifacts.put(STRUCTURE_CIF_PATH, path)
    return context


def test_the_engine_needs_the_structure_file(disulfide_cif: Path) -> None:
    with pytest.raises(RuntimeError, match="No structure file was available for 1dsf"):
        DISULFIDE_ENGINE.run(_context(disulfide_cif, with_file=False))


def test_the_engine_reports_the_fixture(disulfide_cif: Path) -> None:
    context = _context(disulfide_cif)

    result = DISULFIDE_ENGINE.run(context)

    assert set(result.tables) == {"disulfides"}
    table = result.tables["disulfides"]
    assert len(table) == 8
    assert table["metal_bridged"].sum() == 1
    # The ID comes from the file name, not the data block XXXX.
    assert set(table["pdb_id"]) == {"1DSF"}
    assert result.columns is not None
    assert list(result.columns.columns) == [DISULFIDE_LINK_COLUMN]
    assert result.columns.index.equals(context.atom_dataframe.index)
    assert set(result.columns[DISULFIDE_LINK_COLUMN].dropna()) == _LINKED_SITES
    messages = (
        "_struct_conn record 'disulf4' names a cysteine that has no SG atom, so "
        "it gives no row.",
        "_struct_conn record 'disulf5' names a cysteine with no SG or CB atom in "
        "the structure, so it gives no row.",
        "cysteine A 150 in model 1 has no SG in the conformer the atom table "
        "keeps, so its rows name SG conformer A and no atom links to it.",
    )
    assert result.warnings == tuple(f"Disulfides: {message}" for message in messages)
    assert result.report == {
        "disulfide_status": "degraded",
        "disulfide_unmatched_record_count": 2,
        "disulfide_symmetry_record_count": 1,
        "disulfide_cysteine_without_sg_count": 1,
        "disulfide_message": " | ".join(messages),
    }


def test_a_structure_without_disulfides_writes_no_rows(minimal_cif: Path) -> None:
    result = DISULFIDE_ENGINE.run(_context(minimal_cif))

    assert result.tables == {}
    assert result.columns is not None
    assert result.columns[DISULFIDE_LINK_COLUMN].isna().all()
    assert result.warnings == ()
    assert result.report == {
        "disulfide_status": "ok",
        "disulfide_unmatched_record_count": 0,
        "disulfide_symmetry_record_count": 0,
        "disulfide_cysteine_without_sg_count": 0,
        "disulfide_message": None,
    }


def test_the_engine_is_off_by_default() -> None:
    assert not DISULFIDE_ENGINE.spec.is_enabled(
        get_engine_registry().with_defaults(EngineOptions())
    )


def test_a_default_build_writes_no_disulfides(
    disulfide_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    result = build_atom_site_dataset_from_mmcif_files(
        [disulfide_cif],
        tmp_path / "work",
        run_id="disulfides-off",
        engine_options=offline_options,
    )

    assert "disulfides" not in result.additional_csv_paths
    assert DISULFIDE_LINK_COLUMN not in pd.read_csv(result.csv_path).columns
    partials = tmp_path / "work" / "runs" / "disulfides-off" / "partials"
    assert not (partials / "disulfides").exists()


@pytest.mark.parametrize(
    ("profile_id", "turned_on"),
    [
        (PROFILE_RESEARCH_READY, True),
        (PROFILE_RADIATION_SENSITIVITY, True),
        (PROFILE_ML_MINIMAL, False),
    ],
)
def test_only_the_research_profiles_turn_the_engine_on(
    profile_id: str, turned_on: bool
) -> None:
    applied = apply_profile(EngineOptions(), profile_id)
    resolved = resolve_run_settings(
        {}, technique="x_ray_diffraction", profile_id=profile_id
    )

    assert ("include_disulfides" in applied.toggles_turned_on) is turned_on
    assert resolved.options.flag("include_disulfides") is turned_on
    assert resolved.column_filter_list is not None
    assert (DISULFIDE_LINK_COLUMN in resolved.column_filter_list) is turned_on


def test_the_engine_may_run_on_neutron_structures() -> None:
    options = get_engine_registry().with_defaults(
        EngineOptions(technique="neutron_diffraction", values=_ON)
    )

    require_compatible_options(options, source="upload")
    require_compatible_options(options, source="download")


@pytest.mark.parametrize("dataset_shape", ["flat", "star_schema"])
def test_a_build_writes_the_disulfides_table_and_links_atoms_to_it(
    disulfide_cif: Path,
    tmp_path: Path,
    dataset_shape: DatasetShapeName,
    offline_options: EngineOptions,
) -> None:
    working_directory = tmp_path / "work"
    result = build_atom_site_dataset_from_mmcif_files(
        [disulfide_cif],
        working_directory,
        run_id="disulfides",
        engine_options=dataclasses.replace(
            offline_options, dataset_shape=dataset_shape
        ).with_values(_ON),
    )

    table_path = result.additional_csv_paths["disulfides"]
    assert table_path.name == "pdb_atom_site_dataset_disulfides.csv"
    table = pd.read_csv(
        table_path,
        dtype={column.name: column.dtype for column in DISULFIDES_TABLE.columns},
    )
    assert list(table.columns) == list(DISULFIDES_TABLE.column_names)
    assert len(table.columns) == 29
    assert len(table) == 8
    assert set(table["pdb_id"]) == {"1DSF"}
    assert table["model_num"].tolist() == [1, 1, 1, 1, 1, 1, 1, 2]
    assert table["detection_method"].tolist() == [
        "struct_conn_and_distance",
        "struct_conn_and_distance",
        "distance",
        "struct_conn_and_distance",
        "distance",
        "struct_conn",
        "distance",
        "struct_conn_and_distance",
    ]
    # The flag is written and read back as a Boolean that is never missing,
    # True for the pair that shares a zinc alone.
    assert table["metal_bridged"].dtype == "boolean"
    assert table["metal_bridged"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    assert table["sg_has_alternate_conformers"].tolist() == [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert table["ss_distance_max_A"].tolist() == pytest.approx(
        [2.04, 3.6, 2.7, 2.03, 2.05, 3.6, 2.9, 2.6], abs=1e-9
    )

    # Only the representative SG atoms carry the link, so no zinc-site SG, CB,
    # or hydrogen or deuterium atom does.
    atoms = pd.read_csv(result.csv_path)
    links = atoms[DISULFIDE_LINK_COLUMN].dropna().tolist()
    assert len(links) == len(_LINKED_SITES)
    assert set(links) == _LINKED_SITES

    report = pd.read_csv(result.run_report_path)
    assert report["disulfide_status"].tolist() == ["degraded"]
    assert report["disulfide_unmatched_record_count"].tolist() == [2]
    assert report["disulfide_symmetry_record_count"].tolist() == [1]
    assert report["disulfide_cysteine_without_sg_count"].tolist() == [1]
    assert report["disulfide_message"].str.contains("'disulf4'").tolist() == [True]
    assert report["disulfides_row_count"].tolist() == [8]

    run_root = working_directory / "runs" / "disulfides"
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    records = {record["name"]: record for record in manifest["engine_tables"]}
    record = records["disulfides"]
    assert record["engine"] == "disulfides"
    assert record["key"] == ["site_id", "partner_site_id"]
    assert [link["role"] for link in record["links"]] == ["", "partner"]
    categories = {column["name"]: column["categories"] for column in record["columns"]}
    assert categories["detection_method"] == list(DETECTION_METHODS)
    assert categories["pair_status"] == list(PAIR_STATUSES)
    assert "no_usable_sg" in categories["pair_status"]
    # The manifest records both distances the run used.
    assert "at most 3.0 Å apart" in record["description"]
    assert "within 2.8 Å of one metal atom" in record["description"]
    assert "1.5–3.0" not in record["description"]
    names = [column["name"] for column in record["columns"]]
    assert names[names.index("detection_method") + 1] == "metal_bridged"
    assert (run_root / "partials" / "disulfides" / "1DSF.csv").is_file()


def test_a_build_finds_cysteines_named_by_their_label_fields_alone(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    path = _write_label_named_cif(tmp_path / "9lab.cif", residue_numbers=True)

    result = build_atom_site_dataset_from_mmcif_files(
        [path],
        tmp_path / "work",
        run_id="label-names",
        engine_options=offline_options.with_values(_ON),
    )

    table = pd.read_csv(
        result.additional_csv_paths["disulfides"],
        dtype={column.name: column.dtype for column in DISULFIDES_TABLE.columns},
    )
    assert table["site_id"].tolist() == ["9LAB:1:A:18::SG:"]
    assert table["partner_site_id"].tolist() == ["9LAB:1:A:107::SG:"]
    assert table["detection_method"].tolist() == ["distance"]
    assert table["ss_distance_min_A"].tolist() == pytest.approx([2.03], abs=1e-9)
    # The atom table holds the label residue name for these atoms, and the
    # table's key columns hold what it holds.
    assert table["res_name"].tolist() == ["CYS"]
    assert table["partner_res_name"].tolist() == ["CYS"]
    atoms = pd.read_csv(result.csv_path)
    assert set(atoms["res_name"]) == {"CYS"}
    assert set(atoms["residue_numbering"]) == {"author"}
    assert atoms[DISULFIDE_LINK_COLUMN].dropna().tolist() == [
        "9LAB:1:A:18::SG:",
        "9LAB:1:A:107::SG:",
    ]
    report = pd.read_csv(result.run_report_path)
    assert report["disulfide_status"].tolist() == ["ok"]
    assert report["disulfides_row_count"].tolist() == [1]


def test_a_build_names_cysteines_without_author_numbers_by_label_numbers(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    path = _write_label_named_cif(tmp_path / "9lab.cif", residue_numbers=False)

    result = build_atom_site_dataset_from_mmcif_files(
        [path],
        tmp_path / "work",
        run_id="label-numbering",
        engine_options=offline_options.with_values(_ON),
    )

    # The structure is numbered by its label fields, so its two cysteines are
    # told apart and the engine gives their row under the label chain.
    report = pd.read_csv(result.run_report_path)
    assert report["disulfide_status"].tolist() == ["ok"]
    assert report["disulfides_row_count"].tolist() == [1]
    table = pd.read_csv(
        result.additional_csv_paths["disulfides"],
        dtype={column.name: column.dtype for column in DISULFIDES_TABLE.columns},
    )
    assert table["site_id"].tolist() == ["9LAB:1:L:18::SG:"]
    assert table["partner_site_id"].tolist() == ["9LAB:1:L:107::SG:"]
    assert table["ss_distance_min_A"].tolist() == pytest.approx([2.03], abs=1e-9)
    atoms = pd.read_csv(result.csv_path)
    assert set(atoms["residue_numbering"]) == {"label"}
    assert atoms[DISULFIDE_LINK_COLUMN].dropna().tolist() == [
        "9LAB:1:L:18::SG:",
        "9LAB:1:L:107::SG:",
    ]
