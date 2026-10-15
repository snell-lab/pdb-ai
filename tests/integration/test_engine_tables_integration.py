from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from pdbai.engines.core import (
    ColumnGranularity,
    ColumnSpec,
    DatasetShapeName,
    EngineContext,
    EngineOptions,
    EnginePhase,
    EngineResult,
    EngineSpec,
    TableColumnSpec,
    TableLink,
    TableSpec,
    site_atom_link,
    site_key_columns,
    site_key_dataframe,
)
from pdbai.orchestration import build_atom_site_dataset_from_mmcif_files
from pdbai.runs import open_run

SITES = TableSpec(
    name="stub_sites",
    description="One row per metal atom.",
    key=("site_id",),
    columns=(
        *site_key_columns(),
        TableColumnSpec("metal_element", categories=("ZN", "FE")),
        TableColumnSpec("coordination_number", dtype="Int64"),
        TableColumnSpec("ligand_distance_min_A", dtype="float64"),
    ),
    links=(site_atom_link(),),
)
BONDS = TableSpec(
    name="stub_bonds",
    description="One row per metal-ligand pair.",
    key=("site_id", "ligand_site_id"),
    columns=(
        TableColumnSpec("site_id"),
        *site_key_columns("ligand_"),
        TableColumnSpec("distance_A", dtype="float64"),
    ),
    links=(
        TableLink(target="stub_sites", columns=("site_id",)),
        site_atom_link("ligand_", role="ligand"),
    ),
)


class StubSiteEngine:
    """Stub engine that pairs each zinc atom with every non-zinc atom.

    It writes a sites table and a bonds table, enough to take an engine
    table through every step of a build.
    """

    spec = EngineSpec(
        name="stub_sites",
        display_name="Stub metal sites",
        phase=EnginePhase.ANNOTATION,
        default_enabled=True,
        columns=(
            ColumnSpec(
                "stub_site_id",
                group="Stub Fields",
                granularity=ColumnGranularity.ATOM,
            ),
        ),
        tables=(SITES, BONDS),
    )

    def run(self, context: EngineContext) -> EngineResult:
        atoms = context.atom_dataframe
        is_metal = atoms["element"].astype(str).eq("ZN")
        metal_keys = site_key_dataframe(atoms[is_metal])
        ligands = atoms[~is_metal]
        ligand_keys = site_key_dataframe(ligands, prefix="ligand_")

        link = pd.Series(pd.NA, index=atoms.index, dtype=object)
        if metal_keys.empty:
            # A declared table the result leaves out is written as its header
            # alone.
            return EngineResult(columns=pd.DataFrame({"stub_site_id": link}))

        def points(rows: pd.DataFrame) -> list[tuple[float, float, float]]:
            return list(
                zip(
                    rows["x"].astype(float).tolist(),
                    rows["y"].astype(float).tolist(),
                    rows["z"].astype(float).tolist(),
                    strict=True,
                )
            )

        ligand_points = points(ligands)
        bonds: list[pd.DataFrame] = []
        for metal_point, site_id in zip(
            points(atoms[is_metal]), metal_keys["site_id"].tolist(), strict=True
        ):
            distances = [math.dist(metal_point, point) for point in ligand_points]
            bonds.append(ligand_keys.assign(site_id=site_id, distance_A=distances))
        bond_rows = pd.concat(bonds, ignore_index=True)

        sites = metal_keys.assign(
            metal_element="ZN",
            coordination_number=len(ligands),
            ligand_distance_min_A=float(bond_rows["distance_A"].min()),
        )
        link[is_metal] = metal_keys["site_id"].astype(object)
        return EngineResult(
            columns=pd.DataFrame({"stub_site_id": link}),
            tables={"stub_sites": sites, "stub_bonds": bond_rows},
        )


@pytest.fixture
def stub_site_engine(install_engines: Callable[..., None]) -> None:
    install_engines(StubSiteEngine())


@pytest.mark.parametrize(
    ("dataset_shape", "base_tables"),
    [
        ("flat", set[str]()),
        ("star_schema", {"residues", "chains", "structures", "citations"}),
    ],
)
@pytest.mark.usefixtures("stub_site_engine")
def test_engine_tables_are_separate_files_in_both_output_modes(
    minimal_cif: Path,
    tmp_path: Path,
    dataset_shape: DatasetShapeName,
    base_tables: set[str],
    offline_options: EngineOptions,
) -> None:
    working_directory = tmp_path / "work"
    result = build_atom_site_dataset_from_mmcif_files(
        [minimal_cif],
        working_directory,
        run_id="tables",
        engine_options=dataclasses.replace(
            offline_options, dataset_shape=dataset_shape
        ),
    )

    assert set(result.additional_csv_paths) == base_tables | {
        "stub_sites",
        "stub_bonds",
    }
    assert result.additional_csv_paths["stub_sites"].name == (
        "pdb_atom_site_dataset_stub_sites.csv"
    )
    sites = pd.read_csv(
        result.additional_csv_paths["stub_sites"], keep_default_na=False
    )
    bonds = pd.read_csv(
        result.additional_csv_paths["stub_bonds"], keep_default_na=False
    )
    assert list(sites.columns) == list(SITES.column_names)
    assert list(bonds.columns) == list(BONDS.column_names)
    assert sites["site_id"].tolist() == ["1ABC:1:X:8::ZN:"]
    assert sites["coordination_number"].tolist() == [3]
    assert len(bonds) == 3
    assert set(bonds["site_id"]) == {"1ABC:1:X:8::ZN:"}
    assert "1ABC:1:X:7::N:" in set(bonds["ligand_site_id"])
    assert bonds["distance_A"].min() == pytest.approx(1.0)

    # Atoms carry only the link back to their site.
    atoms = pd.read_csv(result.csv_path)
    assert atoms["stub_site_id"].dropna().tolist() == ["1ABC:1:X:8::ZN:"]

    run_root = working_directory / "runs" / "tables"
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    records = {record["name"]: record for record in manifest["engine_tables"]}
    assert set(records) == {"stub_sites", "stub_bonds"}
    assert records["stub_bonds"]["engine"] == "stub_sites"
    assert records["stub_bonds"]["key"] == ["site_id", "ligand_site_id"]
    assert records["stub_bonds"]["links"][0]["target"] == "stub_sites"
    assert records["stub_sites"]["columns"][9]["categories"] == ["ZN", "FE"]
    entry = manifest["entries"]["1ABC"]
    assert entry["engine_table_row_counts"] == {"stub_sites": 1, "stub_bonds": 3}
    assert entry["partial_paths"]["stub_bonds"] == "partials/stub_bonds/1ABC.csv"
    roles = {output["role"] for output in manifest["provenance"]["outputs"]}
    assert {"dataset_stub_sites", "published_dataset_stub_bonds"} <= roles

    report = pd.read_csv(result.run_report_path)
    assert report["stub_sites_row_count"].tolist() == [1]
    assert report["stub_bonds_row_count"].tolist() == [3]

    summary = result.run_summary_path.read_text(encoding="utf-8")
    assert "## Engine Tables" in summary
    assert "| `stub_bonds` | stub_sites | dataset_stub_bonds.csv | 3 |" in summary
    assert "### Table: `stub_sites`" in summary
    assert "One row per metal atom." in summary


@pytest.mark.usefixtures("stub_site_engine")
def test_structures_without_sites_still_write_header_only_tables(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    no_metal = tmp_path / "2xyz.cif"
    no_metal.write_text(
        "\n".join(
            line
            for line in minimal_cif.read_text(encoding="utf-8").splitlines()
            if not line.startswith("HETATM")
        ).replace("data_1abc", "data_2xyz")
        + "\n",
        encoding="utf-8",
    )

    result = build_atom_site_dataset_from_mmcif_files(
        [no_metal],
        tmp_path / "work",
        run_id="empty-tables",
        engine_options=offline_options,
    )

    partial = tmp_path / "work" / "runs" / "empty-tables" / "partials"
    header = (partial / "stub_bonds" / "2XYZ.csv").read_text(encoding="utf-8")
    assert header.strip() == ",".join(BONDS.column_names)
    sites = pd.read_csv(result.additional_csv_paths["stub_sites"])
    assert sites.empty
    assert list(sites.columns) == list(SITES.column_names)


@pytest.mark.usefixtures("stub_site_engine")
def test_resume_redoes_structures_missing_an_engine_table_partial(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    working_directory = tmp_path / "work"
    build_atom_site_dataset_from_mmcif_files(
        [minimal_cif],
        working_directory,
        run_id="resume-tables",
        engine_options=offline_options,
    )
    state = open_run(working_directory, "resume-tables")
    tables = ["flat", "stub_sites", "stub_bonds"]
    assert state.done_pdb_ids(required_tables=tables) == {"1ABC"}

    # An entry missing the partial of any table the run writes is redone.
    manifest = state.manifest()
    entry = manifest["entries"]["1ABC"]
    partial_paths = dict(entry["partial_paths"])
    del partial_paths["stub_bonds"]
    state.update_entry("1ABC", partial_paths=partial_paths)
    assert state.done_pdb_ids() == {"1ABC"}
    assert state.done_pdb_ids(required_tables=tables) == set()
