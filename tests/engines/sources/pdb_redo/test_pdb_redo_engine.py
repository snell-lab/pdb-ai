from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fake_http import FakeResponse, FakeSession

from pdbai.engines.core import (
    ArtifactKey,
    ColumnGranularity,
    EngineContext,
    EngineOptions,
    EngineServices,
)
from pdbai.engines.sources.pdb_redo import (
    PDB_REDO_CIF_PATH,
    PDB_REDO_DATAFRAME,
    PDB_REDO_MTZ_PATH,
)
from pdbai.engines.sources.pdb_redo import engine as engine_module
from pdbai.engines.sources.pdb_redo.comparison import compare_with_rcsb_atoms


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


def _acquisition_context(session: FakeSession, cache_root: Path) -> EngineContext:
    return EngineContext(
        "1ABC",
        pd.DataFrame(),
        EngineOptions(values={"include_pdb_redo": True}),
        services=EngineServices(
            session=session.as_session(), timeout=2.5, cache_root=cache_root
        ),
    )


def test_pdb_redo_acquisition_downloads_into_its_cache_directory(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "pdb_redo" / "1abc.cif"
    url = "https://pdb-redo.eu/db/1abc/1abc_final.cif"
    session = FakeSession(FakeResponse(content=b"data_1abc\n"))

    result = engine_module.PDB_REDO_ACQUISITION_ENGINE.run(
        _acquisition_context(session, tmp_path)
    )

    assert result.artifacts == {PDB_REDO_CIF_PATH: expected}
    assert session.calls == [(url, 2.5)]
    assert [
        (source.role, source.path, source.source_url, source.cached_before)
        for source in result.source_files
    ] == [("pdb_redo_mmcif", expected, url, False)]


def test_an_entry_pdb_redo_lacks_is_not_available_rather_than_failed(
    tmp_path: Path,
) -> None:
    context = _acquisition_context(FakeSession(FakeResponse(status_code=404)), tmp_path)

    acquired = engine_module.PDB_REDO_ACQUISITION_ENGINE.run(context)
    parsed = engine_module.PDB_REDO_STRUCTURE_ENGINE.run(context)

    assert acquired.artifacts == {}
    assert parsed.report == {"pdb_redo_structure_parsed": False}
    assert parsed.warnings and "PDB-Redo has no entry for 1ABC" in parsed.warnings[0]


def test_pdb_redo_structure_publishes_the_parsed_model(minimal_cif: Path) -> None:
    context = _context(
        "1ABC",
        pd.DataFrame(),
        EngineOptions(values={"include_pdb_redo": True}),
        artifacts={PDB_REDO_CIF_PATH: minimal_cif},
    )

    result = engine_module.PDB_REDO_STRUCTURE_ENGINE.run(context)

    assert result.report == {"pdb_redo_structure_parsed": True}
    dataframe = result.artifacts[PDB_REDO_DATAFRAME]
    assert isinstance(dataframe, pd.DataFrame) and not dataframe.empty


def test_pdb_redo_atoms_carry_the_entry_id_the_rcsb_atoms_were_parsed_under(
    minimal_cif: Path,
) -> None:
    # The alignment pairs atoms only within one entry ID. The cache file's
    # name is not what the RCSB atoms were parsed under.
    renamed = minimal_cif.rename(minimal_cif.with_name("cached_model.cif"))
    context = _context(
        "1abc",
        pd.DataFrame(),
        EngineOptions(values={"include_pdb_redo": True}),
        artifacts={PDB_REDO_CIF_PATH: renamed},
    )

    result = engine_module.PDB_REDO_STRUCTURE_ENGINE.run(context)

    dataframe = result.artifacts[PDB_REDO_DATAFRAME]
    assert isinstance(dataframe, pd.DataFrame)
    assert set(dataframe["pdb_id"]) == {"1ABC"}


def test_a_file_that_does_not_parse_is_a_warning_rather_than_a_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "1abc.cif"
    path.write_text("data_1abc\n_atom_site.id 1\n", encoding="utf-8")
    context = _context(
        "1ABC",
        pd.DataFrame(),
        EngineOptions(values={"include_pdb_redo": True}),
        artifacts={PDB_REDO_CIF_PATH: path},
    )

    result = engine_module.PDB_REDO_STRUCTURE_ENGINE.run(context)

    assert result.artifacts == {}
    assert result.report == {"pdb_redo_structure_parsed": False}
    assert len(result.warnings) == 1
    assert result.warnings[0].startswith("PDB-Redo parse failed for 1ABC")


def test_a_failed_download_is_not_warned_about_a_second_time() -> None:
    context = _context(
        "1ABC", pd.DataFrame(), EngineOptions(values={"include_pdb_redo": True})
    )
    context.failures["pdb_redo_source"] = "PDB-Redo 1ABC download failed: timeout"

    result = engine_module.PDB_REDO_STRUCTURE_ENGINE.run(context)

    assert result.warnings == ()
    assert result.report == {"pdb_redo_structure_parsed": False}


def test_pdb_redo_comparison_reports_numeric_deltas() -> None:
    identity = {
        "chain_id": "A",
        "res_num": "7",
        "res_name": "GLY",
        "atom_name": "CA",
    }
    rcsb = pd.DataFrame(
        [
            {
                **identity,
                "x": 1.0,
                "y": 2.0,
                "z": 3.0,
                "occupancy": 1.0,
                "_atom_site.B_iso_or_equiv": 10.0,
                "u_equivalent": 0.10,
            }
        ],
        index=[17],
    )
    pdb_redo = pd.DataFrame(
        [
            {
                **identity,
                "x": 1.25,
                "y": 2.0,
                "z": 3.0,
                "occupancy": 0.8,
                "_atom_site.B_iso_or_equiv": 12.0,
                "u_equivalent": 0.12,
            }
        ]
    )
    context = _context(
        "1ABC",
        rcsb,
        EngineOptions(values={"include_pdb_redo": True}),
        artifacts={PDB_REDO_DATAFRAME: pdb_redo},
    )

    result = engine_module.PDB_REDO_COMPARISON_ENGINE.run(context)

    assert result.columns is not None
    assert result.columns.index.equals(rcsb.index)
    assert result.columns.loc[17, "pdb_redo_x"] == pytest.approx(1.25)
    assert result.columns.loc[17, "pdb_redo_x_delta"] == pytest.approx(0.25)
    assert result.columns.loc[17, "pdb_redo_bfactor_delta"] == pytest.approx(2.0)
    assert bool(result.columns.loc[17, "pdb_redo_revised"])
    assert result.columns.loc[17, "pdb_redo_alignment_match_rate"] == pytest.approx(1.0)
    assert result.columns.loc[17, "pdb_redo_alignment_rcsb_ambiguous_atom_count"] == 0
    assert (
        result.columns.loc[17, "pdb_redo_alignment_pdb_redo_ambiguous_atom_count"] == 0
    )
    assert result.report["pdb_redo_comparison_success"] is True
    assert result.report["pdb_redo_alignment_match_rate"] == 1.0


def test_pdb_redo_alignment_columns_are_structure_level() -> None:
    alignment_specs = [
        column
        for column in engine_module.PDB_REDO_COMPARISON_ENGINE.spec.columns
        if column.name.startswith("pdb_redo_alignment_")
    ]

    assert len(alignment_specs) == 3
    assert all(
        column.granularity is ColumnGranularity.STRUCTURE for column in alignment_specs
    )


def test_pdb_redo_comparison_is_a_noop_without_an_acquired_dataframe() -> None:
    context = _context(
        "1ABC",
        pd.DataFrame({"x": [1.0]}),
        EngineOptions(values={"include_pdb_redo": True}),
    )

    result = engine_module.PDB_REDO_COMPARISON_ENGINE.run(context)
    columns = result.columns

    assert columns is not None
    assert columns.index.equals(context.atom_dataframe.index)
    assert columns["pdb_redo_x"].isna().all()
    assert columns["pdb_redo_revised"].isna().all()
    assert columns["pdb_redo_alignment_match_rate"].isna().all()
    assert result.report["pdb_redo_comparison_success"] is False


def test_pdb_redo_comparison_declares_exactly_the_columns_compute_returns() -> None:
    declared = [
        column.name for column in engine_module.PDB_REDO_COMPARISON_ENGINE.spec.columns
    ]
    rcsb = pd.DataFrame(
        [{"chain_id": "A", "res_num": "7", "res_name": "GLY", "atom_name": "CA"}]
    )

    with_partner = compare_with_rcsb_atoms(rcsb, rcsb.copy(), pdb_id="1ABC")
    without_partner = compare_with_rcsb_atoms(rcsb, pd.DataFrame(), pdb_id="1ABC")

    assert sorted(with_partner.columns.columns) == sorted(declared)
    assert sorted(without_partner.columns.columns) == sorted(declared)
    assert declared[:3] == ["pdb_redo_x", "pdb_redo_y", "pdb_redo_z"]


def test_pdb_redo_coordinates_are_written_when_rcsb_lacks_the_column() -> None:
    identity = {"chain_id": "A", "res_num": "7", "res_name": "GLY", "atom_name": "CA"}
    rcsb = pd.DataFrame([{**identity, "x": 1.0, "y": 2.0}])
    pdb_redo = pd.DataFrame([{**identity, "x": 1.5, "y": 2.0, "z": 3.0}])

    columns = compare_with_rcsb_atoms(rcsb, pdb_redo, pdb_id="1ABC").columns

    assert columns["pdb_redo_z"].tolist() == [3.0]
    assert columns["pdb_redo_z_delta"].isna().all()
    assert columns["pdb_redo_x_delta"].tolist() == [0.5]


def test_a_comparison_that_paired_no_atom_is_not_a_success() -> None:
    rcsb = pd.DataFrame(
        [{"chain_id": "A", "res_num": "7", "res_name": "GLY", "atom_name": "CA"}]
    )
    pdb_redo = rcsb.assign(res_num="8")

    result = engine_module.PDB_REDO_COMPARISON_ENGINE.run(
        _context(
            "1ABC",
            rcsb,
            EngineOptions(values={"include_pdb_redo": True}),
            artifacts={PDB_REDO_DATAFRAME: pdb_redo},
        )
    )

    assert result.report["pdb_redo_comparison_success"] is False
    assert result.report["pdb_redo_alignment_match_rate"] == 0.0
    assert len(result.warnings) == 1
    assert "paired no PDB-Redo atom" in result.warnings[0]


def test_a_comparison_on_label_chains_succeeds_with_a_warning() -> None:
    # PDB-Redo wrote the RCSB label chains A and B as its author chains in
    # place of the deposited AAA and BBB, as for 9BB0.
    def atoms(auth_chains: list[str], label_chains: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "pdb_id": "9BB0",
                "model_num": "1",
                "_atom_site.auth_asym_id": auth_chains,
                "_atom_site.label_asym_id": label_chains,
                "_atom_site.auth_seq_id": "10",
                "res_name": "SER",
                "atom_name": "CA",
                "x": [1.0, 50.0],
            }
        )

    result = engine_module.PDB_REDO_COMPARISON_ENGINE.run(
        _context(
            "9BB0",
            atoms(["AAA", "BBB"], ["A", "B"]),
            EngineOptions(values={"include_pdb_redo": True}),
            artifacts={PDB_REDO_DATAFRAME: atoms(["A", "B"], ["A", "B"])},
        )
    )

    assert result.report["pdb_redo_comparison_success"] is True
    assert len(result.warnings) == 1
    assert "aligned on label chain IDs" in result.warnings[0]
    assert result.columns is not None
    assert result.columns["pdb_redo_x"].tolist() == [1.0, 50.0]


def test_pdb_redo_comparison_reports_ambiguous_atom_counts() -> None:
    identity = {
        "chain_id": "A",
        "res_num": "7",
        "res_name": "GLY",
        "atom_name": "CA",
        "x": 1.0,
    }
    rcsb = pd.DataFrame([identity, identity])
    pdb_redo = pd.DataFrame([identity])

    result = engine_module.PDB_REDO_COMPARISON_ENGINE.run(
        _context(
            "1ABC",
            rcsb,
            EngineOptions(values={"include_pdb_redo": True}),
            artifacts={PDB_REDO_DATAFRAME: pdb_redo},
        )
    )

    assert result.columns is not None
    assert result.columns["pdb_redo_alignment_match_rate"].tolist() == [0.0, 0.0]
    assert result.columns["pdb_redo_alignment_rcsb_ambiguous_atom_count"].tolist() == [
        2,
        2,
    ]
    assert result.columns[
        "pdb_redo_alignment_pdb_redo_ambiguous_atom_count"
    ].tolist() == [0, 0]
    assert result.report["pdb_redo_alignment_rcsb_ambiguous_atom_count"] == 2
    assert result.report["pdb_redo_alignment_pdb_redo_ambiguous_atom_count"] == 0


def test_pdb_redo_comparison_keeps_models_distinct() -> None:
    identity = {
        "chain_id": "A",
        "res_num": "7",
        "res_name": "GLY",
        "atom_name": "CA",
    }
    rcsb = pd.DataFrame(
        [
            {**identity, "model_num": "1", "x": 1.0},
            {**identity, "model_num": "2", "x": 10.0},
        ],
        index=[10, 20],
    )
    pdb_redo = pd.DataFrame(
        [
            {**identity, "model_num": "2", "x": 20.0},
            {**identity, "model_num": "1", "x": 2.0},
        ]
    )

    columns = engine_module.PDB_REDO_COMPARISON_ENGINE.run(
        _context(
            "1ABC",
            rcsb,
            EngineOptions(values={"include_pdb_redo": True}),
            artifacts={PDB_REDO_DATAFRAME: pdb_redo},
        )
    ).columns

    assert columns is not None
    assert columns.index.equals(rcsb.index)
    assert columns["pdb_redo_x"].tolist() == [2.0, 20.0]
    assert columns["pdb_redo_x_delta"].tolist() == [1.0, 10.0]


def test_pdb_redo_comparison_keeps_insertion_codes_distinct() -> None:
    identity = {
        "chain_id": "A",
        "res_num": "7",
        "res_name": "GLY",
        "atom_name": "CA",
        "model_num": "1",
    }
    insertion_column = "_atom_site.pdbx_PDB_ins_code"
    rcsb = pd.DataFrame(
        [
            {**identity, insertion_column: "A", "x": 1.0},
            {**identity, insertion_column: "B", "x": 10.0},
        ]
    )
    pdb_redo = pd.DataFrame(
        [
            {**identity, insertion_column: "B", "x": 20.0},
            {**identity, insertion_column: "A", "x": 2.0},
        ]
    )

    columns = engine_module.PDB_REDO_COMPARISON_ENGINE.run(
        _context(
            "1ABC",
            rcsb,
            EngineOptions(values={"include_pdb_redo": True}),
            artifacts={PDB_REDO_DATAFRAME: pdb_redo},
        )
    ).columns

    assert columns is not None
    assert columns["pdb_redo_x"].tolist() == [2.0, 20.0]


def test_pdb_redo_revised_is_missing_without_a_matched_atom() -> None:
    identity = {
        "chain_id": "A",
        "res_num": "7",
        "res_name": "GLY",
        "atom_name": "CA",
    }
    rcsb = pd.DataFrame(
        [
            {**identity, "x": 1.0},
            {**identity, "res_num": "8", "x": 3.0},
        ]
    )
    pdb_redo = pd.DataFrame([{**identity, "x": 1.0}])

    columns = engine_module.PDB_REDO_COMPARISON_ENGINE.run(
        _context(
            "1ABC",
            rcsb,
            EngineOptions(values={"include_pdb_redo": True}),
            artifacts={PDB_REDO_DATAFRAME: pdb_redo},
        )
    ).columns

    assert columns is not None
    assert not bool(columns.loc[0, "pdb_redo_revised"])
    assert pd.isna(columns.loc[1, "pdb_redo_revised"])


def test_pdb_redo_comparison_warns_when_alignment_is_rejected() -> None:
    identity = {
        "chain_id": "A",
        "res_num": "7",
        "res_name": "GLY",
        "atom_name": "CA",
        "y": 0.0,
        "z": 0.0,
        "occupancy": 1.0,
        "_atom_site.B_iso_or_equiv": 10.0,
    }
    rcsb = pd.DataFrame([{**identity, "x": 1.0}], index=[17])
    # The identities agree, but the PDB-Redo atom sits 40 Å away. It cannot be
    # the same atom. In 9NQ3, matching on identity alone paired every atom with
    # an atom in the wrong chain.
    pdb_redo = pd.DataFrame([{**identity, "x": 41.0}])
    context = _context(
        "9NQ3",
        rcsb,
        EngineOptions(values={"include_pdb_redo": True}),
        artifacts={PDB_REDO_DATAFRAME: pdb_redo},
    )

    result = engine_module.PDB_REDO_COMPARISON_ENGINE.run(context)

    assert result.columns is not None
    assert len(result.warnings) == 1
    assert "9NQ3" in result.warnings[0]
    assert "40.0 Å" in result.warnings[0]
    assert pd.isna(result.columns.loc[17, "pdb_redo_x_delta"])
    assert pd.isna(result.columns.loc[17, "pdb_redo_bfactor_delta"])
    assert pd.isna(result.columns.loc[17, "pdb_redo_revised"])
    assert result.columns.loc[17, "pdb_redo_alignment_match_rate"] == pytest.approx(0.0)


def test_pdb_redo_comparison_is_not_a_success_when_the_alignment_is_rejected() -> None:
    identity = {
        "chain_id": "A",
        "res_num": "7",
        "res_name": "GLY",
        "atom_name": "CA",
        "y": 0.0,
        "z": 0.0,
    }
    rcsb = pd.DataFrame([{**identity, "x": 1.0}])
    pdb_redo = pd.DataFrame([{**identity, "x": 41.0}])
    context = _context(
        "9NQ3",
        rcsb,
        EngineOptions(values={"include_pdb_redo": True}),
        artifacts={PDB_REDO_DATAFRAME: pdb_redo},
    )

    result = engine_module.PDB_REDO_COMPARISON_ENGINE.run(context)

    assert result.report["pdb_redo_comparison_success"] is False
    assert len(result.warnings) == 1
    assert "was rejected" in result.warnings[0]


def _protein_dataframe(*, with_hydrogen: bool) -> pd.DataFrame:
    protein = {
        "pdb_id": "1ABC",
        "model_num": "1",
        "chain_id": "A",
        "res_num": "10",
        "res_name": "SER",
        "_pdbai_entity_type": "polymer",
        "_pdbai_polymer_type": "polypeptide(L)",
        "y": 0.0,
        "z": 0.0,
    }
    rows = [
        {**protein, "atom_name": "N", "element": "N", "x": 0.0},
        {**protein, "atom_name": "CA", "element": "C", "x": 1.0},
    ]
    if with_hydrogen:
        rows.append({**protein, "atom_name": "H", "element": "H", "x": 0.5})
    return pd.DataFrame(rows)


def test_comparison_warns_when_protein_heavy_atoms_go_unmatched(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rcsb = _protein_dataframe(with_hydrogen=False)
    pdb_redo = rcsb.iloc[[0]].copy()

    result = compare_with_rcsb_atoms(rcsb, pdb_redo, pdb_id="1ABC")

    assert result.alignment is not None
    assert result.alignment.match_rate == 0.5
    assert "matched only 50.0% of protein heavy atoms (1/2)" in caplog.text
    assert "1ABC" in caplog.text


def test_comparison_does_not_warn_about_stripped_hydrogens(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rcsb = _protein_dataframe(with_hydrogen=True)
    pdb_redo = _protein_dataframe(with_hydrogen=False)

    result = compare_with_rcsb_atoms(rcsb, pdb_redo, pdb_id="1ABC")

    assert result.alignment is not None
    assert result.alignment.match_rate == 1.0
    assert "matched only" not in caplog.text


_MTZ_BYTES = b"MTZ \x00\x00\x00\x00reflections"


def _map_context(session: FakeSession, cache_root: Path) -> EngineContext:
    return EngineContext(
        "1ABC",
        pd.DataFrame(),
        EngineOptions(
            values={"include_density": True, "density_map_source": "pdb_redo"}
        ),
        services=EngineServices(
            session=session.as_session(), timeout=2.5, cache_root=cache_root
        ),
    )


def _mirror_entry(cache_root: Path) -> Path:
    directory = cache_root / "pdb_redo_mirror" / "ab" / "1abc"
    directory.mkdir(parents=True)
    return directory


def test_the_map_source_downloads_the_reflection_file_into_the_cache(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "pdb_redo" / "1abc_final.mtz"
    url = "https://pdb-redo.eu/db/1abc/1abc_final.mtz"
    session = FakeSession(FakeResponse(content=_MTZ_BYTES))

    result = engine_module.PDB_REDO_MAP_ACQUISITION_ENGINE.run(
        _map_context(session, tmp_path)
    )

    assert result.artifacts == {PDB_REDO_MTZ_PATH: expected}
    assert expected.read_bytes() == _MTZ_BYTES
    assert session.calls == [(url, 2.5)]
    assert [
        (source.role, source.path, source.source_url, source.cached_before)
        for source in result.source_files
    ] == [("pdb_redo_mtz", expected, url, False)]


def test_an_entry_without_a_reflection_file_is_not_available_rather_than_failed(
    tmp_path: Path,
) -> None:
    result = engine_module.PDB_REDO_MAP_ACQUISITION_ENGINE.run(
        _map_context(FakeSession(FakeResponse(status_code=404)), tmp_path)
    )

    assert result.artifacts == {}
    assert [source.cached_before for source in result.source_files] == [False]


def test_a_download_that_is_not_an_mtz_file_fails(tmp_path: Path) -> None:
    session = FakeSession(FakeResponse(content=b"<html>Not Found</html>"))

    with pytest.raises(RuntimeError, match="not an MTZ reflection file"):
        engine_module.PDB_REDO_MAP_ACQUISITION_ENGINE.run(
            _map_context(session, tmp_path)
        )
    assert not (tmp_path / "pdb_redo" / "1abc_final.mtz").exists()


def test_both_source_engines_read_a_local_copy_before_the_network(
    tmp_path: Path,
) -> None:
    directory = _mirror_entry(tmp_path)
    model = directory / "1abc_final.cif"
    model.write_bytes(b"data_1abc\n")
    reflections = directory / "1abc_final.mtz"
    reflections.write_bytes(_MTZ_BYTES)
    # A session with no script fails the test on any request.
    session = FakeSession()

    acquired = engine_module.PDB_REDO_ACQUISITION_ENGINE.run(
        _acquisition_context(session, tmp_path)
    )
    mapped = engine_module.PDB_REDO_MAP_ACQUISITION_ENGINE.run(
        _map_context(session, tmp_path)
    )

    assert acquired.artifacts == {PDB_REDO_CIF_PATH: model}
    assert mapped.artifacts == {PDB_REDO_MTZ_PATH: reflections}
    assert session.calls == []
    assert [
        (source.role, source.source_url, source.cached_before)
        for source in (*acquired.source_files, *mapped.source_files)
    ] == [
        ("pdb_redo_mmcif", model.as_uri(), True),
        ("pdb_redo_mtz", reflections.as_uri(), True),
    ]


def test_an_entry_the_local_copy_lacks_is_requested(tmp_path: Path) -> None:
    # An empty file is what an interrupted copy leaves.
    (_mirror_entry(tmp_path) / "1abc_final.mtz").write_bytes(b"")
    session = FakeSession(FakeResponse(content=_MTZ_BYTES))

    result = engine_module.PDB_REDO_MAP_ACQUISITION_ENGINE.run(
        _map_context(session, tmp_path)
    )

    assert result.artifacts == {
        PDB_REDO_MTZ_PATH: tmp_path / "pdb_redo" / "1abc_final.mtz"
    }
    assert len(session.calls) == 1


def test_a_local_reflection_file_that_is_not_mtz_fails(tmp_path: Path) -> None:
    (_mirror_entry(tmp_path) / "1abc_final.mtz").write_bytes(b"<html></html>")

    with pytest.raises(RuntimeError, match="local copy is not an MTZ reflection"):
        engine_module.PDB_REDO_MAP_ACQUISITION_ENGINE.run(
            _map_context(FakeSession(), tmp_path)
        )


def test_a_local_model_file_that_is_not_mmcif_fails(tmp_path: Path) -> None:
    # The copy is checked as a download would be, so an error page saved in
    # its place never reaches the parser.
    (_mirror_entry(tmp_path) / "1abc_final.cif").write_bytes(b"<html></html>")

    with pytest.raises(RuntimeError, match="does not contain an mmCIF data block"):
        engine_module.PDB_REDO_ACQUISITION_ENGINE.run(
            _acquisition_context(FakeSession(), tmp_path)
        )
