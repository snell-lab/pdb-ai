"""Live contract tests against the installed rabdam-custom library.

Every other RABDAM test builds its own RabdamCustomResult. Nothing else in the
suite would notice if upstream renamed a symbol, moved a module, or changed the
shape of a result object. These tests call the real library end to end and
assert on the surface that integration.py actually consumes.

They deliberately do not check scientific values against a reference, because
that is the job of upstream's own test suite. What they pin is the integration
contract and the fact that PDB-AI runs on upstream's default parameters.
"""

from __future__ import annotations

import dataclasses
import gzip
import shutil
from pathlib import Path
from typing import Any

import pytest

from pdbai.engines.analyses.rabdam import integration
from pdbai.engines.analyses.rabdam.integration import (
    RabdamCustomError,
    RabdamCustomResult,
    run_rabdam_custom_for_structure,
)

# Only ASP and GLU are used, so the structure always carries the carboxyl
# oxygens that Bnet site selection requires.
_SIDE_CHAINS: dict[str, tuple[tuple[str, str], ...]] = {
    "ASP": (
        ("N", "N"),
        ("CA", "C"),
        ("C", "C"),
        ("O", "O"),
        ("CB", "C"),
        ("CG", "C"),
        ("OD1", "O"),
        ("OD2", "O"),
    ),
    "GLU": (
        ("N", "N"),
        ("CA", "C"),
        ("C", "C"),
        ("O", "O"),
        ("CB", "C"),
        ("CG", "C"),
        ("CD", "C"),
        ("OE1", "O"),
        ("OE2", "O"),
    ),
}

_ATOM_SITE_COLUMNS = """loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num"""


def _write_structure(path: Path, residue_count: int) -> Path:
    # The structure is a single P1 chain of alternating ASP and GLU. The
    # coordinates only need enough spread for packing density to vary between
    # atoms.
    rows: list[str] = []
    serial = 0
    for index in range(residue_count):
        component = "ASP" if index % 2 == 0 else "GLU"
        for offset, (atom, element) in enumerate(_SIDE_CHAINS[component]):
            serial += 1
            x = 4.0 * index + (offset % 3) * 1.3
            y = (offset // 3) * 1.3 + (index % 2) * 0.7
            z = (offset % 2) * 1.1
            b_factor = 20.0 + (index * 1.7 + offset) % 25
            rows.append(
                f"ATOM {serial} {element} {atom} . {component} A 1 {index + 1} ? "
                f"{x:.3f} {y:.3f} {z:.3f} 1.00 {b_factor:.2f} "
                f"{index + 1} {component} A {atom} 1"
            )

    path.write_text(
        "data_TEST\n"
        "_cell.length_a 60.000\n"
        "_cell.length_b 60.000\n"
        "_cell.length_c 60.000\n"
        "_cell.angle_alpha 90.000\n"
        "_cell.angle_beta 90.000\n"
        "_cell.angle_gamma 90.000\n"
        "_symmetry.space_group_name_H-M 'P 1'\n"
        "_refine.ls_d_res_high 2.00\n\n"
        f"{_ATOM_SITE_COLUMNS}\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def scored_structure(tmp_path_factory: pytest.TempPathFactory) -> RabdamCustomResult:
    # 140 residues make 1190 atoms. The structure is scored once per module
    # because each run takes about a second.
    path = _write_structure(
        tmp_path_factory.mktemp("rabdam") / "test.cif", residue_count=140
    )
    return run_rabdam_custom_for_structure("TEST", path)


def test_rabdam_custom_returns_every_field_the_integration_reads(
    scored_structure: RabdamCustomResult,
) -> None:
    result = scored_structure

    assert result.scored_atom_count == 1190
    assert list(result.per_atom.columns) == [
        "_atom_site.id",
        "bdamage",
        "packing_density",
    ]
    # Serials must survive as the join key the atom table is merged on.
    assert result.per_atom["_atom_site.id"].tolist() == [
        str(serial) for serial in range(1, 1191)
    ]
    assert (result.per_atom["bdamage"] > 0).all()
    assert (result.per_atom["packing_density"] > 0).all()

    assert result.bnet is not None
    assert result.site_count is not None and result.site_count > 0
    assert result.resolution_angstrom == 2.0
    assert result.warnings == ()


def test_bnet_percentile_resolves_against_the_vendored_reference_database(
    scored_structure: RabdamCustomResult,
) -> None:
    # This test exercises the bundled reference table, which the installed
    # package omits. A missing or unreadable copy would give None and a warning
    # instead.
    assert scored_structure.bnet_percentile is not None
    assert 0.0 <= scored_structure.bnet_percentile <= 100.0


def test_rabdam_custom_reads_the_bundled_table_under_its_manifest() -> None:
    # rabdam-custom finds the manifest beside the table by its name. The
    # percentile is then ranked in the table the run manifest's receipt names.
    database = integration._load_reference_db()  # pyright: ignore[reportPrivateUsage]

    assert database is not None
    assert database.entry_count == 124948
    assert database.metadata.database_id == "pdb_redo_8_20_202606_strict_cryo_bnet"


def _window_size(path: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    # The window size is not part of the result, so it is read off the
    # workflow rabdam-custom returns to the integration.
    upstream = integration._import_rabdam_custom()  # pyright: ignore[reportPrivateUsage]
    workflows: list[Any] = []

    def calculate(structure_data: object) -> Any:
        workflow = upstream.calculate_bdamage_for_structure_data(structure_data)
        workflows.append(workflow)
        return workflow

    with monkeypatch.context() as patch:
        patch.setattr(
            integration,
            "_import_rabdam_custom",
            lambda: dataclasses.replace(
                upstream, calculate_bdamage_for_structure_data=calculate
            ),
        )
        run_rabdam_custom_for_structure("TEST", path)
    return int(workflows[0].bdamage_score_result.window_size)


def test_window_size_shows_upstream_defaults_are_still_in_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # PDB-AI passes no workflow options, so upstream's defaults decide the
    # window size. This test pins both branches of that rule. Above the
    # minimum, the window is 2% of the atom count. That value is rounded and
    # then raised to an odd number. For 1190 atoms, round(23.8) = 24 is raised
    # to 25. Below the minimum, the window is 11. If either default changes
    # upstream, scores stop matching the defaults the bundled reference table
    # was built at, and this test fails.
    large = _write_structure(tmp_path / "large.cif", residue_count=140)
    small = _write_structure(tmp_path / "small.cif", residue_count=12)

    assert _window_size(large, monkeypatch) == 25
    assert _window_size(small, monkeypatch) == 11


def test_a_gzipped_file_scores_as_the_plain_file_does(tmp_path: Path) -> None:
    # An upload given to the command line may be gzipped, and rabdam-custom's
    # own input resolver refuses a .gz name. The integration bypasses that
    # resolver, so a gzipped file gets the same scores as the plain one.
    plain = _write_structure(tmp_path / "small.cif", residue_count=12)
    gzipped = tmp_path / "small.cif.gz"
    with plain.open("rb") as source, gzip.open(gzipped, "wb") as target:
        shutil.copyfileobj(source, target)

    expected = run_rabdam_custom_for_structure("TEST", plain)
    result = run_rabdam_custom_for_structure("TEST", gzipped)

    assert result.bnet == expected.bnet
    assert result.bnet_percentile == expected.bnet_percentile
    assert result.resolution_angstrom == 2.0
    assert result.per_atom.equals(expected.per_atom)


def test_a_missing_file_raises_without_its_absolute_path(tmp_path: Path) -> None:
    # The reason reaches the run report and the manifest, which name files
    # without the directories they lie in.
    missing = tmp_path / "absent.cif"

    with pytest.raises(RabdamCustomError, match="absent.cif") as raised:
        run_rabdam_custom_for_structure("TEST", missing)

    assert str(tmp_path) not in str(raised.value)
