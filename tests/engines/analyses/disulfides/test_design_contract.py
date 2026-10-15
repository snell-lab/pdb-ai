"""Black-box tests of the disulfide engine against the edge cases of its design.

Each test writes a small mmCIF file, parses it as the host does, and hands the
atom table and the flat dictionary to compute_structure_disulfides. Every
expectation is taken from the design's edge-case table and the sections it
refers to, not from the implementation. Author residue numbers are label
numbers plus 100, so a test can tell which of the two a lookup used. Each
cysteine's CB sits 1.8 Å below its SG on the z axis, and pairs lie on the x
axis, so every distance is exact.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

from pdbai.engines.analyses.disulfides.columns import DISULFIDES_TABLE
from pdbai.engines.analyses.disulfides.compute import (
    DisulfideComputation,
    compute_structure_disulfides,
)
from pdbai.engines.core.tables import validate_tables
from pdbai.structure import parse_mmcif_atom_rows, read_mmcif_as_flat_dict

_ATOM_SITE_ITEMS = (
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_entity_id",
    "label_seq_id",
    "pdbx_PDB_ins_code",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "B_iso_or_equiv",
    "auth_seq_id",
    "auth_comp_id",
    "auth_asym_id",
    "auth_atom_id",
    "pdbx_PDB_model_num",
)

_STRUCT_CONN_ITEMS = (
    "id",
    "conn_type_id",
    "ptnr1_label_asym_id",
    "ptnr1_label_comp_id",
    "ptnr1_label_seq_id",
    "ptnr1_label_atom_id",
    "ptnr1_auth_asym_id",
    "ptnr1_auth_comp_id",
    "ptnr1_auth_seq_id",
    "pdbx_ptnr1_PDB_ins_code",
    "ptnr1_symmetry",
    "ptnr2_label_asym_id",
    "ptnr2_label_comp_id",
    "ptnr2_label_seq_id",
    "ptnr2_label_atom_id",
    "ptnr2_auth_asym_id",
    "ptnr2_auth_comp_id",
    "ptnr2_auth_seq_id",
    "pdbx_ptnr2_PDB_ins_code",
    "ptnr2_symmetry",
)

_CB_DROP = 1.8


@dataclass(frozen=True)
class Atom:
    """One _atom_site row; numbers are written as given."""

    res: str
    name: str
    x: str
    y: str = "0"
    z: str = "0"
    alt: str = "."
    occ: str = "1.00"
    b: str = "20"
    chain: str = "A"
    ins: str = "?"
    model: str = "1"
    comp: str = "CYS"
    group: str = "ATOM"
    entity: str = "1"
    label_seq: str | None = None
    element: str | None = None

    def values(self, atom_id: int) -> dict[str, str]:
        label_seq = self.label_seq
        if label_seq is None:
            digits = "".join(character for character in self.res if character.isdigit())
            label_seq = str(int(digits) - 100)
        return {
            "group_PDB": self.group,
            "id": str(atom_id),
            "type_symbol": self.element or self.name[0],
            "label_atom_id": self.name,
            "label_alt_id": self.alt,
            "label_comp_id": self.comp,
            "label_asym_id": self.chain,
            "label_entity_id": self.entity,
            "label_seq_id": label_seq,
            "pdbx_PDB_ins_code": self.ins,
            "Cartn_x": self.x,
            "Cartn_y": self.y,
            "Cartn_z": self.z,
            "occupancy": self.occ,
            "B_iso_or_equiv": self.b,
            "auth_seq_id": self.res,
            "auth_comp_id": self.comp,
            "auth_asym_id": self.chain,
            "auth_atom_id": self.name,
            "pdbx_PDB_model_num": self.model,
        }


def _number(value: float) -> str:
    return f"{value:.3f}"


def cys(
    res: str,
    x: float,
    y: float = 0.0,
    *,
    occ: str = "1.00",
    b: str = "20",
    cb_b: str = "20",
    chain: str = "A",
    ins: str = "?",
    model: str = "1",
    alt: str = ".",
    cb_alt: str = ".",
) -> list[Atom]:
    """Return a cysteine's CB and SG, the SG at (x, y, 0)."""
    common = {"chain": chain, "ins": ins, "model": model}
    return [
        Atom(
            res,
            "CB",
            _number(x),
            _number(y),
            _number(-_CB_DROP),
            alt=cb_alt,
            occ=occ,
            b=cb_b,
            **common,
        ),
        Atom(res, "SG", _number(x), _number(y), "0", alt=alt, occ=occ, b=b, **common),
    ]


@dataclass(frozen=True)
class Partner:
    """One partner of a _struct_conn record."""

    res: str
    chain: str = "A"
    ins: str = "?"
    comp: str = "CYS"
    atom: str = "SG"
    symmetry: str = "1_555"
    author: bool = True
    label_seq: str | None = None

    def values(self, number: int) -> dict[str, str]:
        label_seq = self.label_seq or str(int(self.res) - 100)
        return {
            f"ptnr{number}_label_asym_id": self.chain,
            f"ptnr{number}_label_comp_id": self.comp,
            f"ptnr{number}_label_seq_id": label_seq,
            f"ptnr{number}_label_atom_id": self.atom,
            f"ptnr{number}_auth_asym_id": self.chain if self.author else "?",
            f"ptnr{number}_auth_comp_id": self.comp,
            f"ptnr{number}_auth_seq_id": self.res if self.author else "?",
            f"pdbx_ptnr{number}_PDB_ins_code": self.ins,
            f"ptnr{number}_symmetry": self.symmetry,
        }


@dataclass(frozen=True)
class Record:
    """One _struct_conn row."""

    record_id: str
    first: Partner
    second: Partner
    conn_type: str = "disulf"

    def values(self) -> dict[str, str]:
        return {
            "id": self.record_id,
            "conn_type_id": self.conn_type,
            **self.first.values(1),
            **self.second.values(2),
        }


def disulf(record_id: str, first: str, second: str, **kwargs: str) -> Record:
    return Record(record_id, Partner(first), Partner(second), **kwargs)


def _loop(category: str, items: Sequence[str], rows: Iterable[dict[str, str]]) -> str:
    lines = ["loop_", *(f"_{category}.{item}" for item in items)]
    lines.extend(" ".join(row[item] for item in items) for row in rows)
    return "\n".join(lines) + "\n"


@dataclass
class Structure:
    """A hand-written mmCIF file and what the engine computes from it."""

    atoms: list[Atom]
    records: list[Record] = field(default_factory=list[Record])
    omit: frozenset[str] = frozenset()
    block: str = "1tst"

    def text(self) -> str:
        parts = [
            f"data_{self.block}",
            "",
            "loop_\n_entity.id\n_entity.type\n1 polymer\n2 non-polymer\n",
            "loop_\n_entity_poly.entity_id\n_entity_poly.type\n1 'polypeptide(L)'\n",
            "loop_\n_chem_comp.id\n_chem_comp.type\nCYS 'L-peptide linking'\n"
            "ZN non-polymer\nBME non-polymer\n",
        ]
        if self.records:
            items = [item for item in _STRUCT_CONN_ITEMS if item not in self.omit]
            parts.append(
                _loop("struct_conn", items, (r.values() for r in self.records))
            )
        atom_items = [item for item in _ATOM_SITE_ITEMS if item not in self.omit]
        parts.append(
            _loop(
                "atom_site",
                atom_items,
                (atom.values(index) for index, atom in enumerate(self.atoms, 1)),
            )
        )
        return "\n".join(parts)

    def write(self, directory: Path, name: str = "1tst.cif") -> Path:
        path = directory / name
        path.write_text(self.text(), encoding="utf-8")
        return path


@dataclass(frozen=True)
class Outcome:
    computation: DisulfideComputation
    atoms: pd.DataFrame

    @property
    def table(self) -> pd.DataFrame:
        return self.computation.table

    def rows(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = [
            {str(key): value for key, value in row.items()}
            for row in self.table.to_dict("records")
        ]
        return records

    def only_row(self) -> dict[str, object]:
        rows = self.rows()
        assert len(rows) == 1, self.table
        return rows[0]

    def linked(self) -> dict[str, str]:
        """Return each linked atom's link value by its 'chain+res:atom' label."""
        linked: dict[str, str] = {}
        link: list[object] = self.computation.link.reindex(self.atoms.index).tolist()
        chains: list[object] = self.atoms["chain_id"].tolist()
        residues: list[object] = self.atoms["res_num"].tolist()
        names: list[object] = self.atoms["atom_name"].tolist()
        for value, chain, residue, name in zip(
            link, chains, residues, names, strict=True
        ):
            if value is pd.NA or value is None:
                continue
            linked[f"{chain}{residue}:{name}"] = str(value)
        return linked


def compute(
    tmp_path: Path,
    structure: Structure,
    *,
    pdb_id: str = "1tst",
    file_name: str = "1tst.cif",
) -> Outcome:
    path = structure.write(tmp_path, file_name)
    atoms = parse_mmcif_atom_rows(path, {}, structure_id=pdb_id).dataframe
    flat = read_mmcif_as_flat_dict(path)
    return Outcome(compute_structure_disulfides(atoms, flat, pdb_id=pdb_id), atoms)


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _intact_pair() -> list[Atom]:
    return [*cys("101", 0.0), *cys("140", 2.04)]


# Section 8: no CYS, or CYS without SG.


def test_no_cysteine_sg_gives_no_rows_na_link_and_ok(tmp_path: Path) -> None:
    structure = Structure(
        [
            Atom("101", "CA", "0", comp="GLY"),
            Atom("140", "CB", "2.04"),
        ]
    )
    outcome = compute(tmp_path, structure)
    assert outcome.table.empty
    assert outcome.computation.link.isna().all()
    assert outcome.computation.status == "ok"
    assert outcome.computation.warnings == ()


# Section 8: no _struct_conn, no conn_type_id, or conn_type_id "?".


@pytest.mark.parametrize("variant", ["no_category", "no_type_item", "type_unknown"])
def test_records_without_a_type_leave_distance_detection_only(
    tmp_path: Path, variant: str
) -> None:
    records = [disulf("disulf1", "101", "140", conn_type="?")]
    omit: frozenset[str] = frozenset()
    if variant == "no_category":
        records = []
    elif variant == "no_type_item":
        omit = frozenset({"conn_type_id"})
    outcome = compute(tmp_path, Structure(_intact_pair(), records, omit=omit))
    row = outcome.only_row()
    assert row["detection_method"] == "distance"
    assert outcome.computation.status == "ok"
    assert outcome.computation.unmatched_record_count == 0
    assert outcome.computation.symmetry_record_count == 0


def test_upper_case_disulf_type_is_matched(tmp_path: Path) -> None:
    structure = Structure(
        _intact_pair(), [disulf("disulf1", "101", "140", conn_type="DISULF")]
    )
    row = compute(tmp_path, structure).only_row()
    assert row["detection_method"] == "struct_conn_and_distance"


def test_intact_annotated_bond(tmp_path: Path) -> None:
    structure = Structure(_intact_pair(), [disulf("disulf1", "101", "140")])
    outcome = compute(tmp_path, structure)
    row = outcome.only_row()
    assert row["detection_method"] == "struct_conn_and_distance"
    assert row["ss_distance_min_A"] == pytest.approx(2.04, abs=1e-9)
    assert row["ss_distance_max_A"] == pytest.approx(2.04, abs=1e-9)
    assert row["pair_status"] == "ok"
    assert row["site_id"] == "1TST:1:A:101::SG:"
    assert row["partner_site_id"] == "1TST:1:A:140::SG:"
    assert row["sg_cb_b_factor_ratio_min"] == 1.0
    assert outcome.computation.status == "ok"


# Section 8: Cys{A 0.6, B 0.4} + Cys{A, B}; per-altloc, reversed, duplicates.


def _partially_broken_pair() -> list[Atom]:
    # A–A 2.05, B–B 3.60, A–B (105A to 130B) 2.50.
    return [
        Atom("105", "CB", "0", "0", "-1.800"),
        Atom("105", "SG", "0", "0", "0", alt="A", occ="0.60"),
        Atom("105", "SG", "-1.100", "0", "0", alt="B", occ="0.40"),
        Atom("130", "CB", "2.050", "0", "-1.800"),
        Atom("130", "SG", "2.050", "0", "0", alt="A", occ="0.60"),
        Atom("130", "SG", "2.500", "0", "0", alt="B", occ="0.40"),
    ]


def test_partially_broken_pair_summarizes_compatible_conformers(
    tmp_path: Path,
) -> None:
    structure = Structure(_partially_broken_pair())
    row = compute(tmp_path, structure).only_row()
    assert row["ss_distance_min_A"] == pytest.approx(2.05, abs=1e-9)
    assert row["ss_distance_max_A"] == pytest.approx(3.6, abs=1e-9)
    assert row["sg_has_alternate_conformers"] is True
    assert row["sg_conformer_occupancy_min"] == pytest.approx(0.4, abs=1e-12)
    assert row["sg_total_occupancy_min"] == 1.0
    assert row["altloc"] == "A"
    assert row["partner_altloc"] == "A"
    assert row["site_id"] == "1TST:1:A:105::SG:A"
    assert row["pair_status"] == "ok"


def test_per_altloc_reversed_and_duplicate_records_give_one_row(
    tmp_path: Path,
) -> None:
    records = [
        disulf("disulf2", "105", "130"),
        disulf("disulf3", "130", "105"),
        disulf("disulf2", "105", "130"),
    ]
    outcome = compute(tmp_path, Structure(_partially_broken_pair(), records))
    row = outcome.only_row()
    assert row["res_num"] == "105"
    assert row["partner_res_num"] == "130"
    assert row["detection_method"] == "struct_conn_and_distance"
    assert outcome.computation.warnings == ()
    assert outcome.computation.status == "ok"


# Section 8: distance thresholds.


def test_fully_broken_pair_without_record_is_not_found(tmp_path: Path) -> None:
    outcome = compute(tmp_path, Structure([*cys("101", 0.0), *cys("140", 3.5)]))
    assert outcome.table.empty


def test_unannotated_lengthened_pair_is_found_by_distance(tmp_path: Path) -> None:
    outcome = compute(tmp_path, Structure([*cys("101", 0.0), *cys("140", 2.7)]))
    row = outcome.only_row()
    assert row["detection_method"] == "distance"
    assert row["ss_distance_min_A"] == pytest.approx(2.7, abs=1e-9)


def test_annotated_but_cleaved_pair_stays_struct_conn(tmp_path: Path) -> None:
    structure = Structure(
        [*cys("176", 0.0, occ="0.70"), *cys("194", 3.6)],
        [disulf("disulf8", "176", "194")],
    )
    row = compute(tmp_path, structure).only_row()
    assert row["detection_method"] == "struct_conn"
    assert row["ss_distance_min_A"] == pytest.approx(3.6, abs=1e-9)
    assert row["sg_total_occupancy_min"] == pytest.approx(0.7, abs=1e-12)
    assert row["sg_has_alternate_conformers"] is False


@pytest.mark.parametrize(
    ("separation", "found"),
    [(0.5, True), (1.499, True), (1.5, True), (3.0, True), (3.001, False)],
)
def test_search_includes_its_limit_and_has_no_lower_limit(
    tmp_path: Path, separation: float, found: bool
) -> None:
    outcome = compute(tmp_path, Structure([*cys("101", 0.0), *cys("140", separation)]))
    assert len(outcome.table) == (1 if found else 0)
    if found:
        row = outcome.only_row()
        assert row["detection_method"] == "distance"
        assert row["ss_distance_min_A"] == pytest.approx(separation, abs=1e-9)
        assert row["pair_status"] == "ok"


def test_annotated_pair_below_bonding_distance_is_found_by_both(
    tmp_path: Path,
) -> None:
    # With a lower limit of 1.5 Å such a row said struct_conn alone, as if the
    # distance search had not seen the pair.
    structure = Structure(
        [*cys("101", 0.0), *cys("140", 1.16)], [disulf("disulf1", "101", "140")]
    )
    row = compute(tmp_path, structure).only_row()
    assert row["detection_method"] == "struct_conn_and_distance"
    assert row["ss_distance_min_A"] == pytest.approx(1.16, abs=1e-9)
    assert row["ss_distance_max_A"] == pytest.approx(1.16, abs=1e-9)


def _zinc(x: float, y: float, z: float) -> Atom:
    return Atom(
        "401",
        "ZN",
        _number(x),
        _number(y),
        _number(z),
        comp="ZN",
        group="HETATM",
        entity="2",
        label_seq=".",
        element="ZN",
    )


def test_zinc_cys4_site_gives_no_rows(tmp_path: Path) -> None:
    # Tetrahedral corners at distance 2.33 from the zinc; SG–SG = 2.33 * sqrt(8/3).
    a = 2.33 / math.sqrt(3)
    corners = [(a, a, a), (a, -a, -a), (-a, a, -a), (-a, -a, a)]
    atoms: list[Atom] = [_zinc(0, 0, 0)]
    for res, (x, y, z) in zip(("170", "173", "186", "189"), corners, strict=True):
        atoms.append(Atom(res, "SG", _number(x), _number(y), _number(z)))
    outcome = compute(tmp_path, Structure(atoms))
    assert outcome.table.empty


def test_close_pair_around_a_zinc_is_kept_and_flagged(tmp_path: Path) -> None:
    # The zinc is sqrt(1.45² + 1²) = 1.761 Å from each SG.
    atoms = [_zinc(1.45, 1.0, 0), *cys("101", 0.0), *cys("140", 2.9)]
    row = compute(tmp_path, Structure(atoms)).only_row()
    assert row["detection_method"] == "distance"
    assert row["ss_distance_min_A"] == pytest.approx(2.9, abs=1e-9)
    assert row["metal_bridged"] is True
    assert row["pair_status"] == "ok"


def test_pair_with_one_sg_near_a_zinc_is_not_flagged(tmp_path: Path) -> None:
    # The zinc is 2.33 Å from SG 101 and 5.23 Å from SG 140.
    atoms = [_zinc(-2.33, 0, 0), *cys("101", 0.0), *cys("140", 2.9)]
    row = compute(tmp_path, Structure(atoms)).only_row()
    assert row["ss_distance_min_A"] == pytest.approx(2.9, abs=1e-9)
    assert row["metal_bridged"] is False


def test_metal_flag_changes_no_other_value_and_no_row(tmp_path: Path) -> None:
    pairs = [*cys("101", 0.0), *cys("140", 2.9), *cys("176", 20.0), *cys("194", 22.04)]
    with_zinc = compute(tmp_path, Structure([_zinc(1.45, 1.0, 0), *pairs]))
    without_zinc = compute(tmp_path, Structure(pairs))
    assert with_zinc.table["metal_bridged"].tolist() == [True, False]
    assert without_zinc.table["metal_bridged"].tolist() == [False, False]
    pd.testing.assert_frame_equal(
        with_zinc.table.drop(columns="metal_bridged"),
        without_zinc.table.drop(columns="metal_bridged"),
    )
    assert with_zinc.linked() == without_zinc.linked()
    assert with_zinc.computation.status == without_zinc.computation.status == "ok"


# Section 8: unmatched records degrade; symmetry and out-of-scope do not.


@pytest.mark.parametrize(
    ("atoms", "record"),
    [
        (_intact_pair(), disulf("disulf5", "101", "199")),
        (
            [*_intact_pair(), Atom("195", "CB", "0", "20", "0")],
            disulf("disulf4", "195", "140"),
        ),
        (_intact_pair(), disulf("disulf9", "101", "101")),
    ],
    ids=["absent_residue", "residue_without_sg", "self_pair"],
)
def test_unmatched_in_scope_record_degrades(
    tmp_path: Path, atoms: list[Atom], record: Record
) -> None:
    outcome = compute(tmp_path, Structure(atoms, [record]))
    computation = outcome.computation
    assert computation.status == "degraded"
    assert computation.unmatched_record_count == 1
    assert computation.symmetry_record_count == 0
    assert len(computation.warnings) == 1
    assert record.record_id in computation.warnings[0]
    # The distance pair 101–140 is still found.
    assert outcome.only_row()["detection_method"] == "distance"


def test_symmetry_mate_record_is_counted_only(tmp_path: Path) -> None:
    record = Record("disulf6", Partner("180"), Partner("180", symmetry="2_655"))
    outcome = compute(tmp_path, Structure(cys("180", 0.0), [record]))
    computation = outcome.computation
    assert outcome.table.empty
    assert computation.symmetry_record_count == 1
    assert computation.unmatched_record_count == 0
    assert computation.status == "ok"
    assert computation.warnings == ()


@pytest.mark.parametrize(
    ("first", "second"),
    [("2_555", "2_555"), ("?", "?"), ("1_555", "?"), ("1555", "1_555")],
)
def test_equal_or_missing_operators_are_within_one_molecule(
    tmp_path: Path, first: str, second: str
) -> None:
    record = Record(
        "disulf1", Partner("101", symmetry=first), Partner("140", symmetry=second)
    )
    # 3.6 Å apart, so only the record can find the pair.
    outcome = compute(
        tmp_path, Structure([*cys("101", 0.0), *cys("140", 3.6)], [record])
    )
    row = outcome.only_row()
    assert row["detection_method"] == "struct_conn"
    assert outcome.computation.symmetry_record_count == 0
    assert outcome.computation.status == "ok"


@pytest.mark.parametrize(
    "second",
    [
        Partner("301", comp="BME", atom="S2", label_seq="."),
        Partner("301", comp="BME", atom="S2", label_seq=".", symmetry="2_655"),
        Partner("140", atom="CB"),
        Partner("199", comp="CSO"),
    ],
    ids=["bme", "bme_symmetry_mate", "non_sg_atom", "oxidized_cys"],
)
def test_out_of_scope_record_is_skipped_and_not_counted(
    tmp_path: Path, second: Partner
) -> None:
    atoms = [
        *cys("101", 0.0),
        *cys("140", 20.0),
        Atom(
            "301",
            "S2",
            "0",
            "2.040",
            "0",
            comp="BME",
            group="HETATM",
            entity="2",
            label_seq=".",
            element="S",
        ),
    ]
    record = Record("disulf1", Partner("101"), second)
    outcome = compute(tmp_path, Structure(atoms, [record]))
    computation = outcome.computation
    assert outcome.table.empty
    assert computation.unmatched_record_count == 0
    assert computation.symmetry_record_count == 0
    assert computation.status == "ok"
    assert computation.warnings == ()


# Section 8: the label route.


def test_record_without_author_fields_resolves_by_label(tmp_path: Path) -> None:
    record = Record(
        "disulf1", Partner("101", author=False), Partner("140", author=False)
    )
    structure = Structure([*cys("101", 0.0), *cys("140", 3.6)], [record])
    outcome = compute(tmp_path, structure)
    row = outcome.only_row()
    assert row["detection_method"] == "struct_conn"
    assert row["res_num"] == "101"
    assert outcome.computation.status == "ok"


def test_author_miss_is_not_retried_by_label(tmp_path: Path) -> None:
    # Author numbers point at absent residues; label numbers (1, 40) would match.
    record = Record(
        "disulf1",
        Partner("901", label_seq="1"),
        Partner("940", label_seq="40"),
    )
    structure = Structure([*cys("101", 0.0), *cys("140", 3.6)], [record])
    outcome = compute(tmp_path, structure)
    assert outcome.table.empty
    assert outcome.computation.unmatched_record_count == 1
    assert outcome.computation.status == "degraded"
    assert "disulf1" in outcome.computation.warnings[0]


def test_cysteines_without_author_component_are_found_by_label_component(
    tmp_path: Path,
) -> None:
    structure = Structure(_intact_pair(), omit=frozenset({"auth_comp_id"}))
    outcome = compute(tmp_path, structure)
    row = outcome.only_row()
    assert row["detection_method"] == "distance"
    assert row["ss_distance_min_A"] == pytest.approx(2.04, abs=1e-9)
    assert row["site_id"] == "1TST:1:A:101::SG:"
    # The atom table holds the label residue name, and the row holds the same.
    assert set(outcome.atoms["res_name"]) == {"CYS"}
    assert row["res_name"] == "CYS"
    assert row["partner_res_name"] == "CYS"
    assert outcome.linked() == {
        "A101:SG": "1TST:1:A:101::SG:",
        "A140:SG": "1TST:1:A:140::SG:",
    }
    assert outcome.computation.status == "ok"
    assert outcome.computation.warnings == ()


def test_cysteines_without_author_numbers_are_named_by_label_numbers(
    tmp_path: Path,
) -> None:
    structure = Structure(
        _intact_pair(), omit=frozenset({"auth_comp_id", "auth_seq_id"})
    )
    outcome = compute(tmp_path, structure)
    row = outcome.only_row()
    # The file is numbered by its label fields, which are 1 and 40 here.
    assert set(outcome.atoms["residue_numbering"]) == {"label"}
    assert row["detection_method"] == "distance"
    assert row["site_id"] == "1TST:1:A:1::SG:"
    assert row["partner_site_id"] == "1TST:1:A:40::SG:"
    assert row["res_name"] == "CYS"
    assert outcome.linked() == {
        "A1:SG": "1TST:1:A:1::SG:",
        "A40:SG": "1TST:1:A:40::SG:",
    }
    assert outcome.computation.status == "ok"


def test_record_with_author_fields_resolves_by_label_under_label_numbering(
    tmp_path: Path,
) -> None:
    # The record names author residues 101 and 140, which no residue key of a
    # label-numbered file holds, and label residues 1 and 40, which do match.
    structure = Structure(
        [*cys("101", 0.0), *cys("140", 3.6)],
        [disulf("disulf1", "101", "140")],
        omit=frozenset({"auth_seq_id"}),
    )
    outcome = compute(tmp_path, structure)
    row = outcome.only_row()
    assert row["detection_method"] == "struct_conn"
    assert row["site_id"] == "1TST:1:A:1::SG:"
    assert row["partner_site_id"] == "1TST:1:A:40::SG:"
    assert outcome.computation.unmatched_record_count == 0
    assert outcome.computation.status == "ok"
    assert outcome.computation.warnings == ()


# Section 8: multi-model.


def test_record_resolving_in_some_models_only_gives_no_warning(
    tmp_path: Path,
) -> None:
    atoms = [*cys("101", 0.0), *cys("140", 3.6), *cys("101", 0.0, model="2")]
    structure = Structure(atoms, [disulf("disulf1", "101", "140")])
    outcome = compute(tmp_path, structure)
    row = outcome.only_row()
    assert row["model_num"] == 1
    assert row["detection_method"] == "struct_conn"
    assert outcome.computation.warnings == ()
    assert outcome.computation.unmatched_record_count == 0
    assert outcome.computation.status == "ok"


def test_rows_per_model_and_search_never_crosses_models(tmp_path: Path) -> None:
    # Model 1 at 2.04; model 2 at 3.6. Model 1's 140 lies 2.04 from model 2's 101.
    atoms = [
        *cys("101", 0.0),
        *cys("140", 2.04),
        *cys("101", 0.0, model="2"),
        *cys("140", 3.6, model="2"),
    ]
    outcome = compute(tmp_path, Structure(atoms))
    row = outcome.only_row()
    assert row["model_num"] == 1
    assert row["partner_model_num"] == 1

    with_record = compute(tmp_path, Structure(atoms, [disulf("disulf1", "101", "140")]))
    rows = with_record.rows()
    assert [(r["model_num"], r["detection_method"]) for r in rows] == [
        (1, "struct_conn_and_distance"),
        (2, "struct_conn"),
    ]


# Section 8: conformers that cannot coexist.


def test_a_only_against_b_only_gives_no_compatible_conformers(
    tmp_path: Path,
) -> None:
    atoms = [
        *cys("101", 0.0, alt="A", cb_alt="A"),
        *cys("140", 2.04, alt="B", cb_alt="B"),
    ]
    without_record = compute(tmp_path, Structure(atoms))
    assert without_record.table.empty

    outcome = compute(tmp_path, Structure(atoms, [disulf("disulf1", "101", "140")]))
    row = outcome.only_row()
    assert row["detection_method"] == "struct_conn"
    assert _is_nan(row["ss_distance_min_A"])
    assert _is_nan(row["ss_distance_max_A"])
    assert row["pair_status"] == "no_compatible_conformers"
    assert outcome.computation.status == "ok"


# Section 8: zero occupancy and unknown coordinates.


def test_zero_occupancy_sg_is_not_searched(tmp_path: Path) -> None:
    atoms = [*cys("101", 0.0), *cys("140", 2.04, occ="0.00")]
    assert compute(tmp_path, Structure(atoms)).table.empty


def test_zero_occupancy_sg_with_record_is_no_usable_sg(tmp_path: Path) -> None:
    atoms = [*cys("101", 0.0), *cys("140", 2.04, occ="0.00")]
    outcome = compute(tmp_path, Structure(atoms, [disulf("disulf1", "101", "140")]))
    row = outcome.only_row()
    assert row["detection_method"] == "struct_conn"
    assert _is_nan(row["ss_distance_min_A"])
    assert _is_nan(row["ss_distance_max_A"])
    assert row["pair_status"] == "no_usable_sg"
    assert row["sg_conformer_occupancy_min"] == 0.0
    assert row["sg_total_occupancy_min"] == 0.0
    assert outcome.computation.status == "ok"


def test_unknown_coordinates_do_not_crash(tmp_path: Path) -> None:
    atoms = [
        *cys("101", 0.0),
        Atom("140", "CB", "2.040", "0", "-1.800"),
        Atom("140", "SG", "?", "?", "?"),
    ]
    assert compute(tmp_path, Structure(atoms)).table.empty

    outcome = compute(tmp_path, Structure(atoms, [disulf("disulf1", "101", "140")]))
    row = outcome.only_row()
    assert row["pair_status"] == "no_usable_sg"
    assert _is_nan(row["ss_distance_min_A"])


def test_unknown_coordinates_on_a_dropped_conformer_leave_the_other(
    tmp_path: Path,
) -> None:
    atoms = [
        *cys("101", 0.0),
        Atom("140", "CB", "2.040", "0", "-1.800"),
        Atom("140", "SG", "2.040", "0", "0", alt="A", occ="0.60"),
        Atom("140", "SG", "?", "?", "?", alt="B", occ="0.40"),
    ]
    row = compute(tmp_path, Structure(atoms)).only_row()
    assert row["ss_distance_min_A"] == pytest.approx(2.04, abs=1e-9)
    assert row["ss_distance_max_A"] == pytest.approx(2.04, abs=1e-9)
    assert row["sg_conformer_occupancy_min"] == pytest.approx(0.4, abs=1e-12)
    assert row["pair_status"] == "ok"


def test_unknown_occupancy_is_searched_but_reported_missing(
    tmp_path: Path,
) -> None:
    atoms = [*cys("101", 0.0), *cys("140", 2.04)]
    atoms[3] = Atom("140", "SG", "2.040", "0", "0", occ="?")
    row = compute(tmp_path, Structure(atoms)).only_row()
    assert row["detection_method"] == "distance"
    assert row["ss_distance_min_A"] == pytest.approx(2.04, abs=1e-9)
    assert _is_nan(row["sg_conformer_occupancy_min"])
    assert _is_nan(row["sg_total_occupancy_min"])
    assert row["pair_status"] == "occupancy_missing"


def test_occupancies_summing_above_one_are_reported_as_is(tmp_path: Path) -> None:
    atoms = [
        Atom("101", "CB", "0", "0", "-1.800"),
        Atom("101", "SG", "0", "0", "0", alt="A", occ="0.70"),
        Atom("101", "SG", "0", "1.000", "0", alt="B", occ="0.60"),
        Atom("140", "CB", "2.040", "0", "-1.800"),
        Atom("140", "SG", "2.040", "0", "0", alt="A", occ="0.70"),
        Atom("140", "SG", "2.040", "1.000", "0", alt="B", occ="0.60"),
    ]
    row = compute(tmp_path, Structure(atoms)).only_row()
    assert row["sg_total_occupancy_min"] == 1.3
    assert row["sg_conformer_occupancy_min"] == pytest.approx(0.6, abs=1e-12)


def test_three_conformers_of_a_third_sum_to_exactly_one(tmp_path: Path) -> None:
    atoms = [
        Atom("101", "CB", "0", "0", "-1.800"),
        Atom("101", "SG", "0", "0", "0", alt="A", occ="0.34"),
        Atom("101", "SG", "0", "0.500", "0", alt="B", occ="0.33"),
        Atom("101", "SG", "0", "-0.500", "0", alt="C", occ="0.33"),
        *cys("140", 2.04),
    ]
    row = compute(tmp_path, Structure(atoms)).only_row()
    assert row["sg_total_occupancy_min"] == 1.0


# Section 8: B-factors.


@pytest.mark.parametrize("variant", ["absent", "unknown", "zero"])
def test_unavailable_b_factors_give_nan_ratios(tmp_path: Path, variant: str) -> None:
    omit: frozenset[str] = frozenset()
    b = "20"
    if variant == "absent":
        omit = frozenset({"B_iso_or_equiv"})
    elif variant == "unknown":
        b = "?"
    else:
        b = "0.00"
    atoms = [*cys("101", 0.0, b=b, cb_b=b), *cys("140", 2.04, b=b, cb_b=b)]
    outcome = compute(tmp_path, Structure(atoms, omit=omit))
    row = outcome.only_row()
    assert _is_nan(row["sg_cb_b_factor_ratio_min"])
    assert _is_nan(row["sg_cb_b_factor_ratio_max"])
    assert row["pair_status"] == "b_factor_unavailable"
    assert outcome.computation.status == "ok"
    assert outcome.computation.warnings == ()


def test_cb_b_factor_zero_on_one_side_makes_both_ratios_nan(tmp_path: Path) -> None:
    atoms = [*cys("101", 0.0, b="30", cb_b="20"), *cys("140", 2.04, cb_b="0.00")]
    row = compute(tmp_path, Structure(atoms)).only_row()
    assert _is_nan(row["sg_cb_b_factor_ratio_min"])
    assert _is_nan(row["sg_cb_b_factor_ratio_max"])
    assert row["pair_status"] == "b_factor_unavailable"


def test_constant_b_gives_ratio_exactly_one(tmp_path: Path) -> None:
    atoms = [
        *cys("101", 0.0, b="30", cb_b="30"),
        *cys("140", 2.04, b="30", cb_b="30"),
    ]
    row = compute(tmp_path, Structure(atoms)).only_row()
    assert row["sg_cb_b_factor_ratio_min"] == 1.0
    assert row["sg_cb_b_factor_ratio_max"] == 1.0
    assert row["pair_status"] == "ok"


def test_ratios_are_the_smaller_and_larger_residue_ratio(tmp_path: Path) -> None:
    atoms = [
        *cys("101", 0.0, b="30", cb_b="20"),
        *cys("140", 2.04, b="25", cb_b="20"),
    ]
    row = compute(tmp_path, Structure(atoms)).only_row()
    assert row["sg_cb_b_factor_ratio_min"] == 1.25
    assert row["sg_cb_b_factor_ratio_max"] == 1.5


def test_missing_representative_cb_gives_cb_missing(tmp_path: Path) -> None:
    atoms = [*cys("101", 0.0), Atom("140", "SG", "2.040")]
    row = compute(tmp_path, Structure(atoms)).only_row()
    assert row["pair_status"] == "cb_missing"
    assert _is_nan(row["sg_cb_b_factor_ratio_min"])


# Section 8: identifier columns.


def test_absent_model_number_leaves_model_part_empty(tmp_path: Path) -> None:
    outcome = compute(
        tmp_path,
        Structure(_intact_pair(), omit=frozenset({"pdbx_PDB_model_num"})),
    )
    row = outcome.only_row()
    assert str(outcome.table["model_num"].dtype) == "Int64"
    assert outcome.table["model_num"].isna().all()
    assert outcome.table["partner_model_num"].isna().all()
    assert row["site_id"] == "1TST::A:101::SG:"
    assert row["partner_site_id"] == "1TST::A:140::SG:"
    validated = validate_tables(
        "disulfides", (DISULFIDES_TABLE,), {"disulfides": outcome.table}
    )
    assert len(validated["disulfides"]) == 1


def test_absent_alt_id_makes_everything_compatible(tmp_path: Path) -> None:
    outcome = compute(
        tmp_path, Structure(_intact_pair(), omit=frozenset({"label_alt_id"}))
    )
    row = outcome.only_row()
    assert row["sg_has_alternate_conformers"] is False
    assert row["altloc"] == ""
    assert row["partner_altloc"] == ""
    assert row["site_id"] == "1TST:1:A:101::SG:"


def test_multi_character_altlocs_are_compared_as_whole_strings(
    tmp_path: Path,
) -> None:
    atoms = [
        Atom("105", "CB", "0", "0", "-1.800"),
        Atom("105", "SG", "0", "0", "0", alt="AA", occ="0.60"),
        Atom("105", "SG", "-1.100", "0", "0", alt="BB", occ="0.40"),
        Atom("130", "CB", "2.050", "0", "-1.800"),
        Atom("130", "SG", "2.050", "0", "0", alt="AA", occ="0.60"),
        Atom("130", "SG", "2.500", "0", "0", alt="BB", occ="0.40"),
    ]
    row = compute(tmp_path, Structure(atoms)).only_row()
    assert row["altloc"] == "AA"
    assert row["site_id"] == "1TST:1:A:105::SG:AA"
    assert row["ss_distance_min_A"] == pytest.approx(2.05, abs=1e-9)
    assert row["ss_distance_max_A"] == pytest.approx(3.6, abs=1e-9)
    assert row["sg_has_alternate_conformers"] is True


def test_single_and_double_letter_altlocs_are_incompatible(tmp_path: Path) -> None:
    atoms = [
        *cys("101", 0.0, alt="A", cb_alt="A"),
        *cys("140", 2.04, alt="AA", cb_alt="AA"),
    ]
    outcome = compute(tmp_path, Structure(atoms, [disulf("disulf1", "101", "140")]))
    row = outcome.only_row()
    assert row["pair_status"] == "no_compatible_conformers"


def test_sg_altlocs_with_unlabelled_cb_use_the_unlabelled_cb(tmp_path: Path) -> None:
    atoms = [
        Atom("101", "CB", "0", "0", "-1.800", b="20"),
        Atom("101", "SG", "0", "0", "0", alt="A", occ="0.60", b="30"),
        Atom("101", "SG", "0", "1.000", "0", alt="B", occ="0.40", b="40"),
        *cys("140", 2.04, b="20", cb_b="20"),
    ]
    row = compute(tmp_path, Structure(atoms)).only_row()
    assert row["sg_cb_b_factor_ratio_max"] == 1.5
    assert row["sg_cb_b_factor_ratio_min"] == 1.0
    assert row["pair_status"] == "ok"


def test_lone_labelled_sg_has_no_alternates(tmp_path: Path) -> None:
    atoms = [*cys("101", 0.0, alt="A", occ="0.60"), *cys("140", 2.04)]
    row = compute(tmp_path, Structure(atoms)).only_row()
    assert row["sg_has_alternate_conformers"] is False
    assert row["sg_total_occupancy_min"] == pytest.approx(0.6, abs=1e-12)


def test_kept_conformer_without_sg_falls_back(tmp_path: Path) -> None:
    atoms = [
        Atom("150", "CB", "0", "0", "-1.800", alt="A", occ="0.45", b="30"),
        Atom("150", "SG", "0", "0", "0", alt="A", occ="0.45", b="45"),
        Atom("150", "CB", "0", "0.500", "-1.800", alt="B", occ="0.55"),
        *cys("155", 2.05),
    ]
    outcome = compute(tmp_path, Structure(atoms))
    row = outcome.only_row()
    assert row["detection_method"] == "distance"
    assert row["pair_status"] == "sg_not_in_atom_table"
    assert row["altloc"] == "A"
    assert row["site_id"] == "1TST:1:A:150::SG:A"
    assert row["sg_cb_b_factor_ratio_max"] == 1.5
    assert outcome.linked() == {"A155:SG": "1TST:1:A:155::SG:"}
    assert outcome.computation.status == "ok"
    assert len(outcome.computation.warnings) == 1


def test_duplicate_author_key_raises_value_error(tmp_path: Path) -> None:
    # Two residues share chain A, number 101, and no insertion code, but differ
    # in their label numbers.
    atoms = [
        *cys("101", 0.0),
        Atom("101", "SG", "20.000", label_seq="5"),
        Atom("101", "CB", "20.000", "0", "-1.800", label_seq="5"),
    ]
    path = Structure(atoms).write(tmp_path)
    parsed = parse_mmcif_atom_rows(path, {}, structure_id="1tst").dataframe
    with pytest.raises(ValueError, match=r"1TST:1:A:101::(SG|CB):"):
        compute_structure_disulfides(
            parsed, read_mmcif_as_flat_dict(path), pdb_id="1tst"
        )


def test_atom_table_of_another_file_raises_runtime_error(tmp_path: Path) -> None:
    first = Structure(_intact_pair()).write(tmp_path, "a.cif")
    second = Structure([*cys("101", 0.0), *cys("141", 2.04)]).write(tmp_path, "b.cif")
    atoms = parse_mmcif_atom_rows(first, {}, structure_id="1tst").dataframe
    with pytest.raises(RuntimeError):
        compute_structure_disulfides(
            atoms, read_mmcif_as_flat_dict(second), pdb_id="1tst"
        )


def test_overlapping_chain_copies_are_paired_and_reported_as_measured(
    tmp_path: Path,
) -> None:
    # A residue and its copy in another chain, 1.0 Å apart. The search has no
    # lower limit, so the pair is a row and its distance shows what it is.
    atoms = [*cys("101", 0.0, chain="A"), *cys("101", 1.0, chain="B")]
    row = compute(tmp_path, Structure(atoms)).only_row()
    assert row["detection_method"] == "distance"
    assert (row["chain_id"], row["partner_chain_id"]) == ("A", "B")
    assert (row["res_num"], row["partner_res_num"]) == ("101", "101")
    assert row["ss_distance_min_A"] == pytest.approx(1.0, abs=1e-9)


def test_sg_near_two_partners_gives_two_rows_and_own_links(tmp_path: Path) -> None:
    atoms = [*cys("101", -2.04), *cys("120", 0.0), *cys("140", 2.5)]
    outcome = compute(tmp_path, Structure(atoms))
    rows = outcome.rows()
    assert [(r["res_num"], r["partner_res_num"]) for r in rows] == [
        ("101", "120"),
        ("120", "140"),
    ]
    assert outcome.computation.link.index.equals(outcome.atoms.index)
    assert outcome.linked() == {
        "A101:SG": "1TST:1:A:101::SG:",
        "A120:SG": "1TST:1:A:120::SG:",
        "A140:SG": "1TST:1:A:140::SG:",
    }


def test_hydrogen_and_deuterium_atoms_are_never_read(tmp_path: Path) -> None:
    atoms = [
        *cys("101", 0.0),
        Atom("101", "HG", "1.000", "0", "0", alt="A", occ="0.60", element="H"),
        Atom("101", "HG", "1.000", "0.300", "0", alt="B", occ="0.40", element="H"),
        Atom("101", "HB2", "0", "0.900", "-1.800", element="H"),
        *cys("140", 2.04),
        Atom("140", "DG", "1.100", "0", "0", alt="A", occ="0.50", element="D"),
        Atom("140", "DG", "1.100", "0.300", "0", alt="B", occ="0.50", element="D"),
    ]
    outcome = compute(tmp_path, Structure(atoms, [disulf("disulf1", "101", "140")]))
    row = outcome.only_row()
    assert row["detection_method"] == "struct_conn_and_distance"
    assert row["sg_has_alternate_conformers"] is False
    assert row["sg_conformer_occupancy_min"] == 1.0
    assert row["sg_total_occupancy_min"] == 1.0
    assert row["ss_distance_min_A"] == pytest.approx(2.04, abs=1e-9)
    assert row["pair_status"] == "ok"
    assert set(outcome.linked()) == {"A101:SG", "A140:SG"}


def test_pdb_id_comes_from_the_host_not_the_data_block(tmp_path: Path) -> None:
    structure = Structure(_intact_pair(), block="XXXX")
    outcome = compute(tmp_path, structure, pdb_id="1dsf", file_name="upload.cif")
    row = outcome.only_row()
    assert row["pdb_id"] == "1DSF"
    assert row["partner_pdb_id"] == "1DSF"
    assert row["site_id"] == "1DSF:1:A:101::SG:"


def test_hetatm_cysteine_ligand_resolves_through_author_fields(
    tmp_path: Path,
) -> None:
    ligand = [
        Atom(
            "301",
            name,
            x,
            "0",
            z,
            chain="B",
            group="HETATM",
            entity="2",
            label_seq=".",
        )
        for name, x, z in (("CB", "2.030", "-1.800"), ("SG", "2.030", "0"))
    ]
    record = Record("disulf7", Partner("125"), Partner("301", chain="B", label_seq="."))
    outcome = compute(tmp_path, Structure([*cys("125", 0.0), *ligand], [record]))
    row = outcome.only_row()
    assert row["detection_method"] == "struct_conn_and_distance"
    assert row["partner_chain_id"] == "B"
    assert row["partner_res_num"] == "301"
    assert outcome.computation.status == "ok"


def test_insertion_codes_are_distinct_residues(tmp_path: Path) -> None:
    atoms = [
        *cys("120", 0.0, 20.0),
        *cys("120", 0.0, ins="A"),
        *cys("160", 2.7),
        *cys("170", 3.6, 20.0),
    ]
    records = [
        Record("disulf1", Partner("120", ins="A"), Partner("160")),
        disulf("disulf2", "120", "170"),
    ]
    outcome = compute(tmp_path, Structure(atoms, records))
    rows = outcome.rows()
    assert [
        (r["site_id"], r["partner_site_id"], r["detection_method"]) for r in rows
    ] == [
        ("1TST:1:A:120::SG:", "1TST:1:A:170::SG:", "struct_conn"),
        ("1TST:1:A:120:A:SG:", "1TST:1:A:160::SG:", "struct_conn_and_distance"),
    ]
    assert outcome.computation.status == "ok"


def test_canonical_order_puts_smaller_residue_first(tmp_path: Path) -> None:
    atoms = [
        *cys("110", 0.0),
        *cys("109", 2.04),
        *cys("152", 0.0, 20.0, ins="A"),
        *cys("152", 2.04, 20.0),
    ]
    outcome = compute(tmp_path, Structure(atoms, [disulf("disulf1", "110", "109")]))
    rows = outcome.rows()
    assert [(r["res_num"], r["ins_code"], r["partner_res_num"]) for r in rows] == [
        ("109", "", "110"),
        ("152", "", "152"),
    ]
    assert rows[1]["partner_ins_code"] == "A"


# The table fits its declaration.


def test_table_passes_validate_tables(tmp_path: Path) -> None:
    atoms = [
        *_partially_broken_pair(),
        *cys("101", 0.0, 20.0),
        *cys("140", 2.7, 20.0, b="0.00"),
        *cys("176", 0.0, 40.0, occ="0.00"),
        *cys("194", 3.6, 40.0),
    ]
    records = [disulf("disulf1", "105", "130"), disulf("disulf8", "176", "194")]
    outcome = compute(tmp_path, Structure(atoms, records))
    table = outcome.table
    assert list(table.columns) == list(DISULFIDES_TABLE.column_names)
    assert len(table.columns) == 29
    assert not table["metal_bridged"].isna().any()
    validated = validate_tables(
        "disulfides", (DISULFIDES_TABLE,), {"disulfides": table}
    )["disulfides"]
    assert len(validated) == 3
    pd.testing.assert_frame_equal(
        validated, table.reset_index(drop=True), check_dtype=True
    )
    assert set(validated["pair_status"]) == {
        "ok",
        "b_factor_unavailable",
        "no_usable_sg",
    }
    assert outcome.computation.status == "ok"
