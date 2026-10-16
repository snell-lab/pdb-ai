"""Black-box tests of disulfide edge cases that test_compute.py does not cover.

The cases are a symmetry record alone, a record in a file numbered by its label
fields, unreadable coordinates on one conformer, occupancies that sum above
one, a B-factor ratio of exactly one, missing model numbers and alternate
location labels, labels of more than one character, hydrogen and deuterium
conformers, and records that tell residues apart by insertion code. Each
expectation follows from the module docstring of compute.py, not from the
implementation.

Each test writes a small mmCIF file, parses it as the host does, and hands the
atom table and the flat dictionary to compute_structure_disulfides. Author
residue numbers are label numbers plus 100, so a test can tell which of the
two a lookup used. Each cysteine's CB sits 1.8 Å below its SG on the z axis,
and pairs lie on the x axis, so every distance is exact.
"""

from __future__ import annotations

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


def _intact_pair() -> list[Atom]:
    return [*cys("101", 0.0), *cys("140", 2.04)]


def test_symmetry_mate_record_is_counted_only(tmp_path: Path) -> None:
    record = Record("disulf6", Partner("180"), Partner("180", symmetry="2_655"))
    outcome = compute(tmp_path, Structure(cys("180", 0.0), [record]))
    computation = outcome.computation
    assert outcome.table.empty
    assert computation.symmetry_record_count == 1
    assert computation.unmatched_record_count == 0
    assert computation.status == "ok"
    assert computation.warnings == ()


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


def test_constant_b_gives_ratio_exactly_one(tmp_path: Path) -> None:
    atoms = [
        *cys("101", 0.0, b="30", cb_b="30"),
        *cys("140", 2.04, b="30", cb_b="30"),
    ]
    row = compute(tmp_path, Structure(atoms)).only_row()
    assert row["sg_cb_b_factor_ratio_min"] == 1.0
    assert row["sg_cb_b_factor_ratio_max"] == 1.0
    assert row["pair_status"] == "ok"


def test_absent_model_number_leaves_model_part_empty(tmp_path: Path) -> None:
    outcome = compute(
        tmp_path,
        Structure(_intact_pair(), omit=frozenset({"pdbx_PDB_model_num"})),
    )
    row = outcome.only_row()
    assert row["site_id"] == "1TST::A:101::SG:"
    assert row["partner_site_id"] == "1TST::A:140::SG:"
    validated = validate_tables(
        "disulfides", (DISULFIDES_TABLE,), {"disulfides": outcome.table}
    )["disulfides"]
    assert len(validated) == 1
    assert str(validated["model_num"].dtype) == "Int64"
    assert validated["model_num"].isna().all()
    assert validated["partner_model_num"].isna().all()


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
