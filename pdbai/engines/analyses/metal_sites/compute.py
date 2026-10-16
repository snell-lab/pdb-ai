"""Finding the donor atoms of each metal atom and measuring each bond.

compute_structure_metal_sites takes the atom table of one structure and the
flat mmCIF dictionary of its file, and returns the three tables, the atom link
column, and the counts the run report records. Nothing here judges a site.

The atom table gives the atoms, one conformer of each residue as the parser
chose it, so a site is measured in the conformer every other column
describes. The file gives what the atom table does not hold: the unit cell
and space group, the deposited _struct_conn metalc records, and the
reflection count and R-free the precision index needs.

Model by model, the usable atoms are searched with gemmi's neighbour search,
which also finds atoms of symmetry mates. An atom is usable when it has
coordinates, is not hydrogen or deuterium, and has an occupancy that is not
zero or below. Around each metal atom the search collects the nitrogen,
oxygen, and sulfur atoms of water and of the standard amino acids within
CANDIDATE_SEARCH_RADIUS_A, leaving out the atoms of the metal's own
component. Two images of one atom closer than SPECIAL_POSITION_CUTOFF_A are
one atom on a special position, and the nearer is kept.

A candidate becomes a bond in two ways. The distance search infers it when
its donor rule allows and it lies within the first sphere, the target
distance for the pair plus FIRST_SPHERE_TOLERANCE_A. The target is the
reference table's own row for the metal and donor, or else the longest row
for the same metal and donor element. A deposited record makes a bond of any
atom it names, whatever its element, residue, or distance, since the
depositor asserted it. A record whose partners take different symmetry
operators is matched to the nearest image of the donor.

Each bond with a reference row of its own gets its deviation from the
reference distance, and its z value when the entry has a precision index.
The index follows equation 7 of Blow (2002) and is computed from the
deposited model and its own refinement statistics.

Strict non-crystallographic symmetry operators that a file declares without
giving the copies are not applied, so contacts between such copies are not
found. Entries of that kind are mostly virus capsids.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import gemmi
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial import cKDTree

from ....structure.cif_numbers import coerce_cif_numbers
from ...core import site_key_dataframe
from ..site_geometry import (
    SPECIAL_POSITION_CUTOFF_A,
    CrystalSymmetry,
    finite_mean,
    number_array,
    on_special_position,
    positive_ratio,
    read_file_crystal_symmetry,
    text_array,
)
from ..struct_conn import (
    IDENTITY_SYMMETRY_OPERATOR,
    PartnerAddress,
    StructConnRecord,
    detection_method,
)
from .columns import (
    DONOR_PREFIX,
    METAL_BONDS_TABLE,
    METAL_PAIRS_TABLE,
    METAL_SITES_TABLE,
    PARTNER_PREFIX,
)
from .constants import (
    BACKBONE_CARBONYL_ATOM,
    BOND_STATUS_NO_DPI,
    BOND_STATUS_NO_REFERENCE,
    BOND_STATUS_OK,
    C_TERMINAL_DONOR_ATOMS,
    CANDIDATE_SEARCH_RADIUS_A,
    DISTANCE_EPSILON_A,
    DONOR_AMINO_ACID,
    DONOR_ELEMENTS,
    DONOR_NUCLEOTIDE,
    DONOR_OTHER,
    DONOR_WATER,
    FIRST_SPHERE_TOLERANCE_A,
    HYDROGEN_ELEMENTS,
    INFERRED_DONOR_ATOMS,
    METAL_ELEMENTS,
    METAL_STATUS_DEGRADED,
    METAL_STATUS_NO_METALS,
    METAL_STATUS_OK,
    N_TERMINAL_DONOR_ATOM,
    NEARBY_METAL_RADIUS_A,
    PARENT_CLUSTER,
    PARENT_HEME,
    PARENT_ION,
    PARENT_OTHER,
    REFERENCE_BACKBONE,
    REFERENCE_C_TERMINUS,
    REFERENCE_ELEMENT_FALLBACK,
    REFERENCE_EXACT,
    REFERENCE_MISSING,
    REFERENCE_N_TERMINUS,
    REFERENCE_WATER,
    RULE_BACKBONE_CARBONYL,
    RULE_C_TERMINUS,
    RULE_N_TERMINUS,
    RULE_NOT_TYPICAL,
    RULE_SIDECHAIN,
    RULE_WATER_OXYGEN,
    SITE_STATUS_NO_DONORS,
    SITE_STATUS_OK,
    STANDARD_AMINO_ACIDS,
)
from .records import read_metal_records
from .reference import cofactor_classes, element_targets, literature_distances

_REFLECTION_COUNT_ITEM = "_refine.ls_number_reflns_obs"
_R_FREE_ITEM = "_refine.ls_R_factor_R_free"


@dataclass(frozen=True)
class MetalComputation:
    """The metal-site tables of one structure, and what the run report records."""

    sites: pd.DataFrame
    bonds: pd.DataFrame
    # Every pair of metal atoms of one model within NEARBY_METAL_RADIUS_A.
    pairs: pd.DataFrame
    # The metal atoms' own site_id on their rows of the atom table, on its index.
    link: pd.Series
    status: str
    # Deposited metalc records that named a metal atom of the table and found
    # no donor atom in it.
    unresolved_record_count: int
    dpi: float
    warnings: tuple[str, ...]


@dataclass
class _Atoms:
    """The atom table's columns as arrays, a value per row in row order."""

    element: npt.NDArray[np.object_]
    res_name: npt.NDArray[np.object_]
    atom_name: npt.NDArray[np.object_]
    residue_key: npt.NDArray[np.object_]
    model: npt.NDArray[np.object_]
    altloc: npt.NDArray[np.object_]
    label_seq: npt.NDArray[np.object_]
    label_asym: npt.NDArray[np.object_]
    auth_asym: npt.NDArray[np.object_]
    auth_seq: npt.NDArray[np.object_]
    ins_code: npt.NDArray[np.object_]
    xyz: npt.NDArray[np.float64]
    occupancy: npt.NDArray[np.float64]
    b_factor: npt.NDArray[np.float64]
    usable: npt.NDArray[np.bool_]


@dataclass
class _Contact:
    """One donor atom in one image around a metal atom."""

    row: int
    operator: str
    distance: float
    position: tuple[float, float, float] | None
    declared: bool = False
    inferred: bool = False
    rule: str = RULE_NOT_TYPICAL
    reference_kind: str = REFERENCE_MISSING
    reference_key: str = ""
    target: float = math.nan
    cutoff: float = math.nan
    reference_distance: float = math.nan
    reference_sd: float = math.nan

    @property
    def is_symmetry_mate(self) -> bool:
        return self.operator != IDENTITY_SYMMETRY_OPERATOR


@dataclass
class _ModelSearch:
    """The neighbour search of one model and the atom rows it answers with."""

    model: gemmi.Model
    search: gemmi.NeighborSearch
    rows: npt.NDArray[np.intp]
    by_label: dict[tuple[str, str], list[int]] = field(
        default_factory=dict[tuple[str, str], list[int]]
    )
    by_author: dict[tuple[str, str], list[int]] = field(
        default_factory=dict[tuple[str, str], list[int]]
    )


def _read_atoms(atoms: pd.DataFrame) -> _Atoms:
    """Return the atom table's columns as arrays."""
    element = text_array(atoms, "element", upper=True)
    xyz = np.column_stack([number_array(atoms, name) for name in ("x", "y", "z")])
    occupancy = number_array(atoms, "occupancy")
    model = text_array(atoms, "model_num")
    # A component is one residue of one model, told apart by chain, residue
    # number, insertion code, and name.
    residue_key = np.array(
        [
            "|".join(parts)
            for parts in zip(
                model,
                text_array(atoms, "chain_id"),
                text_array(atoms, "res_num"),
                text_array(atoms, "_atom_site.pdbx_PDB_ins_code"),
                text_array(atoms, "res_name"),
                strict=True,
            )
        ],
        dtype=object,
    )
    usable = (
        np.isfinite(xyz).all(axis=1)
        & ~np.isin(element, list(HYDROGEN_ELEMENTS))
        & ~(np.isfinite(occupancy) & (occupancy <= 0.0))
    )
    return _Atoms(
        element=element,
        res_name=text_array(atoms, "res_name", upper=True),
        atom_name=text_array(atoms, "atom_name"),
        residue_key=residue_key,
        model=model,
        altloc=text_array(atoms, "_atom_site.label_alt_id"),
        label_seq=text_array(atoms, "_atom_site.label_seq_id"),
        label_asym=text_array(atoms, "_atom_site.label_asym_id"),
        auth_asym=text_array(atoms, "_atom_site.auth_asym_id"),
        auth_seq=text_array(atoms, "_atom_site.auth_seq_id"),
        ins_code=text_array(atoms, "_atom_site.pdbx_PDB_ins_code"),
        xyz=xyz,
        occupancy=occupancy,
        b_factor=number_array(atoms, "_atom_site.B_iso_or_equiv"),
        usable=usable,
    )


def _refinement_terms(
    mmcif_dict: Mapping[str, Sequence[str | None]],
) -> tuple[float, float]:
    """Return the first reflection count and R-free the file gives together."""
    counts = mmcif_dict.get(_REFLECTION_COUNT_ITEM) or []
    r_frees = mmcif_dict.get(_R_FREE_ITEM) or []
    if not counts or len(counts) != len(r_frees):
        return math.nan, math.nan
    count_numbers = coerce_cif_numbers(pd.Series(list(counts), dtype=object))
    r_free_numbers = coerce_cif_numbers(pd.Series(list(r_frees), dtype=object))
    for count, r_free in zip(count_numbers, r_free_numbers, strict=True):
        if pd.notna(count) and pd.notna(r_free) and count > 0 and r_free > 0:
            return float(count), float(r_free)
    return math.nan, math.nan


def precision_index(
    atoms: _Atoms,
    crystal: CrystalSymmetry | None,
    mmcif_dict: Mapping[str, Sequence[str | None]],
) -> float:
    """Return the entry's diffraction-component precision index in Å, or NaN.

    The index is 1.28 Ni^(1/2) Va^(1/3) nobs^(-5/6) Rfree, equation 7 of Blow
    (2002). Ni sums the occupancies of the first model's atoms that are not
    hydrogen, with a kept alternate conformer counted whole, since the atom
    table holds one conformer where the file holds several that sum to one.
    NaN stands for a file without a usable cell and space group, without a
    reflection count and R-free given together, or with an occupancy among
    those atoms that is not a number.
    """
    if crystal is None:
        return math.nan
    reflection_count, r_free = _refinement_terms(mmcif_dict)
    if not (math.isfinite(reflection_count) and math.isfinite(r_free)):
        return math.nan
    models = list(dict.fromkeys(atoms.model.tolist()))
    if not models:
        return math.nan
    counted = (atoms.model == models[0]) & ~np.isin(
        atoms.element, list(HYDROGEN_ELEMENTS)
    )
    occupancy = np.where(atoms.altloc[counted] != "", 1.0, atoms.occupancy[counted])
    if occupancy.size == 0 or not np.isfinite(occupancy).all():
        return math.nan
    atom_count = float(occupancy.sum())
    asu_volume = float(crystal.cell.volume) / crystal.operation_count
    if atom_count <= 0 or asu_volume <= 0:
        return math.nan
    return float(
        1.28
        * atom_count**0.5
        * asu_volume ** (1 / 3)
        * reflection_count ** (-5 / 6)
        * r_free
    )


def _donor_class(res_name: str) -> str:
    if res_name in STANDARD_AMINO_ACIDS:
        return DONOR_AMINO_ACID
    info = gemmi.find_tabulated_residue(res_name)
    if info.is_water():
        return DONOR_WATER
    if info.is_nucleic_acid():
        return DONOR_NUCLEOTIDE
    return DONOR_OTHER


def donor_rule(
    res_name: str, atom_name: str, element: str, label_seq: str, donor_class: str
) -> str:
    """Return which donor rule an atom falls under.

    Every rule but RULE_NOT_TYPICAL lets the distance search infer a bond.
    The backbone nitrogen counts only in the residue whose label_seq_id is 1,
    the first of its polymer sequence, so the first modelled residue of a
    chain with a disordered start is not taken for a terminus.
    """
    if donor_class == DONOR_WATER:
        return RULE_WATER_OXYGEN if element == "O" else RULE_NOT_TYPICAL
    allowed = INFERRED_DONOR_ATOMS.get(res_name)
    if allowed is None:
        return RULE_NOT_TYPICAL
    name = atom_name.upper()
    if name in allowed:
        return (
            RULE_BACKBONE_CARBONYL if name == BACKBONE_CARBONYL_ATOM else RULE_SIDECHAIN
        )
    if name == N_TERMINAL_DONOR_ATOM and label_seq == "1":
        return RULE_N_TERMINUS
    if name in C_TERMINAL_DONOR_ATOMS:
        return RULE_C_TERMINUS
    return RULE_NOT_TYPICAL


def reference_key(
    res_name: str, atom_name: str, element: str, donor_class: str, metal: str
) -> tuple[str, str, str]:
    """Return the reference table key for a donor atom and a metal element.

    A water oxygen is keyed as HOH, the backbone carbonyl oxygen as residue
    CA, any other oxygen by its residue, and any other atom by its residue
    and element. The backbone nitrogen and the terminal oxygens are keyed as
    NTERM and CTERM, which the table has no rows for, so they fall back to
    the element.
    """
    name = atom_name.upper()
    if donor_class == DONOR_WATER and element == "O":
        return (REFERENCE_WATER, "O", metal)
    if name == N_TERMINAL_DONOR_ATOM:
        return (REFERENCE_N_TERMINUS, "N", metal)
    if name in C_TERMINAL_DONOR_ATOMS:
        return (REFERENCE_C_TERMINUS, "O", metal)
    if name == BACKBONE_CARBONYL_ATOM:
        return (REFERENCE_BACKBONE, "O", metal)
    if name.startswith("O"):
        return (res_name, "O", metal)
    return (res_name, element, metal)


def _annotate(contact: _Contact, atoms: _Atoms, metal_element: str) -> None:
    """Fill a contact's donor rule, reference, and first-sphere verdict."""
    row = contact.row
    res_name = str(atoms.res_name[row])
    atom_name = str(atoms.atom_name[row])
    element = str(atoms.element[row])
    donor_class = _donor_class(res_name)
    contact.rule = donor_rule(
        res_name, atom_name, element, str(atoms.label_seq[row]), donor_class
    )
    key = reference_key(res_name, atom_name, element, donor_class, metal_element)
    exact = literature_distances().get(key)
    if exact is not None:
        contact.reference_kind = REFERENCE_EXACT
        contact.reference_key = ":".join(key)
        contact.target = exact[0]
        contact.reference_distance, contact.reference_sd = exact
    else:
        fallback = element_targets().get((metal_element, element))
        if fallback is not None:
            contact.reference_kind = REFERENCE_ELEMENT_FALLBACK
            contact.reference_key = ":".join(("*", element, metal_element))
            contact.target = fallback
    if math.isfinite(contact.target):
        contact.cutoff = min(
            CANDIDATE_SEARCH_RADIUS_A, contact.target + FIRST_SPHERE_TOLERANCE_A
        )
        contact.inferred = (
            contact.rule != RULE_NOT_TYPICAL
            and donor_class in (DONOR_WATER, DONOR_AMINO_ACID)
            and element in DONOR_ELEMENTS
            and contact.distance <= contact.cutoff + DISTANCE_EPSILON_A
        )


def _build_search(
    atoms: _Atoms, rows: npt.NDArray[np.intp], crystal: CrystalSymmetry | None
) -> _ModelSearch:
    """Return the neighbour search over the given atom rows of one model.

    Each atom is a residue of its own in the search model, so a hit's residue
    index is the atom's position in ``rows``.
    """
    structure = gemmi.Structure()
    if crystal is not None:
        structure.cell = crystal.cell
    model = gemmi.Model(1)
    chain = gemmi.Chain("A")
    for row in rows.tolist():
        residue = gemmi.Residue()
        residue.name = "X"
        atom = gemmi.Atom()
        atom.name = "X"
        # The search skips hydrogens by element, so every atom is entered as
        # one that is kept.
        atom.element = gemmi.Element("O")
        x, y, z = atoms.xyz[row]
        atom.pos = gemmi.Position(float(x), float(y), float(z))
        atom.occ = 1.0
        residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    cell = crystal.cell if crystal is not None else structure.cell
    search = gemmi.NeighborSearch(
        structure[0], cell, CANDIDATE_SEARCH_RADIUS_A + 1.0
    ).populate(include_h=True)
    result = _ModelSearch(model=structure[0], search=search, rows=rows)
    for row in rows.tolist():
        name = str(atoms.atom_name[row])
        result.by_label.setdefault((str(atoms.label_asym[row]), name), []).append(row)
        result.by_author.setdefault((str(atoms.auth_asym[row]), name), []).append(row)
    return result


def _proximal_contacts(
    metal_row: int,
    atoms: _Atoms,
    search: _ModelSearch,
    crystal: CrystalSymmetry | None,
) -> dict[tuple[int, str], _Contact]:
    """Return the candidate donors within the search radius of one metal atom.

    The candidates are nitrogen, oxygen, and sulfur atoms of water and of the
    standard amino acids, outside the metal's own component, keyed by atom
    row and symmetry operator. Images of one atom closer than
    SPECIAL_POSITION_CUTOFF_A are collapsed to the nearest, with an image of
    the file itself preferred.
    """
    x, y, z = atoms.xyz[metal_row]
    metal_position = gemmi.Position(float(x), float(y), float(z))
    found: dict[tuple[int, str], _Contact] = {}
    marks = search.search.find_atoms(
        metal_position,
        "\0",
        min_dist=0.0,
        radius=CANDIDATE_SEARCH_RADIUS_A + DISTANCE_EPSILON_A,
    )
    for mark in marks:
        row = int(search.rows[mark.residue_idx])
        if row == metal_row and mark.image_idx == 0:
            continue
        if atoms.element[row] not in DONOR_ELEMENTS:
            continue
        if _donor_class(str(atoms.res_name[row])) not in (
            DONOR_WATER,
            DONOR_AMINO_ACID,
        ):
            continue
        ax, ay, az = atoms.xyz[row]
        atom_position = gemmi.Position(float(ax), float(ay), float(az))
        if crystal is not None:
            image = crystal.cell.find_nearest_pbc_image(
                metal_position, atom_position, mark.image_idx
            )
            moved = crystal.cell.find_nearest_pbc_position(
                metal_position, atom_position, mark.image_idx
            )
            operator = image.symmetry_code(True)
            distance = metal_position.dist(moved)
            position = (moved.x, moved.y, moved.z)
        else:
            operator = IDENTITY_SYMMETRY_OPERATOR
            distance = metal_position.dist(atom_position)
            position = (atom_position.x, atom_position.y, atom_position.z)
        if (
            operator == IDENTITY_SYMMETRY_OPERATOR
            and atoms.residue_key[row] == atoms.residue_key[metal_row]
        ):
            continue
        if not (0.0 < distance <= CANDIDATE_SEARCH_RADIUS_A + DISTANCE_EPSILON_A):
            continue
        key = (row, operator)
        current = found.get(key)
        if current is None or distance < current.distance:
            found[key] = _Contact(row, operator, distance, position)

    # One atom on a special position answers under several operators.
    by_row: dict[int, list[_Contact]] = {}
    for contact in found.values():
        by_row.setdefault(contact.row, []).append(contact)
    kept: dict[tuple[int, str], _Contact] = {}
    for contacts in by_row.values():
        retained: list[_Contact] = []
        for contact in sorted(
            contacts,
            key=lambda item: (item.is_symmetry_mate, item.distance, item.operator),
        ):
            if any(
                contact.position is not None
                and other.position is not None
                and math.dist(contact.position, other.position)
                <= SPECIAL_POSITION_CUTOFF_A
                for other in retained
            ):
                continue
            retained.append(contact)
        for contact in retained:
            kept[(contact.row, contact.operator)] = contact
    return kept


def _resolve(
    address: PartnerAddress, atoms: _Atoms, search: _ModelSearch
) -> int | None:
    """Return the atom row a record's partner names in one model, or None.

    The label chain and atom name find the candidates, or the author chain
    when the record gives no label chain. The residue numbers and the
    insertion code the record gives then narrow them. None stands for no
    atom, or for more than one.
    """
    if address.label_asym_id:
        rows = search.by_label.get((address.label_asym_id, address.atom_name), [])
    else:
        rows = search.by_author.get((address.auth_asym_id, address.atom_name), [])
    if address.auth_seq_id:
        rows = [row for row in rows if atoms.auth_seq[row] == address.auth_seq_id]
    if address.label_seq_id:
        rows = [row for row in rows if atoms.label_seq[row] == address.label_seq_id]
    rows = [row for row in rows if atoms.ins_code[row] == address.ins_code]
    return rows[0] if len(rows) == 1 else None


def _declared_contacts(
    records: Sequence[StructConnRecord],
    atoms: _Atoms,
    search: _ModelSearch,
    crystal: CrystalSymmetry | None,
    metal_rows: frozenset[int],
    proximal: dict[int, dict[tuple[int, str], _Contact]],
) -> int:
    """Mark or add the contacts deposited records name, and count the unresolved.

    ``proximal`` holds each metal row's candidates and is changed in place.
    A record counts as unresolved when one partner is a metal atom of the
    model and the other is no usable atom of it, or when its partners take
    different operators and the file gives no usable symmetry. A record
    between two metal atoms, or within one component, names no donor and is
    passed over.
    """
    unresolved = 0
    for record in records:
        first = _resolve(record.first, atoms, search)
        second = _resolve(record.second, atoms, search)
        pairs = [(first, second), (second, first)]
        metal_pairs = [
            (metal, donor)
            for metal, donor in pairs
            if metal is not None and metal in metal_rows
        ]
        if not metal_pairs:
            continue
        if len(metal_pairs) == 2:
            # Both partners are metal atoms, so neither is a donor.
            continue
        metal_row, donor_row = metal_pairs[0]
        if donor_row is None:
            unresolved += 1
            continue
        contacts = proximal[metal_row]
        if record.same_operator:
            if atoms.residue_key[donor_row] == atoms.residue_key[metal_row]:
                continue
            key = (donor_row, IDENTITY_SYMMETRY_OPERATOR)
            contact = contacts.get(key)
            if contact is None:
                contact = _Contact(
                    donor_row,
                    IDENTITY_SYMMETRY_OPERATOR,
                    math.dist(
                        atoms.xyz[metal_row].tolist(), atoms.xyz[donor_row].tolist()
                    ),
                    None,
                )
                contacts[key] = contact
            contact.declared = True
            continue
        mates = sorted(
            (
                contact
                for contact in contacts.values()
                if contact.row == donor_row and contact.is_symmetry_mate
            ),
            key=lambda item: item.distance,
        )
        if mates:
            mates[0].declared = True
            continue
        if crystal is None:
            unresolved += 1
            continue
        image = crystal.cell.find_nearest_image(
            gemmi.Position(*(float(value) for value in atoms.xyz[metal_row])),
            gemmi.Position(*(float(value) for value in atoms.xyz[donor_row])),
            gemmi.Asu.Different,
        )
        operator = image.symmetry_code(True)
        contact = _Contact(
            donor_row, operator, float(image.dist()), None, declared=True
        )
        contacts[(donor_row, operator)] = contact
    return unresolved


def compute_structure_metal_sites(
    atoms: pd.DataFrame,
    mmcif_dict: Mapping[str, Sequence[str | None]],
) -> MetalComputation:
    """Return the metal-site tables, link column, and report of one structure.

    ``atoms`` is the structure's atom table and ``mmcif_dict`` the flat
    dictionary of its file. A structure without metal atoms gives empty
    tables and the status no_metals. Raises what building a table from
    repeated atom identifiers raises when the host validates it.
    """
    table = _read_atoms(atoms)
    link = pd.Series(pd.NA, index=atoms.index, dtype=object)
    metal_mask = table.usable & np.isin(table.element, list(METAL_ELEMENTS))
    if not metal_mask.any():
        return MetalComputation(
            sites=METAL_SITES_TABLE.empty_dataframe(),
            bonds=METAL_BONDS_TABLE.empty_dataframe(),
            pairs=METAL_PAIRS_TABLE.empty_dataframe(),
            link=link,
            status=METAL_STATUS_NO_METALS,
            unresolved_record_count=0,
            dpi=math.nan,
            warnings=(),
        )
    warnings: list[str] = []
    symmetry = read_file_crystal_symmetry(mmcif_dict)
    crystal = symmetry if isinstance(symmetry, CrystalSymmetry) else None
    if isinstance(symmetry, str):
        warnings.append(
            f"Symmetry mates were not searched, since {symmetry}. Donor atoms "
            "of neighbouring molecules are missing from the tables."
        )
    dpi = precision_index(table, crystal, mmcif_dict)
    records = read_metal_records(mmcif_dict)
    site_keys = site_key_dataframe(atoms)
    site_ids = site_keys["site_id"].to_numpy(dtype=object)
    component_sizes = (
        pd.Series(table.residue_key[~np.isin(table.element, list(HYDROGEN_ELEMENTS))])
        .value_counts()
        .to_dict()
    )
    classes = cofactor_classes()

    site_rows: list[dict[str, object]] = []
    bond_rows: list[dict[str, object]] = []
    site_positions: list[int] = []
    bond_metal_positions: list[int] = []
    bond_donor_positions: list[int] = []
    pair_rows: list[dict[str, object]] = []
    pair_first_positions: list[int] = []
    pair_second_positions: list[int] = []
    unresolved_total = 0

    for model in dict.fromkeys(table.model.tolist()):
        in_model = table.model == model
        metal_rows = np.flatnonzero(in_model & metal_mask)
        if metal_rows.size == 0:
            continue
        search = _build_search(table, np.flatnonzero(in_model & table.usable), crystal)
        proximal = {
            int(row): _proximal_contacts(int(row), table, search, crystal)
            for row in metal_rows
        }
        unresolved_total += _declared_contacts(
            records,
            table,
            search,
            crystal,
            frozenset(int(row) for row in metal_rows),
            proximal,
        )
        metal_positions = table.xyz[metal_rows]
        for first, second in sorted(
            cKDTree(metal_positions).query_pairs(
                NEARBY_METAL_RADIUS_A + DISTANCE_EPSILON_A
            )
        ):
            first_row, second_row = int(metal_rows[first]), int(metal_rows[second])
            pair_first_positions.append(first_row)
            pair_second_positions.append(second_row)
            pair_rows.append(
                {
                    "metal_element": str(table.element[first_row]),
                    "partner_element": str(table.element[second_row]),
                    "distance_A": math.dist(
                        metal_positions[first].tolist(),
                        metal_positions[second].tolist(),
                    ),
                    "same_component": bool(
                        table.residue_key[first_row] == table.residue_key[second_row]
                    ),
                }
            )
        for index, metal_row in enumerate(metal_rows.tolist()):
            metal_element = str(table.element[metal_row])
            contacts = list(proximal[metal_row].values())
            for contact in contacts:
                _annotate(contact, table, metal_element)
            bonds = sorted(
                (
                    contact
                    for contact in contacts
                    if contact.inferred or contact.declared
                ),
                key=lambda item: (item.row, item.operator),
            )
            deviations: list[float] = []
            z_values: list[float] = []
            for contact in bonds:
                deviation = contact.distance - contact.reference_distance
                z_value = math.nan
                if math.isfinite(deviation) and math.isfinite(dpi):
                    denominator = math.sqrt(dpi**2 + contact.reference_sd**2)
                    z_value = deviation / denominator if denominator > 0 else math.nan
                if not math.isfinite(deviation):
                    bond_status = BOND_STATUS_NO_REFERENCE
                elif not math.isfinite(z_value):
                    bond_status = BOND_STATUS_NO_DPI
                else:
                    bond_status = BOND_STATUS_OK
                deviations.append(deviation)
                z_values.append(z_value)
                bond_metal_positions.append(metal_row)
                bond_donor_positions.append(contact.row)
                bond_rows.append(
                    {
                        "donor_symmetry_operator": contact.operator,
                        "is_symmetry_mate": contact.is_symmetry_mate,
                        "metal_element": metal_element,
                        "donor_element": str(table.element[contact.row]),
                        "donor_class": _donor_class(str(table.res_name[contact.row])),
                        "donor_rule": contact.rule,
                        "detection_method": detection_method(
                            by_record=contact.declared, by_distance=contact.inferred
                        ),
                        "distance_A": contact.distance,
                        "reference_kind": contact.reference_kind,
                        "reference_key": contact.reference_key,
                        "reference_distance_A": contact.reference_distance,
                        "reference_sd_A": contact.reference_sd,
                        "first_sphere_cutoff_A": contact.cutoff,
                        "distance_deviation_A": deviation,
                        "distance_z": z_value,
                        "donor_occupancy": float(table.occupancy[contact.row]),
                        "donor_b_factor_A2": float(table.b_factor[contact.row]),
                        "bond_status": bond_status,
                    }
                )

            residue_name = str(table.res_name[metal_row])
            component_size = int(component_sizes.get(table.residue_key[metal_row], 1))
            structural_class = classes.get(residue_name, "")
            if structural_class == PARENT_CLUSTER:
                parent_type = PARENT_CLUSTER
            elif structural_class == PARENT_HEME:
                parent_type = PARENT_HEME
            elif component_size == 1:
                parent_type = PARENT_ION
            else:
                parent_type = PARENT_OTHER

            others = np.delete(np.arange(len(metal_rows)), index)
            nearest_distance = math.nan
            nearest_element: object = pd.NA
            nearest_site: object = pd.NA
            nearby = 0
            if others.size:
                offsets = metal_positions[others] - metal_positions[index]
                separations: npt.NDArray[np.float64] = np.sqrt(
                    np.square(offsets).sum(axis=1)
                )
                nearest = int(np.argmin(separations))
                nearest_distance = float(separations[nearest])
                nearest_row = int(metal_rows[others[nearest]])
                nearest_element = str(table.element[nearest_row])
                nearest_site = str(site_ids[nearest_row])
                nearby = int(
                    (separations <= NEARBY_METAL_RADIUS_A + DISTANCE_EPSILON_A).sum()
                )

            special = on_special_position(
                crystal.cell if crystal is not None else None,
                table.xyz[metal_row].tolist(),
            )

            distances = [contact.distance for contact in bonds]
            donor_b = finite_mean(
                float(table.b_factor[contact.row]) for contact in bonds
            )
            metal_b = float(table.b_factor[metal_row])
            finite_z = [value for value in z_values if math.isfinite(value)]
            site_positions.append(metal_row)
            link.iloc[metal_row] = str(site_ids[metal_row])
            site_rows.append(
                {
                    "metal_element": metal_element,
                    "parent_type": parent_type,
                    "component_atom_count": component_size,
                    "coordination_number": len(bonds),
                    "n_symmetry_bonds": sum(c.is_symmetry_mate for c in bonds),
                    "n_declared_bonds": sum(c.declared for c in bonds),
                    "n_donor_N": sum(table.element[c.row] == "N" for c in bonds),
                    "n_donor_O": sum(table.element[c.row] == "O" for c in bonds),
                    "n_donor_S": sum(table.element[c.row] == "S" for c in bonds),
                    "n_donor_water": sum(
                        _donor_class(str(table.res_name[c.row])) == DONOR_WATER
                        and table.element[c.row] == "O"
                        for c in bonds
                    ),
                    "donor_distance_min_A": min(distances) if distances else math.nan,
                    "donor_distance_mean_A": finite_mean(distances),
                    "donor_distance_max_A": max(distances) if distances else math.nan,
                    "n_reference_bonds": sum(math.isfinite(d) for d in deviations),
                    "distance_deviation_mean_A": finite_mean(deviations),
                    "distance_z_rms": (
                        float(np.sqrt(np.mean(np.square(finite_z))))
                        if finite_z
                        else math.nan
                    ),
                    "occupancy": float(table.occupancy[metal_row]),
                    "b_factor_A2": metal_b,
                    "donor_b_factor_mean_A2": donor_b,
                    "b_factor_ratio_to_donors": positive_ratio(metal_b, donor_b),
                    "nearest_metal_distance_A": nearest_distance,
                    "nearest_metal_element": nearest_element,
                    "nearest_metal_site_id": nearest_site,
                    "n_metals_within_6A": nearby,
                    "on_special_position": pd.NA if special is None else special,
                    "dpi_A": dpi,
                    "metal_site_status": (
                        SITE_STATUS_OK if bonds else SITE_STATUS_NO_DONORS
                    ),
                }
            )

    sites = pd.concat(
        [
            site_keys.iloc[site_positions].reset_index(drop=True),
            pd.DataFrame(site_rows),
        ],
        axis=1,
    )
    if bond_rows:
        donor_keys = site_key_dataframe(atoms.iloc[bond_donor_positions], DONOR_PREFIX)
        bonds_frame = pd.concat(
            [
                site_keys.iloc[bond_metal_positions].reset_index(drop=True),
                donor_keys.reset_index(drop=True),
                pd.DataFrame(bond_rows),
            ],
            axis=1,
        )
    else:
        bonds_frame = METAL_BONDS_TABLE.empty_dataframe()
    if pair_rows:
        pairs_frame = pd.concat(
            [
                site_keys.iloc[pair_first_positions].reset_index(drop=True),
                site_key_dataframe(
                    atoms.iloc[pair_second_positions], PARTNER_PREFIX
                ).reset_index(drop=True),
                pd.DataFrame(pair_rows),
            ],
            axis=1,
        )
    else:
        pairs_frame = METAL_PAIRS_TABLE.empty_dataframe()
    if unresolved_total:
        warnings.append(
            f"{unresolved_total} deposited metal coordination record(s) named a "
            "metal atom and no usable donor atom of the atom table, so no bond "
            "row holds them."
        )
    return MetalComputation(
        sites=sites,
        bonds=bonds_frame,
        pairs=pairs_frame,
        link=link,
        status=METAL_STATUS_DEGRADED if unresolved_total else METAL_STATUS_OK,
        unresolved_record_count=unresolved_total,
        dpi=dpi,
        warnings=tuple(warnings),
    )


__all__ = [
    "MetalComputation",
    "compute_structure_metal_sites",
    "donor_rule",
    "precision_index",
    "reference_key",
]
