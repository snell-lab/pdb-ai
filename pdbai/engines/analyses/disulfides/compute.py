"""The candidate disulfides of one structure, from every conformer its file holds.

compute_structure_disulfides turns one structure's atom table and its flat
mmCIF dictionary into the rows of the disulfides table and the atom link
column. The atom table alone is not enough. The parser keeps one conformer
of each residue, so a disulfide modelled as an intact conformer A and a
broken conformer B shows only one of them there. The cysteine SG and CB rows
are therefore read again from the dictionary's _atom_site items, every
conformer of every model. Each gets the column names the atom table uses, so
site_key_dataframe gives it the identifier the atom table would. Hydrogen and
deuterium atoms are never read, since rows are chosen by residue and atom
name. A cysteine recorded as HETATM is read like any other.

A row is a cysteine's when its residue name is CYS. The name, the chain, and
the residue number of a row are the ones atom_site_residue_identity gives.
The parser fills the atom table by the same function, so the engine holds no
rule of its own and its rows carry the atom table's names. The name is the
author component, or the label component where the author one is missing,
row by row, which is also the rule records.py applies to a partner of a
record. Some integrative models and every file written by gemmi's default
writer leave _atom_site.auth_comp_id out, and read by the author name alone
they would give no rows and no warning. The chain and the residue number are
the author ones, or the label ones for the whole file when it gives no
author chain or no author residue number on any atom. That numbering is
decided once, on every atom of the file and before the cysteine rows are
chosen, since a selection of rows could decide otherwise than the parser
did.

The rows the atom table kept are found again by model number and atom id. A
kept row must have the site_id the atom table gives it, and every cysteine
SG and CB of the atom table must be found. Otherwise RuntimeError is raised,
since rows that silently disagree with the atom table would be worse than
none. Two rows with one site_id raise ValueError, since the table could not
tell them apart.

A residue is one cysteine of one model, keyed by model, chain, residue
number, and insertion code. Its representative SG is the SG row the atom
table keeps. When the kept conformer has no SG, the representative is the SG
of highest occupancy, a missing occupancy ranking lowest and file order
breaking a tie. No atom then links to that residue, a warning names it, and
the row says sg_not_in_atom_table unless an earlier gap applies. Its
representative CB is the CB that can coexist with that SG, preferring the one
the atom table keeps and otherwise ranked the same way. Two conformers can
coexist when their alternate location labels are equal or either has none.

A pair is found in two ways, model by model. A deposited record from
records.py names two residues by their author fields, or by their label
fields when the author fields are missing, and applies to every model,
since _struct_conn names no model. In a file numbered by its label
identifiers the residue keys hold label values, so a record there names its
residues by its label fields alone, whether or not it gives author fields. A
record that finds no pair of distinct
cysteines with SG atoms in any model is unmatched. It is counted and warned
about, and it makes the structure degraded. The distance search pairs SG
conformers of different residues that can coexist and lie at most
DISULFIDE_SEARCH_CUTOFF_A apart. It uses only usable conformers, those with
coordinates and an occupancy that is not zero or below. A missing occupancy
does not make a conformer unusable, since unknown is not zero. The search
never crosses models, and it has no lower limit, so two sulfurs modelled
implausibly close are a row with the distance as measured. A pair a record
names but whose SG atoms lie beyond the limit, or have no usable conformers
to measure, keeps its record as its only source and whatever distances could
be measured. Such a row is not evidence of a cleaved bond. A cysteine with a
CB but no SG row is in no pair found by distance. It is counted as a cysteine
the table cannot see, which is not a count of damage, since side chains are
cut back for many reasons.

Either way a row is a candidate, and two cysteines bound to one metal atom
can be one. No pair is left out for that, and no metal record is read. The
metal atoms of the file are read instead, chosen by element before any frame
is built, and a row is metal_bridged when a usable SG conformer of each
residue lies within METAL_CONTACT_DISTANCE_A of the same metal atom of the
same model, where the two SG conformers can coexist with each other and each
with the metal atom. A metal atom is one _atom_site row whose type symbol
names a metallic element, with coordinates and an occupancy that is not zero
or below. Only the atoms of the file are read, so a metal of a symmetry mate
is not seen. The flag changes no other value of a row.

The measures summarize every SG conformer of the two residues, as columns.py
describes. Each row's pair_status names the first gap that applies. The two
residues of a row are ordered by chain, then residue number, numerically
where it is a whole number and after all such numbers otherwise, then
insertion code and site_id. Rows are ordered by model and then by those
keys, so a rerun, or a file listing its atoms or records in another order,
gives the same table.
"""

from __future__ import annotations

import math
import re
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial import cKDTree

from ....structure import (
    RESIDUE_NUMBERING_LABEL,
    ResidueNumbering,
    atom_site_residue_identity,
    atom_site_residue_names,
    atom_site_residue_numbering,
    coerce_cif_numbers,
)
from ....structure.residue_identity import clean_string_series
from ...core import site_key_columns, site_key_dataframe
from ..metal_sites.constants import METAL_ELEMENTS
from ..struct_conn import PartnerAddress, StructConnRecord, detection_method
from .columns import DISULFIDES_TABLE, PARTNER_PREFIX
from .constants import (
    BETA_CARBON_ATOM_NAME,
    CYSTEINE_RESIDUE_NAME,
    DISULFIDE_SEARCH_CUTOFF_A,
    DISULFIDE_STATUS_DEGRADED,
    DISULFIDE_STATUS_OK,
    METAL_CONTACT_DISTANCE_A,
    OCCUPANCY_SUM_DECIMALS,
    PAIR_STATUS_B_FACTOR_UNAVAILABLE,
    PAIR_STATUS_CB_MISSING,
    PAIR_STATUS_NO_COMPATIBLE_CONFORMERS,
    PAIR_STATUS_NO_USABLE_SG,
    PAIR_STATUS_OCCUPANCY_MISSING,
    PAIR_STATUS_OK,
    PAIR_STATUS_SG_NOT_IN_ATOM_TABLE,
    SULFUR_ATOM_NAME,
)
from .records import read_disulfide_records

_ATOM_SITE_PREFIX = "_atom_site."
_ELEMENT_ITEM = "_atom_site.type_symbol"
_ATOM_NAME_ITEM = "_atom_site.label_atom_id"
_MODEL_ITEM = "_atom_site.pdbx_PDB_model_num"
_ATOM_ID_ITEM = "_atom_site.id"
_ALT_ID_ITEM = "_atom_site.label_alt_id"
_LABEL_ASYM_ITEM = "_atom_site.label_asym_id"
_LABEL_SEQ_ITEM = "_atom_site.label_seq_id"

# The items kept under their own names, as the atom table keeps them, so that
# site_key_dataframe and the match to the atom table read them alike.
_RAW_ITEMS = (
    "_atom_site.pdbx_PDB_ins_code",
    _ALT_ID_ITEM,
    _MODEL_ITEM,
    _ATOM_ID_ITEM,
    _LABEL_ASYM_ITEM,
    _LABEL_SEQ_ITEM,
)
# The items that name a residue, which atom_site_residue_names reads.
_COMPONENT_ITEMS = ("_atom_site.auth_comp_id", "_atom_site.label_comp_id")
# The items atom_site_residue_identity reads to name a row as the atom table
# does.
_IDENTITY_ITEMS = (
    "_atom_site.auth_asym_id",
    "_atom_site.auth_seq_id",
    _LABEL_ASYM_ITEM,
    _LABEL_SEQ_ITEM,
    *_COMPONENT_ITEMS,
)
_COORDINATE_ITEMS = (
    "_atom_site.Cartn_x",
    "_atom_site.Cartn_y",
    "_atom_site.Cartn_z",
)
_OCCUPANCY_ITEM = "_atom_site.occupancy"
_NUMBER_COLUMNS = (
    *zip(("x", "y", "z"), _COORDINATE_ITEMS, strict=True),
    ("occupancy", _OCCUPANCY_ITEM),
    ("b_factor", "_atom_site.B_iso_or_equiv"),
)
_READ_ITEMS = frozenset(
    {
        *_RAW_ITEMS,
        *_IDENTITY_ITEMS,
        _ATOM_NAME_ITEM,
        *(item for _, item in _NUMBER_COLUMNS),
    }
)
# The items read of a metal atom, which give its model, its conformer, its
# position, and whether it is there at all.
_METAL_ITEMS = frozenset(
    {_MODEL_ITEM, _ALT_ID_ITEM, *_COORDINATE_ITEMS, _OCCUPANCY_ITEM}
)
_SITE_KEY_COLUMNS = tuple(column.name for column in site_key_columns())

_WHOLE_NUMBER = re.compile(r"[+-]?\d+")

# An element symbol as _atom_site.type_symbol writes it, with or without a
# charge such as the 2+ of ZN2+ or the 1+ of K1+. The letters alone name the
# element, in either case.
_ELEMENT_SYMBOL = re.compile(r"([A-Za-z]{1,2})[0-9+-]*")

# The decimals a distance is rounded to before it is compared with a limit,
# so that coordinates written to three decimals fall on the side of the limit
# their decimal values put them, whatever the floating-point error.
_DISTANCE_COMPARISON_DECIMALS = 6
# How far beyond a limit the KD-tree looks, so that it drops no pair the
# rounded comparison would keep.
_SEARCH_MARGIN_A = 1e-6

# Why a record found no pair in a model, as its warning words it.
_RESIDUE_ABSENT = "names a cysteine with no SG or CB atom in the structure"
_RESIDUE_WITHOUT_SULFUR = "names a cysteine that has no SG atom"
_SAME_RESIDUE_TWICE = "names the same cysteine twice"

# A residue within the structure: model text, chain, residue number, and
# insertion code, as the residue's site identifiers spell them.
type _ResidueKey = tuple[str, str, str, str]
# Orders residues by chain, then by residue number. A whole number sorts as 0
# and its value, before any other text, which sorts as 1 and the text.
# Insertion code and the representative site_id come last.
type _ResidueSortKey = tuple[str, int, int, str, str, str]
type _ResiduePair = tuple[_ResidueKey, _ResidueKey]


@dataclass(frozen=True)
class DisulfideComputation:
    """The disulfides of one structure, with its status and warnings.

    ``table`` holds the rows of the disulfides table in their declared columns,
    which validate_tables casts to their declared types. ``link`` holds the
    atom link column on the atom table's index. ``status`` is "ok" or
    "degraded". ``unmatched_record_count`` counts the in-scope records that
    matched in no model, and ``symmetry_record_count`` the records that join a
    symmetry mate. ``cysteine_without_sg_count`` counts the cysteines, once per
    model, that have a CB row but no SG row in any conformer. ``warnings``
    holds each distinct message once, in the order raised, without the engine's
    name.
    """

    table: pd.DataFrame
    link: pd.Series
    status: str
    unmatched_record_count: int
    symmetry_record_count: int
    cysteine_without_sg_count: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _Conformers:
    """The cysteine SG and CB rows of every conformer, by row position.

    Each sequence holds one value per row of the full-conformer frame, in
    _atom_site order.
    """

    # The numbering the chain and residue number of every residue key use.
    numbering: ResidueNumbering
    # Each row's site key columns, by name, as a table row takes them.
    site_keys: tuple[Mapping[str, object], ...]
    site_ids: tuple[str, ...]
    residues: tuple[_ResidueKey, ...]
    # Each row's label chain and label sequence number.
    label_addresses: tuple[tuple[str, str], ...]
    is_sulfur: tuple[bool, ...]
    altlocs: tuple[str, ...]
    occupancies: tuple[float, ...]
    b_factors: tuple[float, ...]
    coordinates: tuple[tuple[float, float, float], ...]
    usable: tuple[bool, ...]
    file_positions: tuple[int, ...]
    # The atom table's index label of each row the atom table kept.
    atom_labels: tuple[Hashable | None, ...]

    def rank(self, row: int) -> tuple[bool, float, int]:
        """Return the order in which ``row`` is preferred as a representative.

        A row the atom table keeps comes first, then the highest occupancy,
        with a missing one last, then the earliest in the file.
        """
        occupancy = self.occupancies[row]
        return (
            self.atom_labels[row] is None,
            math.inf if math.isnan(occupancy) else -occupancy,
            self.file_positions[row],
        )


@dataclass(frozen=True)
class _Metals:
    """The usable metal atoms of the structure file, one per _atom_site row.

    Each sequence holds one value per metal atom, in _atom_site order. Two
    conformers of one metal are two atoms here, so two sulfurs share a metal
    only when they are bound to the same conformer of it.
    """

    # Each atom's model, spelled as the model of a residue key is.
    models: tuple[str, ...]
    altlocs: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class _Residue:
    """One cysteine of one model that has SG rows, and its representatives."""

    key: _ResidueKey
    sulfur_rows: tuple[int, ...]
    carbon_rows: tuple[int, ...]
    usable_sulfur_rows: tuple[int, ...]
    representative_sulfur: int
    # None when no CB can coexist with the representative SG.
    representative_carbon: int | None

    def sulfur_in_atom_table(self, conformers: _Conformers) -> bool:
        """Return whether the atom table keeps the representative SG."""
        return conformers.atom_labels[self.representative_sulfur] is not None


def _compatible(altloc: str, other: str) -> bool:
    """Return whether conformers with these labels can coexist."""
    return altloc == other or not altloc or not other


def _numbers(raw: pd.DataFrame, item: str) -> pd.Series:
    """Return an item's values as floats, NaN where missing or unreadable.

    Numbers are read by the parser's grammar, so a standard uncertainty such
    as the (3) of 2.040(3) is dropped and the kept rows get the values the
    atom table holds. Only text that is not a CIF number at all becomes NaN,
    which the parser would refuse on a kept row, so a dropped conformer's bad
    number cannot fail the engine.
    """
    if item not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return coerce_cif_numbers(raw[item])


def _model_numbers(raw: pd.DataFrame) -> pd.Series:
    """Return each row's model number as Int64, as the atom table holds it.

    A fractional value, which the parser refuses on a kept row, becomes
    missing rather than cut to a whole number by the cast.
    """
    numbers = _numbers(raw, _MODEL_ITEM)
    return numbers.where(numbers % 1 == 0).astype("Int64")


def _model_texts(model_numbers: pd.Series) -> list[str]:
    """Return each model number as a residue key spells it, "" where missing."""
    return ["" if pd.isna(model) else str(model) for model in model_numbers]


def _usable(point: tuple[float, float, float], occupancy: float) -> bool:
    """Return whether an atom has coordinates and is not at zero occupancy.

    A missing occupancy is unknown rather than zero, so it stays usable.
    """
    return all(math.isfinite(axis) for axis in point) and not occupancy <= 0


def _is_cysteine_atom(residue_names: pd.Series, atom_names: pd.Series) -> pd.Series:
    """Return whether each row is the SG or CB of a cysteine, from its names.

    ``residue_names`` holds the names atom_site_residue_names gives, which
    the atom table keeps as res_name, so the rows of the file and the rows of
    the atom table are chosen alike. The case and the spaces around a residue
    name do not matter.
    """
    return residue_names.str.strip().str.upper().eq(
        CYSTEINE_RESIDUE_NAME
    ) & atom_names.isin((SULFUR_ATOM_NAME, BETA_CARBON_ATOM_NAME))


def _atom_site_items(
    mmcif_dict: Mapping[str, Sequence[str | None]],
) -> dict[str, Sequence[str | None]]:
    """Return the _atom_site items of a flat mmCIF dictionary."""
    return {
        item: values
        for item, values in mmcif_dict.items()
        if item.startswith(_ATOM_SITE_PREFIX)
    }


def _rows_at(
    atom_site: Mapping[str, Sequence[str | None]],
    positions: Sequence[int],
    items: frozenset[str],
) -> pd.DataFrame:
    """Return the given items of the _atom_site rows at ``positions``.

    Only the rows asked for are copied, so a large structure costs what its
    cysteines and metals do.
    """
    return pd.DataFrame(
        {
            item: [values[position] for position in positions]
            for item, values in atom_site.items()
            if item in items
        },
        index=pd.RangeIndex(len(positions)),
    )


def _conformer_frame(
    atom_site: Mapping[str, Sequence[str | None]],
    pdb_id: str,
    numbering: ResidueNumbering,
) -> pd.DataFrame:
    """Return every cysteine SG and CB row of _atom_site, as the atom table names it.

    Rows are chosen before the frame is built, by the residue name the parser
    gives them. Their chain, residue number, and residue name then come from
    atom_site_residue_identity under ``numbering``, the numbering of the
    whole file. ``pdb_id`` is upper-cased, as the parser does.
    """
    row_count = next((len(values) for values in atom_site.values()), 0)
    names = pd.DataFrame(
        {
            item: list(atom_site[item])
            for item in (*_COMPONENT_ITEMS, _ATOM_NAME_ITEM)
            if item in atom_site
        },
        index=pd.RangeIndex(row_count),
    )
    selected = _is_cysteine_atom(
        atom_site_residue_names(names), clean_string_series(names, _ATOM_NAME_ITEM)
    )
    positions: list[int] = np.flatnonzero(selected.to_numpy(dtype=bool)).tolist()
    raw = _rows_at(atom_site, positions, _READ_ITEMS)
    identity = atom_site_residue_identity(raw, numbering=numbering)

    frame = pd.DataFrame(index=raw.index)
    frame["pdb_id"] = pdb_id.upper()
    frame["chain_id"] = identity.chain_id
    frame["res_num"] = identity.res_num
    frame["res_name"] = identity.res_name
    frame["atom_name"] = clean_string_series(raw, _ATOM_NAME_ITEM)
    # site_key_dataframe spells the model of site_id from model_num, which the
    # atom table holds as Int64, so the same type gives the same spelling.
    frame["model_num"] = _model_numbers(raw)
    for item in _RAW_ITEMS:
        if item in raw.columns:
            frame[item] = raw[item]
    for column, item in _NUMBER_COLUMNS:
        frame[column] = _numbers(raw, item)
    frame["file_position"] = positions
    return frame


def _refuse_repeated_sites(site_ids: pd.Series) -> None:
    """Raise ValueError naming the first site_id two rows share."""
    repeated = site_ids[site_ids.duplicated()]
    if not repeated.empty:
        raise ValueError(
            f"Two atoms share the site identifier {repeated.iloc[0]}, so the "
            "disulfides table cannot tell them apart."
        )


def _kept_atom_labels(
    frame: pd.DataFrame, site_ids: pd.Series, atom_dataframe: pd.DataFrame
) -> list[Hashable | None]:
    """Return the atom table's index label of each frame row it kept, or None.

    A row is kept when the atom table has a cysteine SG or CB of its model
    number and atom id. The atom table's cysteines are chosen by res_name,
    which the parser fills by the rule that chose the file's. Raises
    RuntimeError when a kept row's site_id differs from the atom table's, or
    when a cysteine SG or CB of the atom table has no row in the frame.
    """
    atoms = atom_dataframe[
        _is_cysteine_atom(
            clean_string_series(atom_dataframe, "res_name"),
            clean_string_series(atom_dataframe, "atom_name"),
        )
    ]
    atom_site_ids: dict[Hashable, str] = dict(
        zip(atoms.index, site_key_dataframe(atoms)["site_id"].tolist(), strict=True)
    )
    label_by_atom: dict[tuple[str, str], Hashable] = dict(
        zip(
            zip(
                clean_string_series(atoms, _MODEL_ITEM),
                clean_string_series(atoms, _ATOM_ID_ITEM),
                strict=True,
            ),
            atoms.index,
            strict=True,
        )
    )
    labels: list[Hashable | None] = [
        label_by_atom.get(atom)
        for atom in zip(
            clean_string_series(frame, _MODEL_ITEM),
            clean_string_series(frame, _ATOM_ID_ITEM),
            strict=True,
        )
    ]
    for label, site_id in zip(labels, site_ids.tolist(), strict=True):
        if label is not None and atom_site_ids[label] != site_id:
            raise RuntimeError(
                "The structure file does not match the atom table, which names "
                f"atom {site_id} as {atom_site_ids[label]}."
            )
    found = set(labels)
    for label, site_id in atom_site_ids.items():
        if label not in found:
            raise RuntimeError(
                "The structure file does not match the atom table, since it "
                f"holds no atom for {site_id} of the atom table."
            )
    return labels


def _read_conformers(
    atom_dataframe: pd.DataFrame,
    atom_site: Mapping[str, Sequence[str | None]],
    pdb_id: str,
) -> _Conformers:
    """Return every cysteine SG and CB conformer, matched to the atom table.

    The numbering is decided here, on every atom of the file, as the parser
    decides it. Raises ValueError when two rows share a site_id, and
    RuntimeError when the file does not match the atom table.
    """
    numbering = atom_site_residue_numbering(atom_site)
    frame = _conformer_frame(atom_site, pdb_id, numbering)
    site_keys = site_key_dataframe(frame)
    _refuse_repeated_sites(site_keys["site_id"])
    atom_labels = _kept_atom_labels(frame, site_keys["site_id"], atom_dataframe)
    models = _model_texts(site_keys["model_num"])
    residues: tuple[_ResidueKey, ...] = tuple(
        zip(
            models,
            site_keys["chain_id"].tolist(),
            site_keys["res_num"].tolist(),
            site_keys["ins_code"].tolist(),
            strict=True,
        )
    )
    occupancies: tuple[float, ...] = tuple(frame["occupancy"].tolist())
    coordinates: tuple[tuple[float, float, float], ...] = tuple(
        zip(frame["x"].tolist(), frame["y"].tolist(), frame["z"].tolist(), strict=True)
    )
    return _Conformers(
        numbering=numbering,
        site_keys=tuple(
            dict(zip(_SITE_KEY_COLUMNS, values, strict=True))
            for values in site_keys[list(_SITE_KEY_COLUMNS)].itertuples(index=False)
        ),
        site_ids=tuple(str(site_id) for site_id in site_keys["site_id"].tolist()),
        residues=residues,
        label_addresses=tuple(
            zip(
                clean_string_series(frame, _LABEL_ASYM_ITEM),
                clean_string_series(frame, _LABEL_SEQ_ITEM),
                strict=True,
            )
        ),
        is_sulfur=tuple(frame["atom_name"].eq(SULFUR_ATOM_NAME).tolist()),
        altlocs=tuple(clean_string_series(frame, _ALT_ID_ITEM).tolist()),
        occupancies=occupancies,
        b_factors=tuple(frame["b_factor"].tolist()),
        coordinates=coordinates,
        usable=tuple(
            _usable(point, occupancy)
            for point, occupancy in zip(coordinates, occupancies, strict=True)
        ),
        file_positions=tuple(frame["file_position"].tolist()),
        atom_labels=tuple(atom_labels),
    )


def _names_metal(symbol: str | None) -> bool:
    """Return whether an _atom_site.type_symbol value names a metallic element.

    The metals are METAL_ELEMENTS, the elements the metal-site engine counts
    as metals. Case and a charge suffix do not matter, so ZN, Zn, and Zn2+
    all name zinc. A missing value, or text that is not an element symbol
    with at most a charge, names no metal.
    """
    match = _ELEMENT_SYMBOL.fullmatch(symbol.strip()) if symbol else None
    return match is not None and match.group(1).upper() in METAL_ELEMENTS


def _read_metals(atom_site: Mapping[str, Sequence[str | None]]) -> _Metals:
    """Return the usable metal atoms of _atom_site, of any residue and record type.

    Rows are chosen by element before the frame is built, and each distinct
    symbol is looked up once. A file without _atom_site.type_symbol names no
    element and so gives no metal atom. An atom without coordinates, or at an
    occupancy of zero or below, is left out, as such an SG is from the search.
    """
    symbols = atom_site.get(_ELEMENT_ITEM, ())
    metal_symbols = {symbol for symbol in set(symbols) if _names_metal(symbol)}
    positions = [
        position for position, symbol in enumerate(symbols) if symbol in metal_symbols
    ]
    raw = _rows_at(atom_site, positions, _METAL_ITEMS)
    coordinates: list[tuple[float, float, float]] = list(
        zip(*(_numbers(raw, item).tolist() for item in _COORDINATE_ITEMS), strict=True)
    )
    occupancies: list[float] = _numbers(raw, _OCCUPANCY_ITEM).tolist()
    kept = [
        row
        for row, (point, occupancy) in enumerate(
            zip(coordinates, occupancies, strict=True)
        )
        if _usable(point, occupancy)
    ]
    models = _model_texts(_model_numbers(raw))
    altlocs: list[str] = clean_string_series(raw, _ALT_ID_ITEM).tolist()
    return _Metals(
        models=tuple(models[row] for row in kept),
        altlocs=tuple(altlocs[row] for row in kept),
        coordinates=tuple(coordinates[row] for row in kept),
    )


def _build_residues(
    conformers: _Conformers,
) -> tuple[dict[_ResidueKey, _Residue], frozenset[_ResidueKey]]:
    """Return the residues with SG rows, and the keys of those with none.

    Each residue comes with its representative conformers, in file order.
    """
    rows_by_residue: dict[_ResidueKey, list[int]] = {}
    for row, key in enumerate(conformers.residues):
        rows_by_residue.setdefault(key, []).append(row)
    residues: dict[_ResidueKey, _Residue] = {}
    for key, rows in rows_by_residue.items():
        sulfur_rows = tuple(row for row in rows if conformers.is_sulfur[row])
        if not sulfur_rows:
            continue
        sulfur = min(sulfur_rows, key=conformers.rank)
        carbon = min(
            (
                row
                for row in rows
                if not conformers.is_sulfur[row]
                and _compatible(conformers.altlocs[row], conformers.altlocs[sulfur])
            ),
            key=conformers.rank,
            default=None,
        )
        residues[key] = _Residue(
            key=key,
            sulfur_rows=sulfur_rows,
            carbon_rows=tuple(row for row in rows if not conformers.is_sulfur[row]),
            usable_sulfur_rows=tuple(
                row for row in sulfur_rows if conformers.usable[row]
            ),
            representative_sulfur=sulfur,
            representative_carbon=carbon,
        )
    return residues, frozenset(rows_by_residue.keys() - residues.keys())


def _sort_key(residue: _Residue, conformers: _Conformers) -> _ResidueSortKey:
    """Return the key that orders residues within a model, as the module says."""
    _model, chain_id, res_num, ins_code = residue.key
    site_id = conformers.site_ids[residue.representative_sulfur]
    if _WHOLE_NUMBER.fullmatch(res_num):
        return (chain_id, 0, int(res_num), "", ins_code, site_id)
    return (chain_id, 1, 0, res_num, ins_code, site_id)


def _ordered_pair(
    first: _Residue, second: _Residue, conformers: _Conformers
) -> _ResiduePair:
    """Return the two residues' keys, the one that sorts first leading."""
    if _sort_key(second, conformers) < _sort_key(first, conformers):
        first, second = second, first
    return (first.key, second.key)


def _model_order(model: str) -> tuple[int, int]:
    """Return the key that orders models by number, a missing number last."""
    return (0, int(model)) if model else (1, 0)


class _RecordResolver:
    """Finds the two residues a deposited record joins, model by model."""

    def __init__(
        self,
        residues: Mapping[_ResidueKey, _Residue],
        without_sulfur: frozenset[_ResidueKey],
        conformers: _Conformers,
    ) -> None:
        """Index every cysteine by its label fields, model by model."""
        self.residues = residues
        self.without_sulfur = without_sulfur
        self.label_numbered = conformers.numbering == RESIDUE_NUMBERING_LABEL
        self.by_label: dict[tuple[str, str, str], _ResidueKey] = {}
        for key, (label_asym_id, label_seq_id) in zip(
            conformers.residues, conformers.label_addresses, strict=True
        ):
            # A residue's rows share their label fields, and a label pair is an
            # address only when both parts are given.
            if label_asym_id and label_seq_id:
                self.by_label.setdefault((key[0], label_asym_id, label_seq_id), key)

    def _residue(self, partner: PartnerAddress, model: str) -> _ResidueKey | None:
        """Return the cysteine a partner names in ``model``, or None.

        The author fields name it when both are given. The label fields are
        tried only when they are not, never after the author fields miss. In
        a structure numbered by its label identifiers the residue keys hold
        label values, which author fields cannot match, so the label fields
        alone name it there.
        """
        if self.label_numbered or not partner.has_author_fields:
            return self.by_label.get(
                (model, partner.label_asym_id, partner.label_seq_id)
            )
        key = (model, partner.auth_asym_id, partner.auth_seq_id, partner.ins_code)
        return key if key in self.residues or key in self.without_sulfur else None

    def resolve(
        self, record: StructConnRecord, model: str
    ) -> tuple[_Residue, _Residue] | str:
        """Return the two residues ``record`` joins in ``model``, or why none."""
        first = self._residue(record.first, model)
        second = self._residue(record.second, model)
        if first is None or second is None:
            return _RESIDUE_ABSENT
        if first == second:
            return _SAME_RESIDUE_TWICE
        if first not in self.residues or second not in self.residues:
            return _RESIDUE_WITHOUT_SULFUR
        return (self.residues[first], self.residues[second])


@dataclass(frozen=True)
class _RecordPairs:
    """The residue pairs deposited records give, and the records that give none."""

    pairs: frozenset[_ResiduePair]
    unmatched_record_count: int
    warnings: tuple[str, ...]


def _record_pairs(
    records: Iterable[StructConnRecord],
    models: Sequence[str],
    resolver: _RecordResolver,
    conformers: _Conformers,
) -> _RecordPairs:
    """Return the pairs each record gives in each model, and those that fail.

    A record that gives a pair in no model is unmatched, and its warning
    names it with the reason it failed in the first model. Reversed and
    repeated records give the same pair, which is kept once.
    """
    pairs: set[_ResiduePair] = set()
    unmatched_record_count = 0
    warnings: list[str] = []
    for record in records:
        reasons: list[str] = []
        for model in models:
            resolved = resolver.resolve(record, model)
            if isinstance(resolved, str):
                reasons.append(resolved)
            else:
                pairs.add(_ordered_pair(*resolved, conformers))
        if len(reasons) == len(models):
            unmatched_record_count += 1
            reason = reasons[0] if reasons else _RESIDUE_ABSENT
            warnings.append(
                f"_struct_conn record {record.record_id!r} {reason}, so it gives "
                "no row."
            )
    return _RecordPairs(frozenset(pairs), unmatched_record_count, tuple(warnings))


def _distance_pairs(
    residues: Mapping[_ResidueKey, _Residue], conformers: _Conformers
) -> frozenset[_ResiduePair]:
    """Return the residue pairs whose SG conformers lie within the search limit.

    Only usable conformers of different residues of one model that can
    coexist are paired. The KD-tree finds the candidates, and the distance
    rounded to _DISTANCE_COMPARISON_DECIMALS decides, with the limit
    included. There is no lower limit.
    """
    rows_by_model: dict[str, list[int]] = {}
    for residue in residues.values():
        rows_by_model.setdefault(residue.key[0], []).extend(residue.usable_sulfur_rows)
    pairs: set[_ResiduePair] = set()
    for rows in rows_by_model.values():
        if len(rows) < 2:
            continue
        tree = cKDTree(
            np.asarray([conformers.coordinates[row] for row in rows], dtype=float)
        )
        candidates: npt.NDArray[np.intp] = tree.query_pairs(
            r=DISULFIDE_SEARCH_CUTOFF_A + _SEARCH_MARGIN_A, output_type="ndarray"
        )
        positions: list[list[int]] = candidates.tolist()
        for first_position, second_position in positions:
            row, other = rows[first_position], rows[second_position]
            first, second = conformers.residues[row], conformers.residues[other]
            if first == second or not _compatible(
                conformers.altlocs[row], conformers.altlocs[other]
            ):
                continue
            distance = round(
                math.dist(conformers.coordinates[row], conformers.coordinates[other]),
                _DISTANCE_COMPARISON_DECIMALS,
            )
            if distance <= DISULFIDE_SEARCH_CUTOFF_A:
                pairs.add(_ordered_pair(residues[first], residues[second], conformers))
    return frozenset(pairs)


def _metal_contacts(
    rows: Iterable[int], conformers: _Conformers, metals: _Metals
) -> dict[int, frozenset[int]]:
    """Return the metal atoms each of the given SG rows is bound to.

    An SG is bound to a metal atom of its own model that it can coexist with
    and that lies within METAL_CONTACT_DISTANCE_A. The KD-tree finds the
    candidates, and the distance rounded to _DISTANCE_COMPARISON_DECIMALS
    decides, with the limit included. A row bound to no metal is left out.
    """
    metals_by_model: dict[str, list[int]] = {}
    for metal, model in enumerate(metals.models):
        metals_by_model.setdefault(model, []).append(metal)
    rows_by_model: dict[str, list[int]] = {}
    for row in rows:
        rows_by_model.setdefault(conformers.residues[row][0], []).append(row)
    contacts: dict[int, frozenset[int]] = {}
    for model, model_rows in rows_by_model.items():
        model_metals = metals_by_model.get(model)
        if not model_metals:
            continue
        tree = cKDTree(
            np.asarray(
                [metals.coordinates[metal] for metal in model_metals], dtype=float
            )
        )
        for row in model_rows:
            point = conformers.coordinates[row]
            positions = tree.query_ball_point(
                point, r=METAL_CONTACT_DISTANCE_A + _SEARCH_MARGIN_A
            )
            bound = frozenset(
                metal
                for metal in (model_metals[position] for position in positions)
                if _compatible(conformers.altlocs[row], metals.altlocs[metal])
                and round(
                    math.dist(point, metals.coordinates[metal]),
                    _DISTANCE_COMPARISON_DECIMALS,
                )
                <= METAL_CONTACT_DISTANCE_A
            )
            if bound:
                contacts[row] = bound
    return contacts


def _metal_bridged(
    first: _Residue,
    second: _Residue,
    conformers: _Conformers,
    contacts: Mapping[int, frozenset[int]],
) -> bool:
    """Return whether an SG conformer of each residue is bound to one metal atom.

    The two conformers must be usable and able to coexist. ``contacts`` holds
    the metal atoms each is bound to, as _metal_contacts returns them.
    """
    return any(
        _compatible(conformers.altlocs[row], conformers.altlocs[other])
        and not contacts[row].isdisjoint(contacts[other])
        for row in first.usable_sulfur_rows
        if row in contacts
        for other in second.usable_sulfur_rows
        if other in contacts
    )


def _b_factor_ratio(residue: _Residue, conformers: _Conformers) -> float:
    """Return a residue's representative SG/CB B-factor ratio, or NaN.

    The ratio is NaN without a representative CB, and when either B-factor
    is missing, not finite, or zero or below. It is also NaN when either
    atom's occupancy is zero or below, since such an atom was placed without
    being seen in the data and its B-factor is not a measurement.
    """
    if residue.representative_carbon is None:
        return math.nan
    atoms = (residue.representative_sulfur, residue.representative_carbon)
    if any(conformers.occupancies[row] <= 0 for row in atoms):
        return math.nan
    sulfur, carbon = (conformers.b_factors[row] for row in atoms)
    if (
        not (math.isfinite(sulfur) and math.isfinite(carbon))
        or min(sulfur, carbon) <= 0
    ):
        return math.nan
    return sulfur / carbon


def _occupancy_deficit(residue: _Residue, conformers: _Conformers) -> float:
    """Return a residue's CB occupancy sum less its SG occupancy sum, or NaN.

    The deficit is NaN when the residue has no CB row or any of its SG or CB
    occupancies is missing, and it is rounded as the occupancy sums are.
    """
    if not residue.carbon_rows:
        return math.nan
    deficit = sum(conformers.occupancies[row] for row in residue.carbon_rows) - sum(
        conformers.occupancies[row] for row in residue.sulfur_rows
    )
    return round(deficit, OCCUPANCY_SUM_DECIMALS)


def _has_alternate_conformers(residue: _Residue, conformers: _Conformers) -> bool:
    """Return whether the residue's SG rows carry two or more distinct labels."""
    labels = {conformers.altlocs[row] for row in residue.sulfur_rows}
    return len(labels - {""}) >= 2


def _pair_measures(
    first: _Residue, second: _Residue, conformers: _Conformers
) -> dict[str, object]:
    """Return the measure columns of one pair, from its distances onward."""
    distances = [
        math.dist(conformers.coordinates[row], conformers.coordinates[other])
        for row in first.usable_sulfur_rows
        for other in second.usable_sulfur_rows
        if _compatible(conformers.altlocs[row], conformers.altlocs[other])
    ]
    occupancy_sums = [
        sum(conformers.occupancies[row] for row in residue.sulfur_rows)
        for residue in (first, second)
    ]
    occupancies = [
        conformers.occupancies[row]
        for residue in (first, second)
        for row in residue.sulfur_rows
    ]
    occupancy_missing = any(math.isnan(occupancy) for occupancy in occupancies)
    deficits = [_occupancy_deficit(residue, conformers) for residue in (first, second)]
    carbon_occupancy_missing = any(
        math.isnan(conformers.occupancies[row])
        for residue in (first, second)
        for row in residue.carbon_rows
    )
    ratios = [_b_factor_ratio(residue, conformers) for residue in (first, second)]
    ratio_missing = any(math.isnan(ratio) for ratio in ratios)

    if not (first.usable_sulfur_rows and second.usable_sulfur_rows):
        status = PAIR_STATUS_NO_USABLE_SG
    elif not distances:
        status = PAIR_STATUS_NO_COMPATIBLE_CONFORMERS
    elif occupancy_missing or carbon_occupancy_missing:
        status = PAIR_STATUS_OCCUPANCY_MISSING
    elif first.representative_carbon is None or second.representative_carbon is None:
        status = PAIR_STATUS_CB_MISSING
    elif ratio_missing:
        status = PAIR_STATUS_B_FACTOR_UNAVAILABLE
    elif not (
        first.sulfur_in_atom_table(conformers)
        and second.sulfur_in_atom_table(conformers)
    ):
        status = PAIR_STATUS_SG_NOT_IN_ATOM_TABLE
    else:
        status = PAIR_STATUS_OK
    return {
        "ss_distance_min_A": min(distances, default=math.nan),
        "ss_distance_max_A": max(distances, default=math.nan),
        "sg_conformer_occupancy_min": (
            math.nan if occupancy_missing else min(occupancies)
        ),
        "sg_total_occupancy_min": (
            math.nan
            if occupancy_missing
            else min(round(total, OCCUPANCY_SUM_DECIMALS) for total in occupancy_sums)
        ),
        "sg_occupancy_deficit_max": (
            math.nan
            if any(math.isnan(deficit) for deficit in deficits)
            else max(deficits)
        ),
        "sg_has_alternate_conformers": _has_alternate_conformers(first, conformers)
        or _has_alternate_conformers(second, conformers),
        "sg_cb_b_factor_ratio_min": math.nan if ratio_missing else min(ratios),
        "sg_cb_b_factor_ratio_max": math.nan if ratio_missing else max(ratios),
        "pair_status": status,
    }


def _fallback_warning(residue: _Residue, conformers: _Conformers) -> str:
    """Return the warning for a residue whose kept conformer has no SG."""
    model, chain_id, res_num, ins_code = residue.key
    in_model = f" in model {model}" if model else ""
    altloc = conformers.altlocs[residue.representative_sulfur]
    return (
        f"cysteine {chain_id} {res_num}{ins_code}{in_model} has no SG in the "
        f"conformer the atom table keeps, so its rows name SG conformer {altloc} "
        "and no atom links to it."
    )


def _table(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Return the rows as the disulfides table, in its declared columns.

    The host casts each column to its declared type when it validates the
    table, as it does every engine's.
    """
    if not rows:
        return DISULFIDES_TABLE.empty_dataframe()
    return pd.DataFrame(rows, columns=list(DISULFIDES_TABLE.column_names))


def compute_structure_disulfides(
    atom_dataframe: pd.DataFrame,
    mmcif_dict: Mapping[str, Sequence[str | None]],
    *,
    pdb_id: str,
) -> DisulfideComputation:
    """Return the disulfides table and atom link column of one structure.

    ``atom_dataframe`` is the structure's parsed atom table and
    ``mmcif_dict`` the flat dictionary of the file it was parsed from, as
    read_mmcif_as_flat_dict reads it. ``pdb_id`` names the entry, as the
    atom table's pdb_id column does. A structure without cysteine SG atoms
    gives an empty table and an ok status. A cysteine with a CB but no SG
    row is counted, once per model, rather than paired.

    Raises ValueError when two cysteine SG or CB rows share a site_id, as
    two cysteines of a chain without residue numbers do in a file that
    numbers its other residues, and RuntimeError when the file does not match
    the atom table.
    """
    atom_site = _atom_site_items(mmcif_dict)
    conformers = _read_conformers(atom_dataframe, atom_site, pdb_id)
    residues, without_sulfur = _build_residues(conformers)
    models = list(dict.fromkeys(model for model, *_ in conformers.residues))
    deposited = read_disulfide_records(mmcif_dict)
    by_record = _record_pairs(
        deposited.records,
        models,
        _RecordResolver(residues, without_sulfur, conformers),
        conformers,
    )
    by_distance = _distance_pairs(residues, conformers)
    pairs = by_record.pairs | by_distance
    # Only the sulfurs of the pairs are looked up, so a structure with many
    # metals and few candidate pairs costs little.
    metal_contacts = _metal_contacts(
        {
            row
            for pair in pairs
            for key in pair
            for row in residues[key].usable_sulfur_rows
        },
        conformers,
        _read_metals(atom_site),
    )

    def pair_order(
        pair: _ResiduePair,
    ) -> tuple[tuple[int, int], _ResidueSortKey, _ResidueSortKey]:
        first, second = pair
        return (
            _model_order(first[0]),
            _sort_key(residues[first], conformers),
            _sort_key(residues[second], conformers),
        )

    rows: list[dict[str, object]] = []
    link = pd.Series(pd.NA, index=atom_dataframe.index, dtype=object)
    fallback_residues: dict[_ResidueKey, _Residue] = {}
    for pair in sorted(pairs, key=pair_order):
        first, second = residues[pair[0]], residues[pair[1]]
        partner_keys = conformers.site_keys[second.representative_sulfur]
        rows.append(
            {
                **conformers.site_keys[first.representative_sulfur],
                **{
                    f"{PARTNER_PREFIX}{name}": value
                    for name, value in partner_keys.items()
                },
                "detection_method": detection_method(
                    by_record=pair in by_record.pairs, by_distance=pair in by_distance
                ),
                "metal_bridged": _metal_bridged(
                    first, second, conformers, metal_contacts
                ),
                **_pair_measures(first, second, conformers),
            }
        )
        for residue in (first, second):
            label = conformers.atom_labels[residue.representative_sulfur]
            if label is None:
                fallback_residues[residue.key] = residue
            else:
                link.at[label] = conformers.site_ids[residue.representative_sulfur]

    # Each warning once, so the run's warnings and the report's message agree.
    warnings = tuple(
        dict.fromkeys(
            (
                *by_record.warnings,
                *(
                    _fallback_warning(residue, conformers)
                    for residue in fallback_residues.values()
                ),
            )
        )
    )
    return DisulfideComputation(
        table=_table(rows),
        link=link,
        status=(
            DISULFIDE_STATUS_DEGRADED
            if by_record.unmatched_record_count
            else DISULFIDE_STATUS_OK
        ),
        unmatched_record_count=by_record.unmatched_record_count,
        symmetry_record_count=deposited.symmetry_record_count,
        cysteine_without_sg_count=len(without_sulfur),
        warnings=warnings,
    )


__all__ = ["DisulfideComputation", "compute_structure_disulfides"]
