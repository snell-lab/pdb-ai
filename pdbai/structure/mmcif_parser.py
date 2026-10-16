"""Reading an mmCIF file into one row per atom, with its entry's values.

parse_mmcif_atom_rows reads the single data block of an mmCIF file
with gemmi. It returns one row per atom, for every model in the file.
mmcif_fields.py turns the _atom_site items, and the items of categories that
describe the whole entry, into columns. Each column keeps the text the file
writes, so the deposited digits survive. coordinate_preprocessing.py then
reduces each residue to one conformer. read_mmcif_as_flat_dict is the reader
beneath it, and an engine that needs every conformer or a category of several
rows reads the file through it. Numbers are read by the grammar of
cif_numbers.py, which such an engine shares.

The parser also writes named columns beside the raw items. x, y, z, and
occupancy are numbers, and model_num is a nullable whole number. chain_id,
res_num, and res_name come from the author items, because papers and the PDB
format number residues that way. A file that gives no author chain or no
author residue number gets both from the label items instead, and
residue_numbering says which of the two numberings the structure uses. A
row without an author residue name takes its label one.
atom_site_residue_identity of residue_identity.py is that rule, and the
parse result warns when it chose label numbering. atom_name comes from the
label atom name. is_hetatm, is_backbone, and is_sidechain classify each atom.
residue_mean_bfactor averages the B-factors of each model-aware residue. The
anisotropic displacement tensor is joined to its atom, with its isotropic
equivalent and its anisotropy.

The entry-level columns are the same on every row. Resolution, unit-cell
volume, and solvent content come from the file. The RCSB record's resolution
is used only when the file reports none. The RCSB record's other values are
repeated under their own names. Experimental methods and radiation sources
come from the RCSB record, or from the file when the record lists none. An
entry-level item of a category with several rows is read from the first row,
as the RCSB record reads it. The two _refine rows of a joint X-ray and
neutron refinement are one example, and either method may come first. So
the resolution, r_work, and r_free columns suffixed _xray and _neutron hold
each method's own refinement row, whatever the order of the rows. How the
entry was refined comes from the file too, as refinement.py reads it: the
refinement programs, the number of TLS groups, the archive's note on
B-factors that leave out the TLS contribution, and the share of atom sites
with an anisotropic record.

A value that is not reported stays missing, never zero or False, so that it
is not mistaken for a measurement. A value that is reported but cannot be
read is refused rather than dropped. The refusals are a number that is not
one, a record type other than ATOM or HETATM, and an identifier given to two
rows of a category the parser looks values up in. A file without the atom
ids, record types, or coordinates that every row needs is refused too. The
parser raises RuntimeError naming the entry for such a file, as for a file it
cannot read at all. Its caller then fails that entry alone.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import gemmi
import numpy as np
import pandas as pd

from ..engines.core.techniques import classify_methods, normalize_source
from ..json_values import str_list
from ..refused_values import listed_values
from .cif_numbers import CIF_NUMBER, coerce_cif_numbers
from .coordinate_preprocessing import highest_occupancy_conformer_mask
from .mmcif_fields import calculate_unit_cell_volume, expand_mmcif_atom_fields
from .refinement import read_refinement_fields
from .residue_identity import (
    RESIDUE_NUMBERING_COLUMN,
    RESIDUE_NUMBERING_LABEL,
    ResidueIdentities,
    atom_site_residue_identity,
    atom_site_residue_numbering,
    build_residue_identities,
    hetatm_record_flags,
    text_series,
)

# The polypeptide backbone atom names. A non-HETATM atom with one of these
# names is backbone. Any other non-HETATM atom is side chain, whatever its
# residue. A HETATM atom is neither.
BACKBONE_ATOM_NAMES: tuple[str, ...] = ("N", "CA", "C", "O", "OXT")

# The radiation dose in MGy above which an entry is flagged as high.
HIGH_RADIATION_DOSE_MGY = 5.0

# Matthews (1968) gives the protein-occupied fraction of a crystal as
# 1.23 / Vm, with Vm in Å³ per dalton. 110 Da is the customary mean residue
# mass.
_MATTHEWS_PROTEIN_FRACTION_CONSTANT = 1.23
MEAN_RESIDUE_MASS_DA = 110.0

# A file read flat, from each item name to its values, one per row of its
# category. A value is None where the file writes a placeholder. It is the
# empty text where the file writes a quoted empty value. No number or
# identifier can be read from the empty text, so both read as missing.
_MmcifDict = Mapping[str, Sequence[str | None]]

# The _atom_site items every atom row needs. They are the id that anisotropic
# tensors and every engine join on, the record type that says whether an atom
# is HETATM, and the coordinates.
_REQUIRED_ATOM_SITE_ITEMS = (
    "_atom_site.id",
    "_atom_site.group_PDB",
    "_atom_site.Cartn_x",
    "_atom_site.Cartn_y",
    "_atom_site.Cartn_z",
)

# The resolution items read from the file, in order of preference. The
# refinement resolution comes first, as in the RCSB record and in RABDAM's own
# reading of the file. The data-collection limit can extend beyond the data
# the model was refined against, so it is used only when there is no
# refinement resolution.
_RESOLUTION_ITEMS = (
    "_refine.ls_d_res_high",
    "_refine_hist.d_res_high",
    "_reflns.d_resolution_high",
)

# The _refine items read for each diffraction method, with the suffix that
# names the method's column, by the technique identifier of the method that
# _refine.pdbx_refine_id names. A joint X-ray/neutron entry has one _refine
# row for each method, in either order.
_METHOD_COLUMN_SUFFIXES = {
    "x_ray_diffraction": "xray",
    "neutron_diffraction": "neutron",
}
_METHOD_REFINEMENT_ITEMS = (
    ("resolution", "_refine.ls_d_res_high"),
    ("r_work", "_refine.ls_R_factor_R_work"),
    ("r_free", "_refine.ls_R_factor_R_free"),
)

# The column of each anisotropic tensor component and the item it is read from,
# diagonal components first.
_ANISOTROPIC_ITEMS = {
    "u11": "_atom_site_anisotrop.U[1][1]",
    "u22": "_atom_site_anisotrop.U[2][2]",
    "u33": "_atom_site_anisotrop.U[3][3]",
    "u12": "_atom_site_anisotrop.U[1][2]",
    "u13": "_atom_site_anisotrop.U[1][3]",
    "u23": "_atom_site_anisotrop.U[2][3]",
}

# The positions in (u11, u22, u33, u12, u13, u23) of the nine elements of the
# symmetric tensor, row by row.
_TENSOR_ELEMENT_POSITIONS = [0, 3, 4, 3, 1, 5, 4, 5, 2]

# The RCSB entry record's values repeated on every atom row, each under its own
# name except those renamed below.
_RCSB_METADATA_KEYS = (
    "title",
    "keywords",
    "beamline",
    "wavelength",
    "radiation_dose_MGy",
    "temperature_K",
    "crystal_pH",
    "crystallization_method",
    "num_crystals",
    "sample_delivery_method",
    "xfel_pulse_duration_fs",
    "r_work",
    "r_free",
    "space_group",
    "reflns_completeness",
    "cc_half",
    "i_over_sigma",
    "wilson_b",
    "multiplicity",
    "first_radiation_source",
    "synchrotron_site",
    "detector_type",
    "collection_year",
    "deposition_year",
    "citation_title",
    "citation_journal_abbrev",
    "citation_year",
    "citation_doi",
    "citation_pubmed_id",
    "citation_authors",
    "deposit_authors",
)
_RENAMED_RCSB_METADATA = {"wavelength": "radiation_wavelength"}

# The kinds of value one of those columns holds.
type _MetadataScalar = str | bool | int | float | None


# gemmi's stubs leave these return types partly unknown. One cast to this
# protocol states what the reader relies on, instead of a cast at every call.
class _MmcifBlock(Protocol):
    def get_mmcif_category_names(self) -> list[str]: ...

    def get_mmcif_category(
        self, name: str, raw: bool = False
    ) -> dict[str, list[str]]: ...


@dataclass(frozen=True)
class AtomSiteParseResult:
    """A structure file's atom rows and what the file says of its experiment.

    The DataFrame is empty when the file has no atoms. The experiment fields
    are the file's own, whatever the RCSB record says.
    """

    dataframe: pd.DataFrame
    # Each distinct _exptl.method value in the file, in file order. A joint
    # X-ray and neutron entry reports both.
    experimental_methods: tuple[str, ...] = ()
    # True when the file has an _ihm_ category, which marks an integrative
    # model. Such a model has no experimental method of its own, so this
    # stands in for one.
    is_integrative: bool = False
    # Each distinct _diffrn_source.source value in the file, in file order.
    # Whitespace is collapsed, but the spelling is not made canonical.
    radiation_sources: tuple[str, ...] = ()
    # What the file itself says is wrong with it, such as an entry the
    # archive made obsolete or residues it lists but holds no atoms for.
    warnings: tuple[str, ...] = ()


def read_mmcif_as_flat_dict(cif_file_path: Path) -> dict[str, list[str | None]]:
    """Return every item of the file's single data block with its values.

    Quoting and surrounding whitespace are removed. The bare placeholders "?"
    and "." become None. A quoted '?' or '.' in the file stays text, as it is
    in CIF. Gemmi's reader is described at
    https://gemmi.readthedocs.io/en/latest/cif.html.

    The exceptions gemmi raises for a file it cannot read, of several types,
    pass through unchanged and do not name the entry.
    """
    doc = gemmi.cif.read(str(cif_file_path))
    block = cast(_MmcifBlock, doc.sole_block())
    flat: dict[str, list[str | None]] = {}
    for category in block.get_mmcif_category_names():
        raw_values = block.get_mmcif_category(category, raw=True)
        for suffix, values in raw_values.items():
            flat[f"{category}{suffix}"] = [
                None if gemmi.cif.is_null(value) else gemmi.cif.as_string(value).strip()
                for value in values
            ]
    return flat


def _cif_number(text: str, item: str) -> float:
    """Return a CIF number's value, without its standard uncertainty.

    Raises ValueError naming the item when the text is not a finite number.
    """
    match = CIF_NUMBER.fullmatch(text)
    number = float(match.group("value")) if match else math.nan
    if not math.isfinite(number):
        raise ValueError(f"mmCIF item {item} is {text!r}, which is not a number.")
    return number


def _first_value(mmcif_dict: _MmcifDict, item: str) -> str | None:
    """Return an item's value in the first row of its category, or None.

    A placeholder and the empty text both read as None.
    """
    values = mmcif_dict.get(item)
    first = values[0] if values else None
    return None if first == "" else first


def _item_number(mmcif_dict: _MmcifDict, item: str) -> float | None:
    """Return an item's number in its category's first row, None when missing.

    Raises ValueError naming the item when its value is not a number.
    """
    text = _first_value(mmcif_dict, item)
    return None if text is None else _cif_number(text, item)


def _lookup(mmcif_dict: _MmcifDict, key_item: str, value_item: str) -> dict[str, str]:
    """Return each row's value of one item by its value of another.

    A row whose key is missing is left out, and a missing value becomes the
    empty text. Raises ValueError naming the key item when two rows share a
    key, since either value would be a guess.
    """
    keys = mmcif_dict.get(key_item)
    values = mmcif_dict.get(value_item)
    if not keys or not values:
        return {}
    lookup: dict[str, str] = {}
    for key, value in zip(keys, values, strict=True):
        if key is None or key == "":
            continue
        if key in lookup:
            raise ValueError(f"mmCIF item {key_item} names {key!r} on two rows.")
        lookup[key] = value or ""
    return lookup


def _distinct_texts(values: Iterable[str | None]) -> tuple[str, ...]:
    """Return the values with whitespace collapsed, once each, in order.

    Missing and empty values are left out.
    """
    texts = (" ".join(value.split()) for value in values if value is not None)
    return tuple(dict.fromkeys(text for text in texts if text))


def _cif_numbers(values: pd.Series, item: str) -> pd.Series:
    """Return an item's values as floats, NaN where missing.

    Raises ValueError naming the item and the values that are not numbers.
    """
    present = values.notna() & (values != "")
    numbers = coerce_cif_numbers(values)
    refused = present & numbers.isna()
    if refused.any():
        raise ValueError(
            f"mmCIF item {item} holds values that are not numbers: "
            f"{listed_values(values[refused])}."
        )
    return numbers


def _optional_numbers(dataframe: pd.DataFrame, item: str) -> pd.Series:
    """Return an atom item's values as floats, all NaN when it is absent.

    Raises ValueError naming the item and the values that are not numbers.
    """
    if item not in dataframe.columns:
        return pd.Series(np.nan, index=dataframe.index, dtype=float)
    return _cif_numbers(dataframe[item], item)


def _model_numbers(dataframe: pd.DataFrame) -> pd.Series:
    """Return each atom's model number as a nullable integer, NA when missing.

    Raises ValueError naming _atom_site.pdbx_PDB_model_num when a value is not
    a number or not a whole one.
    """
    item = "_atom_site.pdbx_PDB_model_num"
    numbers = _optional_numbers(dataframe, item)
    fractional = numbers.notna() & (numbers != np.floor(numbers))
    if fractional.any():
        raise ValueError(
            f"mmCIF item {item} holds values that are not whole numbers: "
            f"{listed_values(dataframe.loc[fractional, item])}."
        )
    return numbers.astype("Int64")


def _anisotropic_columns(mmcif_dict: _MmcifDict, atom_ids: pd.Series) -> pd.DataFrame:
    """Return each atom's anisotropic tensor, U equivalent, and anisotropy.

    The tensor rows are joined to the atoms by _atom_site.id. A tensor id
    given to two rows is joined to no atom, since choosing one of the two
    tensors would be arbitrary. The anisotropy is the ratio of the largest
    to the smallest eigenvalue. It is missing where a component is missing,
    or where the tensor is not positive definite.
    """
    tensor_ids = pd.Index(
        np.array(mmcif_dict.get("_atom_site_anisotrop.id", []), dtype=object)
    )
    tensors = pd.DataFrame(
        {
            column: _cif_numbers(pd.Series(mmcif_dict[item], index=tensor_ids), item)
            for column, item in _ANISOTROPIC_ITEMS.items()
            if item in mmcif_dict
        },
        index=tensor_ids,
        columns=list(_ANISOTROPIC_ITEMS),
        dtype=float,
    )
    tensors = tensors[~tensors.index.duplicated(keep=False)]

    columns = pd.DataFrame(
        {column: atom_ids.map(tensors[column]) for column in _ANISOTROPIC_ITEMS},
        index=atom_ids.index,
        dtype=float,
    )
    columns["u_equivalent"] = (columns["u11"] + columns["u22"] + columns["u33"]) / 3.0

    anisotropy = np.full(len(columns), np.nan)
    complete = columns[list(_ANISOTROPIC_ITEMS)].notna().all(axis=1).to_numpy()
    if complete.any():
        components = columns[list(_ANISOTROPIC_ITEMS)].to_numpy()[complete]
        matrices = components[:, _TENSOR_ELEMENT_POSITIONS].reshape(-1, 3, 3)
        eigenvalues = np.linalg.eigvalsh(matrices)
        smallest, largest = eigenvalues[:, 0], eigenvalues[:, 2]
        anisotropy[complete] = np.divide(
            largest, smallest, out=np.full(len(smallest), np.nan), where=smallest > 0
        )
    columns["b_anisotropy"] = anisotropy
    return columns


def _file_resolution(mmcif_dict: _MmcifDict) -> float:
    """Return the resolution the file reports, or NaN when it reports none.

    A resolution that is not positive counts as not reported. The
    _reflns_shell loop is read last. It gives the smallest limit of its
    shells, which is the limit of the whole data set in whatever order the
    shells are listed.
    """
    for item in _RESOLUTION_ITEMS:
        resolution = _item_number(mmcif_dict, item)
        if resolution is not None and resolution > 0:
            return resolution
    shell_item = "_reflns_shell.d_res_high"
    shell_limits = [
        _cif_number(text, shell_item)
        for text in mmcif_dict.get(shell_item, [])
        if text is not None and text != ""
    ]
    positive = [limit for limit in shell_limits if limit > 0]
    return min(positive) if positive else math.nan


def _refinement_methods(
    mmcif_dict: _MmcifDict, methods: Iterable[str]
) -> list[tuple[str, ...]]:
    """Return the technique identifiers each _refine row belongs to.

    _refine.pdbx_refine_id names the row's method. A single row that names
    no method is taken to belong to the entry's one method, when ``methods``
    lists exactly one. Any other unnamed row belongs to no method.
    """
    row_count = max(
        (len(mmcif_dict.get(item, [])) for _, item in _METHOD_REFINEMENT_ITEMS),
        default=0,
    )
    refine_ids = mmcif_dict.get("_refine.pdbx_refine_id", [])
    entry_methods = _distinct_texts(methods)
    named: list[tuple[str, ...]] = []
    for row in range(row_count):
        refine_id = refine_ids[row] if row < len(refine_ids) else None
        if refine_id:
            named.append(classify_methods([refine_id]))
        elif row_count == 1 and len(entry_methods) == 1:
            named.append(classify_methods(entry_methods))
        else:
            named.append(())
    return named


def _method_refinement_values(
    mmcif_dict: _MmcifDict, methods: Iterable[str]
) -> dict[str, float]:
    """Return each method's refinement resolution and R factors, by column.

    The columns are resolution_xray, r_work_xray, r_free_xray, and the same
    three for neutron. Each holds the value of the first _refine row of its
    method, as _refinement_methods assigns rows. A resolution that is not
    positive counts as not reported. A method the file has no row for, and a
    value its row leaves out, give NaN.

    Raises ValueError naming the item when a value is not a number.
    """
    values = {
        f"{prefix}_{suffix}": math.nan
        for prefix, _ in _METHOD_REFINEMENT_ITEMS
        for suffix in _METHOD_COLUMN_SUFFIXES.values()
    }
    for row, technique_ids in enumerate(_refinement_methods(mmcif_dict, methods)):
        suffixes = [
            _METHOD_COLUMN_SUFFIXES[technique_id]
            for technique_id in technique_ids
            if technique_id in _METHOD_COLUMN_SUFFIXES
        ]
        for prefix, item in _METHOD_REFINEMENT_ITEMS:
            texts = mmcif_dict.get(item, [])
            text = texts[row] if row < len(texts) else None
            if text is None or text == "":
                continue
            value = _cif_number(text, item)
            if prefix == "resolution" and value <= 0:
                continue
            for suffix in suffixes:
                column = f"{prefix}_{suffix}"
                if math.isnan(values[column]):
                    values[column] = value
    return values


def _symmetry_operation_count(mmcif_dict: _MmcifDict) -> int | None:
    """Return the number of symmetry operations in the unit cell, or None.

    The count includes the centring translations, since the Matthews
    coefficient divides the whole cell among the copies of the asymmetric
    unit. The cell angles choose between the rhombohedral and hexagonal
    settings of an R space group, which the PDB writes under the same name.
    _cell.Z_PDB is deliberately not used. It counts polymer chains per unit
    cell, so it would count the chains of the asymmetric unit twice against
    the residue total.
    """
    alpha = _item_number(mmcif_dict, "_cell.angle_alpha") or 0.0
    gamma = _item_number(mmcif_dict, "_cell.angle_gamma") or 0.0
    for item in (
        "_symmetry.space_group_name_H-M",
        "_symmetry.pdbx_full_space_group_name_H-M",
    ):
        name = _first_value(mmcif_dict, item)
        if name is None:
            continue
        # gemmi returns None for an unknown name, although its stubs say not.
        space_group = cast(
            "gemmi.SpaceGroup | None",
            gemmi.find_spacegroup_by_name(name, alpha, gamma),
        )
        if space_group is not None:
            return len(space_group.operations())
    number = _item_number(mmcif_dict, "_symmetry.Int_Tables_number")
    if number is None:
        return None
    if not number.is_integer():
        raise ValueError(
            f"mmCIF item _symmetry.Int_Tables_number is {number}, which is not a "
            f"space-group number."
        )
    # gemmi reads 0 as P 1, so only the 230 numbers are looked up.
    if not 1 <= number <= 230:
        return None
    space_group = cast(
        "gemmi.SpaceGroup | None", gemmi.find_spacegroup_by_number(int(number))
    )
    return None if space_group is None else len(space_group.operations())


def _estimated_matthews_coefficient(
    mmcif_dict: _MmcifDict,
    unit_cell_volume: float,
    identities: ResidueIdentities,
) -> float | None:
    """Return the Matthews coefficient implied by the cell and model, or None.

    The asymmetric unit is taken to hold the polymer residues of the first
    model at the mean residue mass each. None is returned without a cell, a
    known space group, or a polymer residue.
    """
    if not unit_cell_volume > 0:
        return None
    symmetry_count = _symmetry_operation_count(mmcif_dict)
    if symmetry_count is None:
        return None
    polymer = identities.polymer
    if not polymer.any():
        return None
    first_model = str(identities.model[polymer].iloc[0])
    first_model_polymer = polymer & identities.model.eq(first_model)
    residue_count = int(identities.residue_join_keys[first_model_polymer].nunique())
    if residue_count == 0:
        return None
    return unit_cell_volume / (symmetry_count * residue_count * MEAN_RESIDUE_MASS_DA)


def _solvent_fraction_from_matthews(coefficient: float | None) -> float | None:
    """Return the solvent fraction 1 - 1.23 / Vm, or None outside (0, 1)."""
    # A coefficient above the constant is what gives a fraction in (0, 1).
    if coefficient is None or coefficient <= _MATTHEWS_PROTEIN_FRACTION_CONSTANT:
        return None
    return 1.0 - _MATTHEWS_PROTEIN_FRACTION_CONSTANT / coefficient


def _solvent_content(
    mmcif_dict: _MmcifDict,
    unit_cell_volume: float,
    identities: ResidueIdentities,
) -> float:
    """Return the crystal's solvent-volume fraction, or NaN when none is known.

    The deposited _exptl_crystal.density_percent_sol comes first. Next is the
    fraction 1 - 1.23 / Vm from the deposited Matthews coefficient,
    _exptl_crystal.density_Matthews. Last is the same fraction from a
    coefficient estimated from the unit-cell volume, the space-group symmetry
    count, and the modelled polymer residues. A source that is missing or
    gives a fraction outside (0, 1) passes to the next. When none gives one,
    the value stays missing rather than clipped. A clipped 0.0 or 1.0 would
    read as a real value.
    """
    percent = _item_number(mmcif_dict, "_exptl_crystal.density_percent_sol")
    if percent is not None and 0.0 < percent < 100.0:
        return percent / 100.0
    fraction = _solvent_fraction_from_matthews(
        _item_number(mmcif_dict, "_exptl_crystal.density_Matthews")
    )
    if fraction is None:
        fraction = _solvent_fraction_from_matthews(
            _estimated_matthews_coefficient(mmcif_dict, unit_cell_volume, identities)
        )
    return math.nan if fraction is None else fraction


def _metadata_number(metadata: Mapping[str, object], key: str) -> float | None:
    """Return a numeric value of the RCSB metadata, None when missing.

    Raises ValueError naming the key when the value is not a finite number.
    """
    value = metadata.get(key)
    # NaN is how pandas marks a missing number.
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    number = math.nan
    if isinstance(value, (int, float, str)) and not isinstance(value, bool):
        with contextlib.suppress(ValueError):
            number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"RCSB metadata {key!r} is {value!r}, which is not a finite number."
        )
    return number


def _metadata_scalar(metadata: Mapping[str, object], key: str) -> _MetadataScalar:
    """Return an RCSB metadata value as a column holds it, None when missing.

    Text is stripped of surrounding whitespace. Raises ValueError naming the
    key when the value is a list or other container, which one cell cannot
    hold.
    """
    value = metadata.get(key)
    if isinstance(value, str):
        return value.strip()
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise ValueError(f"RCSB metadata {key!r} is {value!r}, which is not one value.")


def _threshold_flag(
    value: float | None, threshold: float, index: pd.Index
) -> pd.Series:
    """Return whether the value exceeds the threshold, missing without one."""
    if value is None:
        return pd.Series(pd.NA, index=index, dtype="boolean")
    return pd.Series(value > threshold, index=index, dtype="boolean")


def _atom_rows(
    mmcif_dict: _MmcifDict,
    rcsb_metadata: Mapping[str, object],
    pdb_id: str,
    file_methods: tuple[str, ...],
    file_sources: tuple[str, ...],
) -> pd.DataFrame:
    """Return the atom rows of a parsed file, empty when it has no atoms.

    Raises ValueError for a value the parser cannot read.
    """
    expanded_fields, atom_count = expand_mmcif_atom_fields(mmcif_dict)
    if atom_count == 0:
        return pd.DataFrame()
    dataframe = pd.DataFrame(expanded_fields)
    missing_items = [
        item for item in _REQUIRED_ATOM_SITE_ITEMS if item not in dataframe.columns
    ]
    if missing_items:
        raise ValueError(
            f"The _atom_site category has no {', '.join(missing_items)}, which "
            f"every atom row needs."
        )

    label_entity = text_series(dataframe, "_atom_site.label_entity_id")
    label_component = text_series(dataframe, "_atom_site.label_comp_id")
    entity_types = _lookup(mmcif_dict, "_entity.id", "_entity.type")
    polymer_types = _lookup(mmcif_dict, "_entity_poly.entity_id", "_entity_poly.type")
    component_types = _lookup(mmcif_dict, "_chem_comp.id", "_chem_comp.type")
    dataframe["_pdbai_entity_type"] = label_entity.map(entity_types).fillna("")
    dataframe["_pdbai_polymer_type"] = label_entity.map(polymer_types).fillna("")
    dataframe["_pdbai_chem_comp_type"] = label_component.map(component_types).fillna("")

    # Record types are checked on every row, before the conformer choice
    # drops any. A bad record type on a dropped conformer is still refused,
    # and the refusal is this parser's rather than the residue reader's.
    is_hetatm = hetatm_record_flags(dataframe)
    # The numbering is the file's, so it is decided on every row too. An
    # engine that reads the file again then decides as the parser did.
    numbering = atom_site_residue_numbering(dataframe)
    selected = highest_occupancy_conformer_mask(dataframe)
    if not selected.all():
        dataframe = dataframe.loc[selected].reset_index(drop=True)
        is_hetatm = is_hetatm.loc[selected].reset_index(drop=True)

    resolution = _file_resolution(mmcif_dict)
    if math.isnan(resolution):
        rcsb_resolution = _metadata_number(rcsb_metadata, "resolution_rcsb")
        if rcsb_resolution is not None and rcsb_resolution > 0:
            resolution = rcsb_resolution
    unit_cell_volume = calculate_unit_cell_volume(
        *(
            _item_number(mmcif_dict, f"_cell.{parameter}")
            for parameter in (
                "length_a",
                "length_b",
                "length_c",
                "angle_alpha",
                "angle_beta",
                "angle_gamma",
            )
        )
    )

    dataframe["pdb_id"] = pdb_id
    dataframe["resolution"] = resolution
    dataframe["unit_cell_volume"] = unit_cell_volume
    residue_identity = atom_site_residue_identity(dataframe, numbering=numbering)
    dataframe["chain_id"] = residue_identity.chain_id
    dataframe["res_num"] = residue_identity.res_num
    dataframe["res_name"] = residue_identity.res_name
    dataframe[RESIDUE_NUMBERING_COLUMN] = numbering
    dataframe["atom_name"] = text_series(dataframe, "_atom_site.label_atom_id")
    dataframe["element"] = dataframe.get("_atom_site.type_symbol")
    is_backbone_name = dataframe["atom_name"].isin(BACKBONE_ATOM_NAMES)
    dataframe["is_hetatm"] = is_hetatm
    dataframe["is_backbone"] = is_backbone_name & ~is_hetatm
    dataframe["is_sidechain"] = ~is_backbone_name & ~is_hetatm
    dataframe["occupancy"] = _optional_numbers(dataframe, "_atom_site.occupancy")
    dataframe["model_num"] = _model_numbers(dataframe)
    for column, item in (
        ("x", "_atom_site.Cartn_x"),
        ("y", "_atom_site.Cartn_y"),
        ("z", "_atom_site.Cartn_z"),
    ):
        dataframe[column] = _cif_numbers(dataframe[item], item)

    identities = build_residue_identities(dataframe)
    b_factors = _optional_numbers(dataframe, "_atom_site.B_iso_or_equiv")
    # Author residue numbers alone would merge NMR models and insertion codes.
    # The model-aware residue identity keeps them apart.
    dataframe["residue_mean_bfactor"] = b_factors.groupby(
        identities.residue_join_keys, sort=False, dropna=True
    ).transform("mean")

    metadata_columns = pd.DataFrame(
        {
            _RENAMED_RCSB_METADATA.get(key, key): _metadata_scalar(rcsb_metadata, key)
            for key in _RCSB_METADATA_KEYS
        },
        index=dataframe.index,
    )
    dataframe = pd.concat([dataframe, metadata_columns], axis=1)
    dataframe["is_high_radiation_dose"] = _threshold_flag(
        _metadata_number(rcsb_metadata, "radiation_dose_MGy"),
        HIGH_RADIATION_DOSE_MGY,
        dataframe.index,
    )
    # These two are kept missing, never false, when the record says nothing.
    for flag in ("is_serial_crystallography", "mentions_time_resolved"):
        value = rcsb_metadata.get(flag)
        dataframe[flag] = pd.Series(
            value if isinstance(value, bool) else pd.NA,
            index=dataframe.index,
            dtype="boolean",
        )

    # The RCSB entry record is authoritative when it lists any. An uploaded
    # file has only its own _exptl and _diffrn_source categories to go on.
    methods = (
        _distinct_texts(str_list(rcsb_metadata.get("experimental_methods")))
        or file_methods
    )
    sources = (
        _distinct_texts(str_list(rcsb_metadata.get("radiation_sources")))
        or file_sources
    )
    # Two spellings of one source, or one source per data set, are listed once.
    canonical_sources = dict.fromkeys(map(normalize_source, sources))
    for column, value in _method_refinement_values(mmcif_dict, methods).items():
        dataframe[column] = value
    dataframe["experimental_methods"] = "; ".join(methods) or None
    dataframe["radiation_sources"] = "; ".join(canonical_sources) or None
    dataframe["is_multi_method"] = pd.Series(
        len(methods) > 1 if methods else pd.NA, index=dataframe.index, dtype="boolean"
    )

    dataframe["solvent_content"] = _solvent_content(
        mmcif_dict, unit_cell_volume, identities
    )
    refinement = read_refinement_fields(mmcif_dict)
    dataframe["refinement_program"] = refinement.refinement_program
    dataframe["refinement_program_version"] = refinement.refinement_program_version
    dataframe["refinement_programs"] = refinement.refinement_programs
    # A count is a whole number, and a nullable one keeps a file without a
    # refinement missing instead of turning the column into floats.
    dataframe["tls_group_count"] = pd.Series(
        refinement.tls_group_count, index=dataframe.index, dtype="Int64"
    )
    dataframe["tls_residual_adp_flag"] = refinement.tls_residual_adp_flag
    dataframe["anisotropic_atom_fraction"] = pd.Series(
        refinement.anisotropic_atom_fraction, index=dataframe.index, dtype=float
    )
    anisotropic = _anisotropic_columns(mmcif_dict, dataframe["_atom_site.id"])
    return pd.concat([dataframe, anisotropic], axis=1)


def _obsolete_warning(mmcif_dict: _MmcifDict, pdb_id: str) -> str | None:
    """Return a warning when the file says the archive made the entry obsolete.

    The archive still serves an obsolete entry's file, and the file marks it
    with status code OBS and names the entries that replaced it.
    """
    status = _first_value(mmcif_dict, "_pdbx_database_status.status_code")
    if status is None or status.upper() != "OBS":
        return None
    replacements = " ".join(
        value
        for value in mmcif_dict.get("_pdbx_database_PDB_obs_spr.pdb_id", [])
        if value
    ).split()
    replaced_by = (
        f"; it was replaced by {', '.join(replacements)}" if replacements else ""
    )
    return f"{pdb_id} is obsolete in the PDB archive{replaced_by}."


def _truncation_warning(mmcif_dict: _MmcifDict, pdb_id: str) -> str | None:
    """Return a warning when the file lists residues it holds no atoms for.

    An archive file lists each modelled residue in _pdbx_poly_seq_scheme or
    _pdbx_nonpoly_scheme before its atom sites. A file cut short inside the
    atom sites still lists the residues it lost, so their count tells. A
    file without either scheme is not checked.
    """
    asym_ids = mmcif_dict.get("_atom_site.label_asym_id", [])
    # A polymer residue is numbered by its label position in the scheme, and
    # any other residue by its author number.
    polymer_atoms = set(
        zip(asym_ids, mmcif_dict.get("_atom_site.label_seq_id", []), strict=False)
    )
    other_atoms = set(
        zip(asym_ids, mmcif_dict.get("_atom_site.auth_seq_id", []), strict=False)
    )
    # A polymer residue the depositors did not model has no author residue
    # name, so only the modelled ones are expected to have atoms.
    listed_polymer = {
        (asym_id, seq_id)
        for asym_id, seq_id, modelled in zip(
            mmcif_dict.get("_pdbx_poly_seq_scheme.asym_id", []),
            mmcif_dict.get("_pdbx_poly_seq_scheme.seq_id", []),
            mmcif_dict.get("_pdbx_poly_seq_scheme.pdb_mon_id", []),
            strict=False,
        )
        if modelled is not None
    }
    listed_other = set(
        zip(
            mmcif_dict.get("_pdbx_nonpoly_scheme.asym_id", []),
            mmcif_dict.get("_pdbx_nonpoly_scheme.pdb_seq_num", []),
            strict=False,
        )
    )
    missing = len(listed_polymer - polymer_atoms) + len(listed_other - other_atoms)
    if not missing:
        return None
    return (
        f"{pdb_id}: the file lists {missing} modelled residue(s) that have no "
        "atom sites, so it may be truncated."
    )


def _label_numbering_warning(dataframe: pd.DataFrame, pdb_id: str) -> str | None:
    """Return a warning when the atom rows are named by their label items.

    Residue numbers a paper or a validation report quotes are author numbers,
    so a reader must know when the table's are not.
    """
    if (
        dataframe.empty
        or not dataframe[RESIDUE_NUMBERING_COLUMN].eq(RESIDUE_NUMBERING_LABEL).any()
    ):
        return None
    return (
        f"{pdb_id}: the file gives no author chain IDs or no author residue "
        "numbers, so chain_id and res_num hold its label identifiers "
        "(_atom_site.label_asym_id and _atom_site.label_seq_id)."
    )


def parse_mmcif_atom_rows(
    cif_file_path: Path,
    rcsb_metadata: Mapping[str, object],
    *,
    structure_id: str,
) -> AtomSiteParseResult:
    """Return a structure file's atom rows and what it says of its experiment.

    rcsb_metadata is the entry's RCSB record as fetch_rcsb_metadata extracts
    it, or an empty mapping for an uploaded file. The pdb_id column holds
    structure_id upper-cased, whatever the file's name or data block says,
    so that the atoms of two files of one entry share an ID. Raises
    RuntimeError naming the entry when the file cannot be read or holds a
    value the parser cannot read, such as a number that is not one. The
    result's warnings say when the archive made the entry obsolete, when
    the file lists modelled residues it holds no atoms for, as a file cut
    short does, and when chain_id and res_num hold label identifiers since
    the file gives no author ones.
    """
    pdb_id = structure_id.upper()

    try:
        mmcif_dict = read_mmcif_as_flat_dict(cif_file_path)
    except Exception as exception:
        # gemmi reports an unreadable file with several exception types.
        raise RuntimeError(f"Error parsing {pdb_id}: {exception}") from exception

    file_methods = _distinct_texts(mmcif_dict.get("_exptl.method", []))
    file_sources = _distinct_texts(mmcif_dict.get("_diffrn_source.source", []))
    try:
        dataframe = _atom_rows(
            mmcif_dict, rcsb_metadata, pdb_id, file_methods, file_sources
        )
    except ValueError as exception:
        raise RuntimeError(f"Error parsing {pdb_id}: {exception}") from exception
    return AtomSiteParseResult(
        dataframe=dataframe,
        experimental_methods=file_methods,
        is_integrative=any(item.startswith("_ihm_") for item in mmcif_dict),
        radiation_sources=file_sources,
        warnings=tuple(
            warning
            for warning in (
                _obsolete_warning(mmcif_dict, pdb_id),
                _truncation_warning(mmcif_dict, pdb_id),
                _label_numbering_warning(dataframe, pdb_id),
            )
            if warning is not None
        ),
    )


__all__ = [
    "BACKBONE_ATOM_NAMES",
    "HIGH_RADIATION_DOSE_MGY",
    "MEAN_RESIDUE_MASS_DA",
    "AtomSiteParseResult",
    "parse_mmcif_atom_rows",
    "read_mmcif_as_flat_dict",
]
