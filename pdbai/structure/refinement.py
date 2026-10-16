"""How an entry was refined, read from a flat mmCIF dictionary.

A B-factor depends on how it was refined: by which program, with or without
TLS groups, and as one isotropic value or as an anisotropic tensor. A study
that compares B-factors between entries must be able to set those choices
apart, so read_refinement_fields returns them, one value per entry. Each
attribute of the result is named after the column the parser writes it to.

Every value comes from the structure file. The RCSB entry record lists the
software as the file does, but it holds neither the TLS groups nor the
anisotropic records, and an uploaded file has no record. Reading all of
them from the file gives them one source and gives an upload the same
columns.

_software has one row per program, classified by the step it was used for.
The refinement programs are the rows classified as refinement.
refinement_program is the name in the first of them, and
refinement_program_version is the version that row states. An entry can
name several, as one refined with REFMAC and then with PHENIX does, and the
file does not say which wrote the deposited B-factors. So refinement_programs
lists every name once, in file order, joined with "; " as
experimental_methods is. A name is kept as the file writes it, with its
whitespace collapsed. The archive writes one family of programs under
several names, such as SHELXL, SHELXL-97, and SHELX, or BUSTER and
BUSTER-TNT. They are not merged, since some of those names state a version
and which of them count as one program is for a study to decide.

_pdbx_refine_tls has one row per TLS group, so tls_group_count is its row
count. The archive writes the TLS groups of a refinement into that
category, so a file that has a _refine category and no TLS group gives 0.
A file without a _refine category reports no refinement, which says nothing
of TLS, so it gives None. tls_residual_adp_flag is the archive's own note,
_refine.pdbx_TLS_residual_ADP_flag, on an entry whose deposited B-factors
probably leave out the TLS contribution.

anisotropic_atom_fraction is the share of the file's atom sites that have an
anisotropic record, a row of _atom_site_anisotrop. Hydrogen atoms are left
out of both counts, since a refinement that gives every other atom a tensor
often still lets its hydrogens ride on their parents with one isotropic
value. The atom sites are those of the file, every alternate conformer and
every model, so the fraction does not depend on the conformer the parser
keeps. A file without any anisotropic record gives 0.0, which is what its
atom sites show and not a missing value.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .elements import HYDROGEN_ELEMENTS

# The _software.classification of a program the model was refined with. It is
# matched in any letter case.
_REFINEMENT_CLASSIFICATION = "refinement"

# What separates the names in refinement_programs, as in experimental_methods
# and radiation_sources.
_PROGRAM_SEPARATOR = "; "

# A file read flat, as read_mmcif_as_flat_dict of mmcif_parser.py returns it.
_MmcifDict = Mapping[str, Sequence[str | None]]


@dataclass(frozen=True)
class RefinementFields:
    """What a structure file says of its refinement, by output column.

    A value the file does not state is None, never zero or an empty text.
    """

    # The name in the first _software row classified as refinement.
    refinement_program: str | None
    # The version that same row states.
    refinement_program_version: str | None
    # Every refinement program's name, once each, in file order, joined with
    # "; ".
    refinement_programs: str | None
    # The number of rows of _pdbx_refine_tls. It is 0 for a file with a
    # _refine category and no TLS group.
    tls_group_count: int | None
    # The first _refine.pdbx_TLS_residual_ADP_flag the file states.
    tls_residual_adp_flag: str | None
    # The share, from 0 to 1, of the non-hydrogen atom sites that have an
    # anisotropic record.
    anisotropic_atom_fraction: float | None


def _text(value: str | None) -> str | None:
    """Return the value with whitespace collapsed, None when missing or blank."""
    if value is None:
        return None
    return " ".join(value.split()) or None


def _category_row_count(mmcif_dict: _MmcifDict, category: str) -> int:
    """Return the number of rows of a category, 0 when the file has none."""
    # The point ends the category name, so _refine does not match
    # _refine_hist.
    prefix = f"{category}."
    return max(
        (len(values) for item, values in mmcif_dict.items() if item.startswith(prefix)),
        default=0,
    )


def _item_texts(mmcif_dict: _MmcifDict, item: str, row_count: int) -> list[str | None]:
    """Return an item's text in each row of its category, None where missing.

    An item the category leaves out is missing in every row.
    """
    values = mmcif_dict.get(item)
    if values is None:
        return [None] * row_count
    return [_text(value) for value in values]


def _refinement_programs(mmcif_dict: _MmcifDict) -> list[tuple[str, str | None]]:
    """Return the name and version of each refinement program, in file order.

    A row classified as refinement that names no program is left out.
    """
    row_count = _category_row_count(mmcif_dict, "_software")
    classifications, names, versions = (
        _item_texts(mmcif_dict, f"_software.{item}", row_count)
        for item in ("classification", "name", "version")
    )
    return [
        (name, version)
        for classification, name, version in zip(
            classifications, names, versions, strict=True
        )
        if name is not None
        and classification is not None
        and classification.lower() == _REFINEMENT_CLASSIFICATION
    ]


def _tls_group_count(mmcif_dict: _MmcifDict) -> int | None:
    """Return the number of TLS groups, None for a file without a refinement."""
    count = _category_row_count(mmcif_dict, "_pdbx_refine_tls")
    if count == 0 and _category_row_count(mmcif_dict, "_refine") == 0:
        return None
    return count


def _anisotropic_atom_fraction(mmcif_dict: _MmcifDict) -> float | None:
    """Return the share of non-hydrogen atom sites with an anisotropic record.

    A record is joined to its atom by _atom_site.id, as the parser joins it.
    A record id given to two rows counts for no atom, since the parser joins
    neither tensor. None is returned for a file without a non-hydrogen atom
    site.
    """
    atom_ids = mmcif_dict.get("_atom_site.id", [])
    symbols = mmcif_dict.get("_atom_site.type_symbol") or [None] * len(atom_ids)
    heavy_atom_ids = [
        atom_id
        for atom_id, symbol in zip(atom_ids, symbols, strict=True)
        if symbol is None or symbol.upper() not in HYDROGEN_ELEMENTS
    ]
    if not heavy_atom_ids:
        return None
    record_counts = Counter(mmcif_dict.get("_atom_site_anisotrop.id", []))
    anisotropic = sum(
        1
        for atom_id in heavy_atom_ids
        if atom_id is not None and record_counts[atom_id] == 1
    )
    return anisotropic / len(heavy_atom_ids)


def read_refinement_fields(mmcif_dict: _MmcifDict) -> RefinementFields:
    """Return what a file read flat says of its refinement.

    The module docstring gives the rule of each value. A category the file
    lacks, or one that leaves out an item read here, gives None for what
    would be read from it, except where the rule says otherwise.
    """
    programs = _refinement_programs(mmcif_dict)
    first_name, first_version = programs[0] if programs else (None, None)
    row_count = _category_row_count(mmcif_dict, "_refine")
    flags = _item_texts(mmcif_dict, "_refine.pdbx_TLS_residual_ADP_flag", row_count)
    return RefinementFields(
        refinement_program=first_name,
        refinement_program_version=first_version,
        refinement_programs=_PROGRAM_SEPARATOR.join(
            dict.fromkeys(name for name, _ in programs)
        )
        or None,
        tls_group_count=_tls_group_count(mmcif_dict),
        tls_residual_adp_flag=next((flag for flag in flags if flag is not None), None),
        anisotropic_atom_fraction=_anisotropic_atom_fraction(mmcif_dict),
    )


__all__ = ["RefinementFields", "read_refinement_fields"]
