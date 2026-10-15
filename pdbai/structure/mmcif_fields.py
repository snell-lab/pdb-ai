"""Atom columns from a flat mmCIF dictionary, and the volume of its unit cell.

The parser reads an mmCIF file into one flat dictionary. It maps each item
name, such as _atom_site.Cartn_x, to its values, one per row of its category.
expand_mmcif_atom_fields turns that dictionary into columns of one value per
atom. An item outside _atom_site becomes a column only when its category has
one row. Such a category describes the whole entry, as _exptl does, so its
value is repeated on every atom. A category of several rows, such as _entity
or _struct_conn, has no positional link to the atoms. Spreading it over them
would invent one, so it is left out and the parser reads it from the
dictionary. _atom_site_anisotrop is left out too, because each of its rows
belongs to one atom.

calculate_unit_cell_volume returns NaN, the dataset's missing value, for a
cell that is absent, impossible, or the wwPDB placeholder for an entry
without a crystal. A number there would read as a measurement.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

import gemmi

_ATOM_SITE_PREFIX = "_atom_site."

# Categories whose rows each belong to one atom. The parser joins them to the
# atoms by atom id, so even a one-row loop describes one atom.
_ATOM_KEYED_CATEGORIES = ("_atom_site_anisotrop.",)

# The unit cell the wwPDB writes for NMR, electron microscopy, and other
# entries without a crystal. No crystal has 1 Å edges, so this volume is never
# a measurement.
_NON_CRYSTAL_PLACEHOLDER_CELL = (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)


def _unwrap_lines(value: str | None) -> str | None:
    if value is None:
        return None
    if "\n" in value or "\r" in value:
        # A CIF text block wraps its lines for display only. A line break
        # inside a CSV field breaks many spreadsheet tools.
        return re.sub(r"\r\n|\r|\n", " ", value).strip()
    return value


def expand_mmcif_atom_fields(
    mmcif_dict: Mapping[str, Sequence[str | None]],
) -> tuple[dict[str, list[str | None]], int]:
    """Return a flat mmCIF dictionary's atom columns and its atom count.

    The _atom_site items are the columns of one CIF loop, so each holds one
    value per atom. Every _atom_site item becomes a column. So does every
    item of a one-row category, repeated on each atom row, unless the
    category is atom-keyed like _atom_site_anisotrop. A None, which the
    reader gives for the placeholders "?" and ".", stays None. A multi-line
    value has its line breaks replaced by spaces. A dictionary without atoms
    gives no columns and a count of zero.
    """
    atom_count = next(
        (
            len(values)
            for key, values in mmcif_dict.items()
            if key.startswith(_ATOM_SITE_PREFIX)
        ),
        0,
    )
    if atom_count == 0:
        return {}, 0

    expanded_fields: dict[str, list[str | None]] = {}
    for key, raw_values in mmcif_dict.items():
        if key.startswith(_ATOM_SITE_PREFIX):
            expanded_fields[key] = [_unwrap_lines(value) for value in raw_values]
        elif len(raw_values) == 1 and not key.startswith(_ATOM_KEYED_CATEGORIES):
            # Every item of a category holds one value per row, so an item with
            # one value belongs to a one-row category.
            expanded_fields[key] = [_unwrap_lines(raw_values[0])] * atom_count
    return expanded_fields, atom_count


def _is_physical_cell(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> bool:
    if not all(math.isfinite(value) for value in (a, b, c, alpha, beta, gamma)):
        return False
    if min(a, b, c) <= 0:
        return False
    if not all(0 < angle < 180 for angle in (alpha, beta, gamma)):
        return False
    # The triclinic volume is abc times the square root of
    # 1 - cos²α - cos²β - cos²γ + 2 cosα cosβ cosγ. That root is real and
    # positive exactly when these conditions hold. Checking the angles as
    # written refuses a flat cell such as 120, 120, 120, whose computed
    # radicand rounding could leave slightly positive.
    return (
        alpha + beta + gamma < 360
        and alpha < beta + gamma
        and beta < alpha + gamma
        and gamma < alpha + beta
    )


def calculate_unit_cell_volume(
    a: float | None,
    b: float | None,
    c: float | None,
    alpha: float | None,
    beta: float | None,
    gamma: float | None,
) -> float:
    """Return the unit-cell volume in cubic ångströms, or NaN without a cell.

    The lengths are in ångströms and the angles in degrees. NaN is returned
    when a parameter is None or not finite, or a length is not positive. It
    is also returned when an angle lies outside (0, 180) or the angles cannot
    close a cell. They cannot when they sum to 360 or more, or when one is at
    least the sum of the other two. The wwPDB placeholder cell of 1 Å edges
    and right angles gives NaN too.
    """
    if a is None or b is None or c is None:
        return math.nan
    if alpha is None or beta is None or gamma is None:
        return math.nan
    if (a, b, c, alpha, beta, gamma) == _NON_CRYSTAL_PLACEHOLDER_CELL:
        return math.nan
    if not _is_physical_cell(a, b, c, alpha, beta, gamma):
        return math.nan
    volume = float(gemmi.UnitCell(a, b, c, alpha, beta, gamma).volume)
    # Rounding can still leave a cell that only just closes with no volume.
    return volume if volume > 0 else math.nan


__all__ = ["calculate_unit_cell_volume", "expand_mmcif_atom_fields"]
