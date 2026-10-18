"""The deposited unit cell, read from the atom table.

The density engine compares the cell of the file whose atoms it places with
the cell of the map it reads them in. The atom table repeats the file's six
_cell items on every row, so the cell is read from the first row. A number is
read by the parser's grammar, so a cell length written with its uncertainty,
as 79.1(2), is 79.1 here too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ....structure.cif_numbers import coerce_cif_numbers

_CELL_ITEMS = (
    "_cell.length_a",
    "_cell.length_b",
    "_cell.length_c",
    "_cell.angle_alpha",
    "_cell.angle_beta",
    "_cell.angle_gamma",
)

# A unit cell's lengths a, b, and c in Å, and its angles alpha, beta, and
# gamma in degrees.
type CellParameters = tuple[float, float, float, float, float, float]


def read_cell_parameters(atoms: pd.DataFrame) -> CellParameters | None:
    """Return the deposited unit cell the atom table records, or None.

    The six _cell items are read from the first row of the atom table, which
    repeats them on every row. None stands for a table without them, or with
    a value among them that is not a positive number.
    """
    if atoms.empty:
        return None
    first = [
        atoms[item].iloc[0] if item in atoms.columns else None for item in _CELL_ITEMS
    ]
    numbers = coerce_cif_numbers(pd.Series(first, dtype=object)).to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    if not (np.isfinite(numbers) & (numbers > 0)).all():
        return None
    a, b, c, alpha, beta, gamma = (float(value) for value in numbers)
    return (a, b, c, alpha, beta, gamma)


__all__ = ["CellParameters", "read_cell_parameters"]
