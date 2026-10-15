"""Nullable Booleans read from a column without guessing.

Every column declared boolean, in the schema catalog or in an engine's own
table, is read through to_nullable_boolean, so a flag means the same
wherever it is stored. A plain conversion to bool would read the text
"false" and the number 2 as True. A value that is not plainly a Boolean is
therefore refused rather than read as one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .refused_values import listed_values

# The numbers read as False and True, of any numeric type.
_ACCEPTED_NUMBERS = (0, 1)


def _is_accepted(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value in _ACCEPTED_NUMBERS
    return False


def _refusal(column: str, refused: list[object]) -> ValueError:
    return ValueError(
        f"Column {column!r} is declared boolean but holds non-boolean values "
        f"other than True, False, 1, 0, or a missing value, which are "
        f"{listed_values(refused)}."
    )


def to_nullable_boolean(values: pd.Series, *, column: str) -> pd.Series:
    """Return a column's values as nullable Booleans, under the same index.

    True, False, 1, and 0 of any numeric type are converted, and None, NaN,
    and NA stay missing. Any other value raises ValueError naming ``column``
    and the first few distinct values refused, in the order they appear.
    """
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype("boolean")

    present = values[values.notna()]
    if pd.api.types.is_numeric_dtype(values.dtype):
        accepted = present.isin(_ACCEPTED_NUMBERS)
        if not accepted.all():
            raise _refusal(column, present[~accepted].tolist())
        return values.astype("boolean")

    refused = [value for value in present.tolist() if not _is_accepted(value)]
    if refused:
        raise _refusal(column, refused)
    result = pd.Series(pd.NA, index=values.index, dtype="boolean")
    result[present.index] = present.map(bool).astype("boolean")
    return result


__all__ = ["to_nullable_boolean"]
