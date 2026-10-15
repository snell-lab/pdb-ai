"""The check that every declared Boolean column holds True, False, or missing.

Every column the catalog declares "boolean" leaves annotation as a pandas
nullable Boolean column. A flag has to keep "unknown" apart from False. A
float column cannot do that safely, since Python reads NaN as true. The check
runs once, after the last engine, so it covers every producer and no engine
has to convert its own flags. nullable_booleans sets which values convert
and which are refused.

A column name that appears twice is refused rather than checked. The two
columns could disagree, and pandas would return both under the one name.
"""

from __future__ import annotations

import pandas as pd

from ..nullable_booleans import to_nullable_boolean
from .catalog import declared_boolean_columns


def normalize_boolean_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame whose declared Boolean columns are nullable Booleans.

    The DataFrame given is returned as it is when those columns are nullable
    Booleans already, and otherwise a converted copy is returned. Raises
    ValueError when a declared Boolean column appears more than once, or holds
    a value other than True, False, 1, 0, or a missing value.
    """
    declared = declared_boolean_columns()
    names = [str(name) for name in dataframe.columns if str(name) in declared]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(
            f"Boolean column(s) {', '.join(repeated)} appear more than once"
        )
    pending = [
        name for name in names if not isinstance(dataframe[name].dtype, pd.BooleanDtype)
    ]
    if not pending:
        return dataframe

    normalized = dataframe.copy()
    for name in pending:
        normalized[name] = to_nullable_boolean(dataframe[name], column=name)
    return normalized


__all__ = ["normalize_boolean_columns"]
