"""The refused values an error message quotes.

A reader that refuses values names them in its error, so that a user can
find them in the file. A column can hold thousands of bad values, so the
message lists a few distinct ones and counts the rest.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

# How many distinct values a message lists before it counts the rest.
_MAX_LISTED_VALUES = 5


def listed_values(values: Iterable[object]) -> str:
    """Return the distinct values as a list for an error message.

    Values are listed once each, in order of first appearance, as their
    repr. None, pandas NA, and NaN are listed as "a missing value". After
    five values the rest are counted, as in "'a', 'b', 'c', 'd', 'e', and 2
    more".
    """
    distinct = list(
        dict.fromkeys(
            "a missing value"
            if value is None
            or value is pd.NA
            or (isinstance(value, float) and math.isnan(value))
            else repr(value)
            for value in values
        )
    )
    listed = ", ".join(distinct[:_MAX_LISTED_VALUES])
    more = len(distinct) - _MAX_LISTED_VALUES
    return f"{listed}, and {more} more" if more > 0 else listed


__all__ = ["listed_values"]
