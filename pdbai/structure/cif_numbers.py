"""The grammar by which every reader of the package reads a CIF number.

A CIF number may carry its standard uncertainty in parentheses, as in
40.123(4) or an occupancy of 0.50(2). Only the value before the parentheses
is a measurement the dataset keeps. The parser, the conformer choice, and
the engines that read a file again must agree on that, or one of them
refuses a file the others read. The grammar therefore lives here, below all
of them, so that the conformer choice can use it without importing the
parser that imports the conformer choice.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# A CIF number, which may carry its standard uncertainty in parentheses, as in
# 40.123(4). Only the value before the parentheses is kept.
CIF_NUMBER = re.compile(
    r"(?P<value>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(?:\(\d+\))?"
)


def coerce_cif_numbers(values: pd.Series) -> pd.Series:
    """Return CIF number texts as floats, NaN where missing or not a number.

    A standard uncertainty in parentheses, as in 2.040(3), is dropped, as the
    parser drops it. Text that is not a finite CIF number becomes NaN rather
    than an error, for a reader of rows the atom table may not have kept.
    """
    present = values.notna() & (values != "")
    text = values[present].astype(str)
    number_text = text.str.extract(f"^{CIF_NUMBER.pattern}$", expand=False)
    numbers = pd.Series(np.nan, index=values.index, dtype=float)
    numbers[present] = number_text.astype(float)
    return numbers.where(np.isfinite(numbers))


__all__ = ["CIF_NUMBER", "coerce_cif_numbers"]
