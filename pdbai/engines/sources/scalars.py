"""Numbers read from source responses, with None for a value that is not one.

Remote records give a number as a JSON number or as text, and a field can
hold an empty string or text that is no number at all. The source engines
read numbers from their responses through to_float and to_int. Both return
None for a value that is not a number, so one malformed field leaves its
column empty. It does not fail the entry.

Both are stricter than float() and int(), in the way pdbai.integers is.
Text must be ASCII decimal digits, with an optional sign in front and, for
to_float, one decimal point. Text with an exponent, an underscore, a sign
anywhere else, or a word such as "nan" or "inf" is refused, although Python
reads it as a number. The sources write plain decimals, so such text marks
a malformed field rather than a value. A Boolean and a value that is not
finite are refused too, since neither is a measurement.

to_int neither rounds nor truncates. Reading 5.7 as 5 would change the value
without a trace, so 5.7 reads as missing, and so does the text "5.0".
"""

from __future__ import annotations

import math
import numbers
import re

# The text each function reads once surrounding whitespace is removed.
_INTEGER_TEXT = re.compile(r"[+-]?[0-9]+")
_DECIMAL_TEXT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)")


def to_float(value: object) -> float | None:
    """Return the value as a finite float, or None when it is not a number.

    A real number other than a Boolean is accepted, and so is the text of a
    decimal number with optional surrounding whitespace. Anything else
    returns None, as does a value that is not finite or too large for a
    float.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        try:
            number = float(value)
        except OverflowError:
            return None
    elif isinstance(value, str) and _DECIMAL_TEXT.fullmatch(text := value.strip()):
        number = float(text)
    else:
        return None
    return number if math.isfinite(number) else None


def to_int(value: object) -> int | None:
    """Return the value as an integer, or None unless it is an integer.

    An integral number other than a Boolean is accepted, and so is the text
    of an integer with optional surrounding whitespace. A float is refused
    even when it is whole, and so is its text, such as "5.0".
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, str) and _INTEGER_TEXT.fullmatch(text := value.strip()):
        # Python refuses to read text of more than a few thousand digits.
        try:
            return int(text)
        except ValueError:
            return None
    return None


__all__ = ["to_float", "to_int"]
