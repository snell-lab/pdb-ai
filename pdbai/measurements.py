"""Positive measurements, read and checked by one rule at every front end.

A temperature or a resolution passes or fails the same way, whichever front
end it came from. The command line reads its range filters through this
module, and the web app reads the same filters from its JSON keys.
parse_positive_number reads a value as a person or a client supplied it.
positive_number checks a value that code or storage should already hold as a
number. Each refusal raises ValueError with a message such as
"min_temperature_K must be a positive number; got 0.", where the caller
supplies the name.

Text is read only in its plain decimal form, such as "273" or "2.5". Text
with a sign, an exponent, an underscore, or no digit before the point is
refused although float() accepts it, as is "inf" and "nan". A Boolean is
refused although Python counts it as a number, since True is not a
measurement. pdbai.integers reads counts by the same rule.
"""

from __future__ import annotations

import math
import numbers
import re

# The text parse_positive_number reads once surrounding whitespace is removed.
# Non-ASCII digits are refused as well, though float() accepts them.
_DECIMAL_TEXT = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def _not_positive_message(name: str, value: object) -> str:
    return f"{name} must be a positive number; got {value!r}."


def positive_number(value: object, name: str) -> float:
    """Return ``value`` as a plain float if it is a positive, finite number.

    Any real number is accepted, including an integer and a NumPy number,
    because a stored whole value such as 273 is still a measurement. A
    Boolean, text, any other type, and a number that is not finite or not
    positive raise ValueError naming ``name``. Use parse_positive_number for
    input that a person or a client supplied.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(_not_positive_message(name, value))
    try:
        number = float(value)
    except OverflowError:
        # An integer too large for a float is not a measurement either.
        number = math.inf
    if not math.isfinite(number) or number <= 0:
        raise ValueError(_not_positive_message(name, value))
    return number


def parse_positive_number(value: object, name: str) -> float:
    """Return the positive number a person or a client supplied.

    The value may be a real number, or the text of one in ASCII decimal
    digits with an optional fraction and optional surrounding whitespace. A
    Boolean, bytes, text in any other form, and a value that is not finite
    or not positive raise ValueError naming ``name``.
    """
    if isinstance(value, str):
        if _DECIMAL_TEXT.fullmatch(value.strip()) is None:
            raise ValueError(_not_positive_message(name, value))
        try:
            # The pattern admits only digits and one point, so float() reads
            # the text exactly as written. Text of too many digits reads as
            # infinity, which is refused below with the same message.
            return positive_number(float(value), name)
        except ValueError:
            raise ValueError(_not_positive_message(name, value)) from None
    return positive_number(value, name)


__all__ = ["parse_positive_number", "positive_number"]
