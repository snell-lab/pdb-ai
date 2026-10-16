"""Positive counts and measurements, read and checked by one rule everywhere.

A limit, a count, a temperature, or a resolution passes or fails the same
way, whichever front end it came from. The command line reads --limit,
--workers, and its range filters through this module. The web app reads its
form fields, JSON keys, port, configured caps, and range filters through it,
and the engines read their own count settings. Each kind of value has two
readers. parse_positive_int and parse_positive_number read a value as a
person or a client supplied it. positive_int and positive_number check a
value that code or storage should already hold as a number. Each refusal
raises ValueError with a message such as "limit must be a positive integer;
got 0." or "min_temperature_K must be a positive number; got 0.", where the
caller supplies the name.

Text is read only in its plain decimal form, such as "5", "273", or "2.5".
Text with a sign, an exponent, an underscore, or no digit on either side of
the point is refused although int() and float() accept it, as are "inf" and
"nan". A count may also be a whole float or its text, such as 5.0 or "5.0",
because web forms and JSON clients send them. A Boolean is refused although
Python counts it as a number, since True is neither a count nor a
measurement.
"""

from __future__ import annotations

import contextlib
import math
import numbers
import re
from typing import Literal

# The text the parsers read once surrounding whitespace is removed. The
# decimal point and zeros of a count admit the text of a whole float such as
# "5.0". Non-ASCII digits are refused as well, though int() and float()
# accept them.
_WHOLE_NUMBER_TEXT = re.compile(r"([0-9]+)(?:\.0+)?")
_DECIMAL_TEXT = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def _not_positive_message(
    name: str, value: object, kind: Literal["integer", "number"]
) -> str:
    return f"{name} must be a positive {kind}; got {value!r}."


def positive_int(value: object, name: str) -> int:
    """Return ``value`` as a plain int if it is a positive integer.

    Any integral number is accepted, including a NumPy integer, because
    counts computed with NumPy reach the same checks. A Boolean, a float,
    text, any other type, and an integer that is not positive raise
    ValueError naming ``name``. Use parse_positive_int for input that a
    person or a client supplied.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(_not_positive_message(name, value, "integer"))
    integer = int(value)
    if integer <= 0:
        raise ValueError(_not_positive_message(name, value, "integer"))
    return integer


def parse_positive_int(value: object, name: str) -> int:
    """Return the positive integer a person or a client supplied.

    The value may be an integral number or a whole float. It may also be the
    text of either, in ASCII decimal digits, with optional surrounding
    whitespace. Integer text is read exactly, never through a float, so a
    number too large for a float keeps every digit. A Boolean, bytes, text in
    any other form, a float that is not whole or not finite, and a value that
    is not positive raise ValueError naming ``name``.
    """
    if isinstance(value, numbers.Integral):
        return positive_int(value, name)
    integer = 0
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            integer = int(value)
    elif isinstance(value, str):
        match = _WHOLE_NUMBER_TEXT.fullmatch(value.strip())
        if match is not None:
            # Python refuses to read text of more than a few thousand digits,
            # and such text is refused here with the same message as any other.
            with contextlib.suppress(ValueError):
                integer = int(match.group(1))
    # Zero stands for every refusal here, so that the message names the value
    # as it was supplied rather than what it was read as.
    if integer <= 0:
        raise ValueError(_not_positive_message(name, value, "integer"))
    return integer


def positive_number(value: object, name: str) -> float:
    """Return ``value`` as a plain float if it is a positive, finite number.

    Any real number is accepted, including an integer and a NumPy number,
    because a stored whole value such as 273 is still a measurement. A
    Boolean, text, any other type, and a number that is not finite or not
    positive raise ValueError naming ``name``. Use parse_positive_number for
    input that a person or a client supplied.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(_not_positive_message(name, value, "number"))
    try:
        number = float(value)
    except OverflowError:
        # An integer too large for a float is not a measurement either.
        number = math.inf
    if not math.isfinite(number) or number <= 0:
        raise ValueError(_not_positive_message(name, value, "number"))
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
            raise ValueError(_not_positive_message(name, value, "number"))
        try:
            # The pattern admits only digits and one point, so float() reads
            # the text exactly as written. Text of too many digits reads as
            # infinity, which is refused below with the same message.
            return positive_number(float(value), name)
        except ValueError:
            raise ValueError(_not_positive_message(name, value, "number")) from None
    return positive_number(value, name)


__all__ = [
    "parse_positive_int",
    "parse_positive_number",
    "positive_int",
    "positive_number",
]
