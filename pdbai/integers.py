"""Positive integers, read and checked by one rule at every front end.

A limit or a count passes or fails the same way, whichever front end it came
from. The command line reads --limit and --workers through this module. The
web app reads its form fields, JSON keys, port, and configured caps through
it, and the engines read their own count settings. parse_positive_int reads
a value as a person or a client supplied it. positive_int checks a value
that code or storage should already hold as an integer. Each refusal raises
ValueError with a message such as "limit must be a positive integer; got
0.", where the caller supplies the name.

A whole float and its text, such as 5.0 or "5.0", are accepted because web
forms and JSON clients send them. Text with a sign, an exponent, or an
underscore is refused although int() and float() accept it, so that only
the plain text of a whole number passes. A Boolean is refused although
Python counts it as an integer, since True is not a count.
"""

from __future__ import annotations

import contextlib
import math
import numbers
import re

# The text parse_positive_int reads once surrounding whitespace is removed.
# The decimal point and zeros admit the text of a whole float such as "5.0".
# Non-ASCII digits are refused as well, though int() accepts them.
_WHOLE_NUMBER_TEXT = re.compile(r"([0-9]+)(?:\.0+)?")


def _not_positive_message(name: str, value: object) -> str:
    return f"{name} must be a positive integer; got {value!r}."


def positive_int(value: object, name: str) -> int:
    """Return ``value`` as a plain int if it is a positive integer.

    Any integral number is accepted, including a NumPy integer, because
    counts computed with NumPy reach the same checks. A Boolean, a float,
    text, any other type, and an integer that is not positive raise
    ValueError naming ``name``. Use parse_positive_int for input that a
    person or a client supplied.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(_not_positive_message(name, value))
    integer = int(value)
    if integer <= 0:
        raise ValueError(_not_positive_message(name, value))
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
        raise ValueError(_not_positive_message(name, value))
    return integer


__all__ = ["parse_positive_int", "positive_int"]
