from __future__ import annotations

import numpy as np
import pytest

from pdbai.engines.sources.scalars import to_float, to_int


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("5", 5),
        (" 5 ", 5),
        (5, 5),
        (-3, -3),
        ("-3", -3),
        ("+3", 3),
        (np.int64(7), 7),
        ("1" * 30, int("1" * 30)),
    ],
)
def test_to_int_reads_integers_and_their_text(value: object, expected: int) -> None:
    assert to_int(value) == expected


@pytest.mark.parametrize(
    "value",
    [None, "", " ", "abc", "-", [1], {"a": 1}, b"5", "٥", "1" * 5000],
)
def test_to_int_refuses_values_that_are_not_integers(value: object) -> None:
    assert to_int(value) is None


@pytest.mark.parametrize("value", [True, False])
def test_to_int_refuses_booleans(value: object) -> None:
    assert to_int(value) is None


@pytest.mark.parametrize("value", [5.0, 5.7, "5.0", float("nan"), float("inf")])
def test_to_int_refuses_floats_even_when_whole(value: object) -> None:
    assert to_int(value) is None


@pytest.mark.parametrize(
    "value", ["1_000", "5-", "- 5", "--5", "+-5", "1e3", "1E3", "0x10", "١٢"]
)
def test_to_int_refuses_text_python_reads_but_sources_do_not_write(
    value: str,
) -> None:
    assert to_int(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1.5", 1.5),
        (1.5, 1.5),
        (" 2 ", 2.0),
        (3, 3.0),
        ("-0.25", -0.25),
        ("+0.25", 0.25),
        (".5", 0.5),
        ("5.", 5.0),
        (np.float32(0.5), 0.5),
    ],
)
def test_to_float_reads_numbers_and_their_text(value: object, expected: float) -> None:
    assert to_float(value) == expected


@pytest.mark.parametrize("value", [None, "", " ", "abc", ".", "-", [1], b"1.5"])
def test_to_float_refuses_values_that_are_not_numbers(value: object) -> None:
    assert to_float(value) is None


@pytest.mark.parametrize("value", [True, False])
def test_to_float_refuses_booleans(value: object) -> None:
    assert to_float(value) is None


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        np.float64("nan"),
        10**400,
        "1" * 400,
    ],
)
def test_to_float_refuses_values_that_are_not_finite(value: object) -> None:
    assert to_float(value) is None


@pytest.mark.parametrize(
    "value",
    [
        "nan",
        "NaN",
        "inf",
        "-Infinity",
        "1_000.5",
        "1e3",
        "1.5E-3",
        "5-",
        "- 5",
        "1.2.3",
        "٥.5",
    ],
)
def test_to_float_refuses_text_python_reads_but_sources_do_not_write(
    value: str,
) -> None:
    assert to_float(value) is None
