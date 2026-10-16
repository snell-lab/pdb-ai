from __future__ import annotations

import numpy as np
import pytest

from pdbai.positive_numbers import (
    parse_positive_int,
    parse_positive_number,
    positive_int,
    positive_number,
)


@pytest.mark.parametrize(
    "raw", [5, "5", " 5 ", "\t5\n", "005", 5.0, "5.0", " 5.00 ", np.int64(5)]
)
def test_parse_positive_int_accepts_whole_numbers_as_supplied(raw: object) -> None:
    assert parse_positive_int(raw, "limit") == 5


def test_parse_positive_int_reads_integer_text_exactly() -> None:
    # This is 2**53 + 1, the first integer a float cannot hold. Reading the
    # text through float would turn it into 2**53.
    assert parse_positive_int("9007199254740993", "limit") == 9007199254740993
    assert parse_positive_int("9007199254740993.0", "limit") == 9007199254740993


def test_parse_positive_int_returns_a_plain_int_for_a_numpy_integer() -> None:
    result = parse_positive_int(np.uint16(7), "limit")

    assert result == 7
    assert type(result) is int


@pytest.mark.parametrize(
    "raw",
    [
        0,
        -1,
        "0",
        "0.0",
        "-1",
        "-1.0",
        "1.5",
        "5.",
        ".5",
        "five",
        "",
        "  ",
        "inf",
        "nan",
        "1e3",
        "1E3",
        "1_000",
        "+5",
        "5 5",
        "٥",
        b"7",
        bytearray(b"7"),
        "1" * 5000,
        0.0,
        -5.0,
        1.5,
        float("inf"),
        float("nan"),
        np.int64(0),
        np.bool_(True),
        None,
        [5],
        True,
        False,
    ],
)
def test_parse_positive_int_rejects_anything_else(raw: object) -> None:
    with pytest.raises(ValueError, match="^limit must be a positive integer; got"):
        parse_positive_int(raw, "limit")


def test_parse_positive_int_names_the_value_as_supplied() -> None:
    with pytest.raises(ValueError) as caught:
        parse_positive_int("0.0", "limit")

    assert str(caught.value) == "limit must be a positive integer; got '0.0'."


def test_positive_int_returns_a_positive_integer() -> None:
    assert positive_int(3, "limit") == 3


def test_positive_int_returns_a_plain_int_for_a_numpy_integer() -> None:
    result = positive_int(np.int32(3), "limit")

    assert result == 3
    assert type(result) is int


@pytest.mark.parametrize(
    "value", [0, -1, True, False, "5", 5.0, b"5", None, np.int64(-2), np.bool_(True)]
)
def test_positive_int_rejects_anything_else(value: object) -> None:
    with pytest.raises(ValueError, match="^limit must be a positive integer; got"):
        positive_int(value, "limit")


def test_positive_int_message_ends_with_a_period() -> None:
    with pytest.raises(ValueError) as caught:
        positive_int(0, "limit")

    assert str(caught.value) == "limit must be a positive integer; got 0."


@pytest.mark.parametrize(
    ("value", "expected"),
    [(273, 273.0), (2.5, 2.5), (np.float64(1.8), 1.8), (np.int64(100), 100.0)],
)
def test_positive_number_takes_any_real_number_as_a_float(
    value: object, expected: float
) -> None:
    number = positive_number(value, "bound")

    assert number == expected
    assert type(number) is float


@pytest.mark.parametrize(
    "value",
    [True, False, 0, 0.0, -1, -2.5, float("nan"), float("inf"), 10**400, "273", None],
)
def test_positive_number_refuses_anything_else_naming_the_value(value: object) -> None:
    with pytest.raises(ValueError, match="^bound must be a positive number; got "):
        positive_number(value, "bound")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("273", 273.0), ("2.5", 2.5), (" 1.80 ", 1.8), ("0.5", 0.5), (273, 273.0)],
)
def test_parse_positive_number_reads_plain_decimal_text(
    value: object, expected: float
) -> None:
    assert parse_positive_number(value, "bound") == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "0",
        "0.0",
        "-5",
        "+5",
        "1e2",
        "1_000",
        ".5",
        "5.",
        "2,5",
        "inf",
        "nan",
        "٢٧٣",
        "9" * 400,
        b"273",
        True,
        None,
    ],
)
def test_parse_positive_number_refuses_text_in_any_other_form(value: object) -> None:
    with pytest.raises(ValueError) as caught:
        parse_positive_number(value, "--min-temperature")

    assert str(caught.value) == (
        f"--min-temperature must be a positive number; got {value!r}."
    )
