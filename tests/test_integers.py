from __future__ import annotations

import numpy as np
import pytest

from pdbai.integers import parse_positive_int, positive_int


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
