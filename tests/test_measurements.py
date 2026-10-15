from __future__ import annotations

import numpy as np
import pytest

from pdbai.measurements import parse_positive_number, positive_number


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
