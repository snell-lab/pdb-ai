from __future__ import annotations

import pytest

from pdbai.engines.core import EngineOptions, OptionSpec, OptionValue


def _count(default: OptionValue) -> OptionSpec:
    return OptionSpec(
        "points", label="Points", help="Points.", kind="count", default=default
    )


def _choice() -> OptionSpec:
    return OptionSpec(
        "mode",
        label="Mode",
        help="Mode.",
        kind="choice",
        default="fast",
        choices=("fast", "slow"),
    )


@pytest.mark.parametrize("default", [0, -5, True, 2.0, "3"])
def test_a_count_setting_refuses_a_default_its_check_would_refuse(
    default: OptionValue,
) -> None:
    # A default of 0 is refused when the setting is declared. Otherwise every
    # run that left the setting out would fail validation.
    with pytest.raises(
        ValueError,
        match=(
            "^Count option 'points' needs a positive integer default or None; "
            f"got {default!r}$"
        ),
    ):
        _count(default)


@pytest.mark.parametrize("default", [None, 1, 250])
def test_a_count_setting_accepts_a_positive_default_or_none(
    default: OptionValue,
) -> None:
    option = _count(default)

    assert option.default == default
    assert option.check(option.default) == default


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(" TRUE ", True), ("false", False), (False, False), (None, False), ("  ", False)],
)
def test_a_flag_setting_parses_booleans_and_their_text(
    raw: object, expected: bool
) -> None:
    option = OptionSpec("include_x", label="X", help="X.")

    assert option.parse(raw) is expected


@pytest.mark.parametrize(
    ("raw", "expected"), [(" slow ", "slow"), ("", "fast"), (None, "fast")]
)
def test_a_choice_setting_parses_text_and_gives_its_default_for_blank(
    raw: object, expected: str
) -> None:
    assert _choice().parse(raw) == expected


def test_a_choice_setting_refuses_a_value_that_is_not_text() -> None:
    # A value that is not text is refused rather than converted with str(),
    # which would let True match a choice named "True".
    option = OptionSpec(
        "mode",
        label="Mode",
        help="Mode.",
        kind="choice",
        default="False",
        choices=("True", "False"),
    )

    with pytest.raises(ValueError, match="^mode must be one of True, False; got True$"):
        option.parse(True)


def test_a_count_setting_parses_text_and_gives_its_default_for_blank() -> None:
    assert _count(None).parse(" 12 ") == 12
    assert _count(5).parse("") == 5


def test_with_values_returns_a_copy_and_leaves_the_original() -> None:
    original = EngineOptions(values={"include_sasa": True, "mode": "fast"})

    changed = original.with_values({"mode": "slow", "points": 3})

    assert changed.values == {"include_sasa": True, "mode": "slow", "points": 3}
    assert original.values == {"include_sasa": True, "mode": "fast"}
    assert changed.dataset_shape == original.dataset_shape
    assert changed.technique == original.technique


def test_options_keep_their_settings_when_the_caller_changes_its_mapping() -> None:
    values: dict[str, OptionValue] = {"include_sasa": True}

    options = EngineOptions(values=values)
    values["include_sasa"] = False

    assert options.flag("include_sasa")


def test_option_accessors_refuse_a_value_of_another_kind() -> None:
    options = EngineOptions(values={"mode": 1, "points": True, "flag": "true"})

    with pytest.raises(KeyError, match="'mode' holds no choice; got 1"):
        options.choice("mode")
    with pytest.raises(KeyError, match="'points' holds no count; got True"):
        options.count("points")
    # Only True counts as on, so the text "true" does not.
    assert not options.flag("flag")
    assert options.count("unset") is None
