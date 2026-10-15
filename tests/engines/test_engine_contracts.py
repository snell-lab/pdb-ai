from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import pytest

from pdbai.engines.core import (
    PROFILE_ML_MINIMAL,
    ArtifactKey,
    ArtifactStore,
    ColumnGranularity,
    ColumnSpec,
    DatasetSource,
    EngineOptions,
    EnginePhase,
    EngineResult,
    EngineSpec,
    OptionSpec,
    ReportField,
    ReportKind,
    TableColumnSpec,
    TableLink,
    TableSpec,
    column_specs,
)

PATH_KEY = ArtifactKey("structure_file", Path)


def _engine(**overrides: Any) -> EngineSpec:
    arguments: dict[str, Any] = {
        "name": "scorer",
        "display_name": "Scorer",
        "phase": EnginePhase.ANNOTATION,
    }
    arguments.update(overrides)
    return EngineSpec(**arguments)


def test_an_artifact_key_refuses_a_name_or_type_it_cannot_publish_under() -> None:
    with pytest.raises(ValueError, match="names must not be empty"):
        ArtifactKey("", Path)
    # A parameterized generic is not a class, so isinstance cannot check
    # values against it.
    generic: Any = list[str]
    with pytest.raises(
        ValueError, match=r"'names' declares type list\[str\], which is not a class"
    ):
        ArtifactKey("names", generic)


def test_the_store_publishes_and_returns_values_of_the_key_type() -> None:
    store = ArtifactStore()

    assert store.get(PATH_KEY) is None
    assert PATH_KEY not in store

    store.put(PATH_KEY, Path("first.cif"))
    store.put(PATH_KEY, Path("second.cif"))

    assert PATH_KEY in store
    assert store.get(PATH_KEY) == Path("second.cif")


def test_the_store_refuses_a_value_of_the_wrong_type_and_keeps_the_old_one() -> None:
    store = ArtifactStore()
    store.put(PATH_KEY, Path("kept.cif"))

    with pytest.raises(
        TypeError, match="Artifact 'structure_file' must be a Path; got str"
    ):
        store.put(PATH_KEY, cast(Path, "replaced.cif"))

    assert store.get(PATH_KEY) == Path("kept.cif")


def test_the_store_refuses_a_value_read_with_a_key_of_another_type() -> None:
    # The registry matches keys by name, so a consumer that declares the
    # name with the wrong type passes its check. Reading must not then look
    # like a producer that had nothing to publish.
    store = ArtifactStore()
    store.put(PATH_KEY, Path("1abc.cif"))

    with pytest.raises(
        TypeError,
        match=(
            "Artifact 'structure_file' holds a PosixPath, but the key it is read "
            "with promises a str"
        ),
    ):
        store.get(ArtifactKey("structure_file", str))


def test_the_store_publishes_several_values_until_one_has_the_wrong_type() -> None:
    count = ArtifactKey("count", int)
    label = ArtifactKey("label", str)
    store = ArtifactStore()

    with pytest.raises(TypeError, match="'label' must be a str"):
        store.update({count: 3, label: 4})

    assert store.get(count) == 3
    assert label not in store


def test_column_specs_share_settings_and_override_them_by_name() -> None:
    first, second = column_specs(
        ["score", "flagged"],
        group="Scorer Fields",
        toggle="include_scores",
        granularity=ColumnGranularity.RESIDUE,
        granularity_by_name={"flagged": ColumnGranularity.STRUCTURE},
        dtype_by_name={"score": "numeric"},
        descriptions={"score": "The score."},
        profiles=frozenset({PROFILE_ML_MINIMAL}),
        profiles_by_name={"flagged": frozenset()},
        sources_by_name={"flagged": frozenset({"download"})},
    )

    assert (first.name, first.dtype, first.granularity) == (
        "score",
        "numeric",
        ColumnGranularity.RESIDUE,
    )
    assert first.description == "The score."
    assert first.profiles == frozenset({PROFILE_ML_MINIMAL})
    assert (second.dtype, second.granularity) == ("text", ColumnGranularity.STRUCTURE)
    assert second.description == "Scorer engine output: flagged."
    assert second.profiles == frozenset()
    assert second.sources == frozenset({"download"})
    assert first.toggle == second.toggle == "include_scores"


def _always(_options: EngineOptions) -> bool:
    return True


@pytest.mark.parametrize(
    ("mapping", "value"),
    [
        ("enabled_if_by_name", _always),
        ("granularity_by_name", ColumnGranularity.CHAIN),
        ("dtype_by_name", "numeric"),
        ("descriptions", "Misspelled."),
        ("profiles_by_name", frozenset[str]()),
        ("sources_by_name", frozenset({"download"})),
    ],
)
def test_column_specs_refuse_a_setting_for_a_column_they_do_not_declare(
    mapping: str, value: object
) -> None:
    # A misspelled name would otherwise leave "score" with the shared setting.
    settings: dict[str, Any] = {mapping: {"scroe": value}}

    with pytest.raises(
        ValueError,
        match=rf"{mapping} names columns that are not declared: \['scroe'\]",
    ):
        column_specs(("score",), group="Scorer Fields", **settings)


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        (frozenset[str](), r"declares dataset sources \[\]"),
        (frozenset({"uploads"}), r"declares dataset sources \['uploads'\]"),
    ],
)
def test_a_column_and_an_engine_refuse_sources_no_run_can_have(
    sources: frozenset[str], message: str
) -> None:
    declared = cast(frozenset[DatasetSource], sources)

    with pytest.raises(ValueError, match=f"Column 'score' {message}"):
        ColumnSpec("score", group="Scorer Fields", sources=declared)
    with pytest.raises(ValueError, match=f"Engine 'scorer' {message}"):
        _engine(sources=declared)


def test_a_column_and_an_engine_refuse_an_unknown_profile() -> None:
    with pytest.raises(
        ValueError, match=r"Column 'score' names unknown profile\(s\) \['ml-minimal'\]"
    ):
        ColumnSpec("score", group="Scorer Fields", profiles=frozenset({"ml-minimal"}))
    with pytest.raises(
        ValueError, match=r"Engine 'scorer' names unknown profile\(s\) \['all'\]"
    ):
        _engine(profile_patterns=(("all", ("score*",)),))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "report_fields": (
                    ReportField("scorer_status", "Outcome."),
                    ReportField("scorer_status", "Outcome again."),
                )
            },
            r"declares a report field twice: \['scorer_status'\]",
        ),
        (
            {
                "options": (
                    OptionSpec("include_scores", label="Scores", help="Score."),
                    OptionSpec("include_scores", label="Again", help="Score."),
                )
            },
            r"declares an option twice: \['include_scores'\]",
        ),
    ],
)
def test_an_engine_refuses_a_report_field_or_option_declared_twice(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=f"'scorer' {message}"):
        _engine(**overrides)


def test_conditional_dependencies_must_be_declared_up_front() -> None:
    def needs(options: EngineOptions) -> tuple[str, ...]:
        return ("pdb_redo_comparison",) if options.flag("include_redo") else ("typo",)

    spec = _engine(
        dependencies=("rcsb_source",),
        conditional_dependency_names=("pdb_redo_comparison",),
        conditional_dependencies=needs,
    )

    assert spec.dependencies_under(None) == ("rcsb_source", "pdb_redo_comparison")
    assert spec.dependencies_under(EngineOptions(values={"include_redo": True})) == (
        "rcsb_source",
        "pdb_redo_comparison",
    )
    with pytest.raises(
        ValueError,
        match=(
            r"'scorer' depends on \['typo'\] without declaring them in "
            "conditional_dependency_names"
        ),
    ):
        spec.dependencies_under(EngineOptions())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"columns": (TableColumnSpec("site_id"), TableColumnSpec("site_id"))},
            r"'sites' declares a column twice: \['site_id'\]",
        ),
        (
            {"key": ("site_id", "site_id")},
            r"'sites' lists a key column twice: \['site_id'\]",
        ),
        (
            {"links": (TableLink(target="atom", columns=()),)},
            "'sites' declares a link with no columns",
        ),
    ],
)
def test_a_table_refuses_repeated_names_and_an_empty_link(
    overrides: dict[str, object], message: str
) -> None:
    arguments: dict[str, Any] = {
        "name": "sites",
        "description": "",
        "key": ("site_id",),
        "columns": (TableColumnSpec("site_id"),),
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        TableSpec(**arguments)


def test_a_table_column_names_the_category_it_lists_twice() -> None:
    with pytest.raises(
        ValueError,
        match=r"'site_type' declares duplicate categories: \['metal'\]",
    ):
        TableColumnSpec("site_type", categories=("metal", "control", "metal"))


def test_a_report_field_refuses_an_unknown_kind() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Report field 'scorer_rate' declares kind 'percent'; use one of "
            "text, flag, number, fraction"
        ),
    ):
        ReportField("scorer_rate", "Rate.", kind=cast(ReportKind, "percent"))


def test_an_engine_result_cannot_change_once_returned() -> None:
    result = EngineResult(warnings=("kept",))

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.warnings = ()  # type: ignore[misc]
