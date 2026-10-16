from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import cast

import pandas as pd
import pytest

from pdbai.engines.compatibility import validate_options
from pdbai.engines.core import (
    ArtifactKey,
    ColumnSpec,
    EngineContext,
    EngineExecutionError,
    EngineOptions,
    EnginePhase,
    EngineRegistry,
    EngineResult,
    EngineSpec,
    OptionSpec,
    ReportField,
    run_engine_phase,
)
from pdbai.engines.core.techniques import get_xray_source

REPORT = ArtifactKey("report", object)
LEAKED = ArtifactKey("leaked", object)


@dataclass
class StubEngine:
    spec: EngineSpec
    values: dict[str, object] = field(default_factory=dict[str, object])
    calls: list[str] = field(default_factory=list[str])
    failure: Exception | None = None
    warnings: tuple[str, ...] = ()
    artifacts: dict[ArtifactKey[object], object] = field(
        default_factory=dict[ArtifactKey[object], object]
    )
    index: pd.Index | None = None
    result_columns: pd.DataFrame | None = None

    def run(self, context: EngineContext) -> EngineResult:
        self.calls.append(self.spec.name)
        if self.failure is not None:
            raise self.failure
        output_index = (
            context.atom_dataframe.index if self.index is None else self.index
        )
        columns = self.result_columns
        if columns is None:
            columns = pd.DataFrame(
                {
                    name: [value] * len(output_index)
                    for name, value in self.values.items()
                },
                index=output_index,
            )
        return EngineResult(
            columns=columns,
            warnings=self.warnings,
            artifacts=self.artifacts,
        )


def _spec(
    name: str,
    *,
    output_columns: tuple[str, ...] = (),
    columns: tuple[ColumnSpec, ...] | None = None,
    dependencies: tuple[str, ...] = (),
    phase: EnginePhase = EnginePhase.ANNOTATION,
    toggle: str | None = None,
    default_enabled: bool = True,
    required: bool = True,
    provides: tuple[ArtifactKey[object], ...] = (),
) -> EngineSpec:
    # Stub engines are required unless a test says otherwise, so a rejected
    # result surfaces as an error rather than a recorded failure.
    return EngineSpec(
        name=name,
        display_name=name.title(),
        phase=phase,
        toggle=toggle,
        dependencies=dependencies,
        required=required,
        provides=provides,
        columns=(
            columns
            if columns is not None
            else tuple(
                ColumnSpec(column, group="Test Fields") for column in output_columns
            )
        ),
        default_enabled=default_enabled,
    )


def _context(options: EngineOptions | None = None) -> EngineContext:
    return EngineContext(
        pdb_id="1ABC",
        atom_dataframe=pd.DataFrame({"atom_name": ["N", "CA"]}, index=[10, 20]),
        options=options or EngineOptions(),
    )


def test_registry_orders_dependencies_before_consumers() -> None:
    calls: list[str] = []
    dependency = StubEngine(_spec("source", default_enabled=False), calls=calls)
    consumer = StubEngine(_spec("consumer", dependencies=("source",)), calls=calls)
    registry = EngineRegistry([consumer, dependency])
    options = EngineOptions()

    ordered = registry.engines_to_run(phase=EnginePhase.ANNOTATION, options=options)

    assert [engine.spec.name for engine in ordered] == ["source", "consumer"]


def test_registry_includes_cross_phase_dependencies_in_their_phase() -> None:
    source = StubEngine(
        _spec(
            "source",
            phase=EnginePhase.ACQUISITION,
            default_enabled=False,
        )
    )
    consumer = StubEngine(_spec("consumer", dependencies=("source",)))
    registry = EngineRegistry([consumer, source])
    options = EngineOptions()

    acquisition = registry.engines_to_run(
        phase=EnginePhase.ACQUISITION, options=options
    )
    annotation = registry.engines_to_run(phase=EnginePhase.ANNOTATION, options=options)

    assert [engine.spec.name for engine in acquisition] == ["source"]
    assert [engine.spec.name for engine in annotation] == ["consumer"]


def test_registry_rejects_duplicate_unknown_and_cyclic_dependencies() -> None:
    duplicate = StubEngine(_spec("same"))
    with pytest.raises(ValueError, match="already registered"):
        EngineRegistry([duplicate, duplicate])

    unknown = StubEngine(_spec("consumer", dependencies=("missing",)))
    with pytest.raises(ValueError, match="unknown engine"):
        EngineRegistry([unknown]).engines_to_run()

    left = StubEngine(_spec("left", dependencies=("right",)))
    right = StubEngine(_spec("right", dependencies=("left",)))
    with pytest.raises(ValueError, match="dependency cycle"):
        EngineRegistry([left, right]).engines_to_run()


@pytest.mark.parametrize("link", ["dependencies", "conditional"])
def test_registry_refuses_a_dependency_on_a_later_phase(link: str) -> None:
    annotation = StubEngine(_spec("annotation"))
    acquisition = StubEngine(
        EngineSpec(
            name="acquisition",
            display_name="Acquisition",
            phase=EnginePhase.ACQUISITION,
            dependencies=("annotation",) if link == "dependencies" else (),
            conditional_dependency_names=(
                ("annotation",) if link == "conditional" else ()
            ),
        )
    )
    registry = EngineRegistry([annotation, acquisition])

    with pytest.raises(
        ValueError,
        match=(
            "Engine 'acquisition' runs in the acquisition phase but needs "
            "engine 'annotation' first, which runs in the later annotation phase"
        ),
    ):
        registry.validate()


def test_registry_accepts_a_dependency_on_an_earlier_phase() -> None:
    source = StubEngine(_spec("source", phase=EnginePhase.ACQUISITION))
    consumer = StubEngine(_spec("consumer", dependencies=("source",)))

    EngineRegistry([consumer, source]).validate()


def test_registry_validation_names_an_unknown_dependency() -> None:
    unknown = StubEngine(_spec("consumer", dependencies=("missing",)))

    with pytest.raises(
        ValueError, match="Engine 'consumer' refers to unknown engine 'missing'"
    ):
        EngineRegistry([unknown]).validate()


def test_registry_validation_refuses_a_dependency_cycle() -> None:
    left = StubEngine(_spec("left", dependencies=("right",)))
    right = StubEngine(_spec("right", dependencies=("left",)))

    with pytest.raises(
        ValueError, match="Engine dependency cycle: left -> right -> left"
    ):
        EngineRegistry([left, right]).validate()


@pytest.mark.parametrize(
    ("refines", "cycle"),
    [
        ({"include_a": "include_a"}, "include_a -> include_a"),
        (
            {"include_a": "include_b", "include_b": "include_a"},
            "include_a -> include_b -> include_a",
        ),
    ],
)
def test_registry_validation_refuses_options_that_refine_one_another(
    refines: dict[str, str], cycle: str
) -> None:
    spec = EngineSpec(
        name="scorer",
        display_name="Scorer",
        phase=EnginePhase.ANNOTATION,
        options=tuple(
            OptionSpec(name, label=name, help="Set.", refines=refined)
            for name, refined in refines.items()
        ),
    )

    with pytest.raises(
        ValueError, match=f"Options refine one another in a cycle: {cycle}"
    ):
        EngineRegistry([StubEngine(spec)]).validate()


def test_coordinator_merges_declared_columns_and_collects_result_metadata() -> None:
    engine = StubEngine(
        _spec("quality", output_columns=("quality_score",), provides=(REPORT,)),
        values={"quality_score": 0.75},
        warnings=("quality warning",),
        artifacts={REPORT: {"source": "test"}},
    )
    context = _context()

    report = run_engine_phase(
        context, EnginePhase.ANNOTATION, registry=EngineRegistry([engine])
    )

    assert context.atom_dataframe["quality_score"].tolist() == [0.75, 0.75]
    assert context.artifacts.get(REPORT) == {"source": "test"}
    assert report.warnings == ("quality warning",)


def test_coordinator_rejects_undeclared_columns_and_wrong_index() -> None:
    undeclared = StubEngine(_spec("undeclared"), values={"surprise": 1})
    with pytest.raises(EngineExecutionError, match="undeclared columns") as caught:
        run_engine_phase(
            _context(),
            EnginePhase.ANNOTATION,
            registry=EngineRegistry([undeclared]),
        )

    assert caught.value.engine_name == "undeclared"
    assert isinstance(caught.value.__cause__, ValueError)

    wrong_index = StubEngine(
        _spec("misaligned", output_columns=("score",)),
        values={"score": 1},
        index=pd.Index([0, 1]),
    )
    with pytest.raises(EngineExecutionError, match="atom DataFrame's index"):
        run_engine_phase(
            _context(),
            EnginePhase.ANNOTATION,
            registry=EngineRegistry([wrong_index]),
        )


def test_coordinator_leaves_the_context_unchanged_when_a_merge_fails() -> None:
    context = _context()
    before = context.atom_dataframe.copy()

    broken = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6]],
        columns=["ok", "bad", "bad"],
        index=context.atom_dataframe.index,
    )
    engine = StubEngine(
        _spec("partial", output_columns=("ok", "bad"), provides=(LEAKED,)),
        result_columns=broken,
        warnings=("late",),
        artifacts={LEAKED: True},
    )

    with pytest.raises(EngineExecutionError):
        run_engine_phase(
            context, EnginePhase.ANNOTATION, registry=EngineRegistry([engine])
        )

    assert context.atom_dataframe.equals(before)
    assert LEAKED not in context.artifacts


def test_coordinator_names_the_empty_result_dataframe_mistake() -> None:
    engine = StubEngine(_spec("empty"), result_columns=pd.DataFrame())

    with pytest.raises(EngineExecutionError, match="return columns=None"):
        run_engine_phase(
            _context(), EnginePhase.ANNOTATION, registry=EngineRegistry([engine])
        )


def test_coordinator_recovers_from_rejected_engine_results() -> None:
    context = _context()
    rejected = StubEngine(
        _spec("optional", output_columns=("kept",), required=False),
        values={"surprise": 1},
    )
    later = StubEngine(_spec("later", output_columns=("score",)), values={"score": 2})

    run_engine_phase(
        context,
        EnginePhase.ANNOTATION,
        registry=EngineRegistry([rejected, later]),
    )

    assert "undeclared columns" in context.failures["optional"]
    assert "surprise" not in context.atom_dataframe.columns
    # The failed engine's declared column is present and empty, so every
    # structure's partial file shares one header.
    assert context.atom_dataframe["kept"].isna().all()
    assert context.atom_dataframe["score"].tolist() == [2, 2]


def test_only_a_required_engine_failure_is_fatal() -> None:
    fatal = StubEngine(_spec("essential"), failure=RuntimeError("offline"))

    with pytest.raises(EngineExecutionError, match="'essential' failed") as caught:
        run_engine_phase(
            _context(), EnginePhase.ANNOTATION, registry=EngineRegistry([fatal])
        )
    assert caught.value.engine_name == "essential"

    optional = StubEngine(
        _spec("optional", required=False), failure=RuntimeError("offline")
    )
    context = _context()
    recovered = run_engine_phase(
        context, EnginePhase.ANNOTATION, registry=EngineRegistry([optional])
    )
    assert context.failures == {"optional": "offline"}
    assert recovered.warnings == ("Optional failed: offline",)


def test_a_failed_engine_reports_its_declared_failure_values() -> None:
    spec = EngineSpec(
        name="scorer",
        display_name="Scorer",
        phase=EnginePhase.ANNOTATION,
        default_enabled=True,
        report_fields=(
            ReportField("scorer_status", "Outcome.", on_failure="failed"),
            ReportField("scorer_message", "Why.", carries_message=True),
            ReportField("scorer_rate", "Rate.", kind="number"),
        ),
    )
    failing = StubEngine(spec, failure=RuntimeError("no data"))

    report = run_engine_phase(
        _context(), EnginePhase.ANNOTATION, registry=EngineRegistry([failing])
    )

    assert report.report == {
        "scorer_status": "failed",
        "scorer_message": "no data",
        "scorer_rate": None,
    }


def test_an_engine_cannot_report_or_publish_what_it_did_not_declare() -> None:
    class Reporter:
        spec = EngineSpec(
            name="reporter",
            display_name="Reporter",
            phase=EnginePhase.ANNOTATION,
            default_enabled=True,
            required=True,
        )

        def __init__(self, result: EngineResult) -> None:
            self.result = result

        def run(self, context: EngineContext) -> EngineResult:
            return self.result

    for result, message in (
        (EngineResult(report={"surprise": 1}), "undeclared fields"),
        (EngineResult(artifacts={REPORT: 1}), "undeclared artifacts"),
    ):
        with pytest.raises(EngineExecutionError, match=message):
            run_engine_phase(
                _context(),
                EnginePhase.ANNOTATION,
                registry=EngineRegistry([Reporter(result)]),
            )


def test_a_wrong_typed_artifact_leaves_the_context_untouched() -> None:
    # Typed as object keys, so that the stub can return a value of any type.
    first: ArtifactKey[object] = ArtifactKey("first", str)
    count: ArtifactKey[object] = ArtifactKey("count", int)
    context = _context()
    engine = StubEngine(
        _spec(
            "publisher",
            output_columns=("score",),
            required=False,
            provides=(first, count),
        ),
        values={"score": 7},
        artifacts={first: "published first", count: "not a count"},
    )

    run_engine_phase(context, EnginePhase.ANNOTATION, registry=EngineRegistry([engine]))

    assert context.failures["publisher"] == (
        "Engine 'publisher' published artifact 'count' of type str; its key "
        "requires type int"
    )
    # The failed engine's column is filled empty, not left with its values.
    assert context.atom_dataframe["score"].isna().all()
    assert first not in context.artifacts
    assert count not in context.artifacts


def test_registry_rejects_engines_gated_by_unknown_options() -> None:
    misspelled = StubEngine(_spec("typo", toggle="include_dsp"))
    with pytest.raises(ValueError, match="'typo' is gated by 'include_dsp'"):
        EngineRegistry([misspelled]).validate()

    gated_column = ColumnSpec("score", group="Test Fields", toggle="include_scores")
    with pytest.raises(ValueError, match="'scorer' is gated by 'include_scores'"):
        EngineRegistry(
            [StubEngine(_spec("scorer", columns=(gated_column,)))]
        ).validate()

    # An undeclared flag reads as off rather than crashing a run.
    assert not misspelled.spec.is_enabled(EngineOptions())
    assert not gated_column.is_enabled(EngineOptions(), source="download")


def test_registry_accepts_a_toggle_its_engine_declares() -> None:
    declared = EngineSpec(
        name="scorer",
        display_name="Scorer",
        phase=EnginePhase.ANNOTATION,
        toggle="include_scores",
        options=(OptionSpec("include_scores", label="Scores", help="Score."),),
    )
    registry = EngineRegistry([StubEngine(declared)])
    registry.validate()

    assert registry.engines_to_run(options=EngineOptions()) == ()
    enabled = EngineOptions(values={"include_scores": True})
    assert [
        engine.spec.name for engine in registry.engines_to_run(options=enabled)
    ] == ["scorer"]


def test_predicate_decides_enablement_while_toggle_stays_attribution() -> None:
    # A column available under either of two flags still names one owning
    # toggle for profiles and reports. That name must not be read as the rule.
    column = ColumnSpec(
        "span",
        group="Test Fields",
        toggle="include_uniprot",
        enabled_if=lambda options: (
            options.flag("include_uniprot") or options.flag("include_alphafold")
        ),
    )
    assert column.is_enabled(
        EngineOptions(values={"include_alphafold": True}), source="download"
    )
    assert not column.is_enabled(EngineOptions(), source="download")


def test_registry_rejects_engines_gated_by_non_boolean_options() -> None:
    choice = EngineSpec(
        name="scorer",
        display_name="Scorer",
        phase=EnginePhase.ANNOTATION,
        toggle="scorer_mode",
        options=(
            OptionSpec(
                "scorer_mode",
                label="Mode",
                help="Mode.",
                kind="choice",
                default="fast",
                choices=("fast", "slow"),
            ),
        ),
    )

    with pytest.raises(ValueError, match="no engine declares as a flag"):
        EngineRegistry([StubEngine(choice)]).validate()


def test_registry_accepts_engines_gated_by_real_toggles() -> None:
    gated = StubEngine(
        _spec(
            "surface",
            toggle="include_sasa",
            columns=(
                ColumnSpec("area", group="Test Fields"),
                ColumnSpec("area_ok", group="Test Fields", toggle="include_validation"),
            ),
        )
    )
    registry = EngineRegistry([gated])

    assert (
        registry.engines_to_run(options=EngineOptions(values={"include_sasa": False}))
        == ()
    )
    enabled = registry.engines_to_run(
        options=EngineOptions(values={"include_sasa": True})
    )
    assert [engine.spec.name for engine in enabled] == ["surface"]
    assert [
        column.name
        for column in gated.spec.columns
        if not column.is_enabled(
            EngineOptions(values={"include_sasa": True}), source="download"
        )
    ] == ["area_ok"]


def test_registry_rejects_one_column_declared_by_two_engines() -> None:
    always = ColumnSpec("score", group="Test Fields", enabled_if=lambda _options: True)
    never = ColumnSpec("score", group="Test Fields", enabled_if=lambda _options: False)

    assert always == never
    registry = EngineRegistry([StubEngine(_spec("first", columns=(always,)))])

    with pytest.raises(
        ValueError, match="declared by both engines 'first' and 'second'"
    ):
        registry.register(StubEngine(_spec("second", columns=(never,))))


def test_spec_rejects_one_column_declared_twice_by_one_engine() -> None:
    duplicated = (
        ColumnSpec("score", group="Test Fields"),
        ColumnSpec("score", group="Test Fields", dtype="numeric"),
    )

    with pytest.raises(ValueError, match="'solo' declares a column twice"):
        _spec("solo", columns=duplicated)


def test_spec_refuses_columns_on_an_acquisition_engine() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Engine 'fetcher' declares columns but runs in the acquisition "
            "phase; only annotation engines add columns"
        ),
    ):
        EngineSpec(
            name="fetcher",
            display_name="Fetcher",
            phase=EnginePhase.ACQUISITION,
            columns=(ColumnSpec("score", group="Test Fields"),),
        )


@pytest.mark.parametrize(
    ("kind", "raw", "message"),
    [
        ("flag", "yes", "^setting must be true or false; got 'yes'$"),
        ("flag", 1, "^setting must be true or false; got 1$"),
        ("choice", "medium", "setting must be one of fast, slow"),
        ("choice", 1, "^setting must be one of fast, slow; got 1$"),
        ("count", "0", "setting must be a positive integer"),
        ("count", True, "setting must be a positive integer"),
    ],
)
def test_option_parse_refuses_every_bad_value_with_value_error(
    kind: str, raw: object, message: str
) -> None:
    option = {
        "flag": OptionSpec("setting", label="Setting", help="Set."),
        "choice": OptionSpec(
            "setting",
            label="Setting",
            help="Set.",
            kind="choice",
            default="fast",
            choices=("fast", "slow"),
        ),
        "count": OptionSpec(
            "setting", label="Setting", help="Set.", kind="count", default=None
        ),
    }[kind]

    with pytest.raises(ValueError, match=message):
        option.parse(raw)


def test_engine_options_reject_values_outside_their_choices() -> None:
    with pytest.raises(ValueError, match="rabdam_source must be one of"):
        validate_options(EngineOptions(values={"rabdam_source": "pdbredo"}))
    with pytest.raises(ValueError, match="sasa_accuracy_mode must be one of"):
        validate_options(EngineOptions(values={"sasa_accuracy_mode": "High"}))
    with pytest.raises(ValueError, match="must be a positive integer"):
        validate_options(EngineOptions(values={"sasa_n_sphere_points": 0}))
    # dataset_shape is a run setting rather than an engine's, so it is checked
    # as soon as the options are built.
    with pytest.raises(ValueError, match="dataset_shape must be one of"):
        EngineOptions.from_settings_dict(
            {**EngineOptions().as_settings_dict(), "dataset_shape": "star"}
        )

    accepted = EngineOptions(
        dataset_shape="star_schema",
        values={
            "rabdam_source": "both",
            "sasa_accuracy_mode": "fast",
            "sasa_n_sphere_points": 250,
        },
    )
    validate_options(accepted)
    assert accepted.choice("rabdam_source") == "both"


def test_settings_no_engine_declares_are_rejected() -> None:
    with pytest.raises(ValueError, match="No engine declares the settings"):
        validate_options(EngineOptions(values={"include_typo": True}))


def test_engine_options_reject_unusable_manifest_values() -> None:
    settings = EngineOptions().as_settings_dict()
    settings["engine_settings"] = {"rabdam_source": "pdbredo"}

    with pytest.raises(ValueError, match="rabdam_source must be one of"):
        validate_options(EngineOptions.from_settings_dict(settings))


def test_engine_options_reject_wrong_typed_manifest_settings() -> None:
    def stored(**overrides: object) -> dict[str, object]:
        settings = EngineOptions().as_settings_dict()
        settings.update(overrides)
        return settings

    def restored(**engine_settings: object) -> EngineOptions:
        return EngineOptions.from_settings_dict(stored(engine_settings=engine_settings))

    with pytest.raises(TypeError, match="must be a string"):
        EngineOptions.from_settings_dict(stored(technique=1))
    with pytest.raises(TypeError, match="'engine_settings' must be an object"):
        EngineOptions.from_settings_dict(stored(engine_settings=["include_sasa"]))
    with pytest.raises(TypeError, match="is not a plain value"):
        restored(include_x=[1])
    # Plain values of the wrong kind survive restoration. Validation then
    # checks each one against its engine's declaration.
    with pytest.raises(ValueError, match="include_sasa must be true or false"):
        validate_options(restored(include_sasa="false"))
    with pytest.raises(ValueError, match="rabdam_source must be one of"):
        validate_options(restored(rabdam_source=1))
    with pytest.raises(ValueError, match="must be a positive integer"):
        validate_options(restored(sasa_n_sphere_points="250"))
    with pytest.raises(ValueError, match="must be a positive integer"):
        validate_options(restored(sasa_n_sphere_points=True))


def test_engine_options_survive_a_json_manifest_round_trip() -> None:
    options = EngineOptions(
        dataset_shape="star_schema",
        values={"include_sasa": True, "sasa_n_sphere_points": 250},
    )

    written = json.dumps(options.as_settings_dict(), sort_keys=True, default=str)
    restored: object = json.loads(written)
    assert isinstance(restored, dict)

    settings = cast("dict[str, object]", restored)
    assert EngineOptions.from_settings_dict(settings) == options


def test_engine_options_round_trip_through_manifest_settings() -> None:
    options = EngineOptions(
        dataset_shape="star_schema",
        values={"include_rabdam": False, "include_sasa": True},
    )

    restored = EngineOptions.from_settings_dict(options.as_settings_dict())

    assert restored == options


def test_engine_options_ignore_the_engine_list_of_older_manifests() -> None:
    # Runs stored while options could switch engines on by name list them.
    settings = EngineOptions().as_settings_dict()
    settings["enabled_engines"] = ["rabdam", 1]

    assert EngineOptions.from_settings_dict(settings) == EngineOptions()


def test_engine_options_reject_incomplete_manifest_settings() -> None:
    with pytest.raises(ValueError, match="missing"):
        EngineOptions.from_settings_dict({"dataset_shape": "flat"})


def test_registry_fills_settings_left_out_with_their_declared_defaults() -> None:
    declared = EngineSpec(
        name="scorer",
        display_name="Scorer",
        phase=EnginePhase.ANNOTATION,
        toggle="include_scores",
        options=(
            OptionSpec("include_scores", label="Scores", help="Score.", default=True),
            OptionSpec(
                "score_mode",
                label="Mode",
                help="Mode.",
                kind="choice",
                default="fast",
                choices=("fast", "slow"),
            ),
            OptionSpec(
                "score_scale",
                label="Scale",
                help="Scale.",
                kind="choice",
                default="linear",
                choices=("linear", "log"),
            ),
            OptionSpec(
                "score_points",
                label="Points",
                help="Points.",
                kind="count",
                default=None,
            ),
        ),
    )
    registry = EngineRegistry([StubEngine(declared)])

    given = EngineOptions(values={"score_mode": "slow"})
    filled = registry.with_defaults(given)

    assert filled.values == {
        "score_mode": "slow",
        "include_scores": True,
        "score_scale": "linear",
        "score_points": None,
    }
    assert filled.flag("include_scores")
    assert filled.choice("score_mode") == "slow"
    assert filled.choice("score_scale") == "linear"
    assert filled.count("score_points") is None
    assert registry.with_defaults(filled) is filled
    # Without the registry's defaults, an unset flag reads as off. An unset
    # choice raises instead, because it holds no value to return.
    assert not given.flag("include_scores")
    with pytest.raises(KeyError, match="'score_scale' holds no choice"):
        given.choice("score_scale")


def test_xray_source_refusal_lists_only_the_sources_it_accepts() -> None:
    with pytest.raises(ValueError) as refusal:
        get_xray_source("all")

    assert str(refusal.value) == (
        "xray_source must be one of synchrotron, free_electron_laser, "
        "rotating_anode, sealed_tube, liquid_anode; got 'all'"
    )
