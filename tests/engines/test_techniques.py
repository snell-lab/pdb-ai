from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import pytest

from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.compatibility import (
    compatible_toggles,
    require_compatible_options,
    technique_conflicts,
    toggle_compatibility,
)
from pdbai.engines.core import (
    ALL_XRAY_SOURCES,
    DEFAULT_TECHNIQUE,
    TECHNIQUE_IDS,
    TECHNIQUES,
    XRAY_SOURCE_IDS,
    XRAY_SOURCES,
    EngineContext,
    EngineOptions,
    EnginePhase,
    EngineResult,
    EngineSpec,
    OptionSpec,
    classify_methods,
    classify_sources,
    get_technique,
    get_xray_source,
    normalize_source,
)


def test_vocabulary_mirrors_the_pdb_method_list_one_to_one() -> None:
    methods = [technique.method for technique in TECHNIQUES]

    assert len(set(methods)) == len(methods)
    assert len(set(TECHNIQUE_IDS)) == len(TECHNIQUE_IDS)
    assert DEFAULT_TECHNIQUE in TECHNIQUE_IDS
    assert get_technique("x_ray_diffraction").method == "X-RAY DIFFRACTION"
    assert get_technique("neutron_diffraction").method == "NEUTRON DIFFRACTION"
    assert get_technique("integrative").method == "INTEGRATIVE"
    with pytest.raises(ValueError, match="technique must be one of"):
        get_technique("xray")


def test_common_acronyms_are_shown_alongside_full_names() -> None:
    by_id = {technique.id: technique for technique in TECHNIQUES}

    assert by_id["electron_microscopy"].display_label == (
        "Cryogenic electron microscopy (cryo-EM)"
    )
    assert by_id["solution_nmr"].display_label == (
        "Solution nuclear magnetic resonance (solution NMR)"
    )
    assert by_id["electron_crystallography"].display_label == (
        "Electron crystallography (MicroED)"
    )
    assert by_id["epr"].display_label == "Electron paramagnetic resonance (EPR)"
    assert by_id["fluorescence_transfer"].display_label == (
        "Fluorescence resonance energy transfer (FRET)"
    )
    # Techniques without a common acronym show the name alone.
    assert by_id["neutron_diffraction"].acronym is None
    assert by_id["neutron_diffraction"].display_label == "Neutron diffraction"
    assert get_xray_source("free_electron_laser").display_label == (
        "X-ray free-electron laser (XFEL)"
    )
    assert get_xray_source("synchrotron").display_label == "Synchrotron"


def test_techniques_and_sources_serialize_under_snake_case_keys() -> None:
    assert get_technique("electron_microscopy").to_dict() == {
        "id": "electron_microscopy",
        "label": "Cryogenic electron microscopy",
        "acronym": "cryo-EM",
        "display_label": "Cryogenic electron microscopy (cryo-EM)",
        "method": "ELECTRON MICROSCOPY",
    }
    assert get_xray_source("synchrotron").to_dict() == {
        "id": "synchrotron",
        "label": "Synchrotron",
        "acronym": None,
        "display_label": "Synchrotron",
        "source": "SYNCHROTRON",
    }


def test_classification_keeps_every_method_a_joint_entry_reports() -> None:
    assert classify_methods(["X-RAY DIFFRACTION"]) == ("x_ray_diffraction",)
    assert classify_methods(["NEUTRON DIFFRACTION", "X-RAY DIFFRACTION"]) == (
        "x_ray_diffraction",
        "neutron_diffraction",
    )
    # Spelling and spacing variations in deposited files are tolerated.
    assert classify_methods([" x-ray   diffraction "]) == ("x_ray_diffraction",)
    # An unrecognized method is ignored rather than mapped to a guess.
    assert classify_methods(["SOMETHING NEW"]) == ()
    assert classify_methods([], integrative=True) == ("integrative",)
    assert classify_methods(["", "  "]) == ()


def test_xray_source_vocabulary_normalizes_deposited_spellings() -> None:
    assert XRAY_SOURCE_IDS[0] == ALL_XRAY_SOURCES
    assert [source.id for source in XRAY_SOURCES] == [
        "synchrotron",
        "free_electron_laser",
        "rotating_anode",
        "sealed_tube",
        "liquid_anode",
    ]
    assert get_xray_source("free_electron_laser").source == "FREE ELECTRON LASER"
    with pytest.raises(ValueError, match="xray_source must be one of"):
        get_xray_source("all")
    with pytest.raises(ValueError, match="xray_source must be one of"):
        get_xray_source("xfel")

    # The three deposited spellings of a rotating anode collapse onto one
    # value.
    for spelling in (
        "ROTATING ANODE",
        "rotating-anode X-ray tube",
        "Rotating Copper Anode",
    ):
        assert normalize_source(spelling) == "ROTATING ANODE"
    assert normalize_source("free-electron laser") == "FREE ELECTRON LASER"
    assert normalize_source("  synchrotron ") == "SYNCHROTRON"
    # An unrecognized source is kept, upper-cased, rather than guessed at.
    assert normalize_source("Nuclear Reactor") == "NUCLEAR REACTOR"
    assert normalize_source("LaB6 THERMOIONIC SOURCE") == "LAB6 THERMOIONIC SOURCE"

    assert classify_sources(["SYNCHROTRON", "NUCLEAR REACTOR"]) == ("synchrotron",)
    assert classify_sources(["rotating-anode X-ray tube"]) == ("rotating_anode",)
    assert classify_sources(["NUCLEAR REACTOR"]) == ()
    assert classify_sources([]) == ()


def test_xray_source_option_applies_only_to_x_ray_diffraction() -> None:
    assert EngineOptions().xray_source == ALL_XRAY_SOURCES
    options = EngineOptions(xray_source="free_electron_laser")
    assert EngineOptions.from_settings_dict(options.as_settings_dict()) == options

    with pytest.raises(ValueError, match="xray_source must be one of"):
        EngineOptions(xray_source="xfel")
    with pytest.raises(ValueError, match="applies only to technique"):
        EngineOptions(
            technique="electron_microscopy",
            xray_source="synchrotron",
            values={"include_rabdam": False},
        )
    # The default X-ray source is all, which filters nothing. It is therefore
    # valid for every technique.
    EngineOptions(technique="electron_microscopy", values={"include_rabdam": False})


def test_engine_options_validate_and_round_trip_the_technique() -> None:
    assert EngineOptions().technique == DEFAULT_TECHNIQUE
    with pytest.raises(ValueError, match="technique must be one of"):
        EngineOptions(technique="cryo")

    options = EngineOptions(
        technique="electron_microscopy", values={"include_rabdam": False}
    )
    settings = options.as_settings_dict()
    assert settings["technique"] == "electron_microscopy"
    assert EngineOptions.from_settings_dict(settings) == options
    with pytest.raises(ValueError, match="missing: technique"):
        EngineOptions.from_settings_dict(
            {key: value for key, value in settings.items() if key != "technique"}
        )


def test_engine_spec_declares_technique_support() -> None:
    everywhere = EngineSpec(
        name="all", display_name="All", phase=EnginePhase.ANNOTATION
    )
    xray_only = EngineSpec(
        name="xray",
        display_name="X-ray",
        phase=EnginePhase.ANNOTATION,
        techniques=frozenset({"x_ray_diffraction"}),
    )

    assert everywhere.supports_technique("solution_nmr")
    assert xray_only.supports_technique("x_ray_diffraction")
    assert not xray_only.supports_technique("solution_nmr")
    with pytest.raises(ValueError, match="unknown technique"):
        EngineSpec(
            name="typo",
            display_name="Typo",
            phase=EnginePhase.ANNOTATION,
            techniques=frozenset({"xray"}),
        )


def test_builtin_engines_declare_the_expected_technique_support() -> None:
    registry = get_engine_registry()

    assert registry.get("rabdam").spec.techniques == frozenset({"x_ray_diffraction"})
    assert registry.get("pdb_redo_comparison").spec.techniques == frozenset(
        {"x_ray_diffraction"}
    )
    assert registry.get("pdb_redo_source").spec.techniques == frozenset(
        {"x_ray_diffraction"}
    )
    assert registry.get("sasa").spec.techniques is None
    assert registry.get("disulfides").spec.techniques is None
    assert registry.get("rcsb_source").spec.techniques is None
    assert registry.get("alphafold_database").spec.techniques is None
    validation = registry.get("validation").spec.techniques
    assert validation is not None
    assert {"x_ray_diffraction", "electron_microscopy", "solution_nmr"} <= validation
    assert "epr" not in validation


def test_toggle_compatibility_covers_every_boolean_option() -> None:
    by_toggle = {record.toggle: record for record in toggle_compatibility()}

    assert by_toggle["include_rabdam"].techniques == frozenset({"x_ray_diffraction"})
    assert by_toggle["include_rabdam"].label == "RABDAM"
    assert by_toggle["include_pdb_redo"].techniques == frozenset({"x_ray_diffraction"})
    # An acquisition and annotation pair reports under the annotation engine's
    # name.
    assert by_toggle["include_pdb_redo"].label == "PDB-Redo comparison"
    assert by_toggle["include_validation"].label == "wwPDB validation"
    assert by_toggle["include_sasa"].techniques is None
    assert by_toggle["include_disulfides"].techniques is None
    assert by_toggle["include_uniprot"].techniques is None
    assert by_toggle["include_rabdam"].to_dict() == {
        "toggle": "include_rabdam",
        "label": "RABDAM",
        "techniques": ["x_ray_diffraction"],
    }
    assert by_toggle["include_sasa"].to_dict()["techniques"] is None

    assert "include_rabdam" in compatible_toggles("x_ray_diffraction")
    assert "include_rabdam" not in compatible_toggles("electron_microscopy")
    assert "include_sasa" in compatible_toggles("electron_microscopy")


def test_technique_conflicts_name_the_offending_toggle_once() -> None:
    assert technique_conflicts(EngineOptions()) == []
    assert (
        technique_conflicts(
            EngineOptions(
                technique="electron_microscopy", values={"include_rabdam": False}
            )
        )
        == []
    )

    # RABDAM pulls in the PDB-Redo source engine. Only the flag the user
    # turned on is named, not the engines it brings along.
    defaults = get_engine_registry().with_defaults
    assert technique_conflicts(
        defaults(
            EngineOptions(
                technique="electron_microscopy", values={"include_rabdam": True}
            )
        )
    ) == ["RABDAM (include_rabdam) is not available for Cryogenic electron microscopy"]
    assert (
        technique_conflicts(defaults(EngineOptions(technique="electron_microscopy")))
        == []
    )

    messages = technique_conflicts(
        EngineOptions(
            technique="solution_nmr",
            values={
                "include_rabdam": False,
                "include_pdb_redo": True,
                "include_sasa": True,
            },
        )
    )
    assert messages == [
        "PDB-Redo comparison (include_pdb_redo) is not available for Solution "
        "nuclear magnetic resonance",
    ]

    with pytest.raises(
        ValueError, match="RABDAM \\(include_rabdam\\) is not available"
    ):
        require_compatible_options(
            defaults(
                EngineOptions(
                    technique="fiber_diffraction", values={"include_rabdam": True}
                )
            )
        )
    require_compatible_options(EngineOptions(technique="fiber_diffraction"))


@dataclass
class _StubEngine:
    spec: EngineSpec

    def run(self, context: EngineContext) -> EngineResult:
        return EngineResult(columns=pd.DataFrame(index=context.atom_dataframe.index))


def test_technique_conflicts_report_an_engine_without_a_toggle_by_name(
    install_engines: Callable[..., None],
) -> None:
    install_engines(
        _StubEngine(
            EngineSpec(
                name="em_only",
                display_name="EM only",
                phase=EnginePhase.ANNOTATION,
                default_enabled=True,
                techniques=frozenset({"electron_microscopy"}),
            )
        )
    )
    options = EngineOptions(values={"include_rabdam": False})

    assert technique_conflicts(options) == [
        "Engine 'em_only' is not available for X-ray diffraction"
    ]
    assert (
        technique_conflicts(
            EngineOptions(
                technique="electron_microscopy", values={"include_rabdam": False}
            )
        )
        == []
    )


def _flag(name: str, *, refines: str | None = None) -> OptionSpec:
    return OptionSpec(name, label=name.title(), help="Set.", refines=refines)


def test_a_supported_engine_cannot_bring_in_an_unsupported_dependency(
    install_engines: Callable[..., None],
) -> None:
    fetcher = EngineSpec(
        name="xray_fetcher",
        display_name="X-ray fetcher",
        phase=EnginePhase.ACQUISITION,
        techniques=frozenset({"x_ray_diffraction"}),
    )
    scorer = EngineSpec(
        name="scorer",
        display_name="Scorer",
        phase=EnginePhase.ANNOTATION,
        toggle="include_scorer",
        dependencies=("xray_fetcher",),
        options=(_flag("include_scorer"),),
    )
    install_engines(_StubEngine(fetcher), _StubEngine(scorer))
    options = EngineOptions(technique="solution_nmr", values={"include_scorer": True})

    assert technique_conflicts(options) == [
        "Engine 'xray_fetcher' is not available for Solution nuclear magnetic resonance"
    ]


def test_a_refining_option_applies_only_where_its_flag_applies(
    install_engines: Callable[..., None],
) -> None:
    spec = EngineSpec(
        name="xray",
        display_name="X-ray",
        phase=EnginePhase.ANNOTATION,
        toggle="include_xray",
        techniques=frozenset({"x_ray_diffraction"}),
        options=(
            _flag("include_xray"),
            _flag("xray_detail", refines="include_xray"),
        ),
    )
    install_engines(_StubEngine(spec))

    by_toggle = {record.toggle: record for record in toggle_compatibility()}

    assert by_toggle["xray_detail"].techniques == frozenset({"x_ray_diffraction"})
    assert {"include_xray", "xray_detail"} <= compatible_toggles("x_ray_diffraction")
    assert not {"include_xray", "xray_detail"} & compatible_toggles("solution_nmr")


def test_compatible_toggles_refuses_an_unknown_technique() -> None:
    with pytest.raises(ValueError, match="technique must be one of"):
        compatible_toggles("cryo")
