from __future__ import annotations

import re
from collections.abc import Callable
from typing import cast

import pytest

from pdbai.column_filtering import matches_glob
from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.core import (
    PROFILE_RESEARCH_READY,
    PROFILE_STRUCTURE_QUALITY,
    ColumnDtype,
    ColumnSpec,
    DatasetSource,
    EngineContext,
    EngineOptions,
    EnginePhase,
    EngineResult,
    EngineSpec,
    OptionSpec,
)
from pdbai.profiles import (
    PROFILES,
    ColumnProfile,
    apply_profile,
    get_profile,
    resolve_run_settings,
)
from pdbai.schema import (
    DatasetColumnSpec,
    column_granularity,
    columns_by_group,
    columns_with_role,
    declared_column_specs,
    describe_column,
    disabled_columns,
    get_column_spec,
    order_columns,
    schema_fingerprint,
)


class _Stub:
    def __init__(self, spec: EngineSpec) -> None:
        self.spec = spec

    def run(self, context: EngineContext) -> EngineResult:
        return EngineResult()


def _stub(
    name: str,
    *columns: ColumnSpec,
    toggle: str | None = None,
    dependencies: tuple[str, ...] = (),
    options: tuple[OptionSpec, ...] = (),
) -> _Stub:
    return _Stub(
        EngineSpec(
            name=name,
            display_name=name,
            phase=EnginePhase.ANNOTATION,
            toggle=toggle,
            dependencies=dependencies,
            columns=columns,
            options=options,
        )
    )


def test_curated_schema_has_owned_described_columns() -> None:
    specs = declared_column_specs()

    assert all(spec.owner for spec in specs)
    assert all(spec.column.group for spec in specs)
    assert all(spec.column.description for spec in specs)
    assert describe_column("pdb_id")


def test_adp_summaries_are_parser_owned_base_columns() -> None:
    for name in ("u_equivalent", "b_anisotropy"):
        spec = get_column_spec(name)
        assert spec is not None
        assert spec.owner == "parser"
        assert spec.toggle is None
        assert name not in disabled_columns(EngineOptions(), source="download")


def test_curated_schema_descriptions_are_semantic_not_generated_placeholders() -> None:
    placeholder_fragments = (
        "engine output:",
        "pipeline-derived analysis field:",
        "base mmcif or rcsb metadata field:",
        "publication metadata field:",
    )

    for spec in declared_column_specs():
        normalized = spec.column.description.casefold()
        assert not any(fragment in normalized for fragment in placeholder_fragments), (
            spec.name
        )

    assert "Cartesian x coordinate in Å" in describe_column("x")
    assert "serious all-atom steric clashes" in describe_column("clashscore")
    assert "PDB-Redo x coordinate minus" in describe_column("pdb_redo_x_delta")


def test_every_engine_column_is_exposed_by_the_application_schema() -> None:
    for engine in get_engine_registry().engines_to_run():
        for engine_column in engine.spec.columns:
            application_column = get_column_spec(engine_column.name)
            assert application_column is not None
            assert application_column.owner == engine.spec.name
            assert application_column.column is engine_column


def test_disabled_columns_follow_engine_options() -> None:
    defaults = disabled_columns(EngineOptions(), source="download")
    with_sasa = disabled_columns(
        EngineOptions(values={"include_sasa": True}), source="download"
    )

    assert "atom_sasa_protein" in defaults
    assert "atom_sasa_protein" not in with_sasa
    assert "pdb_id" not in defaults


def test_get_profile_ignores_case_and_rejects_an_unknown_identifier() -> None:
    for profile in PROFILES:
        assert get_profile(f" {profile.id.upper()} ") is profile
    with pytest.raises(ValueError, match="profile must be one of all, research_ready"):
        get_profile("not-a-profile")


def test_profiles_keep_their_member_columns_and_turn_on_their_toggles() -> None:
    flags = get_engine_registry().flag_names()
    for profile in PROFILES:
        members = [
            spec
            for spec in declared_column_specs()
            if profile.id in spec.column.profiles
        ]
        for spec in members:
            assert any(
                matches_glob(spec.name, selector) for selector in profile.columns
            ), (profile.id, spec.name)
        names = [selector for selector in profile.columns if "*" not in selector]
        assert set(names) <= {spec.name for spec in members}
        assert len(set(profile.columns)) == len(profile.columns)
        assert set(profile.toggles) == {spec.toggle for spec in members} - {None}
        assert list(profile.toggles) == sorted(profile.toggles, key=flags.index)


def test_the_all_profile_filters_nothing_and_turns_nothing_on() -> None:
    profile = get_profile("all")

    assert profile.filter_mode is None
    assert profile.columns == ()
    assert profile.toggles == ()


def test_a_profile_lists_source_variants_under_their_narrowest_glob() -> None:
    columns = get_profile(PROFILE_RESEARCH_READY).columns

    # bnet* also matches the percentile columns. Those are listed under the
    # narrower bnet_percentile*, so bnet* stays for the plain Bnet columns.
    assert "bnet*" in columns
    assert "bnet_percentile*" in columns
    assert [name for name in columns if name.startswith("bdamage")] == ["bdamage*"]


def test_uniprot_id_is_attributed_to_one_toggle_but_enabled_by_either() -> None:
    # This is the column that needs a toggle separate from the rule that
    # enables it. AlphaFold needs the UniProt mapping, so the column is
    # written under either switch. It is still reported as UniProt's.
    assert "uniprot_id" not in disabled_columns(
        EngineOptions(values={"include_uniprot": True}), source="download"
    )
    assert "uniprot_id" not in disabled_columns(
        EngineOptions(values={"include_alphafold": True}), source="download"
    )
    assert "uniprot_id" in disabled_columns(EngineOptions(), source="download")
    spec = get_column_spec("uniprot_id")
    assert spec is not None
    assert spec.toggle == "include_uniprot"


def test_column_groups_follow_engine_registration_order() -> None:
    assert [group for group, _columns in columns_by_group()] == [
        "PDB Fields",
        "Citation Fields",
        "PDB-Redo Fields",
        "RABDAM Fields",
        "SASA Fields",
        "Density Fields",
        "Disulfide Fields",
        "Metal Site Fields",
        "Residue Site Fields",
        "Selenium and Halogen Fields",
        "Cofactor Fields",
        "Site Environment Fields",
        "AlphaFold Database Fields",
    ]


def test_column_groups_follow_registration_order_not_dependency_order(
    install_engines: Callable[..., None],
) -> None:
    first = _stub(
        "first",
        ColumnSpec("first_value", group="First Fields"),
        dependencies=("second",),
    )
    second = _stub("second", ColumnSpec("second_value", group="Second Fields"))

    install_engines(first, second)

    assert columns_by_group()[-2:] == (
        ("First Fields", ("first_value",)),
        ("Second Fields", ("second_value",)),
    )
    assert order_columns(["second_value", "first_value"]) == [
        "first_value",
        "second_value",
    ]


def test_the_catalog_follows_the_registered_engines(
    install_engines: Callable[..., None],
) -> None:
    assert get_column_spec("extra_value") is None

    install_engines(_stub("extra", ColumnSpec("extra_value", group="Extra Fields")))

    spec = get_column_spec("extra_value")
    assert spec is not None
    assert spec.owner == "extra"


def test_an_engine_column_named_after_a_base_column_is_refused(
    install_engines: Callable[..., None],
) -> None:
    install_engines(_stub("clash", ColumnSpec("pdb_id", group="Clash Fields")))

    with pytest.raises(
        ValueError, match="'pdb_id' is declared by both 'parser' and 'clash'"
    ):
        declared_column_specs()


def test_an_undeclared_column_has_no_specification() -> None:
    assert get_column_spec("_atom_site.label_atom_id") is None
    assert column_granularity("_atom_site.label_atom_id") is None
    assert describe_column("_atom_site.label_atom_id") == ""


def test_columns_with_role_lists_key_columns_in_display_order() -> None:
    assert columns_with_role("structure_key") == ("pdb_id",)
    assert columns_with_role("chain_author_identity") == ("pdb_id", "chain_id")
    assert columns_with_role("residue_author_identity") == (
        "pdb_id",
        "chain_id",
        "res_num",
        "res_name",
    )


def test_a_column_refuses_a_join_column_name() -> None:
    with pytest.raises(ValueError, match="reserved for a star-schema join column"):
        DatasetColumnSpec(ColumnSpec("chain_key", group="Fields"), engine=None)


def test_an_engine_column_refuses_the_parser_name() -> None:
    column = ColumnSpec("value", group="Fields")

    assert DatasetColumnSpec(column, _stub("owner", column).spec).owner == "owner"
    with pytest.raises(ValueError, match="'parser' is reserved"):
        DatasetColumnSpec(column, _stub("parser", column).spec)


@pytest.mark.parametrize("name", ["", " value", "a,b", "a;b", "a\nb"])
def test_a_column_name_a_filter_could_not_select_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ColumnSpec(name, group="Fields")


def test_an_unknown_column_dtype_is_refused() -> None:
    with pytest.raises(ValueError, match="declares dtype 'bolean'"):
        ColumnSpec("value", group="Fields", dtype=cast(ColumnDtype, "bolean"))


def test_an_engine_gates_its_columns_whatever_their_own_toggle_says(
    install_engines: Callable[..., None],
) -> None:
    flags = tuple(
        OptionSpec(name=name, kind="flag", default=False, label=name, help=name)
        for name in ("include_gate", "include_detail")
    )
    install_engines(
        _stub(
            "gated",
            ColumnSpec("gated_detail", group="Gated Fields", toggle="include_detail"),
            toggle="include_gate",
            options=flags,
        )
    )

    def disabled(**values: bool) -> bool:
        options = EngineOptions(values=dict(values))
        return "gated_detail" in disabled_columns(options, source="download")

    assert disabled(include_detail=True)
    assert disabled(include_gate=True)
    assert not disabled(include_gate=True, include_detail=True)


def _written(engine: str, options: EngineOptions, source: DatasetSource) -> set[str]:
    disabled = disabled_columns(options, source=source)
    return {
        column.name
        for column in get_engine_registry().get(engine).spec.columns
        if column.name not in disabled
    }


def test_the_uniprot_switch_alone_writes_only_the_mapping() -> None:
    options = EngineOptions(values={"include_uniprot": True})

    assert _written("alphafold_database", options, "download") == {
        "uniprot_id",
        "sequence_cluster_30",
        "sequence_cluster_50",
        "sequence_cluster_70",
        "sequence_cluster_90",
        "sequence_cluster_95",
        "sequence_cluster_100",
    }


def test_a_column_is_disabled_for_a_source_its_engine_does_not_apply_to() -> None:
    options = EngineOptions(values={"include_pdb_redo": True})

    assert "pdb_redo_x" not in disabled_columns(options, source="download")
    assert "pdb_redo_x" in disabled_columns(options, source="upload")


def test_the_schema_fingerprint_follows_the_columns_a_run_writes() -> None:
    defaults = schema_fingerprint(EngineOptions(), source="download")
    with_sasa = schema_fingerprint(
        EngineOptions(values={"include_sasa": True}), source="download"
    )
    redo = EngineOptions(values={"include_pdb_redo": True})

    assert re.fullmatch(r"[0-9a-f]{16}", defaults)
    assert defaults == schema_fingerprint(EngineOptions(), source="download")
    assert defaults == schema_fingerprint(
        EngineOptions(technique="electron_microscopy"), source="download"
    )
    assert with_sasa != defaults
    assert schema_fingerprint(redo, source="upload") != schema_fingerprint(
        redo, source="download"
    )


def test_order_columns_refuses_a_repeated_name() -> None:
    with pytest.raises(ValueError, match="repeated name\\(s\\): x"):
        order_columns(["x", "pdb_id", "x"])


def test_order_columns_uses_display_order_then_the_given_order() -> None:
    columns = [
        "_atom_site.label_atom_id",
        "atom_sasa_protein",
        "x",
        "bnet",
        "citation_title",
        "eds_rscc",
        "pdb_id",
        "_atom_site.auth_comp_id",
    ]

    assert order_columns(columns) == [
        "pdb_id",
        "x",
        "eds_rscc",
        "citation_title",
        "bnet",
        "atom_sasa_protein",
        "_atom_site.label_atom_id",
        "_atom_site.auth_comp_id",
    ]


def test_a_profile_turns_on_only_the_toggles_the_technique_allows() -> None:
    options = get_engine_registry().with_defaults(
        EngineOptions(technique="electron_microscopy", values={"include_uniprot": True})
    )

    settings = apply_profile(options, PROFILE_STRUCTURE_QUALITY)

    # UniProt was already on, and RABDAM does not apply to electron microscopy.
    assert settings.toggles_turned_on == ("include_validation",)
    assert settings.toggles_ruled_out == ("include_rabdam",)
    assert settings.options.flag("include_validation") is True
    assert settings.options.flag("include_rabdam") is False
    assert settings.uses_profile_filter is True
    assert settings.column_filter_mode == "include"


def test_no_profile_leaves_the_options_and_the_requested_filter_alone() -> None:
    options = EngineOptions()

    settings = apply_profile(
        options, None, column_filter_mode="exclude", column_filter_list=["x"]
    )

    assert settings.options == options
    assert settings.profile is None
    assert settings.column_filter_mode == "exclude"
    assert settings.column_filter_list == ("x",)
    assert settings.uses_profile_filter is False


_CURATED_PROFILES = [profile for profile in PROFILES if profile.filter_mode]


def _unmatched_selectors(
    selectors: tuple[str, ...] | None, options: EngineOptions
) -> list[str]:
    # Settlement warns about each include selector that matches no column the
    # run writes. This returns the selectors it would warn about.
    disabled = disabled_columns(options, source="download")
    written = [
        spec.name for spec in declared_column_specs() if spec.name not in disabled
    ]
    return [
        selector
        for selector in selectors or ()
        if not any(matches_glob(name, selector) for name in written)
    ]


@pytest.mark.parametrize(
    "technique", ["electron_microscopy", "solution_nmr", "solution_scattering"]
)
@pytest.mark.parametrize("profile", _CURATED_PROFILES, ids=lambda p: p.id)
def test_a_profile_filter_leaves_out_the_columns_of_ruled_out_engines(
    technique: str, profile: ColumnProfile
) -> None:
    settings = resolve_run_settings({}, technique=technique, profile_id=profile.id)

    # The ruled-out engines are still reported rather than refused.
    assert "include_rabdam" in settings.toggles_ruled_out
    assert settings.options.flag("include_rabdam") is False
    assert settings.uses_profile_filter is True
    assert settings.column_filter_list == profile.columns_for(technique)
    assert _unmatched_selectors(settings.column_filter_list, settings.options) == []
    assert set(settings.column_filter_list or ()) < set(profile.columns)


def test_electron_microscopy_drops_the_rabdam_selectors_and_keeps_the_rest() -> None:
    profile = get_profile(PROFILE_RESEARCH_READY)

    xray = resolve_run_settings(
        {}, technique="x_ray_diffraction", profile_id=profile.id
    )
    em = resolve_run_settings(
        {}, technique="electron_microscopy", profile_id=profile.id
    )

    assert xray.column_filter_list == profile.columns
    assert em.column_filter_list is not None
    dropped = set(profile.columns) - set(em.column_filter_list)
    assert {"bdamage*", "packing_density*", "bnet*", "bnet_percentile*"} <= dropped
    for selector in dropped:
        owners = {
            spec.owner
            for spec in declared_column_specs()
            if matches_glob(spec.name, selector)
        }
        assert owners == {"rabdam"}, selector
    # Validation applies to electron microscopy, so its columns stay.
    assert "clashscore" in em.column_filter_list
    assert em.toggles_ruled_out == ("include_rabdam",)


def test_a_requested_filter_is_kept_whole_under_a_ruling_out_technique() -> None:
    settings = resolve_run_settings(
        {},
        technique="electron_microscopy",
        profile_id=PROFILE_RESEARCH_READY,
        column_filter_mode="include",
        column_filter_list=["pdb_id", "bnet*"],
    )

    assert settings.uses_profile_filter is False
    assert settings.column_filter_list == ("pdb_id", "bnet*")
