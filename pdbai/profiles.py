"""Column profiles, and the assembly of run settings that applies them.

A column profile is a named preset of dataset columns. A user chooses one
with --profile on the command line or profileId in the web app. A profile is
more than a column filter. It also switches on every engine whose columns it
keeps, since those columns cannot exist otherwise. Both front ends assemble
a run's settings through resolve_run_settings, so a profile is applied by
one rule. The profile named all filters nothing. Every other profile is an
include filter, and like any column filter it applies to the base tables
only.

A profile's columns are not listed here. Each column names the profiles it
belongs to in ColumnSpec.profiles, where the column is declared. Columns
name a profile by an identifier from engines.core.profiles. Engines import
those identifiers rather than this module, because this module imports the
engine registry and the schema. The profile itself is not recorded. The
manifest stores the engine options and the column filter it resolved to.
The settings hash therefore depends only on what the profile did, and a run
resumes without knowing which profile started it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .column_filtering import ColumnFilterMode, matches_glob
from .engines.catalog import get_engine_registry
from .engines.compatibility import compatible_toggles, require_compatible_options
from .engines.core import (
    ALL_XRAY_SOURCES,
    DEFAULT_DATASET_SHAPE,
    PROFILE_ML_MINIMAL,
    PROFILE_RADIATION_SENSITIVITY,
    PROFILE_RESEARCH_READY,
    PROFILE_STRUCTURE_QUALITY,
    DatasetShapeName,
    EngineOptions,
    OptionValue,
)
from .schema import DatasetColumnSpec, declared_column_specs


def _column_selector(column: str, globs: Iterable[str]) -> str:
    """Return the glob that stands for a column in a preset, or its name.

    Globs match as they do in a column filter. When several match, the one
    with the most literal characters wins. A broad glob such as bnet* may
    reach columns outside the profile, so it stands only for a column that
    no narrower glob covers.
    """
    matching = [glob for glob in globs if matches_glob(column, glob)]
    return max(matching, key=lambda glob: len(glob.replace("*", "")), default=column)


def _member_columns(profile_id: str) -> tuple[DatasetColumnSpec, ...]:
    """Return the declared columns that name a profile, in catalog order."""
    return tuple(
        spec for spec in declared_column_specs() if profile_id in spec.column.profiles
    )


def _selectors(profile_id: str, specs: Iterable[DatasetColumnSpec]) -> tuple[str, ...]:
    """Return the names and globs that stand for columns in a profile's filter.

    Each column is given by the narrowest glob declared for the profile, or
    else by its own name. Each selector is listed once.
    """
    globs = get_engine_registry().profile_patterns().get(profile_id, ())
    return tuple(dict.fromkeys(_column_selector(spec.name, globs) for spec in specs))


@dataclass(frozen=True)
class ColumnProfile:
    """A named preset of dataset columns, and the toggles that write them.

    The columns and toggles are read from the schema catalog each time they
    are asked for. They follow whichever columns name the profile there.
    """

    id: str
    name: str
    description: str
    # None for a profile that filters nothing and so keeps every column.
    filter_mode: ColumnFilterMode | None

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the names and globs the profile keeps, in catalog order.

        An engine may declare a glob for the profile. The glob is listed once,
        in place of every column it is the narrowest match for, so the preset
        covers whichever source-suffixed variants a run writes. The list
        holds the columns of every engine, whatever the technique.
        columns_for narrows it to one technique.
        """
        return _selectors(self.id, _member_columns(self.id))

    def columns_for(self, technique: str) -> tuple[str, ...]:
        """Return the columns the profile keeps in a run of one technique.

        A column is left out when the technique rules out its toggle or the
        engine that declares it. A run of that technique cannot write the
        column, and an include filter naming it would only report it missing.
        A glob is left out once every column it stood for is left out. The
        technique is assumed valid, since EngineOptions has checked it.
        """
        applicable = compatible_toggles(technique)
        return _selectors(
            self.id,
            (
                spec
                for spec in _member_columns(self.id)
                if (spec.toggle is None or spec.toggle in applicable)
                and (spec.engine is None or spec.engine.supports_technique(technique))
            ),
        )

    @property
    def toggles(self) -> tuple[str, ...]:
        """Return the toggles its columns are attributed to, in flag order."""
        attributed = {spec.toggle for spec in _member_columns(self.id)}
        return tuple(
            name for name in get_engine_registry().flag_names() if name in attributed
        )

    def to_dict(self) -> dict[str, object]:
        """Return the profile as JSON values, with its columns and toggles."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "filter_mode": self.filter_mode,
            "columns": list(self.columns),
            "toggles": list(self.toggles),
        }


PROFILES: tuple[ColumnProfile, ...] = (
    # The one profile that filters nothing. No column declares membership in
    # it, so engines.core does not define its identifier.
    ColumnProfile(
        id="all",
        name="All Columns",
        description=(
            "No filtering. The base tables keep every column the run writes, "
            "namely each _atom_site item, each item of a one-row mmCIF "
            "category repeated on every atom, the parser's derived and "
            "metadata columns, and the columns of every engine the run "
            "enables. It switches no engine on."
        ),
        filter_mode=None,
    ),
    ColumnProfile(
        id=PROFILE_RESEARCH_READY,
        name="Research-Ready",
        description=(
            "The default for analysis. Keeps atom identity, B factors and "
            "anisotropic displacement parameters, experimental metadata, "
            "RABDAM's BDamage, packing density, and Bnet scores, solvent "
            "accessibility, the UniProt mapping, "
            "AlphaFold pLDDT, and wwPDB validation metrics. Of the raw mmCIF "
            "fields, only "
            "the identity items and the B factor remain."
        ),
        filter_mode="include",
    ),
    ColumnProfile(
        id=PROFILE_RADIATION_SENSITIVITY,
        name="Radiation Sensitivity",
        description=(
            "Features for models of radiation-damage susceptibility. Keeps "
            "atom identity, B factors and anisotropic displacement parameters, "
            "RABDAM's BDamage, packing density, and Bnet scores, solvent "
            "accessibility, electron density at atoms, data-quality "
            "and wwPDB "
            "validation metrics, radiation dose "
            "and collection conditions, and the UniProt mapping."
        ),
        filter_mode="include",
    ),
    ColumnProfile(
        id=PROFILE_ML_MINIMAL,
        name="ML Minimal",
        description=(
            "The smallest complete feature set for training and inference. "
            "Keeps atom identity, B factors and their anisotropy summaries, "
            "resolution and R factors, the technique and radiation source, "
            "RABDAM's damage scores, relative solvent accessibility, and the "
            "real-space fit and Ramachandran class from wwPDB validation. Of "
            "the raw mmCIF fields, only the identity items and the B factor "
            "remain."
        ),
        filter_mode="include",
    ),
    ColumnProfile(
        id=PROFILE_STRUCTURE_QUALITY,
        name="Structure Quality",
        description=(
            "Experimental metadata and quality indicators for choosing "
            "structures before training. Keeps atom identity and B factors, "
            "resolution, R factors, data-collection statistics, radiation "
            "dose, crystallization conditions, the wwPDB clashscore and "
            "geometry outliers, RABDAM's structure-level Bnet and coverage, "
            "and the UniProt mapping."
        ),
        filter_mode="include",
    ),
)

_PROFILES_BY_ID = {profile.id: profile for profile in PROFILES}


def get_profile(profile_id: str) -> ColumnProfile:
    """Return the profile with the given identifier.

    Case and surrounding whitespace are ignored, and an identifier no profile
    has raises ValueError.
    """
    try:
        return _PROFILES_BY_ID[profile_id.strip().lower()]
    except KeyError:
        raise ValueError(
            f"profile must be one of {', '.join(_PROFILES_BY_ID)}; got {profile_id!r}"
        ) from None


@dataclass(frozen=True)
class RunSettings:
    """A run's engine options and column filter, with its profile applied."""

    options: EngineOptions
    column_filter_mode: ColumnFilterMode | None
    column_filter_list: tuple[str, ...] | None
    profile: ColumnProfile | None = None
    # toggles_turned_on holds the profile's toggles that were off until it
    # turned them on. toggles_ruled_out holds those the technique rules out.
    # They stay off, and the profile's filter leaves out their columns. The
    # command line reports both.
    toggles_turned_on: tuple[str, ...] = ()
    toggles_ruled_out: tuple[str, ...] = ()
    # True when the request named no column filter and the profile's applies.
    uses_profile_filter: bool = False


def apply_profile(
    options: EngineOptions,
    profile_id: str | None,
    *,
    column_filter_mode: ColumnFilterMode | None = None,
    column_filter_list: Iterable[str] | None = None,
) -> RunSettings:
    """Apply a column profile to a run's options and column filter.

    A profile turns on the toggles its columns are attributed to, even one
    the request switched off. A toggle the run's technique rules out stays
    off and is reported rather than refused. Every other toggle stays as
    given, so a profile never switches a toggle off. The profile's column
    filter is the one ColumnProfile.columns_for gives for the technique. It
    applies unless the request names a filter of its own, which is then kept
    whole. A ``profile_id`` of None applies no profile, and an unknown one
    raises ValueError.
    """
    filter_list = tuple(column_filter_list) if column_filter_list is not None else None
    if profile_id is None:
        return RunSettings(options, column_filter_mode, filter_list)

    profile = get_profile(profile_id)
    applicable = compatible_toggles(options.technique)
    toggles = profile.toggles
    turned_on = tuple(
        toggle
        for toggle in toggles
        if toggle in applicable and not options.flag(toggle)
    )
    ruled_out = tuple(toggle for toggle in toggles if toggle not in applicable)
    uses_profile_filter = (
        profile.filter_mode is not None
        and column_filter_mode is None
        and not filter_list
    )
    if uses_profile_filter:
        column_filter_mode = profile.filter_mode
        filter_list = profile.columns_for(options.technique)
    return RunSettings(
        options.with_values(dict.fromkeys(turned_on, True)),
        column_filter_mode,
        filter_list,
        profile=profile,
        toggles_turned_on=turned_on,
        toggles_ruled_out=ruled_out,
        uses_profile_filter=uses_profile_filter,
    )


def resolve_run_settings(
    values: Mapping[str, OptionValue],
    *,
    technique: str,
    xray_source: str = ALL_XRAY_SOURCES,
    dataset_shape: DatasetShapeName = DEFAULT_DATASET_SHAPE,
    profile_id: str | None = None,
    column_filter_mode: ColumnFilterMode | None = None,
    column_filter_list: Iterable[str] | None = None,
) -> RunSettings:
    """Build a run's engine options and column filter, and check them.

    ``values`` maps option names to the engine settings a user gave. A
    setting that is missing or None takes its declared default. The profile
    is then applied as apply_profile describes. Last, the result is checked
    against the technique.

    ValueError is raised for an unknown technique, X-ray source, dataset
    shape, or profile. It is also raised for an undeclared setting or an
    invalid value. It is raised too for an enabled engine the technique does
    not support.
    """
    options = get_engine_registry().with_defaults(
        EngineOptions(
            dataset_shape=dataset_shape,
            technique=technique,
            xray_source=xray_source,
            values={name: value for name, value in values.items() if value is not None},
        )
    )
    settings = apply_profile(
        options,
        profile_id,
        column_filter_mode=column_filter_mode,
        column_filter_list=column_filter_list,
    )
    require_compatible_options(settings.options)
    return settings


__all__ = [
    "PROFILES",
    "ColumnProfile",
    "RunSettings",
    "apply_profile",
    "get_profile",
    "resolve_run_settings",
]
