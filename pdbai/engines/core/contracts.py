"""The contract between the host and an engine, and what an engine declares.

An engine is an object with a spec and a run method. The spec is an
EngineSpec. The run method takes an EngineContext for one structure and
returns an EngineResult. The spec declares everything the rest of PDB-AI
shows about the engine. The command line, the web app, the schema catalog,
the run report, the run summary, and the manifest are all built from it, so
no code outside the engine's own package names the engine. Each declaration
is checked when it is made and raises ValueError when it is inconsistent. A
broken engine therefore fails when it is declared, not partway through a
run.

Each structure passes through two phases. Acquisition engines run first and
fetch what the structure needs, such as its file and its validation report.
The host then parses the structure file into an atom DataFrame. Annotation
engines run last and compute from those atoms. Only annotation engines may
declare columns or tables. The host replaces the atom DataFrame after
acquisition, so an acquisition engine's columns would never reach the
output. Its tables would not either, since the host writes only the
annotation phase's tables.

A dataset column is written when two things hold. Its engine must run, and
the column's own rule must allow it. The engine decides whether its columns
are written at all. The column's rule only picks which of a running engine's
columns are written, so it never repeats the engine's rule.

An engine and a column state their rule the same way. A predicate in
enabled_if decides when it is set. Otherwise the flag named by toggle
decides. That flag is also the one the engine or column is attributed to
in profiles, the data dictionary, and run reports. A rule that spans
several flags is therefore a predicate, and its toggle only says which flag
it belongs to.

Dataset columns and engine table columns use separate type vocabularies. A
dataset column sits beside the base columns in the schema catalog, so it is
text, numeric, or boolean as they are. An engine table is written on its
own, so each of its columns names the pandas dtype it is written with.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, get_args

import pandas as pd
import requests

from .artifacts import ArtifactKey, ArtifactStore
from .options import DATASET_SHAPES, EngineOptions, OptionSpec
from .profiles import CURATED_PROFILE_IDS
from .techniques import TECHNIQUE_IDS


class EnginePhase(str, Enum):
    """The two phases each structure passes through, in the order they run."""

    ACQUISITION = "acquisition"
    ANNOTATION = "annotation"


# Where a run's structures come from. A download is an archive entry, and an
# upload is a file a person supplied.
type DatasetSource = Literal["download", "upload"]
ALL_DATASET_SOURCES: frozenset[DatasetSource] = frozenset({"download", "upload"})


def _repeated(names: Iterable[str]) -> list[str]:
    """Return each name that occurs more than once in ``names``, sorted."""
    seen: set[str] = set()
    repeated: set[str] = set()
    for name in names:
        (repeated if name in seen else seen).add(name)
    return sorted(repeated)


def _check_sources(owner: str, sources: frozenset[str]) -> None:
    """Refuse dataset sources that are empty or that name an unknown source.

    ``owner`` names the declaration in the message, such as "Column 'x'". A
    declaration with no source would never be written or run, and an unknown
    source would never match a run, so either raises ValueError.
    """
    unknown = sorted(sources - ALL_DATASET_SOURCES)
    if unknown or not sources:
        raise ValueError(
            f"{owner} declares dataset sources {sorted(sources)}; use one or "
            f"more of {', '.join(sorted(ALL_DATASET_SOURCES))}"
        )


def _check_profiles(owner: str, profile_ids: Iterable[str]) -> None:
    """Refuse profile identifiers that name no curated profile.

    ``owner`` names the declaration in the message. A misspelled identifier
    would leave the column out of its profile without a word, so it raises
    ValueError.
    """
    unknown = sorted(set(profile_ids) - CURATED_PROFILE_IDS)
    if unknown:
        raise ValueError(
            f"{owner} names unknown profile(s) {unknown}; use one or more of "
            f"{', '.join(sorted(CURATED_PROFILE_IDS))}"
        )


class ColumnGranularity(str, Enum):
    """The level of the dataset a column describes.

    The level places the column in a star-schema table. In the flat shape
    every value is repeated on each atom row it applies to.
    """

    ATOM = "atom"
    RESIDUE = "residue"
    CHAIN = "chain"
    STRUCTURE = "structure"
    CITATION = "citation"


# What a dataset column holds. A "boolean" column holds True, False, or a
# missing value, which the schema checks once every producer has run.
type ColumnDtype = Literal["text", "numeric", "boolean"]

COLUMN_DTYPES: tuple[ColumnDtype, ...] = get_args(ColumnDtype.__value__)

# The characters that separate names in a requested column list. No column
# name may hold one, since a column filter could then never select it.
_COLUMN_NAME_SEPARATORS = re.compile(r"[\r\n,;]")


@dataclass(frozen=True)
class ColumnSpec:
    """One dataset column an engine writes.

    A name that is empty, has spaces around it, or holds a comma, a
    semicolon, or a line break raises ValueError. So do an unknown dtype, an
    unknown profile identifier, and sources that are empty or name an
    unknown dataset source.
    """

    name: str
    # The display group the column is listed under, such as "RABDAM Fields".
    group: str
    # The column's own flag. None attributes the column to its engine's
    # toggle, and without a predicate the column is then written whenever
    # its engine runs.
    toggle: str | None = None
    # The column's predicate. None leaves the decision to toggle. It is left
    # out of equality and repr, since a function is behavior and not part of
    # the schema.
    enabled_if: Callable[[EngineOptions], bool] | None = field(
        default=None, repr=False, compare=False
    )
    granularity: ColumnGranularity = ColumnGranularity.ATOM
    dtype: ColumnDtype = "text"
    description: str = ""
    # The curated profiles that keep the column, by identifier.
    profiles: frozenset[str] = frozenset()
    # The dataset sources the column exists for. An upload has no RCSB and
    # PDB-Redo versions to compare, for one, so a column whose name says which
    # version it scores exists only for downloads.
    sources: frozenset[DatasetSource] = ALL_DATASET_SOURCES

    def __post_init__(self) -> None:
        """Check the name, the dtype, the profiles, and the sources."""
        if (
            not self.name
            or self.name != self.name.strip()
            or _COLUMN_NAME_SEPARATORS.search(self.name)
        ):
            raise ValueError(
                f"Column name {self.name!r} must not be empty, have spaces around "
                "it, or hold a comma, a semicolon, or a line break"
            )
        if self.dtype not in COLUMN_DTYPES:
            raise ValueError(
                f"Column {self.name!r} declares dtype {self.dtype!r}; "
                f"use one of {', '.join(COLUMN_DTYPES)}"
            )
        _check_profiles(f"Column {self.name!r}", self.profiles)
        _check_sources(f"Column {self.name!r}", self.sources)

    def is_enabled(self, options: EngineOptions, *, source: DatasetSource) -> bool:
        """Return whether a run writes the column, given that its engine runs.

        A column whose sources leave out ``source`` is never written. Then the
        predicate decides, or else the toggle. A column with neither is
        written whenever its engine runs.
        """
        if source not in self.sources:
            return False
        if self.enabled_if is not None:
            return self.enabled_if(options)
        if self.toggle is not None:
            return options.flag(self.toggle)
        return True


def column_specs(
    names: Sequence[str],
    *,
    group: str,
    toggle: str | None = None,
    enabled_if: Callable[[EngineOptions], bool] | None = None,
    enabled_if_by_name: Mapping[str, Callable[[EngineOptions], bool]] | None = None,
    granularity: ColumnGranularity = ColumnGranularity.ATOM,
    granularity_by_name: Mapping[str, ColumnGranularity] | None = None,
    dtype_by_name: Mapping[str, ColumnDtype] | None = None,
    descriptions: Mapping[str, str] | None = None,
    profiles: frozenset[str] = frozenset(),
    profiles_by_name: Mapping[str, frozenset[str]] | None = None,
    sources: frozenset[DatasetSource] = ALL_DATASET_SOURCES,
    sources_by_name: Mapping[str, frozenset[DatasetSource]] | None = None,
) -> tuple[ColumnSpec, ...]:
    """Return one ColumnSpec for each of ``names``, sharing the settings given.

    Each mapping whose name ends in _by_name overrides the shared setting
    for the columns it names. A column missing from ``dtype_by_name`` is
    text. A column missing from ``descriptions`` gets a placeholder that
    names its group and itself.

    A mapping that names a column missing from ``names`` raises ValueError,
    since a misspelled name would otherwise leave its column with the shared
    setting. Each ColumnSpec raises ValueError as its own declaration does.
    """
    column_sources = sources_by_name or {}
    granularities = granularity_by_name or {}
    dtypes = dtype_by_name or {}
    column_descriptions = descriptions or {}
    column_profiles = profiles_by_name or {}
    column_predicates = enabled_if_by_name or {}
    declared = set(names)
    for label, mapping in (
        ("enabled_if_by_name", column_predicates),
        ("granularity_by_name", granularities),
        ("dtype_by_name", dtypes),
        ("descriptions", column_descriptions),
        ("profiles_by_name", column_profiles),
        ("sources_by_name", column_sources),
    ):
        unknown = sorted(set(mapping) - declared)
        if unknown:
            raise ValueError(f"{label} names columns that are not declared: {unknown}")
    return tuple(
        ColumnSpec(
            name=name,
            group=group,
            toggle=toggle,
            enabled_if=column_predicates.get(name, enabled_if),
            granularity=granularities.get(name, granularity),
            dtype=dtypes.get(name, "text"),
            description=column_descriptions.get(
                name, f"{group.removesuffix(' Fields')} engine output: {name}."
            ),
            profiles=column_profiles.get(name, profiles),
            sources=column_sources.get(name, sources),
        )
        for name in names
    )


# The base tables of every output shape. Engine tables share the partials
# directory and the output file names with them, so no engine table may take
# one of these names.
BASE_TABLE_NAMES: frozenset[str] = frozenset(
    table for shape in DATASET_SHAPES.values() for table in shape.tables
)

# Column types an engine table may declare. Each is a pandas dtype that keeps
# missing values distinct from false or zero and maps directly to Parquet.
TABLE_COLUMN_DTYPES: tuple[str, ...] = ("string", "boolean", "Int64", "float64")

# Dataset levels a table row can link back to, besides another table of the
# same engine.
TABLE_LINK_LEVELS: tuple[str, ...] = tuple(
    level.value
    for level in ColumnGranularity
    if level is not ColumnGranularity.CITATION
)

_TABLE_NAME = re.compile(r"[a-z][a-z0-9_]*")


def is_engine_table_name(name: str) -> bool:
    """Return whether an engine table may be called ``name``.

    The name becomes part of file names, so it must be lowercase snake_case.
    It must also differ from the name of every base table.
    """
    return _TABLE_NAME.fullmatch(name) is not None and name not in BASE_TABLE_NAMES


@dataclass(frozen=True)
class TableColumnSpec:
    """One column of an engine table.

    An unknown dtype raises ValueError. So do categories on a column that is
    not a string column, and a category listed twice.
    """

    name: str
    dtype: str = "string"
    description: str = ""
    # The fixed values a category column may hold, so one-hot encodings stay
    # stable across runs. None means the column is not categorical.
    categories: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Check the dtype and the categories."""
        if self.dtype not in TABLE_COLUMN_DTYPES:
            raise ValueError(
                f"Table column {self.name!r} declares dtype {self.dtype!r}; "
                f"use one of {', '.join(TABLE_COLUMN_DTYPES)}"
            )
        if self.categories is not None:
            if self.dtype != "string":
                raise ValueError(
                    f"Table column {self.name!r} declares categories, which "
                    "apply only to string columns"
                )
            repeated = _repeated(self.categories)
            if repeated:
                raise ValueError(
                    f"Table column {self.name!r} declares duplicate categories: "
                    f"{repeated}"
                )


@dataclass(frozen=True)
class TableLink:
    """The columns that join a table row to a dataset level or a table."""

    # A dataset level from TABLE_LINK_LEVELS, or another table of the same
    # engine.
    target: str
    # The columns of this table that hold the link.
    columns: tuple[str, ...]
    # The matching columns in the target, pair by pair. None means the same
    # names. A bond table, for one, joins its ligand atom through
    # ligand_chain_id to chain_id.
    target_columns: tuple[str, ...] | None = None
    # Tells apart several links to one target, such as the two atoms of a
    # bond.
    role: str = ""

    @property
    def resolved_target_columns(self) -> tuple[str, ...]:
        """Return the target columns, defaulting to the link's own names."""
        return self.target_columns if self.target_columns is not None else self.columns


@dataclass(frozen=True)
class TableSpec:
    """An engine's own output table, with one row per multi-atom record.

    A record is a site, a bond, or anything else that spans several atoms.
    The declaration raises ValueError for any of these problems. The name is
    not lowercase snake_case, or a base table has it. A column is declared
    twice. The key is empty, lists a column twice, or names a column that is
    not declared. A link has no columns, uses a column that is not declared,
    or has target columns that do not pair with its own. A link to another
    table is checked against that table by the engine's EngineSpec.
    """

    name: str
    description: str
    # Columns that together are unique within the table and never missing.
    key: tuple[str, ...]
    columns: tuple[TableColumnSpec, ...]
    links: tuple[TableLink, ...] = ()

    def __post_init__(self) -> None:
        """Check the name, the columns, the key, and the links."""
        if not _TABLE_NAME.fullmatch(self.name):
            raise ValueError(
                f"Table name {self.name!r} must be lowercase snake_case, since it "
                "becomes part of output file names"
            )
        if self.name in BASE_TABLE_NAMES:
            raise ValueError(
                f"Table name {self.name!r} is reserved for a base dataset table"
            )
        names = self.column_names
        repeated = _repeated(names)
        if repeated:
            raise ValueError(f"Table {self.name!r} declares a column twice: {repeated}")
        if not self.key:
            raise ValueError(f"Table {self.name!r} must declare key columns")
        repeated = _repeated(self.key)
        if repeated:
            raise ValueError(
                f"Table {self.name!r} lists a key column twice: {repeated}"
            )
        undeclared_key = [column for column in self.key if column not in names]
        if undeclared_key:
            raise ValueError(
                f"Table {self.name!r} key columns are not declared: {undeclared_key}"
            )
        for link in self.links:
            if not link.columns:
                raise ValueError(f"Table {self.name!r} declares a link with no columns")
            undeclared = [column for column in link.columns if column not in names]
            if undeclared:
                raise ValueError(
                    f"Table {self.name!r} link to {link.target!r} uses undeclared "
                    f"columns: {undeclared}"
                )
            if len(link.resolved_target_columns) != len(link.columns):
                raise ValueError(
                    f"Table {self.name!r} link to {link.target!r} pairs "
                    f"{len(link.columns)} column(s) with "
                    f"{len(link.resolved_target_columns)} target column(s)"
                )

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the declared column names in output order."""
        return tuple(column.name for column in self.columns)

    @property
    def link_columns(self) -> tuple[str, ...]:
        """Return every column a link uses, once each, in declaration order."""
        seen: dict[str, None] = {}
        for link in self.links:
            for column in link.columns:
                seen.setdefault(column)
        return tuple(seen)

    def empty_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with no rows and every declared column."""
        return pd.DataFrame(
            {column.name: pd.Series([], dtype=column.dtype) for column in self.columns}
        )

    def to_record(self) -> dict[str, object]:
        """Return the declaration as the JSON object a manifest holds."""
        return {
            "name": self.name,
            "description": self.description,
            "key": list(self.key),
            "links": [
                {
                    "target": link.target,
                    "columns": list(link.columns),
                    "target_columns": list(link.resolved_target_columns),
                    "role": link.role,
                }
                for link in self.links
            ],
            "columns": [
                {
                    "name": column.name,
                    "dtype": column.dtype,
                    "description": column.description,
                    "categories": (
                        None if column.categories is None else list(column.categories)
                    ),
                }
                for column in self.columns
            ],
        }


@dataclass(frozen=True)
class EngineSpec:
    """Everything an engine declares about itself.

    The declaration raises ValueError for any of these problems. A technique
    is unknown. The sources are empty or name an unknown dataset source. A
    profile pattern names an unknown profile. A table, column, report field,
    or option is declared twice. An acquisition engine declares tables or
    columns. A table link targets neither a dataset level nor a table of
    this engine, or names target columns that table does not declare.
    """

    # The identifier that dependencies and context.failures name the engine
    # by.
    name: str
    # The name shown to people, as in a warning that the engine failed.
    display_name: str
    phase: EnginePhase
    # The engine's own flag. None leaves the decision to enabled_if or, when
    # that is None too, to default_enabled.
    toggle: str | None = None
    # The engine's predicate. None leaves the decision to toggle.
    enabled_if: Callable[[EngineOptions], bool] | None = field(
        default=None, repr=False, compare=False
    )
    # Engines that must run before this one. Each runs whenever this one does,
    # even if the run does not enable it. A dependency that does not apply to
    # the run's dataset source is dropped.
    dependencies: tuple[str, ...] = ()
    columns: tuple[ColumnSpec, ...] = ()
    # Whether the engine runs when it has neither a toggle nor a predicate.
    default_enabled: bool = False
    # Technique identifiers this engine's science applies to. None means every
    # technique. A run that enables the engine for another technique is
    # refused before it starts, rather than producing meaningless numbers.
    techniques: frozenset[str] | None = None
    # Every engine that conditional_dependencies may name. Listing them lets
    # the whole registry be ordered without settings.
    conditional_dependency_names: tuple[str, ...] = ()
    # The dataset sources this engine applies to. PDB-Redo, for instance, has
    # nothing to offer an uploaded file with no archive entry behind it.
    sources: frozenset[DatasetSource] = ALL_DATASET_SOURCES
    # The engine's own output tables. Each is written for every structure the
    # engine runs on, with no rows when the engine fails.
    tables: tuple[TableSpec, ...] = ()
    # Whether the engine's failure fails the entry. Without it, a failure is
    # recorded and the entry still completes.
    required: bool = False
    # The typed values this engine publishes. An engine that reads one of
    # them depends on this engine or runs in a later phase, so that it runs
    # after it.
    provides: tuple[ArtifactKey[Any], ...] = ()
    # Engines this one needs only under some settings. RABDAM, for one, needs
    # PDB-Redo only when it scores PDB-Redo's model.
    conditional_dependencies: Callable[[EngineOptions], tuple[str, ...]] | None = field(
        default=None, repr=False, compare=False
    )
    # The settings this engine reads, declared once for the command line, the
    # web form, persisted settings, and the run summary.
    options: tuple[OptionSpec, ...] = ()
    # Per-structure outcome fields for the run report and manifest.
    report_fields: tuple[ReportField, ...] = ()
    # Returns receipts, keyed by name, for data files the engine ships with
    # and a run's results depend on, such as a bundled reference distribution.
    # An error it raises is recorded in place of its receipts.
    reference_data: Callable[[], Mapping[str, object]] | None = field(
        default=None, repr=False, compare=False
    )
    # Checks the environment before a run that includes this engine starts.
    # It raises when the engine cannot run at all, as with a missing external
    # package, so the run stops before every structure fails one by one.
    preflight: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )
    # Pairs of a profile identifier and the glob selectors that stand for this
    # engine's columns in that profile. A glob such as bnet* covers every
    # variant of a column named for its scoring source.
    profile_patterns: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        """Check the techniques, the sources, the profiles, and the outputs."""
        if self.techniques is not None:
            unknown = sorted(self.techniques - set(TECHNIQUE_IDS))
            if unknown:
                raise ValueError(
                    f"Engine {self.name!r} declares unknown technique(s): "
                    + ", ".join(unknown)
                )
        _check_sources(f"Engine {self.name!r}", self.sources)
        _check_profiles(
            f"Engine {self.name!r}",
            (profile_id for profile_id, _ in self.profile_patterns),
        )
        for label, names in (
            ("a table", [item.name for item in self.tables]),
            ("a column", [item.name for item in self.columns]),
            ("a report field", [item.name for item in self.report_fields]),
            ("an option", [item.name for item in self.options]),
        ):
            repeated = _repeated(names)
            if repeated:
                raise ValueError(
                    f"Engine {self.name!r} declares {label} twice: {repeated}"
                )
        if self.tables and self.phase is not EnginePhase.ANNOTATION:
            raise ValueError(
                f"Engine {self.name!r} declares tables but runs in the "
                f"{self.phase.value} phase; only annotation engines write tables"
            )
        if self.columns and self.phase is not EnginePhase.ANNOTATION:
            raise ValueError(
                f"Engine {self.name!r} declares columns but runs in the "
                f"{self.phase.value} phase; only annotation engines add columns"
            )
        tables_by_name = {table.name: table for table in self.tables}
        for table in self.tables:
            for link in table.links:
                if link.target in TABLE_LINK_LEVELS:
                    continue
                target = tables_by_name.get(link.target)
                if target is None:
                    raise ValueError(
                        f"Table {table.name!r} links to {link.target!r}, which is "
                        "neither a dataset level ("
                        + ", ".join(TABLE_LINK_LEVELS)
                        + f") nor a table of engine {self.name!r}"
                    )
                missing = [
                    column
                    for column in link.resolved_target_columns
                    if column not in target.column_names
                ]
                if missing:
                    raise ValueError(
                        f"Table {table.name!r} links to {target.name!r} columns "
                        f"that table does not declare: {missing}"
                    )

    def is_enabled(self, options: EngineOptions) -> bool:
        """Return whether the run switches the engine on itself.

        An engine that runs only because another depends on it is not counted
        here.
        """
        if self.enabled_if is not None:
            return self.enabled_if(options)
        if self.toggle is not None:
            return options.flag(self.toggle)
        return self.default_enabled

    def dependencies_under(self, options: EngineOptions | None) -> tuple[str, ...]:
        """Return the engines this one needs under ``options``.

        When ``options`` is None, every engine it could need is returned, which
        is what ordering the full registry requires. A conditional dependency
        missing from conditional_dependency_names raises ValueError.
        """
        if self.conditional_dependencies is None:
            return self.dependencies
        if options is None:
            return self.dependencies + self.conditional_dependency_names
        conditional = self.conditional_dependencies(options)
        undeclared = [
            name
            for name in conditional
            if name not in self.conditional_dependency_names
        ]
        if undeclared:
            raise ValueError(
                f"Engine {self.name!r} depends on {undeclared} without declaring "
                "them in conditional_dependency_names"
            )
        return self.dependencies + conditional

    @property
    def report_field_names(self) -> tuple[str, ...]:
        """Return the declared report field names in declaration order."""
        return tuple(report_field.name for report_field in self.report_fields)

    def supports_technique(self, technique: str) -> bool:
        """Return whether this engine applies to ``technique``."""
        return self.techniques is None or technique in self.techniques


@dataclass(frozen=True)
class EngineServices:
    """The network access and download cache the host lends to engines."""

    session: requests.Session
    # Seconds to wait for each request.
    timeout: float
    # Every download is cached beneath this directory and reused by later
    # runs until it expires. The host keeps each entry's RCSB record at the
    # top level, as {pdb_id.lower()}_metadata.json. Each engine names its own
    # files, in a directory of its own or at the top level.
    cache_root: Path


@dataclass
class EngineContext:
    """The state that one structure's engines share.

    One context lives through both phases, so annotation engines read what
    acquisition engines publish. The atom DataFrame is empty during
    acquisition. Each annotation engine's columns are merged into it as that
    engine finishes, so a later engine sees them.
    """

    # The entry's identifier. An upload's is taken from its file name, so it
    # need not be a PDB ID.
    pdb_id: str
    atom_dataframe: pd.DataFrame
    options: EngineOptions
    source: DatasetSource = "download"
    # None for an upload, which needs no network.
    services: EngineServices | None = None
    # Which coordinates the atom DataFrame holds. A download's are the
    # asymmetric unit, "asymmetric_unit", and an upload's are the file's as
    # given, "as_provided".
    structure_context: str = "as_provided"
    artifacts: ArtifactStore = field(default_factory=ArtifactStore)
    # The failure message of each engine that raised, by engine name, so a
    # later engine can say why an input it expected is missing.
    failures: dict[str, str] = field(default_factory=dict[str, str])

    def require_services(self) -> EngineServices:
        """Return the host's services, or raise RuntimeError without them."""
        if self.services is None:
            raise RuntimeError(
                "This engine needs network services the host did not provide"
            )
        return self.services


# One value of a report field. None means the structure has no value.
type ReportValue = bool | int | float | str | None
# What a report field's values are, which decides how the run summary shows
# them. A "fraction" is a share from 0 to 1 and is shown as a percentage. A
# "number" is shown as it is.
type ReportKind = Literal["text", "flag", "number", "fraction"]

_REPORT_KINDS: tuple[ReportKind, ...] = get_args(ReportKind.__value__)

# The run report's own columns, in order. They record what the host did with
# each entry. Engine report fields follow them in the same rows, so no field
# may take one of these names.
HOST_REPORT_COLUMNS: tuple[str, ...] = (
    "pdb_id",
    "source",
    "state",
    "success",
    "technique_status",
    "technique_message",
    "experimental_methods",
    "radiation_sources",
    "download_success",
    "download_error",
    "parse_success",
    "parse_error",
    "output_error",
    "metadata_success",
    "metadata_message",
    "metadata_level",
)

# The run report gives each engine table a column that holds its row count.
# The column is named after the table with this suffix, so no report field
# may end with it.
ROW_COUNT_COLUMN_SUFFIX = "_row_count"


@dataclass(frozen=True)
class ReportField:
    """One per-structure outcome field an engine reports.

    Values reach the run report, the manifest, and the web results under this
    name, so the host reports an engine's outcome without knowing the engine.
    A name the run report keeps for its own columns raises ValueError. So
    do a name that ends in ROW_COUNT_COLUMN_SUFFIX and an unknown kind.
    """

    name: str
    description: str
    kind: ReportKind = "text"
    # The value recorded when the engine raises.
    on_failure: ReportValue = None
    # Whether the field records the failure's message instead of on_failure.
    carries_message: bool = False
    # Whether the run summary's per-structure table shows this field, and the
    # heading it has there. The name serves when there is no label.
    in_summary: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        """Check the name and the kind."""
        if self.name in HOST_REPORT_COLUMNS:
            raise ValueError(
                f"Report field name {self.name!r} is reserved for a run report "
                "column the host writes"
            )
        if self.name.endswith(ROW_COUNT_COLUMN_SUFFIX):
            raise ValueError(
                f"Report field name {self.name!r} ends with "
                f"{ROW_COUNT_COLUMN_SUFFIX!r}, which the run report reserves for "
                "engine table row counts"
            )
        if self.kind not in _REPORT_KINDS:
            raise ValueError(
                f"Report field {self.name!r} declares kind {self.kind!r}; "
                f"use one of {', '.join(_REPORT_KINDS)}"
            )


@dataclass(frozen=True)
class SourceFile:
    """One file an acquisition engine used, for the manifest's receipts."""

    # What the file is to the run, such as "rcsb_mmcif".
    role: str
    path: Path
    # The URL the file was fetched from, or several joined by "; " when it
    # was assembled from more than one. None when no URL is known.
    source_url: str | None
    # Whether the cache held the file before this structure was processed.
    # None when that is not known.
    cached_before: bool | None = None


@dataclass(frozen=True)
class EngineResult:
    """What an engine produced for one structure."""

    # New atom columns, on the atom DataFrame's own index. A column the atoms
    # already have is replaced. None adds no columns.
    columns: pd.DataFrame | None = None
    # Rows for the engine's declared tables, keyed by table name. A declared
    # table left out is written with its header only.
    tables: Mapping[str, pd.DataFrame] = field(default_factory=dict[str, pd.DataFrame])
    # Messages added to the run's warnings under the structure's ID.
    warnings: tuple[str, ...] = ()
    # Values for the engine's declared report fields. A field left out records
    # None.
    report: Mapping[str, ReportValue] = field(default_factory=dict[str, ReportValue])
    # Values to publish under keys the engine declares in provides.
    artifacts: Mapping[ArtifactKey[Any], object] = field(
        default_factory=dict[ArtifactKey[Any], object]
    )
    # Files the engine read, which the manifest records as receipts. Only an
    # acquisition engine's files are recorded.
    source_files: tuple[SourceFile, ...] = ()


class Engine(Protocol):
    """The interface every engine implements."""

    spec: EngineSpec

    def run(self, context: EngineContext) -> EngineResult:
        """Return what the engine produces for the structure in ``context``.

        Raise when the engine cannot produce its output. The engine reads the
        atom DataFrame and the artifacts from ``context``, and it should change
        them only through the result it returns.
        """
        ...


__all__ = [
    "ALL_DATASET_SOURCES",
    "BASE_TABLE_NAMES",
    "COLUMN_DTYPES",
    "HOST_REPORT_COLUMNS",
    "ROW_COUNT_COLUMN_SUFFIX",
    "TABLE_COLUMN_DTYPES",
    "TABLE_LINK_LEVELS",
    "ColumnDtype",
    "ColumnGranularity",
    "ColumnSpec",
    "DatasetSource",
    "Engine",
    "EngineContext",
    "EnginePhase",
    "EngineResult",
    "EngineServices",
    "EngineSpec",
    "ReportField",
    "ReportKind",
    "ReportValue",
    "SourceFile",
    "TableColumnSpec",
    "TableLink",
    "TableSpec",
    "column_specs",
    "is_engine_table_name",
]
