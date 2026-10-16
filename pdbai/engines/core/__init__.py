"""The engine system, which runs every engine and names none of them.

techniques.py holds the PDB's experimental techniques and X-ray radiation
sources, and matches what an entry reports against them.

options.py holds EngineOptions, the settings a run is configured with. It
also holds OptionSpec, which declares one setting an engine reads, and names
the two dataset shapes and the base tables each writes.

artifacts.py holds ArtifactKey and ArtifactStore, for the typed values that
engines hand to one another. It also names the artifacts the host publishes.

profiles.py holds the identifiers of the curated column profiles, which a
column names to say which profiles keep it.

contracts.py holds the Engine protocol and the records an engine declares
and returns, such as EngineSpec, ColumnSpec, TableSpec, and EngineResult.

tables.py checks the tables an engine returns. It also builds site_id, the
identifier engine tables share for an atom site.

registry.py holds EngineRegistry. It registers engines, checks their
declarations against one another, orders them for a run, and derives the
settings, tables, and report fields a run has.

coordinator.py runs one phase of engines on a structure. It checks and
merges each result. A failed engine is recorded rather than failing the
structure, unless the engine is required.

Nothing in this package imports a concrete engine, so a new engine needs no
change here. Everything the rest of PDB-AI shows about an engine comes from
its declarations. Every result it returns is checked against them, and a
result that breaks them counts as a failure of the engine.
"""

from .artifacts import (
    ENTRY_RECORD,
    STRUCTURE_CIF_PATH,
    STRUCTURE_METADATA,
    ArtifactKey,
    ArtifactStore,
)
from .contracts import (
    ALL_DATASET_SOURCES,
    COLUMN_DTYPES,
    HOST_REPORT_COLUMNS,
    ROW_COUNT_COLUMN_SUFFIX,
    TABLE_COLUMN_DTYPES,
    TABLE_LINK_LEVELS,
    ColumnDtype,
    ColumnGranularity,
    ColumnSpec,
    DatasetSource,
    Engine,
    EngineContext,
    EnginePhase,
    EngineResult,
    EngineServices,
    EngineSpec,
    ReportField,
    ReportKind,
    ReportValue,
    SourceFile,
    TableColumnSpec,
    TableLink,
    TableSpec,
    column_specs,
    is_engine_table_name,
)
from .coordinator import (
    EngineExecutionError,
    EnginePhaseResult,
    run_engine_phase,
)
from .options import (
    DATASET_SHAPES,
    DEFAULT_DATASET_SHAPE,
    DatasetShapeName,
    EngineOptions,
    OptionSpec,
    OptionValue,
    camel_case,
    parse_dataset_shape,
)
from .profiles import (
    CURATED_PROFILE_IDS,
    PROFILE_ML_MINIMAL,
    PROFILE_RADIATION_SENSITIVITY,
    PROFILE_RESEARCH_READY,
    PROFILE_STRUCTURE_QUALITY,
)
from .registry import EngineRegistry
from .tables import (
    SITE_ID_COLUMN,
    site_atom_link,
    site_key_columns,
    site_key_dataframe,
)
from .techniques import (
    ALL_XRAY_SOURCES,
    DEFAULT_TECHNIQUE,
    INTEGRATIVE_METHOD,
    TECHNIQUE_IDS,
    TECHNIQUES,
    XRAY_SOURCE_IDS,
    XRAY_SOURCE_TECHNIQUE,
    XRAY_SOURCES,
    RadiationSource,
    Technique,
    classify_methods,
    classify_sources,
    get_technique,
    get_xray_source,
    normalize_source,
)

__all__ = [
    "ALL_DATASET_SOURCES",
    "ALL_XRAY_SOURCES",
    "COLUMN_DTYPES",
    "CURATED_PROFILE_IDS",
    "DATASET_SHAPES",
    "DEFAULT_DATASET_SHAPE",
    "DEFAULT_TECHNIQUE",
    "ENTRY_RECORD",
    "HOST_REPORT_COLUMNS",
    "INTEGRATIVE_METHOD",
    "PROFILE_ML_MINIMAL",
    "PROFILE_RADIATION_SENSITIVITY",
    "PROFILE_RESEARCH_READY",
    "PROFILE_STRUCTURE_QUALITY",
    "ROW_COUNT_COLUMN_SUFFIX",
    "SITE_ID_COLUMN",
    "STRUCTURE_CIF_PATH",
    "STRUCTURE_METADATA",
    "TABLE_COLUMN_DTYPES",
    "TABLE_LINK_LEVELS",
    "TECHNIQUES",
    "TECHNIQUE_IDS",
    "XRAY_SOURCES",
    "XRAY_SOURCE_IDS",
    "XRAY_SOURCE_TECHNIQUE",
    "ArtifactKey",
    "ArtifactStore",
    "ColumnDtype",
    "ColumnGranularity",
    "ColumnSpec",
    "DatasetShapeName",
    "DatasetSource",
    "Engine",
    "EngineContext",
    "EngineExecutionError",
    "EngineOptions",
    "EnginePhase",
    "EnginePhaseResult",
    "EngineRegistry",
    "EngineResult",
    "EngineServices",
    "EngineSpec",
    "OptionSpec",
    "OptionValue",
    "RadiationSource",
    "ReportField",
    "ReportKind",
    "ReportValue",
    "SourceFile",
    "TableColumnSpec",
    "TableLink",
    "TableSpec",
    "Technique",
    "camel_case",
    "classify_methods",
    "classify_sources",
    "column_specs",
    "get_technique",
    "get_xray_source",
    "is_engine_table_name",
    "normalize_source",
    "parse_dataset_shape",
    "run_engine_phase",
    "site_atom_link",
    "site_key_columns",
    "site_key_dataframe",
]
