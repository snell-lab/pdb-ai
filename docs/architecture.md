# Architecture

PDB-AI separates acquisition, analysis, schema declarations, orchestration,
persistence, and reporting.

[Project README](../README.md)

## Overview

- **RCSB's standard mmCIF is the only mandatory download**, parsed via `gemmi`
  (no Biopython dependency). Every other data source or computation is an
  "engine" you turn on or off.
- **Every integration has its own package in `pdbai/engines/`.** Calculation,
  download, or comparison code stays separate from the package's registered
  `engine.py` entry point. Each engine remains independently readable and is
  coordinated through the central catalog.

## Engines

- **Every engine implements the same registered contract, and declares
  everything the rest of the application shows about it.** An `EngineSpec`
  declares its phase, toggle, dependencies, output `ColumnSpec` and `TableSpec`
  records, the settings it reads (`OptionSpec`), the per-structure outcome
  fields it reports (`ReportField`), the typed artifacts it provides, the
  dataset sources it applies to, and optional preflight and reference-data
  hooks. `run(EngineContext) -> EngineResult` is the single
  integration point. The command line, the web form and its JSON API, persisted
  settings, technique compatibility, the run report, the run summary, and the
  manifest are all generated from those declarations, so no code outside
  `pdbai/engines/` dispatches on an engine's name or settings. The one source
  the host fetches itself is each downloaded entry's RCSB entry record, which
  technique intake reads before any engine runs.
- **Engines hand each other typed artifacts, not strings.** For a downloaded
  entry, one `EngineContext` lives through both phases, so what an acquisition
  engine publishes (a downloaded file, a parsed report) is what the annotation
  engines read. An upload starts at the annotation phase, with the host
  publishing the uploaded file. Keys are `ArtifactKey[T]` values that carry
  their type. For a download, the RCSB engine publishes the structure file, and
  the host publishes the entry's metadata and, when it could be fetched, the
  archive's entry record. An engine publishes only the keys it declares in
  `provides`, each as a value of the key's type. Nothing declares what an
  engine reads. An annotation engine reads what acquisition published, and
  otherwise a reader names its producer in `dependencies` (or
  `conditional_dependencies`), which is what runs the producer first. A
  producer may fail or have nothing to publish, so a missing artifact is an
  expected case for the reader.
- **Only required engines are fatal.** The mandatory RCSB download is
  `required`. Any other engine that raises is recorded, and the structure still
  completes. Its enabled columns are left empty, its tables header-only, and its
  report fields set to their declared failure values, and a warning names it.
  Engines that depend on a failed one read `context.failures` to say why their
  input is missing.
- **Dependencies can depend on settings.** RABDAM pulls in PDB-Redo only when it
  scores PDB-Redo's model (`conditional_dependencies`), so PDB-Redo's engines do
  not depend on RABDAM. The density engine needs the engines of each map source
  it reads (`--density-source`). For PDB-Redo's maps those are the map source
  engine and the comparison engine, whose coordinates of PDB-Redo's model it
  reads from the atom table. They then run with `--include-pdb-redo` off, and
  their own columns are left out of the output. For the wwPDB validation maps
  it is the validation map source engine alone.
- **RABDAM's scoring source is a 3-way choice.** `--rabdam-source
  {pdb_redo,rcsb,both}` picks which mmCIF `rabdam-custom` scores for a
  downloaded entry, and every column name says which one it was. An uploaded
  file is scored as given, under unsuffixed names. See [RABDAM column
  names](scientific-methods.md#rabdam-column-names).

## Technique selection

**One run processes one experimental technique.** `EngineOptions.technique` is
one value from the PDB's own `_exptl.method` vocabulary (X-ray diffraction by
default, see `pdbai/engines/core/techniques.py`). At intake the RCSB entry
record is fetched first and its reported methods compared against the technique.
An entry refused by its record never reaches the structure download or any
engine, and is persisted as `state = refused` with its reason. An upload, or a
download whose record could not be fetched or names no method, is checked
against the file's own `_exptl.method` after parsing. An entry with no method
anywhere is admitted as unverified rather than refused. Every `EngineSpec`
declares the techniques its science applies to (`techniques=None` means all),
and `pdbai/engines/compatibility.py` rejects, before any work starts, a run that
would execute an engine for a technique it does not support. That covers an
engine switched on by its flag, and an unsupported dependency of a supported
engine. Within X-ray diffraction,
`EngineOptions.xray_source` is a second-level intake filter on the reported
radiation source (how XFEL entries are distinguished, since the PDB files them
as X-ray diffraction). It normalizes deposited spellings and refuses entries
that record no source. Temperature and the like remain ordinary columns to
filter on. They are not techniques.

## Columns and tables

- **Every curated output column has one owner and one declaration.** The
  application schema combines parser-owned base fields and engine `ColumnSpec`
  records into one catalog carrying order, group, type, description,
  granularity, owning toggle, and profile membership. Column ordering, profiles,
  star-schema placement, data-dictionary text, and disabled-column selection are
  all derived from that catalog. Unknown raw mmCIF fields remain supported
  through the documented dynamic fallback.
- **Only per-atom mmCIF data becomes per-atom columns.** Every `_atom_site.*`
  field is carried through, and a non-atom category is broadcast onto atom rows
  only when it holds a single structure-level value (`_exptl.method`). Multi-row
  categories, the per-residue, per-entity, or per-chain tables such as
  `_pdbx_poly_seq_scheme`, are not atom-level facts and are not emitted.
  Flattening them into one semicolon-joined string repeated on every row would
  turn a 1.4 MB input into a ~700 MB CSV. What a study needs from such a
  category becomes a curated column with a stated rule, as `refinement_programs`
  is read from `_software` and `tls_group_count` from `_pdbx_refine_tls`.
- **Relationships between atoms go in engine tables.** An annotation engine's
  `EngineSpec` can declare `TableSpec` records for data with many atoms per row,
  such as a pair of bonded atoms or a metal and its donor atoms. None of the
  current built-in engines declares one. `EngineResult.tables` carries the rows. Before any
  atom column merges, the coordinator rejects undeclared tables or columns and
  missing key or link columns, then casts every column to its declared type,
  then rejects missing or duplicate keys and values outside a column's
  categories. Each structure writes `partials/<table>/<entry>.csv`, header-only
  when it has no rows, and a structure is resumed as complete only when every
  enabled table's partial exists. Assembly publishes each table as its own file
  in both output shapes. See [Engine tables](dataset-schema.md#engine-tables).
- **Disabling an engine drops its columns entirely** from the output. Toggles
  are inclusion/exclusion, not "compute it anyway and blank it out."
- **PDB-Redo's actual re-refined coordinates are exposed directly**, not just as
  deltas. `pdb_redo_x`/`pdb_redo_y`/`pdb_redo_z` appear when
  `--include-pdb-redo` enables the comparison engine. RABDAM may acquire the
  same source file internally without exposing comparison columns.
- **Non-scientific publication metadata is split out.** In `star_schema` mode,
  citation/author/DOI fields land in their own `citations` table rather than
  mixed into the `structures` table. See [Dataset
  shape](dataset-schema.md#dataset-shape-flat-vs-star_schema).
- **Output columns are grouped, not scattered.** The final CSV follows the
  schema catalog's display order (`schema/catalog.py`) instead of whatever order
  parsing produced. The catalog starts from declaration order, with the base
  columns as `schema/base.py` declares them (identity, coordinates, B-factors,
  crystallographic quality, experimental conditions, citation) followed by each
  engine's columns in the order the engines are registered in
  `engines/catalog.py`. The columns are then gathered by display group, and each
  group sits where its first column falls. So an engine column declared in a
  base group sits among the base columns. AlphaFold and validation declare some
  of theirs in "PDB Fields", which therefore come before the citation columns.
  `schema/base.py` also declares several raw `_atom_site` items, such as
  `_atom_site.id` and `_atom_site.B_iso_or_equiv`, which take their catalog
  positions. Columns the catalog does not declare, such as most raw mmCIF
  fields, come last.

## Runs and outputs

- **Every run gets a reproducibility receipt and data dictionary.**
  `manifest.json` records each execution environment plus source URLs and
  SHA-256 fingerprints for inputs, outputs, and any reference data the run's
  engines ship with (RABDAM's Bnet reference database, for one). An input's
  fingerprint is that of the bytes the run used. Runs that share a working
  directory share its download cache, and a later run may refresh a cached file,
  so the file on disk can differ from an earlier run's receipt while that
  receipt stays true of the run. `run_summary.md` renders that receipt in
  readable tables, describes what every toggle did, and includes per-column
  statistics (type, % null, minimum and maximum, mean and standard deviation,
  distinct values) for every output table. The statistics are computed in
  chunked reads, so a full multi-structure dataset is never loaded into memory.
- **Run records scale with the cohort.** Per-structure entry updates are
  appended to `entries.jsonl` and replayed on load. `manifest.json` holds
  everything else and absorbs the log whenever the run's own record changes,
  such as when it settles. A run is claimed by an operating-system lock on
  `run.lock`, so the command line and the web app cannot process one run at
  once. Resuming is refused when the settings differ from the run's, or when the
  column schema that wrote its done entries' partials differs from the current
  one.
- **Outputs appear whole.** Partials, assembled tables, and published copies are
  written to a scratch file and moved into place, so a reader never sees a table
  still being written. Every CSV file ends its lines with CRLF on every
  operating system, so the same data gives the same bytes and SHA-256
  everywhere. An automatically numbered output name is claimed for every table
  by creating each file exclusively, and a run that cannot be published leaves
  no table behind.
- **A run's status is decided when it settles.** A run is left `completed` or
  `paused` once its tables, published copies, and reports are written, and
  `error`, with the reason in `manifest.json`, when any of that fails or an
  error stops the build after it claimed the run. A run with no done entry gets
  its reports but no tables, and is left `error` (`NoDatasetError`), or `paused`
  if it was cancelled. An interruption such as Ctrl-C is not an error and leaves
  the run `running`. A run still recorded as `running` with no worker is stale,
  whatever its entries say, and `--resume` continues it.
- **Caches expire.** Entry metadata, validation reports, AlphaFold lookups, and
  downloaded mmCIF files are reused for 30 days
  (`engines/sources/http_cache.py`), since entries are revised after deposition.

## Adding an engine

A new engine is a package under `pdbai/engines/sources/` or
`pdbai/engines/analyses/` whose `engine.py` implements `spec` plus
`run(context)`, and one line adding it to `BUILTIN_ENGINES` in
`pdbai/engines/catalog.py`. The catalog's registry is built and validated when
it is first imported. Registering checks that the engine's name, columns,
tables, settings, and report fields clash with no other engine's. Validating
checks that toggles and settings name declared flags, that no settings refine
one another in a cycle, that dependencies exist and run in the same or an
earlier phase, and that no dependencies form a cycle. Only annotation engines
may declare columns or tables. The engine is turned on through its own declared
flag. The core names no engine. Every engine's settings live in
`EngineOptions.values` keyed by the names their `OptionSpec`s declare, and a
build fills any left out with their declared defaults. The engine's declared
columns then participate in enablement, ordering, documentation, filtering, and
star-schema placement. Its tables are written, assembled, and described in the
manifest and run summary. Its settings appear as command-line flags and web form
controls, and its report fields appear in the run report, the manifest, and the
run summary. `tests/integration/test_stub_engine.py` installs a registry holding
a stub engine pair that nothing else names and runs it end to end.

## Project structure

```
pdb-ai/
├── pdbai/                               # Main Python package
│   ├── __init__.py                      # Public API: the two dataset builders
│   ├── __main__.py                      # CLI entry point: python -m pdbai
│   ├── cli.py                           # Argument parsing and the command-line run
│   ├── results.py                       # Run and per-entry outcomes, run errors
│   ├── profiles.py                      # Column profiles and run-settings assembly
│   ├── column_filtering.py              # Column parsing, matching, and filtering
│   ├── dataset_shapes.py                # Flat vs. star_schema splitting
│   ├── provenance.py                    # Reproducibility receipts
│   ├── atomic.py                        # Atomic file replacement
│   ├── network.py                       # Retry sessions and network logging
│   ├── console_logging.py               # [pdbai]-prefixed standard-error logging
│   ├── json_values.py                   # Type narrowing for values read from JSON
│   ├── positive_numbers.py              # Shared positive-integer and positive-number checks
│   ├── nullable_booleans.py             # Strict nullable-boolean reading
│   ├── refused_values.py                # Refused values quoted in error messages
│   ├── missing_text.py                  # Texts read as missing values
│   ├── identifiers.py                   # PDB ID normalization and validation
│   ├── selection.py                     # Archive-wide selection (RCSB Search API)
│   ├── selection_filters.py             # Range filters of the selection, declared once
│   ├── licenses.py                      # Dependency license auditor
│   ├── structure/                       # Shared structural-data infrastructure
│   │   ├── cif_numbers.py               # CIF number grammar shared by every reader
│   │   ├── elements.py                  # Hydrogen element symbols shared by every reader
│   │   ├── mmcif_parser.py              # mmCIF → base atom frame
│   │   ├── mmcif_fields.py              # Raw mmCIF expansion and cell geometry
│   │   ├── refinement.py                # Refinement programs, TLS groups, anisotropic share
│   │   ├── coordinate_preprocessing.py  # Alternate-conformer selection
│   │   ├── residue_identity.py          # Residue identity and the named columns
│   │   └── alignment.py                 # RCSB↔PDB-Redo atom alignment
│   ├── orchestration/                   # Builds and per-entry processing
│   │   ├── api.py                       # Public build functions
│   │   ├── download.py                  # Acquisition for download builds
│   │   ├── upload.py                    # Entries from uploaded mmCIF files
│   │   ├── processing.py                # One entry to its recorded outcome
│   │   ├── annotation.py                # Adds the engines' columns
│   │   ├── technique_intake.py          # Refuses entries of other techniques
│   │   └── build.py                     # One build of a claimed run
│   ├── schema/                          # Application column catalog
│   │   ├── contracts.py                 # DatasetColumnSpec, roles, join columns
│   │   ├── base.py                      # Base mmCIF column declarations
│   │   ├── boolean_columns.py           # Boolean column checks
│   │   └── catalog.py                   # Base and engine columns in display order
│   ├── engines/                         # Toggleable enrichment engines
│   │   ├── core/                        # Engine system; imports no concrete engine
│   │   │   ├── contracts.py             # Spec, context, result, column contracts
│   │   │   ├── options.py               # EngineOptions and OptionSpec
│   │   │   ├── artifacts.py             # Typed ArtifactKey values
│   │   │   ├── profiles.py              # Curated profile identifiers
│   │   │   ├── registry.py              # Registration, ordering, artifact checks
│   │   │   ├── coordinator.py           # Phase execution, merging, failures
│   │   │   ├── tables.py                # Engine-table validation and site_id
│   │   │   └── techniques.py            # Experimental-technique vocabulary
│   │   ├── catalog.py                   # Built-in engines and the live registry
│   │   ├── compatibility.py             # Setting and technique compatibility
│   │   ├── sources/                     # Database acquisition and parsing
│   │   │   ├── http_cache.py            # Expiring caches and file downloads
│   │   │   ├── map_sources.py           # Which maps the density engine reads
│   │   │   ├── scalars.py               # Lenient numeric parsing
│   │   │   ├── rcsb/                    # Entry fetch and curated fields
│   │   │   ├── pdb_redo/                # PDB-Redo model, maps, and comparison
│   │   │   ├── alphafold/               # UniProt mapping and AFDB pLDDT
│   │   │   └── validation/              # wwPDB validation reports and maps
│   │   └── analyses/                    # Structural analysis engines
│   │       ├── density/                 # PDB-Redo map density at every atom
│   │       ├── rabdam/                  # rabdam-custom adapter, Bnet reference
│   │       └── sasa/                    # Solvent accessibility (SASA, RASA)
│   ├── runs/                            # Persisted runs
│   │   ├── state.py                     # Manifest, entry log, lock, resume checks
│   │   ├── storage.py                   # Run directory layout and name checks
│   │   ├── listing.py                   # Run summaries for the Runs panel
│   │   ├── partials.py                  # Per-structure partial CSVs
│   │   ├── assembly.py                  # Joins partials into the final tables
│   │   ├── publish.py                   # Copies tables into the working directory
│   │   └── settlement.py                # Assembly, publishing, reports, result
│   ├── reporting/                       # Run reporting
│   │   ├── run_report.py                # Per-PDB outcome CSV
│   │   ├── summary.py                   # Run summary and data dictionary
│   │   └── column_statistics.py         # Data-dictionary statistics
│   └── webapp/                          # Flask app (python -m pdbai.webapp)
│       ├── __main__.py                  # Development server
│       ├── factory.py                   # create_app(), guards, JSON errors
│       ├── options.py                   # Request options and run settings
│       ├── jobs.py                      # Background builds and job state
│       ├── route_helpers.py             # Shared request checks
│       ├── build_routes.py              # /api/build
│       ├── upload_routes.py             # mmCIF upload and CSV-filter routes
│       ├── select_routes.py             # /api/select
│       ├── run_routes.py                # Run status, files, pause, resume, delete
│       ├── templates/                   # index.html
│       └── static/                      # styles.css, app.js, logo.svg, favicon.svg
├── tests/                               # Offline pytest suite
├── docs/                                # These guides, plus plans/
├── pyproject.toml                       # Project metadata and tool settings
├── requirements.txt                     # Runtime dependencies
├── requirements-dev.txt                 # Test, lint, and type-check tools
└── THIRD_PARTY_NOTICES.md               # Dependency licenses and data provenance
```

### Runtime workspace layout

```
workspace/
├── mmcif/                            # Download caches
│   ├── <id>_metadata.json            # RCSB entry records
│   ├── <id>_validation.json          # wwPDB validation reports
│   ├── rcsb/                         # RCSB mmCIF files
│   ├── pdb_redo/                     # PDB-Redo mmCIF and reflection files
│   ├── pdb_redo_mirror/              # Optional local copy of PDB-Redo
│   ├── validation_maps/              # wwPDB validation map coefficients
│   └── afdb/                         # AlphaFold DB lookups
│       ├── entities/                 # Entity-to-UniProt alignments
│       └── models/                   # pLDDT and model provenance
├── uploads/                          # Web uploads, removed when their build ends
├── datasets/                         # Web filter-tool CSVs, kept at least an hour
├── tmp/                              # "Download All" archives
├── runs/<run_id>/                    # One directory per run
│   ├── manifest.json                 # Settings, state, provenance, receipts
│   ├── entries.jsonl                 # Entry updates not yet in the manifest
│   ├── events.jsonl                  # Append-only progress log
│   ├── run.lock                      # Held while a process works on the run
│   ├── partials/<table>/<entry>.csv  # Per-structure partial CSVs
│   ├── dataset.csv                   # Flat mode: the assembled table
│   ├── dataset_<table>.csv           # Star-schema and engine tables
│   ├── run_report.csv                # Per-PDB outcomes and engine report fields
│   ├── run_summary.md                # Readable provenance and data dictionary
│   └── job.json                      # Web job state
├── runs/.deleting-<run_id>-<hex>/    # A run being deleted; never listed
└── pdb_atom_site_dataset.csv         # Published primary table (CLI and library)
```

