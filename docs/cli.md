# CLI and Library Reference

Use the command-line interface for scripted runs. Use the Python API to
integrate PDB-AI into another workflow.

[Project README](../README.md)

## CLI reference

```
python -m pdbai [PDB_IDS ...] [OPTIONS]
```

`python -m pdbai.cli` is the same command.

| Option | Default | Description |
|--------|---------|-------------|
| `PDB_IDS` | *(required unless `--select`)* | One or more PDB identifiers (e.g. `5MCC 4HHB`). |
| `--select` | off | Take the identifiers from an archive search instead of listing them. The search returns every current PDB entry matching `--technique` and `--xray-source`, narrowed by any range filter below (see [Selecting entries from the archive](#selecting-entries-from-the-archive)). Cannot be combined with positional IDs. |
| `--list-only` | off | With `--select`, print the matching identifiers, one per line, to standard output and exit without building. Everything else goes to standard error, so the output can be piped or saved. |
| `--limit N` | unset | With `--select`, keep at most `N` matching identifiers, taken in sorted order so the sample is deterministic. The full match count is still reported and recorded. |
| `--min-temperature KELVIN` | unset | With `--select`, keep an entry only when every data-collection temperature it records is at least this many kelvin (see [Range filters](#range-filters)). |
| `--max-temperature KELVIN` | unset | With `--select`, keep an entry only when every data-collection temperature it records is at most this many kelvin. |
| `--max-resolution ANGSTROMS` | unset | With `--select`, keep an entry only when every resolution it records is at most this many ångströms. A smaller value is a finer resolution. |
| `--uniprot-accessions FILE` | unset | With `--select`, keep an entry only when one of its polymer entities is mapped to a UniProt accession listed in `FILE` (see [Matching by UniProt accession](#matching-by-uniprot-accession)). |
| `--technique ID` | `x_ray_diffraction` | The experimental technique this run processes, one per run (see [Technique selection](#technique-selection)). Entries the PDB records under any other method are refused at intake, before download, and listed as refused in the run report. |
| `--xray-source ID` | `all` | Second-level filter within X-ray diffraction, one of `synchrotron`, `free_electron_laser`, `rotating_anode`, `sealed_tube`, or `liquid_anode` (see [X-ray source filter](#x-ray-source-filter)). Only valid with `--technique x_ray_diffraction`. |
| `--workdir PATH` | `./workspace` | Directory for the downloads, the run directories with their per-structure partials, and the published output CSVs. |
| `--output FILENAME` | auto | Filename of the primary table, a single name that ends in `.csv` and does not start with a period, not a path. Every other table is written beside it as `<name>_<table>.csv`. The files replace any of the same names. Without it, the run takes the first free of `pdb_atom_site_dataset.csv`, `pdb_atom_site_dataset_1.csv`, …, and keeps that name when it is resumed. |
| `--profile {all,standard,minimal}` | none | Apply a pre-defined column profile (see [Column profiles](dataset-schema.md#column-profiles)). `minimal` switches on the engines its columns need; `all` and `standard` switch nothing on. |
| `--column-filter-mode {include,exclude}` | none | Manual column filter mode. Case is ignored. Replaces the column filter of `--profile`, whose engines, if any, are still switched on. Must be used together with `--column-filter-list`. |
| `--column-filter-list COL [COL ...]` | none | Column names for manual filtering. Each value may also hold several names separated by commas, semicolons, or line breaks. A name matches the column spelled the same, else one spelled the same ignoring case, else one with the same letters and digits, so `_cell_length_a` finds `_cell.length_a`. A `*` matches any run of characters, ignoring case (`*factor*`). A name that matches nothing is ignored with a warning. Must be used together with `--column-filter-mode`. |
| `--workers N` | `min(8, n)` | Positive number of workers downloading entries' data in parallel, ahead of processing, never more than `n`, the number of entries still to process. With `1`, one entry downloads at a time. |
| `--run-id ID` | auto-uuid | Name the run directory (`workspace/runs/<ID>/`) using 1 to 128 ASCII letters, digits, periods, underscores, or hyphens, starting with a letter or digit. Paths, `.`/`..`, and reserved filenames are rejected. Without `--resume` the ID must name no existing run, so a finished run is never built over. Pair with `--resume` to continue an interrupted run. |
| `--resume` | off | Continue the run `--run-id` names, which must exist and have the same settings. PDB IDs given now that the run lacks are added to it. Each of its IDs the manifest records as done, whose partial file for every table still exists in the run directory, is skipped, and any other is processed again. The dataset and run report cover the whole run, not only the IDs given now. |
| `--dataset-shape {flat,star_schema}` | `flat` | `flat` writes one CSV, one row per atom. `star_schema` writes `atoms`/`residues`/`chains`/`structures`/`citations` CSVs. See [Dataset shape](dataset-schema.md#dataset-shape-flat-vs-star_schema). |

### Optional engines

RCSB's standard mmCIF is always the base structure source. Every other source
and analysis is an engine, and its flags, help text, and defaults are generated
from the settings the engine declares, so `python -m pdbai --help` is the full
reference. The Python API takes the same settings by name in
`EngineOptions.values`.

| Flag | Default | Setting | Purpose |
|------|---------|---------|---------|
| `--include-pdb-redo` | off | `include_pdb_redo` | PDB-Redo's re-refined coordinates and deltas versus RCSB. |
| `--include-rabdam` | off | `include_rabdam` | BDamage, Bnet, and packing density (X-ray diffraction only). |
| `--rabdam-source {pdb_redo,rcsb,both}` | `pdb_redo` | `rabdam_source` | Which mmCIF RABDAM scores. `pdb_redo` and `both` fetch PDB-Redo's file even without `--include-pdb-redo` (see [Radiation damage (RABDAM)](scientific-methods.md#radiation-damage-rabdam)). |
| `--include-sasa` | off | `include_sasa` | Protein-only and bound-context SASA/RASA. |
| `--sasa-accuracy {fast,default,high}` | `default` | `sasa_accuracy_mode` | 92, 1,000, or 5,000 sphere points per atom. |
| `--sasa-points N` | unset | `sasa_n_sphere_points` | Explicit point count overriding `--sasa-accuracy`. |
| `--sasa-include-experimental-additives` | off | `sasa_include_experimental_additives` | Count crystallization additives as bound-context occluders. |
| `--sasa-exclude-unknown-nonpolymers` | off | `sasa_include_unknown_nonpolymers` (default `True`) | Leave unclassified non-polymers out of the bound context. |
| `--include-density` | off | `include_density` | Density of an entry's 2mFo−DFc and mFo−DFc maps at every atom, in units of each map's root-mean-square density, and, from PDB-Redo's maps, the model's calculated density and the difference density on one scale (X-ray diffraction only). |
| `--density-source {pdb_redo,wwpdb,both}` | `pdb_redo` | `density_map_source` | Which maps the density columns are read from. `pdb_redo` fetches PDB-Redo's model and reflection file and reads its maps at the deposited position and at PDB-Redo's. `wwpdb` fetches the wwPDB validation map coefficients, calculated about the deposited model, and reads them at the deposited position. `both` does both (see [Electron density at atoms](scientific-methods.md#electron-density-at-atoms)). |
| `--include-alphafold` | off | `include_alphafold` | AlphaFold Database status, provenance, and per-residue pLDDT. Keeps `uniprot_id`. |
| `--include-uniprot` | off | `include_uniprot` | Per-residue UniProt accession (`uniprot_id`). |
| `--include-validation` | off | `include_validation` | wwPDB validation metrics (RSCC, RSR, Ramachandran, rotamer, clashscore). |

Engines that do not apply to the run's technique stay off. See [Technique
selection](#technique-selection).

### Technique selection

One run processes one experimental technique, chosen with `--technique`. The
options are the PDB's own `_exptl.method` vocabulary, one to one, plus the
integrative methodology the archive classifies separately:

| `--technique` | PDB method | Shown as |
|---------------|------------|----------|
| `x_ray_diffraction` (default) | `X-RAY DIFFRACTION` | X-ray diffraction (XRD) |
| `electron_microscopy` | `ELECTRON MICROSCOPY` | Cryogenic electron microscopy (cryo-EM) |
| `solution_nmr` | `SOLUTION NMR` | Solution nuclear magnetic resonance (solution NMR) |
| `electron_crystallography` | `ELECTRON CRYSTALLOGRAPHY` | Electron crystallography (MicroED) |
| `neutron_diffraction` | `NEUTRON DIFFRACTION` | Neutron diffraction |
| `solid_state_nmr` | `SOLID-STATE NMR` | Solid-state nuclear magnetic resonance (ssNMR) |
| `solution_scattering` | `SOLUTION SCATTERING` | Small-angle solution scattering (SAXS/SANS) |
| `fiber_diffraction` | `FIBER DIFFRACTION` | Fiber diffraction |
| `powder_diffraction` | `POWDER DIFFRACTION` | Powder diffraction |
| `epr` | `EPR` | Electron paramagnetic resonance (EPR) |
| `theoretical_model` | `THEORETICAL MODEL` | Theoretical model |
| `infrared_spectroscopy` | `INFRARED SPECTROSCOPY` | Infrared spectroscopy (IR) |
| `fluorescence_transfer` | `FLUORESCENCE TRANSFER` | Fluorescence resonance energy transfer (FRET) |
| `integrative` | integrative methodology | Integrative/hybrid methods (IHM) |

Before anything is downloaded, each entry's RCSB record is checked. An entry is
admitted when the selected technique is among the methods it reports (a joint
X-ray/neutron refinement reports both and is admitted to either run), refused
otherwise, and admitted as *unverified* when no method is recorded anywhere.
When the record cannot be fetched or names no method, the downloaded file's own
`_exptl.method` and `_diffrn_source.source` decide once it is parsed. An entry
refused by its record never reaches the structure download. One refused by its
downloaded file has already been through the engines that fetch its data, but
none of the engines that annotate its atoms runs. Refused entries appear in
`run_report.csv` with `state = refused`, `technique_status = refused`, and their
reported `experimental_methods`, and the run finishes normally. A refusal is an
expected outcome of the filter, not a failure, and does not affect the exit code
as long as some entry is built. Uploaded mmCIF files are checked the same way,
against their own `_exptl.method` and `_diffrn_source.source`, after parsing.

Every engine declares which techniques its science applies to. Enabling one for
a technique it does not support is rejected before the run starts:

| Engine / option | Applies to |
|-----------------|------------|
| RABDAM, PDB-Redo comparison, Density | X-ray diffraction only |
| wwPDB validation | X-ray, electron microscopy, neutron, electron crystallography, solution and solid-state NMR |
| SASA, AlphaFold Database, UniProt | every technique |

A column profile that lists an engine the technique rules out leaves that engine
off and says so, rather than failing, and leaves that engine's columns out of
its column filter.

### X-ray source filter

Within X-ray diffraction, `--xray-source` selects on the radiation source the
entry reports (`_diffrn_source.source`). This is how the PDB distinguishes XFEL
data from everything else, since XFEL entries are filed as X-ray diffraction
with a free-electron-laser source, not as a method of their own.

| `--xray-source` | PDB source | Shown as |
|-----------------|------------|----------|
| `all` (default) | no filter | All sources |
| `synchrotron` | `SYNCHROTRON` | Synchrotron |
| `free_electron_laser` | `FREE ELECTRON LASER` | X-ray free-electron laser (XFEL) |
| `rotating_anode` | `ROTATING ANODE` | Rotating anode |
| `sealed_tube` | `SEALED TUBE` | Sealed tube |
| `liquid_anode` | `LIQUID ANODE` | Liquid anode |

The technique is checked first, then the source. An entry is admitted when the
selected source is among the sources it reports (a joint X-ray/neutron entry
lists its X-ray source alongside a reactor or spallation source). It is refused
when it reports only other sources. Unlike the technique check, it is also
refused when it records **no** source at all, because a missing source cannot be
shown to match. The refusal says which case applied. Deposited spellings such as
`rotating-anode X-ray tube` or `Rotating Copper Anode` are normalized to the
canonical value before matching, and the same normalized values are written to
the `radiation_sources` column so the filter and the output agree.

```bash
# Every XFEL entry in a list, refusing the synchrotron ones before download.
python -m pdbai 9EZQ 9G3W 5MCC --xray-source free_electron_laser
```

### Selecting entries from the archive

PDB-AI treats the PDB as the hub. PDB-Redo, AlphaFold, wwPDB validation, and
UniProt are all looked up by PDB identifier. So a cohort only ever has to be
selected once, against the archive itself, and `--select` does exactly that.
Instead of listing identifiers, it sends one query to the [RCSB Search
API](https://search.rcsb.org) for every current entry matching `--technique` and
`--xray-source`, narrowed by any [range filter](#range-filters), and uses the
answer as the run's identifier list. Nothing is crawled or snapshotted locally.

```bash
# Every XFEL entry in the PDB, as a list you can save or pipe.
python -m pdbai --select --xray-source free_electron_laser --list-only > xfel_ids.txt

# Build the whole XFEL cohort (about a thousand entries at the time of writing).
python -m pdbai --select --xray-source free_electron_laser --workdir ./workspace

# A deterministic 50-entry sample of cryo-EM entries.
python -m pdbai --select --technique electron_microscopy --limit 50 --include-sasa

# Room-temperature synchrotron entries at 2.5 Å or finer.
python -m pdbai --select --xray-source synchrotron \
  --min-temperature 273 --max-resolution 2.5 --list-only > rt_synchrotron_ids.txt
```

The query uses the same vocabulary as intake, `exptl.method` for the technique
(or the structure-determination methodology for integrative entries) and
`diffrn_source.source` for the X-ray source, with exact matches on the archive's
canonical values. So what the search returns is what intake admits. Intake still
re-reads each entry's own record before download, so if the search index and an
entry ever disagree the entry is refused with the reason in the run report
rather than trusted. Searching titles or keywords instead would over-select:
synchrotron control structures deposited alongside XFEL papers routinely mention
"XFEL" or "serial femtosecond" in their titles.

The search result is recorded in the run manifest under `selection` (`filters`,
service, the exact query sent, the retrieval time, the match count, and the
identifiers kept) and summarized in `run_summary.md`, so a cohort can be
reproduced or compared against a later archive. `filters` holds the technique,
the X-ray source, the limit, and one key for every range filter, which is null
when the filter was not set. It is provenance of the input list rather than a
setting. A `--resume` of the run is compatible with or without `--select`, keeps
the original cohort record, and appends any newer search to its execution
session (new matches are added to the run).

#### Range filters

A range filter narrows the search by a number the archive records for each
entry. Every filter is optional, needs `--select`, and takes a positive number
in plain decimal form, such as `273` or `2.5`.

| Flag | Record key | Archive attribute | Keeps an entry when |
|---|---|---|---|
| `--min-temperature` | `min_temperature_K` | `diffrn.ambient_temp` | every data-collection temperature is at least this many kelvin |
| `--max-temperature` | `max_temperature_K` | `diffrn.ambient_temp` | every data-collection temperature is at most this many kelvin |
| `--max-resolution` | `max_resolution_A` | `rcsb_entry_info.resolution_combined` | every resolution is at most this many ångströms |

Three rules hold for every range filter:

- **A range is strict.** An entry may record a quantity more than once. It
  holds a temperature for each data-collection record, and a joint
  X-ray/neutron entry holds a resolution for each of its two data sets. Those
  are measurements of different experiments, so none speaks for the others, and
  the entry matches only when every value it records is in range. An entry with
  room-temperature and cryogenic data sets therefore matches neither
  `--min-temperature 273` nor `--max-temperature 120`. In a joint entry the
  neutron data is usually the coarser, so `--max-resolution` in effect holds a
  joint entry to its neutron resolution.
- **An entry that records no value is left out.** Nothing then shows what the
  entry holds. Some entries record no data-collection temperature, and any
  temperature filter drops them.
- **A range narrows the search only.** Intake checks each entry's technique and
  X-ray source again, but not its ranges. The dataset's own columns, such as
  `temperature_K`, remain the record of what each entry holds. `temperature_K`
  is read from the first data-collection record, and under a temperature filter
  it is in range whenever that record holds a temperature.

The resolution searched is the refinement limit, which the dataset's
`resolution` column reads first too. The data-processing limit is recorded as
well, but it can be finer than the data the model was refined against.

Giving both temperature flags selects a window. A minimum above the maximum is a
usage error, since no entry could match. The filters are declared once, in
`pdbai/selection_filters.py`, and the flags, the web form fields, the JSON keys,
and the record keys are generated from that declaration.

#### Matching by UniProt accession

`--uniprot-accessions FILE` keeps the entries whose polymer entities RCSB maps
to any accession listed in `FILE`, one per line or separated by commas or
spaces. Case is ignored and an isoform suffix such as `-2` is dropped, since the
archive maps an entity to the accession itself. A token that is not a UniProt
accession, or a file that names none, is a usage error.

This is how a cohort of the same proteins as another cohort is selected without
building the whole archive. The distinct `uniprot_id` values of one run's
dataset, written to a file, select the matching entries of another technique:

```bash
# The cryo synchrotron entries of the proteins in a neutron run's dataset.
python -m pdbai --select --xray-source synchrotron --max-temperature 120 \
  --uniprot-accessions neutron_accessions.txt --list-only
```

The accessions are recorded in the manifest under
`selection.filters.uniprot_accessions`, and the criteria text counts them.

Identifiers are returned in sorted order. `--limit N` keeps the first `N`, so
the same limit on the same archive gives the same sample. An empty match ends
the run with an error unless `--list-only` is given, in which case nothing is
printed and the exit status is `0`. A search failure ends the run with status
`2`.

The CLI exits with status `0` when the run finishes and no entry failed or was
left pending, and with status `2` when any entry failed to download, parse, or
write its output, or was left pending. An entry refused at intake, or one whose
metadata fetch failed, is listed in the run summary but does not change the exit
status, so `0` does not mean every requested identifier produced rows. Neither
does a warning that the archive made an entry obsolete, naming the entries that
replaced it, that a file lists modelled residues it holds no atoms for, as a
file cut short does, or that a file without author chain IDs or author residue
numbers is numbered by its label identifiers. Each such entry is built as given
and named in the warnings. When no entry is built at all, because every one was
refused or failed, no dataset is written, and the command exits with `2` and the
reason after writing the run report and summary. Usage errors (a malformed PDB
identifier, conflicting flags, a setting of an engine the run leaves off, such
as `--sasa-points` without `--include-sasa`, or a setting the technique or
profile rules out), a `--run-id` naming no run to `--resume` or an existing run
without it, a run another process holds or one created with other settings, a
failed archive search, and a search matching nothing also exit with `2`, before
anything is downloaded, and print their message on standard error as `<prog>:
error: <message>`. Other errors end the command with a traceback and status `1`.


## Library API

Select a cohort from the archive and record how it was chosen:

```python
from pathlib import Path
from pdbai import build_atom_site_dataset
from pdbai.engines.core import EngineOptions
from pdbai.selection import Selection, search_archive

search_result = search_archive(
    Selection(technique="x_ray_diffraction", xray_source="free_electron_laser")
)
# e.g. 1041 ('11YO', '13CH', ...)
print(search_result.total_count, search_result.pdb_ids[:5])

result = build_atom_site_dataset(
    pdb_ids=search_result.pdb_ids,
    working_directory=Path("./workspace"),
    engine_options=EngineOptions(xray_source="free_electron_laser"),
    search_result=search_result,  # recorded in manifest.json as input provenance
)
```

`Selection(..., limit=N)` keeps the first `N` identifiers in sorted order.
`Selection(..., bounds={"min_temperature_K": 273, "max_resolution_A": 2.5})`
adds [range filters](#range-filters) by their record keys. `search_archive`
raises `ArchiveSearchError` when the request fails or the service does not
answer with a result set.

Build from a typed list:

```python
from pathlib import Path
from pdbai import build_atom_site_dataset
from pdbai.engines.core import EngineOptions

result = build_atom_site_dataset(
    pdb_ids=["5MCC", "4HHB"],
    working_directory=Path("./workspace"),
    column_filter_mode="include",  # or "exclude", or None
    column_filter_list=[
        "pdb_id",
        "chain_id",
        "bdamage_pdb_redo",
        "rasa_protein",
        "rasa_bound",
        "nonprotein_occluded_fraction",
    ],
    engine_options=EngineOptions(
        dataset_shape="flat",  # or "star_schema"
        technique="x_ray_diffraction",  # default, one technique per run
        xray_source="all",  # or "synchrotron", "free_electron_laser", ...
        values={  # engine settings by name; any left out take their defaults
            "include_pdb_redo": False,  # default
            "include_rabdam": True,  # off by default
            "rabdam_source": "pdb_redo",  # default, or "rcsb" / "both"
            "include_sasa": True,  # off by default
            "sasa_accuracy_mode": "default",  # fast (92), default (1,000), or high (5,000)
            "sasa_n_sphere_points": None,  # optional explicit point-count override
            "sasa_include_experimental_additives": False,
            "sasa_include_unknown_nonpolymers": True,
            "include_density": False,  # default
            "density_map_source": "pdb_redo",  # default, or "wwpdb" / "both"
            "include_alphafold": False,  # default
            "include_uniprot": False,  # default
            "include_validation": False,  # default
        },
    ),
)

print(result.csv_path)  # The primary table
print(result.additional_csv_paths)  # Every other table, by name
print(result.atom_count)  # Rows in the primary table
print(result.run_report_path)  # Per-PDB outcome report
print(result.run_summary_path)  # Settings and data dictionary (Markdown)
print(result.pdb_results)  # One PdbProcessResult per PDB ID
print(result.warnings)  # Non-fatal warnings
```

`EngineOptions` holds the run-level fields (`technique`, `xray_source`,
`dataset_shape`) and, in `values`, every engine's settings keyed by the option
name the engine declares. It is frozen. Derive a variant with
`options.with_values({"include_sasa": True})` or, for a run-level field,
`dataclasses.replace(options, technique=...)`. Setting values are checked
against each engine's declaration by
`pdbai.engines.compatibility.validate_options`, which the build functions call
for you. Pass `publish=False` to keep the assembled tables only in the run
directory instead of also copying them into the working directory.

A setting left out of `values` takes its declared default when the build starts
(`get_engine_registry().with_defaults(options)`, from `pdbai.engines.catalog`,
shows the result). Every engine is off, and RABDAM, once enabled, scores
`rabdam_source="pdb_redo"`. `dataset_shape` defaults to `"flat"`, and
`technique` to `"x_ray_diffraction"`. Omit `engine_options` entirely to get that
default combination, which is the same on the command line and in the web UI.
Enabling an engine the technique rules out, such as RABDAM for
`"electron_microscopy"`, makes `build_atom_site_dataset` raise `ValueError`
naming it (`pdbai.engines.compatibility.compatible_toggles(technique)` lists
what applies).

Builds from local files use `build_atom_site_dataset_from_mmcif_files`, also
exported from `pdbai`. It takes `mmcif_files`, an iterable of paths, in place of
`pdb_ids`, and otherwise the same arguments except `search_result` and
`max_workers`. Both builders return a `DatasetBuildResult`, which lives in
`pdbai.results` together with `PdbProcessResult`. When no entry is done, because
every one was refused or failed, they raise `NoDatasetError` instead, and when
the build is cancelled before any entry is done, `RunPausedError`. Both errors
are in `pdbai.results`, and the run report and summary are still written.

`DatasetBuildResult` has these fields: `csv_path`, `additional_csv_paths`,
`run_report_path`, `run_summary_path`, `atom_count`, `column_count`, `pdb_ids`,
`preview_rows`, `pdb_results`, `warnings`, `column_filter`, and `paused`.
`paused` is true when the build was cancelled after some entries were done but
before it recorded every entry, which leaves the run paused with a dataset of
the entries done so far. Each `PdbProcessResult` carries `technique_status`
(`matched`, `refused`, or `unverified`), `technique_message`, the entry's
`experimental_methods`, and `engine_report`, the report fields of every engine
that ran for it (the same values as in `run_report.csv`).
`PdbProcessResult.refused()` is true for entries refused at intake, whose
`state` is `"refused"`. The step methods `metadata_succeeded()`,
`download_succeeded()`, and `parse_succeeded()` return `None` for a step the
entry never reached or whose outcome was never recorded, so a step that may not
have been tried is never reported as failed. `metadata_ok` is likewise `None`
until a download's RCSB record is sought, and always `None` for an upload.

