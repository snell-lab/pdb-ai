# User Guide

This guide covers what PDB-AI builds, common runs, the files a run writes, and
the three ways to use it.

[Project README](../README.md)

## Scope

PDB-AI measures; the analysis judges.

- **Selection filters decide what is built.** Technique, X-ray source,
  temperature, resolution, and UniProt accessions narrow the archive before
  download.
- **Columns record facts.** Archive fields, model data, and engine measurements
  are columns. A derived flag is a column only when its rule is fixed,
  documented, and sits beside its evidence.
- **Study choices belong to the analysis.** Cohort labels, quality thresholds,
  matching, sampling, and anything needing human review are left to you.

## Common runs

These command-line examples show what a run can do. The same choices are
available in the [web application](web-application.md).

### Build from a list of entries

```bash
python -m pdbai 5MCC 4HHB --workdir ./workspace
```

### Add engines and write the star schema

```bash
python -m pdbai 5MCC --workdir ./workspace \
  --dataset-shape star_schema --profile standard \
  --include-rabdam --include-sasa --include-pdb-redo
```

### Choose a technique

Each run handles one technique, X-ray diffraction by default. Entries recorded
under another method are refused before download.

```bash
python -m pdbai 6VXX --workdir ./workspace \
  --technique electron_microscopy --include-sasa
```

### Select entries from the archive

```bash
# List every XFEL entry without building anything
python -m pdbai --select --xray-source free_electron_laser --list-only

# Build room-temperature synchrotron entries at 2.5 Å or finer
python -m pdbai --select --xray-source synchrotron \
  --min-temperature 273 --max-resolution 2.5 --workdir ./workspace
```

The query is recorded in the run manifest so the cohort can be reproduced.

Optional engines are off by default unless a profile enables them. See the
[CLI reference](cli.md) for every option.

## Outputs

Flat mode writes one row per atom to `pdb_atom_site_dataset.csv`, or to the
next free `_N` name; `--output` sets another. Star-schema mode writes linked
atom, residue, chain, structure, and citation tables, each suffixed `_<table>`.

Each run also keeps a record under `workspace/runs/<run_id>/`:

| File                            | Contents                                   |
|---------------------------------|--------------------------------------------|
| `manifest.json`                 | Settings, state, provenance, file hashes   |
| `run_report.csv`                | Outcome per structure and engine           |
| `run_summary.md`                | Configuration and column profiles          |
| `events.jsonl`, `entries.jsonl` | Progress log and pending updates           |
| `partials/`                     | Per-structure data for resume and assembly |

Every invocation records the Git revision, Python environment, dependency
versions, source URLs, and SHA-256 hashes. See the
[dataset schema](dataset-schema.md) for column details.

## Interfaces

### Web application

The web application is the intended way to use PDB-AI.

```bash
python -m pdbai.webapp
```

Then open <http://localhost:8001>. The server binds to `127.0.0.1` and has no
authentication. See the [web application guide](web-application.md).

### Command line

The command line and Python API are optional, for scripted or automated runs.

```text
python -m pdbai [PDB_IDS ...] [OPTIONS]
```

Run `python -m pdbai --help` for built-in help.

### Python API

```python
from pathlib import Path

from pdbai import build_atom_site_dataset
from pdbai.engines.core import EngineOptions

result = build_atom_site_dataset(
    pdb_ids=["5MCC", "4HHB"],
    working_directory=Path("./workspace"),
    engine_options=EngineOptions(values={"include_sasa": True}),
)
print(result.csv_path)
```

Use `build_atom_site_dataset_from_mmcif_files` to build from local mmCIF files.
See the [library reference](cli.md#library-api).
