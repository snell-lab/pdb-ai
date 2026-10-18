# PDB-AI

PDB-AI builds analysis-ready structural biology datasets for artificial
intelligence and machine learning (AI/ML). It converts Protein Data Bank (PDB)
structures into consistent atom-, residue-, chain-, and structure-level data.

The pipeline preserves RCSB mmCIF records and enriches them through modular
engines. Current analysis engines emphasize radiation-damage metrics, solvent
accessibility, and electron density at atoms.
Data-source engines add PDB-Redo comparisons, map coefficients from
PDB-Redo and the wwPDB, wwPDB validation data, UniProt mappings, and AlphaFold
confidence scores. The engine system can support additional structural biology analyses
without changing the dataset pipeline.

PDB-AI provides a command-line interface, a Python API, and a local Flask web
application.

## What PDB-AI does, and leaves to the analysis

PDB-AI builds data and decides which entries to build. It does not interpret
the data. Three rules follow from that:

- **Selection filters decide what is built.** Technique, X-ray source,
  temperature, resolution, and a list of UniProt accessions narrow the archive
  before anything is downloaded. They exist because a cohort of a few thousand
  entries cannot be picked out of two hundred thousand after the fact. A
  decision that can be made from the dataset's own columns is not a selection
  filter.
- **Columns record facts about an entry.** Anything readable from the archive
  record or the model, reusable beyond one study, is a column. The engines
  compute measurements, such as an atom's BDamage or its solvent accessibility,
  and those are facts too. A derived flag is one only when its rule is fixed,
  documented, and sits beside its evidence.
- **The analysis makes a study's choices.** Cohort labels, quality thresholds,
  matching and sampling, pairing controls across cohorts, and anything that
  needs a person to look belong to the analysis, not to PDB-AI.

Measurement in, judgment out. The output is one table per level of structure,
with a fixed set of columns, explicit missing values, and a manifest recording
how every value was obtained.

## Features

- Build general-purpose structural biology feature tables from PDB structures.
- Select a cohort from the whole archive (for example every XFEL entry) with one
  search, narrowed by temperature, resolution, or a list of UniProt accessions,
  and recorded in the run manifest for reproducibility.
- Preserve atom-level mmCIF fields and model-aware structure identity.
- Add experimental, validation, citation, UniProt, and AlphaFold metadata.
- Write one flat CSV or linked atom, residue, chain, structure, and citation
  tables.
- Extend the pipeline through registered source and analysis engines.
- Use the current SASA, RABDAM, density, and PDB-Redo engines when those
  features apply.
- Resume interrupted runs from per-structure partial files.
- Record source files, hashes, software versions, settings, and run outcomes.

## Installation

PDB-AI requires Python 3.12 or later. Run it from a cloned repository.

```bash
git clone https://github.com/snell-lab/pdb-ai.git
cd pdb-ai
python -m venv .venv
```

Activate the environment:

```text
Windows:  .venv\Scripts\activate
Unix:     source .venv/bin/activate
```

Install the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

The dependencies include `gemmi`, NumPy, pandas, SciPy, Flask, Requests, and
`rabdam`, installed from the rabdam-custom repository. PDB-AI does not require
Biopython, Docker, or external scientific binaries.

See [third-party notices](THIRD_PARTY_NOTICES.md) for license and bundled-data
provenance.

## Quick start

Build a flat dataset for two structures:

```bash
python -m pdbai 5MCC 4HHB --workdir ./workspace
```

Enable SASA, AlphaFold, and the PDB-Redo comparison:

```bash
python -m pdbai 5MCC --workdir ./workspace \
  --include-sasa \
  --include-alphafold \
  --include-pdb-redo
```

Build the five-table star schema with the standard profile, which keeps every
column PDB-AI defines for the engines you enable and leaves out the raw mmCIF
items it only passes through.

```bash
python -m pdbai 5MCC --workdir ./workspace \
  --dataset-shape star_schema \
  --include-rabdam --include-sasa \
  --profile standard
```

Build an electron-microscopy cohort. RABDAM and PDB-Redo do not apply to it and
are left off, and any X-ray entry in the list is refused at intake, before
download.

```bash
python -m pdbai 6VXX --workdir ./workspace \
  --technique electron_microscopy \
  --include-sasa
```

Let the archive pick the entries instead of listing them. One query to the RCSB
Search API returns every current PDB entry matching the technique and X-ray
source, narrowed by optional temperature and resolution ranges. The query and
time are recorded in the run manifest, so the cohort can be reproduced.

```bash
# List every XFEL entry in the PDB without building anything.
python -m pdbai --select --xray-source free_electron_laser --list-only

# Build a dataset from all of them.
python -m pdbai --select --xray-source free_electron_laser --workdir ./workspace

# List the room-temperature synchrotron entries at 2.5 Å or finer.
python -m pdbai --select --xray-source synchrotron \
  --min-temperature 273 --max-resolution 2.5 --list-only
```

Every optional engine is off by default, on the command line, in the Python API,
and in the web application alike, unless a profile turns it on. One run
processes one experimental technique (`--technique`, X-ray diffraction by
default). See the [CLI reference](docs/cli.md#technique-selection) and
[Selecting entries from the
archive](docs/cli.md#selecting-entries-from-the-archive).

## Outputs

Flat mode writes one row per atom to `pdb_atom_site_dataset.csv` in the working
directory, or to the first free `pdb_atom_site_dataset_N.csv` when that file
exists. `--output` sets another name. Web builds keep their files in the run
directory instead. Star-schema mode writes linked atom, residue, chain,
structure, and citation tables. The atom table takes the name above, and each
other table is written beside it with `_<table>` added before `.csv`, as in
`pdb_atom_site_dataset_residues.csv`.

Each run also writes persistent artifacts under `workspace/runs/<run_id>/`:

- `manifest.json` contains settings, state, provenance, and file receipts.
- `entries.jsonl` holds per-structure updates not yet folded into the manifest.
- `events.jsonl` contains the append-only progress log.
- `run_report.csv` records the outcome for each structure and engine, including
  entries refused at technique intake and the methods they reported.
- `run_summary.md` explains the configuration and profiles every output column.
- `dataset.csv`, or `dataset_<table>.csv` in star-schema mode, holds the
  assembled tables.
- `partials/` contains finalized per-structure data used for resume and
  assembly.

See the [dataset schema](docs/dataset-schema.md) for table and column details.

## Interfaces

### Command line

```text
python -m pdbai [PDB_IDS ...] [OPTIONS]
```

Run `python -m pdbai --help` for built-in help (`python -m pdbai.cli` is the
same command). The [CLI reference](docs/cli.md) explains every option and its
default.

### Web application

```bash
python -m pdbai.webapp
```

Open <http://localhost:8001>. The server binds to `127.0.0.1` and disables the
debugger by default. It is intended for local use and does not provide
authentication.

See the [web application guide](docs/web-application.md) for uploads, filtering,
run management, and deployment cautions.

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
print(result.run_summary_path)
```

`build_atom_site_dataset_from_mmcif_files`, exported alongside it, builds the
same way from local mmCIF files. See the [library
reference](docs/cli.md#library-api) for filtering, engine options, and result
fields.

## Documentation

- [CLI and library reference](docs/cli.md)
- [Web application](docs/web-application.md)
- [Dataset schema](docs/dataset-schema.md)
- [Scientific methods](docs/scientific-methods.md)
- [Data sources and limitations](docs/data-sources.md)
- [Architecture](docs/architecture.md)

## Reproducibility

Every invocation of a run, whether it completes, pauses, or fails, records the
Git revision and dirty state it started from, with the Python environment and
dependency versions. A resumed run adds a record for each later invocation and
keeps the earlier ones. The run also records the source URLs, cache state, and
SHA-256 hashes of the files it reads and writes.

The machine-readable record is in `manifest.json`. The same information appears
in `run_summary.md` with the configuration and data dictionary.

## Citation

Citation metadata is available in [CITATION.cff](CITATION.cff). Cite the
generated `run_summary.md` with the repository when reporting a specific
dataset.

## License

PDB-AI is available under the [Apache License 2.0](LICENSE). Third-party
licenses and bundled-data provenance are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
