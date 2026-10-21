# PDB-AI

PDB-AI turns Protein Data Bank (PDB) structures into analysis-ready datasets for
AI/ML. It builds consistent atom-, residue-, chain-, and structure-level tables
from RCSB mmCIF records.

Optional engines add radiation-damage metrics (RABDAM), solvent accessibility
(SASA), electron density at atoms, PDB-Redo comparisons, wwPDB validation data,
UniProt mappings, and AlphaFold confidence scores.

PDB-AI is meant to be used through its local web application. A command line
and Python API are also available for scripting.

## Installation

Requires Python 3.12 or later.

```bash
git clone https://github.com/snell-lab/pdb-ai.git
cd pdb-ai
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Quick start

```bash
python -m pdbai.webapp
```

Then open <http://localhost:8001>, choose entries and engines, and download the
dataset when the run finishes. The server is for local use only and has no
authentication; see the [web application guide](docs/web-application.md).

To script runs instead, see the [user guide](docs/user-guide.md).

## Requesting an engine

Anyone can ask for a new engine, whether a data source or an analysis.
[Open an issue](https://github.com/snell-lab/pdb-ai/issues) describing what it
should compute or fetch and where the data comes from. To write one yourself,
see [Adding an engine](docs/architecture.md#adding-an-engine).

## Documentation

- [Web application](docs/web-application.md)
- [User guide](docs/user-guide.md)
- [CLI and library reference](docs/cli.md)
- [Dataset schema](docs/dataset-schema.md)
- [Scientific methods](docs/scientific-methods.md)
- [Data sources and limitations](docs/data-sources.md)
- [Architecture](docs/architecture.md)

## Citation and license

Cite PDB-AI using [CITATION.cff](CITATION.cff). When reporting a specific
dataset, also cite its generated `run_summary.md`.

PDB-AI is licensed under the [Apache License 2.0](LICENSE). Third-party licenses
are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
