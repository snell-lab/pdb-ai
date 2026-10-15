"""Structural biology datasets for AI, built from Protein Data Bank entries.

A build turns PDB entries into tables of atom-, residue-, chain-, and
structure-level data. It starts from PDB identifiers, typed or found by one
archive search, or from uploaded mmCIF files. It writes one flat CSV of atom
rows or a star schema of linked atom, residue, chain, structure, and citation
tables, and each enabled engine may add tables of its own. Each atom row keeps
every _atom_site item of the file. Engines add data from PDB-Redo, wwPDB
validation reports, UniProt, AlphaFold, and the RABDAM and SASA analyses. The
command line, the web app, and the Python API all run the same build through
the builders exported here. These are build_atom_site_dataset for identifiers
and build_atom_site_dataset_from_mmcif_files for files.

The subpackages follow the order of a build. orchestration runs it. engines
acquire each entry's files, structure parses the mmCIF file into atom rows,
and engines then annotate those rows. schema declares and orders every column.
runs keeps each run's manifest and partial tables, and reporting writes the
reports the run settles with. webapp serves the local web interface.

Nothing is guessed. A value an entry does not report stays missing, and one
that cannot be read is refused. A failing engine that is not required empties
only its own columns. One process at a time works on a run. Every file the run
reads back whole is replaced in one step, and the manifest records the code,
environment, and SHA-256 receipts that reproduce the run.
"""

from __future__ import annotations

from .orchestration import (
    build_atom_site_dataset,
    build_atom_site_dataset_from_mmcif_files,
)

__all__ = ["build_atom_site_dataset", "build_atom_site_dataset_from_mmcif_files"]
