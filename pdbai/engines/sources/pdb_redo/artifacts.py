"""The values the PDB-Redo engines hand to other engines.

PDB_REDO_CIF_PATH is the path of PDB-Redo's mmCIF file for the entry. The
source engine publishes it only when PDB-Redo has the entry.

PDB_REDO_DATAFRAME holds that file's atom rows. The structure engine
publishes it only when the file parsed into at least one atom.

PDB_REDO_MTZ_PATH is the path of PDB-Redo's reflection file for the entry,
which holds the map coefficients of its re-refined model. The map source
engine publishes it only when PDB-Redo has the file.

The comparison engine reads the atom rows, RABDAM reads the first two
values, and the density engine reads the reflection file. A reader gets None
for any of them when PDB-Redo has nothing usable for the entry. The reader
then leaves its own PDB-Redo output empty.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...core import ArtifactKey

PDB_REDO_CIF_PATH = ArtifactKey(
    "pdb_redo_cif_path",
    Path,
    "PDB-Redo's re-refined mmCIF for the entry, when PDB-Redo has one.",
)
PDB_REDO_DATAFRAME = ArtifactKey(
    "pdb_redo_dataframe",
    pd.DataFrame,
    "PDB-Redo's re-refined model parsed into an atom DataFrame.",
)
PDB_REDO_MTZ_PATH = ArtifactKey(
    "pdb_redo_mtz_path",
    Path,
    "PDB-Redo's reflection file for the entry, with the map coefficients of "
    "its re-refined model, when PDB-Redo has one.",
)

__all__ = ["PDB_REDO_CIF_PATH", "PDB_REDO_DATAFRAME", "PDB_REDO_MTZ_PATH"]
