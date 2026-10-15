"""PDB-Redo's re-refined model of an entry, fetched, parsed, and compared.

artifacts.py names the three values the PDB-Redo engines hand to other
engines. They are the path of the model file, its parsed atom rows, and the
path of the reflection file that holds the model's map coefficients.

comparison.py pairs PDB-Redo's atoms with the RCSB atoms. For each pair it
takes the difference in coordinates, B factor, occupancy, and equivalent U.

mirror.py finds an entry's files in a local copy of PDB-Redo's databank,
which both source engines read before they make a request.

engine.py registers four engines. The source engine downloads the model
file, the structure engine parses it, and the comparison engine adds the
comparison columns. RABDAM runs the first two without the third when it
scores PDB-Redo's model. The map source engine downloads the reflection file
for the density engine, which runs the other three as well.

PDB-Redo re-refines X-ray entries only, and not all of them. So all four
engines run only for X-ray diffraction. An entry PDB-Redo has not re-refined
gets a warning and empty PDB-Redo columns, and the entry is not failed.
"""

from .artifacts import PDB_REDO_CIF_PATH, PDB_REDO_DATAFRAME, PDB_REDO_MTZ_PATH

__all__ = ["PDB_REDO_CIF_PATH", "PDB_REDO_DATAFRAME", "PDB_REDO_MTZ_PATH"]
