"""Electron density at every atom, read from an entry's map coefficients.

constants.py holds the map coefficient labels, the sampling of the maps, the
limit on their size, and the vocabularies of the two statuses.

compute.py turns a set of map coefficients into the 2mFo−DFc and mFo−DFc maps
of the entry, one at a time, and reads each at the positions of the atoms.
From PDB-Redo's file it also makes the map of the model's calculated
structure factors and reads it at the same positions.

columns.py declares the atom columns and the structure columns of each map
source, with a full description of each.

engine.py registers the engine with its options and report fields, and hands
each structure's map files and atom table to compute.py.

The engine records what the data show at an atom and makes no judgment about
it. It reads maps from two sources, and density_map_source says which. The
first is PDB-Redo, whose maps describe its re-refined model. Each is read at
the atom's deposited position and at the position PDB-Redo's model gives the
same atom. The second is the wwPDB validation pipeline, whose maps describe
the deposited model and are read at the deposited position. Values are in
units of each map's root-mean-square density, and the root-mean-square
densities are structure columns, so a value can be put back on the scale of
the map coefficients.

A value in sigma grows with the number of electrons of the atom, so it does
not compare one element with another. For PDB-Redo's maps the engine
therefore also writes the calculated density at each atom and the difference
density beside it, both on the scale of the map coefficients. The second
over the first is the share of the model's density that the data do not
support. The engine writes the two measurements and not their ratio, which
is unstable where the calculated density is small. The wwPDB validation
files hold no calculated structure factors, so their columns have neither.

The value at an atom cannot show an atom that restraints hold short of where
the data put it, since such an atom sits between a negative and a positive
lobe of the difference map. So each source also gives the gradient of its
difference map at the atoms of the model it was calculated about, in sigma
per Å, and PDB-Redo's maps the shift of the atom that would level it, which
divides the gradient by the curvature of the calculated map.

Both sources hold maps of X-ray entries only, so the engine runs for X-ray
diffraction alone. An entry a source has no maps for gets empty values and a
status that says so, and the entry is not failed.
"""

from .engine import DENSITY_ENGINE

__all__ = ["DENSITY_ENGINE"]
