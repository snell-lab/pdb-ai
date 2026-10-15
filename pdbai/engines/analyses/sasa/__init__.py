"""Solvent-accessible surface area of protein atoms and residues.

constants.py holds the probe radius and sphere-grid sizes, the DSSP protein
radii, the Alvarez radii of other occluders, the Tien maximum areas and the
burial cut-offs, and the component lists that sort non-protein atoms.

columns.py names the output columns, says which are Boolean and which
numeric, and describes each one for the dataset schema.

sphere.py spreads test points over a sphere and runs the Shrake-Rupley
calculation on an array of coordinates and radii.

classification.py sorts atoms into component categories, gives protein atoms
their radii, finds the residues at chain ends and breaks, and turns a residue
area into a relative one.

compute.py runs the calculation on the atom table of one structure. It
computes each model in a protein context and a bound context. It then sums
atoms into residues, decides which residues may be normalized, and sets the
status and warnings.

engine.py registers the engine with its options, columns, and report fields,
and hands each structure to compute.py.

The areas are those of the coordinates as given, model by model, with no
symmetry or assembly transform applied. Every row of a structure records the
probe radius, sphere grid, radius schemes, normalization, and selection
policies that produced its areas.
"""

from .engine import SASA_ENGINE

__all__ = ["SASA_ENGINE"]
