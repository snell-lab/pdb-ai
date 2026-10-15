"""The surroundings of each radiation-sensitive group, in a table of their own.

constants.py holds the distances, the atom sets, and the vocabularies of the
table, with what each distance rests on.

compute.py sorts the atoms of each model into what a group's measures look
for, builds the symmetry mates of an entry with a unit cell, and measures
each group of the residue_sites table.

columns.py declares the site_environment table, with a full description of
each column, and the atom column of depths.

engine.py registers the engine with its option, table, and report fields.

Two groups of one residue type in one protein can lose density at different
rates. What sets them apart is what surrounds them: what they are hydrogen
bonded to, whether a disulfide, a metal, or an aromatic ring lies near, how
deeply they are buried, and whether a neighbouring molecule of the crystal
touches them. A row records those for one group, keyed by the site_id of its
residue_sites row, so the two tables join one to one, and the density of
the group's atoms is one more join away in the atom table.

Everything but the crystal contacts and the waters is taken from the
protein atoms alone, so it can be taken from a predicted model as well as a
deposited one. The crystal contacts need the unit cell and the space group,
and a row says when there are none.
"""

from .engine import SITE_ENVIRONMENT_ENGINE

__all__ = ["SITE_ENVIRONMENT_ENGINE"]
