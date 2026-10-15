"""Cofactors and chromophores, in a table of their own.

constants.py holds the cofactor classes, each with the atom names that
recognize it, the atoms of its reactive group, and its measures, beside the
literature behind them, and the vocabularies of the table.

compute.py recognizes each cofactor in the atom table and measures its
reactive group.

columns.py declares the cofactor_sites table, with a full description of
each column, and the atom column that links each group atom to its row.

engine.py registers the engine with its option, table, and report fields.

A cofactor absorbs the electrons that X-rays free in a crystal, so it is
reduced, and a chromophore can be changed as light would change it. The
engine reads the geometry where that shows: the bend of a flavin's
isoalloxazine ring, the distance of a heme or chlorophyll metal from the
plane of its four nitrogens, the twist of a fluorescent-protein chromophore's
bridge, and the torsions about the bonds that light isomerizes in retinal and
in the p-coumaric acid of photoactive yellow protein.

The bonds from a heme or chlorophyll metal to the protein are in the
metal_sites table, and a row here names that metal atom by its site_id.
Bilins, quinones, and nicotinamides are not read.
"""

from .engine import COFACTOR_ENGINE

__all__ = ["COFACTOR_ENGINE"]
