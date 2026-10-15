"""The radiation-sensitive groups of amino acids, in a table of their own.

constants.py holds the site classes, each with its residue, its atoms, and
the class it is a control for, beside the literature behind it, the metal
contact distance, and the vocabularies of the table.

compute.py finds the groups of each residue in the atom table and measures
each one.

parents.py holds the bonds of the 20 standard amino acids and takes every
atom's B-factor over that of the atom it hangs from.

columns.py declares the residue_sites table, with a full description of each
column, the atom column that links each site atom to its row, and the atom
column of B-factor ratios.

reference.py reads the bundled table of radiolysis rate constants, those of
each amino acid with the hydroxyl radical and the hydrated electron, and
gives the receipt each run records of it.

engine.py registers the engine with its option, table, and report fields.

A sensitive group is the set of atoms that specific radiation damage removes
or changes: the carboxyl carbon and oxygens of Asp, Glu, and the C-terminus,
which leave as carbon dioxide, the sulfurs of Met and Cys, the hydroxyl
oxygen of Tyr, and the rings of Trp and His, which the radicals of water
radiolysis attack. A control group has the same shape and is not removed in
the same way: the amides of Asn and Gln, and, as approximate controls, the
ring carbons of Phe, the delta carbons of Leu and Ile, and the hydroxyls of
Thr and Ser. The oxygens a residue gained, where it was deposited oxidized
at one of these groups, are a group as well.

A row states what the model holds, how many of the group's atoms, at what
occupancy, with what B-factors against the atom the group is attached to.
What the data show at the same atoms is in the atom table, where the density
engine writes the map density of every atom, and the atom link column joins
the two. Disulfides have their own engine and table, which measures a bond
over every conformer of its two cysteines. A cysteine's row here sets its
sulfur beside its CB in the one conformer the atom table keeps, bonded or
not.
"""

from .engine import RESIDUE_SITE_ENGINE

__all__ = ["RESIDUE_SITE_ENGINE"]
