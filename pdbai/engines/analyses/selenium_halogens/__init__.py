"""Selenium and halogen atoms, in a table of their own.

constants.py holds the elements, the bond tolerance, and the vocabularies of
the table, each beside its basis.

compute.py finds each such atom in the atom table, the atoms it is bonded
to, and the other atoms of its component, and measures the site.

columns.py declares the selenium_halogen_sites table, with a full
description of each column, and the atom column that links each such atom to
its row.

engine.py registers the engine with its option, table, and report fields.

Selenium, bromine, and iodine absorb X-rays strongly, and their bonds to
carbon are among the first that radiation breaks: Se–C in selenomethionine,
and C–Br and C–I in halogenated ligands and nucleotides. Chlorine and
fluorine are read with them, although neither is heavy, so that the four
halogens form one series in which the bond to carbon weakens from fluorine
to iodine. Halide ions have rows too, without a bonded atom. Mercury,
platinum, gold, and the other heavy metals are rows of the metal_sites
table.

A row states what the model holds, the atom's occupancy and B-factor
beside those of the atoms it is bonded to and of the rest of its component.
What the data show at the atom is in the atom table, where the density
engine writes the map density of every atom, and the atom link column joins
the two.
"""

from .engine import SELENIUM_HALOGEN_ENGINE

__all__ = ["SELENIUM_HALOGEN_ENGINE"]
