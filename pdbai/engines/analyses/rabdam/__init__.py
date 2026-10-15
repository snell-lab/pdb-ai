"""RABDAM radiation-damage scores computed with rabdam-custom.

reference.py locates the bundled Bnet reference table and builds the
receipt that records it in the run manifest.

integration.py runs rabdam-custom in this process on one mmCIF file,
gzipped or not. It returns the per-atom BDamage and packing density, the
structure's Bnet, and the Bnet percentile against the reference table.

calculation.py puts one file's scores on the atom table. It joins on serial
numbers only when the scores came from the file the atoms were parsed from.
Otherwise it bridges by atom identity. It says why when no score reached
any atom, and it flags whether the PDB-Redo and RCSB scores agree.

engine.py is the registered engine. A setting chooses which files it
scores. Each column name says which file its values came from.

Every score comes from rabdam-custom at its upstream defaults, which are
the defaults the reference table was built with. integration.py explains
why no parameter can be set. engine.py defines when a source counts as
delivered and how that sets rabdam_status.
"""

__all__: list[str] = []
