"""Candidate disulfides between cysteine residues, in a table of their own.

constants.py holds the search limit and the metal contact distance with their
basis, the residue, atom, and connection names, and the vocabularies of the
detection method, the pair status, and the structure status.

columns.py declares the disulfides table, with a full description of each
measure, and the atom column that links each representative SG to it.

records.py reads the deposited _struct_conn disulf records that join two
cysteine SG atoms within one molecule. It counts those that join a symmetry
mate and skips those to other residues or atoms.

compute.py reads every cysteine SG and CB conformer and every metal atom from
the structure file, checks the cysteines against the atom table, finds pairs
from the records and by distance, and measures each pair.

engine.py registers the engine with its option, table, and report fields,
and hands each structure's file and atom table to compute.py.

The atom table keeps one conformer of each residue, so it cannot show a
disulfide that is intact in one conformer and broken in another. The engine
therefore reads the structure file again, every alternate conformer of every
model, and summarizes all of them in one row per pair of cysteines and model.
A pair is found from a deposited record, from the distance between SG
conformers that can coexist, or from both, and the row says which. Each
residue is named by its representative SG, the conformer the atom table
keeps, and that atom's row of the atom table holds the same identifier in
disulfide_site_id. An SG with two partners is in two rows and still links by
its own identifier.

A row is a candidate and does not assert a bond. Two cysteines bound to one
metal atom, as at a zinc site or a [NiFe] hydrogenase site, can be a row. The
metal_bridged column marks such rows without removing them, so an analysis of
disulfides can set them apart. A second row for one cysteine is rarely a
second bond. Such a cysteine is usually a ligand of a metal site, a residue
of an NMR ensemble, or one of two neighbouring disulfides modelled in
contact.
"""

from .engine import DISULFIDE_ENGINE

__all__ = ["DISULFIDE_ENGINE"]
