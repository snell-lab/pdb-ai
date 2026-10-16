"""Atom rows read from mmCIF files, and the identity that joins and pairs them.

residue_identity.py decides which atoms of an atom table form one residue and
one chain, so that the parser, the engines, the star schema's join keys, and
the PDB-Redo alignment all agree. Keys within one file put label fields first,
and keys across files use only author fields, which PDB-Redo keeps.

coordinate_preprocessing.py keeps one alternate conformer of each residue.
The choice is made for the whole residue by mean occupancy, so every
calculation sees one set of coordinates that coexisted.

mmcif_fields.py turns the flat mmCIF dictionary the parser reads into columns
of one value per atom. Only what describes the whole entry is repeated on
every atom. The module also computes the unit-cell volume.

mmcif_parser.py reads an mmCIF file into one row per atom. Each row keeps the
_atom_site text the file writes. Named columns add numbers, identity, and
classification. The entry's experimental values come from the file and from
its RCSB record. Its reader, read_mmcif_as_flat_dict, is exported too, for an
engine that needs every alternate conformer or a category of several rows.
Such an engine reads that reader's rows as the parser does through two more
exports. coerce_cif_numbers of cif_numbers.py reads their numbers.
atom_site_residue_identity of residue_identity.py gives their chain_id,
res_num, and res_name, from the author items or, for a file without them,
the label items, and atom_site_residue_numbering and atom_site_residue_names
are its two parts for an engine that selects rows before it names them.

refinement.py reads how an entry was refined from the same flat dictionary:
its refinement programs, its TLS groups, and the share of its atom sites
with an anisotropic record. The parser writes them as entry-level columns.

elements.py holds the element symbols of hydrogen and its isotopes, so that
every reader that leaves hydrogens out leaves out the same atoms.

alignment.py pairs each atom of an RCSB atom table with the same atom in its
PDB-Redo re-refinement, one to one or not at all. It rejects the whole
alignment when paired atoms lie a median of more than 1 Å apart.

Nothing in the package fills a gap by guess. A value the file does not report
stays missing, never zero or False. A residue or atom whose identity is
incomplete or ambiguous gets no key or partner. A value that is reported but
cannot be read raises ValueError naming it and listing its values. The
parser raises that again as RuntimeError naming the
entry, so that only that entry fails.
"""

from .cif_numbers import coerce_cif_numbers
from .mmcif_parser import parse_mmcif_atom_rows, read_mmcif_as_flat_dict
from .residue_identity import (
    RESIDUE_NUMBERING_COLUMN,
    RESIDUE_NUMBERING_LABEL,
    ResidueNumbering,
    atom_site_residue_identity,
    atom_site_residue_names,
    atom_site_residue_numbering,
)

__all__ = [
    "RESIDUE_NUMBERING_COLUMN",
    "RESIDUE_NUMBERING_LABEL",
    "ResidueNumbering",
    "atom_site_residue_identity",
    "atom_site_residue_names",
    "atom_site_residue_numbering",
    "coerce_cif_numbers",
    "parse_mmcif_atom_rows",
    "read_mmcif_as_flat_dict",
]
