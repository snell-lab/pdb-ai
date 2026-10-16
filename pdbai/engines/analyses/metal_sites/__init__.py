"""Metal sites, their donor atoms, and metal–metal distances, in three tables.

constants.py holds the search radius, the first-sphere tolerance, the donor
rules, the metal elements, and the vocabularies of the tables, each beside
its basis.

reference.py loads the two bundled reference tables, the literature
metal–donor distances and the catalog of metal-containing components, and
gives the receipts a run records for them.

records.py reads the deposited _struct_conn metalc records.

compute.py finds each metal atom's donor atoms, among the atoms of the file
and of its symmetry mates, measures each bond, and summarizes each site.

columns.py declares the metal_sites, metal_ligand_bonds, and
metal_metal_distances tables, with a full description of each column, and
the atom column that links each metal atom to its site.

engine.py registers the engine with its option, tables, and report fields.
Unlike the other engine packages, this one does not import its engine, so
METAL_SITE_ENGINE is imported from engine.py. compute.py reads the unit cell
through site_geometry.py, which reads the metal elements from constants.py,
and an import of constants.py that ran engine.py would come back round to
site_geometry.py before it was defined.

The method is that of the coordination module of Alchemy
(https://github.com/snell-lab/alchemy, MIT), carried over without its
verdicts. Alchemy classifies a site as PASS, REVIEW, or SUSPECT and ranks it
against a reference cohort. This engine writes the distances, the counts, and
each bond's deviation from a cited reference distance, and leaves every
judgment to the analysis that reads the tables. It also reads the deposited
model, where Alchemy reads PDB-Redo's, so that neutron and X-ray entries are
measured by the same code.

A metal atom is any atom whose element is one of METAL_ELEMENTS, with
coordinates and an occupancy that is not zero. Its donor atoms are found in
two ways, and each bond row says which. A deposited _struct_conn metalc
record names a donor outright. The distance search infers one where a
nitrogen, oxygen, or sulfur atom of water or a standard amino acid, of the
kind that commonly binds metals, lies within the first coordination sphere,
the literature distance for the pair plus 0.75 Å (Harding, 2004). Hydrogen
and deuterium atoms are never donors, so a neutron model is read as an X-ray
model is.

The atom table keeps one conformer of each residue, and the engine reads that
table, so a site is measured in the conformer the parser kept.
"""
