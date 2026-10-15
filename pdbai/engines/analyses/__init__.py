"""The engines that compute new values from an entry's structure.

rabdam scores radiation damage with rabdam-custom. It gives BDamage and
packing density per atom, and Bnet and its percentile per structure. It
scores the RCSB model, PDB-Redo's re-refined model, or both.

sasa computes solvent accessibility with its own Shrake-Rupley calculation,
for the protein alone and with what is bound to it.

disulfides writes a table of candidate disulfides, cysteine pairs found from
the deposited _struct_conn records and from SG–SG distances. Each row gives
the pair's S–S distances, SG occupancy, alternate conformers, and SG/CB
B-factor ratios, and says whether a metal atom bridges the pair. The atom
table links each representative SG to its rows.

density reads an entry's 2mFo−DFc and mFo−DFc maps at every atom, in units of
each map's root-mean-square density. The maps are PDB-Redo's, read at the
deposited position and at the position PDB-Redo's model gives the atom, or
the wwPDB validation maps of the deposited model, or both. From PDB-Redo's
maps it also gives the model's calculated density and the difference density
at each atom on one scale.

metal_sites writes three tables, one row per metal atom, one per bond to a
donor atom, and one per pair of metal atoms that lie near each other. Donors
come from the deposited _struct_conn records and from distances within the
first coordination sphere, symmetry mates included, and each bond is set
beside a literature distance.

residue_sites writes a table of the amino-acid groups that radiation damage
removes, the carboxylates of Asp, Glu, and the C-terminus, the sulfurs of Met
and Cys, and the hydroxyl of Tyr, beside control groups of the same shape, with the
atoms modelled, their occupancy, and their B-factors. It also writes every
atom's B-factor over that of the atom it hangs from.

selenium_halogens writes a table of selenium and halogen atoms, each
beside the atoms it is bonded to and the rest of its component.

cofactors writes a table of flavins, tetrapyrroles, fluorescent-protein
chromophores, retinal, and p-coumaric acid, with the geometry of each one's
reactive group. site_geometry.py holds what these three share.

None of the eight fetches anything. They read the structure a source engine
parsed, or an uploaded file. PDB-Redo's model and reflection file reach
RABDAM and the density engine through the PDB-Redo engines, and the wwPDB
validation maps reach the density engine through their own source engine. The disulfide
engine also reads the structure file again, since the parsed atoms keep one
conformer of each residue and it needs every one. The metal-site engine reads
it again for the unit cell, the space group, and the deposited records. The
residue-site, selenium and halogen, and cofactor engines read the atom table
alone.
"""

__all__: list[str] = []
