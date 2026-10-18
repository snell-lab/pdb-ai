"""The engines that compute new values from an entry's structure.

rabdam scores radiation damage with rabdam-custom. It gives BDamage and
packing density per atom, and Bnet and its percentile per structure. It
scores the RCSB model, PDB-Redo's re-refined model, or both.

sasa computes solvent accessibility with its own Shrake-Rupley calculation,
for the protein alone and with what is bound to it.

density reads an entry's 2mFo−DFc and mFo−DFc maps at every atom, in units of
each map's root-mean-square density. The maps are PDB-Redo's, read at the
deposited position and at the position PDB-Redo's model gives the atom, or
the wwPDB validation maps of the deposited model, or both. From PDB-Redo's
maps it also gives the model's calculated density and the difference density
at each atom on one scale.

None of the three fetches anything. They read the structure a source engine
parsed, or an uploaded file. PDB-Redo's model and reflection file reach
RABDAM and the density engine through the PDB-Redo engines, and the wwPDB
validation maps reach the density engine through their own source engine.
"""

__all__: list[str] = []
