"""The distances, atom sets, and vocabularies of the site_environment table.

Each distance is a choice of this engine, given with what it rests on so
that a reader can judge it. Each source should be checked before it is
cited.
"""

from __future__ import annotations

# The largest distance in Å between a nitrogen or oxygen of the group and a
# nitrogen or oxygen of a partner at which the two count as hydrogen bonded.
# Hydrogen bonds between N and O atoms in proteins lie at about 2.6–3.2 Å,
# and 3.5 Å is the usual outer limit of a search by distance alone (Baker &
# Hubbard, 1984).
HBOND_DISTANCE_A = 3.5
# The same limit for the sulfur of a cysteine or methionine, whose hydrogen
# bonds are longer, at up to about 4 Å (Zhou et al., 2009).
SULFUR_HBOND_DISTANCE_A = 4.0
# The largest distance in Å between a carboxylate oxygen and a cationic
# nitrogen at which the two form a salt bridge, the limit of Barlow and
# Thornton (1983) and of Kumar and Nussinov (2002).
SALT_BRIDGE_DISTANCE_A = 4.0

# The radii in Å of the spheres about the group's centroid in which heavy
# atoms are counted. Six Å takes in the first shell of neighbours, and 10 Å
# about what a residue's half-sphere exposure counts (Hamelryck, 2005).
PACKING_INNER_RADIUS_A = 6.0
PACKING_OUTER_RADIUS_A = 10.0
# The radius in Å about each group atom within which water oxygens and the
# atoms of symmetry mates are counted, the inner packing radius.
CONTACT_COUNT_RADIUS_A = PACKING_INNER_RADIUS_A

# Test points per atom of the solvent-accessible area. The SASA engine's
# fast count is 92 and its default 1,000. 200 keeps the sampling error of
# one atom near 1 Å² at a fifth of the default's cost, since every atom of
# the model is computed.
SPHERE_POINT_COUNT = 200
# The least solvent-accessible area in Å² at which an atom counts as at the
# surface, from which depth is measured. It is about two test points of a
# carbon.
EXPOSED_ATOM_AREA_A2 = 1.0

# How far in Å from the model the atoms of symmetry mates are looked for. A
# group farther than this from every symmetry mate has no nearest distance.
SYMMETRY_SEARCH_RADIUS_A = 15.0

# The components read as water.
WATER_RESIDUE_NAMES: frozenset[str] = frozenset({"HOH", "DOD", "H2O", "WAT"})

# The nitrogens that carry a positive charge at neutral pH, or may: the
# amine of Lys, the guanidinium of Arg, and the imidazole of His, whose
# charge depends on its pH and surroundings.
CATIONIC_NITROGENS: dict[str, frozenset[str]] = {
    "LYS": frozenset({"NZ"}),
    "ARG": frozenset({"NE", "NH1", "NH2"}),
    "HIS": frozenset({"ND1", "NE2"}),
}

# The atoms of each aromatic ring, whose centroid is the ring's position.
# Trp has two rings, and each is a ring of its own.
AROMATIC_RINGS: dict[str, tuple[tuple[str, ...], ...]] = {
    "PHE": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TYR": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TRP": (
        ("CG", "CD1", "NE1", "CE2", "CD2"),
        ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    ),
    "HIS": (("CG", "ND1", "CD2", "CE1", "NE2"),),
}
AROMATIC_RESIDUE_NAMES: tuple[str, ...] = tuple(AROMATIC_RINGS)

# The site classes whose group is a carboxylate, the only ones with salt
# bridges.
CARBOXYLATE_CLASSES: frozenset[str] = frozenset(
    {"asp_carboxylate", "glu_carboxylate", "c_terminal_carboxylate"}
)
# The site class of the cysteine sulfur, the only one in_disulfide applies to.
CYS_SULFUR_CLASS = "cys_sulfur"

# ok, or why a row's measures are missing.
ENVIRONMENT_STATUS_OK = "ok"
ENVIRONMENT_STATUS_NO_GROUP_ATOMS = "no_group_atoms"
ENVIRONMENT_STATUSES: tuple[str, ...] = (
    ENVIRONMENT_STATUS_OK,
    ENVIRONMENT_STATUS_NO_GROUP_ATOMS,
)

# Whether the crystal contacts could be read. no_unit_cell for a model
# without a usable cell and space group, such as a predicted model or an
# NMR ensemble. none_in_range when no symmetry mate comes within
# SYMMETRY_SEARCH_RADIUS_A of the group.
CONTACTS_STATUS_OK = "ok"
CONTACTS_STATUS_NONE_IN_RANGE = "none_in_range"
CONTACTS_STATUS_NO_UNIT_CELL = "no_unit_cell"
CONTACTS_STATUSES: tuple[str, ...] = (
    CONTACTS_STATUS_OK,
    CONTACTS_STATUS_NONE_IN_RANGE,
    CONTACTS_STATUS_NO_UNIT_CELL,
)

# The structure-level outcomes the run report records.
SITE_ENVIRONMENT_STATUS_OK = "ok"
SITE_ENVIRONMENT_STATUS_NO_SITES = "no_sites"
SITE_ENVIRONMENT_STATUS_FAILED = "failed"

__all__ = [
    "AROMATIC_RESIDUE_NAMES",
    "AROMATIC_RINGS",
    "CARBOXYLATE_CLASSES",
    "CATIONIC_NITROGENS",
    "CONTACTS_STATUSES",
    "CONTACTS_STATUS_NONE_IN_RANGE",
    "CONTACTS_STATUS_NO_UNIT_CELL",
    "CONTACTS_STATUS_OK",
    "CONTACT_COUNT_RADIUS_A",
    "CYS_SULFUR_CLASS",
    "ENVIRONMENT_STATUSES",
    "ENVIRONMENT_STATUS_NO_GROUP_ATOMS",
    "ENVIRONMENT_STATUS_OK",
    "EXPOSED_ATOM_AREA_A2",
    "HBOND_DISTANCE_A",
    "PACKING_INNER_RADIUS_A",
    "PACKING_OUTER_RADIUS_A",
    "SALT_BRIDGE_DISTANCE_A",
    "SITE_ENVIRONMENT_STATUS_FAILED",
    "SITE_ENVIRONMENT_STATUS_NO_SITES",
    "SITE_ENVIRONMENT_STATUS_OK",
    "SPHERE_POINT_COUNT",
    "SULFUR_HBOND_DISTANCE_A",
    "SYMMETRY_SEARCH_RADIUS_A",
    "WATER_RESIDUE_NAMES",
]
