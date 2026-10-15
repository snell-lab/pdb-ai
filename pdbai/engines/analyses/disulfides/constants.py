"""The cut-offs, names, and vocabularies of the disulfide table.

Every distance the engine uses is here beside its basis. The literature values
are given so that a reader can judge the cut-offs, and each should be checked
against its source before it is cited. Both distances are written into the
table's descriptions, and the search limit into the run summary, so every run
records the ones it used. A change to the search limit changes which pairs the
table holds, and a change to the metal contact distance changes which rows are
flagged.

Residue, atom, and connection names are compared as the file writes them
after surrounding whitespace is removed. Only residue names and connection
types are compared without regard to case.
"""

from __future__ import annotations

# The largest SG–SG distance in Å at which the distance search pairs two
# cysteines. A pair that a deposited record joins is a row at any distance.
# Restraint dictionaries put the S–S target near 2.03 Å (Engh and Huber,
# 2001). An intact bond is 2.02 Å and the disulfide radical anion about 2.8 Å
# (computed 2.78–2.79 Å; Bhattacharyya et al., 2020). Electron capture and
# cleavage are described by Weik et al. (2000), Carpentier et al. (2010), and
# Sutton et al. (2013). Two sulfur atoms that merely touch lie at the van der
# Waals contact of 3.6 Å (Bondi, 1964) to 3.8 Å (Alvarez, 2013), and two
# cysteines on one well-determined tetrahedral zinc lie about 3.8 Å apart,
# since Zn–S is 2.33 Å (Harding, 2006). Phenix uses the same limit, as its
# pdb_interpretation.disulfide_distance_cutoff defaults to 3 Å. Check each
# literature value against its source before citing it.
#
# The limit is a convention, not a boundary the data show, since deposited
# pairs lie on both sides of it. Few of either are partially broken bonds.
# Those at 2.3–3.0 Å are mostly metal sites, NMR and EM models, and annotated
# bonds refined long at 2–3.5 Å resolution. Those just beyond 3.0 Å are
# mostly metal sites and contacts between neighbouring disulfides, which
# begin near 3.15 Å, so a larger limit would add more pairs that are not
# bonds than pairs that are.
#
# The limit therefore leaves out the most separated conformers of a damaged
# bond unless a deposited record or a bonded conformer of the same pair brings
# the pair in.
DISULFIDE_SEARCH_CUTOFF_A = 3.0

# The largest distance in Å between a cysteine SG and a metal atom at which
# the sulfur counts as bound to the metal, for the metal_bridged flag. Bonds
# from a metal to a cysteine sulfur are about 2.1–2.4 Å, and Zn–S is about
# 2.33 Å (Harding, 2006), so 2.8 Å allows for poor geometry. The choice
# matters little, since few rows change between 2.6 and 3.0 Å.
METAL_CONTACT_DISTANCE_A = 2.8

# The residue and atoms the table is about. The SG atoms make the bond, and
# the CB atom is the reference for the SG's B-factor.
CYSTEINE_RESIDUE_NAME = "CYS"
SULFUR_ATOM_NAME = "SG"
BETA_CARBON_ATOM_NAME = "CB"

# The _struct_conn.conn_type_id of a deposited disulfide.
DISULFIDE_CONNECTION_TYPE = "disulf"

# The symmetry operator that leaves an atom where the file puts it. The
# _struct_conn record of a partner without an operator uses this one.
IDENTITY_SYMMETRY_OPERATOR = "1_555"

# The decimal places an SG occupancy sum is rounded to. Occupancies are
# deposited to two or three decimals, so a sum that differs from 1 beyond
# this differs only by floating-point error.
OCCUPANCY_SUM_DECIMALS = 6

# How a pair was found in one model. The names are those the planned metal-site
# table uses for the same two sources, so later site tables can share them.
DETECTED_BY_RECORD = "struct_conn"
DETECTED_BY_DISTANCE = "distance"
DETECTED_BY_RECORD_AND_DISTANCE = "struct_conn_and_distance"
DETECTION_METHODS: tuple[str, ...] = (
    DETECTED_BY_RECORD,
    DETECTED_BY_DISTANCE,
    DETECTED_BY_RECORD_AND_DISTANCE,
)

# Why a value of a row is missing, or ok. A row is ok, or takes the first gap
# that applies, in the order listed after ok, so the most fundamental gap is
# the one named.
PAIR_STATUS_OK = "ok"
PAIR_STATUS_NO_USABLE_SG = "no_usable_sg"
PAIR_STATUS_NO_COMPATIBLE_CONFORMERS = "no_compatible_conformers"
PAIR_STATUS_OCCUPANCY_MISSING = "occupancy_missing"
PAIR_STATUS_CB_MISSING = "cb_missing"
PAIR_STATUS_B_FACTOR_UNAVAILABLE = "b_factor_unavailable"
PAIR_STATUS_SG_NOT_IN_ATOM_TABLE = "sg_not_in_atom_table"
PAIR_STATUSES: tuple[str, ...] = (
    PAIR_STATUS_OK,
    PAIR_STATUS_NO_USABLE_SG,
    PAIR_STATUS_NO_COMPATIBLE_CONFORMERS,
    PAIR_STATUS_OCCUPANCY_MISSING,
    PAIR_STATUS_CB_MISSING,
    PAIR_STATUS_B_FACTOR_UNAVAILABLE,
    PAIR_STATUS_SG_NOT_IN_ATOM_TABLE,
)

# The structure-level outcomes the run report records.
DISULFIDE_STATUS_OK = "ok"
DISULFIDE_STATUS_DEGRADED = "degraded"
DISULFIDE_STATUS_FAILED = "failed"

__all__ = [
    "BETA_CARBON_ATOM_NAME",
    "CYSTEINE_RESIDUE_NAME",
    "DETECTED_BY_DISTANCE",
    "DETECTED_BY_RECORD",
    "DETECTED_BY_RECORD_AND_DISTANCE",
    "DETECTION_METHODS",
    "DISULFIDE_CONNECTION_TYPE",
    "DISULFIDE_SEARCH_CUTOFF_A",
    "DISULFIDE_STATUS_DEGRADED",
    "DISULFIDE_STATUS_FAILED",
    "DISULFIDE_STATUS_OK",
    "IDENTITY_SYMMETRY_OPERATOR",
    "METAL_CONTACT_DISTANCE_A",
    "OCCUPANCY_SUM_DECIMALS",
    "PAIR_STATUSES",
    "PAIR_STATUS_B_FACTOR_UNAVAILABLE",
    "PAIR_STATUS_CB_MISSING",
    "PAIR_STATUS_NO_COMPATIBLE_CONFORMERS",
    "PAIR_STATUS_OCCUPANCY_MISSING",
    "PAIR_STATUS_OK",
    "PAIR_STATUS_SG_NOT_IN_ATOM_TABLE",
    "PAIR_STATUS_NO_USABLE_SG",
]
