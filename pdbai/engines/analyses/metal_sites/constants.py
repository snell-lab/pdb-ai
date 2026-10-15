"""The radii, donor rules, elements, and vocabularies of the metal-site tables.

Every threshold the engine uses is here beside its basis. The search radius
and the first-sphere tolerance are written into the table descriptions and
the run summary, so every run records the ones it used. The rules are those
of Alchemy's coordination module, and each should be checked against its
source before it is cited.
"""

from __future__ import annotations

# The radius in Å within which donor atoms are looked for around a metal.
# It is a search radius only. A bond is inferred within the first sphere,
# which is always smaller.
CANDIDATE_SEARCH_RADIUS_A = 4.0

# A donor belongs to the first coordination sphere when its distance is at
# most the target distance for the pair plus this tolerance in Å, the
# coordination-group definition of Harding (2004, Acta Cryst. D60, 849-859).
FIRST_SPHERE_TOLERANCE_A = 0.75

# Two images of one atom closer than this in Å are one atom on a special
# position, the cut-off gemmi's contact search uses for the same purpose.
SPECIAL_POSITION_CUTOFF_A = 0.8

# The radius in Å within which other metal atoms of the file are counted
# around a metal. It gives context and asserts no metal–metal bond.
NEARBY_METAL_RADIUS_A = 6.0

# A pad for comparing distances against a radius, far below any resolvable
# displacement, so an atom exactly on a radius is inside it.
DISTANCE_EPSILON_A = 1e-6

# The symmetry operator that leaves an atom where the file puts it.
IDENTITY_SYMMETRY_OPERATOR = "1_555"

# Elements counted as metals, the list of Alchemy 1.0.0.
METAL_ELEMENTS: frozenset[str] = frozenset(
    {
        "NA",
        "MG",
        "K",
        "CA",
        "MN",
        "FE",
        "CO",
        "NI",
        "CU",
        "ZN",
        "CD",
        "HG",
        "PT",
        "MO",
        "AL",
        "BE",
        "BA",
        "RU",
        "V",
        "SR",
        "CS",
        "W",
        "AU",
        "IR",
        "YB",
        "LI",
        "GD",
        "PB",
        "TL",
        "U",
        "Y",
        "LR",
        "TI",
        "RB",
        "AG",
        "SM",
        "OS",
        "PR",
        "PD",
        "EU",
        "TB",
        "RE",
        "RH",
        "HF",
        "TA",
        "LU",
        "HO",
        "CR",
        "GA",
        "LA",
        "SN",
        "SB",
        "CE",
        "ZR",
        "ER",
        "TH",
        "IN",
        "SC",
        "DY",
        "BI",
        "PA",
        "PU",
        "AM",
        "CM",
        "CF",
        "GE",
        "NB",
        "TC",
        "ND",
        "PM",
        "TM",
        "PO",
        "FR",
        "RA",
        "AC",
        "NP",
        "BK",
        "ES",
        "FM",
        "MD",
        "NO",
        "RF",
        "DB",
        "SG",
    }
)

# The elements a donor atom may have. Hydrogen and deuterium are absent, so
# they are never donors, in a neutron model or any other.
DONOR_ELEMENTS: frozenset[str] = frozenset({"N", "O", "S"})

HYDROGEN_ELEMENTS: frozenset[str] = frozenset({"H", "D", "T"})

# The atoms of each standard amino acid that the distance search may infer as
# donors. The backbone carbonyl oxygen is allowed for every residue. Amide
# nitrogens (ASN ND2, GLN NE2), the TRP pyrrole nitrogen, the ARG guanidinium
# nitrogens, and the peptide nitrogen are left out, since they rarely bind a
# metal. Such an atom becomes a bond only when a deposited record names it.
INFERRED_DONOR_ATOMS: dict[str, frozenset[str]] = {
    "ALA": frozenset({"O"}),
    "ARG": frozenset({"O"}),
    "ASN": frozenset({"O", "OD1"}),
    "ASP": frozenset({"O", "OD1", "OD2"}),
    "CYS": frozenset({"O", "SG"}),
    "GLN": frozenset({"O", "OE1"}),
    "GLU": frozenset({"O", "OE1", "OE2"}),
    "GLY": frozenset({"O"}),
    "HIS": frozenset({"O", "ND1", "NE2"}),
    "ILE": frozenset({"O"}),
    "LEU": frozenset({"O"}),
    "LYS": frozenset({"O", "NZ"}),
    "MET": frozenset({"O", "SD"}),
    "PHE": frozenset({"O"}),
    "PRO": frozenset({"O"}),
    "SER": frozenset({"O", "OG"}),
    "THR": frozenset({"O", "OG1"}),
    "TRP": frozenset({"O"}),
    "TYR": frozenset({"O", "OH"}),
    "VAL": frozenset({"O"}),
}
STANDARD_AMINO_ACIDS: frozenset[str] = frozenset(INFERRED_DONOR_ATOMS)

# Terminal atoms the distance search may infer as donors. The backbone
# nitrogen counts only in the first residue of its polymer sequence, and a
# terminal oxygen wherever the file models one.
N_TERMINAL_DONOR_ATOM = "N"
C_TERMINAL_DONOR_ATOMS: frozenset[str] = frozenset({"OXT", "OT1", "OT2"})
BACKBONE_CARBONYL_ATOM = "O"

# The pseudo residues the reference table and its keys use. The table keys
# the backbone carbonyl oxygen as residue CA, which is not calcium there.
REFERENCE_WATER = "HOH"
REFERENCE_BACKBONE = "CA"
REFERENCE_N_TERMINUS = "NTERM"
REFERENCE_C_TERMINUS = "CTERM"

# The _struct_conn.conn_type_id of a deposited metal coordination bond.
METAL_CONNECTION_TYPE = "metalc"

# How a bond was found. The names are those the disulfides table uses.
DETECTED_BY_RECORD = "struct_conn"
DETECTED_BY_DISTANCE = "distance"
DETECTED_BY_RECORD_AND_DISTANCE = "struct_conn_and_distance"
DETECTION_METHODS = (
    DETECTED_BY_RECORD,
    DETECTED_BY_DISTANCE,
    DETECTED_BY_RECORD_AND_DISTANCE,
)

# The kind of component a metal atom belongs to. A cluster and a heme are
# components the catalog lists as such, an ion is a component of one atom,
# and other is any other component that holds a metal.
PARENT_ION = "ion"
PARENT_HEME = "heme"
PARENT_CLUSTER = "cluster"
PARENT_OTHER = "other"
PARENT_TYPES = (PARENT_ION, PARENT_HEME, PARENT_CLUSTER, PARENT_OTHER)

# The kind of residue a donor atom belongs to.
DONOR_WATER = "water"
DONOR_AMINO_ACID = "amino_acid"
DONOR_NUCLEOTIDE = "nucleotide"
DONOR_OTHER = "other"
DONOR_CLASSES = (DONOR_WATER, DONOR_AMINO_ACID, DONOR_NUCLEOTIDE, DONOR_OTHER)

# Which donor rule an atom falls under. Only the last forbids inferring a
# bond from distance.
RULE_WATER_OXYGEN = "water_oxygen"
RULE_BACKBONE_CARBONYL = "backbone_carbonyl_oxygen"
RULE_SIDECHAIN = "typical_sidechain_donor"
RULE_N_TERMINUS = "n_terminal_nitrogen"
RULE_C_TERMINUS = "c_terminal_oxygen"
RULE_NOT_TYPICAL = "outside_typical_donor_list"
DONOR_RULES = (
    RULE_WATER_OXYGEN,
    RULE_BACKBONE_CARBONYL,
    RULE_SIDECHAIN,
    RULE_N_TERMINUS,
    RULE_C_TERMINUS,
    RULE_NOT_TYPICAL,
)

# Where a bond's target distance came from. exact is the table's own row for
# the metal and donor. element_fallback is the longest row for the same metal
# and donor element, which decides first-sphere membership only. missing
# means the table has no row for the metal and donor element at all.
REFERENCE_EXACT = "exact"
REFERENCE_ELEMENT_FALLBACK = "element_fallback"
REFERENCE_MISSING = "missing"
REFERENCE_KINDS = (REFERENCE_EXACT, REFERENCE_ELEMENT_FALLBACK, REFERENCE_MISSING)

# ok, or why a derived value of a bond row is missing. no_reference comes
# first, since without a reference distance there is no deviation and no z.
BOND_STATUS_OK = "ok"
BOND_STATUS_NO_REFERENCE = "no_reference"
BOND_STATUS_NO_DPI = "no_dpi"
BOND_STATUSES = (BOND_STATUS_OK, BOND_STATUS_NO_REFERENCE, BOND_STATUS_NO_DPI)

# ok, or why a site row has no bond measures.
SITE_STATUS_OK = "ok"
SITE_STATUS_NO_DONORS = "no_donors_found"
SITE_STATUSES = (SITE_STATUS_OK, SITE_STATUS_NO_DONORS)

# The structure-level outcomes the run report records.
METAL_STATUS_OK = "ok"
METAL_STATUS_NO_METALS = "no_metals"
METAL_STATUS_DEGRADED = "degraded"
METAL_STATUS_FAILED = "failed"

__all__ = [
    "BACKBONE_CARBONYL_ATOM",
    "BOND_STATUSES",
    "BOND_STATUS_NO_DPI",
    "BOND_STATUS_NO_REFERENCE",
    "BOND_STATUS_OK",
    "CANDIDATE_SEARCH_RADIUS_A",
    "C_TERMINAL_DONOR_ATOMS",
    "DETECTED_BY_DISTANCE",
    "DETECTED_BY_RECORD",
    "DETECTED_BY_RECORD_AND_DISTANCE",
    "DETECTION_METHODS",
    "DISTANCE_EPSILON_A",
    "DONOR_AMINO_ACID",
    "DONOR_CLASSES",
    "DONOR_ELEMENTS",
    "DONOR_NUCLEOTIDE",
    "DONOR_OTHER",
    "DONOR_RULES",
    "DONOR_WATER",
    "FIRST_SPHERE_TOLERANCE_A",
    "HYDROGEN_ELEMENTS",
    "IDENTITY_SYMMETRY_OPERATOR",
    "INFERRED_DONOR_ATOMS",
    "METAL_CONNECTION_TYPE",
    "METAL_ELEMENTS",
    "METAL_STATUS_DEGRADED",
    "METAL_STATUS_FAILED",
    "METAL_STATUS_NO_METALS",
    "METAL_STATUS_OK",
    "NEARBY_METAL_RADIUS_A",
    "N_TERMINAL_DONOR_ATOM",
    "PARENT_CLUSTER",
    "PARENT_HEME",
    "PARENT_ION",
    "PARENT_OTHER",
    "PARENT_TYPES",
    "REFERENCE_BACKBONE",
    "REFERENCE_C_TERMINUS",
    "REFERENCE_ELEMENT_FALLBACK",
    "REFERENCE_EXACT",
    "REFERENCE_KINDS",
    "REFERENCE_MISSING",
    "REFERENCE_N_TERMINUS",
    "REFERENCE_WATER",
    "RULE_BACKBONE_CARBONYL",
    "RULE_C_TERMINUS",
    "RULE_NOT_TYPICAL",
    "RULE_N_TERMINUS",
    "RULE_SIDECHAIN",
    "RULE_WATER_OXYGEN",
    "SITE_STATUSES",
    "SITE_STATUS_NO_DONORS",
    "SITE_STATUS_OK",
    "SPECIAL_POSITION_CUTOFF_A",
    "STANDARD_AMINO_ACIDS",
]
