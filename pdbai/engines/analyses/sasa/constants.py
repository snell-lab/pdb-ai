"""Reference radii, areas, grid sizes, and component lists for SASA.

Every number the calculation takes from the literature is here beside its
source, along with the names of the schemes each result records. Element
symbols, component IDs, and atom names are in upper case, as
classification.py normalizes them. A change to a radius, a maximum area, or a
component list changes what results report.

Protein atoms take the DSSP radii rather than element radii. The Tien maxima
that normalize residue areas were computed with DSSP, and a relative area
means something only when its numerator and denominator use the same radii.
Every other occluder takes the van der Waals radius of its element from
Alvarez (2013), a table that covers nearly every element.
"""

from __future__ import annotations

from collections.abc import Mapping

# The probe radius in Å, that of a water molecule. DSSP uses it, so the Tien
# maxima assume it.
SASA_PROBE_RADIUS: float = 1.4
# Test points per atom. The fast count is the 92 that Shrake and Rupley used.
# The default and high counts are this engine's choice. The larger counts cut
# the sampling error, and the time taken grows in proportion to the count.
SASA_FAST_N_SPHERE_POINTS: int = 92
SASA_DEFAULT_N_SPHERE_POINTS: int = 1_000
SASA_HIGH_N_SPHERE_POINTS: int = 5_000

# The test points per atom of each accuracy mode a run may name.
SASA_ACCURACY_POINT_COUNTS: Mapping[str, int] = {
    "fast": SASA_FAST_N_SPHERE_POINTS,
    "default": SASA_DEFAULT_N_SPHERE_POINTS,
    "high": SASA_HIGH_N_SPHERE_POINTS,
}

# Every result records the names of its normalization and its radius schemes.
SASA_NORMALIZATION_NAME = "Tien2013_ALLOWED_DSSP"
SASA_PROTEIN_RADIUS_SCHEME = "DSSP"
SASA_NONPROTEIN_RADIUS_SCHEME = "Alvarez2013"


# A single-atom non-polymer residue of one of these elements is a metal ion.
# The list is this engine's choice. It leaves out the metalloids, such as
# boron, silicon, germanium, arsenic, antimony, and tellurium.
METAL_ELEMENTS: frozenset[str] = frozenset(
    {
        "LI",
        "NA",
        "K",
        "RB",
        "CS",
        "BE",
        "MG",
        "CA",
        "SR",
        "BA",
        "AL",
        "GA",
        "IN",
        "SN",
        "TL",
        "PB",
        "BI",
        "SC",
        "TI",
        "V",
        "CR",
        "MN",
        "FE",
        "CO",
        "NI",
        "CU",
        "ZN",
        "Y",
        "ZR",
        "NB",
        "MO",
        "TC",
        "RU",
        "RH",
        "PD",
        "AG",
        "CD",
        "HF",
        "TA",
        "W",
        "RE",
        "OS",
        "IR",
        "PT",
        "AU",
        "HG",
        "LA",
        "CE",
        "PR",
        "ND",
        "SM",
        "EU",
        "GD",
        "TB",
        "DY",
        "HO",
        "ER",
        "TM",
        "YB",
        "LU",
        "TH",
        "U",
        "NP",
        "PU",
        "AM",
    }
)


# Radii in Å by atom role, as mkdssp assigns them after Kabsch and Sander
# (1983), https://github.com/PDB-REDO/dssp.
DSSP_RADIUS_N = 1.65
DSSP_RADIUS_CA = 1.87
DSSP_RADIUS_C = 1.76
DSSP_RADIUS_O = 1.40
DSSP_RADIUS_SIDECHAIN = 1.80
# Keyed by atom name, so CA is the alpha carbon, not calcium.
DSSP_BACKBONE_RADII: Mapping[str, float] = {
    "N": DSSP_RADIUS_N,
    "CA": DSSP_RADIUS_CA,
    "C": DSSP_RADIUS_C,
    "O": DSSP_RADIUS_O,
}


# Van der Waals radii in Å from Alvarez (2013),
# https://doi.org/10.1039/C3DT50599E.
ALVAREZ_VDW_RADII: Mapping[str, float] = {
    "H": 1.20,
    "HE": 1.43,
    "LI": 2.12,
    "BE": 1.98,
    "B": 1.91,
    "C": 1.77,
    "N": 1.66,
    "O": 1.50,
    "F": 1.46,
    "NE": 1.58,
    "NA": 2.50,
    "MG": 2.51,
    "AL": 2.25,
    "SI": 2.19,
    "P": 1.90,
    "S": 1.89,
    "CL": 1.82,
    "AR": 1.83,
    "K": 2.73,
    "CA": 2.62,
    "SC": 2.58,
    "TI": 2.46,
    "V": 2.42,
    "CR": 2.45,
    "MN": 2.45,
    "FE": 2.44,
    "CO": 2.40,
    "NI": 2.40,
    "CU": 2.38,
    "ZN": 2.39,
    "GA": 2.32,
    "GE": 2.29,
    "AS": 1.88,
    "SE": 1.82,
    "BR": 1.86,
    "KR": 2.25,
    "RB": 3.21,
    "SR": 2.84,
    "Y": 2.75,
    "ZR": 2.52,
    "NB": 2.56,
    "MO": 2.45,
    "TC": 2.44,
    "RU": 2.46,
    "RH": 2.44,
    "PD": 2.15,
    "AG": 2.53,
    "CD": 2.49,
    "IN": 2.43,
    "SN": 2.42,
    "SB": 2.47,
    "TE": 1.99,
    "I": 2.04,
    "XE": 2.06,
    "CS": 3.48,
    "BA": 3.03,
    "LA": 2.98,
    "CE": 2.88,
    "PR": 2.92,
    "ND": 2.95,
    "SM": 2.90,
    "EU": 2.87,
    "GD": 2.83,
    "TB": 2.79,
    "DY": 2.87,
    "HO": 2.81,
    "ER": 2.83,
    "TM": 2.79,
    "YB": 2.80,
    "LU": 2.74,
    "HF": 2.63,
    "TA": 2.53,
    "W": 2.57,
    "RE": 2.49,
    "OS": 2.48,
    "IR": 2.41,
    "PT": 2.29,
    "AU": 2.32,
    "HG": 2.45,
    "TL": 2.47,
    "PB": 2.60,
    "BI": 2.54,
    "TH": 2.93,
    "U": 2.71,
    "NP": 2.82,
    "PU": 2.81,
    "AM": 2.83,
}


# Maximum residue areas in Å², the theoretical values of Tien et al. (2013),
# https://doi.org/10.1371/journal.pone.0080635. Each is the largest area DSSP
# gives the residue over its allowed backbone and side-chain conformations.
# The paper gives none for MSE, SEC, or CSE. Selenomethionine therefore takes
# the methionine maximum, and the selenocysteines take the cysteine one.
MAX_ASA_TIEN2013: Mapping[str, float] = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
    "MSE": 224.0,
    "SEC": 167.0,
    "CSE": 167.0,
}

# Burial cut-offs, as fractions of the residue's Tien maximum. A residue is
# buried when its relative area is below RASA_BURIED_BELOW, and exposed when
# it is at or above RASA_EXPOSED_FROM. A residue between the two is neither.
# Tien et al. give no cut-offs, so these are this engine's choice.
RASA_BURIED_BELOW = 0.25
RASA_EXPOSED_FROM = 0.40

# The heavy atoms of each residue inside a chain, by their atom names in the
# Chemical Component Dictionary. OXT is left out because only the last residue
# of a chain has it. A residue lacking one of these atoms has no complete area
# and no relative area. The Chemical Component Dictionary marks CSE obsolete
# and replaces it with SEC. Both name the selenium SE where cysteine has SG.
EXPECTED_HEAVY_ATOMS: Mapping[str, frozenset[str]] = {
    "ALA": frozenset({"N", "CA", "C", "O", "CB"}),
    "ARG": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"}),
    "ASN": frozenset({"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"}),
    "ASP": frozenset({"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"}),
    "CYS": frozenset({"N", "CA", "C", "O", "CB", "SG"}),
    "GLN": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"}),
    "GLU": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"}),
    "GLY": frozenset({"N", "CA", "C", "O"}),
    "HIS": frozenset({"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"}),
    "ILE": frozenset({"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"}),
    "LEU": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"}),
    "LYS": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"}),
    "MET": frozenset({"N", "CA", "C", "O", "CB", "CG", "SD", "CE"}),
    "PHE": frozenset(
        {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"}
    ),
    "PRO": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD"}),
    "SER": frozenset({"N", "CA", "C", "O", "CB", "OG"}),
    "THR": frozenset({"N", "CA", "C", "O", "CB", "OG1", "CG2"}),
    "TRP": frozenset(
        {
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "NE1",
            "CE2",
            "CE3",
            "CZ2",
            "CZ3",
            "CH2",
        }
    ),
    "TYR": frozenset(
        {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"}
    ),
    "VAL": frozenset({"N", "CA", "C", "O", "CB", "CG1", "CG2"}),
    "MSE": frozenset({"N", "CA", "C", "O", "CB", "CG", "SE", "CE"}),
    "SEC": frozenset({"N", "CA", "C", "O", "CB", "SE"}),
    "CSE": frozenset({"N", "CA", "C", "O", "CB", "SE"}),
}

# Water, which occludes in neither context.
WATER_COMPONENT_IDS = frozenset({"HOH", "DOD", "H2O", "WAT"})
# Common crystallization additives, cryoprotectants, buffers, and reducing
# agents, which occlude only when the user includes them. The list is short
# by choice. An additive it does not name is classified as a ligand.
EXPERIMENTAL_ADDITIVE_IDS = frozenset(
    {
        "ACT",
        "BME",
        "DMS",
        "DTT",
        "EDO",
        "EOH",
        "GOL",
        "IPA",
        "MPD",
        "PEG",
        "PGE",
        "PG4",
        "PG5",
        "PG6",
        "SO4",
        "MES",
        "TRS",
        "FMT",
    }
)
# UNK, UNL, and UNX stand for an unknown residue, ligand, and atom or ion. A
# non-polymer residue of one of these is never a ligand.
UNKNOWN_COMPONENT_IDS = frozenset({"UNK", "UNL", "UNX"})
# Cofactors classified as ligands by component ID alone, even on atoms that
# have no residue identity. Any other named non-polymer residue is classified
# as a ligand as well.
KNOWN_COFACTOR_IDS = frozenset(
    {
        "ADP",
        "AMP",
        "ATP",
        "COA",
        "FAD",
        "FMN",
        "HEM",
        "HEC",
        "NAD",
        "NAP",
        "NDP",
        "PLP",
        "SAM",
        "SAH",
        "TPP",
    }
)


__all__ = [
    "ALVAREZ_VDW_RADII",
    "DSSP_BACKBONE_RADII",
    "DSSP_RADIUS_C",
    "DSSP_RADIUS_CA",
    "DSSP_RADIUS_N",
    "DSSP_RADIUS_O",
    "DSSP_RADIUS_SIDECHAIN",
    "EXPECTED_HEAVY_ATOMS",
    "EXPERIMENTAL_ADDITIVE_IDS",
    "KNOWN_COFACTOR_IDS",
    "MAX_ASA_TIEN2013",
    "METAL_ELEMENTS",
    "RASA_BURIED_BELOW",
    "RASA_EXPOSED_FROM",
    "SASA_ACCURACY_POINT_COUNTS",
    "SASA_DEFAULT_N_SPHERE_POINTS",
    "SASA_FAST_N_SPHERE_POINTS",
    "SASA_HIGH_N_SPHERE_POINTS",
    "SASA_NONPROTEIN_RADIUS_SCHEME",
    "SASA_NORMALIZATION_NAME",
    "SASA_PROBE_RADIUS",
    "SASA_PROTEIN_RADIUS_SCHEME",
    "UNKNOWN_COMPONENT_IDS",
    "WATER_COMPONENT_IDS",
]
