"""The elements, the bond tolerance, and the vocabularies of the table.

Every threshold the engine uses is here beside its basis. The literature
values are given so that a reader can judge them, and each should be checked
against its source before it is cited.
"""

from __future__ import annotations

# The elements a row is written for: selenium and the four halogens.
# Selenium, bromine, and iodine absorb X-rays far more strongly than the
# light atoms of a protein, and their bonds to carbon are among the first to
# break: C–Br in brominated nucleotides (Ennifar et al., 2002), C–Br and C–I
# in halogenated ligands (Rodrigues et al., 2024), and Se–C in
# selenomethionine (Holton, 2007). Chlorine and fluorine are neither heavy
# nor strong absorbers. They are read so that the halogens form one series,
# since the bond to carbon weakens from fluorine to iodine, and the
# carbon–fluorine bond, the strongest, is the comparison the others are set
# against. Chlorine is also common enough to be found in every cohort.
# Mercury, platinum, gold, and the other heavy metals are rows of the
# metal_sites table instead.
SITE_ELEMENTS: tuple[str, ...] = ("SE", "F", "CL", "BR", "I")

# Two atoms are bonded when their distance is at most the sum of their
# covalent radii plus this tolerance in Å. The radii are gemmi's, those of
# Cordero et al. (2008), which put C–F at 1.30 Å, C–Cl at 1.75 Å, C–Se and
# C–Br at 1.93 Å, and C–I at 2.12 Å.
# The tolerance is the one Open Babel uses to perceive bonds from distances
# (O'Boyle et al., 2011). A bond refined longer than that is not found, and
# its row then says no_bonded_atom.
BOND_TOLERANCE_A = 0.45

# The component whose selenium replaces the sulfur of methionine.
SELENOMETHIONINE_RESIDUE_NAME = "MSE"

# The kind of component an atom of the table belongs to. An ion is a
# component whose only non-hydrogen atom is that atom, as a chloride of the buffer is, or
# the bromide or iodide of a soak.
PARENT_ION = "ion"
PARENT_SELENOMETHIONINE = "selenomethionine"
PARENT_POLYMER_RESIDUE = "polymer_residue"
PARENT_LIGAND = "ligand"
PARENT_TYPES: tuple[str, ...] = (
    PARENT_ION,
    PARENT_SELENOMETHIONINE,
    PARENT_POLYMER_RESIDUE,
    PARENT_LIGAND,
)

# ok, or why a value of a row is missing. A row takes the first gap that
# applies, in the order listed after ok.
SITE_STATUS_OK = "ok"
SITE_STATUS_NO_BONDED_ATOM = "no_bonded_atom"
SITE_STATUS_B_FACTOR_UNAVAILABLE = "b_factor_unavailable"
SITE_STATUSES: tuple[str, ...] = (
    SITE_STATUS_OK,
    SITE_STATUS_NO_BONDED_ATOM,
    SITE_STATUS_B_FACTOR_UNAVAILABLE,
)

# The structure-level outcomes the run report records.
SELENIUM_HALOGENS_STATUS_OK = "ok"
SELENIUM_HALOGENS_STATUS_NONE = "no_sites"
SELENIUM_HALOGENS_STATUS_FAILED = "failed"

__all__ = [
    "BOND_TOLERANCE_A",
    "SELENIUM_HALOGENS_STATUS_FAILED",
    "SELENIUM_HALOGENS_STATUS_NONE",
    "SELENIUM_HALOGENS_STATUS_OK",
    "SITE_ELEMENTS",
    "PARENT_ION",
    "PARENT_LIGAND",
    "PARENT_POLYMER_RESIDUE",
    "PARENT_SELENOMETHIONINE",
    "PARENT_TYPES",
    "SELENOMETHIONINE_RESIDUE_NAME",
    "SITE_STATUSES",
    "SITE_STATUS_B_FACTOR_UNAVAILABLE",
    "SITE_STATUS_NO_BONDED_ATOM",
    "SITE_STATUS_OK",
]
