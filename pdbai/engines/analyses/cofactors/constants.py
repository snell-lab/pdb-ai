"""The cofactor classes, their measures, and the vocabularies of the table.

A class is recognized by atom names, the ones the wwPDB chemical component
dictionary gives the atoms of its reactive group, so that an analogue or an
adduct deposited under a component name of its own is read like the common
one. Two classes are limited to one component, where the names alone would
not tell the cofactor from other molecules. The literature behind each
measure is given so that a reader can judge it, and each source should be
checked before it is cited.
"""

from __future__ import annotations

from dataclasses import dataclass

# The radius in Å within which the protein atom a cofactor is linked to is
# looked for. A Schiff base C=N is about 1.3 Å and a thioester C–S about
# 1.8 Å, so the radius also finds a link modelled as broken. The distance is
# reported as measured and no bond is asserted.
LINK_SEARCH_RADIUS_A = 4.0

# One atom of a group, by every name it goes by, the preferred name first.
AtomNames = tuple[str, ...]


@dataclass(frozen=True)
class CofactorClass:
    """One kind of cofactor, the atoms of its reactive group, and its measures."""

    name: str
    # Atom names a component must hold to be of the class.
    signature: frozenset[str]
    # The atom that names the row, one of the signature's.
    key_atom: str
    # The atoms of the reactive group.
    group: tuple[AtomNames, ...]
    # Component names the class is limited to, or None for any component.
    components: frozenset[str] | None = None
    # Whether the component must have a place in a polymer sequence.
    polymer_only: bool = False
    # Each torsion column of the class, with the four atoms that define it.
    torsions: tuple[tuple[str, tuple[AtomNames, ...]], ...] = ()
    # The two rings whose planes the butterfly angle is taken between.
    fold_planes: tuple[tuple[AtomNames, ...], tuple[AtomNames, ...]] | None = None
    # The cofactor atom, the residue, and the atom of that residue it links to.
    link: tuple[str, str, str] | None = None
    # The atoms that surround the central metal, for its out-of-plane distance.
    metal_plane: tuple[str, ...] | None = None


def _atoms(*names: str) -> tuple[AtomNames, ...]:
    return tuple((name,) for name in names)


FLAVIN = "flavin"
RETINAL = "retinal"
TETRAPYRROLE = "tetrapyrrole"
FP_CHROMOPHORE = "fp_chromophore"
COUMARIC_ACID = "coumaric_acid"

# FAD names two ring carbons C4X and C5X, since C4A and C5A are atoms of its
# adenine. FMN and riboflavin name the same two C4A and C5A.
_FLAVIN_C4A: AtomNames = ("C4X", "C4A")
_FLAVIN_C5A: AtomNames = ("C5X", "C5A")
_PYRROLE_NITROGENS = ("NA", "NB", "NC", "ND")

COFACTOR_CLASSES: tuple[CofactorClass, ...] = (
    # The isoalloxazine ring of FAD, FMN, and their analogues. Reduction, by
    # photoelectrons as by chemistry, bends the ring about the N5–N10 axis,
    # the butterfly bend (Røhr et al., 2010). The angle is taken between the
    # planes of the two outer rings, so a flat ring gives 0.
    CofactorClass(
        FLAVIN,
        signature=frozenset({"N5", "N10", "C9A", "C10", "C7M", "C8M"}),
        key_atom="N5",
        group=(
            *_atoms("N1", "C2", "N3", "C4"),
            _FLAVIN_C4A,
            ("N5",),
            _FLAVIN_C5A,
            *_atoms("C6", "C7", "C8", "C9", "C9A", "N10", "C10"),
        ),
        fold_planes=(
            (*_atoms("N1", "C2", "N3", "C4"), _FLAVIN_C4A, ("C10",)),
            (_FLAVIN_C5A, *_atoms("C6", "C7", "C8", "C9", "C9A")),
        ),
    ),
    # The macrocycle of hemes, chlorophylls, pheophytins, and their
    # relatives, with or without a central metal. A metal that moves out of
    # the plane of the four nitrogens marks a change of spin or oxidation
    # state in a heme (Perutz, 1970).
    CofactorClass(
        TETRAPYRROLE,
        signature=frozenset({*_PYRROLE_NITROGENS, "CHA", "CHB", "CHC", "CHD"}),
        key_atom="NA",
        group=_atoms(
            *_PYRROLE_NITROGENS,
            "CHA",
            "CHB",
            "CHC",
            "CHD",
            *(f"C{number}{ring}" for ring in "ABCD" for number in "1234"),
        ),
        metal_plane=_PYRROLE_NITROGENS,
    ),
    # The chromophore of green fluorescent protein and its relatives, a
    # residue formed from three of the chain. The methine bridge between its
    # two rings twists about tau, N2–CA2–CB2–CG2, and phi, CA2–CB2–CG2–CD1,
    # and X-rays take the bridge carbon out of the plane (Adam et al., 2009).
    CofactorClass(
        FP_CHROMOPHORE,
        signature=frozenset({"C1", "N2", "N3", "C2", "CA2", "CB2", "CG2"}),
        key_atom="CB2",
        group=_atoms("C1", "N2", "N3", "C2", "CA2", "CB2", "CG2"),
        polymer_only=True,
        torsions=(
            ("chromophore_tau_deg", _atoms("N2", "CA2", "CB2", "CG2")),
            (
                "chromophore_phi_deg",
                (*_atoms("CA2", "CB2", "CG2"), ("CD1", "CD2")),
            ),
        ),
    ),
    # Retinal, bound to a lysine through a Schiff base. Light isomerizes the
    # C13=C14 bond in microbial rhodopsins and the C11=C12 bond in visual
    # ones, and X-rays change the same region (Borshchevskiy et al., 2014).
    CofactorClass(
        RETINAL,
        signature=frozenset({"C11", "C12", "C13", "C14", "C15"}),
        key_atom="C15",
        group=_atoms(*(f"C{number}" for number in range(5, 16))),
        components=frozenset({"RET"}),
        torsions=(
            ("retinal_c11_c12_torsion_deg", _atoms("C10", "C11", "C12", "C13")),
            ("retinal_c13_c14_torsion_deg", _atoms("C12", "C13", "C14", "C15")),
        ),
        link=("C15", "LYS", "NZ"),
    ),
    # The p-coumaric acid of photoactive yellow protein, a thioester of a
    # cysteine. Light isomerizes its C2=C3 bond (Genick et al., 1998).
    CofactorClass(
        COUMARIC_ACID,
        signature=frozenset({"C1", "C2", "C3", "C1'"}),
        key_atom="C3",
        group=_atoms("C1", "C2", "C3", "C1'", "C2'", "C3'", "C4'", "C5'", "C6'"),
        components=frozenset({"HC4"}),
        torsions=(("coumaric_c2_c3_torsion_deg", _atoms("C1", "C2", "C3", "C1'")),),
        link=("C1", "CYS", "SG"),
    ),
)
COFACTOR_CLASS_NAMES: tuple[str, ...] = tuple(
    cofactor.name for cofactor in COFACTOR_CLASSES
)
TORSION_COLUMNS: tuple[str, ...] = tuple(
    column for cofactor in COFACTOR_CLASSES for column, _atoms_ in cofactor.torsions
)

# ok, or why a value of a row is missing. A row takes the first gap that
# applies, in the order listed after ok.
COFACTOR_STATUS_OK = "ok"
COFACTOR_STATUS_ATOMS_MISSING = "group_atoms_missing"
COFACTOR_STATUS_B_FACTOR_UNAVAILABLE = "b_factor_unavailable"
COFACTOR_STATUSES: tuple[str, ...] = (
    COFACTOR_STATUS_OK,
    COFACTOR_STATUS_ATOMS_MISSING,
    COFACTOR_STATUS_B_FACTOR_UNAVAILABLE,
)

# The structure-level outcomes the run report records.
COFACTORS_STATUS_OK = "ok"
COFACTORS_STATUS_NONE = "no_cofactors"
COFACTORS_STATUS_FAILED = "failed"

__all__ = [
    "COFACTORS_STATUS_FAILED",
    "COFACTORS_STATUS_NONE",
    "COFACTORS_STATUS_OK",
    "COFACTOR_CLASSES",
    "COFACTOR_CLASS_NAMES",
    "COFACTOR_STATUSES",
    "COFACTOR_STATUS_ATOMS_MISSING",
    "COFACTOR_STATUS_B_FACTOR_UNAVAILABLE",
    "COFACTOR_STATUS_OK",
    "COUMARIC_ACID",
    "FLAVIN",
    "FP_CHROMOPHORE",
    "LINK_SEARCH_RADIUS_A",
    "RETINAL",
    "TETRAPYRROLE",
    "TORSION_COLUMNS",
    "AtomNames",
    "CofactorClass",
]
