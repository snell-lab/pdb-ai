"""The site classes, the metal contact distance, and the vocabularies of the table.

Each class names the atoms of one residue that specific radiation damage
removes, or the atoms of a residue of the same shape that it does not. The
literature behind each class is given so that a reader can judge the choice,
and each source should be checked before it is cited.
"""

from __future__ import annotations

from dataclasses import dataclass

# The largest distance in Å between a site atom and a metal atom at which the
# group counts as bound to the metal, the limit the disulfides table uses.
# Metal–oxygen bonds to a carboxylate are about 2.0–2.4 Å (Harding, 2006), so
# 2.8 Å allows for poor geometry.
METAL_CONTACT_DISTANCE_A = 2.8


@dataclass(frozen=True)
class ResidueSiteClass:
    """One kind of site: the atoms of a residue that leave, or their control."""

    name: str
    # The residue the class belongs to, or None for any standard amino acid.
    residue: str | None
    # The atoms of the group, those that damage removes in a sensitive class.
    site_atoms: tuple[str, ...]
    # The atoms the group is bonded to, which stay when it is lost.
    reference_atoms: tuple[str, ...]
    # The reference atom that names the row. A residue without it has no row.
    key_atom: str
    # Each bond that attaches the group, as a reference atom and a site atom.
    attachments: tuple[tuple[str, str], ...]
    # The sensitive class this one is the control for, or None.
    control_for: str | None = None
    # Atoms that must be modelled for the class to apply to a residue.
    requires: tuple[str, ...] = ()
    # Whether the residue is a modified one, deposited as HETATM records
    # within a polymer, where every other class is read from ATOM records.
    modified: bool = False


ASP_CARBOXYLATE = "asp_carboxylate"
GLU_CARBOXYLATE = "glu_carboxylate"
C_TERMINAL_CARBOXYLATE = "c_terminal_carboxylate"
MET_THIOETHER = "met_thioether"
TYR_HYDROXYL = "tyr_hydroxyl"
CYS_SULFUR = "cys_sulfur"
CYS_SULFENIC_ACID = "cys_sulfenic_acid"
CYS_SULFINIC_ACID = "cys_sulfinic_acid"
CYS_SULFONIC_ACID = "cys_sulfonic_acid"
TRP_INDOLE = "trp_indole"
HIS_IMIDAZOLE = "his_imidazole"
MET_SULFOXIDE = "met_sulfoxide"
MET_SULFONE = "met_sulfone"
TRP_HYDROXYLATED = "trp_hydroxylated"
TRP_HYDROPEROXIDE = "trp_hydroperoxide"
TRP_KYNURENINE = "trp_kynurenine"
HIS_OXO = "his_oxo"
TYR_DOPA = "tyr_dopa"
PHE_HYDROXYLATED = "phe_hydroxylated"

# Decarboxylation removes the carboxyl carbon with its two oxygens, so the
# group is those three atoms and the bond that breaks is the one to the atom
# before them (Burmeister, 2000; Weik et al., 2000). The amides of Asn and
# Gln have the same shape and do not decarboxylate. The methionine sulfur
# loses a bond to either carbon beside it, and the tyrosine hydroxyl is lost
# from the ring (Burmeister, 2000). No standard residue has the shape of
# either without the sensitive atom, so their controls are approximate: the
# ring carbon of Phe where Tyr carries its oxygen, and the delta carbons of
# Leu and Ile, which sit where the sulfur of Met does.
#
# A control of another element is a weak one, since the same error of the
# model leaves a deeper hole in the density at a heavier atom. So tyrosine
# has a second control of its own element, the hydroxyl of Thr, which is an
# oxygen on a carbon as its own is, though on a chain and not a ring. Whether
# tyrosine loses its hydroxyl at all is disputed (Bury et al., 2017), and
# whether a hydroxyl on a chain is untouched by X-rays is not established, so
# Thr and Ser are comparisons and not a zero.
#
# The sulfur of every cysteine is a class too, whether or not it is in a
# disulfide, which the disulfides table says. Its control is the hydroxyl of
# Ser, which is Cys with an oxygen in place of the sulfur, so that control is
# of the right shape and the wrong element.
#
# A cysteine oxidized at its sulfur is deposited as a residue of its own, CSO
# with one oxygen (a sulfenic acid), CSD with two (sulfinic), and OCS with
# three (sulfonic). The oxygens are the group and the sulfur is what they
# hang from. X-rays are reported both to reduce a sulfenic acid and to
# oxidize a cysteine, so these classes are read in either direction: how
# often a cohort models them, and how well the oxygens hold where it does.
# Check both reports against their sources before citing either.
#
# The classes above come from cryo crystallography, where X-rays damage a
# protein largely by direct ionization and by the electrons and holes that
# move afterwards. In water at room temperature most damage is done instead
# by the hydroxyl radical and the hydrated electron of radiolysis. The
# hydroxyl radical reacts fastest with Cys, Trp, Tyr, Met, Phe, and His, at
# 5 x 10^9 to 3 x 10^10 per molar per second, and the hydrated electron with
# Cys, then Tyr, Trp, and the imidazolium of His (Buxton et al., 1988, whose
# values are in radiolysis_rate_constants.csv). The indole of Trp and the
# imidazole of His are therefore groups too. Neither leaves: the hydroxyl radical adds to the ring,
# one-electron oxidation takes an electron from it, and the products keep
# the ring or open it. So the group is every ring atom beyond CG, the atom
# that joins the ring to the chain, since the products modify or remove
# atoms across the ring: an oxygen added at CD1 or at any carbon of the
# benzene ring of Trp, a hydroperoxide on NE1, the pyrrole opened to
# kynurenine, and an oxygen added at CE1 of His.
#
# Phe is not made a group. The hydroxyl radical adds to its ring a little
# faster than to that of His, but the hydrated electron reacts with it at
# half the rate of Trp and Tyr and a hundredth that of Cys, and cryo
# crystallography reports no damage to it. It stays the control it was, and
# so it is also the residue that tells the two chemistries apart: what
# hydroxyl radicals do should reach Phe as it reaches His, and what
# electrons do should reach neither much.
#
# The control for His is the four ring carbons of Phe beyond CG, an
# aromatic ring on the same atom, with carbon where His has its two
# nitrogens. CZ is the control for Tyr and a row of its own. Trp has no
# control. No standard residue has a fused aromatic ring, and the other
# aromatic rings are targets themselves. Its carbons can be set beside the
# ring carbons of Phe, and its nitrogen beside another nitrogen, from the
# atom table.
#
# A residue oxidized at one of these groups is deposited as a component of
# its own, and is read as the oxidized cysteines are: the oxygens added are
# the group, and the atom each hangs from is what they are attached to. A
# kind of product deposited under more than one code, as methionine
# sulfoxide is under SME and MHO, has one class name for all of them, and
# the row's key atom says where on the ring an oxygen was added. The codes
# and their atom names are those of the wwPDB Chemical Component Dictionary.
# Products made by enzymes, as 7-hydroxytryptophan and DOPA are on the way
# to protein cofactors, are deposited under the same codes as those made by
# radiation, so a row says that a model holds the product and not how it
# formed. N'-formylkynurenine is a component of the dictionary only as a
# free molecule, NFK, and is not read. 3-hydroxyhistidine, HSK, carries its
# oxygen on a ring nitrogen, which radiation chemistry does not report, and
# is not read either.
SITE_CLASSES: tuple[ResidueSiteClass, ...] = (
    ResidueSiteClass(
        ASP_CARBOXYLATE, "ASP", ("CG", "OD1", "OD2"), ("CB",), "CB", (("CB", "CG"),)
    ),
    ResidueSiteClass(
        GLU_CARBOXYLATE, "GLU", ("CD", "OE1", "OE2"), ("CG",), "CG", (("CG", "CD"),)
    ),
    # A C-terminus is seen where the file models the second oxygen. The last
    # modelled residue of a chain with a disordered end is not a terminus.
    ResidueSiteClass(
        C_TERMINAL_CARBOXYLATE,
        None,
        ("C", "O", "OXT"),
        ("CA",),
        "CA",
        (("CA", "C"),),
        requires=("OXT",),
    ),
    ResidueSiteClass(
        MET_THIOETHER, "MET", ("SD",), ("CG", "CE"), "CG", (("CG", "SD"), ("CE", "SD"))
    ),
    ResidueSiteClass(TYR_HYDROXYL, "TYR", ("OH",), ("CZ",), "CZ", (("CZ", "OH"),)),
    ResidueSiteClass(CYS_SULFUR, "CYS", ("SG",), ("CB",), "CB", (("CB", "SG"),)),
    ResidueSiteClass(
        TRP_INDOLE,
        "TRP",
        ("CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
        ("CG",),
        "CG",
        (("CG", "CD1"), ("CG", "CD2")),
    ),
    ResidueSiteClass(
        HIS_IMIDAZOLE,
        "HIS",
        ("ND1", "CD2", "CE1", "NE2"),
        ("CG",),
        "CG",
        (("CG", "ND1"), ("CG", "CD2")),
    ),
    ResidueSiteClass(
        CYS_SULFENIC_ACID,
        "CSO",
        ("OD",),
        ("SG",),
        "SG",
        (("SG", "OD"),),
        modified=True,
    ),
    ResidueSiteClass(
        CYS_SULFINIC_ACID,
        "CSD",
        ("OD1", "OD2"),
        ("SG",),
        "SG",
        (("SG", "OD1"), ("SG", "OD2")),
        modified=True,
    ),
    ResidueSiteClass(
        CYS_SULFONIC_ACID,
        "OCS",
        ("OD1", "OD2", "OD3"),
        ("SG",),
        "SG",
        (("SG", "OD1"), ("SG", "OD2"), ("SG", "OD3")),
        modified=True,
    ),
    ResidueSiteClass(
        MET_SULFOXIDE, "SME", ("OE",), ("S",), "S", (("S", "OE"),), modified=True
    ),
    ResidueSiteClass(
        MET_SULFOXIDE, "MHO", ("OD1",), ("SD",), "SD", (("SD", "OD1"),), modified=True
    ),
    ResidueSiteClass(
        MET_SULFONE,
        "OMT",
        ("OD1", "OD2"),
        ("SD",),
        "SD",
        (("SD", "OD1"), ("SD", "OD2")),
        modified=True,
    ),
    # 2-hydroxytryptophan, with its oxygen on CD1, and 4-, 5-, 6-, and
    # 7-hydroxytryptophan, with theirs on CE3, CZ3, CH2, and CZ2. 5- has two
    # codes. 6,7-dihydroxytryptophan and the dioxindolylalanines carry two.
    *(
        ResidueSiteClass(
            TRP_HYDROXYLATED,
            code,
            tuple(oxygen for _carbon, oxygen in bonds),
            tuple(carbon for carbon, _oxygen in bonds),
            bonds[0][0],
            bonds,
            modified=True,
        )
        for code, bonds in (
            ("TRO", (("CD1", "OD1"),)),
            ("4HT", (("CE3", "OE3"),)),
            ("HRP", (("CZ3", "OX"),)),
            ("4PQ", (("CZ3", "OAD"),)),
            ("TRX", (("CH2", "OH2"),)),
            ("0AF", (("CZ2", "O1"),)),
            ("TOQ", (("CH2", "O6"), ("CZ2", "O7"))),
            ("RE0", (("CD1", "OD1"), ("CG", "OG"))),
            ("RE3", (("CD1", "OD1"), ("CG", "OG"))),
        )
    ),
    ResidueSiteClass(
        TRP_HYDROPEROXIDE,
        "TOX",
        ("O1", "O2"),
        ("NE1",),
        "NE1",
        (("NE1", "O1"),),
        modified=True,
    ),
    # Kynurenine is named afresh in the dictionary. C1 is the carbon that
    # was CG of the tryptophan, and O2 the oxygen it gained when the pyrrole
    # opened. The carbon that was CD1 is gone.
    ResidueSiteClass(
        TRP_KYNURENINE, "KYN", ("O2",), ("C1",), "C1", (("C1", "O2"),), modified=True
    ),
    ResidueSiteClass(
        HIS_OXO, "OHI", ("O12",), ("CE1",), "CE1", (("CE1", "O12"),), modified=True
    ),
    ResidueSiteClass(
        HIS_OXO,
        "HOO",
        ("OAG", "OAM"),
        ("CE1", "CD2"),
        "CE1",
        (("CE1", "OAG"), ("CD2", "OAM")),
        modified=True,
    ),
    ResidueSiteClass(
        TYR_DOPA, "DAH", ("OE2",), ("CE2",), "CE2", (("CE2", "OE2"),), modified=True
    ),
    # meta-Tyrosine, which the dictionary gives Tyr as its parent, is Phe with
    # an oxygen on CE2.
    ResidueSiteClass(
        PHE_HYDROXYLATED,
        "MTY",
        ("OH",),
        ("CE2",),
        "CE2",
        (("CE2", "OH"),),
        modified=True,
    ),
    ResidueSiteClass(
        "asn_amide",
        "ASN",
        ("CG", "OD1", "ND2"),
        ("CB",),
        "CB",
        (("CB", "CG"),),
        control_for=ASP_CARBOXYLATE,
    ),
    ResidueSiteClass(
        "gln_amide",
        "GLN",
        ("CD", "OE1", "NE2"),
        ("CG",),
        "CG",
        (("CG", "CD"),),
        control_for=GLU_CARBOXYLATE,
    ),
    ResidueSiteClass(
        "phe_para_carbon",
        "PHE",
        ("CZ",),
        ("CE1", "CE2"),
        "CE1",
        (("CE1", "CZ"), ("CE2", "CZ")),
        control_for=TYR_HYDROXYL,
    ),
    ResidueSiteClass(
        "phe_ring_carbons",
        "PHE",
        ("CD1", "CD2", "CE1", "CE2"),
        ("CG",),
        "CG",
        (("CG", "CD1"), ("CG", "CD2")),
        control_for=HIS_IMIDAZOLE,
    ),
    ResidueSiteClass(
        "leu_delta_carbons",
        "LEU",
        ("CD1", "CD2"),
        ("CG",),
        "CG",
        (("CG", "CD1"), ("CG", "CD2")),
        control_for=MET_THIOETHER,
    ),
    ResidueSiteClass(
        "ile_delta_carbon",
        "ILE",
        ("CD1",),
        ("CG1",),
        "CG1",
        (("CG1", "CD1"),),
        control_for=MET_THIOETHER,
    ),
    ResidueSiteClass(
        "thr_hydroxyl",
        "THR",
        ("OG1",),
        ("CB",),
        "CB",
        (("CB", "OG1"),),
        control_for=TYR_HYDROXYL,
    ),
    ResidueSiteClass(
        "ser_hydroxyl",
        "SER",
        ("OG",),
        ("CB",),
        "CB",
        (("CB", "OG"),),
        control_for=CYS_SULFUR,
    ),
)
# A class deposited under more than one code is named once.
SITE_CLASS_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys(site.name for site in SITE_CLASSES)
)
SENSITIVE_CLASS_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys(site.name for site in SITE_CLASSES if site.control_for is None)
)

# ok, or why a value of a row is missing. A row takes the first gap that
# applies, in the order listed after ok.
SITE_STATUS_OK = "ok"
SITE_STATUS_ATOMS_MISSING = "site_atoms_missing"
SITE_STATUS_ZERO_OCCUPANCY = "zero_occupancy"
SITE_STATUS_B_FACTOR_UNAVAILABLE = "b_factor_unavailable"
SITE_STATUSES: tuple[str, ...] = (
    SITE_STATUS_OK,
    SITE_STATUS_ATOMS_MISSING,
    SITE_STATUS_ZERO_OCCUPANCY,
    SITE_STATUS_B_FACTOR_UNAVAILABLE,
)

# The structure-level outcomes the run report records.
RESIDUE_SITES_STATUS_OK = "ok"
RESIDUE_SITES_STATUS_NO_SITES = "no_sites"
RESIDUE_SITES_STATUS_FAILED = "failed"

__all__ = [
    "ASP_CARBOXYLATE",
    "CYS_SULFUR",
    "C_TERMINAL_CARBOXYLATE",
    "GLU_CARBOXYLATE",
    "HIS_IMIDAZOLE",
    "HIS_OXO",
    "METAL_CONTACT_DISTANCE_A",
    "MET_SULFONE",
    "MET_SULFOXIDE",
    "MET_THIOETHER",
    "PHE_HYDROXYLATED",
    "RESIDUE_SITES_STATUS_FAILED",
    "RESIDUE_SITES_STATUS_NO_SITES",
    "RESIDUE_SITES_STATUS_OK",
    "SENSITIVE_CLASS_NAMES",
    "SITE_CLASSES",
    "SITE_CLASS_NAMES",
    "SITE_STATUSES",
    "SITE_STATUS_ATOMS_MISSING",
    "SITE_STATUS_B_FACTOR_UNAVAILABLE",
    "SITE_STATUS_OK",
    "SITE_STATUS_ZERO_OCCUPANCY",
    "TRP_HYDROPEROXIDE",
    "TRP_HYDROXYLATED",
    "TRP_INDOLE",
    "TRP_KYNURENINE",
    "TYR_DOPA",
    "TYR_HYDROXYL",
    "ResidueSiteClass",
]
