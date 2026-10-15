"""The structure file the metal-site engine's tests share.

Only the tests in this directory read it, so it lives here rather than in
the suite's shared fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_HEADER = """data_XXXX

_cell.length_a 30.000
_cell.length_b 30.000
_cell.length_c 30.000
_cell.angle_alpha 90.00
_cell.angle_beta 90.00
_cell.angle_gamma 90.00
_symmetry.space_group_name_H-M 'P 1'

_refine.ls_number_reflns_obs 20000
_refine.ls_R_factor_R_free 0.2000

loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.pdbx_ptnr1_PDB_ins_code
_struct_conn.ptnr1_symmetry
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.pdbx_ptnr2_PDB_ins_code
_struct_conn.ptnr2_symmetry
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_seq_id
metalc1 metalc B ZN  . ZN  ? 1_555 A HIS 2  NE2 ? 1_555 A 401 A 102
metalc2 metalc B ZN  . ZN  ? 1_555 A ASN 5  ND2 ? 1_555 A 401 A 105
metalc3 metalc C CA  . CA  ? 1_555 A ASP 10 OD1 ? 1_455 A 402 A 110
metalc4 metalc E HEM . FE  ? 1_555 F HOH .  O   ? 1_555 A 404 A 599
metalc5 metalc E HEM . FE  ? 1_555 E HEM .  NA  ? 1_555 A 404 A 404
disulf1 disulf A CYS 4  SG  ? 1_555 A CYS 4  SG  ? 1_555 A 104 A 104
"""

_ATOM_HEADER = """
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
"""

_ATOMS = """ATOM   1  N  N   . MET A 1 1  ? 7.21   25     25     1.00 20   101 MET A N   1
ATOM   2  N  NE2 . HIS A 1 2  ? 12.03  10     10     1.00 20   102 HIS A NE2 1
ATOM   3  H  HE2 . HIS A 1 2  ? 11.00  10     10     1.00 25   102 HIS A HE2 1
ATOM   4  N  ND1 . HIS A 1 3  ? 10     12.23  10     1.00 30   103 HIS A ND1 1
ATOM   5  S  SG  . CYS A 1 4  ? 10     10     12.31  1.00 10   104 CYS A SG  1
ATOM   6  N  ND2 . ASN A 1 5  ? 10     10     6.90   1.00 20   105 ASN A ND2 1
ATOM   7  O  O   . ALA A 1 6  ? 8.5    8.5    10     0.00 20   106 ALA A O   1
ATOM   8  N  N   . GLY A 1 7  ? 11.5   11.5   10     1.00 20   107 GLY A N   1
ATOM   9  O  OD1 . ASP A 1 10 ? 1.86   20     20     1.00 20   110 ASP A OD1 1
ATOM   10 N  NE2 . HIS A 1 20 ? 22.2   5      5      1.00 20   120 HIS A NE2 1
ATOM   11 N  NE2 . HIS A 1 30 ? 20     20     7.16   1.00 20   130 HIS A NE2 1
HETATM 12 ZN ZN  . ZN  B 2 .  ? 10     10     10     1.00 15   401 ZN  A ZN  1
HETATM 13 CA CA  . CA  C 3 .  ? 29.5   20     20     1.00 15   402 CA  A CA  1
HETATM 14 CD CD  . CD  D 4 .  ? 20     5      5      1.00 15   403 CD  A CD  1
HETATM 15 FE FE  . HEM E 5 .  ? 20     20     5      1.00 12   404 HEM A FE  1
HETATM 16 N  NA  . HEM E 5 .  ? 22     20     5      1.00 12   404 HEM A NA  1
HETATM 17 N  NB  . HEM E 5 .  ? 20     22     5      1.00 12   404 HEM A NB  1
HETATM 18 MN MN  . MN  G 6 .  ? 5      25     25     1.00 15   405 MN  A MN  1
HETATM 19 O  O   . HOH F 7 .  ? 7.5    10     10     1.00 30   501 HOH A O   1
HETATM 20 O  O   . HOH F 7 .  ? 10     7.0    10     1.00 30   502 HOH A O   1
ATOM   21 N  NE2 . HIS A 1 2  ? 12.50  10     10     1.00 20   102 HIS A NE2 2
HETATM 22 ZN ZN  . ZN  B 2 .  ? 10     10     10     1.00 15   401 ZN  A ZN  2
"""


def metal_cif_text(*, cell: bool = True, atoms: str = _ATOMS) -> str:
    """Return the fixture's text, without its cell and space group on request."""
    header = _HEADER
    if not cell:
        header = "\n".join(
            line
            for line in header.splitlines()
            if not line.startswith(("_cell.", "_symmetry."))
        )
    return header + _ATOM_HEADER + atoms


@pytest.fixture
def metal_cif(tmp_path: Path) -> Path:
    """Write entry 1MTL, every case the metal-site engine handles, and return it.

    The file is 1mtl.cif with the data block XXXX. The cell is a 30 Å cube in
    P 1, so the only symmetry mates are translations. Author residue numbers
    are label numbers plus 100. The file states 20,000 reflections and an
    R-free of 0.20. Distances lie along an axis, so they are exact.

    Model 1 holds five metal atoms, in file order ZN 401, CA 402, CD 403, the
    FE of HEM 404, and MN 405:

    1. ZN 401 at (10, 10, 10), an ion.
       - His 102 NE2 at 2.03 Å, the reference distance (HIS:N:ZN 2.03, sd
         0.05), named by record metalc1: struct_conn_and_distance, deviation
         0. Its HE2, a hydrogen 1.0 Å from the zinc, is never a donor.
       - His 103 ND1 at 2.23 Å: distance, deviation +0.20.
       - Cys 104 SG at 2.31 Å, the reference distance (CYS:S:ZN 2.31).
       - HOH 501 at 2.50 Å: inside the first sphere of water (2.09 + 0.75 =
         2.84), deviation +0.41.
       - HOH 502 at 3.00 Å: outside it, so no row.
       - Asn 105 ND2 at 3.10 Å: not a typical donor and beyond the first
         sphere of nitrogen (2.03 + 0.75), but record metalc2 names it:
         struct_conn, element fallback reference *:N:ZN, no deviation.
       - Ala 106 O at 2.12 Å with occupancy 0.00: not usable, no row.
       - Gly 107 N at 2.12 Å: a backbone nitrogen that is not the first of
         its sequence, no row.
       Five bonds: 3 N, 1 S, 1 O (water), 2 declared.
    2. CA 402 at (29.5, 20, 20), an ion. Asp 110 OD1 at (1.86, 20, 20) is
       27.64 Å away in the file and 2.36 Å away one cell on along a, the
       reference distance (ASP:O:CA 2.36). Record metalc3 names it under a
       different operator from the metal's: struct_conn_and_distance,
       operator 1_655, a symmetry mate.
    3. CD 403 at (20, 5, 5), an ion. His 120 NE2 lies 2.2 Å away, but the
       reference table has no cadmium, so nothing is inferred: no_donors_found.
    4. The FE of HEM 404 at (20, 20, 5), a heme with three atoms. Its own NA
       and NB at 2.0 Å are never donors, and record metalc5 to NA is passed
       over. His 130 NE2 at 2.16 Å is the reference distance (HIS:N:FE
       2.16). Record metalc4 names HOH 599, which the file lacks: unresolved.
    5. MN 405 at (5, 25, 25), an ion. Met 101 N, the first residue of its
       sequence, at 2.21 Å: n_terminal_nitrogen, element fallback *:N:MN
       (2.21), so a bond without a deviation.

    Model 2 holds ZN 401 and His 102 NE2 at 2.50 Å, within the first sphere
    (2.78): one bond, by distance alone on this model's own coordinates,
    which record metalc1 also names.

    In all, 6 site rows and 9 bond rows, 8 in model 1 and 1 in model 2. The
    status is degraded, with 2 unresolved records, metalc4 in model 1 and
    metalc2 in model 2, which has no Asn 105. The precision index counts the
    19 non-hydrogen atoms of model 1 by occupancy, 18.0, in an asymmetric
    unit of 27,000 Å³.
    """
    path = tmp_path / "1mtl.cif"
    path.write_text(metal_cif_text(), encoding="utf-8")
    return path
