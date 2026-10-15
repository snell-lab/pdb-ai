"""The structure file the disulfide engine's tests share.

Only the tests in this directory read it, so it lives here rather than in
the suite's shared fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_DISULFIDE_CIF = """data_XXXX

loop_
_entity.id
_entity.type
1 polymer
2 non-polymer
3 non-polymer

loop_
_entity_poly.entity_id
_entity_poly.type
1 'polypeptide(L)'

loop_
_chem_comp.id
_chem_comp.type
CYS 'L-peptide linking'
ZN non-polymer

loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.pdbx_ptnr1_label_alt_id
_struct_conn.pdbx_ptnr1_PDB_ins_code
_struct_conn.ptnr1_symmetry
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.pdbx_ptnr2_label_alt_id
_struct_conn.pdbx_ptnr2_PDB_ins_code
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_comp_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_comp_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_symmetry
_struct_conn.pdbx_dist_value
disulf1 disulf A CYS 1  SG . ? 1_555 A CYS 40 SG . ? A CYS 101 A CYS 140 1_555 2.040
disulf2 disulf A CYS 5  SG A ? 1_555 A CYS 30 SG A ? A CYS 105 A CYS 130 1_555 2.050
disulf3 disulf A CYS 30 SG B ? 1_555 A CYS 5  SG B ? A CYS 130 A CYS 105 1_555 3.600
disulf4 disulf A CYS 95 SG . ? 1_555 A CYS 40 SG . ? A CYS 195 A CYS 140 1_555 ?
disulf5 disulf A CYS 1  SG . ? 1_555 A CYS 99 SG . ? A CYS 101 A CYS 199 1_555 ?
disulf6 disulf A CYS 80 SG . ? 1_555 A CYS 80 SG . ? A CYS 180 A CYS 180 2_555 2.050
disulf7 disulf A CYS 25 SG . ? 1_555 C CYS .  SG . ? A CYS 125 B CYS 301 1_555 2.030
disulf8 disulf A CYS 76 SG . ? 1_555 A CYS 94 SG . ? A CYS 176 A CYS 194 1_555 3.600
covale1 covale A CYS 1 SG . ? 1_555 A CYS 5 SG A ? A CYS 101 A CYS 105 1_555 ?
metalc1 metalc B ZN . ZN . ? 1_555 A CYS 70 SG . ? A ZN 401 A CYS 170 1_555 2.330
metalc2 metalc B ZN . ZN . ? 1_555 A CYS 73 SG . ? A ZN 401 A CYS 173 1_555 2.330
metalc3 metalc B ZN . ZN . ? 1_555 A CYS 86 SG . ? A ZN 401 A CYS 186 1_555 2.330
metalc4 metalc B ZN . ZN . ? 1_555 A CYS 89 SG . ? A ZN 401 A CYS 189 1_555 2.330
metalc5 metalc B ZN . ZN . ? 1_555 D HOH . O . ? A ZN 401 A HOH 501 1_555 2.100
metalc6 metalc E ZN . ZN . ? 1_555 A CYS 110 SG . ? A ZN 402 A CYS 210 1_555 2.330
metalc7 metalc E ZN . ZN . ? 1_555 A CYS 113 SG . ? A ZN 402 A CYS 213 1_555 2.330

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
ATOM   1  C  CB  . CYS A 1 1  ? 0       0      -1.8   1.00 18   101 CYS A CB  1
ATOM   2  S  SG  . CYS A 1 1  ? 0       0       0     1.00 20   101 CYS A SG  1
ATOM   3  H  HB2 . CYS A 1 1  ? 0       1      -1.8   1.00 25   101 CYS A HB2 1
ATOM   4  C  CB  . CYS A 1 40 ? 2.04    0      -1.8   1.00 18   140 CYS A CB  1
ATOM   5  S  SG  . CYS A 1 40 ? 2.04    0       0     1.00 20   140 CYS A SG  1
ATOM   6  D  DG  A CYS A 1 40 ? 2.04    0       1.3   0.50 30   140 CYS A DG  1
ATOM   7  C  CB  . CYS A 1 5  ? 20      0      -1.8   1.00 20   105 CYS A CB  1
ATOM   8  S  SG  A CYS A 1 5  ? 20      0       0     0.60 30   105 CYS A SG  1
ATOM   9  S  SG  B CYS A 1 5  ? 20     -1.1     0     0.40 40   105 CYS A SG  1
ATOM   10 C  CB  . CYS A 1 30 ? 22.05   0      -1.8   1.00 20   130 CYS A CB  1
ATOM   11 S  SG  A CYS A 1 30 ? 22.05   0       0     0.60 25   130 CYS A SG  1
ATOM   12 S  SG  B CYS A 1 30 ? 20      2.5     0     0.40 35   130 CYS A SG  1
ATOM   13 C  CB  . CYS A 1 20 ? 40      0       8.2   1.00 10   120 CYS A CB  1
ATOM   14 S  SG  . CYS A 1 20 ? 40      0       10    1.00 12   120 CYS A SG  1
ATOM   15 C  CB  . CYS A 1 21 A 40      0      -1.8   1.00 10   120 CYS A CB  1
ATOM   16 S  SG  . CYS A 1 21 A 40      0       0     1.00 15   120 CYS A SG  1
ATOM   17 C  CB  . CYS A 1 60 ? 42.7    0      -1.8   1.00 10   160 CYS A CB  1
ATOM   18 S  SG  . CYS A 1 60 ? 42.7    0       0     1.00 12   160 CYS A SG  1
ATOM   19 C  CB  . CYS A 1 70 ? 62.69   2.69    2.69  1.00 20   170 CYS A CB  1
ATOM   20 S  SG  . CYS A 1 70 ? 61.345  1.345   1.345 1.00 20   170 CYS A SG  1
ATOM   21 C  CB  . CYS A 1 73 ? 62.69  -2.69   -2.69  1.00 20   173 CYS A CB  1
ATOM   22 S  SG  . CYS A 1 73 ? 61.345 -1.345  -1.345 1.00 20   173 CYS A SG  1
ATOM   23 C  CB  . CYS A 1 86 ? 57.31   2.69   -2.69  1.00 20   186 CYS A CB  1
ATOM   24 S  SG  . CYS A 1 86 ? 58.655  1.345  -1.345 1.00 20   186 CYS A SG  1
ATOM   25 C  CB  . CYS A 1 89 ? 57.31  -2.69    2.69  1.00 20   189 CYS A CB  1
ATOM   26 S  SG  . CYS A 1 89 ? 58.655 -1.345   1.345 1.00 20   189 CYS A SG  1
HETATM 27 ZN ZN  . ZN  B 2 .  ? 60      0       0     1.00 15   401 ZN  A ZN  1
ATOM   28 C  CB  . CYS A 1 95 ? 80      0       0     1.00 20   195 CYS A CB  1
ATOM   29 C  CB  . CYS A 1 80 ? 100     0      -1.8   1.00 20   180 CYS A CB  1
ATOM   30 S  SG  . CYS A 1 80 ? 100     0       0     1.00 20   180 CYS A SG  1
ATOM   31 C  CB  . CYS A 1 25 ? 120     0      -1.8   1.00 16   125 CYS A CB  1
ATOM   32 S  SG  . CYS A 1 25 ? 120     0       0     1.00 20   125 CYS A SG  1
HETATM 33 C  CB  . CYS C 3 .  ? 122.03  0      -1.8   1.00 15   301 CYS B CB  1
HETATM 34 S  SG  . CYS C 3 .  ? 122.03  0       0     1.00 0.00 301 CYS B SG  1
ATOM   35 C  CB  A CYS A 1 50 ? 140     0      -1.8   0.45 20   150 CYS A CB  1
ATOM   36 S  SG  A CYS A 1 50 ? 140     0       0     0.45 30   150 CYS A SG  1
ATOM   37 C  CB  B CYS A 1 50 ? 140     1      -1.8   0.55 25   150 CYS A CB  1
ATOM   38 C  CB  . CYS A 1 55 ? 142.05  0      -1.8   1.00 20   155 CYS A CB  1
ATOM   39 S  SG  . CYS A 1 55 ? 142.05  0       0     1.00 22   155 CYS A SG  1
ATOM   40 C  CB  . CYS A 1 76 ? 160     0      -1.8   1.00 20   176 CYS A CB  1
ATOM   41 S  SG  . CYS A 1 76 ? 160     0       0     0.70 25   176 CYS A SG  1
ATOM   42 C  CB  . CYS A 1 94 ? 163.6   0      -1.8   1.00 20   194 CYS A CB  1
ATOM   43 S  SG  . CYS A 1 94 ? 163.6   0       0     1.00 20   194 CYS A SG  1
ATOM   44 C  CB  . CYS A 1 110 ? 178.55 1.824  -1.8   1.00 20   210 CYS A CB  1
ATOM   45 S  SG  . CYS A 1 110 ? 178.55 1.824   0     1.00 24   210 CYS A SG  1
ATOM   46 C  CB  . CYS A 1 113 ? 181.45 1.824  -1.8   1.00 20   213 CYS A CB  1
ATOM   47 S  SG  . CYS A 1 113 ? 181.45 1.824   0     1.00 22   213 CYS A SG  1
HETATM 48 ZN ZN  . ZN  E 2 .  ? 180     0       0     1.00 15   402 ZN  A ZN  1
ATOM   49 C  CB  . CYS A 1 1  ? 0       0      -1.8   1.00 0.00 101 CYS A CB  2
ATOM   50 S  SG  . CYS A 1 1  ? 0       0       0     1.00 0.00 101 CYS A SG  2
ATOM   51 C  CB  . CYS A 1 40 ? 2.6     0      -1.8   1.00 0.00 140 CYS A CB  2
ATOM   52 S  SG  . CYS A 1 40 ? 2.6     0       0     1.00 0.00 140 CYS A SG  2
"""


@pytest.fixture
def disulfide_cif(tmp_path: Path) -> Path:
    """Write entry 1DSF, every case the disulfide engine handles, and return it.

    The file is 1dsf.cif with the data block XXXX, so the entry's ID can only
    come from the file name. Author residue numbers are label numbers plus
    100, except Cys 120A, whose label number is 21, so reading the wrong
    field shows. Every residue is in chain A (label chain A) unless stated.
    The SG atoms of different cases lie at least 10 Å apart, so only the
    SG–SG distances stated here are within 3.0 Å. Where it matters, the
    coordinates lie on an axis, so those distances are exact. The two zinc
    atoms are the only metals, and metal_bridged is True on the row of case
    12 alone.

    Model 1:

    1. Cys 101 and Cys 140, an intact bond: SG at (0, 0, 0) and (2.04, 0, 0),
       2.04 Å, B(SG) 20 and B(CB) 18 on both, record disulf1. Cys 101 also
       has HB2 (element H) and Cys 140 has DG (element D) with altloc A at
       occupancy 0.50, which nothing reads. Row: struct_conn_and_distance,
       distances 2.04 and 2.04, conformer occupancy 1.0, total 1.0, no
       alternates, ratios 20/18 and 20/18, ok.
    2. Cys 105 and Cys 130, partially broken. Each CB is unlabelled, and each
       SG has conformer A at 0.60 and B at 0.40. 105A (20, 0, 0) to 130A
       (22.05, 0, 0) is 2.05 Å, and 105B (20, -1.1, 0) to 130B (20, 2.5, 0) is
       3.60 Å. The traps 105A–130B at 2.50 Å and 105B–130A at
       sqrt(2.05² + 1.1²) = 2.33 Å join conformers that cannot coexist, so
       neither is measured. Records disulf2 (A–A) and disulf3 (reversed,
       B–B) give one pair. The parser keeps conformer A of both residues.
       B: 105 SG A 30, SG B 40, CB 20; 130 SG A 25, SG B 35, CB 20. Row:
       struct_conn_and_distance, 2.05 and 3.60, conformer occupancy 0.40,
       total 0.6 + 0.4 = 1.0 on both, alternates True, ratios 25/20 = 1.25
       and 30/20 = 1.5, ok, site_id 1DSF:1:A:105::SG:A.
    3. Cys 120A (insertion code A) at (40, 0, 0) and Cys 160 at (42.7, 0, 0),
       2.70 Å with no record. Cys 120 without an insertion code is free at
       (40, 0, 10), 10 Å away. B: 120A SG 15, CB 10; 160 SG 12, CB 10. Row:
       distance, 2.70 and 2.70, occupancies 1.0, no alternates, ratios 1.2
       and 1.5, ok, site_id 1DSF:1:A:120:A:SG:.
    4. A zinc site: ZN A 401 (HETATM, label chain B) at (60, 0, 0) with the
       SGs of Cys 170, 173, 186, and 189 at the corners (60 ± 1.345, ± 1.345,
       ± 1.345) of a tetrahedron, 1.345·√3 = 2.330 Å from the zinc and
       2·1.345·√2 = 3.804 Å from one another. Records metalc1–metalc4 join
       them to the zinc. No row, since no pair is within 3.0 Å and no disulf
       record joins one.
    5. Cys 195 has a CB and no SG. Record disulf4 joins its SG to Cys 140,
       so it matches no pair: unmatched. Cys 195 is the one cysteine counted
       as having a CB but no SG.
    6. Record disulf5 joins Cys 101 to Cys 199, which the file does not hold:
       unmatched.
    7. Cys 180 is free. Record disulf6 joins it to itself under 2_555, a
       symmetry mate: counted, no row.
    8. CYS B 301, a free cysteine (HETATM, entity 3, label chain C, label
       sequence number "."), has its SG at (122.03, 0, 0), 2.03 Å from the SG
       of Cys 125 at (120, 0, 0). Record disulf7 names it by its author
       fields. B: 125 SG 20, CB 16; 301 SG 0.00, CB 15. Row: Cys 125 first
       (chain A before B), struct_conn_and_distance, 2.03 and 2.03,
       occupancies 1.0, no alternates, ratios missing since B(SG) of 301 is
       0, b_factor_unavailable.
    9. Cys 150 has CB A and SG A at 0.45 and CB B at 0.55 with no SG B, so the
       parser keeps conformer B, which has no SG. Cys 155's SG at
       (142.05, 0, 0) is 2.05 Å from 150's SG A at (140, 0, 0), with no
       record. 150's representative SG is SG A and its CB is CB A, the one
       that can coexist with it. B: 150 SG A 30, CB A 20, CB B 25; 155 SG 22,
       CB 20. Row: distance, 2.05 and 2.05, conformer occupancy 0.45, total
       min(0.45, 1.0) = 0.45, occupancy deficit 0.45 + 0.55 - 0.45 = 0.55,
       no alternates (one label), ratios 22/20 = 1.1 and 30/20 = 1.5,
       sg_not_in_atom_table, site_id 1DSF:1:A:150::SG:A, altloc A. No atom
       links to Cys 150, and a warning names it.
    10. Cys 176 at (160, 0, 0) and Cys 194 at (163.6, 0, 0), 3.60 Å, an
        annotated but cleaved bond with record disulf8. 176's SG has
        occupancy 0.70 and no altloc. B: 176 SG 25, CB 20; 194 SG 20, CB 20.
        Row: struct_conn, 3.60 and 3.60, conformer occupancy 0.70, total
        min(0.70, 1.0) = 0.70, occupancy deficit 1.00 - 0.70 = 0.30, no
        alternates, ratios 1.0 and 1.25, ok.
    11. Record covale1 joins Cys 101 to Cys 105, and metalc5 joins the zinc to
        a water. Neither is a disulf record, so neither gives a row.
    12. A collapsed zinc site, as low-resolution models hold: ZN A 402
        (HETATM, label chain E) at (180, 0, 0) with the SGs of Cys 210 at
        (178.55, 1.824, 0) and Cys 213 at (181.45, 1.824, 0). Each SG is
        sqrt(1.45² + 1.824²) = 2.330 Å from the zinc, within the 2.8 Å metal
        contact distance, and the two are 2.90 Å apart. Records metalc6 and
        metalc7 join them to the zinc, and no disulf record names them. B:
        210 SG 24, CB 20; 213 SG 22, CB 20. Row: distance, 2.90 and 2.90,
        occupancies 1.0, no alternates, ratios 22/20 = 1.1 and 24/20 = 1.2,
        ok, metal_bridged True.

    Model 2 holds Cys 101 at (0, 0, 0) and Cys 140 at (2.6, 0, 0), every B at
    0.00. Record disulf1 applies to every model. Row: struct_conn_and_distance,
    2.60 and 2.60, occupancies 1.0, no alternates, ratios missing,
    b_factor_unavailable.

    In all, 8 rows, 7 in model 1 and 1 in model 2, in the order 101–140,
    105–130, 120A–160, 125–B301, 150–155, 176–194, 210–213, then model 2's
    101–140. The table has 29 columns. The occupancy deficit is 0 on every
    row but those of cases 9 and 10, since each other SG is modelled as fully
    as its CB. metal_bridged is False on every row but that of case 12. No
    other SG of a row is within 2.8 Å of a zinc, and model 2 holds no metal.
    The status is degraded, with 2 unmatched records (disulf4, disulf5), 1
    symmetry record (disulf6), and 1 cysteine with a CB but no SG (Cys 195).
    Fifteen atoms link to the table: the SGs of 101, 140, 105 A, 130 A, 120A,
    160, 125, B 301, 155, 176, 194, 210, and 213 in model 1, and of 101 and
    140 in model 2.
    """
    path = tmp_path / "1dsf.cif"
    path.write_text(_DISULFIDE_CIF, encoding="utf-8")
    return path
