"""A small structure file built atom by atom, for the site engines' tests.

The residue-site, heavy-atom, and cofactor engines read only the atom table,
so their tests state each atom's position, occupancy, and B-factor outright
and need no file on disk beyond the one write_site_cif makes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pdbai.structure import parse_mmcif_atom_rows

_CELL = """_cell.length_a 40.000
_cell.length_b 40.000
_cell.length_c 40.000
_cell.angle_alpha 90.00
_cell.angle_beta 90.00
_cell.angle_gamma 90.00
_symmetry.space_group_name_H-M '{space_group}'
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


@dataclass(frozen=True)
class SiteAtom:
    """One atom of the file. A residue of a polymer has a ``seq``."""

    comp: str
    number: int
    name: str
    xyz: tuple[float, float, float]
    element: str = ""
    b_factor: float = 20.0
    occupancy: float = 1.0
    altloc: str = "."
    # The label sequence number, or None for a component outside a polymer.
    seq: int | None = None
    hetatm: bool = False
    model: int = 1

    def line(self, serial: int) -> str:
        """Return the atom's _atom_site row."""
        element = self.element or self.name[0]
        name = f'"{self.name}"' if "'" in self.name else self.name
        return " ".join(
            [
                "HETATM" if self.hetatm else "ATOM",
                str(serial),
                element,
                name,
                self.altloc,
                self.comp,
                "A",
                "1",
                "." if self.seq is None else str(self.seq),
                "?",
                *(f"{value:.4f}" for value in self.xyz),
                f"{self.occupancy:.2f}",
                f"{self.b_factor:.2f}",
                str(self.number),
                self.comp,
                "A",
                name,
                str(self.model),
            ]
        )


def residue(
    comp: str,
    number: int,
    atoms: dict[str, tuple[float, float, float]],
    *,
    b_factors: dict[str, float] | None = None,
    occupancies: dict[str, float] | None = None,
    elements: dict[str, str] | None = None,
    polymer: bool = True,
    hetatm: bool = False,
    model: int = 1,
) -> list[SiteAtom]:
    """Return the atoms of one residue, each at the position ``atoms`` gives it."""
    return [
        SiteAtom(
            comp,
            number,
            name,
            xyz,
            element=(elements or {}).get(name, ""),
            b_factor=(b_factors or {}).get(name, 20.0),
            occupancy=(occupancies or {}).get(name, 1.0),
            seq=number if polymer else None,
            hetatm=hetatm,
            model=model,
        )
        for name, xyz in atoms.items()
    ]


def site_cif_text(atoms: list[SiteAtom], *, space_group: str | None = "P 1") -> str:
    """Return the file's text, without a cell when ``space_group`` is None."""
    cell = "" if space_group is None else _CELL.format(space_group=space_group)
    rows = "\n".join(atom.line(serial) for serial, atom in enumerate(atoms, start=1))
    return f"data_XXXX\n\n{cell}{_ATOM_HEADER}{rows}\n"


def write_site_cif(
    directory: Path,
    atoms: list[SiteAtom],
    *,
    name: str = "1sit",
    space_group: str | None = "P 1",
) -> Path:
    """Write the atoms as ``name``.cif in ``directory`` and return its path."""
    path = directory / f"{name}.cif"
    path.write_text(site_cif_text(atoms, space_group=space_group), encoding="utf-8")
    return path


def parse_site_atoms(path: Path) -> pd.DataFrame:
    """Return the atom table the parser builds from a file."""
    return parse_mmcif_atom_rows(path, {}, structure_id=path.stem).dataframe
