from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdbai.engines.analyses.sasa.classification import (
    component_categories,
    normalized_elements,
    protein_radii,
    rasa_and_burial,
)
from pdbai.engines.analyses.sasa.constants import (
    DSSP_RADIUS_CA,
    DSSP_RADIUS_SIDECHAIN,
    EXPECTED_HEAVY_ATOMS,
    MAX_ASA_TIEN2013,
)
from pdbai.structure.residue_identity import build_residue_identities


def _nonpolymer_rows(residues: list[tuple[str, list[str]]]) -> pd.DataFrame:
    # Each residue is a non-polymer of its own, listed by component ID and the
    # element of each atom, numbered from 1 in chain B.
    rows: list[dict[str, object]] = []
    for number, (residue_name, elements) in enumerate(residues, start=1):
        for element in elements:
            rows.append(
                {
                    "pdb_id": "1ABC",
                    "model_num": "1",
                    "chain_id": "B",
                    "res_num": str(number),
                    "res_name": residue_name,
                    "atom_name": element,
                    "element": element,
                    "is_hetatm": True,
                    "_pdbai_entity_type": "non-polymer",
                }
            )
    return pd.DataFrame(rows)


def _categories(dataframe: pd.DataFrame) -> list[str]:
    identities = build_residue_identities(dataframe)
    return component_categories(identities, normalized_elements(dataframe)).tolist()


def test_nonpolymer_residues_are_sorted_by_their_lists_and_their_atoms() -> None:
    dataframe = _nonpolymer_rows(
        [
            ("HOH", ["O"]),
            ("GOL", ["C", "O"]),
            ("HEM", ["FE", "C"]),
            ("ZN", ["ZN"]),
            ("CL", ["CL"]),
            ("LIG", ["C", "N"]),
            ("UNL", ["C"]),
        ]
    )

    assert _categories(dataframe) == [
        "water",
        "experimental_additive",
        "experimental_additive",
        "ligand_or_cofactor",
        "ligand_or_cofactor",
        "metal",
        # A chloride ion is no metal, so it falls to the named ligands.
        "ligand_or_cofactor",
        "ligand_or_cofactor",
        "ligand_or_cofactor",
        "unknown_nonpolymer",
    ]


@pytest.mark.parametrize("residue_name", ["GA", "TL", "IN", "BI"])
def test_an_ion_of_any_metal_is_a_metal(residue_name: str) -> None:
    dataframe = _nonpolymer_rows([(residue_name, [residue_name])])

    assert _categories(dataframe) == ["metal"]


def test_an_unknown_atom_or_ion_is_an_unknown_nonpolymer() -> None:
    dataframe = _nonpolymer_rows([("UNX", ["X"]), ("UNX", ["NA", "C"])])

    assert _categories(dataframe) == ["unknown_nonpolymer"] * 3


def test_an_unlisted_nonpolymer_atom_without_identity_stays_unknown() -> None:
    dataframe = _nonpolymer_rows([("LIG", ["C"]), ("HEM", ["FE"])])
    dataframe["res_num"] = ""

    # Only a listed component is classified without a residue identity.
    assert _categories(dataframe) == ["unknown_nonpolymer", "ligand_or_cofactor"]


def test_protein_atoms_take_the_dssp_radius_of_their_role() -> None:
    radii = protein_radii(pd.Series(["N", "CA", "C", "O", "CB", "SE", "OXT"]))

    assert radii[1] == DSSP_RADIUS_CA
    assert np.array_equal(radii[[4, 5, 6]], np.full(3, DSSP_RADIUS_SIDECHAIN))


@pytest.mark.parametrize(
    ("fraction", "buried", "exposed"),
    [
        (0.2, True, False),
        (0.25, False, False),
        (0.4, False, True),
        # A relative area is not clipped at one.
        (1.5, False, True),
    ],
)
def test_relative_area_classes_meet_at_their_cut_offs(
    fraction: float, buried: bool, exposed: bool
) -> None:
    normalized = rasa_and_burial("ALA", fraction * MAX_ASA_TIEN2013["ALA"])

    assert normalized is not None
    rasa, is_buried, is_exposed = normalized
    assert rasa == pytest.approx(fraction)
    assert (is_buried, is_exposed) == (buried, exposed)


def test_a_residue_without_a_tien_maximum_has_no_relative_area() -> None:
    assert rasa_and_burial("SEP", 100.0) is None


def test_every_tien_maximum_has_an_expected_atom_list() -> None:
    assert set(MAX_ASA_TIEN2013) == set(EXPECTED_HEAVY_ATOMS)
    assert all(maximum > 0 for maximum in MAX_ASA_TIEN2013.values())
