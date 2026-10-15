from __future__ import annotations

import pandas as pd
import pytest

from pdbai.structure.coordinate_preprocessing import highest_occupancy_conformer_mask


def _atoms(
    alt_ids: list[str],
    occupancies: list[object],
    *,
    residues: list[str] | None = None,
    models: list[str] | None = None,
    chains: list[str] | None = None,
) -> pd.DataFrame:
    count = len(alt_ids)
    return pd.DataFrame(
        {
            "_atom_site.group_PDB": ["ATOM"] * count,
            "_atom_site.label_alt_id": alt_ids,
            "_atom_site.occupancy": occupancies,
            "_atom_site.auth_asym_id": chains or ["A"] * count,
            "_atom_site.auth_seq_id": residues or ["10"] * count,
            "_atom_site.auth_comp_id": ["SER"] * count,
            "_atom_site.pdbx_PDB_model_num": models or ["1"] * count,
        }
    )


def _kept(dataframe: pd.DataFrame) -> list[bool]:
    return [bool(value) for value in highest_occupancy_conformer_mask(dataframe)]


def test_every_row_is_kept_when_no_atom_has_an_alternate_identifier() -> None:
    dataframe = _atoms([".", "?", "", "."], ["1.0", None, "abc", "0.5"])

    assert _kept(dataframe) == [True, True, True, True]


def test_every_row_is_kept_when_the_alternate_identifier_column_is_absent() -> None:
    dataframe = _atoms(["", ""], ["0.5", "0.5"]).drop(columns="_atom_site.label_alt_id")

    assert _kept(dataframe) == [True, True]


def test_an_empty_table_gives_an_empty_mask() -> None:
    dataframe = _atoms([], [])

    assert _kept(dataframe) == []


def test_the_conformer_with_the_highest_mean_occupancy_is_kept_whole() -> None:
    # Atom by atom, A would win the first pair and B the second, which would
    # assemble a hybrid residue. B has the higher mean, so all of B is kept.
    dataframe = _atoms(
        [".", "A", "B", "A", "B"],
        ["1.0", "0.55", "0.45", "0.2", "0.8"],
    )

    assert _kept(dataframe) == [True, False, True, False, True]


def test_atoms_without_an_identifier_are_kept_beside_the_chosen_conformer() -> None:
    dataframe = _atoms(
        ["", "A", "B", "", "A", "B"],
        ["1.0", "0.7", "0.3", "0.0", "0.7", "0.3"],
    )

    assert _kept(dataframe) == [True, True, False, True, True, False]


def test_a_tie_keeps_the_identifier_that_appears_first() -> None:
    dataframe = _atoms(["B", "A", "B", "A"], ["0.5", "0.5", "0.5", "0.5"])

    assert _kept(dataframe) == [True, False, True, False]


def test_means_equal_up_to_rounding_error_tie() -> None:
    # In floating point, 0.1 and 0.2 average to 0.15000000000000002. Without
    # rounding, B would beat the 0.15 of A although the means are equal.
    dataframe = _atoms(["A", "A", "B", "B"], ["0.15", "0.15", "0.1", "0.2"])

    assert _kept(dataframe) == [True, True, False, False]


def test_numeric_and_multi_character_identifiers_are_conformers() -> None:
    dataframe = _atoms(
        ["1", "2", "AB", "CD"],
        ["0.4", "0.6", "0.6", "0.4"],
        residues=["10", "10", "11", "11"],
    )

    assert _kept(dataframe) == [False, True, True, False]


def test_each_residue_chooses_its_own_conformer() -> None:
    dataframe = _atoms(
        ["A", "B", "A", "B"],
        ["0.7", "0.3", "0.3", "0.7"],
        residues=["10", "10", "11", "11"],
    )

    assert _kept(dataframe) == [True, False, False, True]


def test_each_chain_chooses_its_own_conformer() -> None:
    dataframe = _atoms(
        ["A", "B", "A", "B"],
        ["0.7", "0.3", "0.3", "0.7"],
        chains=["A", "A", "B", "B"],
    )

    assert _kept(dataframe) == [True, False, False, True]


def test_each_model_chooses_its_own_conformer() -> None:
    dataframe = _atoms(
        ["A", "B", "A", "B"],
        ["0.7", "0.3", "0.3", "0.7"],
        models=["1", "1", "2", "2"],
    )

    assert _kept(dataframe) == [True, False, False, True]


def test_alternates_of_different_residue_types_at_one_position_compete() -> None:
    # Microheterogeneity puts two residue types at one sequence position. The
    # label key gives both the same polymer position, so only one is kept.
    dataframe = pd.DataFrame(
        {
            "_atom_site.group_PDB": ["ATOM"] * 4,
            "_atom_site.label_alt_id": ["A", "A", "B", "B"],
            "_atom_site.occupancy": ["0.4", "0.4", "0.6", "0.6"],
            "_atom_site.label_asym_id": ["A"] * 4,
            "_atom_site.label_seq_id": ["5"] * 4,
            "_atom_site.auth_asym_id": ["A"] * 4,
            "_atom_site.auth_seq_id": ["10"] * 4,
            "_atom_site.auth_comp_id": ["SER", "SER", "ALA", "ALA"],
            "_pdbai_polymer_type": ["polypeptide(L)"] * 4,
        }
    )

    assert _kept(dataframe) == [False, False, True, True]


def test_an_atom_without_a_residue_identity_is_its_own_residue() -> None:
    # With no author chain these atoms have no identity. Grouped together,
    # the two atoms would compete although they belong to different residues.
    dataframe = _atoms(
        ["A", "B"],
        ["0.7", "0.3"],
        residues=["301", "302"],
        chains=["?", "?"],
    )

    assert _kept(dataframe) == [True, True]


def test_the_mask_is_positional_whatever_the_index() -> None:
    dataframe = _atoms(["A", "B", "", "B"], ["0.3", "0.7", "1.0", "0.7"])
    dataframe.index = pd.Index([30, 10, 20, 0])

    assert _kept(dataframe) == [False, True, True, True]


def test_occupancies_may_be_numbers() -> None:
    dataframe = _atoms(["A", "B"], [0.3, 0.7])

    assert _kept(dataframe) == [False, True]


def test_an_occupancy_is_read_without_its_standard_uncertainty() -> None:
    dataframe = _atoms(["A", "B"], ["0.30(12)", "0.70(2)"])

    assert _kept(dataframe) == [False, True]


def test_a_missing_occupancy_on_an_alternate_atom_raises() -> None:
    dataframe = _atoms(["A", "B"], ["0.5", None])

    with pytest.raises(ValueError, match=r"'_atom_site\.occupancy'.*a missing value"):
        highest_occupancy_conformer_mask(dataframe)


def test_an_unreadable_occupancy_on_an_alternate_atom_raises() -> None:
    dataframe = _atoms(["A", "B", "B", "B"], ["0.5", "half", "inf", "0.5(x)"])

    with pytest.raises(
        ValueError, match=r"'_atom_site\.occupancy'.*'half', 'inf', '0\.5\(x\)'"
    ):
        highest_occupancy_conformer_mask(dataframe)


def test_a_missing_occupancy_column_raises_when_alternates_exist() -> None:
    dataframe = _atoms(["A", "B"], ["0.5", "0.5"]).drop(columns="_atom_site.occupancy")

    with pytest.raises(ValueError, match=r"'_atom_site\.occupancy' is missing"):
        highest_occupancy_conformer_mask(dataframe)


def test_the_occupancy_of_an_atom_without_an_identifier_is_not_checked() -> None:
    dataframe = _atoms(["", "A", "B"], [None, "0.4", "0.6"])

    assert _kept(dataframe) == [True, False, True]
