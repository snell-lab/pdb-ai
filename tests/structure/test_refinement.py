from __future__ import annotations

import pytest

from pdbai.structure.refinement import RefinementFields, read_refinement_fields


def test_a_file_that_says_nothing_of_its_refinement_gives_nothing() -> None:
    fields = read_refinement_fields({})

    assert fields == RefinementFields(
        refinement_program=None,
        refinement_program_version=None,
        refinement_programs=None,
        tls_group_count=None,
        tls_residual_adp_flag=None,
        anisotropic_atom_fraction=None,
    )


def test_the_refinement_program_is_the_row_classified_as_refinement() -> None:
    fields = read_refinement_fields(
        {
            "_software.classification": ["data reduction", "refinement", "phasing"],
            "_software.name": ["XDS", "PHENIX", "PHASER"],
            "_software.version": [None, "1.19.2_4158", "2.8.3"],
        }
    )

    assert fields.refinement_program == "PHENIX"
    assert fields.refinement_program_version == "1.19.2_4158"
    assert fields.refinement_programs == "PHENIX"


def test_several_refinement_programs_are_listed_once_each_in_file_order() -> None:
    fields = read_refinement_fields(
        {
            "_software.classification": [
                "refinement",
                "data scaling",
                "refinement",
                "refinement",
            ],
            "_software.name": ["REFMAC", "AIMLESS", "PHENIX", "REFMAC"],
            "_software.version": [None, "0.7.4", "1.16", "5.8.0238"],
        }
    )

    # The first row is the program, and its version is that row's own, not
    # the one a later row of the same name states.
    assert fields.refinement_program == "REFMAC"
    assert fields.refinement_program_version is None
    assert fields.refinement_programs == "REFMAC; PHENIX"


def test_program_names_are_kept_as_written_with_whitespace_collapsed() -> None:
    fields = read_refinement_fields(
        {
            "_software.classification": ["REFINEMENT", "refinement", "refinement"],
            "_software.name": ["SHELXL-97", "SHELXL", "phenix.refine"],
            "_software.version": ["  (phenix.refine:  1.8.1_1168) ", None, None],
        }
    )

    # The classification is matched in any letter case. Names of one family
    # of programs are not merged, and no name is put into capitals.
    assert fields.refinement_program == "SHELXL-97"
    assert fields.refinement_program_version == "(phenix.refine: 1.8.1_1168)"
    assert fields.refinement_programs == "SHELXL-97; SHELXL; phenix.refine"


def test_a_refinement_row_without_a_name_is_passed_over() -> None:
    fields = read_refinement_fields(
        {
            "_software.classification": ["refinement", "refinement", None],
            "_software.name": [None, "BUSTER", "CNS"],
            "_software.version": ["9.9", "2.10.3", "1.3"],
        }
    )

    assert fields.refinement_program == "BUSTER"
    assert fields.refinement_program_version == "2.10.3"
    assert fields.refinement_programs == "BUSTER"


def test_software_without_a_version_item_still_names_its_program() -> None:
    fields = read_refinement_fields(
        {"_software.classification": ["refinement"], "_software.name": ["REFMAC"]}
    )

    assert fields.refinement_program == "REFMAC"
    assert fields.refinement_program_version is None


def test_software_without_a_refinement_row_names_no_program() -> None:
    fields = read_refinement_fields(
        {"_software.classification": ["phasing"], "_software.name": ["PHASER"]}
    )

    assert fields.refinement_program is None
    assert fields.refinement_program_version is None
    assert fields.refinement_programs is None


def test_tls_groups_are_counted_by_their_rows() -> None:
    fields = read_refinement_fields(
        {
            "_refine.ls_d_res_high": ["1.8"],
            "_pdbx_refine_tls.id": ["1", "2", "3"],
            # The residue ranges of the groups are another category, and a
            # group can have several.
            "_pdbx_refine_tls_group.id": ["1", "2", "3", "4", "5"],
        }
    )

    assert fields.tls_group_count == 3


def test_a_refinement_without_tls_groups_has_zero_and_no_refinement_has_none() -> None:
    refined = read_refinement_fields({"_refine.ls_d_res_high": ["1.8"]})
    # _refine_hist is not _refine, so it does not show a refinement.
    unrefined = read_refinement_fields({"_refine_hist.d_res_high": ["1.8"]})
    groups_alone = read_refinement_fields({"_pdbx_refine_tls.id": ["1"]})

    assert refined.tls_group_count == 0
    assert unrefined.tls_group_count is None
    assert groups_alone.tls_group_count == 1


def test_the_tls_flag_is_the_first_one_a_refinement_row_states() -> None:
    stated = read_refinement_fields(
        {
            "_refine.pdbx_refine_id": ["NEUTRON DIFFRACTION", "X-RAY DIFFRACTION"],
            "_refine.pdbx_TLS_residual_ADP_flag": [None, "LIKELY RESIDUAL"],
        }
    )
    unstated = read_refinement_fields(
        {"_refine.pdbx_TLS_residual_ADP_flag": [None], "_pdbx_refine_tls.id": ["1"]}
    )

    assert stated.tls_residual_adp_flag == "LIKELY RESIDUAL"
    assert unstated.tls_residual_adp_flag is None


def test_the_anisotropic_fraction_counts_non_hydrogen_atom_sites() -> None:
    fields = read_refinement_fields(
        {
            "_atom_site.id": ["1", "2", "3", "4", "5", "6"],
            "_atom_site.type_symbol": ["N", "C", "C", "O", "H", "d"],
            # The hydrogen's record counts for nothing, as its atom does.
            "_atom_site_anisotrop.id": ["1", "2", "3", "5"],
        }
    )

    assert fields.anisotropic_atom_fraction == pytest.approx(0.75)


def test_a_file_without_anisotropic_records_has_a_fraction_of_zero() -> None:
    fields = read_refinement_fields(
        {"_atom_site.id": ["1", "2"], "_atom_site.type_symbol": ["N", "C"]}
    )

    assert fields.anisotropic_atom_fraction == 0.0


def test_a_record_id_on_two_rows_counts_for_no_atom() -> None:
    fields = read_refinement_fields(
        {
            "_atom_site.id": ["1", "2"],
            "_atom_site.type_symbol": ["N", "C"],
            "_atom_site_anisotrop.id": ["1", "1", "2"],
        }
    )

    assert fields.anisotropic_atom_fraction == pytest.approx(0.5)


def test_an_atom_without_an_element_symbol_is_not_taken_for_a_hydrogen() -> None:
    no_symbols = read_refinement_fields(
        {"_atom_site.id": ["1", "2"], "_atom_site_anisotrop.id": ["1"]}
    )
    one_missing = read_refinement_fields(
        {
            "_atom_site.id": ["1", "2"],
            "_atom_site.type_symbol": [None, "H"],
            "_atom_site_anisotrop.id": ["1"],
        }
    )

    assert no_symbols.anisotropic_atom_fraction == pytest.approx(0.5)
    assert one_missing.anisotropic_atom_fraction == 1.0


def test_a_file_of_hydrogens_alone_has_no_anisotropic_fraction() -> None:
    fields = read_refinement_fields(
        {"_atom_site.id": ["1"], "_atom_site.type_symbol": ["H"]}
    )

    assert fields.anisotropic_atom_fraction is None
