from __future__ import annotations

from pathlib import Path

import pytest

from pdbai.orchestration.upload import identify_uploads, portable_upload_stem


def test_a_file_named_like_a_suffixed_id_keeps_it_and_repeats_skip_past_it(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "x_2.cif", tmp_path / "a" / "x.cif", tmp_path / "b" / "x.cif"]
    for path in paths:
        path.parent.mkdir(exist_ok=True)
        path.write_text("data_x\n", encoding="utf-8")

    uploads = identify_uploads(paths)

    assert [upload.pdb_id for upload in uploads] == ["X_2", "X", "X_3"]


def test_a_file_named_after_a_windows_device_gets_an_id_a_file_can_have(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "con.cif", tmp_path / "nul.cif.gz", tmp_path / "com1.x.cif"]
    for path in paths:
        path.write_text("data_x\n", encoding="utf-8")

    uploads = identify_uploads(paths)

    assert [upload.pdb_id for upload in uploads] == ["CON_", "NUL_", "COM1.X_"]


def test_an_upload_stem_keeps_its_case_and_follows_the_id_rule() -> None:
    assert portable_upload_stem("my file (2)") == "my_file_2"
    assert portable_upload_stem("nul") == "nul_"


def test_an_upload_stem_with_no_ascii_letter_or_digit_is_refused() -> None:
    with pytest.raises(ValueError, match="no ASCII letter or digit"):
        portable_upload_stem("中文")
