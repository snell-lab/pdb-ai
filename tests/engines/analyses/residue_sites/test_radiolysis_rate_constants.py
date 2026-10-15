from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pdbai.engines.analyses.metal_sites.constants import STANDARD_AMINO_ACIDS
from pdbai.engines.analyses.residue_sites import RESIDUE_SITE_ENGINE, reference
from pdbai.engines.analyses.residue_sites.reference import (
    RATE_CONSTANTS_PATH,
    REFERENCE_MANIFEST_PATH,
    radiolysis_rate_constants,
    reference_receipts,
    selected_rate_constants,
)


def test_every_amino_acid_has_a_selected_hydroxyl_radical_value() -> None:
    selected = selected_rate_constants()

    hydroxyl = selected[selected["radical"] == "OH"]
    assert set(hydroxyl["residue"]) == set(STANDARD_AMINO_ACIDS)
    assert not selected.duplicated(subset=["residue", "radical"]).any()
    # The source has no entry for the hydrated electron with Gln or Ile.
    electron = selected[selected["radical"] == "e_aq"]
    assert set(STANDARD_AMINO_ACIDS) - set(electron["residue"]) == {"GLN", "ILE"}
    assert not selected["form"].str.startswith("cystine").any()


def test_the_selected_values_are_those_the_source_prints() -> None:
    selected = selected_rate_constants().set_index(["residue", "radical"])

    assert selected.loc[("TRP", "OH"), "k_M_per_s"] == pytest.approx(1.3e10)
    assert selected.loc[("PHE", "OH"), "k_M_per_s"] == pytest.approx(6.5e9)
    assert selected.loc[("HIS", "e_aq"), "k_M_per_s"] == pytest.approx(6.4e7)
    assert selected.loc[("CYS", "e_aq"), "k_M_per_s"] == pytest.approx(1.2e10)
    assert selected.loc[("VAL", "e_aq"), "k_qualifier"] == "<"
    assert bool(selected.loc[("TRP", "OH"), "is_average"]) is True


def test_every_row_is_a_positive_value_with_its_source() -> None:
    table = radiolysis_rate_constants()

    assert (table["k_M_per_s"] > 0).all()
    assert table["source_location"].str.startswith("Table ").all()
    assert set(table["radical"]) == {"OH", "e_aq"}


def test_the_manifest_states_the_digest_of_the_table() -> None:
    manifest = json.loads(REFERENCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    entry = manifest["radiolysis_rate_constants"]

    digest = hashlib.sha256(RATE_CONSTANTS_PATH.read_bytes()).hexdigest()
    assert entry["sha256"] == digest
    assert entry["row_count"] == len(radiolysis_rate_constants())
    assert entry["selected_count"] == len(selected_rate_constants())
    receipt = reference_receipts()["residue_sites_radiolysis_rate_constants"]
    assert isinstance(receipt, dict)
    assert receipt["hash_matches_manifest"] is True


def test_the_engine_records_a_receipt_of_the_table() -> None:
    hook = RESIDUE_SITE_ENGINE.spec.reference_data

    assert hook is not None
    assert set(hook()) == {"residue_sites_radiolysis_rate_constants"}


def test_a_damaged_table_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    damaged = tmp_path / "radiolysis_rate_constants.csv"
    lines = RATE_CONSTANTS_PATH.read_text(encoding="utf-8").splitlines()
    damaged.write_text(
        "\n".join([lines[0], lines[1].replace("GLY,OH,", "XYZ,OH,", 1)]) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(reference, "RATE_CONSTANTS_PATH", damaged)
    reference._read_rate_constants.cache_clear()  # pyright: ignore[reportPrivateUsage]
    try:
        with pytest.raises(ValueError, match="not a standard amino acid"):
            radiolysis_rate_constants()
    finally:
        reference._read_rate_constants.cache_clear()  # pyright: ignore[reportPrivateUsage]
