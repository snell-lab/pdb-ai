from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from pdbai.engines.analyses.rabdam import reference
from pdbai.engines.analyses.rabdam.reference import (
    REFERENCE_MANIFEST_PATH,
    REFERENCE_PATH,
    reference_receipts,
)


def test_bundled_reference_matches_its_provenance_manifest() -> None:
    manifest = cast(
        dict[str, object],
        json.loads(REFERENCE_MANIFEST_PATH.read_text(encoding="utf-8")),
    )

    assert manifest["artifact"] == REFERENCE_PATH.name
    assert hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest() == manifest["sha256"]

    with REFERENCE_PATH.open(encoding="utf-8", newline="") as handle:
        # The header row is not an entry.
        entry_count = sum(1 for _ in csv.reader(handle)) - 1
    assert entry_count == manifest["entry_count"]

    source = cast(dict[str, object], manifest["source"])
    assert source == {
        "path": "database/reference/database.csv",
        "project": "rabdam-custom",
        "project_version": "1.0.0",
        "ref": "main",
        "repository": "https://github.com/snell-lab/rabdam-custom",
    }


def test_receipt_confirms_the_bundled_table_against_its_manifest() -> None:
    receipt = cast(dict[str, object], reference_receipts()["rabdam_bnet"])

    assert receipt["available"] is True
    assert receipt["path"] == (
        "pdbai/engines/analyses/rabdam/rabdam_custom_bnet_reference.csv"
    )
    assert receipt["hash_matches_manifest"] is True
    assert receipt["database_id"] == "pdb_redo_8_20_202606_strict_cryo_bnet"
    assert receipt["entry_count"] == 124948
    source = cast(dict[str, object], receipt["source"])
    assert source["project_version"] == "1.0.0"


def test_receipt_flags_a_table_that_differs_from_its_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = tmp_path / REFERENCE_PATH.name
    table.write_bytes(REFERENCE_PATH.read_bytes() + b"124948,9ZZZ,2.0,1.0\r\n")
    manifest = tmp_path / REFERENCE_MANIFEST_PATH.name
    manifest.write_bytes(REFERENCE_MANIFEST_PATH.read_bytes())
    monkeypatch.setattr(reference, "REFERENCE_PATH", table)
    monkeypatch.setattr(reference, "REFERENCE_MANIFEST_PATH", manifest)

    receipt = cast(dict[str, object], reference.reference_receipts()["rabdam_bnet"])

    assert receipt["available"] is True
    assert receipt["hash_matches_manifest"] is False
    assert (
        receipt["declared_sha256"]
        == hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest()
    )


def test_receipt_without_a_manifest_declares_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reference, "REFERENCE_MANIFEST_PATH", tmp_path / "none.json")

    receipt = cast(dict[str, object], reference.reference_receipts()["rabdam_bnet"])

    assert receipt["available"] is True
    assert receipt["declared_sha256"] is None
    assert receipt["database_id"] is None
    assert receipt["hash_matches_manifest"] is None
