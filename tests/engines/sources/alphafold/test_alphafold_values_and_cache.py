from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fake_http import FakeResponse, FakeSession

from pdbai.engines.sources.alphafold import model_fetch as model_fetch_module
from pdbai.engines.sources.alphafold.annotation import is_amino_acid_name
from pdbai.engines.sources.alphafold.model_fetch import fetch_afdb_model
from pdbai.engines.sources.alphafold.models import (
    AFDB_STATUS_AVAILABLE,
    AFDBModelResult,
    model_cache_path,
    plddt_value,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (90.0, 90.0),
        ("90.0", 90.0),
        (" 90 ", 90.0),
        (0, 0.0),
        (100, 100.0),
        ("0", 0.0),
        ("100", 100.0),
    ],
)
def test_plddt_values_on_the_scale_are_accepted(value: object, expected: float) -> None:
    assert plddt_value(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        100.1,
        "-0.1",
        "100.1",
        float("nan"),
        float("inf"),
        float("-inf"),
        "nan",
        "inf",
        None,
        "",
        "?",
        ".",
        "abc",
        [1],
        {"a": 1},
    ],
)
def test_plddt_values_off_the_scale_or_unparseable_are_rejected(
    value: object,
) -> None:

    assert plddt_value(value) is None


def test_plddt_refuses_booleans() -> None:
    assert plddt_value(True) is None
    assert plddt_value(False) is None


@pytest.mark.parametrize(
    ("component", "expected"),
    [
        ("ALA", True),
        ("ala", True),
        (" MSE ", True),
        ("SEP", True),
        ("2MR", True),
        (" pca ", True),
        ("HOH", False),
        ("ZN", False),
        ("", False),
        ("   ", False),
        ("XYZ", False),
        ("NOTARESIDUE", False),
    ],
)
def test_amino_acid_classification(component: str, expected: bool) -> None:
    assert is_amino_acid_name(component) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("P12345", "P12345"),
        ("  P1  ", "P1"),
        ("P-1_2.3", "P-1_2.3"),
        ("P1 P2", "P1_P2"),
        ("a\\b", "a_b"),
        ("../../etc/passwd", ".._.._etc_passwd"),
        ("P1/../P2", "P1_.._P2"),
        ("Ünïcode", "_n_code"),
        ("", ""),
    ],
)
def test_cache_file_names_are_sanitized(
    tmp_path: Path, value: str, expected: str
) -> None:
    assert model_cache_path(tmp_path, value) == (
        tmp_path / "models" / f"{expected.lower()}.json"
    )


@pytest.mark.parametrize(
    "accession",
    ["../../etc/passwd", "P1/../P2", "..", "/absolute", "a\\..\\b", "  ../x  "],
)
def test_a_hostile_accession_cannot_escape_the_cache_directory(
    tmp_path: Path, accession: str
) -> None:

    path = model_cache_path(tmp_path, accession)

    assert path.resolve().parent == (tmp_path / "models").resolve()


def test_a_cached_model_keeps_the_names_of_scored_residues_only() -> None:
    model = AFDBModelResult.from_dict(
        {
            "uniprot_id": "P12345",
            "status": AFDB_STATUS_AVAILABLE,
            "plddt_by_position": {"1": 90.0, "2": "bad"},
            "residue_name_by_position": {"1": " ala ", "2": "GLY", "3": "SER"},
        }
    )

    assert model.plddt_by_position == {"1": 90.0}
    assert model.residue_name_by_position == {"1": "ALA"}


# Only the cache holds the score 12.5. Seeing it in a result shows that the
# cache was used.
_CACHED_MODEL: dict[str, object] = {
    "uniprot_id": "P12345",
    "status": AFDB_STATUS_AVAILABLE,
    "plddt_by_position": {"1": 12.5},
    "residue_name_by_position": {"1": "ALA"},
}


def _write_model_cache(cache_directory: Path, model: dict[str, object]) -> None:
    path = model_cache_path(cache_directory, "P12345")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"cached_at": time.time(), "model": model}),
        encoding="utf-8",
    )


def _fetch_fresh_model(
    monkeypatch: pytest.MonkeyPatch, cache_directory: Path
) -> tuple[AFDBModelResult, FakeSession]:
    prediction = {
        "uniprotAccession": "P12345",
        "entryId": "AF-P12345-F1",
        "cifUrl": "https://example.test/AF-P12345-F1-model_v6.cif",
    }
    session = FakeSession(FakeResponse([prediction]), FakeResponse(text="data_x"))

    def fake_parse(_text: str) -> tuple[dict[str, float], dict[str, str]]:
        return {"1": 80.0}, {"1": "MET"}

    monkeypatch.setattr(model_fetch_module, "parse_afdb_model_cif", fake_parse)
    result = fetch_afdb_model(
        "P12345",
        session=session.as_session(),
        timeout=1.0,
        cache_directory=cache_directory,
    )
    return result, session


def test_a_cached_model_is_served_without_a_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_model_cache(tmp_path, _CACHED_MODEL)

    result, session = _fetch_fresh_model(monkeypatch, tmp_path)

    assert result.plddt_by_position == {"1": 12.5}
    assert result.cached
    assert session.calls == []


def test_a_cached_model_of_another_accession_is_fetched_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_model_cache(tmp_path, {**_CACHED_MODEL, "uniprot_id": "Q99999"})

    result, session = _fetch_fresh_model(monkeypatch, tmp_path)

    assert result.plddt_by_position == {"1": 80.0}
    assert len(session.calls) == 2


def test_a_complete_model_is_cached_with_its_residue_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, _ = _fetch_fresh_model(monkeypatch, tmp_path)

    written = json.loads(
        model_cache_path(tmp_path, "P12345").read_text(encoding="utf-8")
    )

    assert not result.cached
    assert written["model"]["residue_name_by_position"] == {"1": "MET"}
    assert "cached" not in written["model"]
