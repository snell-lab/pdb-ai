from __future__ import annotations

from typing import cast

import pytest
import requests

from pdbai.engines.sources.alphafold import fetch as coordinator_module
from pdbai.engines.sources.alphafold.fetch import (
    fetch_afdb_entry,
    unavailable_upload_result,
)
from pdbai.engines.sources.alphafold.models import (
    AFDB_STATUS_AVAILABLE,
    AFDB_STATUS_LOOKUP_FAILED,
    AFDB_STATUS_MAPPING_ONLY,
    AFDB_STATUS_MODEL_NOT_FOUND,
    AFDB_STATUS_NO_MODEL_AVAILABLE,
    AFDB_STATUS_NO_UNIPROT_MAPPING,
    AFDB_STATUS_NOT_AVAILABLE_FOR_UPLOAD,
    AFDB_STATUS_PARSE_FAILED,
    AFDB_STATUS_PARTIAL,
    AFDBEntityReference,
    AFDBEntryResult,
    AFDBModelResult,
    AFDBReferenceDiscovery,
)


def _reference(accession: str, entity_id: str = "1") -> AFDBEntityReference:
    return AFDBEntityReference(
        entity_id=entity_id,
        uniprot_id=accession,
    )


def _model(accession: str, status: str, *, scores: bool = True) -> AFDBModelResult:
    return AFDBModelResult(
        accession, status, plddt_by_position={"1": 90.0} if scores else {}
    )


def _entry(
    monkeypatch: pytest.MonkeyPatch,
    discovery: AFDBReferenceDiscovery,
    models: dict[str, AFDBModelResult],
    *,
    fetch_models: bool = True,
) -> tuple[AFDBEntryResult, list[str]]:

    queried: list[str] = []

    def fake_discover(*args: object, **kwargs: object) -> AFDBReferenceDiscovery:
        del args, kwargs
        return discovery

    def fake_fetch_model(accession: str, **kwargs: object) -> AFDBModelResult:
        del kwargs
        queried.append(accession)
        return models[accession]

    monkeypatch.setattr(coordinator_module, "discover_entry_references", fake_discover)
    monkeypatch.setattr(coordinator_module, "fetch_afdb_model", fake_fetch_model)
    result = fetch_afdb_entry(
        "1ABC",
        session=cast(requests.Session, None),
        timeout=1.0,
        fetch_models=fetch_models,
    )
    return result, queried


def _mapped(*accessions: str) -> AFDBReferenceDiscovery:
    return AFDBReferenceDiscovery(
        references=[_reference(accession) for accession in accessions],
        protein_entity_ids=["1"],
    )


def test_every_model_available_reports_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _entry(
        monkeypatch,
        _mapped("P1", "P2"),
        {
            "P1": _model("P1", AFDB_STATUS_AVAILABLE),
            "P2": _model("P2", AFDB_STATUS_AVAILABLE),
        },
    )

    assert result.status == AFDB_STATUS_AVAILABLE
    assert result.failure_reason is None
    assert result.discovery_complete


def test_one_model_available_reports_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _entry(
        monkeypatch,
        _mapped("P1", "P2"),
        {
            "P1": _model("P1", AFDB_STATUS_AVAILABLE),
            "P2": _model("P2", AFDB_STATUS_MODEL_NOT_FOUND, scores=False),
        },
    )

    assert result.status == AFDB_STATUS_PARTIAL
    assert result.failure_reason is not None
    assert "P2" in result.failure_reason


def test_a_unanimous_negative_status_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    result, _ = _entry(
        monkeypatch,
        _mapped("P1", "P2"),
        {
            "P1": _model("P1", AFDB_STATUS_MODEL_NOT_FOUND, scores=False),
            "P2": _model("P2", AFDB_STATUS_MODEL_NOT_FOUND, scores=False),
        },
    )

    assert result.status == AFDB_STATUS_MODEL_NOT_FOUND


def test_disagreeing_negative_statuses_collapse_to_no_model_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    result, _ = _entry(
        monkeypatch,
        _mapped("P1", "P2"),
        {
            "P1": _model("P1", AFDB_STATUS_MODEL_NOT_FOUND, scores=False),
            "P2": _model("P2", AFDB_STATUS_PARSE_FAILED, scores=False),
        },
    )

    assert result.status == AFDB_STATUS_NO_MODEL_AVAILABLE


def test_complete_discovery_without_mappings_reports_no_uniprot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, queried = _entry(
        monkeypatch, AFDBReferenceDiscovery(protein_entity_ids=["1"]), {}
    )

    assert result.status == AFDB_STATUS_NO_UNIPROT_MAPPING
    assert queried == [], "no model lookup without an accession to look up"


def test_failed_discovery_reports_lookup_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    result, _ = _entry(
        monkeypatch,
        AFDBReferenceDiscovery(
            failure_reason="RCSB lookup failed", failed_entity_ids=["1"]
        ),
        {},
    )

    assert result.status == AFDB_STATUS_LOOKUP_FAILED
    assert not result.discovery_complete
    assert result.failure_reason is not None


def test_mapping_only_run_skips_model_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, queried = _entry(monkeypatch, _mapped("P1"), {}, fetch_models=False)

    assert result.status == AFDB_STATUS_MAPPING_ONLY
    assert queried == []
    assert [r.uniprot_id for r in result.references] == ["P1"]


def test_uploads_report_that_discovery_was_never_attempted() -> None:
    result = unavailable_upload_result("1ABC")

    assert result.status == AFDB_STATUS_NOT_AVAILABLE_FOR_UPLOAD
    assert result.references == []
    assert result.failure_reason is not None


def test_available_models_cannot_claim_available_when_discovery_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    result, _ = _entry(
        monkeypatch,
        AFDBReferenceDiscovery(references=[_reference("P1")], failed_entity_ids=["2"]),
        {"P1": _model("P1", AFDB_STATUS_AVAILABLE)},
    )

    assert result.status == AFDB_STATUS_PARTIAL
    assert not result.discovery_complete


def test_available_models_cannot_claim_available_beside_an_unmapped_entity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _entry(
        monkeypatch,
        AFDBReferenceDiscovery(
            references=[_reference("P1")],
            protein_entity_ids=["1", "2"],
            unmapped_protein_entity_ids=["2"],
        ),
        {"P1": _model("P1", AFDB_STATUS_AVAILABLE)},
    )

    assert result.status == AFDB_STATUS_PARTIAL
    assert result.failure_reason is not None
    assert "Protein entities without a UniProt mapping: 2" in result.failure_reason


def test_a_unanimous_negative_is_not_preserved_beside_an_unmapped_entity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _entry(
        monkeypatch,
        AFDBReferenceDiscovery(
            references=[_reference("P1")],
            protein_entity_ids=["1", "2"],
            unmapped_protein_entity_ids=["2"],
        ),
        {"P1": _model("P1", AFDB_STATUS_MODEL_NOT_FOUND, scores=False)},
    )

    assert result.status == AFDB_STATUS_NO_MODEL_AVAILABLE


def test_a_repeated_accession_is_fetched_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = AFDBReferenceDiscovery(
        references=[_reference("P1", "1"), _reference("P1", "2")],
        protein_entity_ids=["1", "2"],
    )

    result, queried = _entry(
        monkeypatch, discovery, {"P1": _model("P1", AFDB_STATUS_AVAILABLE)}
    )

    assert queried == ["P1"]
    assert result.status == AFDB_STATUS_AVAILABLE
    assert len(result.references) == 2


def test_failure_reasons_from_every_source_are_reported_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = AFDBReferenceDiscovery(
        references=[_reference("P1")],
        protein_entity_ids=["1", "2"],
        unmapped_protein_entity_ids=["2"],
        failed_entity_ids=["3"],
        failure_reason="entity 3: boom",
    )
    failing = AFDBModelResult(
        "P1",
        AFDB_STATUS_PARSE_FAILED,
        failure_reason="no parseable pLDDT values",
    )

    result, _ = _entry(monkeypatch, discovery, {"P1": failing})

    assert result.failure_reason is not None
    assert "entity 3: boom" in result.failure_reason
    assert "Protein entities without a UniProt mapping: 2" in result.failure_reason
    assert "no parseable pLDDT values" in result.failure_reason


def test_an_available_model_without_scores_reports_no_model_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Available means every model is complete, and a model without scores is
    # not, whatever its own status says.
    result, _ = _entry(
        monkeypatch,
        _mapped("P1"),
        {"P1": _model("P1", AFDB_STATUS_AVAILABLE, scores=False)},
    )

    assert result.status == AFDB_STATUS_NO_MODEL_AVAILABLE
    assert result.failure_reason == "P1: no pLDDT scores"


def test_a_partial_model_without_scores_reports_no_model_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _entry(
        monkeypatch,
        _mapped("P1"),
        {"P1": _model("P1", AFDB_STATUS_PARTIAL, scores=False)},
    )

    assert result.status == AFDB_STATUS_NO_MODEL_AVAILABLE
    assert result.failure_reason == "P1: no pLDDT scores"
