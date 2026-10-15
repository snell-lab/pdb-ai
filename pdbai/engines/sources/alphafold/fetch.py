"""One AlphaFold Database lookup for a PDB entry, summed up in one status.

fetch_afdb_entry finds the UniProt references of the entry. It then fetches the
model of each distinct accession once, and gives the entry the status that
the run report shows. An entry is complete only when RCSB answered for every
polymer entity and every protein entity has a UniProt sequence. An entry
with an unmapped protein is therefore partial at best, even when all of its
models are complete.

The failure reason joins every shortfall into one string, separated by
semicolons. It names the discovery error, the unmapped entities, and each
model that was not complete. The run report can then explain a partial
result as well as a failed one.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import requests

from .model_fetch import fetch_afdb_model
from .models import (
    AFDB_STATUS_AVAILABLE,
    AFDB_STATUS_LOOKUP_FAILED,
    AFDB_STATUS_MAPPING_ONLY,
    AFDB_STATUS_NO_MODEL_AVAILABLE,
    AFDB_STATUS_NO_UNIPROT_MAPPING,
    AFDB_STATUS_NOT_AVAILABLE_FOR_UPLOAD,
    AFDB_STATUS_PARTIAL,
    AFDBEntryResult,
    AFDBModelResult,
    combined_model_status,
)
from .references import discover_entry_references


def fetch_afdb_entry(
    pdb_id: str,
    *,
    session: requests.Session,
    timeout: float,
    cache_directory: Path | None = None,
    fetch_models: bool = True,
    entry_json: Mapping[str, object] | None = None,
) -> AFDBEntryResult:
    """Look up a PDB entry's UniProt references and their AlphaFold models.

    Pass ``entry_json``, the RCSB entry record, when the caller already has
    it. With ``fetch_models`` false, only the mapping is looked up, and a
    mapped entry gets the status mapping_only. An entry without references
    gets no_uniprot_mapping, or lookup_failed when a request to RCSB failed.
    An invalid PDB ID raises ValueError. Every other failure is reported in
    the result's status and failure reason.
    """
    discovery = discover_entry_references(
        pdb_id,
        session=session,
        timeout=timeout,
        cache_directory=cache_directory,
        entry_json=entry_json,
    )
    references = discovery.references
    discovery_error = discovery.failure_reason
    failure_parts = [discovery_error] if discovery_error else []
    if discovery.unmapped_protein_entity_ids:
        failure_parts.append(
            "Protein entities without a UniProt mapping: "
            + ", ".join(discovery.unmapped_protein_entity_ids)
        )
    if not references:
        return AFDBEntryResult(
            pdb_id=pdb_id,
            status=(
                AFDB_STATUS_LOOKUP_FAILED
                if discovery_error
                else AFDB_STATUS_NO_UNIPROT_MAPPING
            ),
            failure_reason="; ".join(failure_parts) if failure_parts else None,
            discovery_complete=discovery.complete,
            protein_entity_ids=discovery.protein_entity_ids,
            unmapped_protein_entity_ids=discovery.unmapped_protein_entity_ids,
            sequence_clusters=discovery.sequence_clusters,
        )

    models: dict[str, AFDBModelResult] = {}
    if fetch_models:
        for accession in dict.fromkeys(
            reference.uniprot_id for reference in references
        ):
            models[accession] = fetch_afdb_model(
                accession,
                session=session,
                timeout=timeout,
                cache_directory=cache_directory,
            )

    status = (
        combined_model_status(
            list(models.values()),
            complete=discovery.complete and not discovery.unmapped_protein_entity_ids,
            fallback=AFDB_STATUS_NO_MODEL_AVAILABLE,
        )
        if fetch_models
        else AFDB_STATUS_MAPPING_ONLY
    )
    for accession, model in models.items():
        if model.status == AFDB_STATUS_AVAILABLE and model.has_scores:
            continue
        # A status that claims scores the model does not have is not the
        # shortfall, so the missing scores are named instead.
        claims_scores = model.status in {AFDB_STATUS_AVAILABLE, AFDB_STATUS_PARTIAL}
        if claims_scores and not model.has_scores:
            detail = f"{accession}: no pLDDT scores"
        else:
            detail = f"{accession}: {model.status}"
        if model.failure_reason:
            detail += f" ({model.failure_reason})"
        failure_parts.append(detail)

    return AFDBEntryResult(
        pdb_id=pdb_id,
        references=references,
        models=models,
        status=status,
        failure_reason="; ".join(failure_parts) if failure_parts else None,
        discovery_complete=discovery.complete,
        protein_entity_ids=discovery.protein_entity_ids,
        unmapped_protein_entity_ids=discovery.unmapped_protein_entity_ids,
        sequence_clusters=discovery.sequence_clusters,
    )


def unavailable_upload_result(pdb_id: str) -> AFDBEntryResult:
    """Return the result that stands for an uploaded structure's lookup.

    An upload has no RCSB entry, so nothing is looked up. The status is
    not_available_for_upload, for the reason engine.py gives.
    """
    return AFDBEntryResult(
        pdb_id=pdb_id,
        status=AFDB_STATUS_NOT_AVAILABLE_FOR_UPLOAD,
        failure_reason=(
            "AFDB discovery was not attempted for a direct mmCIF upload because no "
            "RCSB polymer-entity mapping was fetched."
        ),
    )


__all__ = ["fetch_afdb_entry", "unavailable_upload_result"]
