"""Which UniProt sequences each polymer entity of a PDB entry aligns to.

discover_entry_references reads the list of polymer entities from the RCSB
entry record. It then asks the RCSB polymer-entity endpoint about each
entity. Each alignment to UniProt becomes a reference with aligned regions,
and the regions are what let a residue be mapped. Some entities have a
UniProt accession but no alignment. Such an entity gets a reference without
regions. That is enough to fetch and report a model, but maps no residue.

An entity counts as a protein when it has a UniProt reference, or when RCSB
calls its polymer a protein or a polypeptide. A protein without a reference
is recorded as unmapped. An entity that RCSB could not describe is recorded
as failed. A caller can then tell a protein with no UniProt sequence from
one that nobody could look up.

The same entity record names the sequence clusters RCSB places the entity
in, one at each identity level, and those are kept for every entity that
has them, protein or not.

A discovery is cached only when RCSB answered for every entity, at least
one reference was found, and no protein is unmapped. Any other discovery is
fetched again next time, because RCSB adds mappings after an entry is
released and a network failure may not recur.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast
from urllib.parse import quote

import requests

from ....identifiers import normalize_pdb_id
from ....json_values import object_dict, object_list, optional_object_dict, str_list
from ..http_cache import read_envelope, write_envelope
from ..rcsb.metadata import rcsb_entry_url
from ..scalars import to_float, to_int
from .models import (
    SEQUENCE_IDENTITY_LEVELS,
    AFDBAlignedRegion,
    AFDBEntityReference,
    AFDBReferenceDiscovery,
    SequenceClusters,
    discovery_cache_path,
)

RCSB_POLYMER_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/"


def parse_entity_references(
    entity_json: Mapping[str, object],
) -> list[AFDBEntityReference]:
    """Return the UniProt references in one RCSB polymer-entity record.

    Alignments to other databases are ignored. So are aligned regions that
    lack a start or have no positive length. An accession that the record
    names without an alignment becomes a reference only when the entity has
    no UniProt alignment at all. Such an accession is taken from the
    record's reference sequence identifiers, or from its bare list of
    UniProt accessions when those name none.
    """
    container = object_dict(
        entity_json.get("rcsb_polymer_entity_container_identifiers")
    )
    entity_id = str(container.get("entity_id") or "")
    if not entity_id:
        rcsb_id = str(entity_json.get("rcsb_id") or "")
        entity_id = rcsb_id.rsplit("_", 1)[-1]

    references: list[AFDBEntityReference] = []
    for raw_alignment in object_list(entity_json.get("rcsb_polymer_entity_align")):
        alignment = optional_object_dict(raw_alignment)
        if alignment is None:
            continue
        database_name = str(alignment.get("reference_database_name") or "")
        if "uniprot" not in database_name.lower():
            continue
        accession = _isoform_or_accession(
            alignment.get("reference_database_accession"),
            alignment.get("reference_database_isoform"),
        )
        if not accession:
            continue
        regions: list[AFDBAlignedRegion] = []
        for raw_region in object_list(alignment.get("aligned_regions")):
            region = optional_object_dict(raw_region)
            if region is None:
                continue
            entity_beg = to_int(region.get("entity_beg_seq_id"))
            reference_beg = to_int(region.get("ref_beg_seq_id"))
            length = to_int(region.get("length"))
            if (
                entity_beg is None
                or reference_beg is None
                or length is None
                or length <= 0
            ):
                continue
            regions.append(AFDBAlignedRegion(entity_beg, reference_beg, length))
        references.append(AFDBEntityReference(entity_id, accession, tuple(regions)))

    if not references:
        references = [
            AFDBEntityReference(entity_id, accession)
            for accession in _identifier_accessions(container)
        ]
    return references


def _isoform_or_accession(accession: object, isoform: object) -> str:
    """Return the isoform when one is named, else the accession, or ""."""
    return str(isoform or "").strip() or str(accession or "").strip()


def _identifier_accessions(container: Mapping[str, object]) -> list[str]:
    """Return the accessions an entity's container names without alignments.

    The UniProt entries of reference_sequence_identifiers are read first.
    The bare uniprot_ids list is read only when they name no accession.
    """
    accessions: list[str] = []
    for raw_identifier in object_list(container.get("reference_sequence_identifiers")):
        identifier = optional_object_dict(raw_identifier)
        if identifier is None:
            continue
        if "uniprot" not in str(identifier.get("database_name") or "").lower():
            continue
        accession = _isoform_or_accession(
            identifier.get("database_accession"), identifier.get("database_isoform")
        )
        if accession:
            accessions.append(accession)
    if accessions:
        return accessions
    return [
        accession.strip()
        for accession in str_list(container.get("uniprot_ids"))
        if accession.strip()
    ]


def parse_entity_clusters(entity_json: Mapping[str, object]) -> dict[str, str]:
    """Return the entity's sequence cluster at each identity level, by level.

    The levels are the ones SEQUENCE_IDENTITY_LEVELS names, written as text.
    Groups made any other way than by sequence identity are ignored, and so
    is a level the entity has no group at. The first group at a level is
    kept.
    """
    clusters: dict[str, str] = {}
    for raw_group in object_list(
        entity_json.get("rcsb_polymer_entity_group_membership")
    ):
        group = optional_object_dict(raw_group)
        if group is None:
            continue
        if str(group.get("aggregation_method") or "") != "sequence_identity":
            continue
        # RCSB writes the level as a float, such as 95.0.
        cutoff = to_float(group.get("similarity_cutoff"))
        group_id = str(group.get("group_id") or "").strip()
        if cutoff is None or not cutoff.is_integer() or not group_id:
            continue
        level = int(cutoff)
        if level in SEQUENCE_IDENTITY_LEVELS:
            clusters.setdefault(str(level), group_id)
    return clusters


def _stored_clusters(value: object) -> SequenceClusters | None:
    # Return the clusters a cache holds, or None when they are not a mapping
    # of entity identifiers to mappings of levels to cluster identifiers.
    stored = optional_object_dict(value)
    if stored is None:
        return None
    clusters: SequenceClusters = {}
    for entity_id, raw_levels in stored.items():
        levels = optional_object_dict(raw_levels)
        if levels is None or not all(
            isinstance(cluster, str) for cluster in levels.values()
        ):
            return None
        clusters[entity_id] = {
            level: cluster
            for level, cluster in levels.items()
            if isinstance(cluster, str)
        }
    return clusters


def entity_is_protein(entity_json: Mapping[str, object]) -> bool:
    """Return whether RCSB records the entity's polymer as a protein."""
    entity_poly = optional_object_dict(entity_json.get("entity_poly"))
    if entity_poly is None:
        return False
    polymer_types = (
        entity_poly.get("rcsb_entity_polymer_type"),
        entity_poly.get("type"),
    )
    normalized = " ".join(str(value).lower() for value in polymer_types if value)
    return "protein" in normalized or "polypeptide" in normalized


def _cached_discovery(cache_path: Path) -> AFDBReferenceDiscovery | None:
    """Return the discovery cached at ``cache_path``, or None to fetch again.

    A file with no reference, with a reference that lacks an entity or an
    accession, or with an aligned region that cannot be read counts as
    missing, since no discovery is ever cached that way. So does a file
    without the sequence clusters, which was written before they were kept.
    """
    cached = read_envelope(cache_path)
    if cached is None:
        return None
    try:
        references = [
            AFDBEntityReference.from_dict(item_mapping)
            for item in object_list(cached.get("references"))
            if (item_mapping := optional_object_dict(item)) is not None
        ]
    except (KeyError, TypeError, ValueError):
        return None
    if not references or not all(
        reference.entity_id and reference.uniprot_id for reference in references
    ):
        return None
    sequence_clusters = _stored_clusters(cached.get("sequence_clusters"))
    if sequence_clusters is None:
        return None
    # A discovery is cached only when no entity failed or was unmapped, so
    # there is none to restore.
    return AFDBReferenceDiscovery(
        references=references,
        protein_entity_ids=str_list(cached.get("protein_entity_ids")),
        sequence_clusters=sequence_clusters,
        cached=True,
    )


def polymer_entity_url(pdb_id: str, entity_id: str) -> str:
    """Return the RCSB Data API URL of one polymer entity of an entry.

    The entity identifier comes from a remote record, so it is quoted as one
    path segment rather than trusted to be one. Raise ValueError when
    ``pdb_id`` is not a PDB ID.
    """
    pdb_lower = normalize_pdb_id(pdb_id).lower()
    return f"{RCSB_POLYMER_ENTITY_URL}{pdb_lower}/{quote(entity_id, safe='')}"


def discover_entry_references(
    pdb_id: str,
    *,
    session: requests.Session,
    timeout: float,
    cache_directory: Path,
    entry_json: Mapping[str, object] | None = None,
) -> AFDBReferenceDiscovery:
    """Return the UniProt references of every polymer entity in a PDB entry.

    Pass ``entry_json``, the RCSB entry record, when the caller already has
    it. That saves fetching it again. The discovery is read from and
    written to the cache under ``cache_directory``, and says whether it was
    read. An invalid PDB ID raises ValueError. Any other failure is
    reported in the discovery's failure_reason and failed_entity_ids. An
    entry record without container identifiers is such a failure, while
    one whose identifiers list no polymer entity is a complete discovery
    with no reference.
    """
    normalized_id = normalize_pdb_id(pdb_id)
    pdb_lower = normalized_id.lower()
    cache_path = discovery_cache_path(cache_directory, pdb_lower)
    cached = _cached_discovery(cache_path)
    if cached is not None:
        return cached

    try:
        if entry_json is None:
            entry_response = session.get(rcsb_entry_url(pdb_lower), timeout=timeout)
            entry_response.raise_for_status()
            entry_json = optional_object_dict(cast(object, entry_response.json()))
            if entry_json is None:
                raise ValueError("RCSB entry endpoint returned a non-object payload")
        identifiers = optional_object_dict(
            entry_json.get("rcsb_entry_container_identifiers")
        )
        # Every entry record has container identifiers. RCSB leaves out an
        # empty list, so only their absence marks a malformed record.
        if identifiers is None:
            raise ValueError("RCSB entry record has no container identifiers")
        entity_ids = [
            str(entity_id)
            for entity_id in object_list(identifiers.get("polymer_entity_ids"))
        ]
        if not entity_ids:
            return AFDBReferenceDiscovery()

        references: list[AFDBEntityReference] = []
        protein_entity_ids: list[str] = []
        unmapped_protein_entity_ids: list[str] = []
        failed_entity_ids: list[str] = []
        entity_errors: list[str] = []
        sequence_clusters: SequenceClusters = {}
        for entity_id in entity_ids:
            try:
                response = session.get(
                    polymer_entity_url(pdb_lower, entity_id), timeout=timeout
                )
                response.raise_for_status()
                entity_json = optional_object_dict(cast(object, response.json()))
                if entity_json is None:
                    raise ValueError(
                        "RCSB polymer-entity endpoint returned a non-object payload"
                    )
                entity_references = parse_entity_references(entity_json)
                clusters = parse_entity_clusters(entity_json)
                if clusters:
                    sequence_clusters[entity_id] = clusters
                is_protein = bool(entity_references) or entity_is_protein(entity_json)
                if is_protein:
                    protein_entity_ids.append(entity_id)
                    if entity_references:
                        references.extend(entity_references)
                    else:
                        unmapped_protein_entity_ids.append(entity_id)
            except Exception as exception:
                failed_entity_ids.append(entity_id)
                entity_errors.append(f"entity {entity_id}: {exception}")

        if references and not unmapped_protein_entity_ids and not entity_errors:
            write_envelope(
                cache_path,
                {
                    "pdb_id": pdb_lower,
                    "references": [reference.to_dict() for reference in references],
                    "protein_entity_ids": protein_entity_ids,
                    "sequence_clusters": sequence_clusters,
                },
            )
        error = "; ".join(entity_errors) if entity_errors else None
        return AFDBReferenceDiscovery(
            references=references,
            protein_entity_ids=protein_entity_ids,
            unmapped_protein_entity_ids=unmapped_protein_entity_ids,
            failed_entity_ids=failed_entity_ids,
            failure_reason=error,
            sequence_clusters=sequence_clusters,
        )
    except Exception as exception:
        message = f"RCSB polymer-entity lookup failed for {normalized_id}: {exception}"
        return AFDBReferenceDiscovery(failure_reason=message)


__all__ = [
    "parse_entity_clusters",
    "RCSB_POLYMER_ENTITY_URL",
    "discover_entry_references",
    "entity_is_protein",
    "parse_entity_references",
    "polymer_entity_url",
]
