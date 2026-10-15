"""Records, statuses, and cache paths shared by the AlphaFold modules.

The statuses are the values of afdb_status, in the residue table and in the
run report. available means that every model a lookup needed is complete.
partial means that at least one model has scores. mapping_only means that
the run asked for the UniProt mapping alone and fetched no model. The other
statuses say why a protein, an entity, or an entry was left without a model.
combined_model_status sums up the models of several accessions in one
status. It applies the same rule to an entry and to each of its entities.

Some records are cached, so they convert to and from plain JSON. Scores are
therefore keyed by position written as a string. AFDBModelResult.from_dict
checks every score again when it reads one back. A cache file is outside the
program's control, and a bad score must not reach a dataset.

An aligned region moves a position by one fixed offset, and a position
outside every region of a reference maps to nothing. A gap or an insertion
in the deposited sequence is therefore never bridged by guesswork.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from ....json_values import object_dict, object_list, optional_object_dict
from ..scalars import to_float, to_int

AFDB_STATUS_AVAILABLE = "available"
AFDB_STATUS_PARTIAL = "partial"
AFDB_STATUS_NO_MODEL_AVAILABLE = "no_model_available"
AFDB_STATUS_NO_UNIPROT_MAPPING = "no_uniprot_mapping"
AFDB_STATUS_MODEL_NOT_FOUND = "model_not_found"
AFDB_STATUS_LOOKUP_FAILED = "lookup_failed"
AFDB_STATUS_DOWNLOAD_FAILED = "download_failed"
AFDB_STATUS_PARSE_FAILED = "parse_failed"
AFDB_STATUS_NOT_AVAILABLE_FOR_UPLOAD = "not_available_for_upload"
AFDB_STATUS_MAPPING_ONLY = "mapping_only"


def _cache_component(value: str) -> str:
    """Return the value with each run of unsafe characters an underscore."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())


def discovery_cache_path(cache_directory: Path, pdb_id: str) -> Path:
    """Return where an entry's UniProt reference discovery is cached."""
    return cache_directory / "entities" / f"{_cache_component(pdb_id.lower())}.json"


def model_cache_path(cache_directory: Path, accession: str) -> Path:
    """Return where an accession's AlphaFold model scores are cached.

    ``accession`` is the one looked up, an isoform when RCSB named one.
    """
    return cache_directory / "models" / f"{_cache_component(accession.lower())}.json"


def plddt_value(value: object) -> float | None:
    """Return a pLDDT score, or None unless it is finite and from 0 to 100."""
    number = to_float(value)
    if number is None or not np.isfinite(number) or number < 0.0 or number > 100.0:
        return None
    return number


@dataclass(frozen=True)
class AFDBAlignedRegion:
    """A stretch of a polymer entity aligned residue for residue to UniProt.

    Positions count from 1. An entity position is a label_seq_id and a
    reference position is a UniProt residue number. ``length`` counts
    residues and is positive.
    """

    entity_beg_seq_id: int
    reference_beg_seq_id: int
    length: int

    def map_entity_position(self, entity_position: int) -> int | None:
        """Return the UniProt position, or None outside the region."""
        offset = entity_position - self.entity_beg_seq_id
        if offset < 0 or offset >= self.length:
            return None
        return self.reference_beg_seq_id + offset

    def to_dict(self) -> dict[str, int]:
        """Return the region as JSON-ready values for the cache."""
        return {
            "entity_beg_seq_id": self.entity_beg_seq_id,
            "reference_beg_seq_id": self.reference_beg_seq_id,
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AFDBAlignedRegion:
        """Rebuild a region from the values to_dict returned.

        A missing field raises KeyError, and a value int cannot read raises
        ValueError or TypeError.
        """
        return cls(
            entity_beg_seq_id=int(cast(Any, data["entity_beg_seq_id"])),
            reference_beg_seq_id=int(cast(Any, data["reference_beg_seq_id"])),
            length=int(cast(Any, data["length"])),
        )


@dataclass(frozen=True)
class AFDBEntityReference:
    """One UniProt sequence that a polymer entity of the entry aligns to.

    ``uniprot_id`` is the accession to look up in the AlphaFold Database,
    the isoform when RCSB names one. An entity built from segments of
    several proteins has one reference for each. ``aligned_regions`` is
    empty when RCSB named the accession without an alignment, and such a
    reference maps no residue.
    """

    entity_id: str
    uniprot_id: str
    aligned_regions: tuple[AFDBAlignedRegion, ...] = ()

    def map_entity_position(self, entity_position: int) -> int | None:
        """Return the UniProt position, or None outside every region."""
        for region in self.aligned_regions:
            mapped = region.map_entity_position(entity_position)
            if mapped is not None:
                return mapped
        return None

    def to_dict(self) -> dict[str, object]:
        """Return the reference as JSON-ready values for the cache."""
        return {
            "entity_id": self.entity_id,
            "uniprot_id": self.uniprot_id,
            "aligned_regions": [region.to_dict() for region in self.aligned_regions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AFDBEntityReference:
        """Rebuild a reference from the values to_dict returned.

        A missing identifier reads as an empty string. A malformed aligned
        region raises as AFDBAlignedRegion.from_dict does.
        """
        return cls(
            entity_id=str(data.get("entity_id") or ""),
            uniprot_id=str(data.get("uniprot_id") or ""),
            aligned_regions=tuple(
                AFDBAlignedRegion.from_dict(region_mapping)
                for region in object_list(data.get("aligned_regions"))
                if (region_mapping := optional_object_dict(region)) is not None
            ),
        )


@dataclass(frozen=True)
class AFDBModelResult:
    """The AlphaFold pLDDT scores of one UniProt accession and their models.

    ``plddt_by_position`` is keyed by UniProt residue number written as a
    string, and ``model_id_by_position`` names the model that supplied each
    score. ``residue_name_by_position`` holds the model's residue name at
    each scored position, upper case, where the model file gives one.
    ``model_ids`` and ``source_urls`` list every model that supplied a
    score, in the order the family was ranked. ``failure_reason`` explains
    each model of the family that supplied nothing.
    """

    uniprot_id: str
    status: str
    plddt_by_position: dict[str, float] = field(default_factory=dict[str, float])
    model_id_by_position: dict[str, str] = field(default_factory=dict[str, str])
    residue_name_by_position: dict[str, str] = field(default_factory=dict[str, str])
    model_ids: list[str] = field(default_factory=list[str])
    source_urls: list[str] = field(default_factory=list[str])
    failure_reason: str | None = None

    @property
    def has_scores(self) -> bool:
        """Return whether there are pLDDT scores to map, complete or not."""
        return self.status in {AFDB_STATUS_AVAILABLE, AFDB_STATUS_PARTIAL} and bool(
            self.plddt_by_position
        )

    def to_dict(self) -> dict[str, object]:
        """Return the result as JSON-ready values for the cache."""
        return {
            "uniprot_id": self.uniprot_id,
            "status": self.status,
            "model_ids": self.model_ids,
            "source_urls": self.source_urls,
            "failure_reason": self.failure_reason,
            "plddt_by_position": self.plddt_by_position,
            "model_id_by_position": self.model_id_by_position,
            "residue_name_by_position": self.residue_name_by_position,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AFDBModelResult:
        """Rebuild a result from the values to_dict returned.

        A score that is not a valid pLDDT, or sits at a position below 1, is
        dropped. So are the model identifier and the residue name of every
        position without a score. A missing status reads as parse_failed.
        """
        scores: dict[str, float] = {}
        for key, value in object_dict(data.get("plddt_by_position")).items():
            position = to_int(key)
            score = plddt_value(value)
            if position is not None and position > 0 and score is not None:
                scores[str(position)] = score
        model_id_by_position = {
            str(position): str(model_id)
            for position, model_id in object_dict(
                data.get("model_id_by_position")
            ).items()
            if str(position) in scores and model_id
        }
        residue_name_by_position = {
            str(position): name.strip().upper()
            for position, name in object_dict(
                data.get("residue_name_by_position")
            ).items()
            if str(position) in scores and isinstance(name, str) and name.strip()
        }
        return cls(
            uniprot_id=str(data.get("uniprot_id") or ""),
            status=str(data.get("status") or AFDB_STATUS_PARSE_FAILED),
            plddt_by_position=scores,
            model_id_by_position=model_id_by_position,
            residue_name_by_position=residue_name_by_position,
            model_ids=[
                str(value) for value in object_list(data.get("model_ids")) if value
            ],
            source_urls=[
                str(value) for value in object_list(data.get("source_urls")) if value
            ],
            failure_reason=(
                str(data["failure_reason"]) if data.get("failure_reason") else None
            ),
        )


def combined_model_status(
    models: Sequence[AFDBModelResult | None], *, complete: bool, fallback: str
) -> str:
    """Return the one status that sums up the models of several accessions.

    ``models`` holds the model of each accession, None where none was
    fetched. ``complete`` says whether every protein the models stand for
    was mapped. The status is available when ``complete`` is true and every
    model was fetched and is complete. It is partial when any model has
    scores. When every model failed the same way, that failure is the
    status, but only if ``complete`` is true. An unmapped protein did not
    fail that way. A model whose status is available or partial but which
    has no scores did not fail in a way its status names, so it never sets
    the status. In every other case the status is no_model_available, or
    ``fallback`` when no model was fetched at all.
    """
    known = [model for model in models if model is not None]
    if (
        complete
        and known
        and len(known) == len(models)
        and all(
            model.status == AFDB_STATUS_AVAILABLE and model.has_scores
            for model in known
        )
    ):
        return AFDB_STATUS_AVAILABLE
    if any(model.has_scores for model in known):
        return AFDB_STATUS_PARTIAL
    statuses = {model.status for model in known}
    if (
        complete
        and len(statuses) == 1
        and statuses.isdisjoint({AFDB_STATUS_AVAILABLE, AFDB_STATUS_PARTIAL})
    ):
        return next(iter(statuses))
    return AFDB_STATUS_NO_MODEL_AVAILABLE if known else fallback


# The sequence-identity levels at which RCSB clusters polymer entities, as
# percentages. Each gives a column, from sequence_cluster_30 to
# sequence_cluster_100.
SEQUENCE_IDENTITY_LEVELS: tuple[int, ...] = (30, 50, 70, 90, 95, 100)

# The cluster memberships of an entry's polymer entities, keyed by entity
# identifier, then by identity level written as text, holding the cluster
# identifier RCSB gives, such as 391_95.
type SequenceClusters = dict[str, dict[str, str]]


def sequence_cluster_column(level: int) -> str:
    """Return the column that holds the cluster at an identity level."""
    return f"sequence_cluster_{level}"


@dataclass(frozen=True)
class AFDBEntryResult:
    """Everything the AlphaFold lookup found for one PDB entry.

    ``models`` is keyed by the accession each reference looks up, and is
    empty when the run asked for the mapping alone. ``discovery_complete``
    is false when RCSB could not be asked about the entry or one of its
    polymer entities. ``protein_entity_ids`` lists the entities that are
    proteins, and ``unmapped_protein_entity_ids`` those of them with no
    UniProt sequence. ``sequence_clusters`` holds the sequence clusters of
    every polymer entity RCSB answered for, proteins or not.
    """

    pdb_id: str
    references: list[AFDBEntityReference] = field(
        default_factory=list[AFDBEntityReference]
    )
    models: dict[str, AFDBModelResult] = field(
        default_factory=dict[str, AFDBModelResult]
    )
    status: str = AFDB_STATUS_NO_UNIPROT_MAPPING
    failure_reason: str | None = None
    discovery_complete: bool = True
    protein_entity_ids: list[str] = field(default_factory=list[str])
    unmapped_protein_entity_ids: list[str] = field(default_factory=list[str])
    sequence_clusters: SequenceClusters = field(
        default_factory=dict[str, dict[str, str]]
    )


@dataclass(frozen=True)
class AFDBReferenceDiscovery:
    """What RCSB said about the UniProt sequences of an entry's entities.

    ``failed_entity_ids`` lists the entities RCSB could not be asked about.
    ``failure_reason`` joins their errors, or holds the one error that
    stopped discovery for the whole entry. ``sequence_clusters`` holds the
    sequence clusters of every entity RCSB answered for.
    """

    references: list[AFDBEntityReference] = field(
        default_factory=list[AFDBEntityReference]
    )
    protein_entity_ids: list[str] = field(default_factory=list[str])
    unmapped_protein_entity_ids: list[str] = field(default_factory=list[str])
    failed_entity_ids: list[str] = field(default_factory=list[str])
    failure_reason: str | None = None
    sequence_clusters: SequenceClusters = field(
        default_factory=dict[str, dict[str, str]]
    )

    @property
    def complete(self) -> bool:
        """Return whether RCSB answered for the entry and every entity."""
        return not self.failed_entity_ids and not self.failure_reason


__all__ = [
    "AFDB_STATUS_AVAILABLE",
    "AFDB_STATUS_DOWNLOAD_FAILED",
    "AFDB_STATUS_LOOKUP_FAILED",
    "AFDB_STATUS_MAPPING_ONLY",
    "AFDB_STATUS_MODEL_NOT_FOUND",
    "AFDB_STATUS_NOT_AVAILABLE_FOR_UPLOAD",
    "AFDB_STATUS_NO_MODEL_AVAILABLE",
    "AFDB_STATUS_NO_UNIPROT_MAPPING",
    "AFDB_STATUS_PARSE_FAILED",
    "AFDB_STATUS_PARTIAL",
    "AFDBAlignedRegion",
    "AFDBEntityReference",
    "AFDBEntryResult",
    "AFDBModelResult",
    "AFDBReferenceDiscovery",
    "SEQUENCE_IDENTITY_LEVELS",
    "SequenceClusters",
    "combined_model_status",
    "discovery_cache_path",
    "model_cache_path",
    "plddt_value",
    "sequence_cluster_column",
]
