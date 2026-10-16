"""AlphaFold Database confidence mapped onto the atoms of a PDB entry.

annotate_atom_dataframe gives every atom the AlphaFold columns of its
residue. A residue is found by its entity and its label_seq_id, which are
what RCSB aligns to UniProt. Author numbering and insertion codes therefore
never matter, and equal numbers in different entities never collide. Nor
does it matter whether the table's chain_id and res_num hold author or label
values, since neither is read. An
aligned region of the entity's reference turns the position into a UniProt
position. The model of that accession then gives the pLDDT and names the
model file it came from.

An atom takes part when it belongs to a polymer, which means it has a
label_seq_id, a whole number from 1. The record type does not matter, so a
selenomethionine written as HETATM is mapped like any other residue. A
polymer atom belongs to a protein when its entity is known to be one or its
residue is an amino acid. Atoms outside a protein are left empty. Only
amino acids count as residues. Capping groups such as ACE and NH2 have a
label_seq_id of their own inside a protein entity. They never get a pLDDT,
so they are left out of the residue count.

A residue that two references map to different accessions or different
UniProt positions is left unmapped. RCSB lists references in no meaningful
order, and picking one could put another protein's confidence on the
residue. An entity with a single reference and no aligned region still gets
that accession in uniprot_id, but none of its residues gets a score.

A score also needs the model to have the deposited amino acid at that
position. Names that differ still match when both stand for the same
standard amino acid, as MSE stands for MET. pLDDT is the confidence of the
model's own residue, so an engineered mutation gets no score. Neither does a
residue whose UniProt position the model numbers differently, as when the
model was built on another version of the sequence, nor one the model or
the table does not name.

The columns keep apart the reasons a score can be missing. afdb_status gives
the lookup outcome for the residue's entity. A residue mapped to one UniProt
position gets the status of that accession's model instead.
afdb_residue_has_plddt is true for a scored residue, false for an unscored
residue, and missing for every other atom. afdb_plddt_coverage is the
share of an entity's distinct residues that were scored. It is missing for
an entity without a reference. Such an entity is given no_uniprot_mapping
when RCSB answered for it, and lookup_failed when it may be one RCSB could
not be asked about.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import gemmi
import numpy as np
import pandas as pd

from .models import (
    AFDB_STATUS_LOOKUP_FAILED,
    AFDB_STATUS_NO_UNIPROT_MAPPING,
    SEQUENCE_IDENTITY_LEVELS,
    AFDBEntityReference,
    AFDBEntryResult,
    combined_model_status,
    sequence_cluster_column,
)

# Parents from the wwPDB Chemical Component Dictionary
# (mon_nstd_parent_comp_id) for the modified amino acids Gemmi's table gets
# wrong or lacks. Gemmi gives pyroglutamate (PCA) the code of glutamate, but
# it forms from an N-terminal glutamine, and the others it does not list.
_DICTIONARY_PARENT_CODES = {
    "PCA": "Q",
    "2MR": "R",
    "DM0": "K",
    "GLZ": "G",
}


@lru_cache(maxsize=1024)
def is_amino_acid_name(value: str) -> bool:
    """Return whether the name is an amino acid.

    Gemmi's residue table decides, together with the modified amino acids
    in _DICTIONARY_PARENT_CODES. Case and surrounding whitespace are
    ignored. A name neither knows, the empty name included, is not an amino
    acid.
    """
    name = value.strip().upper()
    if name in _DICTIONARY_PARENT_CODES:
        return True
    return gemmi.find_tabulated_residue(name).is_amino_acid()


@lru_cache(maxsize=1024)
def _standard_amino_acid_code(name: str) -> str | None:
    """Return the one-letter code of the standard amino acid a name stands for.

    A modified amino acid gives its parent's code. None stands for a name
    that is not an amino acid, or whose parent neither Gemmi's table nor
    _DICTIONARY_PARENT_CODES knows.
    """
    if name in _DICTIONARY_PARENT_CODES:
        return _DICTIONARY_PARENT_CODES[name]
    residue = gemmi.find_tabulated_residue(name)
    if not residue.is_amino_acid():
        return None
    code = residue.one_letter_code.upper()
    return code if code.isalpha() and code != "X" else None


def residue_names_match(deposited: str, modelled: str) -> bool:
    """Return whether two residue names stand for the same amino acid.

    Case and surrounding whitespace are ignored. Equal names match, and so
    do two amino acids whose standard parent is the same, such as MSE and
    MET. An empty name matches nothing.
    """
    deposited_name = deposited.strip().upper()
    modelled_name = modelled.strip().upper()
    if not deposited_name or not modelled_name:
        return False
    if deposited_name == modelled_name:
        return True
    code = _standard_amino_acid_code(deposited_name)
    return code is not None and code == _standard_amino_acid_code(modelled_name)


def group_references_by_entity(
    entry_result: AFDBEntryResult,
) -> dict[str, list[AFDBEntityReference]]:
    """Return the entry's references grouped by entity identifier."""
    grouped: dict[str, list[AFDBEntityReference]] = {}
    for reference in entry_result.references:
        grouped.setdefault(reference.entity_id, []).append(reference)
    return grouped


def _assign_by_position(
    afdb_columns: pd.DataFrame,
    column: str,
    entity_positions: pd.Series,
    values_by_position: Mapping[int, object],
) -> None:
    """Set ``column`` of each row to the value at that row's position.

    ``entity_positions`` gives the position of each row to set, keyed by the
    index of ``afdb_columns``. A row whose position has no value keeps its current
    one.
    """
    if not values_by_position:
        return
    mapped = entity_positions.map(values_by_position)
    assigned = mapped.notna()
    if bool(assigned.any()):
        afdb_columns.loc[mapped.index[assigned], column] = mapped[assigned]


def _protein_and_residue_rows(
    atom_dataframe: pd.DataFrame,
    entity_ids: pd.Series,
    is_polymer: pd.Series,
    residue_names: pd.Series,
    entry_result: AFDBEntryResult,
) -> tuple[pd.Series, pd.Series]:
    """Return which atoms belong to a protein, and which to an amino acid.

    Both are boolean series on the index of ``atom_dataframe``, and every
    amino acid atom is a protein atom. A polymer atom belongs to a protein
    when its entity has a reference or is listed as a protein, or when its
    residue name is an amino acid.
    """
    known_protein_entities = set(group_references_by_entity(entry_result)) | set(
        entry_result.protein_entity_ids
    )
    name_says_amino_acid = residue_names.map(is_amino_acid_name).astype(bool)
    is_protein = is_polymer & (
        entity_ids.isin(known_protein_entities) | name_says_amino_acid
    )
    # The component type from the file says which components are amino acids.
    # Where the file gives no type, the residue name decides through Gemmi's
    # table, and a name the table does not know is not an amino acid.
    component_type = (
        atom_dataframe["_pdbai_chem_comp_type"].fillna("").astype(str).str.lower()
    )
    typed = component_type.ne("")
    is_amino_acid = (typed & component_type.str.contains("peptide", regex=False)) | (
        ~typed & name_says_amino_acid
    )
    return is_protein, is_protein & is_amino_acid


@dataclass(frozen=True)
class _PositionMapping:
    """What each entity position of one entity maps to.

    Each dict is keyed by entity position. ``uniprot_id`` and ``status``
    hold every position that maps to exactly one UniProt position.
    ``model_id`` and ``plddt`` hold only the positions among them that got a
    score.
    """

    uniprot_id: dict[int, str] = field(default_factory=dict[int, str])
    status: dict[int, str] = field(default_factory=dict[int, str])
    model_id: dict[int, str | None] = field(default_factory=dict[int, str | None])
    plddt: dict[int, float] = field(default_factory=dict[int, float])


def _map_positions(
    deposited_names: Mapping[int, Iterable[str]],
    references: Sequence[AFDBEntityReference],
    entry_result: AFDBEntryResult,
) -> _PositionMapping:
    """Return what each of an entity's positions maps to through references.

    ``deposited_names`` gives the residue names the table records at each
    position to map. The module docstring gives the rules. A position
    recorded under several names gets a score only when every one of them
    matches the model's residue.
    """
    mapping = _PositionMapping()
    for entity_position, names in deposited_names.items():
        mapped_references: list[tuple[AFDBEntityReference, int]] = []
        for reference in references:
            reference_position = reference.map_entity_position(entity_position)
            if reference_position is not None:
                mapped_references.append((reference, reference_position))

        mapping_keys = {
            (reference.uniprot_id, reference_position)
            for reference, reference_position in mapped_references
        }
        if len(mapping_keys) != 1:
            continue
        mapped_reference, reference_position = mapped_references[0]
        mapped_model = entry_result.models.get(mapped_reference.uniprot_id)
        selected_model_id: str | None = None
        mapping.uniprot_id[entity_position] = mapped_reference.uniprot_id
        if mapped_model is not None:
            mapping.status[entity_position] = mapped_model.status
            selected_model_id = mapped_model.model_id_by_position.get(
                str(reference_position)
            )
        plddt = (
            mapped_model.plddt_by_position.get(str(reference_position))
            if mapped_model is not None
            else None
        )
        if plddt is None or mapped_model is None:
            continue
        modelled_name = mapped_model.residue_name_by_position.get(
            str(reference_position), ""
        )
        recorded_names = list(names)
        if not recorded_names or not all(
            residue_names_match(name, modelled_name) for name in recorded_names
        ):
            continue
        mapping.model_id[entity_position] = selected_model_id
        mapping.plddt[entity_position] = plddt
    return mapping


def _annotate_entity(
    afdb_columns: pd.DataFrame,
    entity_mask: pd.Series,
    residue_mask: pd.Series,
    sequence_ids: pd.Series,
    residue_names: pd.Series,
    references: Sequence[AFDBEntityReference],
    entry_result: AFDBEntryResult,
) -> None:
    """Write the AlphaFold columns of one entity's atoms into ``afdb_columns``.

    ``entity_mask`` selects the entity's protein atoms and ``residue_mask``
    those of its amino acids. ``residue_names`` gives each atom's residue
    name, empty where the file gives none. ``references`` holds at least one
    reference.
    """
    accessions = list(dict.fromkeys(reference.uniprot_id for reference in references))
    afdb_columns.loc[entity_mask, "afdb_status"] = combined_model_status(
        [entry_result.models.get(accession) for accession in accessions],
        complete=True,
        fallback=entry_result.status,
    )
    if len(references) == 1 and not references[0].aligned_regions:
        afdb_columns.loc[entity_mask, "uniprot_id"] = references[0].uniprot_id

    afdb_columns.loc[residue_mask, "afdb_residue_has_plddt"] = False
    entity_positions = sequence_ids[residue_mask].astype(int)
    unique_positions = np.unique(entity_positions.to_numpy())
    deposited_names: dict[int, set[str]] = {
        int(position): set() for position in unique_positions
    }
    position_names = pd.DataFrame(
        {"position": entity_positions, "name": residue_names[residue_mask]}
    ).drop_duplicates()
    for position, name in zip(
        position_names["position"], position_names["name"], strict=True
    ):
        if name:
            deposited_names[int(position)].add(str(name))
    mapping = _map_positions(deposited_names, references, entry_result)

    _assign_by_position(
        afdb_columns, "uniprot_id", entity_positions, mapping.uniprot_id
    )
    _assign_by_position(afdb_columns, "afdb_status", entity_positions, mapping.status)
    _assign_by_position(
        afdb_columns, "afdb_model_id", entity_positions, mapping.model_id
    )
    _assign_by_position(afdb_columns, "afdb_plddt", entity_positions, mapping.plddt)
    scored = entity_positions.index[entity_positions.isin(set(mapping.plddt))]
    afdb_columns.loc[scored, "afdb_residue_has_plddt"] = True

    coverage = (
        float(len(mapping.plddt)) / float(len(unique_positions))
        if len(unique_positions)
        else np.nan
    )
    afdb_columns.loc[entity_mask, "afdb_plddt_coverage"] = coverage


def annotate_atom_dataframe(
    atom_dataframe: pd.DataFrame, entry_result: AFDBEntryResult
) -> pd.DataFrame:
    """Return the AlphaFold columns for every atom of the table, on its index.

    The table must have the res_name and _pdbai_chem_comp_type columns the
    parser always writes. res_name is the label component where the file
    gives no author one, so a file without author names still has its amino
    acids recognized. A table without a label_entity_id or label_seq_id
    column, which a file may leave out, gives every atom the entry status
    and nothing else.
    """
    afdb_columns = pd.DataFrame(index=atom_dataframe.index)
    afdb_columns["uniprot_id"] = None
    afdb_columns["afdb_status"] = None
    afdb_columns["afdb_model_id"] = None
    afdb_columns["afdb_plddt_coverage"] = np.nan
    afdb_columns["afdb_plddt"] = np.nan
    afdb_columns["afdb_residue_has_plddt"] = pd.Series(
        pd.NA, index=atom_dataframe.index, dtype="boolean"
    )
    for level in SEQUENCE_IDENTITY_LEVELS:
        afdb_columns[sequence_cluster_column(level)] = None

    entity_col = atom_dataframe.get("_atom_site.label_entity_id")
    seq_col = atom_dataframe.get("_atom_site.label_seq_id")
    if entity_col is None or seq_col is None:
        afdb_columns["afdb_status"] = entry_result.status
        return afdb_columns

    entity_ids = entity_col.fillna("").astype(str).str.strip()
    # The clusters are a fact about each polymer entity, protein or not, so
    # every atom of the entity carries them.
    for entity_id, clusters in entry_result.sequence_clusters.items():
        entity_rows = entity_ids == entity_id
        for level in SEQUENCE_IDENTITY_LEVELS:
            cluster = clusters.get(str(level))
            if cluster is not None:
                afdb_columns.loc[entity_rows, sequence_cluster_column(level)] = cluster
    sequence_ids = pd.to_numeric(seq_col, errors="coerce")
    # A label_seq_id counts from 1 in whole numbers. Any other value names no
    # polymer position, rather than being truncated onto one.
    sequence_ids = sequence_ids.where(sequence_ids.mod(1).eq(0) & sequence_ids.ge(1))
    residue_names = atom_dataframe["res_name"].fillna("").astype(str).str.strip()
    is_protein, is_residue = _protein_and_residue_rows(
        atom_dataframe, entity_ids, sequence_ids.notna(), residue_names, entry_result
    )

    if not entry_result.references:
        afdb_columns.loc[is_protein, "afdb_status"] = entry_result.status
        afdb_columns.loc[is_residue, "afdb_residue_has_plddt"] = False
        return afdb_columns

    references_by_entity = group_references_by_entity(entry_result)
    for entity_id in entity_ids[is_protein].drop_duplicates():
        entity_mask = is_protein & (entity_ids == entity_id)
        references = references_by_entity.get(entity_id, [])
        if references:
            _annotate_entity(
                afdb_columns,
                entity_mask,
                entity_mask & is_residue,
                sequence_ids,
                residue_names,
                references,
                entry_result,
            )
            continue
        # RCSB answered for the entity when discovery was complete, or when
        # it named the entity a protein without a UniProt mapping.
        answered = (
            entry_result.discovery_complete
            or entity_id in entry_result.unmapped_protein_entity_ids
        )
        afdb_columns.loc[entity_mask, "afdb_status"] = (
            AFDB_STATUS_NO_UNIPROT_MAPPING if answered else AFDB_STATUS_LOOKUP_FAILED
        )
        afdb_columns.loc[entity_mask & is_residue, "afdb_residue_has_plddt"] = False

    return afdb_columns


__all__ = [
    "annotate_atom_dataframe",
    "group_references_by_entity",
    "is_amino_acid_name",
    "residue_names_match",
]
