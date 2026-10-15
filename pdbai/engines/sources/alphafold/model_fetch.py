"""AlphaFold models of one UniProt accession and their per-residue pLDDT.

The AlphaFold Database prediction API, documented at
https://alphafold.ebi.ac.uk/api-docs, returns every prediction that involves
an accession. The list can hold fragments of a long protein, isoforms,
complexes, and alternate datasets. Merging them all would invent a sequence
that nobody predicted. So one family is chosen, and only its models are
merged. The family is the set of canonical AF-<accession>-F<n> fragments
when the database has them. Otherwise it is the one best prediction for
exactly this accession. A prediction the API marks as a complex never joins
a family.

Each model is a ModelCIF file. The pLDDT of a residue comes from the local
quality metric that the file declares as pLDDT. The B-factor of the CA atom
is used only when no such metric gives a score. The residue name at each
position is kept beside its score, so that the annotation can check that a
score lands on the residue it was predicted for. A file that gives one
residue two different scores or two different names is rejected whole,
because its numbering cannot be trusted.

Fragments of a long protein are numbered in one of two ways. Some count
from 1, and their scores are moved to UniProt positions by the fragment's
declared start. Others already use UniProt positions and are left alone.
A fragment that fits neither way is rejected as parse_failed, because
moving its scores by a guessed offset would put them on the wrong residues.
Where two fragments overlap, the fragment ranked first keeps the position.

A model is cached only when every member of the family was downloaded and
parsed. Any other outcome may be a passing failure, or a gap that a later
database release fills.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast
from urllib.parse import quote, urlparse

import gemmi
import numpy as np
import requests

from ....json_values import optional_object_dict
from ..http_cache import read_envelope, write_envelope
from ..scalars import to_int
from .models import (
    AFDB_STATUS_AVAILABLE,
    AFDB_STATUS_DOWNLOAD_FAILED,
    AFDB_STATUS_LOOKUP_FAILED,
    AFDB_STATUS_MODEL_NOT_FOUND,
    AFDB_STATUS_NO_UNIPROT_MAPPING,
    AFDB_STATUS_PARSE_FAILED,
    AFDB_STATUS_PARTIAL,
    AFDBModelResult,
    model_cache_path,
    plddt_value,
)

AFDB_PREDICTION_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/"


def _category_rows(block: gemmi.cif.Block, category: str) -> list[dict[str, str]]:
    """Return each row of an mmCIF category as a dict keyed by the full tag.

    The category may be written as a loop or as single items. Quotes are
    removed, and the null values ? and . read as empty strings. A category
    the block lacks has no rows.
    """
    table = block.find_mmcif_category(category)
    tags = [str(tag) for tag in table.tags]
    return [
        {tag: str(row.str(index)) for index, tag in enumerate(tags)} for row in table
    ]


def _plddt_scores(block: gemmi.cif.Block) -> dict[str, float]:
    """Return the pLDDT of each residue of a ModelCIF block, by label_seq_id.

    The result is empty when the block holds no score or gives one residue
    two different scores.
    """
    plddt_metric_ids: set[str] = set()
    declared_metric_ids: set[str] = set()
    for row in _category_rows(block, "_ma_qa_metric."):
        metric_id = row.get("_ma_qa_metric.id", "")
        if metric_id:
            declared_metric_ids.add(metric_id)
        description = " ".join(
            row.get(tag, "")
            for tag in (
                "_ma_qa_metric.name",
                "_ma_qa_metric.type",
                "_ma_qa_metric.description",
            )
        ).lower()
        if metric_id and "plddt" in description.replace("_", ""):
            plddt_metric_ids.add(metric_id)

    scores: dict[str, float] = {}
    for row in _category_rows(block, "_ma_qa_metric_local."):
        key = row.get("_ma_qa_metric_local.label_seq_id", "").strip()
        metric_id = row.get("_ma_qa_metric_local.metric_id", "")
        score = plddt_value(row.get("_ma_qa_metric_local.metric_value"))
        if not key or score is None:
            continue
        # The file declares its metrics and none is pLDDT, so its local values
        # measure something else.
        if declared_metric_ids and not plddt_metric_ids:
            continue
        if plddt_metric_ids and metric_id not in plddt_metric_ids:
            continue
        if key in scores and not np.isclose(scores[key], score, atol=1e-6):
            return {}
        scores[key] = score
    if scores:
        return scores

    # AlphaFold also writes each residue's pLDDT into the B-factor of its
    # atoms. It is the fallback for a file whose local metrics gave no score.
    for row in _category_rows(block, "_atom_site."):
        if row.get("_atom_site.label_atom_id", "").strip() != "CA":
            continue
        key = row.get("_atom_site.label_seq_id", "").strip()
        score = plddt_value(row.get("_atom_site.B_iso_or_equiv"))
        if not key or score is None:
            continue
        if key in scores and not np.isclose(scores[key], score, atol=1e-6):
            return {}
        scores[key] = score
    return scores


def _residue_names(block: gemmi.cif.Block) -> dict[str, str] | None:
    """Return the residue name at each label_seq_id of a ModelCIF block.

    Names are read from the atoms and from the local quality metrics, and
    are upper case. None stands for a block that gives one position two
    different names.
    """
    names: dict[str, str] = {}
    for category, position_tag, name_tag in (
        ("_atom_site.", "_atom_site.label_seq_id", "_atom_site.label_comp_id"),
        (
            "_ma_qa_metric_local.",
            "_ma_qa_metric_local.label_seq_id",
            "_ma_qa_metric_local.label_comp_id",
        ),
    ):
        for row in _category_rows(block, category):
            key = row.get(position_tag, "").strip()
            name = row.get(name_tag, "").strip().upper()
            if key and name and names.setdefault(key, name) != name:
                return None
    return names


def parse_afdb_model_cif(cif_text: str) -> tuple[dict[str, float], dict[str, str]]:
    """Return the pLDDT and the residue name of each residue, by label_seq_id.

    A local metric counts as pLDDT when its name, type, or description says
    so, or when the file declares no metric at all. Names are upper case,
    and only scored positions are named. Both are empty when the file cannot
    be read, holds no score, or gives one residue two different scores or
    two different names.
    """
    try:
        block = gemmi.cif.read_string(cif_text).sole_block()
    except (IndexError, RuntimeError, ValueError):
        # Gemmi raises ValueError for text that is not CIF, IndexError for an
        # empty document, and RuntimeError for more than one data block.
        return {}, {}
    scores = _plddt_scores(block)
    names = _residue_names(block)
    if not scores or names is None:
        return {}, {}
    return scores, {key: name for key, name in names.items() if key in scores}


def prediction_id(prediction: Mapping[str, object]) -> str | None:
    """Return the identifier the API gives the prediction, or None."""
    for key in ("modelEntityId", "entryId", "modelId"):
        value = prediction.get(key)
        if value:
            return str(value)
    return None


def model_id_from_prediction(
    prediction: Mapping[str, object], cif_url: str
) -> str | None:
    """Return the model file's name without .cif, or the prediction's ID.

    The file name carries the model version, as in AF-P69905-F1-model_v4, so
    it says exactly which model supplied a score.
    """
    filename = Path(urlparse(cif_url).path).name
    if filename:
        return filename[:-4] if filename.lower().endswith(".cif") else filename
    return prediction_id(prediction)


def _family_member_id(prediction: Mapping[str, object]) -> str:
    """Return the identifier a prediction is ranked and grouped by."""
    cif_url = str(prediction.get("cifUrl") or "")
    return (
        prediction_id(prediction) or model_id_from_prediction(prediction, cif_url) or ""
    )


def _declared_position(prediction: Mapping[str, object], *keys: str) -> int | None:
    """Return the first value under ``keys`` that is 1 or more, or None."""
    for key in keys:
        position = to_int(prediction.get(key))
        if position is not None and position >= 1:
            return position
    return None


def prediction_range(prediction: Mapping[str, object]) -> tuple[int, int | None]:
    """Return the first and last UniProt residues the prediction declares.

    uniprotStart and uniprotEnd are read, or sequenceStart and sequenceEnd
    where those are missing. A value below 1 counts as not declared. When no
    start is declared it is 1, and when no end is declared it is None.
    """
    start = _declared_position(prediction, "uniprotStart", "sequenceStart") or 1
    end = _declared_position(prediction, "uniprotEnd", "sequenceEnd")
    return start, end


def _score_position(key: str) -> int:
    """Return a model position written as text, or raise ValueError."""
    position = to_int(key)
    if position is None or position < 1:
        raise ValueError(f"pLDDT position {key!r} is not a whole number >= 1")
    return position


def uniprot_offset(positions: Iterable[str], start: int, end: int | None) -> int:
    """Return the offset that turns a model's positions into UniProt positions.

    ``positions`` are the model's label_seq_id values, and ``start`` and
    ``end`` are the UniProt range the prediction declares, with ``end``
    None when it declares none. Positions that all lie within that range
    already use UniProt numbering, and the offset is 0. Positions of a
    fragment that starts after residue 1 and whose lowest position is 1 are
    moved by ``start`` minus 1. When ``end`` is known, the moved positions
    must not pass it. The two cases cannot both hold, and no positions at
    all give 0. Anything else raises ValueError, as does a position that is
    not a whole number of at least 1 or that repeats another.
    """
    seen: set[int] = set()
    for key in positions:
        position = _score_position(key)
        if position in seen:
            raise ValueError(f"pLDDT position {key!r} repeats position {position}")
        seen.add(position)
    if not seen:
        return 0
    lowest = min(seen)
    highest = max(seen)
    if lowest >= start and (end is None or highest <= end):
        return 0
    shift = start - 1
    if start > 1 and lowest == 1 and (end is None or highest + shift <= end):
        return shift
    declared = f"{start} to {end}" if end is not None else f"{start} onwards"
    raise ValueError(
        f"pLDDT positions {lowest} to {highest} do not fit the declared UniProt "
        f"range {declared}. A fragment is renumbered only when its positions "
        "start at 1 and fit that range once moved."
    )


def prediction_sort_key(
    prediction: Mapping[str, object], accession: str
) -> tuple[int, int, int, str, str]:
    """Return the key that ranks a prediction within its accession's family.

    The canonical F1 model comes first, then any prediction whose identifier
    contains the accession, then the rest. Ties go to the earlier start, then
    to the longer span, then to the identifier and the URL. The ranking
    therefore never depends on the order of the API's answer.
    """
    normalized_id = _family_member_id(prediction).lower()
    normalized_accession = accession.lower()
    canonical_id = f"af-{normalized_accession}-f1"
    if normalized_id == canonical_id:
        identity_rank = 0
    elif normalized_accession in normalized_id:
        identity_rank = 1
    else:
        identity_rank = 2
    start, end = prediction_range(prediction)
    cif_url = str(prediction.get("cifUrl") or "")
    return identity_rank, start, -(end or start), normalized_id, cif_url


def select_prediction_family(
    predictions: Iterable[Mapping[str, object]], accession: str
) -> list[Mapping[str, object]]:
    """Return the predictions to merge for an accession, best ranked first.

    Only predictions with a model file count, and a prediction whose
    isComplex is true never does. The family is every canonical
    AF-<accession>-F<n> prediction. Without one, the family is the single
    best prediction recorded for exactly this accession. Best means an
    identifier that contains the accession, then the widest span, then the
    earliest start. The list is empty when there is no such prediction.
    """
    candidates = [
        prediction
        for prediction in predictions
        if prediction.get("cifUrl") and prediction.get("isComplex") is not True
    ]
    canonical_pattern = re.compile(
        rf"^af-{re.escape(accession.lower())}-f\d+(?:-|$)", re.IGNORECASE
    )
    canonical = [
        prediction
        for prediction in candidates
        if canonical_pattern.match(_family_member_id(prediction))
    ]
    if canonical:
        return sorted(canonical, key=lambda item: prediction_sort_key(item, accession))

    exact_accession = [
        prediction
        for prediction in candidates
        if str(prediction.get("uniprotAccession") or "").strip().lower()
        == accession.strip().lower()
    ]
    if not exact_accession:
        return []

    def fallback_key(
        prediction: Mapping[str, object],
    ) -> tuple[int, int, int, int, str, str]:
        basic = prediction_sort_key(prediction, accession)
        start, end = prediction_range(prediction)
        span = (end or start) - start + 1
        return basic[0], -span, start, basic[2], basic[3], basic[4]

    return [min(exact_accession, key=fallback_key)]


def _cached_model(cache_path: Path, accession: str) -> AFDBModelResult | None:
    """Return the complete model cached for the accession, or None.

    A file that is not a complete model with scores, and one that names
    another accession, count as missing.
    """
    cached = read_envelope(cache_path)
    model_data = optional_object_dict(cached.get("model")) if cached else None
    if model_data is None:
        return None
    model = AFDBModelResult.from_dict(model_data)
    # Accessions are cached under their lower-case form, so case alone does
    # not make the file another accession's.
    if (
        model.status != AFDB_STATUS_AVAILABLE
        or not model.has_scores
        or model.uniprot_id.lower() != accession.lower()
    ):
        return None
    return model


def fetch_afdb_model(
    uniprot_id: str,
    *,
    session: requests.Session,
    timeout: float,
    cache_directory: Path | None = None,
) -> AFDBModelResult:
    """Fetch the AlphaFold models of a UniProt accession and merge their pLDDT.

    No failure raises. Each one is returned as a status. An empty accession
    gives no_uniprot_mapping. model_not_found means the database has no
    compatible prediction. lookup_failed means the prediction API failed or
    returned no model file. download_failed and parse_failed mean that no
    model in the family gave a score. When both happened, parse_failed is
    reported. A family that lost a member but still gave scores is partial.
    With ``cache_directory`` None, nothing is cached.
    """
    accession = uniprot_id.strip()
    if not accession:
        return AFDBModelResult("", AFDB_STATUS_NO_UNIPROT_MAPPING)

    cache_path = (
        model_cache_path(cache_directory, accession)
        if cache_directory is not None
        else None
    )
    if cache_path is not None:
        cached_model = _cached_model(cache_path, accession)
        if cached_model is not None:
            return cached_model

    # The accession comes from a remote record, so it is quoted as one path
    # segment rather than trusted to be one.
    api_url = AFDB_PREDICTION_API_URL + quote(accession, safe="")
    try:
        response = session.get(api_url, timeout=timeout)
        if response.status_code == 404:
            return AFDBModelResult(accession, AFDB_STATUS_MODEL_NOT_FOUND)
        response.raise_for_status()
        predictions: object = response.json()
    except Exception as exception:
        return AFDBModelResult(
            accession,
            AFDB_STATUS_LOOKUP_FAILED,
            failure_reason=f"AFDB lookup failed for {accession}: {exception}",
        )

    if not isinstance(predictions, list):
        return AFDBModelResult(
            accession,
            AFDB_STATUS_LOOKUP_FAILED,
            failure_reason=f"AFDB lookup for {accession} returned a non-list payload.",
        )
    if not predictions:
        return AFDBModelResult(accession, AFDB_STATUS_MODEL_NOT_FOUND)
    prediction_items = [
        item_mapping
        for item in cast(list[object], predictions)
        if (item_mapping := optional_object_dict(item)) is not None
    ]
    if not any(item.get("cifUrl") for item in prediction_items):
        return AFDBModelResult(
            accession,
            AFDB_STATUS_LOOKUP_FAILED,
            failure_reason=(
                f"AFDB lookup for {accession} returned predictions without a CIF URL."
            ),
        )
    family = select_prediction_family(prediction_items, accession)
    if not family:
        return AFDBModelResult(
            accession,
            AFDB_STATUS_MODEL_NOT_FOUND,
            failure_reason=(
                f"AFDB lookup for {accession} returned no compatible accession model."
            ),
        )

    merged_scores: dict[str, float] = {}
    model_id_by_position: dict[str, str] = {}
    residue_name_by_position: dict[str, str] = {}
    model_ids: list[str] = []
    source_urls: list[str] = []
    failures: list[str] = []
    parse_failures = 0

    # The family arrives ranked. The first model to score a position keeps it.
    for prediction in family:
        cif_url = str(prediction["cifUrl"])
        model_id = model_id_from_prediction(prediction, cif_url) or cif_url
        try:
            cif_response = session.get(cif_url, timeout=timeout)
            cif_response.raise_for_status()
            cif_text = cif_response.text
        except Exception as exception:
            failures.append(f"{model_id}: download failed ({exception})")
            continue

        scores, names = parse_afdb_model_cif(cif_text)
        if not scores:
            parse_failures += 1
            failures.append(f"{model_id}: no parseable pLDDT values")
            continue

        start, end = prediction_range(prediction)
        try:
            offset = uniprot_offset(scores, start, end)
        except ValueError as error:
            parse_failures += 1
            failures.append(f"{model_id}: {error}")
            continue

        supplied_a_score = False
        for key, value in scores.items():
            position = str(_score_position(key) + offset)
            if position in merged_scores:
                continue
            merged_scores[position] = value
            model_id_by_position[position] = model_id
            if key in names:
                residue_name_by_position[position] = names[key]
            supplied_a_score = True
        if supplied_a_score:
            model_ids.append(model_id)
            source_urls.append(cif_url)

    # Every member of the family either failed, and is named in failures, or
    # gave scores. So no scores means every member failed, and no failure
    # means the whole family was read.
    if not merged_scores:
        return AFDBModelResult(
            accession,
            AFDB_STATUS_PARSE_FAILED if parse_failures else AFDB_STATUS_DOWNLOAD_FAILED,
            failure_reason="; ".join(failures),
        )
    result = AFDBModelResult(
        accession,
        AFDB_STATUS_PARTIAL if failures else AFDB_STATUS_AVAILABLE,
        plddt_by_position=merged_scores,
        model_id_by_position=model_id_by_position,
        residue_name_by_position=residue_name_by_position,
        model_ids=list(dict.fromkeys(model_ids)),
        source_urls=list(dict.fromkeys(source_urls)),
        failure_reason="; ".join(failures) if failures else None,
    )
    if cache_path is not None and result.status == AFDB_STATUS_AVAILABLE:
        write_envelope(cache_path, {"model": result.to_dict()})
    return result


__all__ = [
    "AFDB_PREDICTION_API_URL",
    "fetch_afdb_model",
    "model_id_from_prediction",
    "parse_afdb_model_cif",
    "prediction_id",
    "prediction_range",
    "prediction_sort_key",
    "select_prediction_family",
    "uniprot_offset",
]
