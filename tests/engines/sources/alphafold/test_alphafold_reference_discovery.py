from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fake_http import FakeResponse, FakeSession

from pdbai.engines.sources.alphafold.models import (
    AFDBReferenceDiscovery,
)
from pdbai.engines.sources.alphafold.references import (
    discover_entry_references,
    entity_is_protein,
    parse_entity_clusters,
    parse_entity_references,
)
from pdbai.engines.sources.http_cache import CACHE_TTL_SECONDS

_ENTRY_PAYLOAD: dict[str, object] = {
    "rcsb_entry_container_identifiers": {"polymer_entity_ids": ["1"]}
}
_ENTITY_PAYLOAD: dict[str, object] = {
    "rcsb_polymer_entity_container_identifiers": {
        "entity_id": "1",
        "asym_ids": ["A"],
        "auth_asym_ids": ["A"],
    },
    "entity_poly": {"rcsb_entity_polymer_type": "Protein"},
    "rcsb_polymer_entity_align": [
        {
            "reference_database_name": "UniProt",
            "reference_database_accession": "P12345",
            "aligned_regions": [
                {"entity_beg_seq_id": 1, "ref_beg_seq_id": 1, "length": 10}
            ],
        }
    ],
}


def _discover(
    cache_directory: Path, *responses: FakeResponse
) -> tuple[AFDBReferenceDiscovery, FakeSession]:
    session = FakeSession(*responses)
    discovery = discover_entry_references(
        "1ABC",
        session=session.as_session(),
        timeout=1.0,
        cache_directory=cache_directory,
    )
    return discovery, session


def _cache_file(cache_directory: Path) -> Path:
    return cache_directory / "entities" / "1abc.json"


def _write_cache(cache_directory: Path, payload: str) -> Path:
    path = _cache_file(cache_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _cached_payload(*, age_seconds: float) -> str:
    # Only the cache holds the accession CACHED. Seeing it in a result shows
    # that the cache was used.
    return json.dumps(
        {
            "cached_at": time.time() - age_seconds,
            "pdb_id": "1abc",
            "references": [
                {
                    "entity_id": "1",
                    "uniprot_id": "CACHED",
                    "aligned_regions": [],
                }
            ],
            "protein_entity_ids": ["1"],
            "sequence_clusters": {"1": {"95": "CACHED_95"}},
        }
    )


def test_alignment_regions_become_a_residue_map() -> None:
    references = parse_entity_references(_ENTITY_PAYLOAD)

    assert [reference.uniprot_id for reference in references] == ["P12345"]
    assert references[0].map_entity_position(1) == 1
    assert references[0].map_entity_position(10) == 10
    assert references[0].map_entity_position(11) is None


def test_isoform_is_preferred_over_the_base_accession() -> None:
    payload = json.loads(json.dumps(_ENTITY_PAYLOAD))
    payload["rcsb_polymer_entity_align"][0]["reference_database_isoform"] = "P12345-2"

    references = parse_entity_references(payload)

    assert references[0].uniprot_id == "P12345-2"


def test_non_uniprot_alignments_are_ignored() -> None:
    payload = json.loads(json.dumps(_ENTITY_PAYLOAD))
    payload["rcsb_polymer_entity_align"][0]["reference_database_name"] = "GenBank"

    assert parse_entity_references(payload) == []


@pytest.mark.parametrize(
    "region",
    [
        {"entity_beg_seq_id": 1, "ref_beg_seq_id": 1, "length": 0},
        {"entity_beg_seq_id": 1, "ref_beg_seq_id": 1, "length": -5},
        {"entity_beg_seq_id": 1, "length": 10},
        {"ref_beg_seq_id": 1, "length": 10},
        {},
    ],
)
def test_unusable_aligned_regions_do_not_produce_a_residue_map(
    region: dict[str, object],
) -> None:

    payload = json.loads(json.dumps(_ENTITY_PAYLOAD))
    payload["rcsb_polymer_entity_align"][0]["aligned_regions"] = [region]

    references = parse_entity_references(payload)

    assert references[0].uniprot_id == "P12345"
    assert references[0].aligned_regions == ()
    assert references[0].map_entity_position(1) is None


def test_identifier_only_records_expose_the_accession_without_a_map() -> None:
    payload: dict[str, object] = {
        "rcsb_polymer_entity_container_identifiers": {
            "entity_id": "1",
            "reference_sequence_identifiers": [
                {"database_name": "UniProt", "database_accession": "P99999"}
            ],
        }
    }

    references = parse_entity_references(payload)

    assert references[0].uniprot_id == "P99999"
    assert references[0].aligned_regions == ()


def test_bare_uniprot_ids_are_read_when_the_identifiers_name_no_uniprot() -> None:
    payload: dict[str, object] = {
        "rcsb_polymer_entity_container_identifiers": {
            "entity_id": "1",
            "reference_sequence_identifiers": [
                {"database_name": "GenBank", "database_accession": "AAA12345"}
            ],
            "uniprot_ids": ["P77777"],
        }
    }

    assert [r.uniprot_id for r in parse_entity_references(payload)] == ["P77777"]


def test_bare_uniprot_ids_are_the_last_resort() -> None:
    payload: dict[str, object] = {
        "rcsb_polymer_entity_container_identifiers": {
            "entity_id": "1",
            "uniprot_ids": ["P77777"],
        }
    }

    assert [r.uniprot_id for r in parse_entity_references(payload)] == ["P77777"]


def test_entity_id_falls_back_to_the_rcsb_identifier() -> None:
    payload: dict[str, object] = {
        "rcsb_id": "1ABC_2",
        "rcsb_polymer_entity_container_identifiers": {},
        "rcsb_polymer_entity_align": [
            {
                "reference_database_name": "UniProt",
                "reference_database_accession": "P12345",
            }
        ],
    }

    assert parse_entity_references(payload)[0].entity_id == "2"


@pytest.mark.parametrize(
    ("entity_poly", "expected"),
    [
        ({"rcsb_entity_polymer_type": "Protein"}, True),
        ({"type": "polypeptide(L)"}, True),
        ({"type": "polydeoxyribonucleotide"}, False),
        ({}, False),
    ],
)
def test_protein_entities_are_recognized(
    entity_poly: dict[str, object], expected: bool
) -> None:
    assert entity_is_protein({"entity_poly": entity_poly}) is expected


def test_an_entity_without_a_polymer_record_is_not_protein() -> None:
    assert entity_is_protein({}) is False


def test_successful_discovery_is_fetched_and_persisted(tmp_path: Path) -> None:
    discovery, session = _discover(
        tmp_path, FakeResponse(_ENTRY_PAYLOAD), FakeResponse(_ENTITY_PAYLOAD)
    )

    assert [r.uniprot_id for r in discovery.references] == ["P12345"]
    assert discovery.complete
    assert len(session.calls) == 2
    assert _cache_file(tmp_path).exists()


def test_a_fresh_cache_is_served_without_any_request(tmp_path: Path) -> None:
    _write_cache(tmp_path, _cached_payload(age_seconds=60))

    discovery, session = _discover(tmp_path)

    assert [r.uniprot_id for r in discovery.references] == ["CACHED"]
    assert discovery.cached
    assert session.calls == []


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("stale", None),
        ("future timestamp", None),
        ("malformed json", "{not json"),
        ("missing cached_at", json.dumps({"references": []})),
        ("empty file", ""),
        ("no reference", "no reference"),
        ("a reference without an accession", "blank accession"),
    ],
)
def test_unusable_caches_are_refetched(
    tmp_path: Path, label: str, payload: str | None
) -> None:
    if payload is None:
        # A negative age puts cached_at in the future.
        age = CACHE_TTL_SECONDS + 10 if label == "stale" else -99_999
        payload = _cached_payload(age_seconds=age)
    elif payload in {"no reference", "blank accession"}:
        envelope = json.loads(_cached_payload(age_seconds=60))
        if payload == "no reference":
            envelope["references"] = []
        else:
            envelope["references"][0]["uniprot_id"] = ""
        payload = json.dumps(envelope)
    _write_cache(tmp_path, payload)

    discovery, session = _discover(
        tmp_path, FakeResponse(_ENTRY_PAYLOAD), FakeResponse(_ENTITY_PAYLOAD)
    )

    assert [r.uniprot_id for r in discovery.references] == ["P12345"], label
    assert len(session.calls) == 2, label


def test_a_failed_entity_is_reported_and_never_cached(tmp_path: Path) -> None:
    discovery, _session = _discover(
        tmp_path, FakeResponse(_ENTRY_PAYLOAD), FakeResponse(None, status_code=500)
    )

    assert discovery.references == []
    assert discovery.failed_entity_ids == ["1"]
    assert not discovery.complete
    assert discovery.failure_reason is not None
    assert not _cache_file(tmp_path).exists()


def test_a_failed_entry_lookup_is_reported_and_never_cached(tmp_path: Path) -> None:
    discovery, _session = _discover(tmp_path, FakeResponse(None, status_code=500))

    assert discovery.references == []
    assert not discovery.complete
    assert not _cache_file(tmp_path).exists()


def test_a_protein_without_a_mapping_is_reported_and_never_cached(
    tmp_path: Path,
) -> None:

    unmapped = {
        key: value
        for key, value in _ENTITY_PAYLOAD.items()
        if key != "rcsb_polymer_entity_align"
    }

    discovery, _session = _discover(
        tmp_path, FakeResponse(_ENTRY_PAYLOAD), FakeResponse(unmapped)
    )

    assert discovery.references == []
    assert discovery.unmapped_protein_entity_ids == ["1"]
    assert discovery.protein_entity_ids == ["1"]
    assert not _cache_file(tmp_path).exists()


def test_an_entry_without_polymer_entities_stops_after_one_request(
    tmp_path: Path,
) -> None:
    discovery, session = _discover(
        tmp_path, FakeResponse({"rcsb_entry_container_identifiers": {}})
    )

    assert discovery.references == []
    assert discovery.complete
    assert len(session.calls) == 1
    assert not _cache_file(tmp_path).exists()


def test_an_invalid_pdb_id_is_rejected_before_any_request(tmp_path: Path) -> None:
    session = FakeSession()

    with pytest.raises(ValueError, match="Invalid PDB ID"):
        discover_entry_references(
            "../secret",
            session=session.as_session(),
            timeout=1.0,
            cache_directory=tmp_path,
        )

    assert session.calls == []
    assert list(tmp_path.iterdir()) == []


def test_a_complete_discovery_is_cached(tmp_path: Path) -> None:
    discovery, _ = _discover(
        tmp_path, FakeResponse(_ENTRY_PAYLOAD), FakeResponse(_ENTITY_PAYLOAD)
    )

    written = json.loads(_cache_file(tmp_path).read_text(encoding="utf-8"))
    assert not discovery.cached

    assert [reference["uniprot_id"] for reference in written["references"]] == [
        "P12345"
    ]


def test_an_entry_record_without_container_identifiers_is_a_failed_lookup(
    tmp_path: Path,
) -> None:
    discovery, session = _discover(tmp_path, FakeResponse({"rcsb_id": "1ABC"}))

    assert discovery.references == []
    assert not discovery.complete
    assert discovery.failure_reason is not None
    assert "no container identifiers" in discovery.failure_reason
    assert len(session.calls) == 1
    assert not _cache_file(tmp_path).exists()


def test_an_entity_identifier_is_quoted_as_one_segment_of_its_url(
    tmp_path: Path,
) -> None:
    entry = {"rcsb_entry_container_identifiers": {"polymer_entity_ids": ["1/../2"]}}

    _discovery, session = _discover(
        tmp_path, FakeResponse(entry), FakeResponse(_ENTITY_PAYLOAD)
    )

    assert session.calls[1][0] == (
        "https://data.rcsb.org/rest/v1/core/polymer_entity/1abc/1%2F..%2F2"
    )


_GROUPS: list[object] = [
    {
        "aggregation_method": "sequence_identity",
        "group_id": "433_100",
        "similarity_cutoff": 100.0,
    },
    {
        "aggregation_method": "sequence_identity",
        "group_id": "391_95",
        "similarity_cutoff": 95,
    },
    {
        "aggregation_method": "sequence_identity",
        "group_id": "700_30",
        "similarity_cutoff": 30.0,
    },
    # A second group at a level is not read over the first.
    {
        "aggregation_method": "sequence_identity",
        "group_id": "701_30",
        "similarity_cutoff": 30.0,
    },
    # Groups made another way, at a level RCSB does not cluster at, or
    # without an identifier are left out.
    {
        "aggregation_method": "matching_uniprot_accession",
        "group_id": "P12345",
        "similarity_cutoff": None,
    },
    {
        "aggregation_method": "sequence_identity",
        "group_id": "9_40",
        "similarity_cutoff": 40.0,
    },
    {
        "aggregation_method": "sequence_identity",
        "group_id": " ",
        "similarity_cutoff": 50.0,
    },
    "not a group",
]


def test_sequence_clusters_are_read_at_each_identity_level() -> None:
    clusters = parse_entity_clusters({"rcsb_polymer_entity_group_membership": _GROUPS})

    assert clusters == {"100": "433_100", "95": "391_95", "30": "700_30"}
    assert parse_entity_clusters({}) == {}
    assert parse_entity_clusters({"rcsb_polymer_entity_group_membership": "x"}) == {}


def test_a_discovery_keeps_every_entity_s_clusters_and_caches_them(
    tmp_path: Path,
) -> None:
    entry = {"rcsb_entry_container_identifiers": {"polymer_entity_ids": ["1", "2"]}}
    protein = {**_ENTITY_PAYLOAD, "rcsb_polymer_entity_group_membership": _GROUPS}
    # A nucleic acid is no protein, but its clusters are kept all the same.
    nucleic_acid = {
        "rcsb_polymer_entity_container_identifiers": {"entity_id": "2"},
        "entity_poly": {"rcsb_entity_polymer_type": "DNA"},
        "rcsb_polymer_entity_group_membership": [_GROUPS[1]],
    }

    discovery, _ = _discover(
        tmp_path, FakeResponse(entry), FakeResponse(protein), FakeResponse(nucleic_acid)
    )
    cached = json.loads(_cache_file(tmp_path).read_text(encoding="utf-8"))

    assert discovery.sequence_clusters == {
        "1": {"100": "433_100", "95": "391_95", "30": "700_30"},
        "2": {"95": "391_95"},
    }
    assert discovery.protein_entity_ids == ["1"]
    assert cached["sequence_clusters"] == discovery.sequence_clusters


def test_a_cached_discovery_restores_its_clusters(tmp_path: Path) -> None:
    _write_cache(tmp_path, _cached_payload(age_seconds=60))

    discovery, session = _discover(tmp_path)

    assert discovery.sequence_clusters == {"1": {"95": "CACHED_95"}}
    assert session.calls == []


@pytest.mark.parametrize(
    "stored",
    [None, "x", {"1": "391_95"}, {"1": {"95": 391}}],
    ids=["written before clusters were kept", "not a mapping", "no levels", "not text"],
)
def test_a_cache_without_usable_clusters_is_refetched(
    tmp_path: Path, stored: object
) -> None:
    payload = json.loads(_cached_payload(age_seconds=60))
    if stored is None:
        del payload["sequence_clusters"]
    else:
        payload["sequence_clusters"] = stored
    _write_cache(tmp_path, json.dumps(payload))

    discovery, session = _discover(
        tmp_path, FakeResponse(_ENTRY_PAYLOAD), FakeResponse(_ENTITY_PAYLOAD)
    )

    assert [r.uniprot_id for r in discovery.references] == ["P12345"]
    assert len(session.calls) == 2
