from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pandas as pd
import pytest
import requests
from fake_http import FakeResponse, FakeSession

from pdbai.engines.core import (
    ENTRY_RECORD,
    EngineContext,
    EngineOptions,
    EngineServices,
    SourceFile,
)
from pdbai.engines.sources.alphafold import engine as engine_module
from pdbai.engines.sources.alphafold import model_fetch as model_fetch_module
from pdbai.engines.sources.alphafold.annotation import annotate_atom_dataframe
from pdbai.engines.sources.alphafold.artifacts import AFDB_ENTRY
from pdbai.engines.sources.alphafold.model_fetch import (
    fetch_afdb_model,
    select_prediction_family,
    uniprot_offset,
)
from pdbai.engines.sources.alphafold.models import (
    AFDB_STATUS_AVAILABLE,
    AFDB_STATUS_DOWNLOAD_FAILED,
    AFDB_STATUS_LOOKUP_FAILED,
    AFDB_STATUS_NOT_AVAILABLE_FOR_UPLOAD,
    AFDB_STATUS_PARSE_FAILED,
    AFDB_STATUS_PARTIAL,
    AFDBAlignedRegion,
    AFDBEntityReference,
    AFDBEntryResult,
    AFDBModelResult,
    discovery_cache_path,
    model_cache_path,
)

# parse_afdb_model_cif returns this pair of scores and residue names, each
# keyed by label_seq_id.
_ParsedModel = tuple[dict[str, float], dict[str, str]]


def _alanines(scores: dict[str, float]) -> _ParsedModel:
    """Return a parsed model whose every scored residue is an alanine."""
    return scores, dict.fromkeys(scores, "ALA")


def _entry() -> AFDBEntryResult:
    # Entity position 1 maps to UniProt residue 101. A score on the atom
    # therefore shows that the alignment offset was applied.
    return AFDBEntryResult(
        pdb_id="1ABC",
        references=[
            AFDBEntityReference(
                entity_id="1",
                uniprot_id="P12345",
                aligned_regions=(AFDBAlignedRegion(1, 101, 1),),
            )
        ],
        models={
            "P12345": AFDBModelResult(
                uniprot_id="P12345",
                status=AFDB_STATUS_AVAILABLE,
                plddt_by_position={"101": 87.5},
                residue_name_by_position={"101": "ALA"},
            )
        },
        status=AFDB_STATUS_AVAILABLE,
        protein_entity_ids=["1"],
    )


def test_alphafold_acquisition_passes_fetch_policy_and_the_entry_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entry = _entry()
    session = requests.Session()
    record: dict[str, object] = {
        "rcsb_entry_container_identifiers": {"polymer_entity_ids": ["1"]}
    }

    def fake_fetch(
        pdb_id: str,
        *,
        session: requests.Session,
        timeout: float,
        cache_directory: Path,
        fetch_models: bool,
        entry_json: object,
    ) -> AFDBEntryResult:
        assert (pdb_id, timeout, cache_directory, fetch_models) == (
            "1ABC",
            3.0,
            tmp_path / "afdb",
            True,
        )
        # The download has already fetched the RCSB entry record. The engine
        # reuses it rather than fetching it again.
        assert entry_json == record
        assert isinstance(session, requests.Session)
        return entry

    monkeypatch.setattr(engine_module, "fetch_afdb_entry", fake_fetch)
    context = EngineContext(
        "1ABC",
        pd.DataFrame(),
        EngineOptions(values={"include_alphafold": True}),
        services=EngineServices(session=session, timeout=3, cache_root=tmp_path),
    )
    context.artifacts.put(ENTRY_RECORD, record)

    result = engine_module.ALPHAFOLD_ACQUISITION_ENGINE.run(context)

    assert result.artifacts == {AFDB_ENTRY: entry}


def _acquire(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entry: AFDBEntryResult
) -> tuple[SourceFile, ...]:
    def fake_fetch(pdb_id: str, **kwargs: object) -> AFDBEntryResult:
        del pdb_id, kwargs
        return entry

    monkeypatch.setattr(engine_module, "fetch_afdb_entry", fake_fetch)
    context = EngineContext(
        "1ABC",
        pd.DataFrame(),
        EngineOptions(values={"include_alphafold": True}),
        services=EngineServices(
            session=requests.Session(), timeout=3, cache_root=tmp_path
        ),
    )
    return engine_module.ALPHAFOLD_ACQUISITION_ENGINE.run(context).source_files


def _write_cache_files(tmp_path: Path, *accessions: str) -> None:
    cache_directory = tmp_path / "afdb"
    paths = [discovery_cache_path(cache_directory, "1ABC")]
    paths += [model_cache_path(cache_directory, accession) for accession in accessions]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


def test_a_complete_lookup_lists_its_cache_files_as_source_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_cache_files(tmp_path, "P12345")

    source_files = _acquire(monkeypatch, tmp_path, _entry())

    assert [(source.role, source.path) for source in source_files] == [
        (
            "rcsb_polymer_entity_mapping_cache",
            discovery_cache_path(tmp_path / "afdb", "1ABC"),
        ),
        ("alphafold_model_cache", model_cache_path(tmp_path / "afdb", "P12345")),
    ]


def test_cache_files_say_whether_the_lookup_read_or_wrote_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_cache_files(tmp_path, "P12345")
    # The discovery was read from the cache, and the model was fetched and
    # written there.
    entry = dataclasses.replace(_entry(), discovery_cached=True)

    source_files = _acquire(monkeypatch, tmp_path, entry)

    assert [source.cached_before for source in source_files] == [True, False]
    assert source_files[0].source_url == (
        "https://data.rcsb.org/rest/v1/core/entry/1abc; "
        "https://data.rcsb.org/rest/v1/core/polymer_entity/1abc/1"
    )


def test_a_lookup_that_wrote_no_cache_files_lists_no_source_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_files = _acquire(monkeypatch, tmp_path, _entry())

    assert source_files == ()


def test_a_partial_lookup_lists_only_the_cache_files_it_could_have_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Files left by an earlier run sit beside every part of the result. Only
    # the complete model is one the lookup caches, so only its file counts.
    entry = dataclasses.replace(_entry(), unmapped_protein_entity_ids=["2"])
    entry.models["P99999"] = AFDBModelResult(
        uniprot_id="P99999",
        status=AFDB_STATUS_PARTIAL,
        plddt_by_position={"1": 70.0},
    )
    entry.models["Q11111"] = AFDBModelResult(
        uniprot_id="Q11111", status=AFDB_STATUS_DOWNLOAD_FAILED
    )
    _write_cache_files(tmp_path, "P12345", "P99999", "Q11111")

    source_files = _acquire(monkeypatch, tmp_path, entry)

    assert [source.path for source in source_files] == [
        model_cache_path(tmp_path / "afdb", "P12345")
    ]


def test_alphafold_annotation_reports_a_failed_lookup() -> None:
    dataframe = pd.DataFrame(
        {
            "_atom_site.label_entity_id": ["1", "2"],
            "_atom_site.label_seq_id": ["1", "."],
            "res_name": ["ALA", "HOH"],
            "_pdbai_chem_comp_type": ["", ""],
        }
    )
    context = EngineContext(
        "1ABC", dataframe, EngineOptions(values={"include_alphafold": True})
    )
    context.failures["alphafold_database_source"] = "AFDB offline"

    result = engine_module.ALPHAFOLD_ANNOTATION_ENGINE.run(context)

    assert result.report == {
        "afdb_status": AFDB_STATUS_LOOKUP_FAILED,
        "afdb_message": "AFDB offline",
    }
    # The protein atom says the lookup failed, and the water stays empty.
    assert result.columns is not None
    assert result.columns.loc[0, "afdb_status"] == AFDB_STATUS_LOOKUP_FAILED
    assert not bool(result.columns.loc[0, "afdb_residue_has_plddt"])
    assert pd.isna(result.columns.loc[1, "afdb_status"])
    assert pd.isna(result.columns.loc[1, "afdb_residue_has_plddt"])


def test_alphafold_annotation_marks_uploads_as_not_available() -> None:
    dataframe = pd.DataFrame({"_atom_site.label_entity_id": ["1"]})
    context = EngineContext(
        "UPLOAD",
        dataframe,
        EngineOptions(values={"include_alphafold": True}),
        source="upload",
    )

    result = engine_module.ALPHAFOLD_ANNOTATION_ENGINE.run(context)

    assert result.report["afdb_status"] == AFDB_STATUS_NOT_AVAILABLE_FOR_UPLOAD


def test_alphafold_annotation_maps_entity_sequence_to_plddt() -> None:
    dataframe = pd.DataFrame(
        {
            "_atom_site.label_entity_id": ["1", "1", "2"],
            "_atom_site.label_seq_id": ["1", "1", "."],
            "res_name": ["ALA", "ALA", "ZN"],
            "_pdbai_chem_comp_type": ["", "", ""],
        },
        index=[10, 11, 20],
    )
    context = EngineContext(
        "1ABC", dataframe, EngineOptions(values={"include_alphafold": True})
    )
    context.artifacts.put(AFDB_ENTRY, _entry())

    result = engine_module.ALPHAFOLD_ANNOTATION_ENGINE.run(context)

    assert result.report == {"afdb_status": AFDB_STATUS_AVAILABLE, "afdb_message": None}

    assert result.columns is not None
    assert result.columns.loc[[10, 11], "uniprot_id"].tolist() == [
        "P12345",
        "P12345",
    ]
    assert result.columns.loc[[10, 11], "afdb_plddt"].tolist() == [87.5, 87.5]
    assert result.columns.loc[10, "afdb_plddt_coverage"] == pytest.approx(1.0)
    assert bool(result.columns.loc[10, "afdb_residue_has_plddt"])
    assert pd.isna(result.columns.loc[20, "afdb_status"])


def test_alphafold_annotation_rejects_ambiguous_residue_mapping() -> None:
    dataframe = pd.DataFrame(
        {
            "_atom_site.label_entity_id": ["1"],
            "_atom_site.label_seq_id": ["7"],
            "res_name": ["ALA"],
            "_pdbai_chem_comp_type": [""],
        }
    )
    references = [
        AFDBEntityReference(
            entity_id="1",
            uniprot_id=accession,
            aligned_regions=(AFDBAlignedRegion(7, reference_position, 1),),
        )
        for accession, reference_position in (("P11111", 17), ("P22222", 27))
    ]
    models = {
        reference.uniprot_id: AFDBModelResult(
            uniprot_id=reference.uniprot_id,
            status=AFDB_STATUS_AVAILABLE,
            plddt_by_position={
                str(reference.aligned_regions[0].reference_beg_seq_id): 90.0
            },
        )
        for reference in references
    }
    entry = AFDBEntryResult(
        pdb_id="1ABC",
        references=references,
        models=models,
        status=AFDB_STATUS_AVAILABLE,
        protein_entity_ids=["1"],
    )

    result = annotate_atom_dataframe(dataframe, entry)

    assert pd.isna(result.loc[0, "uniprot_id"])
    assert pd.isna(result.loc[0, "afdb_plddt"])
    assert not bool(result.loc[0, "afdb_residue_has_plddt"])
    assert result.loc[0, "afdb_plddt_coverage"] == pytest.approx(0.0)


def test_alphafold_annotation_does_not_invent_identifier_only_residue_map() -> None:
    dataframe = pd.DataFrame(
        {
            "_atom_site.label_entity_id": ["1"],
            "_atom_site.label_seq_id": ["1"],
            "res_name": ["ALA"],
            "_pdbai_chem_comp_type": [""],
        }
    )
    entry = AFDBEntryResult(
        pdb_id="1ABC",
        references=[
            AFDBEntityReference(
                entity_id="1",
                uniprot_id="P12345",
            )
        ],
        models={
            "P12345": AFDBModelResult(
                uniprot_id="P12345",
                status=AFDB_STATUS_AVAILABLE,
                plddt_by_position={"1": 95.0},
            )
        },
        status=AFDB_STATUS_AVAILABLE,
        protein_entity_ids=["1"],
    )

    result = annotate_atom_dataframe(dataframe, entry)

    assert result.loc[0, "uniprot_id"] == "P12345"
    assert pd.isna(result.loc[0, "afdb_plddt"])
    assert not bool(result.loc[0, "afdb_residue_has_plddt"])
    assert result.loc[0, "afdb_plddt_coverage"] == pytest.approx(0.0)


def test_alphafold_prediction_family_keeps_requested_isoform_separate() -> None:
    canonical = {
        "uniprotAccession": "P12345",
        "entryId": "AF-P12345-F1-v4",
        "cifUrl": "https://example.test/AF-P12345-F1-v4.cif",
    }
    isoform = {
        "uniprotAccession": "P12345-2",
        "entryId": "AF-P12345-2-F1-v4",
        "cifUrl": "https://example.test/AF-P12345-2-F1-v4.cif",
    }

    selected = select_prediction_family([canonical, isoform], "P12345-2")

    assert selected == [isoform]


def test_alphafold_fragment_scores_are_shifted_to_uniprot_coordinates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prediction: dict[str, object] = {
        "uniprotAccession": "P12345",
        "entryId": "AF-P12345-F2-v4",
        "uniprotStart": 201,
        "uniprotEnd": 202,
        "cifUrl": "https://example.test/AF-P12345-F2-v4.cif",
    }
    fake_session = FakeSession(
        FakeResponse([prediction]),
        FakeResponse(text="data_fragment"),
    )

    def fake_parse(_: str) -> _ParsedModel:
        return _alanines({"1": 91.0, "2": 82.0})

    monkeypatch.setattr(model_fetch_module, "parse_afdb_model_cif", fake_parse)

    result = fetch_afdb_model(
        "P12345",
        session=fake_session.as_session(),
        timeout=2.0,
        cache_directory=tmp_path,
    )

    assert result.status == AFDB_STATUS_AVAILABLE
    assert result.plddt_by_position == {"201": 91.0, "202": 82.0}
    assert result.model_id_by_position == {
        "201": "AF-P12345-F2-v4",
        "202": "AF-P12345-F2-v4",
    }


def _fragment_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    start: int,
    end: int | None,
    positions: range,
) -> AFDBModelResult:
    prediction: dict[str, object] = {
        "uniprotAccession": "P12345",
        "entryId": "AF-P12345-F2-v4",
        "uniprotStart": start,
        "cifUrl": "https://example.test/AF-P12345-F2-v4.cif",
    }
    if end is not None:
        prediction["uniprotEnd"] = end
    fake_session = FakeSession(
        FakeResponse([prediction]),
        FakeResponse(text="data_fragment"),
    )

    def fake_parse(_text: str) -> _ParsedModel:
        return _alanines({str(position): 90.0 for position in positions})

    monkeypatch.setattr(model_fetch_module, "parse_afdb_model_cif", fake_parse)
    return fetch_afdb_model(
        "P12345",
        session=fake_session.as_session(),
        timeout=2.0,
        cache_directory=tmp_path,
    )


def _position_span(result: AFDBModelResult) -> tuple[int, int]:
    positions = sorted(int(key) for key in result.plddt_by_position)
    return positions[0], positions[-1]


@pytest.mark.parametrize(
    ("label", "start", "end", "positions", "expected"),
    [
        ("local numbering, complete", 201, 1600, range(1, 1401), (201, 1600)),
        ("global numbering, complete", 201, 1600, range(201, 1601), (201, 1600)),
        ("global numbering, truncated model", 201, 1600, range(201, 1001), (201, 1000)),
        ("global numbering, no uniprotEnd", 201, None, range(201, 1001), (201, 1000)),
        ("local numbering, no uniprotEnd", 201, None, range(1, 801), (201, 1000)),
        ("first fragment starts at one", 1, 1400, range(1, 1401), (1, 1400)),
    ],
)
def test_fragment_positions_end_up_in_uniprot_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    label: str,
    start: int,
    end: int | None,
    positions: range,
    expected: tuple[int, int],
) -> None:

    result = _fragment_result(
        monkeypatch, tmp_path, start=start, end=end, positions=positions
    )

    assert _position_span(result) == expected, label
    assert len(result.plddt_by_position) == len(positions), label


def test_fragment_shift_keeps_scores_attached_to_their_own_residue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prediction: dict[str, object] = {
        "uniprotAccession": "P12345",
        "entryId": "AF-P12345-F2-v4",
        "uniprotStart": 201,
        "uniprotEnd": 203,
        "cifUrl": "https://example.test/AF-P12345-F2-v4.cif",
    }
    fake_session = FakeSession(
        FakeResponse([prediction]),
        FakeResponse(text="data_fragment"),
    )

    def fake_parse(_text: str) -> _ParsedModel:
        return _alanines({"1": 11.0, "2": 22.0, "3": 33.0})

    monkeypatch.setattr(model_fetch_module, "parse_afdb_model_cif", fake_parse)

    result = fetch_afdb_model(
        "P12345",
        session=fake_session.as_session(),
        timeout=2.0,
        cache_directory=tmp_path,
    )

    assert result.plddt_by_position == {"201": 11.0, "202": 22.0, "203": 33.0}


def _fragment_result_from(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    boundaries: dict[str, object],
    positions: range,
) -> AFDBModelResult:
    prediction: dict[str, object] = {
        "uniprotAccession": "P12345",
        "entryId": "AF-P12345-F2-v4",
        "cifUrl": "https://example.test/AF-P12345-F2-v4.cif",
        **boundaries,
    }
    fake_session = FakeSession(
        FakeResponse([prediction]),
        FakeResponse(text="data_fragment"),
    )

    def fake_parse(_text: str) -> _ParsedModel:
        return _alanines({str(position): 90.0 for position in positions})

    monkeypatch.setattr(model_fetch_module, "parse_afdb_model_cif", fake_parse)
    return fetch_afdb_model(
        "P12345",
        session=fake_session.as_session(),
        timeout=2.0,
        cache_directory=tmp_path,
    )


@pytest.mark.parametrize(
    ("label", "boundaries", "positions", "expected"),
    [
        (
            "single-residue fragment",
            {"uniprotStart": 201, "uniprotEnd": 201},
            range(1, 2),
            (201, 201),
        ),
        ("no start declared", {"uniprotEnd": 400}, range(1, 51), (1, 50)),
        (
            "zero start",
            {"uniprotStart": 0, "uniprotEnd": 400},
            range(1, 51),
            (1, 50),
        ),
        (
            "negative start",
            {"uniprotStart": -5, "uniprotEnd": 400},
            range(1, 51),
            (1, 50),
        ),
        (
            "sequenceStart/sequenceEnd instead of uniprot bounds",
            {"sequenceStart": 201, "sequenceEnd": 400},
            range(1, 201),
            (201, 400),
        ),
    ],
)
def test_inconsistent_fragment_boundaries_never_corrupt_positions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    label: str,
    boundaries: dict[str, object],
    positions: range,
    expected: tuple[int, int],
) -> None:

    result = _fragment_result_from(monkeypatch, tmp_path, boundaries, positions)

    assert _position_span(result) == expected, label
    assert len(result.plddt_by_position) == len(positions), label


@pytest.mark.parametrize(
    ("label", "boundaries", "positions"),
    [
        (
            "local numbering with a gap at the start",
            {"uniprotStart": 201, "uniprotEnd": 1600},
            range(5, 1401),
        ),
        (
            "positions straddle the declared start",
            {"uniprotStart": 201, "uniprotEnd": 1600},
            range(150, 251),
        ),
        (
            "positions straddle the start, no end declared",
            {"uniprotStart": 201},
            range(150, 251),
        ),
        (
            "positions run past the declared end",
            {"uniprotStart": 201, "uniprotEnd": 400},
            range(201, 501),
        ),
        (
            "first fragment runs past its declared end",
            {"uniprotStart": 1, "uniprotEnd": 400},
            range(1, 501),
        ),
        (
            "local numbering runs past the declared end once moved",
            {"uniprotStart": 201, "uniprotEnd": 400},
            range(1, 301),
        ),
        (
            "end before start, local positions",
            {"uniprotStart": 201, "uniprotEnd": 100},
            range(1, 51),
        ),
        (
            "end before start, global positions",
            {"uniprotStart": 201, "uniprotEnd": 100},
            range(201, 251),
        ),
    ],
)
def test_a_fragment_that_fits_neither_numbering_is_rejected_as_parse_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    label: str,
    boundaries: dict[str, object],
    positions: range,
) -> None:

    result = _fragment_result_from(monkeypatch, tmp_path, boundaries, positions)

    assert result.status == AFDB_STATUS_PARSE_FAILED, label
    assert result.plddt_by_position == {}, label
    assert result.failure_reason is not None, label
    assert "do not fit the declared UniProt range" in result.failure_reason, label


def test_a_rejected_fragment_leaves_the_rest_of_the_family_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first: dict[str, object] = {
        "uniprotAccession": "P12345",
        "entryId": "AF-P12345-F1-v4",
        "uniprotStart": 1,
        "uniprotEnd": 2,
        "cifUrl": "https://example.test/AF-P12345-F1-v4.cif",
    }
    second: dict[str, object] = {
        "uniprotAccession": "P12345",
        "entryId": "AF-P12345-F2-v4",
        "uniprotStart": 201,
        "uniprotEnd": 400,
        "cifUrl": "https://example.test/AF-P12345-F2-v4.cif",
    }
    fake_session = FakeSession(
        FakeResponse([first, second]),
        FakeResponse(text="first"),
        FakeResponse(text="second"),
    )

    def fake_parse(text: str) -> _ParsedModel:
        if text == "first":
            return _alanines({"1": 90.0, "2": 80.0})
        return _alanines({"150": 70.0, "250": 60.0})

    monkeypatch.setattr(model_fetch_module, "parse_afdb_model_cif", fake_parse)

    result = fetch_afdb_model(
        "P12345",
        session=fake_session.as_session(),
        timeout=2.0,
        cache_directory=tmp_path,
    )

    assert result.status == AFDB_STATUS_PARTIAL
    assert result.plddt_by_position == {"1": 90.0, "2": 80.0}
    assert result.failure_reason is not None
    assert result.failure_reason.startswith("AF-P12345-F2-v4: pLDDT positions 150")


@pytest.mark.parametrize(
    ("label", "start", "end", "positions", "expected"),
    [
        ("already within the declared range", 201, 400, ["205", "300"], 0),
        ("within a range with no declared end", 201, None, ["201", "900"], 0),
        ("numbered from 1 after a later start", 201, 400, ["1", "200"], 200),
        ("numbered from 1 with no declared end", 201, None, ["1", "2"], 200),
        ("no positions", 201, 400, [], 0),
    ],
)
def test_uniprot_offset_keeps_or_moves_a_consistent_fragment(
    label: str,
    start: int,
    end: int | None,
    positions: list[str],
    expected: int,
) -> None:
    assert uniprot_offset(positions, start, end) == expected, label


@pytest.mark.parametrize(
    ("label", "start", "end", "positions", "message"),
    [
        (
            "between the two numberings",
            201,
            1600,
            ["150", "250"],
            "pLDDT positions 150 to 250 do not fit the declared UniProt range "
            "201 to 1600.",
        ),
        (
            "between the two numberings, no declared end",
            201,
            None,
            ["5"],
            "pLDDT positions 5 to 5 do not fit the declared UniProt range 201 onwards.",
        ),
        (
            "moved past the declared end",
            201,
            300,
            ["1", "101"],
            "pLDDT positions 1 to 101 do not fit the declared UniProt range "
            "201 to 300.",
        ),
        ("not a whole number", 1, None, ["."], "pLDDT position '.' is not"),
        ("below 1", 1, None, ["0"], "pLDDT position '0' is not"),
        (
            "one position written twice",
            1,
            None,
            ["5", "05"],
            "pLDDT position '05' repeats position 5",
        ),
    ],
)
def test_uniprot_offset_rejects_a_fragment_it_would_have_to_guess(
    label: str,
    start: int,
    end: int | None,
    positions: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        uniprot_offset(positions, start, end)
