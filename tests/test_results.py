from __future__ import annotations

import pytest

from pdbai.results import PdbProcessResult

# These are entries at each point a run can leave them, flagged as the
# pipeline records them.
_REFUSED_ON_RECORD = PdbProcessResult(
    pdb_id="1ABC", state="refused", technique_status="refused", metadata_ok=True
)
# With no RCSB record, the parsed file decides whether the entry is admitted.
_REFUSED_ON_FILE = PdbProcessResult(
    pdb_id="1ABC",
    state="refused",
    technique_status="refused",
    download_ok=True,
    parse_ok=True,
    metadata_ok=False,
    metadata_message="RCSB record unavailable",
)
_DOWNLOAD_FAILED = PdbProcessResult(
    pdb_id="1ABC",
    state="failed",
    metadata_ok=True,
    download_error="network unavailable",
)
_PARSE_FAILED = PdbProcessResult(
    pdb_id="1ABC",
    state="failed",
    metadata_ok=True,
    download_ok=True,
    parse_error="not an mmCIF file",
)
_OUTPUT_FAILED = PdbProcessResult(
    pdb_id="1ABC",
    state="failed",
    metadata_ok=True,
    download_ok=True,
    parse_ok=True,
    output_error="Annotation failed",
)
_DONE = PdbProcessResult(
    pdb_id="1ABC", state="done", metadata_ok=True, download_ok=True, parse_ok=True
)
_UPLOAD_DONE = PdbProcessResult(
    pdb_id="MODEL", source="upload", state="done", download_ok=True, parse_ok=True
)
# An unexpected error during acquisition is caught after the fact, so whether
# the record was fetched is not known. It reads as unknown rather than failed,
# since the error may have come before the record was asked for.
_ACQUISITION_RAISED = PdbProcessResult(
    pdb_id="1ABC", state="failed", download_error="Download failed for 1ABC: boom"
)


@pytest.mark.parametrize(
    ("result", "metadata", "download", "parse"),
    [
        (PdbProcessResult(pdb_id="1ABC"), None, None, None),
        (
            PdbProcessResult.from_manifest(
                "1ABC", {"state": "pending"}, default_source="download"
            ),
            None,
            None,
            None,
        ),
        (_REFUSED_ON_RECORD, True, None, None),
        (_REFUSED_ON_FILE, False, True, True),
        (_DOWNLOAD_FAILED, True, False, None),
        (_PARSE_FAILED, True, True, False),
        (_OUTPUT_FAILED, True, True, True),
        (_DONE, True, True, True),
        (_UPLOAD_DONE, None, True, True),
        (_ACQUISITION_RAISED, None, False, None),
    ],
)
def test_step_outcomes_are_none_for_a_step_the_entry_never_reached(
    result: PdbProcessResult,
    metadata: bool | None,
    download: bool | None,
    parse: bool | None,
) -> None:
    assert result.metadata_succeeded() is metadata
    assert result.download_succeeded() is download
    assert result.parse_succeeded() is parse


def test_a_failed_entry_without_a_structure_file_failed_at_download() -> None:
    # A manifest entry written without its message still reads as a failed
    # download rather than one that was never tried.
    restored = PdbProcessResult.from_manifest(
        "1ABC", {"state": "failed"}, default_source="download"
    )

    assert restored.download_succeeded() is False
    assert restored.parse_succeeded() is None


@pytest.mark.parametrize(
    ("stored", "restored_source"),
    [
        ("upload", "upload"),
        ("download", "download"),
        ("archive", "upload"),
        (1, "upload"),
    ],
)
def test_an_unknown_manifest_source_reads_as_the_runs_source(
    stored: object, restored_source: str
) -> None:
    restored = PdbProcessResult.from_manifest(
        "MODEL", {"state": "done", "source": stored}, default_source="upload"
    )

    assert restored.source == restored_source


@pytest.mark.parametrize("state", ["done", "failed", "refused"])
def test_a_missing_metadata_flag_reads_as_unknown_in_every_state(state: str) -> None:
    # A done download may have been built from its file without its record, so
    # not even a done entry implies the fetch succeeded.
    restored = PdbProcessResult.from_manifest(
        "1ABC", {"state": state}, default_source="download"
    )

    assert restored.metadata_ok is None
    assert restored.metadata_succeeded() is None


def test_a_stored_metadata_flag_and_message_are_read_back() -> None:
    restored = PdbProcessResult.from_manifest(
        "1ABC",
        {
            "state": "done",
            "metadata_ok": False,
            "metadata_message": "RCSB record unavailable",
        },
        default_source="download",
    )

    assert restored.metadata_succeeded() is False
    assert restored.metadata_message == "RCSB record unavailable"


def test_a_manifest_key_with_no_field_is_ignored() -> None:
    # Manifests of earlier builds also stored a metadata level beside the
    # flag, and such a manifest still reads.
    restored = PdbProcessResult.from_manifest(
        "1ABC",
        {"state": "done", "metadata_ok": True, "metadata_level": "info"},
        default_source="download",
    )

    assert restored.metadata_succeeded() is True
    assert "metadata_level" not in restored.to_manifest()


def test_a_result_survives_its_manifest_round_trip() -> None:
    restored = PdbProcessResult.from_manifest(
        "1ABC", _REFUSED_ON_FILE.to_manifest(), default_source="download"
    )

    assert restored == _REFUSED_ON_FILE
