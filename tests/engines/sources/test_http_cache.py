from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from fake_http import FakeResponse, FakeSession

from pdbai.engines.sources.http_cache import (
    CACHE_TTL_SECONDS,
    NotAvailableError,
    fetch_gzip,
    fetch_mmcif,
    fetch_mtz,
    is_fresh_gzip,
    is_fresh_mmcif,
    is_fresh_mtz,
    is_gzip_file,
    is_mtz_file,
    read_envelope,
    write_envelope,
)


def test_an_envelope_round_trips_until_it_expires(tmp_path: Path) -> None:
    path = tmp_path / "entry.json"
    write_envelope(path, {"entry": {"a": 1}})

    envelope = read_envelope(path)
    assert envelope is not None and envelope["entry"] == {"a": 1}

    stale = json.loads(path.read_text(encoding="utf-8"))
    stale["cached_at"] = time.time() - CACHE_TTL_SECONDS - 1
    path.write_text(json.dumps(stale), encoding="utf-8")
    assert read_envelope(path) is None


@pytest.mark.parametrize(
    "content", ["{not json", "", "[1, 2, 3]", '"a string"', "42", "null", "{}"]
)
def test_unusable_envelopes_read_as_missing(tmp_path: Path, content: str) -> None:
    path = tmp_path / "entry.json"
    path.write_text(content, encoding="utf-8")

    assert read_envelope(path) is None
    assert read_envelope(tmp_path / "absent.json") is None


def test_an_envelope_too_deep_or_not_utf8_reads_as_missing(tmp_path: Path) -> None:
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 100_000, encoding="utf-8")
    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe{}")

    assert read_envelope(deep) is None
    assert read_envelope(binary) is None


def test_a_payload_may_not_replace_the_cache_time(tmp_path: Path) -> None:
    path = tmp_path / "entry.json"

    with pytest.raises(ValueError, match="'cached_at' is reserved"):
        write_envelope(path, {"cached_at": 0, "entry": {}})

    assert not path.exists()


def _fetch(session: FakeSession, path: Path) -> bool:
    return fetch_mmcif(
        "u", path, session=session.as_session(), timeout=1.0, label="RCSB"
    )


def test_a_stale_mmcif_is_downloaded_again(tmp_path: Path) -> None:
    # An expired file may predate a remediation or re-refinement. Its
    # coordinates must not be paired with newer metadata.
    path = tmp_path / "1abc.cif"
    path.write_bytes(b"data_old\n")
    old = time.time() - CACHE_TTL_SECONDS - 1
    os.utime(path, (old, old))
    session = FakeSession(FakeResponse(content=b"data_new\n"))

    assert _fetch(session, path) is False

    assert path.read_bytes() == b"data_new\n"
    assert len(session.calls) == 1


def test_a_download_is_written_once_then_reused(tmp_path: Path) -> None:
    path = tmp_path / "1abc.cif"
    session = FakeSession(FakeResponse(content=b"data_1abc\n"))

    assert _fetch(session, path) is False
    assert _fetch(session, path) is True

    assert path.read_bytes() == b"data_1abc\n"
    assert len(session.calls) == 1


def test_an_invalid_cached_file_is_replaced(tmp_path: Path) -> None:
    path = tmp_path / "1abc.cif"
    path.write_text("upstream error page", encoding="utf-8")
    session = FakeSession(FakeResponse(content=b"# comment\ndata_1abc\n"))

    assert _fetch(session, path) is False

    assert path.read_bytes() == b"# comment\ndata_1abc\n"


def test_a_missing_entry_is_reported_as_not_available(tmp_path: Path) -> None:
    session = FakeSession(FakeResponse(status_code=404))

    with pytest.raises(NotAvailableError):
        _fetch(session, tmp_path / "1abc.cif")
    assert not (tmp_path / "1abc.cif").exists()


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(status_code=410),
        # PDB-Redo's answer for an entry it has not re-refined.
        FakeResponse(status_code=500, content=b'{"error":"Not Found"}'),
    ],
)
def test_other_answers_of_a_missing_file_are_not_available(
    tmp_path: Path, response: FakeResponse
) -> None:
    with pytest.raises(NotAvailableError):
        _fetch(FakeSession(response), tmp_path / "1abc.cif")


@pytest.mark.parametrize(
    "content",
    [b"", b"<html>Internal Server Error</html>", b'{"error":"Timeout"}', b"[]"],
)
def test_a_server_error_is_a_failed_download(tmp_path: Path, content: bytes) -> None:
    session = FakeSession(FakeResponse(status_code=500, content=content))

    with pytest.raises(RuntimeError, match="download failed") as caught:
        _fetch(session, tmp_path / "1abc.cif")
    assert not isinstance(caught.value, NotAvailableError)


def test_an_expired_file_the_source_no_longer_has_is_removed(tmp_path: Path) -> None:
    # The engine names this path in its receipt even when the source has no
    # file. A copy left from an earlier download would be recorded as used.
    path = tmp_path / "1abc.cif"
    path.write_bytes(b"data_old\n")
    old = time.time() - CACHE_TTL_SECONDS - 1
    os.utime(path, (old, old))

    with pytest.raises(NotAvailableError, match="RCSB has no mmCIF at u"):
        _fetch(FakeSession(FakeResponse(status_code=404)), path)

    assert not path.exists()


def test_an_invalid_download_is_not_cached(tmp_path: Path) -> None:
    session = FakeSession(FakeResponse(content=b"<html>upstream error</html>"))

    with pytest.raises(RuntimeError, match="does not contain an mmCIF data block"):
        _fetch(session, tmp_path / "1abc.cif")

    assert list(tmp_path.iterdir()) == []


def test_an_interrupted_download_write_leaves_no_cached_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = FakeSession(FakeResponse(content=b"data_1abc\n_atom_site.id 1\n"))
    original_write_bytes = Path.write_bytes

    def interrupted_write(path: Path, content: bytes) -> int:
        original_write_bytes(path, content[:4])
        raise OSError("interrupted write")

    monkeypatch.setattr(Path, "write_bytes", interrupted_write)

    with pytest.raises(RuntimeError, match="interrupted write"):
        _fetch(session, tmp_path / "1abc.cif")

    assert list(tmp_path.iterdir()) == []


_MMCIF_CASES: list[tuple[str, bytes, bool]] = [
    ("plain block", b"data_1ABC\n_entry.id 1ABC\n", True),
    ("comment preamble", b"#\n# generated\n#\n\ndata_1ABC\n", True),
    # The probe must find the data block even after a very long comment.
    ("long comment preamble", b"#" + b"x" * 20_000 + b"\ndata_1ABC\n", True),
    ("crlf endings", b"#c\r\ndata_1ABC\r\n", True),
    ("cr only endings", b"#comment\rdata_1ABC\r", True),
    ("no trailing newline", b"data_1ABC", True),
    ("html error page", b"<html>404</html>\n", False),
    ("empty", b"", False),
    ("blank lines only", b"\n\n   \n", False),
]


@pytest.mark.parametrize(
    ("content", "expected"),
    [(content, expected) for _, content, expected in _MMCIF_CASES],
    ids=[name for name, _, _ in _MMCIF_CASES],
)
def test_cached_files_and_downloads_are_judged_alike(
    tmp_path: Path, content: bytes, expected: bool
) -> None:
    cached = tmp_path / "cached.cif"
    cached.write_bytes(content)
    session = FakeSession(FakeResponse(content=content))

    assert is_fresh_mmcif(cached) is expected
    if expected:
        assert _fetch(session, tmp_path / "downloaded.cif") is False
    else:
        with pytest.raises(RuntimeError, match="mmCIF data block"):
            _fetch(session, tmp_path / "downloaded.cif")


def test_the_cache_probe_never_loads_the_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "large.cif"
    path.write_bytes(b"data_1ABC\n" + b"ATOM 1 N\n" * 200_000)

    def forbidden_read_bytes(self: Path) -> bytes:
        raise AssertionError("cache probe read the entire file")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)

    assert is_fresh_mmcif(path) is True


def test_missing_and_unreadable_paths_are_not_cache_hits(tmp_path: Path) -> None:
    assert is_fresh_mmcif(tmp_path / "absent.cif") is False
    assert is_fresh_mmcif(tmp_path) is False


_MTZ = b"MTZ \x00\x00reflections"


def _fetch_mtz(session: FakeSession, path: Path) -> bool:
    return fetch_mtz(
        "https://example.org/1abc_final.mtz",
        path,
        session=session.as_session(),
        timeout=3.0,
        label="PDB-Redo 1ABC",
    )


def test_an_mtz_download_is_written_once_then_reused(tmp_path: Path) -> None:
    path = tmp_path / "1abc_final.mtz"
    session = FakeSession(FakeResponse(content=_MTZ))

    assert _fetch_mtz(session, path) is False
    assert _fetch_mtz(session, path) is True

    assert path.read_bytes() == _MTZ
    assert len(session.calls) == 1
    assert is_mtz_file(path)
    assert is_fresh_mtz(path)


def test_a_stale_or_damaged_mtz_is_downloaded_again(tmp_path: Path) -> None:
    stale = tmp_path / "stale.mtz"
    stale.write_bytes(_MTZ)
    old = time.time() - CACHE_TTL_SECONDS - 1
    os.utime(stale, (old, old))
    damaged = tmp_path / "damaged.mtz"
    damaged.write_bytes(b"<html>Not Found</html>")

    assert not is_fresh_mtz(stale)
    assert not is_fresh_mtz(damaged)
    assert not is_mtz_file(tmp_path / "absent.mtz")
    for path in (stale, damaged):
        session = FakeSession(FakeResponse(content=_MTZ + b"new"))
        assert _fetch_mtz(session, path) is False
        assert path.read_bytes() == _MTZ + b"new"


def test_a_missing_mtz_is_not_available_and_its_old_copy_is_removed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "1abc_final.mtz"
    path.write_bytes(_MTZ)
    old = time.time() - CACHE_TTL_SECONDS - 1
    os.utime(path, (old, old))

    with pytest.raises(NotAvailableError, match="has no MTZ"):
        _fetch_mtz(FakeSession(FakeResponse(status_code=404)), path)

    assert not path.exists()


def test_an_mtz_response_that_is_not_mtz_is_not_cached(tmp_path: Path) -> None:
    path = tmp_path / "1abc_final.mtz"

    with pytest.raises(RuntimeError, match="not an MTZ reflection file"):
        _fetch_mtz(FakeSession(FakeResponse(content=b"<html></html>")), path)
    with pytest.raises(RuntimeError, match="download failed"):
        _fetch_mtz(FakeSession(FakeResponse(status_code=503)), path)
    with pytest.raises(RuntimeError, match="download failed"):
        _fetch_mtz(FakeSession(ConnectionError("down")), path)

    assert not path.exists()


def test_a_gzip_download_is_cached_and_a_403_can_mean_missing(tmp_path: Path) -> None:
    path = tmp_path / "maps.cif.gz"
    content = b"\x1f\x8b\x08compressed"

    def fetch(session: FakeSession, missing: frozenset[int] = frozenset()) -> bool:
        return fetch_gzip(
            "https://example.org/maps.cif.gz",
            path,
            session=session.as_session(),
            timeout=3.0,
            label="wwPDB validation map 1ABC",
            missing_statuses=missing,
        )

    session = FakeSession(FakeResponse(content=content))
    assert fetch(session) is False
    assert fetch(session) is True
    assert is_gzip_file(path)
    assert is_fresh_gzip(path)
    old = time.time() - CACHE_TTL_SECONDS - 1
    os.utime(path, (old, old))
    assert not is_fresh_gzip(path)
    # Without the caller's word, a 403 is a failure and the old copy stays.
    with pytest.raises(RuntimeError, match="download failed"):
        fetch(FakeSession(FakeResponse(status_code=403)))
    assert path.exists()
    with pytest.raises(NotAvailableError, match="has no gzip file"):
        fetch(FakeSession(FakeResponse(status_code=403)), frozenset({403}))
    assert not path.exists()
    with pytest.raises(RuntimeError, match="not a gzip file"):
        fetch(FakeSession(FakeResponse(content=b"plain text")))
    assert not path.exists()
