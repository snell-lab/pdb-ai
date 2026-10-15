from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

import pytest

from pdbai.atomic import write_atomically, write_text_atomically


def test_a_write_creates_missing_parents_and_leaves_no_scratch_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "entry.json"

    # The non-ASCII text shows the file is written as UTF-8 whatever the
    # locale.
    write_text_atomically(path, '{"unit": "Å²"}')

    assert path.read_bytes() == '{"unit": "Å²"}'.encode()
    assert [entry.name for entry in path.parent.iterdir()] == ["entry.json"]


def test_the_new_file_gets_the_permissions_of_any_new_file(tmp_path: Path) -> None:
    plain = tmp_path / "plain.csv"
    plain.write_text("a,b\n", encoding="utf-8")
    replaced = tmp_path / "replaced.csv"

    write_text_atomically(replaced, "a,b\n")

    assert stat.S_IMODE(replaced.stat().st_mode) == stat.S_IMODE(plain.stat().st_mode)


def test_a_block_that_raises_leaves_the_previous_file_intact(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    path.write_text("generation 1\n", encoding="utf-8")

    with pytest.raises(OSError, match="disk full"), write_atomically(path) as scratch:
        scratch.write_text("genera", encoding="utf-8")
        raise OSError("disk full")

    assert path.read_text(encoding="utf-8") == "generation 1\n"
    assert [entry.name for entry in tmp_path.iterdir()] == ["table.csv"]


def test_a_failed_replacement_is_not_retried_and_leaves_the_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    write_text_atomically(path, "generation 1")
    attempts: list[Path] = []

    def fail(source: Path, target: Path) -> None:
        attempts.append(target)
        raise OSError("disk full")

    monkeypatch.setattr("pdbai.atomic.os.replace", fail)
    with pytest.raises(OSError, match="disk full"):
        write_text_atomically(path, "generation 2")

    assert attempts == [path]
    assert path.read_text(encoding="utf-8") == "generation 1"
    assert [entry.name for entry in tmp_path.iterdir()] == ["manifest.json"]


def test_a_refused_replacement_is_retried_after_a_growing_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    real_replace = os.replace
    refusals = 2
    waits: list[float] = []

    def replace_once_released(source: Path, target: Path) -> None:
        nonlocal refusals
        if refusals:
            refusals -= 1
            raise PermissionError("locked")
        real_replace(source, target)

    monkeypatch.setattr("pdbai.atomic.os.replace", replace_once_released)
    monkeypatch.setattr("pdbai.atomic.time.sleep", waits.append)
    write_text_atomically(path, "written")

    assert path.read_text(encoding="utf-8") == "written"
    assert len(waits) == 2
    assert waits[0] < waits[1]


def test_a_replacement_refused_throughout_raises_the_last_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    write_text_atomically(path, "generation 1")
    attempts: list[Path] = []
    waits: list[float] = []

    def refuse(source: Path, target: Path) -> None:
        attempts.append(target)
        raise PermissionError(f"locked, attempt {len(attempts)}")

    monkeypatch.setattr("pdbai.atomic.os.replace", refuse)
    monkeypatch.setattr("pdbai.atomic.time.sleep", waits.append)
    with pytest.raises(PermissionError) as refusal:
        write_text_atomically(path, "generation 2")

    assert len(attempts) == len(waits) + 1
    assert str(refusal.value) == f"locked, attempt {len(attempts)}"
    assert path.read_text(encoding="utf-8") == "generation 1"
    assert [entry.name for entry in tmp_path.iterdir()] == ["manifest.json"]


def test_concurrent_writers_leave_one_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def writer(index: int) -> None:
        try:
            barrier.wait()
            for _ in range(25):
                write_text_atomically(path, f"writer {index}")
        except BaseException as exception:
            errors.append(exception)

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert path.read_text(encoding="utf-8") in {f"writer {index}" for index in range(8)}
    assert [entry.name for entry in tmp_path.iterdir()] == ["manifest.json"]
