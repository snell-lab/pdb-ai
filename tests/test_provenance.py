from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdbai import licenses, provenance
from pdbai.json_values import object_dict
from pdbai.provenance import capture_execution_session, file_receipt


def test_file_receipt_fingerprints_a_file_that_is_present(tmp_path: Path) -> None:
    content = b"pdb_id\n1ABC\n"
    path = tmp_path / "atoms.csv"
    path.write_bytes(content)
    modified_at = datetime(2026, 8, 14, 12, 0, 0, 250000, tzinfo=UTC).timestamp()
    os.utime(path, (modified_at, modified_at))

    receipt = file_receipt(path, role="dataset_flat")

    observed_at = str(receipt.pop("observed_at"))
    assert receipt == {
        "role": "dataset_flat",
        "path": "atoms.csv",
        "source_url": None,
        "available": True,
        "size_bytes": len(content),
        "file_modified_at": "2026-08-14T12:00:00.250000Z",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    assert observed_at.endswith("Z")
    assert datetime.fromisoformat(observed_at).tzinfo == UTC


def test_file_receipt_records_a_missing_file_as_not_available(tmp_path: Path) -> None:
    receipt = file_receipt(
        tmp_path / "1abc_metadata.json",
        role="rcsb_entry_metadata",
        source_url="https://data.rcsb.org/rest/v1/core/entry/1ABC",
        cache_present_before_acquisition=False,
    )

    receipt.pop("observed_at")
    assert receipt == {
        "role": "rcsb_entry_metadata",
        "path": "1abc_metadata.json",
        "source_url": "https://data.rcsb.org/rest/v1/core/entry/1ABC",
        "available": False,
        "cache_present_before_acquisition": False,
    }


# Forward slashes keep the path the same whichever platform wrote the
# manifest, as the manifest's partial paths do. A file outside the root is
# named by its bare name, so no absolute path from this machine is recorded.
def test_file_receipt_names_a_file_under_its_root_by_relative_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mmcif" / "1abc.cif"

    under_root = file_receipt(path, role="mmcif", relative_to=tmp_path)
    elsewhere = file_receipt(path, role="mmcif", relative_to=tmp_path / "other")

    assert under_root["path"] == "mmcif/1abc.cif"
    assert elsewhere["path"] == "1abc.cif"


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Git must see only this directory. Clearing GIT_DIR and its siblings
    # matters when pytest runs inside a git hook, which sets them for the
    # outer repository. The ceiling stops git from finding a repository above
    # tmp_path. User and system configuration are ignored, so a setting such
    # as commit signing cannot break the commits a test makes.
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.setattr(provenance, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _git(repository: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *args], check=True, capture_output=True
    )


@pytest.mark.usefixtures("checkout")
def test_session_records_no_git_state_outside_a_repository() -> None:
    code = object_dict(capture_execution_session()["code"])

    assert code == {
        "application": "PDB-AI",
        "repository": "https://github.com/snell-lab/pdb-ai",
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
    }


def test_session_records_the_commit_branch_and_uncommitted_changes(
    checkout: Path,
) -> None:
    _git(checkout, "init", "--quiet", "--initial-branch=main")
    _git(
        checkout,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.test",
        "commit",
        "--quiet",
        "--allow-empty",
        "--message=Initial",
    )

    clean = object_dict(capture_execution_session()["code"])
    (checkout / "notes.txt").write_text("draft\n", encoding="utf-8")
    dirty = object_dict(capture_execution_session()["code"])

    assert clean["git_branch"] == "main"
    commit = clean["git_commit"]
    assert isinstance(commit, str) and len(commit) == 40
    assert clean["git_dirty"] is False
    assert dirty["git_dirty"] is True


@pytest.mark.parametrize(
    ("origin", "recorded"),
    [
        (
            "https://user:token@github.com/example/pdb-ai.git",
            "https://github.com/example/pdb-ai.git",
        ),
        ("git@github.com:example/pdb-ai.git", "github.com:example/pdb-ai.git"),
        (
            "user:token@github.com:example/pdb-ai.git",
            "github.com:example/pdb-ai.git",
        ),
        ("github.com:example/pdb-ai.git", "github.com:example/pdb-ai.git"),
        ("/srv/git/pdb-ai@2.git", "/srv/git/pdb-ai@2.git"),
        ("pdb-ai@2.git", "pdb-ai@2.git"),
    ],
)
def test_session_records_the_origin_url_without_credentials(
    checkout: Path, origin: str, recorded: str
) -> None:
    _git(checkout, "init", "--quiet")
    _git(checkout, "remote", "add", "origin", origin)

    code = object_dict(capture_execution_session()["code"])

    assert code["repository"] == recorded


# A remote URL can hold bytes the locale's encoding cannot decode. The
# surrogate escape passes the raw byte 0xFF to git, and no UTF-8 decoder
# accepts that byte. What replaces it depends on the locale, so only the rest
# of the URL is checked. Windows passes arguments to git as UTF-16, so no
# such byte can reach git there.
@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows cannot pass git an undecodable byte"
)
def test_session_reads_git_output_the_locale_cannot_decode(
    checkout: Path,
) -> None:
    _git(checkout, "init", "--quiet")
    _git(
        checkout,
        "remote",
        "add",
        "origin",
        "https://example.test/" + os.fsdecode(b"\xff") + ".git",
    )

    code = object_dict(capture_execution_session()["code"])

    repository = code["repository"]
    assert isinstance(repository, str)
    assert repository.startswith("https://example.test/")
    assert repository.endswith(".git")


@pytest.mark.usefixtures("checkout")
def test_session_records_reference_data_as_given() -> None:
    receipt = {"available": True, "entry_count": 3, "source": {"ref": None}}

    session = capture_execution_session(reference_data={"bnet": receipt})

    assert session["reference_data"] == {"bnet": receipt}


@pytest.mark.usefixtures("checkout")
@pytest.mark.parametrize(
    "receipt",
    [
        {"path": Path("reference.csv")},
        {"entry_count": float("nan")},
        {"sizes": (1, 2)},
        {1: "one"},
    ],
    ids=["path", "nan", "tuple", "integer_key"],
)
def test_session_refuses_reference_data_that_is_not_json(receipt: object) -> None:
    with pytest.raises(TypeError, match="reference data 'bnet'"):
        capture_execution_session(reference_data={"bnet": receipt})


# This stands in for licenses.PackageLicense with only the fields that
# provenance reads.
def _package(
    name: str, version: str | None, status: str, *, direct: bool
) -> SimpleNamespace:
    return SimpleNamespace(name=name, version=version, status=status, direct=direct)


def test_session_records_package_presence_not_license_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def installed_closure(packages: object) -> list[SimpleNamespace]:
        return [
            _package("numpy", "2.3.0", "ok", direct=True),
            _package("tzdata", "2025.2", "review", direct=False),
            _package("rabdam", None, "missing", direct=True),
        ]

    monkeypatch.setattr(licenses, "collect_dependency_licenses", installed_closure)

    session = capture_execution_session()

    assert session["dependencies"] == [
        {"name": "numpy", "version": "2.3.0", "direct": True, "status": "installed"},
        {"name": "tzdata", "version": "2025.2", "direct": False, "status": "installed"},
        {"name": "rabdam", "version": None, "direct": True, "status": "missing"},
    ]
    assert "dependencies_error" not in session


def test_session_records_why_the_dependency_inventory_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def malformed_closure(packages: object) -> list[SimpleNamespace]:
        raise ValueError("Invalid requirement in metadata for broken: 'demo >='")

    monkeypatch.setattr(licenses, "collect_dependency_licenses", malformed_closure)

    session = capture_execution_session()

    assert session["dependencies"] is None
    assert session["dependencies_error"] == (
        "Invalid requirement in metadata for broken: 'demo >='"
    )
