"""Reproducibility receipts for the code, environment, and files of a run.

A run's manifest.json holds two kinds of receipt, and run_summary.md renders
them as tables. An execution session describes one invocation. A run adds a
session each time it starts or resumes and keeps the earlier ones, so it
lists every environment it ran in. A file receipt describes one file as it
was at a given moment. An entry's input files are receipted as the entry is
acquired, and the assembled tables at settlement. Every timestamp is ISO
8601 in UTC. A receipt holds only JSON values, so it reads back from the
manifest exactly as it was written.

The manifest is published beside the dataset. A receipt therefore records
no more about the machine than reproducing the run needs. It holds no
absolute path, host name, or user name, and a token in the origin URL is
removed. Collecting a receipt never stops a run. A file, git state, or
dependency inventory that cannot be read is recorded as unavailable. The one
exception is engine reference data that is not a JSON value. That is a fault
in the engine's code rather than in the machine, so it raises.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

# The checkout this package runs from. The project is never installed, so
# this checkout is the code that ran, and git can describe it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Recorded when the checkout has no origin remote, as when it was unpacked
# from an archive. It repeats the URL in pyproject.toml, because the project
# is never installed and importlib.metadata has no copy of it to read.
_REPOSITORY_URL = "https://github.com/dmsellit/pdb-ai"


def _iso_utc(timestamp: float) -> str:
    """Format a POSIX timestamp as ISO 8601 in UTC, ending in ``Z``."""
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_receipt(
    path: Path,
    *,
    role: str,
    relative_to: Path | None = None,
    source_url: str | None = None,
    cache_present_before_acquisition: bool | None = None,
) -> dict[str, object]:
    """Return a receipt of a file's size, modification time, and SHA-256 now.

    The file is named by its path under ``relative_to``. It is named by its
    bare name when it lies elsewhere or no root is given, so no absolute path
    reaches the manifest. The path is written with forward slashes, so a
    manifest names a file the same way on every platform.
    ``cache_present_before_acquisition`` is recorded only when it is given. A
    file that is missing or cannot be read is recorded as not available
    instead of raising.
    """
    receipt: dict[str, object] = {
        "role": role,
        "path": (
            path.relative_to(relative_to).as_posix()
            if relative_to is not None and path.is_relative_to(relative_to)
            else path.name
        ),
        "source_url": source_url,
        "observed_at": _iso_utc(time.time()),
        "available": False,
    }
    if cache_present_before_acquisition is not None:
        receipt["cache_present_before_acquisition"] = cache_present_before_acquisition
    try:
        stat = path.stat()
        sha256 = sha256_file(path)
    except OSError:
        return receipt
    receipt.update(
        {
            "available": True,
            "size_bytes": stat.st_size,
            "file_modified_at": _iso_utc(stat.st_mtime),
            "sha256": sha256,
        }
    )
    return receipt


def _git_output(*args: str) -> str | None:
    """Return the trimmed output of a git command on the project checkout.

    None means git could not run or reported an error, as it does outside a
    repository.
    """
    try:
        # A branch name or remote URL can hold bytes the locale's encoding
        # cannot decode. A strict decode would raise and stop the run from
        # starting. Replacing those bytes keeps the rest of the value, and a
        # dirty checkout still reads as dirty.
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


# A remote URL can carry a user name and a token. Both are removed, so the
# token never reaches a manifest published beside the dataset.
def _without_credentials(url: str) -> str:
    if "://" in url:
        parts = urlsplit(url)
        if "@" not in parts.netloc:
            return url
        return urlunsplit(parts._replace(netloc=parts.netloc.rpartition("@")[2]))
    # Without a scheme, the remote is either a local path or git's scp form,
    # user:token@host:path, which urlsplit cannot read. The credentials end
    # at the last "@" before the first "/". A colon must follow that "@",
    # or the text is a local path whose name holds an "@".
    head, slash, rest = url.partition("/")
    _credentials, at, host = head.rpartition("@")
    if not at or ":" not in host:
        return url
    return host + slash + rest


def _code_receipt() -> dict[str, object]:
    origin = _git_output("config", "--get", "remote.origin.url")
    status = _git_output("status", "--porcelain", "--untracked-files=all")
    return {
        "application": "PDB-AI",
        "repository": _without_credentials(origin) if origin else _REPOSITORY_URL,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("symbolic-ref", "--quiet", "--short", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def _runtime_receipt() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
    }


def _dependency_receipt() -> list[dict[str, object]]:
    # Importing the pdbai package imports this module. An import at the top
    # would therefore load pdbai.licenses before python -m pdbai.licenses
    # runs it, and runpy would warn that it may behave unpredictably.
    from .licenses import (
        REQUIREMENTS_PATH,
        collect_dependency_licenses,
        parse_requirements,
    )

    packages = collect_dependency_licenses(parse_requirements(REQUIREMENTS_PATH))
    # What ran depends on which packages were present, not on their
    # licenses, so the license report's statuses reduce to installed or
    # missing.
    return [
        {
            "name": package.name,
            "version": package.version,
            "direct": package.direct,
            "status": "missing" if package.status == "missing" else "installed",
        }
        for package in packages
    ]


def _json_reference_data(
    reference_data: Mapping[str, object],
) -> dict[str, object]:
    # The manifest is dumped with default=str, which turns a value JSON cannot
    # hold into a string. That string would not read back as the value. A
    # round trip refuses such a value here, where its receipt can still be
    # named. A tuple, or a dict with keys that are not strings, passes
    # json.dumps but reads back changed, so the comparison refuses it.
    recorded: dict[str, object] = {}
    for name, receipt in reference_data.items():
        try:
            read_back: object = json.loads(json.dumps(receipt, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"The engine that gave reference data {name!r} returned a"
                f" value that is not JSON ({error})"
            ) from error
        if read_back != receipt:
            raise TypeError(
                f"The engine that gave reference data {name!r} returned a"
                " value that JSON does not read back as given, such as a"
                " tuple or a dict with keys that are not strings"
            )
        recorded[name] = read_back
    return recorded


def capture_execution_session(
    *, reference_data: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Return a receipt of the code, runtime, and dependencies of this process.

    The receipt records the checkout's repository URL, commit, branch, and
    whether it has uncommitted or untracked changes. It also records the
    Python version, operating system, and machine, and each package in the
    closure of requirements.txt as installed or missing, with its version.
    ``reference_data`` holds the receipts EngineRegistry.reference_data
    collected for the data files the run's engines ship with, keyed by name.
    It is recorded only when it is not empty. When the dependency inventory
    fails, ``dependencies`` is None and ``dependencies_error`` says why. A
    reference-data receipt that would not read back from JSON unchanged
    raises TypeError naming its key.
    """
    session: dict[str, object] = {
        "captured_at": _iso_utc(time.time()),
        "code": _code_receipt(),
        "runtime": _runtime_receipt(),
    }
    try:
        session["dependencies"] = _dependency_receipt()
    except Exception as exception:
        # The inventory reads the metadata of every installed package in the
        # closure, which pdbai does not control. Whatever goes wrong there is
        # recorded in its place rather than stopping the run from starting.
        session["dependencies"] = None
        session["dependencies_error"] = str(exception)
    if reference_data:
        session["reference_data"] = _json_reference_data(reference_data)
    return session


__all__ = [
    "PROJECT_ROOT",
    "capture_execution_session",
    "file_receipt",
    "sha256_file",
]
