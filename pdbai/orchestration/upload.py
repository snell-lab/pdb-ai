"""Naming each uploaded mmCIF file as an entry, then processing it.

An uploaded file has no archive record, so the run gives it an entry id taken
from its file name. The id also names the entry's partial files and fills the
pdb_id column of its rows. It is therefore made portable. It holds only
characters that are safe in a file name on every operating system and in a
CSV cell, and its letters are capitals, as in a PDB ID.

The web app saves each upload under a stem made by the same rule, with the
letters left in their case. A file's id is therefore its saved stem in
capitals.

An upload needs no acquisition, and its file alone decides its technique
once it is parsed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..engines.core import STRUCTURE_CIF_PATH, EngineContext, EngineOptions
from ..provenance import file_receipt, sha256_file
from ..results import PdbProcessResult
from ..runs.storage import is_windows_reserved_name
from .build import DatasetBuild
from .processing import AcquiredEntry, process_entry

# An id keeps only ASCII letters, digits, ".", "_" and "-". A "/" or "\"
# would split the name into folders. Windows forbids characters such as ":",
# "*" and "?". Spaces and commas cause trouble in shells and CSV files.
_UNPORTABLE_ID_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")

# In characters, before any numeric suffix. Every character is ASCII, so this
# is also the length in bytes, well within the 255 a file name may have.
_MAX_BASE_ID_LENGTH = 100


@dataclass(frozen=True)
class UploadedFile:
    """One uploaded mmCIF file and the entry id the run gives it."""

    pdb_id: str
    path: Path
    content_sha256: str
    size_bytes: int

    def to_settings(self) -> dict[str, object]:
        """Return the file's identity as recorded in the run's settings.

        The settings are hashed when the run is claimed, so resuming it with a
        file whose content changed is refused instead of mixing two versions.
        """
        return {
            "structure_id": self.pdb_id,
            "source_filename": self.path.name,
            "content_sha256": self.content_sha256,
            "size_bytes": self.size_bytes,
        }


def _portable(text: str) -> str:
    # Every other run of characters becomes one "_". The result is empty when
    # the text holds no ASCII letter or digit.
    portable = _UNPORTABLE_ID_CHARACTERS.sub("_", text).strip("._-")
    return portable[:_MAX_BASE_ID_LENGTH].rstrip("._-")


def _unreserved(name: str) -> str:
    # No file can have a name Windows reserves for a device, such as CON, so
    # such a name gets a "_" after it.
    return f"{name}_" if is_windows_reserved_name(name) else name


def portable_upload_stem(stem: str) -> str:
    """Return the portable stem an uploaded file is saved under.

    ``stem`` is the stem of the name the file was uploaded with. The rule is
    the one that makes an entry id from a file name, but letters keep their
    case. A ``stem`` with no ASCII letter or digit raises ValueError, since
    its file would have no name of its own.
    """
    portable = _portable(stem)
    if not portable:
        raise ValueError(f"File name stem {stem!r} has no ASCII letter or digit.")
    return _unreserved(portable)


def _base_id(path: Path) -> str:
    filename = path.name
    if filename.lower().endswith(".gz"):
        filename = filename[:-3]
    return _unreserved(_portable(Path(filename).stem.upper()) or "UPLOADED")


def identify_uploads(paths: Sequence[Path]) -> list[UploadedFile]:
    """Give each file an entry id taken from its name, and hash its content.

    An id another file already took gets the first free suffix of _2, _3, and
    so on, so every file is an entry of its own. A name with no ASCII letter
    or digit gives the id UPLOADED. A file that cannot be read raises OSError.
    """
    taken: set[str] = set()
    uploads: list[UploadedFile] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        base_id = _base_id(path)
        pdb_id = base_id
        suffix = 1
        while pdb_id in taken:
            suffix += 1
            pdb_id = f"{base_id}_{suffix}"
        taken.add(pdb_id)
        uploads.append(
            UploadedFile(
                pdb_id=pdb_id,
                path=path,
                content_sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return uploads


def _acquired_entry(upload: UploadedFile, options: EngineOptions) -> AcquiredEntry:
    """Return an upload as an acquired entry.

    The file is the entry's only input and is in hand from the start, which
    is what download_ok records.
    """
    context = EngineContext(
        pdb_id=upload.pdb_id,
        atom_dataframe=pd.DataFrame(),
        options=options,
        source="upload",
        structure_context="as_provided",
    )
    context.artifacts.put(STRUCTURE_CIF_PATH, upload.path)
    return AcquiredEntry(
        result=PdbProcessResult(
            pdb_id=upload.pdb_id, source="upload", download_ok=True
        ),
        context=context,
        input_receipts=(file_receipt(upload.path, role="uploaded_mmcif"),),
    )


def process_uploads(build: DatasetBuild, uploads: Sequence[UploadedFile]) -> None:
    """Process and record each of the build's uploads to process, one at a time.

    ``uploads`` must hold every entry of the build, as identify_uploads named
    them. Once the build is cancelled, no further upload is processed.
    """
    uploads_by_id = {upload.pdb_id: upload for upload in uploads}
    for pdb_id in build.ids_to_process:
        if build.cancel_requested():
            return
        process_entry(build, _acquired_entry(uploads_by_id[pdb_id], build.options))


__all__ = [
    "UploadedFile",
    "identify_uploads",
    "portable_upload_stem",
    "process_uploads",
]
