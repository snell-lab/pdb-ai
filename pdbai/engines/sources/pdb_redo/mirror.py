r"""A local copy of PDB-Redo's databank, read before the network.

PDB-Redo offers its databank by rsync as well as file by file, and a run
over thousands of entries is better served by a copy made once. A copy
placed in the download cache, under MIRROR_DIRECTORY, is read before any
request is made. Its layout is PDB-Redo's own, the directory of an entry
beneath the two middle characters of its identifier, as in
pdb_redo_mirror/cb/1cbs/1cbs_final.cif. That is what

    rsync -av --exclude=attic rsync://rsync.pdb-redo.eu/pdb-redo/ \
        <workdir>/mmcif/pdb_redo_mirror/

writes, for the whole databank or for the entries a file list names.

A file of the copy is used as it stands. It does not expire as a downloaded
file does, since the person who made the copy decides when to renew it. It
must still pass the content check a download would, an mmCIF data block or
the MTZ signature, and the engine fails when it does not. An entry the copy
lacks is requested from PDB-Redo as it would be without a copy.
"""

from __future__ import annotations

from pathlib import Path

MIRROR_DIRECTORY = "pdb_redo_mirror"


def mirror_file(cache_root: Path, pdb_id: str, file_name: str) -> Path | None:
    """Return an entry's file from the local copy, or None when it has none.

    ``pdb_id`` is the lower-case entry identifier and ``file_name`` the name
    PDB-Redo gives the file, such as 1cbs_final.mtz. An empty file counts as
    absent, since an interrupted copy leaves one.
    """
    path = cache_root / MIRROR_DIRECTORY / pdb_id[1:3] / pdb_id / file_name
    try:
        if path.is_file() and path.stat().st_size > 0:
            return path
    except OSError:
        return None
    return None


__all__ = ["MIRROR_DIRECTORY", "mirror_file"]
