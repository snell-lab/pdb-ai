"""Dataset builds from PDB identifiers or uploaded mmCIF files.

api.py holds the two public builders, one for identifiers and one for files.
Each checks its arguments before anything is written, claims the run, records
the run's settings, and passes the entries to download.py or upload.py.

download.py acquires each entry from the RCSB on parallel workers. It fetches
the entry's record first, so an entry of the wrong technique is refused before
its files are downloaded.

upload.py gives each uploaded file an entry ID taken from its name.

technique_intake.py decides whether an entry belongs to the run's experimental
technique, from its RCSB record or else from its structure file.

processing.py takes one acquired entry through parsing, the technique check,
and annotation, whatever the entry's source.

annotation.py runs the annotation engines, which add their columns to the
entry's atom rows.

build.py records how each entry ended and settles the run when the build
ends.

A build records each entry before it starts the next, so its memory stays
bounded and an interrupted run resumes where it stopped. Only acquisition runs
ahead of processing.
"""

from .api import build_atom_site_dataset, build_atom_site_dataset_from_mmcif_files
from .download import DEFAULT_MAX_WORKERS

__all__ = [
    "DEFAULT_MAX_WORKERS",
    "build_atom_site_dataset",
    "build_atom_site_dataset_from_mmcif_files",
]
