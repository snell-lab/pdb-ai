"""The entry result the AlphaFold acquisition engine hands on.

Downloading and mapping run in separate phases. The acquisition engine does
all the network work for an entry, and publishes one AFDBEntryResult under
AFDB_ENTRY. The annotation engine reads it to fill the residue columns.

The key is absent in two cases. An uploaded structure has no RCSB entry to
look up, and the acquisition of an entry may have raised. The annotation
engine tells the two apart by the source of the run.
"""

from __future__ import annotations

from ...core import ArtifactKey
from .models import AFDBEntryResult

AFDB_ENTRY = ArtifactKey(
    "afdb_entry",
    AFDBEntryResult,
    "Entity-aware UniProt mappings and AlphaFold models for the entry.",
)

__all__ = ["AFDB_ENTRY"]
