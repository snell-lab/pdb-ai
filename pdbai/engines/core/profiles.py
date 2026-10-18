"""The identifiers of the column profiles a column can belong to.

A column names the profiles it belongs to in ColumnSpec.profiles, and an
engine may add glob selectors per profile in EngineSpec.profile_patterns. The
profiles themselves, with their names, purposes, and the engines they switch
on, live in pdbai.profiles. Only the identifiers are here. That lets an
engine declare membership without importing pdbai.profiles, which imports
the engine registry and the schema.

Only the minimal profile is declared column by column. The profile "all"
keeps every column a run writes, and "standard" every column PDB-AI declares,
so no column needs to name either. MINIMAL_PROFILE is the value a column of
the minimal profile gives ColumnSpec.profiles, and DECLARED_PROFILE_IDS the
profiles a column may name.
"""

from __future__ import annotations

PROFILE_STANDARD = "standard"
PROFILE_MINIMAL = "minimal"

MINIMAL_PROFILE = frozenset({PROFILE_MINIMAL})
DECLARED_PROFILE_IDS = MINIMAL_PROFILE

__all__ = [
    "DECLARED_PROFILE_IDS",
    "MINIMAL_PROFILE",
    "PROFILE_MINIMAL",
    "PROFILE_STANDARD",
]
