"""The identifiers of the curated column profiles a column can belong to.

A column names the profiles it belongs to in ColumnSpec.profiles, and an
engine may add glob selectors per profile in EngineSpec.profile_patterns. The
profiles themselves, with their names, purposes, and the engines they switch
on, live in pdbai.profiles. Only the identifiers are here. That lets an
engine declare membership without importing pdbai.profiles, which imports
the engine registry and the schema.

The profile "all" has no identifier here, since it keeps every column and no
column needs to name it. CURATED_PROFILE_IDS is every other profile.
"""

from __future__ import annotations

PROFILE_RESEARCH_READY = "research_ready"
PROFILE_RADIATION_SENSITIVITY = "radiation_sensitivity"
PROFILE_ML_MINIMAL = "ml_minimal"
PROFILE_STRUCTURE_QUALITY = "structure_quality"

CURATED_PROFILE_IDS = frozenset(
    {
        PROFILE_RESEARCH_READY,
        PROFILE_RADIATION_SENSITIVITY,
        PROFILE_ML_MINIMAL,
        PROFILE_STRUCTURE_QUALITY,
    }
)

__all__ = [
    "CURATED_PROFILE_IDS",
    "PROFILE_ML_MINIMAL",
    "PROFILE_RADIATION_SENSITIVITY",
    "PROFILE_RESEARCH_READY",
    "PROFILE_STRUCTURE_QUALITY",
]
