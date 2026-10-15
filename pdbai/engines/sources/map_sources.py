"""Which maps a run's density engine reads, as the map source engines see it.

The density engine declares the density_map_source setting, and each map
source engine runs only when the density engine will read its maps. The
setting's name and values are kept here, beside the source engines, so that
neither they nor the density engine import the other.
"""

from __future__ import annotations

from ..core import EngineOptions

DENSITY_TOGGLE = "include_density"
DENSITY_MAP_SOURCE_OPTION = "density_map_source"
# PDB-Redo's maps, calculated about its re-refined model.
DENSITY_SOURCE_PDB_REDO = "pdb_redo"
# The wwPDB validation maps, calculated about the deposited model.
DENSITY_SOURCE_WWPDB = "wwpdb"
DENSITY_SOURCE_BOTH = "both"
DENSITY_MAP_SOURCES = (
    DENSITY_SOURCE_PDB_REDO,
    DENSITY_SOURCE_WWPDB,
    DENSITY_SOURCE_BOTH,
)


def density_source_selected(options: EngineOptions, source: str) -> bool:
    """Return whether the density map source setting is ``source`` or both."""
    return options.value(DENSITY_MAP_SOURCE_OPTION) in (source, DENSITY_SOURCE_BOTH)


def density_reads(options: EngineOptions, source: str) -> bool:
    """Return whether a run's density engine reads the maps of ``source``.

    ``source`` is DENSITY_SOURCE_PDB_REDO or DENSITY_SOURCE_WWPDB. The engine
    reads them when include_density is on and the density map source is that
    one or both.
    """
    return options.flag(DENSITY_TOGGLE) and density_source_selected(options, source)


__all__ = [
    "DENSITY_MAP_SOURCES",
    "DENSITY_MAP_SOURCE_OPTION",
    "DENSITY_SOURCE_BOTH",
    "DENSITY_SOURCE_PDB_REDO",
    "DENSITY_SOURCE_WWPDB",
    "DENSITY_TOGGLE",
    "density_reads",
    "density_source_selected",
]
