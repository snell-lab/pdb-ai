"""The built-in engines and the registry every run draws them from.

BUILTIN_ENGINES lists each engine PDB-AI ships with, and get_engine_registry
returns the one registry built from them. Every call returns that same
object, so code outside PDB-AI adds an engine of its own by registering it
there. It then calls validate, since registering checks an engine only
against the engines already registered.

The registry is built and validated when this module is first imported. So
a declaration that clashes with another engine's, names an engine or setting
nobody declares, or closes a dependency cycle, fails at import rather than
partway through a run.

The order of the list is the order engines are shown in, which puts their
column groups and settings in order wherever those are listed. The order
engines run in follows their dependencies instead.
"""

from __future__ import annotations

from .analyses.cofactors.engine import COFACTOR_ENGINE
from .analyses.density.engine import DENSITY_ENGINE
from .analyses.disulfides.engine import DISULFIDE_ENGINE
from .analyses.metal_sites.engine import METAL_SITE_ENGINE
from .analyses.rabdam.engine import RABDAM_ENGINE
from .analyses.residue_sites.engine import RESIDUE_SITE_ENGINE
from .analyses.sasa.engine import SASA_ENGINE
from .analyses.selenium_halogens.engine import SELENIUM_HALOGEN_ENGINE
from .analyses.site_environment.engine import SITE_ENVIRONMENT_ENGINE
from .core import Engine, EngineRegistry
from .sources.alphafold.engine import (
    ALPHAFOLD_ACQUISITION_ENGINE,
    ALPHAFOLD_ANNOTATION_ENGINE,
)
from .sources.pdb_redo.engine import (
    PDB_REDO_ACQUISITION_ENGINE,
    PDB_REDO_COMPARISON_ENGINE,
    PDB_REDO_MAP_ACQUISITION_ENGINE,
    PDB_REDO_STRUCTURE_ENGINE,
)
from .sources.rcsb.engine import RCSB_ACQUISITION_ENGINE
from .sources.validation.engine import (
    VALIDATION_ACQUISITION_ENGINE,
    VALIDATION_ANNOTATION_ENGINE,
)
from .sources.validation.maps import VALIDATION_MAP_ACQUISITION_ENGINE

# Among engines that do not depend on one another, this order also decides
# which runs first. Report fields follow the order engines run in.
BUILTIN_ENGINES: tuple[Engine, ...] = (
    RCSB_ACQUISITION_ENGINE,
    PDB_REDO_ACQUISITION_ENGINE,
    PDB_REDO_STRUCTURE_ENGINE,
    PDB_REDO_COMPARISON_ENGINE,
    PDB_REDO_MAP_ACQUISITION_ENGINE,
    VALIDATION_MAP_ACQUISITION_ENGINE,
    RABDAM_ENGINE,
    SASA_ENGINE,
    DENSITY_ENGINE,
    DISULFIDE_ENGINE,
    METAL_SITE_ENGINE,
    RESIDUE_SITE_ENGINE,
    SELENIUM_HALOGEN_ENGINE,
    COFACTOR_ENGINE,
    SITE_ENVIRONMENT_ENGINE,
    ALPHAFOLD_ACQUISITION_ENGINE,
    ALPHAFOLD_ANNOTATION_ENGINE,
    VALIDATION_ACQUISITION_ENGINE,
    VALIDATION_ANNOTATION_ENGINE,
)

_registry = EngineRegistry(BUILTIN_ENGINES)
_registry.validate()


def get_engine_registry() -> EngineRegistry:
    """Return the registry of the built-in engines, shared by every caller."""
    return _registry


__all__ = ["BUILTIN_ENGINES", "get_engine_registry"]
