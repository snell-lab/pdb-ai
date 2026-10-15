"""The map coefficients the wwPDB validation pipeline computes for an entry.

For an X-ray entry deposited with structure factors, the wwPDB validation
report comes with two reflection files in mmCIF form. One holds the
coefficients of the 2mFo−DFc map (pdbx_FWT, pdbx_PHWT) and the other those of
the mFo−DFc difference map (pdbx_DELFWT, pdbx_DELPHWT). Both were calculated
about the deposited model, with no further refinement, so they describe the
model the atom table holds.

The map source engine downloads both files into the cache, as the archive
compresses them. It runs only for the density engine, and only when that
engine reads the wwPDB maps. An entry the archive has no such files for,
which it signals with HTTP 403 or 404, is an expected outcome and not a
failure. The engine then publishes neither path, and the density engine
reports why its columns are empty.
"""

from __future__ import annotations

from pathlib import Path

from ....identifiers import normalize_pdb_id
from ...core import (
    ArtifactKey,
    EngineContext,
    EnginePhase,
    EngineResult,
    EngineSpec,
    SourceFile,
)
from ..http_cache import NotAvailableError, fetch_gzip
from ..map_sources import DENSITY_SOURCE_WWPDB, DENSITY_TOGGLE, density_reads

VALIDATION_2FOFC_MAP_PATH = ArtifactKey(
    "validation_2fofc_map_path",
    Path,
    "The wwPDB validation 2mFo−DFc map coefficients of the entry, when the "
    "archive has them.",
)
VALIDATION_FOFC_MAP_PATH = ArtifactKey(
    "validation_fofc_map_path",
    Path,
    "The wwPDB validation mFo−DFc map coefficients of the entry, when the "
    "archive has them.",
)

# The wwPDB computes these maps for X-ray entries only.
_VALIDATION_MAP_TECHNIQUES = frozenset({"x_ray_diffraction"})

# files.rcsb.org answers 403, not 404, for a file it does not hold.
_MISSING_STATUSES = frozenset({403})

# Each map's file name suffix in the archive, its cache name suffix, the
# artifact it is published under, and its role in the manifest's receipts.
_MAP_FILES = (
    (
        "validation_2fo-fc_map_coef.cif.gz",
        "2fofc_map_coef.cif.gz",
        VALIDATION_2FOFC_MAP_PATH,
        "wwpdb_validation_2fofc_map",
    ),
    (
        "validation_fo-fc_map_coef.cif.gz",
        "fofc_map_coef.cif.gz",
        VALIDATION_FOFC_MAP_PATH,
        "wwpdb_validation_fofc_map",
    ),
)


def validation_map_url(pdb_id: str, file_suffix: str) -> str:
    """Return the archive URL of one validation map file of an entry."""
    lower = normalize_pdb_id(pdb_id).lower()
    return (
        "https://files.rcsb.org/pub/pdb/validation_reports/"
        f"{lower[1:3]}/{lower}/{lower}_{file_suffix}"
    )


class ValidationMapAcquisitionEngine:
    """The engine that downloads the wwPDB validation map coefficients."""

    spec = EngineSpec(
        name="validation_map_source",
        display_name="wwPDB validation map coefficients",
        phase=EnginePhase.ACQUISITION,
        toggle=DENSITY_TOGGLE,
        enabled_if=lambda options: density_reads(options, DENSITY_SOURCE_WWPDB),
        dependencies=("rcsb_source",),
        provides=(VALIDATION_2FOFC_MAP_PATH, VALIDATION_FOFC_MAP_PATH),
        techniques=_VALIDATION_MAP_TECHNIQUES,
        sources=frozenset({"download"}),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the paths of the entry's two map files, unless the archive lacks one.

        Both paths are published or neither is, since the density engine
        reads the two maps together. Any failed download but the answer that
        the archive has no such file raises RuntimeError.
        """
        services = context.require_services()
        lower = normalize_pdb_id(context.pdb_id).lower()
        paths: dict[ArtifactKey[Path], Path] = {}
        source_files: list[SourceFile] = []
        available = True
        for file_suffix, cache_suffix, key, role in _MAP_FILES:
            url = validation_map_url(lower, file_suffix)
            path = services.cache_root / "validation_maps" / f"{lower}_{cache_suffix}"
            try:
                cached = fetch_gzip(
                    url,
                    path,
                    session=services.session,
                    timeout=services.timeout,
                    label=f"wwPDB validation map {lower.upper()}",
                    missing_statuses=_MISSING_STATUSES,
                )
            except NotAvailableError:
                available = False
                source_files.append(SourceFile(role, path, url, False))
                continue
            paths[key] = path
            source_files.append(SourceFile(role, path, url, cached))
        return EngineResult(
            artifacts=dict(paths) if available else {},
            source_files=tuple(source_files),
        )


VALIDATION_MAP_ACQUISITION_ENGINE = ValidationMapAcquisitionEngine()

__all__ = [
    "VALIDATION_2FOFC_MAP_PATH",
    "VALIDATION_FOFC_MAP_PATH",
    "VALIDATION_MAP_ACQUISITION_ENGINE",
    "ValidationMapAcquisitionEngine",
    "validation_map_url",
]
