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
from ..http_cache import GZIP_FILE, NotAvailableError, fetch_file
from ..map_sources import DENSITY_SOURCE_WWPDB, DENSITY_TOGGLE, density_reads
from .fetch import ARCHIVE_MISSING_STATUSES, validation_file_url

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
            url = validation_file_url(lower, file_suffix)
            path = services.cache_root / "validation_maps" / f"{lower}_{cache_suffix}"
            try:
                cached = fetch_file(
                    url,
                    path,
                    GZIP_FILE,
                    session=services.session,
                    timeout=services.timeout,
                    label=f"wwPDB validation map {lower.upper()}",
                    missing_statuses=ARCHIVE_MISSING_STATUSES,
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
]
