"""The engine that downloads an entry's mmCIF file from RCSB.

The dataset's atom rows are parsed from this file, and every other engine
works from those rows. So every run on downloaded entries requires this
engine, and it is the only required one. Its failure fails the entry. Any
other engine that fails leaves only its own columns empty.

The file's path is published as STRUCTURE_CIF_PATH. The orchestration
parses the atom rows from it, and RABDAM scores it directly.
"""

from __future__ import annotations

from ....identifiers import normalize_pdb_id
from ...core import (
    STRUCTURE_CIF_PATH,
    EngineContext,
    EnginePhase,
    EngineResult,
    EngineSpec,
    SourceFile,
)
from ..http_cache import fetch_mmcif


class RcsbAcquisitionEngine:
    """The engine that downloads the entry's mmCIF file from RCSB."""

    spec = EngineSpec(
        name="rcsb_source",
        display_name="RCSB source",
        phase=EnginePhase.ACQUISITION,
        default_enabled=True,
        required=True,
        provides=(STRUCTURE_CIF_PATH,),
        sources=frozenset({"download"}),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the path of the entry's file, downloading it when needed.

        A failed download raises RuntimeError, or NotAvailableError when RCSB
        has no file for the entry. A ``context`` without network services
        raises RuntimeError too.
        """
        services = context.require_services()
        # Normalizing first keeps a malformed ID out of the cache path.
        pdb_id = normalize_pdb_id(context.pdb_id).lower()
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        path = services.cache_root / "rcsb" / f"{pdb_id}.cif"
        cached = fetch_mmcif(
            url,
            path,
            session=services.session,
            timeout=services.timeout,
            label=f"RCSB {pdb_id.upper()}",
        )
        return EngineResult(
            artifacts={STRUCTURE_CIF_PATH: path},
            source_files=(SourceFile("rcsb_mmcif", path, url, cached),),
        )


RCSB_ACQUISITION_ENGINE = RcsbAcquisitionEngine()

__all__ = ["RCSB_ACQUISITION_ENGINE"]
