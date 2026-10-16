"""Acquisition of the entries of a download build, on worker threads.

Acquiring an entry starts with fetching its RCSB record and checking the
entry's technique against it. Unless the record refuses the entry, the
acquisition engines then run. They download the structure file and whatever
else the run's annotation engines need.

Acquisition spends most of its time waiting on the network, so it runs on
worker threads ahead of processing. Processing stays on the build's own
thread, one entry at a time, because each entry holds a whole structure in
memory. Acquisition runs at most two entries per worker ahead of processing.
That keeps every worker busy, and the build never holds many acquired
entries that processing has not reached.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import pandas as pd

from ..engines.catalog import get_engine_registry
from ..engines.core import (
    ENTRY_RECORD,
    STRUCTURE_CIF_PATH,
    STRUCTURE_METADATA,
    EngineContext,
    EngineExecutionError,
    EngineOptions,
    EnginePhase,
    EngineServices,
    SourceFile,
    run_engine_phase,
)
from ..engines.sources.rcsb import fetch_rcsb_metadata, rcsb_entry_url
from ..network import DEFAULT_TIMEOUT_SECONDS, create_retry_session
from ..provenance import file_receipt
from ..results import PdbProcessResult
from .build import DatasetBuild
from .processing import AcquiredEntry, process_entry
from .technique_intake import ReportedExperiment, decide_technique

DEFAULT_MAX_WORKERS = 8

# The most acquisitions per worker that are submitted but not yet handed to
# processing.
_IN_FLIGHT_PER_WORKER = 2


@dataclass(frozen=True)
class DownloadAcquirer:
    """The acquirer that every worker of a download build calls, per entry."""

    options: EngineOptions
    services: EngineServices
    working_directory: Path

    def acquire(self, pdb_id: str) -> AcquiredEntry:
        """Fetch the entry's RCSB record, then run the acquisition engines.

        An entry the record refuses is returned before any engine runs. This
        never raises. An unexpected error becomes the entry's download error,
        so it is recorded like any failed download and the build goes on.
        """
        try:
            return self._acquire(pdb_id)
        except Exception as exception:
            return AcquiredEntry(
                result=PdbProcessResult(
                    pdb_id=pdb_id,
                    source="download",
                    download_error=f"Download failed for {pdb_id}: {exception}",
                ),
                context=self._new_context(pdb_id),
            )

    def _acquire(self, pdb_id: str) -> AcquiredEntry:
        metadata_path = self.services.cache_root / f"{pdb_id.lower()}_metadata.json"
        # The RCSB record is fetched first because it is small and cached. It
        # also names the experimental method, which decides whether anything
        # else is worth downloading.
        metadata = fetch_rcsb_metadata(
            pdb_id,
            session=self.services.session,
            timeout=self.services.timeout,
            metadata_cache_path=metadata_path,
        )
        result = PdbProcessResult(
            pdb_id=pdb_id,
            source="download",
            metadata_ok=metadata.ok,
            metadata_message=metadata.message,
        )
        # Without a record there is nothing to decide on yet, so the entry is
        # downloaded and its structure file decides once parsed.
        if metadata.ok:
            decide_technique(
                pdb_id,
                ReportedExperiment.from_rcsb_metadata(metadata.metadata),
                self.options,
            ).apply_to(result)
        context = self._new_context(pdb_id)
        context.artifacts.put(STRUCTURE_METADATA, dict(metadata.metadata))
        if metadata.entry_record is not None:
            context.artifacts.put(ENTRY_RECORD, dict(metadata.entry_record))
        metadata_receipt = self._receipt(
            SourceFile(
                "rcsb_entry_metadata",
                metadata_path,
                rcsb_entry_url(pdb_id),
                metadata.cached,
            )
        )
        if result.refused():
            return AcquiredEntry(
                result=result, context=context, input_receipts=(metadata_receipt,)
            )

        try:
            phase_result = run_engine_phase(
                context, EnginePhase.ACQUISITION, registry=get_engine_registry()
            )
        except EngineExecutionError as exception:
            result.download_error = str(exception.__cause__ or exception)
            return AcquiredEntry(
                result=result, context=context, input_receipts=(metadata_receipt,)
            )
        result.download_ok = STRUCTURE_CIF_PATH in context.artifacts
        return AcquiredEntry(
            result=result,
            context=context,
            input_receipts=(
                metadata_receipt,
                *map(self._receipt, phase_result.source_files),
            ),
            acquisition_result=phase_result,
        )

    def _new_context(self, pdb_id: str) -> EngineContext:
        return EngineContext(
            pdb_id=pdb_id,
            atom_dataframe=pd.DataFrame(),
            options=self.options,
            source="download",
            services=self.services,
            structure_context="asymmetric_unit",
        )

    def _receipt(self, source_file: SourceFile) -> dict[str, object]:
        return file_receipt(
            source_file.path,
            role=source_file.role,
            relative_to=self.working_directory,
            source_url=source_file.source_url,
            cache_present_before_acquisition=source_file.cached_before,
        )


def acquire_concurrently(
    acquire: Callable[[str], AcquiredEntry],
    pdb_ids: Iterable[str],
    *,
    build: DatasetBuild,
    worker_count: int,
) -> Iterator[AcquiredEntry]:
    """Run ``acquire`` on worker threads and yield each entry once acquired.

    Of the entries ready at once, the one submitted first comes first. When
    none is ready, the oldest in flight is reported at the download stage.
    Once the build is cancelled, nothing more is yielded. Acquisitions not yet
    started are cancelled, and the generator waits for those running.
    """
    unsubmitted = iter(pdb_ids)
    in_flight: dict[Future[AcquiredEntry], str] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        try:
            while not build.cancel_requested():
                room = worker_count * _IN_FLIGHT_PER_WORKER - len(in_flight)
                for pdb_id in islice(unsubmitted, room):
                    in_flight[executor.submit(acquire, pdb_id)] = pdb_id
                if not in_flight:
                    return
                if not any(future.done() for future in in_flight):
                    # The executor hands out work in submission order, so a
                    # worker has already taken the oldest acquisition in
                    # flight.
                    build.report_progress("download", next(iter(in_flight.values())))
                    wait(in_flight, return_when=FIRST_COMPLETED)
                    if build.cancel_requested():
                        return
                finished = next(future for future in in_flight if future.done())
                del in_flight[finished]
                yield finished.result()
        finally:
            for future in in_flight:
                future.cancel()


def process_downloads(build: DatasetBuild, *, max_workers: int | None) -> None:
    """Acquire, process, and record each of the build's entries to process.

    Acquisition runs on up to max_workers threads, or DEFAULT_MAX_WORKERS
    when it is None, and never on more threads than there are entries.
    Requests use the network module's default timeout and retries.
    Processing runs on the calling thread.
    """
    ids_to_process = build.ids_to_process
    # An executor cannot have zero workers.
    if not ids_to_process:
        return
    worker_count = min(
        DEFAULT_MAX_WORKERS if max_workers is None else max_workers,
        len(ids_to_process),
    )
    # Closing the session releases its pooled sockets when the build ends,
    # not at garbage collection.
    with create_retry_session() as session:
        acquirer = DownloadAcquirer(
            options=build.options,
            services=EngineServices(
                session=session,
                timeout=DEFAULT_TIMEOUT_SECONDS,
                # Every download is cached here, and later runs in the same
                # working directory reuse it until it expires.
                cache_root=build.working_directory / "mmcif",
            ),
            working_directory=build.working_directory,
        )
        for acquired in acquire_concurrently(
            acquirer.acquire, ids_to_process, build=build, worker_count=worker_count
        ):
            process_entry(build, acquired)


__all__ = [
    "DEFAULT_MAX_WORKERS",
    "DownloadAcquirer",
    "acquire_concurrently",
    "process_downloads",
]
