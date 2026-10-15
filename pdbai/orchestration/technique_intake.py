"""Technique intake, which keeps entries of the wrong technique out of a run.

A run builds a dataset for one experimental technique, such as X-ray
diffraction. It may also be narrowed to one X-ray source, such as an XFEL.
The run's engines and columns are only valid for that technique. An entry
determined another way, such as by cryo-EM, is therefore refused instead of
processed.

An entry's RCSB record decides first, because it is fetched before anything
else is downloaded. When there is no record, or the record names no method,
the structure file decides once it is parsed. Both name methods and sources
in the archive's own vocabulary, that of _exptl.method and
_diffrn_source.source.

A refusal is an expected outcome of the filter, not a failure. The entry is
recorded as refused, with the reason and what it reported, and the build goes
on. A refused entry does not count as done, so a resumed run checks it again.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ..engines.core import (
    ALL_XRAY_SOURCES,
    EngineOptions,
    Technique,
    classify_methods,
    classify_sources,
    get_technique,
    get_xray_source,
    normalize_source,
)
from ..json_values import str_list
from ..results import PdbProcessResult, TechniqueStatus
from ..structure.mmcif_parser import AtomSiteParseResult

# The most entries each intake summary names, for refused and for unverified
# entries alike.
_NAMED_ENTRIES_LIMIT = 20


def _canonical_radiation_sources(radiation_sources: Iterable[str]) -> tuple[str, ...]:
    # A file lists a source once for each dataset collected with it. Repeats
    # are dropped and variant spellings unified before the sources are kept.
    return tuple(dict.fromkeys(map(normalize_source, radiation_sources)))


@dataclass(frozen=True)
class ReportedExperiment:
    """What an entry reports about how its structure was determined."""

    experimental_methods: tuple[str, ...] = ()
    radiation_sources: tuple[str, ...] = ()
    # An integrative model combines several kinds of data and may list no
    # experimental method of its own, so it is marked separately.
    is_integrative: bool = False

    @classmethod
    def from_rcsb_metadata(cls, metadata: Mapping[str, object]) -> ReportedExperiment:
        """Return what an entry's RCSB record reports.

        ``metadata`` is the record as fetch_rcsb_metadata extracts it.
        """
        return cls(
            experimental_methods=tuple(str_list(metadata.get("experimental_methods"))),
            radiation_sources=_canonical_radiation_sources(
                str_list(metadata.get("radiation_sources"))
            ),
            is_integrative=(
                metadata.get("structure_determination_methodology") == "integrative"
            ),
        )

    @classmethod
    def from_parsed_file(cls, parsed: AtomSiteParseResult) -> ReportedExperiment:
        """Return what a parsed structure file's own categories report."""
        return cls(
            experimental_methods=parsed.experimental_methods,
            radiation_sources=_canonical_radiation_sources(parsed.radiation_sources),
            is_integrative=parsed.is_integrative,
        )

    @property
    def has_method(self) -> bool:
        """Return whether any method is reported, integrative included."""
        return bool(self.experimental_methods) or self.is_integrative


@dataclass(frozen=True)
class TechniqueDecision:
    """Whether what an entry reports admits it to the run, and why not."""

    status: TechniqueStatus
    # Why the entry was refused or left unverified. None when it matched.
    message: str | None
    reported: ReportedExperiment

    def apply_to(self, result: PdbProcessResult) -> None:
        """Copy this decision and what the entry reported onto ``result``.

        A refusal also marks the entry refused.
        """
        result.technique_status = self.status
        result.technique_message = self.message
        result.experimental_methods = list(self.reported.experimental_methods)
        result.radiation_sources = list(self.reported.radiation_sources)
        if self.status == "refused":
            result.state = "refused"


def _technique_refusal(
    pdb_id: str, reported: ReportedExperiment, technique: Technique
) -> str | None:
    if not reported.has_method or technique.id in classify_methods(
        reported.experimental_methods, integrative=reported.is_integrative
    ):
        return None
    methods = "; ".join(reported.experimental_methods) or "an integrative model"
    return (
        f"{pdb_id} reports {methods}, not {technique.label}; refused at intake "
        f"for technique '{technique.id}'."
    )


def _xray_source_refusal(
    pdb_id: str, reported: ReportedExperiment, xray_source: str
) -> str | None:
    if xray_source == ALL_XRAY_SOURCES:
        return None
    wanted = get_xray_source(xray_source)
    if not reported.radiation_sources:
        return (
            f"{pdb_id} reports no radiation source, so it cannot be verified as "
            f"{wanted.label}; refused at intake for xray_source '{wanted.id}'."
        )
    if wanted.id in classify_sources(reported.radiation_sources):
        return None
    return (
        f"{pdb_id} reports {'; '.join(reported.radiation_sources)}, not "
        f"{wanted.label}; refused at intake for xray_source '{wanted.id}'."
    )


def decide_technique(
    pdb_id: str, reported: ReportedExperiment, options: EngineOptions
) -> TechniqueDecision:
    """Decide whether what an entry reports admits it to the run.

    The entry is checked against the run's technique and its X-ray source.
    Every run filters by technique, so an entry that reports no method is
    admitted as unverified rather than refused. Refusing it would shrink
    ordinary runs by dropping entries whose depositors left the category
    empty. A user turns the X-ray source filter on to narrow a run. There an
    entry that reports no source is refused, because its source cannot be
    confirmed, and its message says so. An entry that fails both checks is
    refused for its technique.
    """
    technique = get_technique(options.technique)
    refusal = _technique_refusal(pdb_id, reported, technique) or _xray_source_refusal(
        pdb_id, reported, options.xray_source
    )
    if refusal is not None:
        return TechniqueDecision("refused", refusal, reported)
    if reported.has_method:
        return TechniqueDecision("matched", None, reported)
    return TechniqueDecision(
        "unverified",
        f"{pdb_id} reports no experimental method, so it cannot be verified as "
        f"{technique.label}; admitted unverified at intake for technique "
        f"'{technique.id}'.",
        reported,
    )


def _entry_count(pdb_ids: Sequence[str]) -> str:
    return "1 entry" if len(pdb_ids) == 1 else f"{len(pdb_ids)} entries"


def _named_entries(pdb_ids: Sequence[str]) -> str:
    named = ", ".join(pdb_ids[:_NAMED_ENTRIES_LIMIT])
    unnamed = len(pdb_ids) - _NAMED_ENTRIES_LIMIT
    return f"{named}, and {unnamed} more" if unnamed > 0 else named


def intake_warnings(
    results: Sequence[PdbProcessResult], options: EngineOptions
) -> list[str]:
    """Return the run's summary warnings for refused and unverified entries.

    Each kind is summarized once rather than warned per entry, because a
    cohort pulled from a mixed list can rightly refuse hundreds of entries.
    """
    settings = f"technique '{options.technique}'"
    if options.xray_source != ALL_XRAY_SOURCES:
        settings += f", xray_source '{options.xray_source}'"
    warnings: list[str] = []
    refused = [result.pdb_id for result in results if result.refused()]
    if refused:
        warnings.append(
            f"Technique intake refused {_entry_count(refused)} ({settings}): "
            f"{_named_entries(refused)}. See the run report for each entry's "
            "reported methods and radiation sources."
        )
    unverified = [
        result.pdb_id for result in results if result.technique_status == "unverified"
    ]
    if unverified:
        warnings.append(
            f"Technique intake admitted {_entry_count(unverified)} unverified "
            f"({settings}), with no experimental method reported: "
            f"{_named_entries(unverified)}."
        )
    return warnings


__all__ = [
    "ReportedExperiment",
    "TechniqueDecision",
    "decide_technique",
    "intake_warnings",
]
