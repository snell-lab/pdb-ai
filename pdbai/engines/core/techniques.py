"""The PDB's experimental techniques and X-ray sources, and entry matching.

One run processes one technique from TECHNIQUES. Intake refuses an entry
that reports methods but not that one, and admits one that reports none as
unverified. The techniques mirror the archive's _exptl.method vocabulary one
to one. The integrative methodology, which the PDB records elsewhere, is
added as one more. So no technique is ruled out by design, and an entry is
refused only for not matching the one the run chose. An entry that reports
several methods, such as a joint X-ray and neutron structure, matches each
of them.

Within X-ray diffraction, a run can also choose one radiation source from
XRAY_SOURCES. That is how XFEL entries are told apart, since the PDB files
them as X-ray diffraction. A chosen source refuses an entry that records
none, while ALL_XRAY_SOURCES admits every X-ray entry.

Reported values are compared in the archive's canonical spelling, and a
value that matches nothing in the vocabulary is ignored rather than guessed
at. The techniques an engine supports, and the run's own choice, are named
by the identifiers here, never by the archive's spelling.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Technique:
    """One experimental technique the PDB recognizes."""

    # The identifier runs and engines name the technique by.
    id: str
    # The archive's _exptl.method value, in canonical spelling.
    method: str
    # The full name shown to users.
    label: str
    # The acronym the field commonly uses, shown beside the full name wherever
    # the choices are listed. None when there is no widely used one.
    acronym: str | None = None

    @property
    def display_label(self) -> str:
        """Return the full name with its common acronym, when one exists."""
        return f"{self.label} ({self.acronym})" if self.acronym else self.label

    def to_dict(self) -> dict[str, object]:
        """Return the technique as a JSON object with snake_case keys."""
        return {
            "id": self.id,
            "label": self.label,
            "acronym": self.acronym,
            "display_label": self.display_label,
            "method": self.method,
        }


def _canonical_spelling(value: str) -> str:
    """Return ``value`` in the archive's spelling of methods and sources.

    The canonical spelling is upper case, with every run of whitespace made
    one space and none at either end.
    """
    return " ".join(str(value).split()).upper()


# Integrative entries are known by the archive's
# rcsb_entry_info.structure_determination_methodology rather than by
# _exptl.method. This made-up method value lets them be matched like any other.
INTEGRATIVE_METHOD = "INTEGRATIVE"

TECHNIQUES: tuple[Technique, ...] = (
    Technique("x_ray_diffraction", "X-RAY DIFFRACTION", "X-ray diffraction", "XRD"),
    Technique(
        "electron_microscopy",
        "ELECTRON MICROSCOPY",
        "Cryogenic electron microscopy",
        "cryo-EM",
    ),
    Technique(
        "solution_nmr",
        "SOLUTION NMR",
        "Solution nuclear magnetic resonance",
        "solution NMR",
    ),
    Technique(
        "electron_crystallography",
        "ELECTRON CRYSTALLOGRAPHY",
        "Electron crystallography",
        "MicroED",
    ),
    Technique("neutron_diffraction", "NEUTRON DIFFRACTION", "Neutron diffraction"),
    Technique(
        "solid_state_nmr",
        "SOLID-STATE NMR",
        "Solid-state nuclear magnetic resonance",
        "ssNMR",
    ),
    Technique(
        "solution_scattering",
        "SOLUTION SCATTERING",
        "Small-angle solution scattering",
        "SAXS/SANS",
    ),
    Technique("fiber_diffraction", "FIBER DIFFRACTION", "Fiber diffraction"),
    Technique("powder_diffraction", "POWDER DIFFRACTION", "Powder diffraction"),
    Technique("epr", "EPR", "Electron paramagnetic resonance", "EPR"),
    Technique("theoretical_model", "THEORETICAL MODEL", "Theoretical model"),
    Technique(
        "infrared_spectroscopy", "INFRARED SPECTROSCOPY", "Infrared spectroscopy", "IR"
    ),
    Technique(
        "fluorescence_transfer",
        "FLUORESCENCE TRANSFER",
        "Fluorescence resonance energy transfer",
        "FRET",
    ),
    Technique("integrative", INTEGRATIVE_METHOD, "Integrative/hybrid methods", "IHM"),
)

TECHNIQUE_IDS: tuple[str, ...] = tuple(technique.id for technique in TECHNIQUES)
_TECHNIQUES_BY_ID: Mapping[str, Technique] = {
    technique.id: technique for technique in TECHNIQUES
}


def get_technique(technique_id: str) -> Technique:
    """Return the technique ``technique_id`` names.

    An identifier that names no technique raises ValueError.
    """
    try:
        return _TECHNIQUES_BY_ID[technique_id]
    except KeyError:
        raise ValueError(
            f"technique must be one of {', '.join(TECHNIQUE_IDS)}; got {technique_id!r}"
        ) from None


def classify_methods(
    methods: Iterable[str],
    *,
    integrative: bool = False,
) -> tuple[str, ...]:
    """Return the technique identifiers an entry's reported methods match.

    Set ``integrative`` when the archive classifies the entry as integrative,
    which it records apart from the methods. Identifiers come back in
    vocabulary order. Blank and unrecognized values are ignored.
    """
    normalized = {
        _canonical_spelling(method) for method in methods if str(method).strip()
    }
    if integrative:
        normalized.add(INTEGRATIVE_METHOD)
    return tuple(
        technique.id for technique in TECHNIQUES if technique.method in normalized
    )


DEFAULT_TECHNIQUE = "x_ray_diffraction"
# The radiation-source filter is a second level inside X-ray diffraction only.
# It is the same technique as DEFAULT_TECHNIQUE, but the two answer different
# questions, so each has its own name.
XRAY_SOURCE_TECHNIQUE = "x_ray_diffraction"


@dataclass(frozen=True)
class RadiationSource:
    """One X-ray radiation source the PDB records."""

    # The identifier the xray_source setting names the source by.
    id: str
    # The archive's _diffrn_source.source value, in canonical spelling.
    source: str
    # The full name shown to users.
    label: str
    # The acronym the field commonly uses, or None when there is no widely
    # used one.
    acronym: str | None = None

    @property
    def display_label(self) -> str:
        """Return the full name with its common acronym, when one exists."""
        return f"{self.label} ({self.acronym})" if self.acronym else self.label

    def to_dict(self) -> dict[str, object]:
        """Return the source as a JSON object with snake_case keys."""
        return {
            "id": self.id,
            "label": self.label,
            "acronym": self.acronym,
            "display_label": self.display_label,
            "source": self.source,
        }


# The archive's _diffrn_source.source vocabulary, restricted to X-ray sources.
# Its reactors and spallation sources are left out, since they appear only as
# the neutron side of joint X-ray and neutron entries.
XRAY_SOURCES: tuple[RadiationSource, ...] = (
    RadiationSource("synchrotron", "SYNCHROTRON", "Synchrotron"),
    RadiationSource(
        "free_electron_laser",
        "FREE ELECTRON LASER",
        "X-ray free-electron laser",
        "XFEL",
    ),
    RadiationSource("rotating_anode", "ROTATING ANODE", "Rotating anode"),
    RadiationSource("sealed_tube", "SEALED TUBE", "Sealed tube"),
    RadiationSource("liquid_anode", "LIQUID ANODE", "Liquid anode"),
)

# The xray_source setting that admits every X-ray entry, including one that
# records no source. It names no RadiationSource.
ALL_XRAY_SOURCES = "all"
XRAY_SOURCE_IDS: tuple[str, ...] = (
    ALL_XRAY_SOURCES,
    *(source.id for source in XRAY_SOURCES),
)
_XRAY_SOURCES_BY_ID: Mapping[str, RadiationSource] = {
    source.id: source for source in XRAY_SOURCES
}


def get_xray_source(source_id: str) -> RadiationSource:
    """Return the X-ray source ``source_id`` names.

    An identifier that names no single source raises ValueError, and that
    includes ALL_XRAY_SOURCES.
    """
    try:
        return _XRAY_SOURCES_BY_ID[source_id]
    except KeyError:
        raise ValueError(
            f"xray_source must be one of {', '.join(_XRAY_SOURCES_BY_ID)}; "
            f"got {source_id!r}"
        ) from None


def normalize_source(value: str) -> str:
    """Return a reported _diffrn_source.source value in canonical spelling.

    Depositors spell one source several ways, such as "rotating-anode X-ray
    tube" and "Rotating Copper Anode", and each is mapped onto the archive's
    canonical value, so a filter and the output column agree. Any other value
    is only upper-cased and has its whitespace collapsed, never guessed at.
    """
    text = _canonical_spelling(value)
    words = text.replace("-", " ")
    if "ROTATING" in words and "ANODE" in words:
        return "ROTATING ANODE"
    if "FREE ELECTRON LASER" in words or words in {"XFEL", "X FEL"}:
        return "FREE ELECTRON LASER"
    if "SEALED" in words and "TUBE" in words:
        return "SEALED TUBE"
    if "LIQUID" in words and "ANODE" in words:
        return "LIQUID ANODE"
    if "SYNCHROTRON" in words:
        return "SYNCHROTRON"
    return text


def classify_sources(sources: Iterable[str]) -> tuple[str, ...]:
    """Return the X-ray source identifiers an entry's reported sources match.

    Identifiers come back in vocabulary order. Blank and unrecognized values
    are ignored.
    """
    normalized = {normalize_source(source) for source in sources if str(source).strip()}
    return tuple(source.id for source in XRAY_SOURCES if source.source in normalized)


__all__ = [
    "ALL_XRAY_SOURCES",
    "DEFAULT_TECHNIQUE",
    "INTEGRATIVE_METHOD",
    "TECHNIQUES",
    "TECHNIQUE_IDS",
    "XRAY_SOURCES",
    "XRAY_SOURCE_IDS",
    "XRAY_SOURCE_TECHNIQUE",
    "RadiationSource",
    "Technique",
    "classify_methods",
    "classify_sources",
    "get_technique",
    "get_xray_source",
    "normalize_source",
]
