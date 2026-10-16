"""The dataset's curated fields, read out of an RCSB Core Entry record.

Each public function takes the whole record and returns a dict keyed by
output column. A field the record lacks, or one that does not parse, comes
back as None. A text field is stripped, and one that is blank or not text
is None. Nothing here raises on a malformed record.

Most categories in the record are lists, with one row per crystal, dataset,
or radiation source. A column that holds one value takes it from the first
row of its category.

The record states no radiation dose. Dose-series entries often give it in
the title instead, as in "after 5.2 MGy", "(56 kGy)", or "at 7.88x10^3 Gy",
so the first such number in the title is taken as the dose, in MGy.
Room-temperature and serial entries mostly state theirs in kGy. A number that
is only the tail of a longer one, such as the 5 of "1,5 MGy", is not taken,
and neither is a dose rate, such as "40 kGy/s", so a dose is missing rather
than wrong.

The record has no field for a time-resolved experiment either. Depositors
say so in the title and the keywords, so both are kept as columns, and
mentions_time_resolved records whether either names a time-resolved or
pump-probe experiment. That is a text match, not a classification. A
resting-state structure from a time-resolved series mentions it too.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any, cast

from ....json_values import object_dict, object_list
from ..scalars import to_float, to_int, to_text


def _first_row(entry_json: Mapping[str, object], category: str) -> dict[str, Any]:
    """Return the first row of a list-valued category, or {} without one."""
    rows = object_list(entry_json.get(category))
    return object_dict(rows[0]) if rows else {}


def _texts_in_every_row(
    entry_json: Mapping[str, object], category: str, key: str
) -> list[str]:
    """Return the stripped, non-blank text of ``key`` in a category's rows."""
    return [
        text
        for row in object_list(entry_json.get(category))
        if (text := to_text(object_dict(row).get(key))) is not None
    ]


def extract_beamline_and_wavelength(
    entry_json: Mapping[str, object],
) -> dict[str, object]:
    """Return the beamline and wavelength of the first radiation source.

    The beamline falls back to the source's make and model when no
    synchrotron beamline is named. The wavelength is the first one in the
    source's list of wavelengths, separated by commas or spaces. A source
    without that list gives its single wavelength instead. A wavelength of
    zero or below is missing.
    """
    first_source = _first_row(entry_json, "diffrn_source")
    beamline = to_text(first_source.get("pdbx_synchrotron_beamline")) or to_text(
        first_source.get("type")
    )
    wavelength_text = to_text(first_source.get("pdbx_wavelength_list"))
    if wavelength_text is not None:
        tokens = [token for token in re.split(r"[,\s]+", wavelength_text) if token]
        first_wavelength: object = tokens[0] if tokens else None
    else:
        # Many entries name their one wavelength in pdbx_wavelength and leave
        # the list out.
        first_wavelength = first_source.get("pdbx_wavelength")
    return {"beamline": beamline, "wavelength": _positive(to_float(first_wavelength))}


def _year_of(date_value: object) -> int | None:
    """Return the year that opens an ISO date, or None without one.

    The date must open with exactly four digits, so neither a signed number
    nor a longer one reads as a year.
    """
    if not isinstance(date_value, str):
        return None
    match = re.match(r"([0-9]{4})(?![0-9])", date_value.strip())
    return int(match.group(1)) if match else None


def _positive(value: float | None) -> float | None:
    return value if value is not None and value > 0 else None


def _positive_int(value: object) -> int | None:
    number = to_int(value)
    return number if number is not None and number > 0 else None


def _positive_float(value: object) -> float | None:
    return _positive(to_float(value))


def _yes_or_no(value: object) -> bool | None:
    """Return True for the archive's Y, False for N, and None for anything else."""
    text = to_text(value)
    if text is None:
        return None
    return {"Y": True, "N": False}.get(text.upper())


# Each row names an output column, the category it is read from, the key in
# that category's first row, and the conversion applied to the value.
_FIRST_ROW_FIELDS: tuple[tuple[str, str, str, Callable[[object], object]], ...] = (
    # Crystal-growth temperature is a different measurement and can look
    # plausible in this column, so it is not used as a fallback.
    ("temperature_K", "diffrn", "ambient_temp", to_float),
    ("crystal_pH", "exptl_crystal_grow", "pH", to_float),
    ("crystallization_method", "exptl_crystal_grow", "method", to_text),
    ("num_crystals", "exptl", "crystals_number", to_int),
    ("first_radiation_source", "diffrn_source", "source", to_text),
    ("synchrotron_site", "diffrn_source", "pdbx_synchrotron_site", to_text),
    ("collection_year", "diffrn_detector", "pdbx_collection_date", _year_of),
    ("wilson_b", "reflns", "B_iso_Wilson_estimate", to_float),
    ("multiplicity", "reflns", "pdbx_redundancy", to_float),
    ("reflns_completeness", "reflns", "percent_possible_obs", to_float),
    ("cc_half", "reflns", "pdbx_CC_half", to_float),
    ("i_over_sigma", "reflns", "pdbx_netI_over_sigmaI", to_float),
    ("r_work", "refine", "ls_R_factor_R_work", to_float),
    ("r_free", "refine", "ls_R_factor_R_free", to_float),
    (
        "is_serial_crystallography",
        "diffrn",
        "pdbx_serial_crystal_experiment",
        _yes_or_no,
    ),
    (
        "sample_delivery_method",
        "pdbx_serial_crystallography_sample_delivery",
        "method",
        to_text,
    ),
    (
        "xfel_pulse_duration_fs",
        "pdbx_serial_crystallography_measurement",
        "pulse_duration",
        _positive_float,
    ),
)


def _resolution(entry_json: Mapping[str, object]) -> float | None:
    # The refinement's high-resolution limit comes first. RCSB's combined
    # resolution covers an entry without one. A value of zero or below is not
    # a resolution, so it reads as missing.
    refined = _positive(to_float(_first_row(entry_json, "refine").get("ls_d_res_high")))
    if refined is not None:
        return refined
    combined: object = object_dict(entry_json.get("rcsb_entry_info")).get(
        "resolution_combined"
    )
    if isinstance(combined, list):
        values = cast(list[object], combined)
        combined = values[0] if values else None
    return _positive(to_float(combined))


# A number directly before the unit MGy, kGy, or Gy, which may be written
# in scientific notation, as in "7.88x10^3 Gy". The number may not continue
# one that a digit, point, comma, or caret precedes, so that "1,5 MGy",
# "1.2.5 MGy", or "10^6 Gy" gives no dose rather than 5, 2.5, or 6. A unit
# followed by a slash, "per", or "s-1" is a dose rate and is passed over.
# Old entries write titles in capitals, so the unit is matched in any case.
_TITLE_DOSE = re.compile(
    r"(?<![0-9.,^])([0-9]+(?:\.[0-9]+)?)(?:\s*[x×]\s*10\^?([0-9]+))?"
    r"\s*([MK]?)Gy\b(?!\s*(?:/|per\b|s\s*-\s*1))",
    re.IGNORECASE,
)
_UNITS_PER_MGY = {"M": 1.0, "K": 1e3, "": 1e6}


def _dose_from_title(title: str | None) -> float | None:
    """Return the first dose a title states, in MGy, or None."""
    match = _TITLE_DOSE.search(title) if title is not None else None
    value = to_float(match.group(1)) if match else None
    if match is None or value is None:
        return None
    if match.group(2) is not None:
        value *= 10 ** int(match.group(2))
    return value / _UNITS_PER_MGY[match.group(3).upper()]


# A time-resolved or pump-probe experiment, as titles and keywords name it,
# with the two words joined by a space, a hyphen, or nothing.
_TIME_RESOLVED = re.compile(r"time[\s-]*resolved|pump[\s-]*probe", re.IGNORECASE)


def _mentions_time_resolved(*texts: str | None) -> bool | None:
    """Return whether any text names a time-resolved experiment, None without text."""
    given = [text for text in texts if text is not None]
    if not given:
        return None
    return any(_TIME_RESOLVED.search(text) for text in given)


def extract_experiment_fields(
    entry_json: Mapping[str, object],
) -> dict[str, object]:
    """Return the entry's experimental conditions and data quality statistics.

    The experimental methods and radiation sources are lists, and the
    structure determination methodology is lower-cased.
    """
    info: dict[str, object] = {
        output: coerce(_first_row(entry_json, category).get(key))
        for output, category, key, coerce in _FIRST_ROW_FIELDS
    }
    detector = _first_row(entry_json, "diffrn_detector")
    info["detector_type"] = to_text(detector.get("detector")) or to_text(
        detector.get("type")
    )
    title = to_text(object_dict(entry_json.get("struct")).get("title"))
    keywords = to_text(object_dict(entry_json.get("struct_keywords")).get("text"))
    info["title"] = title
    info["keywords"] = keywords
    info["mentions_time_resolved"] = _mentions_time_resolved(title, keywords)
    info["radiation_dose_MGy"] = _dose_from_title(title)
    info["resolution_rcsb"] = _resolution(entry_json)
    info["space_group"] = to_text(
        object_dict(entry_json.get("symmetry")).get("space_group_name_H_M")
    )
    info["deposition_year"] = _year_of(
        object_dict(entry_json.get("rcsb_accession_info")).get("deposit_date")
    )
    # Every method and source row is kept, since a joint X-ray and neutron
    # entry reports two methods and lists both beams.
    info["experimental_methods"] = _texts_in_every_row(entry_json, "exptl", "method")
    info["radiation_sources"] = _texts_in_every_row(
        entry_json, "diffrn_source", "source"
    )
    methodology = to_text(
        object_dict(entry_json.get("rcsb_entry_info")).get(
            "structure_determination_methodology"
        )
    )
    info["structure_determination_methodology"] = (
        methodology.lower() if methodology is not None else None
    )
    return info


def extract_citation_fields(entry_json: Mapping[str, object]) -> dict[str, object]:
    """Return the entry's primary citation and its deposition authors.

    Author names are joined with "; ", and a name that is not text is left
    out. The deposition authors follow their ordinals, and an author without
    one comes last. A PubMed ID of zero or below is missing, since the
    archive writes -1 for a citation PubMed does not index.
    """
    primary_citation = object_dict(entry_json.get("rcsb_primary_citation"))
    citation_authors = "; ".join(
        name
        for author in object_list(primary_citation.get("rcsb_authors"))
        if (name := to_text(author)) is not None
    )

    named_authors = [
        (name, object_dict(value).get("pdbx_ordinal"))
        for value in object_list(entry_json.get("audit_author"))
        if (name := to_text(object_dict(value).get("name"))) is not None
    ]

    def author_order(author: tuple[str, object]) -> tuple[int, int]:
        # Ordinals are compared as numbers, since text would put author 10
        # before author 2.
        ordinal = to_int(author[1])
        return (1, 0) if ordinal is None else (0, ordinal)

    deposit_authors = "; ".join(
        name for name, _ in sorted(named_authors, key=author_order)
    )

    return {
        "citation_title": to_text(primary_citation.get("title")),
        "citation_journal_abbrev": to_text(primary_citation.get("journal_abbrev")),
        "citation_year": to_int(primary_citation.get("year")),
        "citation_doi": to_text(primary_citation.get("pdbx_database_id_DOI")),
        "citation_pubmed_id": _positive_int(
            primary_citation.get("pdbx_database_id_PubMed")
        ),
        "citation_authors": citation_authors or None,
        "deposit_authors": deposit_authors or None,
    }


__all__ = [
    "extract_beamline_and_wavelength",
    "extract_citation_fields",
    "extract_experiment_fields",
]
