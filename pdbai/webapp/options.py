"""Reading a web request's fields into validated run settings.

Every value in a request is untrusted JSON or form text. Each value is
checked for its type before it is read, and nothing is converted with str()
or bool(). A stray number or Boolean is therefore refused, not read as the
text "5" or "True". The ValueError names the key as the client spelled it,
such as datasetShape rather than dataset_shape.

One rule covers blanks and case for every field. A missing key, None, and a
string of only whitespace all mean the value was not given, and the declared
default applies. A choice is matched with surrounding whitespace stripped and
case ignored. A flag is a JSON boolean or the text true or false, in any
case. A form field sent more than once is read as its first value, as
Werkzeug reads it.

PDB identifiers are split on the same delimiters the browser uses. The count
the page shows as ready is then the count the server accepts.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import get_args

from ..column_filtering import parse_column_filter
from ..engines.catalog import get_engine_registry
from ..engines.compatibility import require_compatible_options
from ..engines.core import (
    ALL_XRAY_SOURCES,
    DATASET_SHAPES,
    DEFAULT_DATASET_SHAPE,
    DEFAULT_TECHNIQUE,
    TECHNIQUE_IDS,
    XRAY_SOURCE_IDS,
    DatasetSource,
    EngineOptions,
    OptionSpec,
    OptionValue,
)
from ..identifiers import normalize_pdb_id
from ..json_values import object_list, str_list
from ..positive_numbers import parse_positive_int
from ..profiles import PROFILES, RunSettings, resolve_run_settings
from ..selection_filters import SEARCH_FILTERS, UNIPROT_ACCESSIONS

# Whitespace, commas, and semicolons separate PDB identifiers in pasted text.
# The whitespace is spelled out, not written \s, because Python's \s and
# JavaScript's \s differ. JavaScript's also matches U+00A0, U+2028, and
# U+3000. Spelled out this way, a non-breaking space stays inside a token, and
# the token is refused. PDB_ID_DELIMITERS in static/app.js must stay identical
# to this pattern.
PDB_ID_DELIMITERS = re.compile(r"[\t\n\v\f\r ,;]+")

_FLAG_SPELLINGS = {"true": True, "false": False}
_DATASET_SOURCES: tuple[DatasetSource, ...] = get_args(DatasetSource.__value__)
_PROFILE_IDS = tuple(profile.id for profile in PROFILES)


def _is_blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def parse_flag(value: object, key: str) -> bool:
    """Return the flag ``value`` gives for the request key ``key``.

    A bool is returned as it is. The text true or false is read in any case,
    with surrounding whitespace ignored. None or a blank string reads as
    False, because an unticked form checkbox sends nothing. Anything else
    raises ValueError naming ``key`` and the value. That includes 1, "1",
    "yes", and "on".
    """
    if isinstance(value, bool):
        return value
    if _is_blank(value):
        return False
    if isinstance(value, str):
        spelled = value.strip().lower()
        if spelled in _FLAG_SPELLINGS:
            return _FLAG_SPELLINGS[spelled]
    raise ValueError(f"{key} must be true or false; got {value!r}.")


def optional_text(value: object, key: str) -> str | None:
    """Return the text ``value`` gives for ``key``, stripped, or None if blank.

    None and a string of only whitespace give None. Any value that is not a
    string raises ValueError naming ``key`` and the value. A number or a
    Boolean is never read as text.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text; got {value!r}.")
    return value.strip() or None


def _parse_choice[Choice: str](
    value: object, key: str, choices: Iterable[Choice]
) -> Choice | None:
    # Return the choice in its declared spelling, or None when the value is
    # blank. optional_text refuses a value that is not a string.
    text = optional_text(value, key)
    if text is None:
        return None
    by_folded_case = {choice.lower(): choice for choice in choices}
    try:
        return by_folded_case[text.lower()]
    except KeyError:
        raise ValueError(
            f"{key} must be one of {', '.join(by_folded_case.values())}; got {value!r}."
        ) from None


def _optional_text_or_list(value: object, key: str) -> str | list[str] | None:
    # Return None or text as it is, or a list whose items are all text as a
    # new list. Anything else raises ValueError naming ``key``.
    if value is None or isinstance(value, str):
        return value
    # Copy the items before the isinstance check: to the type checker, a list
    # narrowed from object holds items of unknown type.
    items = object_list(value)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be text or a list of text; got {value!r}.")
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f"{key} must hold only text; got {item!r}.")
    return str_list(items)


def parse_limit(value: object) -> int | None:
    """Return the selection limit ``value`` gives, or None when it gives none.

    None and a blank string mean no limit. A positive whole number, or its
    text, is the limit. Anything else raises ValueError, including a Boolean
    and zero.
    """
    if _is_blank(value):
        return None
    return parse_positive_int(value, "limit")


def parse_search_bounds(request_fields: Mapping[str, object]) -> dict[str, float]:
    """Return the range filters a search request sets, keyed by filter identifier.

    Each filter pdbai.selection_filters declares is read under its camelCase
    key, such as minTemperatureK. A missing key, None, and a blank string
    mean the filter is not set. A positive number, or its plain decimal
    text, is the bound. Anything else raises ValueError naming the key,
    including a Boolean, zero, and text with a sign or an exponent.
    """
    bounds: dict[str, float] = {}
    for search_filter in SEARCH_FILTERS:
        value = request_fields.get(search_filter.json_key)
        if not _is_blank(value):
            bounds[search_filter.id] = search_filter.parse(
                value, search_filter.json_key
            )
    return bounds


def parse_search_accessions(request_fields: Mapping[str, object]) -> tuple[str, ...]:
    """Return the UniProt accessions a search request lists, or () for none.

    They are read under the camelCase key of UNIPROT_ACCESSIONS, as text or
    a list of text, the way its parse reads them. A missing key, None, and a
    blank string mean the filter is not set. Anything else that names no
    accession, or a token that is not one, raises ValueError naming the key.
    """
    value = request_fields.get(UNIPROT_ACCESSIONS.json_key)
    if _is_blank(value):
        return ()
    return UNIPROT_ACCESSIONS.parse(value, UNIPROT_ACCESSIONS.json_key)


def parse_technique(value: object) -> str:
    """Return the technique ``value`` names, or the default when it is blank.

    Case and surrounding whitespace are ignored. A value that is not text, or
    names no technique, raises ValueError.
    """
    return _parse_choice(value, "technique", TECHNIQUE_IDS) or DEFAULT_TECHNIQUE


def parse_xray_source(value: object) -> str:
    """Return the X-ray source filter ``value`` names, or all sources if blank.

    Case and surrounding whitespace are ignored. A value that is not text, or
    names no X-ray source, raises ValueError.
    """
    return _parse_choice(value, "xraySource", XRAY_SOURCE_IDS) or ALL_XRAY_SOURCES


def parse_pdb_ids(value: object) -> list[str]:
    """Return the PDB identifiers a request's pdbIds gives, canonicalized.

    Text is split on PDB_ID_DELIMITERS. A list holds one identifier per item,
    and an item is never split further. Blank parts and items are dropped, so
    None and blank text give an empty list. The identifiers keep their order,
    duplicates included. A value that is not None, text, or a list of text
    raises ValueError. So does any part that normalize_pdb_id refuses. A
    non-breaking space is not a delimiter, so "5MCC" and "4HHB" joined by one
    make one invalid identifier.
    """
    given = _optional_text_or_list(value, "pdbIds") or []
    parts = PDB_ID_DELIMITERS.split(given) if isinstance(given, str) else given
    return [normalize_pdb_id(part) for part in parts if part.strip()]


def _parse_engine_setting(option: OptionSpec, value: object) -> OptionValue:
    # Return None for a blank value, which resolve_run_settings reads as not
    # given. OptionSpec.parse is not used, because it gives the default for a
    # blank value.
    if _is_blank(value):
        return None
    key = option.json_key
    if option.kind == "flag":
        return parse_flag(value, key)
    if option.kind == "choice":
        return _parse_choice(value, key, option.choices)
    return parse_positive_int(value, key)


def settings_from_request(
    request_fields: Mapping[str, object], *, source: DatasetSource
) -> RunSettings:
    """Return validated run settings from a JSON build request or upload form.

    Each engine setting is read under its declared camelCase key. A setting
    not given takes its declared default, whatever the technique. Then the
    profile that profileId names turns on the engines its columns need. It
    also supplies its column filter, unless the request names one.
    resolve_run_settings describes this rule in full. columnFilterList is
    text of names separated by commas, semicolons, or line breaks, or a list
    of such text.

    The only exception raised is ValueError. It is raised for a value of the
    wrong type and for a count that is not a positive whole number. It is
    raised for a choice that names none of its values, and for a
    columnFilterMode or columnFilterList given without the other. It is also
    raised when resolve_run_settings refuses the settings, as for an engine
    the technique rules out.
    """
    technique = parse_technique(request_fields.get("technique"))
    xray_source = parse_xray_source(request_fields.get("xraySource"))
    dataset_shape = _parse_choice(
        request_fields.get("datasetShape"), "datasetShape", DATASET_SHAPES
    )
    profile_id = _parse_choice(
        request_fields.get("profileId"), "profileId", _PROFILE_IDS
    )
    given = {
        option.name: _parse_engine_setting(option, request_fields.get(option.json_key))
        for option in get_engine_registry().option_specs()
    }
    # parse_column_filter holds the rule and its messages for every front
    # end. The values are type-checked first, so that a number or a Boolean
    # is refused here rather than read as text.
    column_filter_mode, column_filter_list = parse_column_filter(
        optional_text(request_fields.get("columnFilterMode"), "columnFilterMode"),
        _optional_text_or_list(
            request_fields.get("columnFilterList"), "columnFilterList"
        ),
        mode_key="columnFilterMode",
        list_key="columnFilterList",
    )
    return resolve_run_settings(
        given,
        technique=technique,
        xray_source=xray_source,
        dataset_shape=dataset_shape or DEFAULT_DATASET_SHAPE,
        profile_id=profile_id,
        column_filter_mode=column_filter_mode,
        column_filter_list=column_filter_list or None,
        source=source,
    )


def engine_options_from_settings(settings: Mapping[str, object]) -> EngineOptions:
    """Return the engine options a run's stored settings record, validated.

    The ``source`` key of ``settings`` says where the run's structures came
    from. A manifest without it describes a download. Any value other than
    upload or download raises ValueError. So do settings that do not rebuild
    into options, and options the engines refuse for that source. A caller
    therefore has one error to catch for an unusable manifest.
    """
    recorded_source = settings.get("source", "download")
    matching: list[DatasetSource] = [
        known for known in _DATASET_SOURCES if recorded_source == known
    ]
    if not matching:
        raise ValueError(
            f"source must be one of {', '.join(_DATASET_SOURCES)}; "
            f"got {recorded_source!r}."
        )
    try:
        options = EngineOptions.from_settings_dict(settings)
    except TypeError as exception:
        # from_settings_dict raises TypeError for a value of the wrong type. To
        # the caller, that is one more way a manifest is unusable.
        raise ValueError(str(exception)) from exception
    require_compatible_options(options, source=matching[0])
    return options


__all__ = [
    "PDB_ID_DELIMITERS",
    "engine_options_from_settings",
    "optional_text",
    "parse_flag",
    "parse_limit",
    "parse_pdb_ids",
    "parse_search_accessions",
    "parse_search_bounds",
    "parse_technique",
    "parse_xray_source",
    "settings_from_request",
]
