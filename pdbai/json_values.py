"""Type narrowing for values read from JSON documents and caches.

A value of the wrong shape reads as None or as an empty container rather
than raising. A malformed document thus leaves its reader with missing data,
not an error. Each helper returns a new container, which the caller may
change without changing the document it was read from. The values inside
are shared, not copied. A mapping's keys are taken to be strings, as a JSON
object's always are.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast


def optional_object_dict(value: object) -> dict[str, object] | None:
    """Return the value as a new dict, or None when it is not a mapping."""
    if not isinstance(value, Mapping):
        return None
    return dict(cast(Mapping[str, object], value))


def object_dict(value: object) -> dict[str, object]:
    """Return the value as a new dict, or {} when it is not a mapping."""
    return optional_object_dict(value) or {}


def object_list(value: object) -> list[object]:
    """Return the value as a new list, or [] when it is not a list."""
    if not isinstance(value, list):
        return []
    return list(cast(list[object], value))


def str_list(value: object) -> list[str]:
    """Return the value's string items as a new list, or [] for a non-list."""
    return [item for item in object_list(value) if isinstance(item, str)]


__all__ = ["object_dict", "object_list", "optional_object_dict", "str_list"]
