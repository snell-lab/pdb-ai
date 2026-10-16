"""Typed values that engines hand to one another for one structure.

An engine publishes what later engines read, such as a downloaded file or a
parsed report, under an ArtifactKey that carries the value's type. The store
refuses a value of the wrong type when it is published, so a reader gets the
type its key promises or nothing. One store belongs to each EngineContext and
so lives through both phases of one structure.

An engine declares the keys it publishes, and the coordinator refuses any
other. Nothing declares what an engine reads. An engine that reads another's
artifact either depends on that engine or runs in a later phase, so that it
runs after it. A producer may still fail or have nothing to publish, so a reader must treat a missing
value as an expected case. It can read context.failures to say why the value
is missing.

The host's own artifacts are the structure file, the entry's normalized
metadata, and the archive's entry record. The structure file is present once
acquisition has succeeded. The other two exist only for a download, and the
record only when it could be fetched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class ArtifactKey[T]:
    """The name and type of one value an engine or the host publishes.

    The store tells keys apart by name alone, so two keys of one name stand
    for one value. The description is documentation for
    engine authors and is not shown anywhere.

    An empty name raises ValueError. So does a type that is not a class,
    such as list[str], since the store checks each value with isinstance.
    """

    name: str
    type: type[T]
    description: str = ""

    def __post_init__(self) -> None:
        """Check the name and the type."""
        if not self.name:
            raise ValueError("Artifact key names must not be empty")
        # A parameterized generic is not a class, and isinstance raises on
        # it. Refusing it here fails the engine where it is declared, rather
        # than on its first publish.
        if not isinstance(cast(object, self.type), type):
            raise ValueError(
                f"Artifact {self.name!r} declares type {self.type!r}, which is "
                "not a class the store can check values against"
            )


class ArtifactStore:
    """The artifacts published so far for one structure, by key name."""

    def __init__(self) -> None:
        """Create a store that holds no artifacts."""
        self._values: dict[str, object] = {}

    def get[T](self, key: ArtifactKey[T]) -> T | None:
        """Return the value published under ``key``, or None when none is."""
        # put checked the value against the key it was published under, which
        # is this key, since no two keys share a name.
        return cast(T | None, self._values.get(key.name))

    def put[T](self, key: ArtifactKey[T], value: T) -> None:
        """Publish ``value`` under ``key``, replacing any value already there.

        A value that is not an instance of the key's type raises TypeError and
        leaves the store as it was.
        """
        if not isinstance(value, key.type):
            raise TypeError(
                f"Artifact {key.name!r} must be a {key.type.__name__}; "
                f"got {type(value).__name__}"
            )
        self._values[key.name] = value

    def update(self, values: Mapping[ArtifactKey[Any], object]) -> None:
        """Publish each value under its key, in order.

        A value of the wrong type raises TypeError, and the values before it
        stay published.
        """
        for key, value in values.items():
            self.put(key, cast(Any, value))

    def __contains__(self, key: ArtifactKey[Any]) -> bool:
        """Return whether a value is published under ``key``."""
        return key.name in self._values


# Published for a download by the RCSB acquisition engine, and for an upload
# by the host before any engine runs.
STRUCTURE_CIF_PATH = ArtifactKey(
    "structure_cif_path",
    Path,
    "The structure file the atom DataFrame was parsed from: the RCSB download, or "
    "the uploaded file.",
)
# A plain dict is what the store checks at run time. The cast gives type
# checkers the JSON object it holds, which isinstance cannot check.
_JSON_OBJECT = cast(type[dict[str, object]], dict)

STRUCTURE_METADATA = ArtifactKey(
    "structure_metadata",
    _JSON_OBJECT,
    "A downloaded entry's normalized metadata (experiment, citation, and dose "
    "fields), which the parser broadcasts onto the atom DataFrame; absent for "
    "uploads.",
)

ENTRY_RECORD = ArtifactKey(
    "entry_record",
    _JSON_OBJECT,
    "The archive's raw entry record (RCSB Data API) for a downloaded entry; "
    "absent for uploads and when the record could not be fetched.",
)

__all__ = [
    "ENTRY_RECORD",
    "STRUCTURE_CIF_PATH",
    "STRUCTURE_METADATA",
    "ArtifactKey",
    "ArtifactStore",
]
