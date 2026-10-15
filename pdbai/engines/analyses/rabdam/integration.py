"""The in-process adapter to rabdam-custom, the only BDamage implementation.

rabdam-custom (https://github.com/snell-lab/rabdam-custom) is an
accuracy-focused rewrite of RABDAM 2 (Shelley and Garman, 2022). It runs in
this process on every platform PDB-AI supports, so no subprocess, scratch
file, or timeout is involved. This module scores one structure file with it.
The result holds the per-atom BDamage and packing density, the structure's
Bnet, and the Bnet percentile. The package is imported only when a structure
is scored, so commands that never run RABDAM work without it.

The file is handed to rabdam-custom as a local mmCIF file, gzipped or not,
since that is the only kind of file PDB-AI parses. Its own input resolver is
bypassed. That resolver refuses a .gz name, expands environment variables in
the path, and matches a file name in any case, none of which a path PDB-AI
already holds should go through.

Every scientific parameter stays at rabdam-custom's default, and those
defaults equal RABDAM 2's. They include the 7 Å packing-density radius and
a sliding window of 2% of the eligible atoms, at least 11. They also
include the crystal translation range and the handling of hydrogens,
alternate locations, and occupancy. Scores therefore match a stock
rabdam-custom run, except that a caller may rank the percentile at another
resolution than the file's, as the engine ranks every source of an entry at
the atom table's. The percentile also keeps upstream's neighbourhood, the
1,000 reference structures nearest in resolution plus any ties at the
boundary. The bundled reference table was computed at these defaults, so
the percentile stays valid. A setting for any parameter would rank scores
against a table built differently, so none is offered.
"""

from __future__ import annotations

import functools
import importlib
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .reference import REFERENCE_PATH

logger = logging.getLogger(__name__)


# The bundled reference table records resolution in ångströms to two
# decimals, so the in-range test compares at that precision.
REFERENCE_RESOLUTION_DECIMALS = 2


def _import_callable(module_name: str, attribute: str) -> Callable[..., Any]:
    value: object = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"{module_name}.{attribute} is not callable")
    return cast(Callable[..., Any], value)


class RabdamCustomError(RuntimeError):
    """A structure rabdam-custom could not read or score."""


class RabdamCustomNotInstalledError(RabdamCustomError):
    """rabdam-custom cannot be imported, so no structure can be scored."""

    def __init__(self, detail: str = ""):
        """Build the message that says how to install rabdam-custom.

        ``detail`` is the text of the import error, appended when given.
        """
        message = (
            "rabdam-custom is not installed, but it is the required BDamage engine.\n"
            "Install it with:\n"
            '    pip install "rabdam @ git+https://github.com/snell-lab/'
            'rabdam-custom.git@main"\n'
            "(requires Python >= 3.10)."
        )
        if detail:
            message = f"{message}\n\nUnderlying import error: {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class RabdamCustomResult:
    """What rabdam-custom produced for one structure file.

    per_atom has one row per scored atom that carries a serial number. Its
    columns are _atom_site.id as text, bdamage, and packing_density. The ids
    are those of the scored file. They pair with another file's atoms only
    through an identity bridge. A structure-level value is None when
    rabdam-custom could not produce it, and warnings says why.
    resolution_angstrom is the resolution the percentile was matched on.
    """

    per_atom: pd.DataFrame
    bnet: float | None
    bnet_percentile: float | None
    resolution_angstrom: float | None
    site_count: int | None
    warnings: tuple[str, ...] = ()
    # The resolution span, in ångströms, of the reference structures the
    # percentile was ranked among. When the structure's own resolution lies
    # outside it, the percentile is extrapolated from the nearest edge of the
    # reference.
    percentile_resolution_min: float | None = None
    percentile_resolution_max: float | None = None

    @property
    def scored_atom_count(self) -> int:
        """Return the number of scored atoms that carry a serial number."""
        return len(self.per_atom)

    @property
    def percentile_resolution_in_range(self) -> bool | None:
        """Return whether the resolution lies inside the reference window.

        False means the percentile was extrapolated from the nearest edge of
        the reference. None means no percentile was computed. The comparison
        is made at the reference's precision of two decimals. At common
        resolutions one such value covers more than 1,000 entries, so the
        window can shrink to a single value such as 1.55 Å, and a structure
        at 1.553 Å, ranked against the right neighbours, must still count as
        inside it.
        """
        if (
            self.bnet_percentile is None
            or self.resolution_angstrom is None
            or self.percentile_resolution_min is None
            or self.percentile_resolution_max is None
        ):
            return None
        resolution = round(self.resolution_angstrom, REFERENCE_RESOLUTION_DECIMALS)
        window_min = round(
            self.percentile_resolution_min, REFERENCE_RESOLUTION_DECIMALS
        )
        window_max = round(
            self.percentile_resolution_max, REFERENCE_RESOLUTION_DECIMALS
        )
        return bool(window_min <= resolution <= window_max)


@dataclass(frozen=True)
class _RabdamCustom:
    """The parts of rabdam-custom this module calls."""

    resolved_structure_input: Callable[..., Any]
    structure_source_type: Callable[..., Any]
    structure_file_format: Callable[..., Any]
    read_structure: Callable[..., Any]
    calculate_bdamage_for_structure_data: Callable[..., Any]
    calculate_protein_bnet: Callable[..., Any]
    calculate_bnet_percentile: Callable[..., Any]


def check_installed() -> None:
    """Raise RabdamCustomNotInstalledError when rabdam-custom cannot be imported.

    The error names the underlying import failure.
    """
    _import_rabdam_custom()


def _import_rabdam_custom() -> _RabdamCustom:
    # A failure of any kind, not only ImportError, means the package cannot
    # be used, so every one is reported as a missing installation.
    try:
        return _RabdamCustom(
            resolved_structure_input=_import_callable(
                "input.resolver", "ResolvedStructureInput"
            ),
            structure_source_type=_import_callable(
                "input.resolver", "StructureSourceType"
            ),
            structure_file_format=_import_callable(
                "input.resolver", "StructureFileFormat"
            ),
            read_structure=_import_callable("input.reader", "read_structure"),
            calculate_bdamage_for_structure_data=_import_callable(
                "rabdam.workflow", "calculate_bdamage_for_structure_data"
            ),
            calculate_protein_bnet=_import_callable(
                "bnet.calculate", "calculate_protein_bnet"
            ),
            calculate_bnet_percentile=_import_callable(
                "bnet.percentile", "calculate_bnet_percentile"
            ),
        )
    except Exception as exception:
        raise RabdamCustomNotInstalledError(str(exception)) from exception


@functools.cache
def _load_reference_db() -> Any:
    # The result is cached for the process, None included, so a missing or
    # unreadable table is warned about once. It leaves every percentile empty
    # but fails no structure. rabdam-custom reads the manifest beside the
    # table, which gives the database its identifier.
    try:
        load_bnet_reference_database = _import_callable(
            "bnet.reference", "load_bnet_reference_database"
        )
        return load_bnet_reference_database(REFERENCE_PATH)
    except Exception as exception:
        logger.warning(
            "Could not load the rabdam-custom Bnet reference table, so "
            "bnet_percentile will be unavailable: %s",
            exception,
        )
        return None


def _reason_with_file_name_only(exception: Exception, path: Path) -> str:
    # The reason reaches the run report and the manifest, which never name a
    # file by its absolute path, so the path is cut down to the file's name.
    return str(exception).replace(str(path), path.name)


def run_rabdam_custom_for_structure(
    pdb_id: str,
    cif_path: Path,
    *,
    resolution_override: float | None = None,
) -> RabdamCustomResult:
    """Score one structure file with rabdam-custom.

    ``cif_path`` is a local mmCIF file, gzipped or not. ``resolution_override``
    is a resolution in ångströms. When given, the percentile is matched on it
    instead of the resolution in the file. ``pdb_id`` only labels messages.

    Raise RabdamCustomNotInstalledError when the package cannot be
    imported. Raise RabdamCustomError when the file is missing or cannot be
    read, or BDamage cannot be computed. A Bnet or percentile that cannot be
    computed is left None, with a warning in the result.
    """
    rabdam_custom = _import_rabdam_custom()
    warnings: list[str] = []

    local_path = cif_path.resolve()
    try:
        resolved = rabdam_custom.resolved_structure_input(
            original_input=str(cif_path),
            source_type=rabdam_custom.structure_source_type("local_file"),
            file_format=rabdam_custom.structure_file_format("mmcif"),
            local_path=local_path,
        )
        structure_data = rabdam_custom.read_structure(resolved)
    except Exception as exception:
        raise RabdamCustomError(
            f"rabdam-custom could not read structure file for {pdb_id}: "
            f"{_reason_with_file_name_only(exception, local_path)}"
        ) from exception

    # No BDamageWorkflowOptions or StructurePreparationOptions are passed, so
    # every parameter keeps the upstream default the reference table was
    # computed at. The module docstring explains why none is exposed.
    try:
        workflow = rabdam_custom.calculate_bdamage_for_structure_data(structure_data)
    except Exception as exception:
        raise RabdamCustomError(
            f"rabdam-custom BDamage workflow failed for {pdb_id}: "
            f"{_reason_with_file_name_only(exception, local_path)}"
        ) from exception

    score_result = workflow.bdamage_score_result
    # Without a serial an atom cannot be paired with an atom table row.
    serial_atoms = [
        atom for atom in score_result.atom_results if atom.atom_serial is not None
    ]
    per_atom = pd.DataFrame(
        {
            "_atom_site.id": [str(atom.atom_serial) for atom in serial_atoms],
            "bdamage": [float(atom.bdamage) for atom in serial_atoms],
            "packing_density": [float(atom.packing_density) for atom in serial_atoms],
        }
    )

    bnet: float | None = None
    site_count: int | None = None
    try:
        protein_bnet = rabdam_custom.calculate_protein_bnet(
            prepared_structure=workflow.prepared_structure,
            bdamage_score_result=score_result,
        )
        bnet = float(protein_bnet.bnet)
        site_count = int(protein_bnet.site_count)
    except Exception as exception:
        warnings.append(f"rabdam-custom Bnet unavailable for {pdb_id}: {exception}")

    resolution = (
        resolution_override
        if resolution_override is not None
        else _optional_float_attr(structure_data.metadata, "resolution_angstrom")
    )

    bnet_percentile: float | None = None
    percentile_resolution_min: float | None = None
    percentile_resolution_max: float | None = None
    if bnet is not None:
        if resolution is None or not resolution > 0:
            warnings.append(
                f"rabdam-custom Bnet-percentile unavailable for {pdb_id}: no "
                "positive resolution to match the reference database against."
            )
        elif (reference_db := _load_reference_db()) is None:
            warnings.append(
                f"Bnet-percentile reference database unavailable for {pdb_id}; "
                "percentile not computed."
            )
        else:
            try:
                percentile_result = rabdam_custom.calculate_bnet_percentile(
                    bnet=bnet,
                    resolution_angstrom=resolution,
                    reference_database=reference_db,
                )
                bnet_percentile = float(percentile_result.percentile_percent)
                percentile_resolution_min = _optional_float_attr(
                    percentile_result, "local_resolution_min"
                )
                percentile_resolution_max = _optional_float_attr(
                    percentile_result, "local_resolution_max"
                )
            except Exception as exception:
                warnings.append(
                    f"rabdam-custom Bnet-percentile unavailable for {pdb_id}: "
                    f"{exception}"
                )

    return RabdamCustomResult(
        per_atom=per_atom,
        bnet=bnet,
        bnet_percentile=bnet_percentile,
        resolution_angstrom=resolution,
        site_count=site_count,
        warnings=tuple(warnings),
        percentile_resolution_min=percentile_resolution_min,
        percentile_resolution_max=percentile_resolution_max,
    )


def _optional_float_attr(value: object, name: str) -> float | None:
    # A missing, unreadable, or non-finite value counts as absent, so that
    # NaN never reaches the percentile or the in-range test.
    attribute = getattr(value, name, None)
    if attribute is None:
        return None
    try:
        number = float(attribute)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = [
    "RabdamCustomError",
    "RabdamCustomNotInstalledError",
    "RabdamCustomResult",
    "check_installed",
    "run_rabdam_custom_for_structure",
]
