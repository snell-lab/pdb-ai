"""The registered density engine, with its options, columns, and report fields.

The engine runs in the annotation phase, once per structure, and only when
include_density is set. density_map_source says which maps it reads.

For PDB-Redo's maps it depends on the PDB-Redo map source engine, which
fetches the reflection file, and on the PDB-Redo comparison engine, whose
coordinates of PDB-Redo's model it reads from the atom table. Those engines
run for it even when include_pdb_redo is off, and their own columns are then
left out of the output. For the wwPDB validation maps it depends on the
validation map source engine alone, since those maps were calculated about
the deposited model. A run writes the columns of each source it reads.

An entry a source has no maps for, and one whose maps cannot be made or
trusted, completes with empty values and a status that says why. Only a
reflection file that cannot be read fails the engine, and the entry still
completes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ...core import (
    EngineContext,
    EngineOptions,
    EnginePhase,
    EngineResult,
    EngineSpec,
    OptionSpec,
    ReportField,
    ReportValue,
)
from ...sources.map_sources import (
    DENSITY_MAP_SOURCE_OPTION,
    DENSITY_MAP_SOURCES,
    DENSITY_SOURCE_BOTH,
    DENSITY_SOURCE_PDB_REDO,
    DENSITY_SOURCE_WWPDB,
    DENSITY_TOGGLE,
    density_source_selected,
)
from ...sources.pdb_redo.artifacts import PDB_REDO_MTZ_PATH
from ...sources.validation.maps import (
    VALIDATION_2FOFC_MAP_PATH,
    VALIDATION_FOFC_MAP_PATH,
)
from .columns import DENSITY_COLUMNS
from .compute import Cell, compute_structure_density, compute_validation_density
from .constants import (
    MAP_MAX_GRID_POINTS,
    MAP_MIN_SAMPLE_RATE,
    MAP_SAMPLE_RATE,
    MAP_STATUS_FAILED,
    MAP_STATUS_NO_ENTRY,
    MAP_STATUS_NO_VALIDATION_MAPS,
)

# Both sources hold maps of X-ray entries only. PDB-Redo re-refines no other
# technique, and the wwPDB computes validation maps for no other.
_DENSITY_TECHNIQUES = frozenset({"x_ray_diffraction"})

_CELL_ITEMS = (
    "_cell.length_a",
    "_cell.length_b",
    "_cell.length_c",
    "_cell.angle_alpha",
    "_cell.angle_beta",
    "_cell.angle_gamma",
)

_WARNING_PREFIX = "Density: "

_PDB_REDO_ENGINES = ("pdb_redo_map_source", "pdb_redo_comparison")
_WWPDB_ENGINES = ("validation_map_source",)


def _map_source_dependencies(options: EngineOptions) -> tuple[str, ...]:
    # Each source's engines run only when its maps are read.
    needed: tuple[str, ...] = ()
    if density_source_selected(options, DENSITY_SOURCE_PDB_REDO):
        needed += _PDB_REDO_ENGINES
    if density_source_selected(options, DENSITY_SOURCE_WWPDB):
        needed += _WWPDB_ENGINES
    return needed


def deposited_cell(atoms: pd.DataFrame) -> Cell | None:
    """Return the deposited unit cell the atom table records, or None.

    None stands for an atom table without the six cell items, or with a value
    among them that is not a positive number.
    """
    if atoms.empty or any(item not in atoms.columns for item in _CELL_ITEMS):
        return None
    values = (
        pd.to_numeric(
            pd.Series([atoms[item].iloc[0] for item in _CELL_ITEMS], dtype=object),
            errors="coerce",
        )
        .to_numpy(dtype=np.float64, na_value=np.nan)
        .tolist()
    )
    if not all(math.isfinite(value) and value > 0 for value in values):
        return None
    a, b, c, alpha, beta, gamma = values
    return (a, b, c, alpha, beta, gamma)


_SAMPLING_TEXT = (
    f"Each map was calculated over the unit cell at {MAP_SAMPLE_RATE:g} grid "
    f"points per high-resolution limit, lowered to no less than "
    f"{MAP_MIN_SAMPLE_RATE:g} for a cell that would exceed "
    f"{MAP_MAX_GRID_POINTS:,} grid points, and read at atoms by cubic "
    "interpolation, in units of the map's root-mean-square density, which the "
    "structure columns record."
)

_CALCULATED_TEXT = (
    "The map of PDB-Redo's calculated structure factors (FC_ALL, PHIC_ALL) was "
    "read at the same positions on the scale of the map coefficients, and "
    "the difference density is given on that scale as the 2mFo−DFc density "
    "less the calculated density. The gradient of the mFo−DFc map was read "
    "at PDB-Redo's position, in sigma per Å, and divided by the curvature of "
    "the calculated map there for the shift of the atom that would level it."
)

_WWPDB_GRADIENT_TEXT = (
    "The gradient of the validation mFo−DFc map was read at the deposited "
    "position, in sigma per Å."
)


class DensityEngine:
    """The engine that reads an entry's maps at the atoms of its structure."""

    spec = EngineSpec(
        name="density",
        display_name="Density",
        phase=EnginePhase.ANNOTATION,
        toggle=DENSITY_TOGGLE,
        conditional_dependencies=_map_source_dependencies,
        conditional_dependency_names=(*_PDB_REDO_ENGINES, *_WWPDB_ENGINES),
        techniques=_DENSITY_TECHNIQUES,
        sources=frozenset({"download"}),
        columns=DENSITY_COLUMNS,
        options=(
            OptionSpec(
                DENSITY_TOGGLE,
                label="Electron density at atoms",
                caption=(
                    "Density of the 2mFo−DFc and mFo−DFc maps at every atom, "
                    "from PDB-Redo's maps, the wwPDB validation maps, or both."
                ),
                help=(
                    "Fetch an entry's map coefficients and report the density "
                    "of its 2mFo−DFc and mFo−DFc maps at every atom, in units "
                    "of each map's root-mean-square density. PDB-Redo's maps "
                    "also give the model's calculated density and the "
                    "difference density on one scale. Each source also gives "
                    "the gradient of its mFo−DFc map at the atoms of the "
                    "model it was calculated about, and PDB-Redo's the shift "
                    "of each atom that the gradient implies. "
                    "--density-source chooses the maps. Off by default."
                ),
                # The sampling reaches the run summary and the manifest.
                enabled_text=_SAMPLING_TEXT,
            ),
            OptionSpec(
                DENSITY_MAP_SOURCE_OPTION,
                label="Density map source",
                help=(
                    "Which maps the density columns are read from: pdb_redo, "
                    "PDB-Redo's maps about its re-refined model, read at the "
                    "deposited position and at PDB-Redo's; wwpdb, the wwPDB "
                    "validation maps about the deposited model, read at the "
                    "deposited position; or both."
                ),
                kind="choice",
                default=DENSITY_SOURCE_PDB_REDO,
                choices=DENSITY_MAP_SOURCES,
                choice_labels=(
                    "PDB-Redo maps (re-refined model)",
                    "wwPDB validation maps (deposited model)",
                    "Both",
                ),
                refines=DENSITY_TOGGLE,
                cli_flag="--density-source",
                value_texts=(
                    (
                        DENSITY_SOURCE_PDB_REDO,
                        "PDB-Redo's 2mFo−DFc (FWT, PHWT) and mFo−DFc (DELFWT, "
                        "PHDELWT) maps were read at the deposited position and "
                        "at the position of the paired atom of PDB-Redo's "
                        f"model. {_CALCULATED_TEXT}",
                    ),
                    (
                        DENSITY_SOURCE_WWPDB,
                        "The wwPDB validation 2mFo−DFc (pdbx_FWT, pdbx_PHWT) "
                        "and mFo−DFc (pdbx_DELFWT, pdbx_DELPHWT) maps, "
                        "calculated about the deposited model, were read at "
                        f"the deposited position. {_WWPDB_GRADIENT_TEXT}",
                    ),
                    (
                        DENSITY_SOURCE_BOTH,
                        "PDB-Redo's maps were read at the deposited position "
                        "and at PDB-Redo's, and the wwPDB validation maps, "
                        "calculated about the deposited model, at the "
                        f"deposited position. {_CALCULATED_TEXT} "
                        f"{_WWPDB_GRADIENT_TEXT}",
                    ),
                ),
            ),
        ),
        report_fields=(
            ReportField(
                "density_map_status",
                "Outcome of the entry's PDB-Redo maps: ok, or why their density "
                f"columns are empty. {MAP_STATUS_FAILED} when the engine raised. "
                "Empty when the run does not read PDB-Redo's maps.",
                on_failure=MAP_STATUS_FAILED,
                in_summary=True,
                label="Density",
            ),
            ReportField(
                "density_map_sample_rate",
                "Grid points per high-resolution limit PDB-Redo's maps were "
                "sampled at, or empty when no map was made.",
                kind="number",
            ),
            ReportField(
                "density_map_status_wwpdb",
                "Outcome of the entry's wwPDB validation maps: ok, or why their "
                f"density columns are empty. {MAP_STATUS_FAILED} when the "
                "engine raised. Empty when the run does not read those maps.",
                on_failure=MAP_STATUS_FAILED,
                in_summary=True,
                label="Density (wwPDB)",
            ),
            ReportField(
                "density_map_sample_rate_wwpdb",
                "Grid points per high-resolution limit the wwPDB validation "
                "maps were sampled at, or empty when no map was made.",
                kind="number",
            ),
            ReportField(
                "density_message",
                "Each density warning for the structure, joined, or the "
                "engine's failure message when it raised.",
                carries_message=True,
            ),
        ),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the density columns, warnings, and report of one structure.

        Passes on what reading a reflection file raises.
        """
        atoms = context.atom_dataframe
        cell = deposited_cell(atoms)
        frames: list[pd.DataFrame] = []
        warnings: list[str] = []
        messages: list[str] = []
        report: dict[str, ReportValue] = {}
        if density_source_selected(context.options, DENSITY_SOURCE_PDB_REDO):
            computation = compute_structure_density(
                atoms, context.artifacts.get(PDB_REDO_MTZ_PATH), deposited_cell=cell
            )
            frames.append(computation.columns)
            messages.extend(computation.warnings)
            # The source engine already warned when its download failed.
            if not (
                computation.map_status == MAP_STATUS_NO_ENTRY
                and "pdb_redo_map_source" in context.failures
            ):
                warnings.extend(computation.warnings)
            report["density_map_status"] = computation.map_status
            report["density_map_sample_rate"] = computation.sample_rate
        if density_source_selected(context.options, DENSITY_SOURCE_WWPDB):
            computation = compute_validation_density(
                atoms,
                context.artifacts.get(VALIDATION_2FOFC_MAP_PATH),
                context.artifacts.get(VALIDATION_FOFC_MAP_PATH),
                deposited_cell=cell,
            )
            frames.append(computation.columns)
            messages.extend(computation.warnings)
            if not (
                computation.map_status == MAP_STATUS_NO_VALIDATION_MAPS
                and "validation_map_source" in context.failures
            ):
                warnings.extend(computation.warnings)
            report["density_map_status_wwpdb"] = computation.map_status
            report["density_map_sample_rate_wwpdb"] = computation.sample_rate
        report["density_message"] = " | ".join(messages) or None
        return EngineResult(
            columns=pd.concat(frames, axis=1) if frames else None,
            warnings=tuple(
                f"{_WARNING_PREFIX}{context.pdb_id}: {warning}" for warning in warnings
            ),
            report=report,
        )


DENSITY_ENGINE = DensityEngine()

__all__ = ["DENSITY_ENGINE", "DensityEngine", "deposited_cell"]
