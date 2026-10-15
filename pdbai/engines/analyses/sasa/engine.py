"""The registered SASA engine, with its options, columns, and report fields.

The engine runs in the annotation phase, once per structure, and only when
include_sasa is set. It hands the structure's atom table and the user's
options to compute_structure_sasa and passes the columns back unchanged. Each
warning becomes one message among the run's warnings, prefixed with its
context, model, and code. The run report gets the status, whether it is ok,
and the warning messages.

Every SASA column belongs to the research-ready and radiation-sensitivity
profiles. The ML profile takes only the columns that columns.py lists for it.
"""

from __future__ import annotations

from ...core import (
    PROFILE_ML_MINIMAL,
    PROFILE_RADIATION_SENSITIVITY,
    PROFILE_RESEARCH_READY,
    ColumnGranularity,
    ColumnSpec,
    EngineContext,
    EnginePhase,
    EngineResult,
    EngineSpec,
    OptionSpec,
    ReportField,
)
from .columns import (
    ALL_OUTPUT_COLUMNS,
    BOOLEAN_OUTPUT_COLUMNS,
    ML_PROFILE_COLUMNS,
    NUMERIC_OUTPUT_COLUMNS,
    RESIDUE_OUTPUT_COLUMNS,
    SASA_COLUMN_DESCRIPTIONS,
    STRUCTURE_OUTPUT_COLUMNS,
)
from .compute import compute_structure_sasa
from .constants import (
    SASA_ACCURACY_POINT_COUNTS,
    SASA_DEFAULT_N_SPHERE_POINTS,
    SASA_FAST_N_SPHERE_POINTS,
    SASA_HIGH_N_SPHERE_POINTS,
)


class SasaEngine:
    """The engine that adds solvent-accessible areas to each structure."""

    spec = EngineSpec(
        name="sasa",
        display_name="SASA",
        phase=EnginePhase.ANNOTATION,
        toggle="include_sasa",
        columns=tuple(
            ColumnSpec(
                name=name,
                group="SASA Fields",
                granularity=(
                    ColumnGranularity.RESIDUE
                    if name in RESIDUE_OUTPUT_COLUMNS
                    else ColumnGranularity.STRUCTURE
                    if name in STRUCTURE_OUTPUT_COLUMNS
                    else ColumnGranularity.ATOM
                ),
                dtype=(
                    "boolean"
                    if name in BOOLEAN_OUTPUT_COLUMNS
                    else "numeric"
                    if name in NUMERIC_OUTPUT_COLUMNS
                    else "text"
                ),
                description=SASA_COLUMN_DESCRIPTIONS[name],
                profiles=frozenset(
                    {
                        PROFILE_RESEARCH_READY,
                        PROFILE_RADIATION_SENSITIVITY,
                        *((PROFILE_ML_MINIMAL,) if name in ML_PROFILE_COLUMNS else ()),
                    }
                ),
            )
            for name in ALL_OUTPUT_COLUMNS
        ),
        options=(
            OptionSpec(
                "include_sasa",
                label="SASA",
                caption=(
                    "Protein-only and bound-complex solvent-accessible surface area "
                    "and relative accessibility."
                ),
                help=(
                    "Compute protein-only and bound-context SASA/RASA with explicit "
                    "eligibility and status fields. Off by default."
                ),
                enabled_text=(
                    "Per-model protein-only and bound-context solvent-accessible "
                    "surface areas were computed, with DSSP-compatible protein "
                    "radii, Tien2013_ALLOWED_DSSP normalization for eligible "
                    "residues, Alvarez radii for non-protein occluders, explicit "
                    "quality masks, and structured status/diagnostics."
                ),
            ),
            OptionSpec(
                "sasa_accuracy_mode",
                label="SASA sphere-grid mode",
                help=(
                    "SASA sphere grid: fast "
                    f"({SASA_FAST_N_SPHERE_POINTS:,} points per atom), default "
                    f"({SASA_DEFAULT_N_SPHERE_POINTS:,}), or high "
                    f"({SASA_HIGH_N_SPHERE_POINTS:,})."
                ),
                kind="choice",
                default="default",
                choices=tuple(SASA_ACCURACY_POINT_COUNTS),
                refines="include_sasa",
                cli_flag="--sasa-accuracy",
                # The dataset records the mode a point count overrides as
                # custom, and the summary says so rather than the named mode.
                summary_override=lambda options: (
                    "**sasa_accuracy_mode** (`custom`): the named mode "
                    f"`{options.choice('sasa_accuracy_mode')}` was overridden by "
                    "sasa_n_sphere_points."
                    if options.count("sasa_n_sphere_points") is not None
                    else None
                ),
            ),
            OptionSpec(
                "sasa_n_sphere_points",
                label="SASA sphere points",
                help="Override the named SASA mode with an explicit positive point count.",
                kind="count",
                default=None,
                refines="include_sasa",
                cli_flag="--sasa-points",
            ),
            OptionSpec(
                "sasa_include_experimental_additives",
                label="Count crystallization additives as bound context",
                help=(
                    "Include recognized crystallization additives as bound-context "
                    "occluders."
                ),
                refines="include_sasa",
                enabled_text="recognized crystallization additives occluded solvent.",
                disabled_text="recognized crystallization additives were ignored.",
            ),
            OptionSpec(
                "sasa_include_unknown_nonpolymers",
                label="Count unclassified non-polymers as bound context",
                # The help text is written for the command-line flag, which
                # turns the setting off. The web form's checkbox turns it on.
                caption=(
                    "Unclassified non-polymers occlude the bound SASA context. "
                    "On by default."
                ),
                help="Exclude unclassified non-polymers from the bound SASA context.",
                default=True,
                refines="include_sasa",
                cli_flag="--sasa-exclude-unknown-nonpolymers",
                cli_negated=True,
                enabled_text="unclassified non-polymers occluded solvent.",
                disabled_text="unclassified non-polymers were ignored.",
            ),
        ),
        report_fields=(
            ReportField(
                "sasa_status",
                "SASA computation status for the structure.",
                on_failure="failed",
                in_summary=True,
                label="SASA",
            ),
            ReportField(
                "sasa_message",
                "The messages of SASA warnings, joined, or the engine's failure "
                "message when it raised.",
                carries_message=True,
            ),
        ),
    )

    def run(self, context: EngineContext) -> EngineResult:
        """Return the SASA columns, warnings, and report of one structure."""
        computation = compute_structure_sasa(
            context.atom_dataframe,
            accuracy_mode=context.options.choice("sasa_accuracy_mode"),
            n_sphere_points=context.options.count("sasa_n_sphere_points"),
            include_experimental_additives=context.options.flag(
                "sasa_include_experimental_additives"
            ),
            include_unknown_nonpolymers=context.options.flag(
                "sasa_include_unknown_nonpolymers"
            ),
            structure_context=context.structure_context,
        )
        warnings = tuple(
            "SASA"
            + (f" ({warning.context})" if warning.context else "")
            + (f", model {warning.model}" if warning.model else "")
            + f" [{warning.code}]: {warning.message}"
            for warning in computation.warnings
        )
        return EngineResult(
            columns=computation.columns,
            warnings=warnings,
            report={
                "sasa_status": computation.status,
                "sasa_message": " | ".join(
                    warning.message for warning in computation.warnings
                )
                or None,
            },
        )


SASA_ENGINE = SasaEngine()

__all__ = ["SASA_ENGINE"]
