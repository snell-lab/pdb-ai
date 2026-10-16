"""The RABDAM engine, which adds radiation-damage scores to X-ray structures.

RABDAM gives every eligible protein heavy atom a packing density, the
number of atoms within 7 Å of it in the crystal, and a BDamage. BDamage is
the atom's B factor divided by the mean B factor of the atoms nearest it in
packing density (Gerstel, Deane and Garman, 2015). The structure gets a
Bnet, which fits a kernel density estimate to the BDamage of the Asp and
Glu carboxyl oxygens. Bnet is the area of that estimate above the median
BDamage of all scored atoms, divided by the area below it (Shelley et al.,
2018). It rises with the decarboxylation those side chains suffer. Bnet is
also ranked as a percentile among the 1,000 reference structures nearest
in resolution (Shelley and Garman, 2022).

The rabdam_source setting chooses which file is scored. The choices are
PDB-Redo's re-refinement (pdb_redo, the default), RCSB's deposited file
(rcsb), or both. Each source writes its own family of columns, suffixed
_pdb_redo or _rcsb, so every value names the file it came from. The curated
profiles select these families by wildcard. An upload has no RCSB/PDB-Redo
split. It is scored as given and writes the family without a suffix.

PDB-Redo is the default because the reference table was computed from
PDB-Redo re-refinements. Only a PDB-Redo score is ranked like with like.
PDB-Redo's file is therefore fetched whenever RABDAM scores it, even when
the PDB-Redo comparison is off. No RCSB-calibrated table exists, so an
RCSB score is ranked against the same one. Its column description says so.

A source is delivered when its BDamage reached at least one atom row and
its Bnet and percentile exist. rabdam_status is available when every
requested source is delivered, failed when no source could be scored, and
partial otherwise. A source is not delivered when its file is missing or
rabdam-custom cannot score it, or when its Bnet or percentile is missing.
Nor is it when none of its per-atom scores reach the atom rows, as when
PDB-Redo's identity bridge fails. Bnet needs no pairing, so it is kept.
rabdam_message keeps every reason.

bdamage_coverage_pdb_redo is the share of RCSB protein heavy atoms with a
PDB-Redo BDamage. It is missing when PDB-Redo's model could not be scored.
It is a value to filter on, not a fault, so a low share raises no warning.
When both sources are scored, a matches flag per score says whether the two
agree.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ....structure import alignment
from ...core import (
    CURATED_PROFILE_IDS,
    PROFILE_STRUCTURE_QUALITY,
    STRUCTURE_CIF_PATH,
    ColumnGranularity,
    EngineContext,
    EngineOptions,
    EnginePhase,
    EngineResult,
    EngineSpec,
    OptionSpec,
    ReportField,
    column_specs,
)
from ...sources.pdb_redo import PDB_REDO_CIF_PATH, PDB_REDO_DATAFRAME
from .calculation import (
    AGREEMENT_TOLERANCE,
    COMPARED_SCORES,
    add_agreement_flags,
    place_scores_on_atom_table,
    score_structure_file,
)
from .integration import RabdamCustomResult, check_installed
from .reference import reference_receipts

RABDAM_STATUS_AVAILABLE = "available"
RABDAM_STATUS_PARTIAL = "partial"
RABDAM_STATUS_FAILED = "failed"

# Each source's column suffix and the words its column descriptions and
# warnings use for the scored model.
_SCORED_SOURCES = (
    ("", "the uploaded model"),
    ("_pdb_redo", "the PDB-Redo model"),
    ("_rcsb", "the RCSB model"),
)
_MODEL_BY_SUFFIX = dict(_SCORED_SOURCES)

_DESCRIPTION_TEMPLATES = {
    "bdamage": (
        "Atom BDamage: isotropic B factor divided by the sliding-window mean "
        "B factor of atoms with similar packing density, scored on {model}."
    ),
    "packing_density": (
        "Number of neighboring protein heavy atoms within the RABDAM 7 Å "
        "packing radius, scored on {model}."
    ),
    "bnet": (
        "Structure-level RABDAM Bnet score: the area of the kernel density "
        "estimate of Asp and Glu carboxyl-oxygen BDamage above the median "
        "BDamage of all scored atoms, divided by the area below it, scored on "
        "{model}."
    ),
    "bnet_percentile": (
        "Percentile rank of the Bnet score of {model} in RABDAM's bundled "
        "{reference} reference distribution."
    ),
    "bnet_site_count": (
        "Number of Asp OD1/OD2 and Glu OE1/OE2 atoms of {model} whose BDamage "
        "values fed the Bnet estimate. Bnet is biased upward and noisy when this "
        "is small; filter on it before using Bnet as a target."
    ),
    "bdamage_scored_atom_count": (
        "Number of atoms of {model} that received a BDamage score."
    ),
    "bnet_percentile_resolution_in_range": (
        "True when the structure's resolution lies inside the resolution window "
        "of the reference structures the Bnet of {model} was ranked against; "
        "False when the percentile extrapolates from the nearest edge of the "
        "0.59 to 3.5 Å reference; missing when no percentile was computed."
    ),
}

_COMPARED_SCORE_LABELS = {
    "bdamage": "atom BDamage values",
    "packing_density": "packing-density values",
    "bnet": "Bnet scores",
    "bnet_percentile": "Bnet percentiles",
}

# Only PDB-Redo's scores are bridged onto the RCSB atoms, so the bridge's
# columns name both sides of the pairing and carry no source suffix. They
# share their names with the bridge's report fields.
_BRIDGE_RCSB_AMBIGUOUS_COLUMN = "rabdam_alignment_rcsb_ambiguous_atom_count"
_BRIDGE_PDB_REDO_AMBIGUOUS_COLUMN = "rabdam_alignment_pdb_redo_ambiguous_atom_count"

_RABDAM_DESCRIPTIONS: dict[str, str] = {
    **{
        f"{score}{suffix}": template.format(
            model=model,
            # The reference was built from PDB-Redo refinements. The RCSB
            # description says so, because there a different kind of model
            # is ranked against it.
            reference=(
                "PDB-Redo-calibrated, resolution-matched"
                if suffix == "_rcsb"
                else "resolution-matched"
            ),
        )
        for suffix, model in _SCORED_SOURCES
        for score, template in _DESCRIPTION_TEMPLATES.items()
    },
    **{
        f"{score}_matches": (
            f"True when the PDB-Redo and RCSB {_COMPARED_SCORE_LABELS[score]} "
            f"differ by no more than {AGREEMENT_TOLERANCE:g}; missing when either "
            "value is unavailable."
        )
        for score in COMPARED_SCORES
    },
    "bdamage_coverage_pdb_redo": (
        "Fraction of protein heavy atoms in the RCSB DataFrame that received a "
        "PDB-Redo BDamage score. Below 1 when RABDAM excluded atoms (zero "
        "occupancy, non-positive B) or the RCSB/PDB-Redo bridge could not pair "
        "them; 0 when no score could be bridged; missing when the PDB-Redo "
        "model could not be scored."
    ),
    _BRIDGE_RCSB_AMBIGUOUS_COLUMN: (
        "Number of RCSB atoms excluded from the PDB-Redo RABDAM bridge because "
        "their model-aware residue and atom identity was duplicated."
    ),
    _BRIDGE_PDB_REDO_AMBIGUOUS_COLUMN: (
        "Number of PDB-Redo atoms excluded from the RABDAM bridge because their "
        "model-aware residue and atom identity was duplicated."
    ),
}

_OUTPUT_COLUMNS = tuple(_RABDAM_DESCRIPTIONS)


_STRUCTURE_LEVEL_PREFIXES = (
    "bnet",
    "bdamage_coverage",
    "bdamage_scored_atom_count",
    "rabdam_alignment_",
)


def _is_boolean_column(name: str) -> bool:
    return name.endswith("_matches") or "resolution_in_range" in name


def _write_structure_level_columns(
    output: pd.DataFrame, result: RabdamCustomResult | None, *, suffix: str
) -> None:
    """Write one source's structure-level columns other than Bnet itself.

    Each is missing when the source could not be scored.
    """
    output[f"bnet_percentile{suffix}"] = (
        result.bnet_percentile if result is not None else np.nan
    )
    output[f"bnet_site_count{suffix}"] = (
        result.site_count
        if result is not None and result.site_count is not None
        else np.nan
    )
    output[f"bdamage_scored_atom_count{suffix}"] = (
        result.scored_atom_count if result is not None else np.nan
    )
    in_range = result.percentile_resolution_in_range if result is not None else None
    output[f"bnet_percentile_resolution_in_range{suffix}"] = pd.Series(
        pd.NA if in_range is None else in_range,
        index=output.index,
        dtype="boolean",
    )


def _write_bridge_quality_columns(
    output: pd.DataFrame,
    atom_dataframe: pd.DataFrame,
    result: RabdamCustomResult | None,
    bridge: alignment.AlignmentDiagnostics | None,
) -> None:
    """Write the PDB-Redo bridge's coverage and ambiguous-atom counts.

    Coverage is the share of the atoms of ``atom_dataframe`` selected by
    protein_heavy_atom_mask that hold a PDB-Redo BDamage. It is missing when
    ``result`` is None, since nothing was scored to cover them, and when
    ``atom_dataframe`` has no protein heavy atom. The counts are missing when no
    bridge ran.
    """
    eligible = alignment.protein_heavy_atom_mask(atom_dataframe)
    scored = output["bdamage_pdb_redo"].notna().to_numpy(dtype=bool)
    output["bdamage_coverage_pdb_redo"] = (
        float((scored & eligible).sum()) / float(eligible.sum())
        if result is not None and eligible.any()
        else float("nan")
    )
    output[_BRIDGE_RCSB_AMBIGUOUS_COLUMN] = (
        bridge.rcsb_ambiguous_atom_count if bridge is not None else np.nan
    )
    output[_BRIDGE_PDB_REDO_AMBIGUOUS_COLUMN] = (
        bridge.pdb_redo_ambiguous_atom_count if bridge is not None else np.nan
    )


def _scores(options: EngineOptions, sources: set[str]) -> bool:
    # Only rabdam_source is read, never include_rabdam. Whether the engine
    # runs is decided elsewhere.
    return options.choice("rabdam_source") in sources


def _scores_pdb_redo(options: EngineOptions) -> bool:
    return _scores(options, {"pdb_redo", "both"})


def _scores_rcsb(options: EngineOptions) -> bool:
    return _scores(options, {"rcsb", "both"})


def _scores_both(options: EngineOptions) -> bool:
    return _scores(options, {"both"})


def _source_test(name: str) -> Callable[[EngineOptions], bool] | None:
    """Return the test of whether a run on downloaded entries writes ``name``.

    A suffixed column, an agreement flag, or a bridge column exists only when
    its source is scored. None marks the unsuffixed family an upload writes.
    """
    if name.endswith("_matches"):
        return _scores_both
    if name.endswith("_rcsb"):
        return _scores_rcsb
    if name.endswith("_pdb_redo") or name in (
        _BRIDGE_RCSB_AMBIGUOUS_COLUMN,
        _BRIDGE_PDB_REDO_AMBIGUOUS_COLUMN,
    ):
        return _scores_pdb_redo
    return None


_DOWNLOAD_COLUMN_TESTS = {
    name: test for name in _OUTPUT_COLUMNS if (test := _source_test(name)) is not None
}


def _pdb_redo_dependency(options: EngineOptions) -> tuple[str, ...]:
    # Scoring PDB-Redo's model needs its file. Bridging the scores onto the
    # RCSB atoms needs its parsed atom table.
    return ("pdb_redo_structure",) if _scores_pdb_redo(options) else ()


# A scorer is called with the entry ID, the structure file, the atom table,
# and the warnings list. When it cannot score the file, it appends a warning
# and returns None.
type RabdamScorer = Callable[
    [str, Path, pd.DataFrame, list[str]], RabdamCustomResult | None
]

# A profile names RABDAM's columns by these globs. Each glob matches
# whichever suffixed family a run writes.
_COMMON_PROFILE_PATTERNS = (
    "packing_density*",
    "bdamage*",
    "bnet*",
    "bnet_percentile*",
    "rabdam_alignment*",
)
# The structure-level subset the structure_quality profile keeps.
_STRUCTURE_QUALITY_PROFILE_PATTERNS = (
    "bnet*",
    "bnet_percentile*",
    "bdamage_coverage*",
    "bdamage_scored_atom_count*",
    "rabdam_alignment*",
)

_REPORT_FIELDS = (
    ReportField(
        "rabdam_status",
        "RABDAM outcome: available when every requested source put BDamage on "
        "the atoms and produced Bnet and its percentile, failed when no source "
        "could be scored, and partial otherwise.",
        on_failure="failed",
        in_summary=True,
        label="RABDAM",
    ),
    ReportField(
        "rabdam_delivered_pdb_redo",
        "True when PDB-Redo's model was scored, its BDamage bridged onto the "
        "RCSB atoms, and its Bnet and percentile produced.",
        kind="flag",
        in_summary=True,
        label="RABDAM PDB-Redo",
    ),
    ReportField(
        "rabdam_delivered_rcsb",
        "True when RCSB's model was scored, its BDamage joined onto the atoms, "
        "and its Bnet and percentile produced.",
        kind="flag",
        in_summary=True,
        label="RABDAM RCSB",
    ),
    ReportField(
        "rabdam_alignment_match_rate",
        "Share of protein heavy RCSB atoms the PDB-Redo RABDAM bridge matched.",
        kind="fraction",
        in_summary=True,
        label="RABDAM match rate",
    ),
    ReportField(
        "rabdam_alignment_rcsb_ambiguous_atom_count",
        "RCSB atoms left out of the RABDAM bridge because their identity repeats.",
        kind="number",
    ),
    ReportField(
        "rabdam_alignment_pdb_redo_ambiguous_atom_count",
        "PDB-Redo atoms left out of the RABDAM bridge because their identity repeats.",
        kind="number",
    ),
    ReportField(
        "rabdam_message",
        "Every RABDAM warning for the structure, joined. These include a missing "
        "structure file, a file rabdam-custom could not score, its reasons when "
        "Bnet or its percentile could not be produced, and per-atom scores that "
        "reached no atom row.",
        carries_message=True,
    ),
)


class RabdamEngine:
    """RABDAM scoring of the sources rabdam_source selects."""

    spec = EngineSpec(
        name="rabdam",
        display_name="RABDAM",
        phase=EnginePhase.ANNOTATION,
        toggle="include_rabdam",
        conditional_dependencies=_pdb_redo_dependency,
        conditional_dependency_names=("pdb_redo_structure",),
        # There is no other BDamage implementation, so a missing package
        # stops the run before it starts instead of failing every structure.
        preflight=check_installed,
        reference_data=reference_receipts,
        options=(
            OptionSpec(
                "include_rabdam",
                label="RABDAM",
                caption=(
                    "Radiation damage: BDamage, Bnet, and packing density. When "
                    "RABDAM scores PDB-Redo's model (the default), PDB-Redo's mmCIF "
                    "is fetched for this run even when PDB-Redo comparison is off."
                ),
                help=(
                    "Compute RABDAM (bdamage/packing_density/bnet/bnet_percentile). "
                    "Off by default; available for X-ray diffraction only. With "
                    "--rabdam-source pdb_redo (the default) or both, RABDAM "
                    "fetches PDB-Redo's mmCIF internally for consistency with its "
                    "reference database, regardless of --include-pdb-redo. "
                    "Setting --rabdam-source rcsb is the only way to avoid that "
                    "PDB-Redo fetch."
                ),
                enabled_text="RABDAM scored the structures as set by rabdam_source.",
            ),
            OptionSpec(
                "rabdam_source",
                label="RABDAM scores which structure?",
                help=(
                    "Which mmCIF RABDAM scores. For downloaded entries the output "
                    "columns are source-suffixed (bdamage_pdb_redo/bdamage_rcsb, and "
                    "the same for packing_density/bnet/bnet_percentile), so it's "
                    "always clear which source a value came from; uploaded files "
                    "are scored as given and use the bare names. "
                    "'pdb_redo' (default): matches RABDAM's Bnet-percentile "
                    "reference database (calibrated against PDB-Redo-refined "
                    "B-factors), producing only the _pdb_redo columns. 'rcsb': "
                    "score RCSB's own mmCIF instead (no PDB-Redo fetch needed "
                    "unless --include-pdb-redo is also on), producing only the "
                    "_rcsb columns. 'both': score both, producing both "
                    "column sets plus a boolean *_matches flag per column "
                    "comparing the two sources. Only meaningful when RABDAM is "
                    "enabled."
                ),
                kind="choice",
                default="pdb_redo",
                choices=("pdb_redo", "rcsb", "both"),
                choice_labels=(
                    "PDB-Redo's mmCIF (default, matching the Bnet reference database)",
                    "RCSB's own mmCIF",
                    "Both, giving bdamage_pdb_redo/bdamage_rcsb (+bnet, packing_density, "
                    "bnet_percentile) plus a *_matches agreement flag per column",
                ),
                refines="include_rabdam",
                value_texts=(
                    (
                        "pdb_redo",
                        "RABDAM scored PDB-Redo's mmCIF (the default, matching its "
                        "Bnet-percentile reference database), producing "
                        "bdamage_pdb_redo/packing_density_pdb_redo/bnet_pdb_redo/"
                        "bnet_percentile_pdb_redo.",
                    ),
                    (
                        "rcsb",
                        "RABDAM scored RCSB's own mmCIF directly, producing "
                        "bdamage_rcsb/packing_density_rcsb/bnet_rcsb/"
                        "bnet_percentile_rcsb.",
                    ),
                    (
                        "both",
                        "RABDAM scored both RCSB's and PDB-Redo's mmCIF, producing "
                        "both the _pdb_redo- and _rcsb-suffixed column sets, plus a "
                        "*_matches boolean per column comparing the two.",
                    ),
                ),
            ),
        ),
        report_fields=_REPORT_FIELDS,
        profile_patterns=(
            *(
                (profile_id, _COMMON_PROFILE_PATTERNS)
                for profile_id in sorted(CURATED_PROFILE_IDS)
                if profile_id != PROFILE_STRUCTURE_QUALITY
            ),
            (PROFILE_STRUCTURE_QUALITY, _STRUCTURE_QUALITY_PROFILE_PATTERNS),
        ),
        columns=column_specs(
            _OUTPUT_COLUMNS,
            group="RABDAM Fields",
            sources_by_name={
                # An upload has no RCSB/PDB-Redo split, so it writes the
                # unsuffixed family.
                name: (
                    frozenset({"download"})
                    if name in _DOWNLOAD_COLUMN_TESTS
                    else frozenset({"upload"})
                )
                for name in _OUTPUT_COLUMNS
            },
            enabled_if_by_name=_DOWNLOAD_COLUMN_TESTS,
            dtype_by_name={
                name: "boolean" if _is_boolean_column(name) else "numeric"
                for name in _OUTPUT_COLUMNS
            },
            descriptions=_RABDAM_DESCRIPTIONS,
            granularity_by_name={
                name: ColumnGranularity.STRUCTURE
                for name in _OUTPUT_COLUMNS
                if name.startswith(_STRUCTURE_LEVEL_PREFIXES)
            },
            profiles=CURATED_PROFILE_IDS,
            profiles_by_name={
                name: (
                    CURATED_PROFILE_IDS
                    if name.startswith(_STRUCTURE_LEVEL_PREFIXES)
                    else CURATED_PROFILE_IDS - {PROFILE_STRUCTURE_QUALITY}
                )
                for name in _OUTPUT_COLUMNS
            },
        ),
        # BDamage and Bnet are defined on X-ray B factors, and the bundled Bnet
        # reference database was calibrated on X-ray refinements.
        techniques=frozenset({"x_ray_diffraction"}),
    )

    def __init__(self, scorer: RabdamScorer = score_structure_file) -> None:
        """Build the engine around a scorer, which tests replace."""
        self.scorer = scorer

    def run(self, context: EngineContext) -> EngineResult:
        """Score each requested source and put its columns on the atoms."""
        atom_dataframe = context.atom_dataframe
        warnings: list[str] = []
        # Score every requested source before merging any, so rabdam-custom's
        # own warnings come first.
        scored: list[tuple[str, RabdamCustomResult | None]] = []
        for suffix, path in _requested_sources(context):
            model = _MODEL_BY_SUFFIX[suffix]
            if path is None:
                warnings.append(
                    f"No structure file of {model} was available for "
                    f"{context.pdb_id}, so its RABDAM columns are empty."
                )
                scored.append((suffix, None))
                continue
            scorer_warnings: list[str] = []
            scored.append(
                (
                    suffix,
                    self.scorer(context.pdb_id, path, atom_dataframe, scorer_warnings),
                )
            )
            # rabdam-custom's messages do not say which file they are about,
            # and with both sources scored the two can read alike.
            warnings.extend(
                f"{model[0].upper()}{model[1:]}: {text}" for text in scorer_warnings
            )
        output = pd.DataFrame(index=atom_dataframe.index)
        delivered: dict[str, bool] = {}
        pdb_redo_alignment: alignment.AlignmentDiagnostics | None = None
        for suffix, result in scored:
            # PDB-Redo's scores belong to a different refinement, so they are
            # bridged onto the RCSB atoms by atom identity.
            bridged = suffix == "_pdb_redo"
            placed = place_scores_on_atom_table(
                atom_dataframe,
                result,
                context.artifacts.get(PDB_REDO_DATAFRAME) if bridged else None,
                pdb_id=context.pdb_id,
                suffix=suffix,
                requires_alignment=bridged,
            )
            output = output.join(placed.columns)
            model = _MODEL_BY_SUFFIX[suffix]
            if placed.unaligned_reason is not None:
                warnings.append(
                    f"rabdam-custom scored {model} for {context.pdb_id}, but "
                    f"{placed.unaligned_reason}. Its BDamage and packing-density "
                    "columns are therefore empty for this structure."
                )
            if placed.unplaced_atom_count:
                # Only a PDB-Redo model can hold atoms the deposited model lacks.
                reasons = (
                    "rabdam-custom keeps the highest-occupancy conformer of each "
                    "atom and the atom table one conformer per residue"
                    + (
                        ", and PDB-Redo can build atoms the deposited model lacks"
                        if bridged
                        else ""
                    )
                )
                warnings.append(
                    f"{placed.unplaced_atom_count} of the atoms rabdam-custom "
                    f"scored on {model} for {context.pdb_id} have no row in the "
                    "atom table, so their BDamage and packing density are not in "
                    f"the dataset. {reasons[0].upper()}{reasons[1:]}."
                )
            if (
                placed.alignment is not None
                and placed.alignment.chain_id_source != "author"
            ):
                warnings.append(
                    f"PDB-Redo file for {context.pdb_id} carries the RCSB label "
                    "chain IDs in place of the deposited author chain IDs; its "
                    "RABDAM scores were aligned on label chain IDs instead."
                )
            _write_structure_level_columns(output, result, suffix=suffix)
            if bridged:
                pdb_redo_alignment = placed.alignment
                _write_bridge_quality_columns(
                    output, atom_dataframe, result, pdb_redo_alignment
                )
            # The module docstring defines a delivered source.
            delivered[suffix] = (
                result is not None
                and placed.unaligned_reason is None
                and result.bnet is not None
                and result.bnet_percentile is not None
            )
        if len(scored) == 2:  # Both the PDB-Redo and the RCSB model.
            output = add_agreement_flags(output)

        if all(delivered.values()):
            status = RABDAM_STATUS_AVAILABLE
        elif any(result is not None for _, result in scored):
            status = RABDAM_STATUS_PARTIAL
        else:
            status = RABDAM_STATUS_FAILED
        return EngineResult(
            columns=output,
            warnings=tuple(warnings),
            report={
                "rabdam_status": status,
                "rabdam_delivered_pdb_redo": delivered.get("_pdb_redo"),
                "rabdam_delivered_rcsb": delivered.get("_rcsb"),
                "rabdam_alignment_match_rate": (
                    pdb_redo_alignment.match_rate if pdb_redo_alignment else None
                ),
                "rabdam_alignment_rcsb_ambiguous_atom_count": (
                    pdb_redo_alignment.rcsb_ambiguous_atom_count
                    if pdb_redo_alignment
                    else None
                ),
                "rabdam_alignment_pdb_redo_ambiguous_atom_count": (
                    pdb_redo_alignment.pdb_redo_ambiguous_atom_count
                    if pdb_redo_alignment
                    else None
                ),
                # Every warning travels here once, in order. The manifest and
                # run report then keep rabdam-custom's reasons, such as too
                # few Bnet sites or no resolution, and the bridge's.
                "rabdam_message": (
                    " | ".join(dict.fromkeys(text for text in warnings if text)) or None
                ),
            },
        )


def _requested_sources(context: EngineContext) -> list[tuple[str, Path | None]]:
    """Return the column suffix and structure file of each source to score."""
    structure_path = context.artifacts.get(STRUCTURE_CIF_PATH)
    if context.source != "download":
        # An upload has no archive entry, so it has no RCSB/PDB-Redo split.
        return [("", structure_path)]
    requested: list[tuple[str, Path | None]] = []
    if _scores_pdb_redo(context.options):
        requested.append(("_pdb_redo", context.artifacts.get(PDB_REDO_CIF_PATH)))
    if _scores_rcsb(context.options):
        requested.append(("_rcsb", structure_path))
    return requested


RABDAM_ENGINE = RabdamEngine()

__all__ = [
    "RABDAM_ENGINE",
    "RABDAM_STATUS_AVAILABLE",
    "RABDAM_STATUS_FAILED",
    "RABDAM_STATUS_PARTIAL",
    "RabdamEngine",
    "RabdamScorer",
]
