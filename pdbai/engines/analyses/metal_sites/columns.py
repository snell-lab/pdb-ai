"""The metal_sites, metal_ligand_bonds, and metal_metal_distances tables.

The atom link column is declared here too.

metal_sites holds one row per metal atom, with fixed-size summaries of its
bonds. metal_ligand_bonds holds one row per bond, a metal atom and one donor
atom in one image. A bond row carries the metal's site key columns without a
prefix, so it joins its site row on site_id, and the donor's under donor_.
Both sets link to the atom level through the standard site link, so they hold
what the atom table holds for the same atom.

metal_metal_distances holds one row per pair of metal atoms that lie near
each other, the first under the unprefixed site key columns and the second
under partner_. It gives the distances within a cluster, which no bond row
holds, since the atoms of a metal's own component are never donors.

No table has a cohort, technique, or temperature column. Those come from
the structures table at analysis time. None holds a verdict either. A
deviation or a z value states how far a distance lies from a cited reference
and leaves what that means to the reader.
"""

from __future__ import annotations

from ...core import (
    PROFILE_RADIATION_SENSITIVITY,
    PROFILE_RESEARCH_READY,
    ColumnGranularity,
    ColumnSpec,
    TableColumnSpec,
    TableLink,
    TableSpec,
    site_atom_link,
    site_key_columns,
)
from ..site_geometry import SPECIAL_POSITION_CUTOFF_A
from ..struct_conn import DETECTION_METHODS
from .constants import (
    BOND_STATUSES,
    CANDIDATE_SEARCH_RADIUS_A,
    DONOR_CLASSES,
    DONOR_RULES,
    FIRST_SPHERE_TOLERANCE_A,
    NEARBY_METAL_RADIUS_A,
    PARENT_TYPES,
    REFERENCE_KINDS,
    SITE_STATUSES,
)

DONOR_PREFIX = "donor_"
PARTNER_PREFIX = "partner_"
METAL_SITES_TABLE_NAME = "metal_sites"
METAL_BONDS_TABLE_NAME = "metal_ligand_bonds"
METAL_PAIRS_TABLE_NAME = "metal_metal_distances"
METAL_LINK_COLUMN = "metal_site_id"

_TOLERANCE = f"{FIRST_SPHERE_TOLERANCE_A:g} Å"
_SEARCH = f"{CANDIDATE_SEARCH_RADIUS_A:g} Å"

_SITE_MEASURES: tuple[TableColumnSpec, ...] = (
    TableColumnSpec(
        "metal_element", description="Element symbol of the metal atom, in upper case."
    ),
    TableColumnSpec(
        "parent_type",
        categories=PARENT_TYPES,
        description=(
            "The kind of component the metal atom belongs to. cluster and heme "
            "are components the bundled catalog lists as such, with cluster "
            "taking precedence. ion is a component whose only non-hydrogen "
            "atom is the metal. other is any other component."
        ),
    ),
    TableColumnSpec(
        "component_atom_count",
        dtype="Int64",
        description=(
            "Non-hydrogen atoms the atom table holds for the metal's component, "
            "the metal included."
        ),
    ),
    TableColumnSpec(
        "coordination_number",
        dtype="Int64",
        description=(
            "Rows of metal_ligand_bonds for the site. Atoms of the metal's own "
            "component are never donors, so for a heme or a cluster this counts "
            "the bonds to the protein and to water only, not the whole "
            "coordination sphere."
        ),
    ),
    TableColumnSpec(
        "n_symmetry_bonds",
        dtype="Int64",
        description="Bonds of the site whose donor atom belongs to a symmetry mate.",
    ),
    TableColumnSpec(
        "n_declared_bonds",
        dtype="Int64",
        description="Bonds of the site that a deposited _struct_conn metalc record names.",
    ),
    TableColumnSpec(
        "n_donor_N", dtype="Int64", description="Bonds of the site to a nitrogen atom."
    ),
    TableColumnSpec(
        "n_donor_O",
        dtype="Int64",
        description="Bonds of the site to an oxygen atom, water included.",
    ),
    TableColumnSpec(
        "n_donor_S", dtype="Int64", description="Bonds of the site to a sulfur atom."
    ),
    TableColumnSpec(
        "n_donor_water",
        dtype="Int64",
        description="Bonds of the site to a water oxygen.",
    ),
    TableColumnSpec(
        "donor_distance_min_A",
        dtype="float64",
        description="Shortest metal–donor distance of the site in Å. Missing without bonds.",
    ),
    TableColumnSpec(
        "donor_distance_mean_A",
        dtype="float64",
        description="Mean metal–donor distance of the site in Å. Missing without bonds.",
    ),
    TableColumnSpec(
        "donor_distance_max_A",
        dtype="float64",
        description="Longest metal–donor distance of the site in Å. Missing without bonds.",
    ),
    TableColumnSpec(
        "n_reference_bonds",
        dtype="Int64",
        description=(
            "Bonds of the site with a reference distance of their own, which "
            "the two summaries below are taken over."
        ),
    ),
    TableColumnSpec(
        "distance_deviation_mean_A",
        dtype="float64",
        description=(
            "Mean of distance_deviation_A over the site's bonds that have one, "
            "in Å. Positive when the site's bonds are longer than their "
            "reference distances. Missing when no bond has a reference."
        ),
    ),
    TableColumnSpec(
        "distance_z_rms",
        dtype="float64",
        description=(
            "Root mean square of distance_z over the site's bonds that have "
            "one. Missing when no bond has one."
        ),
    ),
    TableColumnSpec(
        "occupancy", dtype="float64", description="Occupancy of the metal atom."
    ),
    TableColumnSpec(
        "b_factor_A2",
        dtype="float64",
        description=(
            "Isotropic B-factor of the metal atom in Å². A neutron B-factor is "
            "refined against nuclear density and is not directly comparable "
            "with an X-ray one."
        ),
    ),
    TableColumnSpec(
        "donor_b_factor_mean_A2",
        dtype="float64",
        description="Mean isotropic B-factor of the site's donor atoms in Å². Missing without bonds.",
    ),
    TableColumnSpec(
        "b_factor_ratio_to_donors",
        dtype="float64",
        description=(
            "b_factor_A2 divided by donor_b_factor_mean_A2. Missing when either "
            "is missing or not positive."
        ),
    ),
    TableColumnSpec(
        "nearest_metal_distance_A",
        dtype="float64",
        description=(
            "Distance in Å to the nearest other metal atom of the same model, "
            "among the atoms of the file only. Missing when the model has no "
            "other metal atom. Where a cluster is modelled as two overlapping "
            "components, the nearest metal is the atom's own copy, so read "
            "the distances within a cluster from metal_metal_distances."
        ),
    ),
    TableColumnSpec(
        "nearest_metal_element",
        description="Element of that nearest metal atom. Missing when there is none.",
    ),
    TableColumnSpec(
        "nearest_metal_site_id",
        description="site_id of that nearest metal atom. Missing when there is none.",
    ),
    TableColumnSpec(
        "n_metals_within_6A",
        dtype="Int64",
        description=(
            f"Other metal atoms of the same model within {NEARBY_METAL_RADIUS_A:g} Å, "
            "among the atoms of the file only. It gives context and asserts no "
            "metal–metal bond."
        ),
    ),
    TableColumnSpec(
        "on_special_position",
        dtype="boolean",
        description=(
            "True when a symmetry image of the metal atom lies within "
            f"{SPECIAL_POSITION_CUTOFF_A:g} Å of it, so the atom sits on a "
            "symmetry element and its occupancy is expected to be a fraction. "
            "Missing when the file gives no usable unit cell and space group."
        ),
    ),
    TableColumnSpec(
        "dpi_A",
        dtype="float64",
        description=(
            "Diffraction-component precision index of the entry in Å, by "
            "equation 7 of Blow (2002), 1.28 Ni^(1/2) Va^(1/3) nobs^(-5/6) "
            "Rfree. Ni is the occupancy-weighted count of non-hydrogen atoms "
            "of the first model, with a kept alternate conformer counted "
            "whole, Va the unit-cell volume divided by the number of "
            "space-group operations, and nobs and Rfree the first "
            "_refine.ls_number_reflns_obs and _refine.ls_R_factor_R_free the "
            "file gives together. It repeats on every site of the entry, and "
            "is missing when any term is."
        ),
    ),
    TableColumnSpec(
        "metal_site_status",
        categories=SITE_STATUSES,
        description=(
            "ok, or no_donors_found when the site has no bond row, so its bond "
            "summaries are missing. A site without bonds is reported as "
            "found. It is not evidence that the metal is uncoordinated, since "
            "donors outside the supported classes are not searched for."
        ),
    ),
)

METAL_SITES_TABLE = TableSpec(
    name=METAL_SITES_TABLE_NAME,
    description=(
        "One row per metal atom of each model, an atom whose element is a "
        "metal, with coordinates and an occupancy that is not zero, in the "
        "conformer the atom table keeps. The row summarizes the site's rows "
        "of metal_ligand_bonds. It makes no judgment of the site."
    ),
    key=("site_id",),
    columns=(*site_key_columns(), *_SITE_MEASURES),
    links=(site_atom_link(),),
)

_BOND_MEASURES: tuple[TableColumnSpec, ...] = (
    TableColumnSpec(
        "donor_symmetry_operator",
        description=(
            "The symmetry operator and cell translation that bring the donor "
            "atom to the position measured, in the mmCIF form 1_555, where "
            "1_555 leaves the atom where the file puts it."
        ),
    ),
    TableColumnSpec(
        "is_symmetry_mate",
        dtype="boolean",
        description="True when donor_symmetry_operator is not 1_555.",
    ),
    TableColumnSpec(
        "metal_element", description="Element symbol of the metal atom, in upper case."
    ),
    TableColumnSpec(
        "donor_element", description="Element symbol of the donor atom, in upper case."
    ),
    TableColumnSpec(
        "donor_class",
        categories=DONOR_CLASSES,
        description=(
            "The kind of residue the donor atom belongs to: water, one of the "
            "20 standard amino acids, a nucleotide, or other. The distance "
            "search infers bonds to water and amino acids only, so any other "
            "class is on a row a deposited record named."
        ),
    ),
    TableColumnSpec(
        "donor_rule",
        categories=DONOR_RULES,
        description=(
            "Which donor rule the atom falls under. The distance search infers "
            "a bond for every rule but outside_typical_donor_list, which is on "
            "a row only because a deposited record named the atom."
        ),
    ),
    TableColumnSpec(
        "detection_method",
        categories=DETECTION_METHODS,
        description=(
            "How the bond was found. struct_conn when a deposited _struct_conn "
            "metalc record names it, distance when the donor is one the "
            "distance search may infer and lies within first_sphere_cutoff_A, "
            "and struct_conn_and_distance when both hold."
        ),
    ),
    TableColumnSpec(
        "distance_A",
        dtype="float64",
        description="Distance in Å between the metal atom and the donor atom in its image. Not rounded.",
    ),
    TableColumnSpec(
        "reference_kind",
        categories=REFERENCE_KINDS,
        description=(
            "Where the target distance came from. exact is the reference "
            "table's own row for the metal and donor. element_fallback is the "
            "longest row for the same metal and donor element, which decides "
            "first-sphere membership only. missing means the table has no row "
            "for the metal and donor element."
        ),
    ),
    TableColumnSpec(
        "reference_key",
        description=(
            "The reference row used, as residue:atom:metal, where the residue "
            "CA stands for the backbone carbonyl and * for the element "
            "fallback. Empty when reference_kind is missing."
        ),
    ),
    TableColumnSpec(
        "reference_distance_A",
        dtype="float64",
        description=(
            "Literature distance in Å for the metal and donor, from Harding "
            "(2006), or Zheng et al. (2008) for nickel. Given for an exact "
            "reference only. The reference is per element and donor, not per "
            "oxidation state."
        ),
    ),
    TableColumnSpec(
        "reference_sd_A",
        dtype="float64",
        description="The spread in Å the literature gives with reference_distance_A.",
    ),
    TableColumnSpec(
        "first_sphere_cutoff_A",
        dtype="float64",
        description=(
            "The largest distance in Å at which the distance search infers "
            f"this bond, the target distance plus {_TOLERANCE} (Harding, "
            f"2004), never more than the {_SEARCH} search radius. Missing "
            "when reference_kind is missing."
        ),
    ),
    TableColumnSpec(
        "distance_deviation_A",
        dtype="float64",
        description=(
            "distance_A less reference_distance_A, in Å. Missing without an "
            "exact reference."
        ),
    ),
    TableColumnSpec(
        "distance_z",
        dtype="float64",
        description=(
            "distance_deviation_A divided by the square root of dpi_A squared "
            "plus reference_sd_A squared. One precision index stands for the "
            "donor's positional uncertainty, with the metal taken as well "
            "ordered. Missing without an exact reference or without a "
            "precision index. Not thresholded."
        ),
    ),
    TableColumnSpec(
        "donor_occupancy", dtype="float64", description="Occupancy of the donor atom."
    ),
    TableColumnSpec(
        "donor_b_factor_A2",
        dtype="float64",
        description="Isotropic B-factor of the donor atom in Å².",
    ),
    TableColumnSpec(
        "bond_status",
        categories=BOND_STATUSES,
        description=(
            "ok, or why a derived value is missing. no_reference when the bond "
            "has no exact reference distance, so its deviation and z are "
            "missing, and no_dpi when it has one but the entry has no "
            "precision index, so its z is missing."
        ),
    ),
)

METAL_BONDS_TABLE = TableSpec(
    name=METAL_BONDS_TABLE_NAME,
    description=(
        "One row per bond between a metal atom and a donor atom in one image. "
        "A bond is a row when a deposited _struct_conn metalc record names "
        "it, or when a nitrogen, oxygen, or sulfur atom of water or a "
        "standard amino acid, of a kind that commonly binds metals, lies "
        "within the literature distance for the pair plus "
        f"{_TOLERANCE} of the metal, searched to {_SEARCH} among the atoms of "
        "the file and of its symmetry mates. Hydrogen and deuterium atoms are "
        "never donors, and neither are atoms of the metal's own component. A "
        "metal the reference table has no row for gets rows from deposited "
        "records only."
    ),
    key=("site_id", f"{DONOR_PREFIX}site_id", "donor_symmetry_operator"),
    columns=(
        *site_key_columns(),
        *site_key_columns(DONOR_PREFIX),
        *_BOND_MEASURES,
    ),
    links=(
        TableLink(target=METAL_SITES_TABLE_NAME, columns=("site_id",)),
        site_atom_link(role="metal"),
        site_atom_link(DONOR_PREFIX, role="donor"),
    ),
)

_PAIR_MEASURES: tuple[TableColumnSpec, ...] = (
    TableColumnSpec(
        "metal_element",
        description="Element symbol of the first metal atom, in upper case.",
    ),
    TableColumnSpec(
        "partner_element",
        description="Element symbol of the second metal atom, in upper case.",
    ),
    TableColumnSpec(
        "distance_A",
        dtype="float64",
        description="Distance in Å between the two metal atoms. Not rounded.",
    ),
    TableColumnSpec(
        "same_component",
        dtype="boolean",
        description=(
            "True when the two metal atoms belong to one component, as the "
            "metals of one iron–sulfur or manganese cluster do. A cluster "
            "modelled as two overlapping components, each at part occupancy, "
            "puts a copy of every metal within a fraction of an ångström of "
            "its counterpart, and those rows are False."
        ),
    ),
)

METAL_PAIRS_TABLE = TableSpec(
    name=METAL_PAIRS_TABLE_NAME,
    description=(
        "One row per pair of metal atoms of one model that lie within "
        f"{NEARBY_METAL_RADIUS_A:g} Å of each other, among the atoms of the "
        "file only, each pair once with the atom the file lists first as the "
        "first. Both atoms are rows of metal_sites. A row gives a distance "
        "and asserts no bond between the two."
    ),
    key=("site_id", f"{PARTNER_PREFIX}site_id"),
    columns=(
        *site_key_columns(),
        *site_key_columns(PARTNER_PREFIX),
        *_PAIR_MEASURES,
    ),
    links=(
        TableLink(target=METAL_SITES_TABLE_NAME, columns=("site_id",), role="metal"),
        TableLink(
            target=METAL_SITES_TABLE_NAME,
            columns=(f"{PARTNER_PREFIX}site_id",),
            target_columns=("site_id",),
            role="partner",
        ),
        site_atom_link(role="metal"),
        site_atom_link(PARTNER_PREFIX, role="partner"),
    ),
)

METAL_LINK_COLUMN_SPEC = ColumnSpec(
    name=METAL_LINK_COLUMN,
    group="Metal Site Fields",
    granularity=ColumnGranularity.ATOM,
    dtype="text",
    description=(
        "The atom's own site_id when it is a metal atom with a row in the "
        "metal_sites table, and missing on every other atom."
    ),
    profiles=frozenset({PROFILE_RESEARCH_READY, PROFILE_RADIATION_SENSITIVITY}),
)

__all__ = [
    "DONOR_PREFIX",
    "METAL_BONDS_TABLE",
    "METAL_BONDS_TABLE_NAME",
    "METAL_LINK_COLUMN",
    "METAL_LINK_COLUMN_SPEC",
    "METAL_PAIRS_TABLE",
    "METAL_PAIRS_TABLE_NAME",
    "METAL_SITES_TABLE",
    "METAL_SITES_TABLE_NAME",
    "PARTNER_PREFIX",
]
