"""The two bundled reference tables, and the receipts that identify them.

metal_donor_distances.txt gives the literature distance and its spread for a
metal and a donor, from Harding (2006) for every metal but nickel and from
Zheng et al. (2008) for nickel. It covers ten metals, NA, MG, K, CA, MN, FE,
CO, NI, CU, and ZN, and the donors water oxygen, carboxylate oxygen, backbone
carbonyl oxygen, histidine nitrogen, and cysteine sulfur, with hydroxyl
oxygens added from statements in Harding (2006). A pair outside it has no
reference, and its bond row says so.

metallocofactors.txt lists every component of the Chemical Component
Dictionary that holds a metal and more than one atom, with the class of
those whose metals sit in a cluster or a heme.

Both tables are those of Alchemy 1.0.0 and are bundled unchanged. Their
manifest records where each came from and its SHA-256 digest. Every run that
executes the engine records a receipt of each, which measures the file's
digest afresh, so an edited table shows up as a mismatch.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from types import MappingProxyType

from ....json_values import object_dict
from ....provenance import PROJECT_ROOT, file_receipt
from .constants import PARENT_CLUSTER, PARENT_HEME

_DIRECTORY = Path(__file__).resolve().parent
DISTANCE_TABLE_PATH = _DIRECTORY / "metal_donor_distances.txt"
COFACTOR_CATALOG_PATH = _DIRECTORY / "metallocofactors.txt"
REFERENCE_MANIFEST_PATH = _DIRECTORY / "reference_data.manifest.json"

_DISTANCE_TABLE_HEADER = ("residue", "atom", "metal", "avg_bond_dist", "st_dev")

type ReferenceKey = tuple[str, str, str]


@cache
def literature_distances() -> Mapping[ReferenceKey, tuple[float, float]]:
    """Return the literature table as {(residue, atom, metal): (mean, spread)}.

    Distances are in Å. In the residue position, CA stands for the backbone
    carbonyl. A row that does not hold five fields, or whose distances are
    not positive numbers, raises ValueError, and so does a key given twice,
    so a damaged table cannot pass for one with fewer references.
    """
    table: dict[ReferenceKey, tuple[float, float]] = {}
    for number, line in enumerate(
        DISTANCE_TABLE_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split()
        if (
            not parts
            or parts[0].startswith("#")
            or tuple(parts) == _DISTANCE_TABLE_HEADER
        ):
            continue
        if len(parts) != 5:
            raise ValueError(
                f"{DISTANCE_TABLE_PATH.name} line {number} has {len(parts)} fields, "
                "expected 5"
            )
        mean, spread = float(parts[3]), float(parts[4])
        if (
            not (math.isfinite(mean) and math.isfinite(spread))
            or mean <= 0
            or spread <= 0
        ):
            raise ValueError(
                f"{DISTANCE_TABLE_PATH.name} line {number} has distances that are "
                "not positive numbers"
            )
        key = (parts[0], parts[1], parts[2])
        if key in table:
            raise ValueError(
                f"{DISTANCE_TABLE_PATH.name} line {number} repeats the key "
                f"{' '.join(key)!r}"
            )
        table[key] = (mean, spread)
    if not table:
        raise ValueError(f"{DISTANCE_TABLE_PATH.name} holds no reference distances")
    return MappingProxyType(table)


@cache
def element_targets() -> Mapping[tuple[str, str], float]:
    """Return the longest literature distance for each metal and donor element.

    The key is (metal, donor element). The value decides first-sphere
    membership for a donor the table has no row of its own for.
    """
    targets: dict[tuple[str, str], float] = {}
    for (_residue, donor, metal), (mean, _spread) in literature_distances().items():
        key = (metal, donor)
        targets[key] = max(mean, targets.get(key, -math.inf))
    return MappingProxyType(targets)


@cache
def cofactor_classes() -> Mapping[str, str]:
    """Return the cataloged components as {component id: class}.

    The class is cluster, heme, or empty for any other metal-containing
    component. An empty catalog raises ValueError.
    """
    classes: dict[str, str] = {}
    for line in COFACTOR_CATALOG_PATH.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        component = fields[0].strip()
        if not component:
            continue
        structural_class = fields[2].strip() if len(fields) > 2 else ""
        classes[component] = (
            structural_class
            if structural_class in (PARENT_CLUSTER, PARENT_HEME)
            else ""
        )
    if not classes:
        raise ValueError(f"{COFACTOR_CATALOG_PATH.name} is empty")
    return MappingProxyType(classes)


def reference_receipts() -> dict[str, object]:
    """Return a receipt of each bundled table, keyed by the table's name.

    A receipt adds the manifest's declared fields to the file's own receipt.
    hash_matches_manifest is None when either digest is missing. A manifest
    that is missing or not JSON leaves the declared fields None and does not
    raise.
    """
    declared: dict[str, object] = {}
    try:
        raw: object = json.loads(REFERENCE_MANIFEST_PATH.read_text(encoding="utf-8"))
        declared = object_dict(raw)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    receipts: dict[str, object] = {}
    for name, path, label in (
        (
            "metal_donor_distances",
            DISTANCE_TABLE_PATH,
            "Metal–donor reference distances",
        ),
        (
            "metallocofactors",
            COFACTOR_CATALOG_PATH,
            "Metal-containing component catalog",
        ),
    ):
        entry = object_dict(declared.get(name))
        receipt = file_receipt(
            path, role=f"metal_sites_{name}", relative_to=PROJECT_ROOT
        )
        declared_hash = entry.get("sha256")
        receipt.update(
            {
                "label": label,
                "source": entry.get("source"),
                "citations": entry.get("citations"),
                "upstream_provenance": entry.get("upstream_provenance"),
                "declared_sha256": declared_hash,
                "hash_matches_manifest": (
                    receipt.get("sha256") == declared_hash
                    if receipt.get("sha256") is not None and declared_hash is not None
                    else None
                ),
            }
        )
        receipts[f"metal_sites_{name}"] = receipt
    return receipts


__all__ = [
    "COFACTOR_CATALOG_PATH",
    "DISTANCE_TABLE_PATH",
    "REFERENCE_MANIFEST_PATH",
    "ReferenceKey",
    "cofactor_classes",
    "element_targets",
    "literature_distances",
    "reference_receipts",
]
