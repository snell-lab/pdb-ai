"""The bundled table of radiolysis rate constants, and the receipt that identifies it.

radiolysis_rate_constants.csv gives, for each standard amino acid, the
second-order rate constants of its reaction in water with the hydroxyl
radical and with the hydrated electron, the two radicals of water radiolysis
that do most of the damage in solution. Each row is one measurement as a
compilation gives it, with its pH, its method, and where in the source it is
printed, so a residue can have several rows. One row per amino acid and
radical, at near-neutral pH, is marked selected. Every value was read from
the page images of Buxton et al. (1988), the standard compilation. A value
that could not be read there is left out, not filled in, and the manifest
lists what was left out and why. Cystine is listed under CYS, with its form
saying so, and is never the selected row.

The table is a reference shipped with PDB-AI and not columns of the
residue_sites table. A rate constant belongs to a free amino acid in a
stated solution, not to a group in a crystal. It is one value per residue
type, which a join on res_name gives any table, the atom table included,
while residue_sites holds rows for a few residue types only. And it carries
a pH and a method that no row could hold without repeating them on every
row.

Its manifest records the sources and the SHA-256 digest of the file. Every
run that executes the residue-site engine records a receipt of the table,
which measures the file's digest afresh, so an edited table shows up as a
mismatch.
"""

from __future__ import annotations

import json
import math
from functools import cache
from pathlib import Path

import pandas as pd

from ....json_values import object_dict
from ....provenance import PROJECT_ROOT, file_receipt
from ..metal_sites.constants import STANDARD_AMINO_ACIDS

_DIRECTORY = Path(__file__).resolve().parent
RATE_CONSTANTS_PATH = _DIRECTORY / "radiolysis_rate_constants.csv"
REFERENCE_MANIFEST_PATH = _DIRECTORY / "reference_data.manifest.json"

RADICAL_HYDROXYL = "OH"
RADICAL_HYDRATED_ELECTRON = "e_aq"
RADICALS: tuple[str, ...] = (RADICAL_HYDROXYL, RADICAL_HYDRATED_ELECTRON)

_COLUMNS = (
    "residue",
    "radical",
    "form",
    "k_M_per_s",
    "k_qualifier",
    "pH",
    "is_average",
    "selected",
    "buxton_entry",
    "source_location",
    "conditions",
    "notes",
)
# A value printed as an upper limit, or as approximate. Empty for a value.
K_QUALIFIERS: tuple[str, ...] = ("", "<", "~")
_FLAGS = ("is_average", "selected")


@cache
def _read_rate_constants() -> pd.DataFrame:
    table = pd.read_csv(
        RATE_CONSTANTS_PATH, dtype=str, keep_default_na=False, comment=None
    )
    if tuple(table.columns) != _COLUMNS:
        raise ValueError(
            f"{RATE_CONSTANTS_PATH.name} has the columns {list(table.columns)}, "
            f"expected {list(_COLUMNS)}"
        )
    if table.empty:
        raise ValueError(f"{RATE_CONSTANTS_PATH.name} holds no rate constants")
    for number, row in enumerate(table.itertuples(index=False), start=2):
        if row.residue not in STANDARD_AMINO_ACIDS:
            raise ValueError(
                f"{RATE_CONSTANTS_PATH.name} line {number} names {row.residue!r}, "
                "which is not a standard amino acid"
            )
        if row.radical not in RADICALS:
            raise ValueError(
                f"{RATE_CONSTANTS_PATH.name} line {number} names the radical "
                f"{row.radical!r}, expected one of {list(RADICALS)}"
            )
        value = float(str(row.k_M_per_s))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"{RATE_CONSTANTS_PATH.name} line {number} has a rate constant "
                "that is not a positive number"
            )
        if row.k_qualifier not in K_QUALIFIERS:
            raise ValueError(
                f"{RATE_CONSTANTS_PATH.name} line {number} has the qualifier "
                f"{row.k_qualifier!r}, expected one of {list(K_QUALIFIERS)}"
            )
        for flag in _FLAGS:
            if getattr(row, flag) not in ("true", "false"):
                raise ValueError(
                    f"{RATE_CONSTANTS_PATH.name} line {number} has {flag} "
                    f"{getattr(row, flag)!r}, expected true or false"
                )
    table["k_M_per_s"] = table["k_M_per_s"].astype(float)
    for flag in _FLAGS:
        table[flag] = table[flag] == "true"
    chosen = table[table["selected"]]
    repeated = chosen.duplicated(subset=["residue", "radical"])
    if repeated.any():
        first = chosen[repeated].iloc[0]
        raise ValueError(
            f"{RATE_CONSTANTS_PATH.name} selects more than one value for "
            f"{first['residue']} and {first['radical']}"
        )
    return table


def radiolysis_rate_constants() -> pd.DataFrame:
    """Return every row of the table, one measurement each.

    k_M_per_s is the rate constant in dm³ mol⁻¹ s⁻¹, and k_qualifier is <
    where the source prints an upper limit and ~ where it prints an
    approximate value. is_average marks one of the source's averages over
    the measurements listed beneath it, and selected the one value per
    amino acid and radical. A row naming a residue that is not a standard amino acid or a
    radical other than OH and e_aq, a rate constant that is not a positive
    number, or two selected values for one residue and radical raises
    ValueError, so a damaged table cannot pass for one with fewer values.
    """
    return _read_rate_constants().copy()


def selected_rate_constants() -> pd.DataFrame:
    """Return the one selected row per amino acid and radical.

    The selected row is the source's average for the form that dominates at
    pH 7 where it prints one, and otherwise the value nearest pH 7. An amino
    acid without a selected row for a radical has no value for it.
    """
    table = _read_rate_constants()
    return table[table["selected"]].reset_index(drop=True)


def reference_receipts() -> dict[str, object]:
    """Return a receipt of the bundled table, keyed by its name.

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
    entry = object_dict(declared.get("radiolysis_rate_constants"))
    receipt = file_receipt(
        RATE_CONSTANTS_PATH,
        role="residue_sites_radiolysis_rate_constants",
        relative_to=PROJECT_ROOT,
    )
    declared_hash = entry.get("sha256")
    receipt.update(
        {
            "label": "Radiolysis rate constants of the amino acids",
            "source": entry.get("source"),
            "citations": entry.get("citations"),
            "declared_sha256": declared_hash,
            "hash_matches_manifest": (
                receipt.get("sha256") == declared_hash
                if receipt.get("sha256") is not None and declared_hash is not None
                else None
            ),
        }
    )
    return {"residue_sites_radiolysis_rate_constants": receipt}


__all__ = [
    "RADICALS",
    "RADICAL_HYDRATED_ELECTRON",
    "RADICAL_HYDROXYL",
    "K_QUALIFIERS",
    "RATE_CONSTANTS_PATH",
    "REFERENCE_MANIFEST_PATH",
    "radiolysis_rate_constants",
    "reference_receipts",
    "selected_rate_constants",
]
