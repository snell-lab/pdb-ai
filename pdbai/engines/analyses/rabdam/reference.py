"""The bundled Bnet reference table and the receipt that identifies it.

rabdam-custom ranks a structure's Bnet among reference structures of
similar resolution. That percentile means something only relative to one
fixed table. The installed package leaves out the table its repository
builds, so a copy is bundled beside this module as
rabdam_custom_bnet_reference.csv. Its manifest records the upstream
repository, ref, and version of the copy. It also records the PDB-Redo
snapshot the table was built from, its entry count, and its SHA-256 digest.

Every run that executes RABDAM records the receipt in its manifest. Each
bnet_percentile can therefore be traced to the table it was ranked in. The
receipt measures the file's digest afresh and compares it with the
declared one. An edited or replaced table therefore shows up as a mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path

from ....json_values import object_dict
from ....provenance import PROJECT_ROOT, file_receipt

REFERENCE_PATH = Path(__file__).resolve().parent / "rabdam_custom_bnet_reference.csv"
REFERENCE_MANIFEST_PATH = REFERENCE_PATH.with_suffix(".manifest.json")


def reference_receipts() -> dict[str, object]:
    """Return the reference table's receipt under the key rabdam_bnet.

    The receipt adds the manifest's declared fields to the file's own
    receipt. hash_matches_manifest is None when either digest is missing.
    A manifest that is missing or not JSON leaves the declared fields None.
    It does not raise, since the scores do not depend on the manifest.
    """
    declared: dict[str, object] = {}
    try:
        raw: object = json.loads(REFERENCE_MANIFEST_PATH.read_text(encoding="utf-8"))
        declared = object_dict(raw)
    except (OSError, json.JSONDecodeError):
        pass

    receipt = file_receipt(
        REFERENCE_PATH,
        role="rabdam_bnet_reference_database",
        relative_to=PROJECT_ROOT,
    )
    declared_hash = declared.get("sha256")
    receipt.update(
        {
            "label": "RABDAM reference database",
            "database_id": declared.get("database_id"),
            "schema_version": declared.get("schema_version"),
            "entry_count": declared.get("entry_count"),
            "source": declared.get("source"),
            "upstream_provenance": declared.get("upstream_provenance"),
            "declared_sha256": declared_hash,
            "hash_matches_manifest": (
                receipt.get("sha256") == declared_hash
                if receipt.get("sha256") is not None and declared_hash is not None
                else None
            ),
        }
    )
    return {"rabdam_bnet": receipt}


__all__ = ["REFERENCE_MANIFEST_PATH", "REFERENCE_PATH", "reference_receipts"]
