"""Reading a wwPDB validation XML report into the values PDB-AI keeps.

A report's root element holds an Entry element with the structure-level
summary and one ModelledSubgroup element per modelled residue. A residue
modelled in alternate conformers has one ModelledSubgroup per conformer,
each labelled by its altcode. For each residue PDB-AI keeps the real-space
correlation coefficient, the real-space R factor, and the Ramachandran and
rotamer classes. For the entry it keeps the clashscore and the percentages
of Ramachandran and rotamer outliers. Everything else in the report is
ignored.

Residues are keyed by model, author chain, author residue number, and
insertion code. These are the identifiers the report itself uses, so an
atom can find its residue through its own author fields. The model is part
of the key because a multi-model entry lists every residue once per model.
Each conformer is also kept under its own key, so an atom can find the
conformer the structure parser kept.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from ..scalars import to_float


def _text(value: str | None) -> str | None:
    """Return an attribute's trimmed text, or None if it is absent or blank."""
    if value is None:
        return None
    return value.strip() or None


def parse_validation_xml(xml_bytes: bytes) -> dict[str, object]:
    """Return the residue values and structure metrics of a validation report.

    Under residues, the result maps "model:chain:number", with the
    insertion code appended, to that residue's values: rscc, rsr,
    rama_class, and rotamer_class. Its other keys are clashscore,
    percent_rama_outliers, and percent_rotamer_outliers. A value the report
    leaves out, or one that does not parse, is None. A blank class is None
    too. A subgroup without a chain or residue number cannot be keyed, so it
    is skipped. A subgroup without a model belongs to model 1.

    A report without an Entry element but with subgroups gives its residue
    values and no metrics. Its metrics are then None, as a metric the Entry
    leaves out is. The residue values do not depend on the summary, so
    refusing the report would discard what it does say.

    A residue with alternate conformers is listed once per altcode, and the
    residue's key holds the values of one of them. That is the first one
    whose altcode is blank or A, and when there is none, the first one
    listed. Each conformer with an altcode is also kept under the residue's
    key followed by a colon and the altcode, such as "1:A:32:B". The first
    subgroup listed under such a key is kept.

    Raise xml.etree.ElementTree.ParseError when the bytes are not XML, and
    ValueError when the XML holds neither an Entry nor a ModelledSubgroup
    element, since it is then not a validation report.
    """
    root = ET.fromstring(xml_bytes)
    entry = root.find("Entry")
    subgroups = list(root.iter("ModelledSubgroup"))
    if entry is None and not subgroups:
        raise ValueError(
            f"XML with root element {root.tag!r} is not a validation report: "
            "it has no Entry or ModelledSubgroup element."
        )

    residues: dict[str, dict[str, object]] = {}
    # The residue keys whose kept subgroup has a blank or A altcode. A later
    # subgroup replaces the kept one only when the kept one is neither.
    keys_with_primary_conformer: set[str] = set()
    for subgroup in subgroups:
        model = _text(subgroup.get("model")) or "1"
        chain = _text(subgroup.get("chain"))
        residue_number = _text(subgroup.get("resnum"))
        insertion_code = _text(subgroup.get("icode")) or ""
        if chain is None or residue_number is None:
            continue
        key = f"{model}:{chain}:{residue_number}{insertion_code}"
        altcode = _text(subgroup.get("altcode")) or ""
        values: dict[str, object] = {
            "rscc": to_float(subgroup.get("rscc")),
            "rsr": to_float(subgroup.get("rsr")),
            "rama_class": _text(subgroup.get("rama")),
            "rotamer_class": _text(subgroup.get("rota")),
        }
        if altcode:
            residues.setdefault(f"{key}:{altcode}", values)
        is_primary_conformer = altcode in {"", "A"}
        if key in keys_with_primary_conformer or (
            key in residues and not is_primary_conformer
        ):
            continue
        if is_primary_conformer:
            keys_with_primary_conformer.add(key)
        residues[key] = values

    summary = entry.attrib if entry is not None else {}
    return {
        "residues": residues,
        "clashscore": to_float(summary.get("clashscore")),
        "percent_rama_outliers": to_float(summary.get("percent-rama-outliers")),
        "percent_rotamer_outliers": to_float(summary.get("percent-rota-outliers")),
    }


__all__ = ["parse_validation_xml"]
