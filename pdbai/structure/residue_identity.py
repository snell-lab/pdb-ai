"""Model, chain, and residue identity for the rows of an atom table.

The coordinate engines, the star schema's join keys, and the PDB-Redo
alignment must agree on which atoms make up one residue and one chain. They
all read that identity here. Every key names the entry and the coordinate
model, so the models of an NMR ensemble never share a residue. A residue with
too little identity gets no key, rather than one it would share with an
unrelated residue.

Within one file, a polymer residue is keyed by its label fields, which are
unique there. Label fields do not hold across files. PDB-Redo renumbers
label_seq_id from the first residue it models and reassigns label_asym_id in
order. So a key that joins two files uses only the author fields, which
PDB-Redo keeps as deposited.

Polymer and protein membership come first from the entity metadata the
parser attaches. Only where that is missing do they come from the chemical
component type, the label sequence number, and the ATOM or HETATM record.

The names a person reads, chain_id, res_num, and res_name, are decided here
too, by atom_site_residue_identity. They are the author items, since papers
and the PDB format number residues that way. Some files leave author items
out. Some integrative entries give no author component, and some of those
give no author residue number either. A file written by gemmi's default
writer gives no author component. A residue
name then falls back to the label component, row by row, since a name stands
on its own. The chain and the residue number fall back together and for the
whole file, since a label number is unique only within a label chain, and a
table that mixed the two numberings could give two residues one name. The
parser writes the numbering it used to the residue_numbering column. The
keys of this module do not change with it. Under label numbering the named
columns hold label values, so the author fields of a key are then read from
the author items alone, and a key that joins two files stays an author key
or is missing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from ..missing_text import MISSING_TEXTS
from ..refused_values import listed_values

ResidueKey = tuple[str, ...]

# The numbering the named columns chain_id and res_num of one structure use,
# and the column the parser writes it to.
type ResidueNumbering = Literal["author", "label"]
RESIDUE_NUMBERING_AUTHOR: ResidueNumbering = "author"
RESIDUE_NUMBERING_LABEL: ResidueNumbering = "label"
RESIDUE_NUMBERING_COLUMN = "residue_numbering"

_AUTHOR_CHAIN_ITEM = "_atom_site.auth_asym_id"
_AUTHOR_NUMBER_ITEM = "_atom_site.auth_seq_id"
_AUTHOR_COMPONENT_ITEM = "_atom_site.auth_comp_id"
_LABEL_CHAIN_ITEM = "_atom_site.label_asym_id"
_LABEL_NUMBER_ITEM = "_atom_site.label_seq_id"
_LABEL_COMPONENT_ITEM = "_atom_site.label_comp_id"

# The component IDs read as protein when no polymer type says otherwise. They
# are the twenty standard amino acids, the ambiguous ASX, GLX, and UNK, and the
# modified amino acids most often deposited as HETATM records.
_PROTEIN_COMPONENT_IDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "ASX",
        "GLX",
        "UNK",
        "MSE",
        "SEC",
        "CSE",
        "HYP",
        "SEP",
        "TPO",
        "PTR",
    }
)

# The spellings of a true and a false flag, compared in lower case. A missing
# text also reads as false. HETATM and ATOM are here so that the mmCIF record
# type reads as the is_hetatm flag.
_TRUE_TEXTS = frozenset({"1", "1.0", "true", "t", "yes", "y", "hetatm"})
_FALSE_TEXTS = frozenset({"0", "0.0", "false", "f", "no", "n", "atom"})


def _stripped_text(values: pd.Series) -> pd.Series:
    # Going through object dtype turns a missing value of any dtype into the
    # empty text. A nullable integer's missing value would otherwise raise or
    # become "<NA>".
    return values.astype(object).where(values.notna(), "").astype(str).str.strip()


def _refusal(column: str, refused: pd.Series, kind: str) -> ValueError:
    return ValueError(
        f"Column {column!r} holds value(s) that are not {kind}: "
        f"{listed_values(refused.tolist())}."
    )


def text_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Return a column as the text it holds, empty where missing or absent.

    The text is not stripped and no placeholder is read as missing, so a
    named column made this way holds what the file writes.
    """
    if column not in dataframe.columns:
        return pd.Series("", index=dataframe.index, dtype=str)
    return dataframe[column].fillna("").astype(str)


def clean_string_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Return a column as stripped text, with every missing value as "".

    A value is missing when it is null or, stripped and in lower case, one of
    MISSING_TEXTS, such as "?" or ".". A DataFrame without the column gives ""
    for every row.
    """
    if column not in dataframe.columns:
        return pd.Series("", index=dataframe.index, dtype=str)
    values = _stripped_text(dataframe[column])
    return values.mask(values.str.lower().isin(MISSING_TEXTS), "")


def first_nonempty_string_series(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    default: str = "",
) -> pd.Series:
    """Return each row's first nonempty text across columns, or the default.

    The columns are tried in order, each read as clean_string_series reads it,
    and a column the DataFrame lacks is skipped.
    """
    result = pd.Series("", index=dataframe.index, dtype=str)
    for column in columns:
        missing = result.eq("")
        if not missing.any():
            break
        result = result.where(~missing, clean_string_series(dataframe, column))
    return result.where(result.ne(""), default)


def _numbers(dataframe: pd.DataFrame, column: str) -> pd.Series:
    values = dataframe[column]
    if pd.api.types.is_numeric_dtype(values.dtype):
        return values.astype("float64")
    text = _stripped_text(values)
    missing = text.str.lower().isin(MISSING_TEXTS)
    numbers = pd.to_numeric(text.where(~missing), errors="coerce").astype("float64")
    refused = numbers.isna() & ~missing
    if refused.any():
        raise _refusal(column, values[refused], "numbers")
    return numbers


def first_numeric_series(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    *,
    default_if_absent: float = float("nan"),
) -> pd.Series:
    """Return each row's first present number across columns, as floats.

    The columns are tried in order, and a missing value in one falls through
    to the next. When the DataFrame has none of the columns, every row holds
    default_if_absent. When it has some, a row missing from all of them stays
    NaN. Raises ValueError naming the column and its values that are neither
    a number nor missing, rather than reading them as missing.
    """
    present = [column for column in columns if column in dataframe.columns]
    if not present:
        return pd.Series(default_if_absent, index=dataframe.index, dtype="float64")
    result = _numbers(dataframe, present[0])
    for column in present[1:]:
        result = result.fillna(_numbers(dataframe, column))
    return result


def coerce_bool_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Return a flag column as Booleans, reading a missing value as False.

    The column may hold Booleans, the numbers 1 and 0, or text in any case and
    with spaces around it. True is spelled true, t, yes, y, 1, 1.0, or HETATM,
    and false is spelled false, f, no, n, 0, 0.0, or ATOM. A DataFrame without
    the column gives False for every row. Raises ValueError naming the column
    and the values it cannot read, rather than reading them as False.
    """
    if column not in dataframe.columns:
        return pd.Series(False, index=dataframe.index, dtype=bool)
    values = dataframe[column]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = _stripped_text(values).str.lower()
    true = normalized.isin(_TRUE_TEXTS)
    refused = ~(true | normalized.isin(_FALSE_TEXTS | MISSING_TEXTS))
    if refused.any():
        raise _refusal(column, values[refused], "flags")
    return true.astype(bool)


@dataclass(frozen=True)
class AtomSiteResidueIdentity:
    """The named residue columns of _atom_site rows, under the rows' index.

    chain_id, res_num, and res_name hold text, empty where the file gives no
    value. numbering says which items chain_id and res_num were read from.
    """

    chain_id: pd.Series
    res_num: pd.Series
    res_name: pd.Series
    numbering: ResidueNumbering


def _gives_any_value(
    atom_site: pd.DataFrame | Mapping[str, Sequence[str | None]], item: str
) -> bool:
    """Return whether an item is present with a value on at least one row."""
    if isinstance(atom_site, pd.DataFrame):
        return bool(text_series(atom_site, item).ne("").any())
    return any(value is not None and value != "" for value in atom_site.get(item, ()))


def atom_site_residue_numbering(
    atom_site: pd.DataFrame | Mapping[str, Sequence[str | None]],
) -> ResidueNumbering:
    """Return the numbering the chain and residue number of a file use.

    ``atom_site`` holds the _atom_site items of every atom of the file under
    their mmCIF names, as a DataFrame or as the flat dictionary
    read_mmcif_as_flat_dict returns. The numbering is "author" when the file
    gives an author chain on at least one atom and an author residue number
    on at least one atom. It is "label" otherwise, which is when
    _atom_site.auth_asym_id or _atom_site.auth_seq_id is absent, or is a
    placeholder or the empty text on every atom. The decision is made once
    for the file, so it must see every atom and not a selection of them.
    """
    if _gives_any_value(atom_site, _AUTHOR_CHAIN_ITEM) and _gives_any_value(
        atom_site, _AUTHOR_NUMBER_ITEM
    ):
        return RESIDUE_NUMBERING_AUTHOR
    return RESIDUE_NUMBERING_LABEL


def atom_site_residue_names(atom_site: pd.DataFrame) -> pd.Series:
    """Return the residue name of each _atom_site row, empty when it has none.

    The name is _atom_site.auth_comp_id, or _atom_site.label_comp_id on a
    row where the author component is absent, a placeholder, or the empty
    text. The fallback is row by row, since a component name does not depend
    on the numbering of the other rows.
    """
    author = text_series(atom_site, _AUTHOR_COMPONENT_ITEM)
    return author.where(author.ne(""), text_series(atom_site, _LABEL_COMPONENT_ITEM))


def atom_site_residue_identity(
    atom_site: pd.DataFrame, *, numbering: ResidueNumbering | None = None
) -> AtomSiteResidueIdentity:
    """Return the chain_id, res_num, and res_name of each _atom_site row.

    ``atom_site`` holds _atom_site items under their mmCIF names, with None
    or the empty text where the file writes a placeholder. This is the one
    rule for the named columns, which the parser and every engine that reads
    a file again share, so that their rows carry the same names.

    Under author numbering chain_id is _atom_site.auth_asym_id and res_num
    is _atom_site.auth_seq_id on every row, even on a row that lacks one.
    Under label numbering they are _atom_site.label_asym_id and
    _atom_site.label_seq_id on every row. The two are never mixed within a
    structure. res_name is what atom_site_residue_names returns under either
    numbering. A value is the text the file writes, not stripped, and empty
    where the item is absent or a placeholder.

    ``numbering`` is the numbering to apply. Pass it when ``atom_site``
    holds a selection of a file's atoms, with the value
    atom_site_residue_numbering gives for the whole file. Left out, it is
    decided from ``atom_site``, which must then hold every atom of the file.
    """
    if numbering is None:
        numbering = atom_site_residue_numbering(atom_site)
    chain_item, number_item = (
        (_AUTHOR_CHAIN_ITEM, _AUTHOR_NUMBER_ITEM)
        if numbering == RESIDUE_NUMBERING_AUTHOR
        else (_LABEL_CHAIN_ITEM, _LABEL_NUMBER_ITEM)
    )
    return AtomSiteResidueIdentity(
        chain_id=text_series(atom_site, chain_item),
        res_num=text_series(atom_site, number_item),
        res_name=atom_site_residue_names(atom_site),
        numbering=numbering,
    )


def author_chain_and_number(dataframe: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return each row's author chain and author residue number, "" if none.

    Each is read from its named column, chain_id or res_num, where the table
    has it and from its mmCIF author item otherwise. On a row whose
    residue_numbering column says "label" the named columns hold label
    values, so both are read from the mmCIF author items alone. A table
    without that column is taken to use author numbering.
    """
    label_numbered = clean_string_series(dataframe, RESIDUE_NUMBERING_COLUMN).eq(
        RESIDUE_NUMBERING_LABEL
    )
    chain = first_nonempty_string_series(dataframe, ("chain_id", _AUTHOR_CHAIN_ITEM))
    number = first_nonempty_string_series(dataframe, ("res_num", _AUTHOR_NUMBER_ITEM))
    if not label_numbered.any():
        return chain, number
    return (
        chain.where(
            ~label_numbered, clean_string_series(dataframe, _AUTHOR_CHAIN_ITEM)
        ),
        number.where(
            ~label_numbered, clean_string_series(dataframe, _AUTHOR_NUMBER_ITEM)
        ),
    )


@dataclass
class ResidueIdentities:
    """The identity of every row of an atom table, under the table's index.

    pdb_id, model, auth_chain, label_chain, auth_seq, label_seq,
    insertion_code, and res_name hold the identity fields as text. Each is ""
    where a row lacks it, except that a missing model is "1". res_name is in
    upper case. chain is the label chain, or the author chain where there is
    no label chain. polymer and protein_polymer are Boolean masks. auth_chain
    and auth_seq hold author values under either numbering of the named
    columns, as author_chain_and_number reads them.

    residue_keys holds each row's residue key within its file, or None. A polymer
    residue with a label chain and a label sequence number has the key
    ("label", pdb_id, model, label_chain, label_seq). Any other residue has
    the key ("auth", pdb_id, model, auth_chain, auth_seq, insertion_code,
    res_name), or None when it lacks an author chain or an author sequence
    number. residue_join_keys holds the same keys joined by "|", the values of
    the star schema's residue_key column. chain_keys holds
    "chain|pdb_id|model|chain", the values of its chain_key column, or None
    without a chain.
    """

    pdb_id: pd.Series
    model: pd.Series
    auth_chain: pd.Series
    label_chain: pd.Series
    chain: pd.Series
    auth_seq: pd.Series
    label_seq: pd.Series
    insertion_code: pd.Series
    res_name: pd.Series
    polymer: pd.Series
    protein_polymer: pd.Series
    residue_keys: pd.Series
    residue_join_keys: pd.Series
    chain_keys: pd.Series


def _author_residue_keys(
    pdb_id: pd.Series,
    model: pd.Series,
    auth_chain: pd.Series,
    auth_seq: pd.Series,
    insertion_code: pd.Series,
    res_name: pd.Series,
) -> list[ResidueKey | None]:
    """Return each row's author residue key, or None without chain or number."""
    return [
        ("auth", entry, model_value, chain, number, insertion, name)
        if chain and number
        else None
        for entry, model_value, chain, number, insertion, name in zip(
            pdb_id, model, auth_chain, auth_seq, insertion_code, res_name, strict=True
        )
    ]


def _join_key(key: ResidueKey | None) -> str | None:
    return "|".join(key) if key is not None else None


def build_residue_identities(dataframe: pd.DataFrame) -> ResidueIdentities:
    """Return the model, chain, and residue identity of each atom table row.

    Each identity field is read from its short column name where the table has
    it and from its mmCIF column otherwise. The author chain and the author
    residue number are the exception under label numbering, where
    author_chain_and_number reads them from the mmCIF author items alone. So
    a table gets the same keys whichever numbering its named columns use.
    Whether a row is HETATM, which decides polymer membership where entity
    metadata is missing, is read from the is_hetatm column, or else from
    _atom_site.group_PDB. Raises ValueError when that flag holds a value it
    cannot read.
    """
    pdb_id = clean_string_series(dataframe, "pdb_id")
    model = first_nonempty_string_series(
        dataframe,
        ("model_num", "_atom_site.pdbx_PDB_model_num"),
        "1",
    )
    auth_chain, auth_seq = author_chain_and_number(dataframe)
    label_chain = clean_string_series(dataframe, _LABEL_CHAIN_ITEM)
    chain = label_chain.where(label_chain.ne(""), auth_chain)
    label_seq = clean_string_series(dataframe, _LABEL_NUMBER_ITEM)
    insertion_code = clean_string_series(dataframe, "_atom_site.pdbx_PDB_ins_code")
    res_name = first_nonempty_string_series(
        dataframe,
        ("res_name", _AUTHOR_COMPONENT_ITEM, _LABEL_COMPONENT_ITEM),
    ).str.upper()
    hetatm_column = (
        "is_hetatm" if "is_hetatm" in dataframe.columns else "_atom_site.group_PDB"
    )
    is_hetatm = coerce_bool_series(dataframe, hetatm_column)

    polymer_type = clean_string_series(dataframe, "_pdbai_polymer_type").str.lower()
    entity_type = clean_string_series(dataframe, "_pdbai_entity_type").str.lower()
    chem_comp_type = clean_string_series(dataframe, "_pdbai_chem_comp_type").str.lower()
    peptide_linking = chem_comp_type.str.contains(
        "peptide", regex=False
    ) & chem_comp_type.str.contains("linking", regex=False)

    # Entity metadata outranks the ATOM or HETATM record. Modified polymer
    # residues are often HETATM. An unusual ATOM record should not turn a
    # known ligand into a polymer residue.
    metadata_polymer = polymer_type.ne("") | entity_type.eq("polymer")
    metadata_nonpolymer = entity_type.ne("") & ~entity_type.eq("polymer")
    polymer = metadata_polymer | (
        ~metadata_nonpolymer & (label_seq.ne("") | peptide_linking | ~is_hetatm)
    )
    known_polymer_type = polymer_type.ne("")
    protein_polymer = polymer & (
        polymer_type.str.contains("polypeptide", regex=False)
        | (
            ~known_polymer_type
            & (peptide_linking | res_name.isin(_PROTEIN_COMPONENT_IDS))
        )
    )

    author_keys = _author_residue_keys(
        pdb_id, model, auth_chain, auth_seq, insertion_code, res_name
    )
    keyed_by_label = polymer & label_chain.ne("") & label_seq.ne("")
    residue_keys = pd.Series(
        [
            ("label", pdb_id_value, model_value, label_chain_value, label_seq_value)
            if by_label
            else author_key
            for (
                pdb_id_value,
                model_value,
                label_chain_value,
                label_seq_value,
                by_label,
                author_key,
            ) in zip(
                pdb_id,
                model,
                label_chain,
                label_seq,
                keyed_by_label,
                author_keys,
                strict=True,
            )
        ],
        index=dataframe.index,
        dtype=object,
    )
    residue_join_keys = residue_keys.map(_join_key)
    chain_keys = pd.Series(
        [
            f"chain|{pdb_id_value}|{model_value}|{chain_value}" if chain_value else None
            for pdb_id_value, model_value, chain_value in zip(
                pdb_id, model, chain, strict=True
            )
        ],
        index=dataframe.index,
        dtype=object,
    )

    return ResidueIdentities(
        pdb_id=pdb_id,
        model=model,
        auth_chain=auth_chain,
        label_chain=label_chain,
        chain=chain,
        auth_seq=auth_seq,
        label_seq=label_seq,
        insertion_code=insertion_code,
        res_name=res_name,
        polymer=polymer.astype(bool),
        protein_polymer=protein_polymer.astype(bool),
        residue_keys=residue_keys,
        residue_join_keys=residue_join_keys,
        chain_keys=chain_keys,
    )


def build_author_residue_keys(identities: ResidueIdentities) -> pd.Series:
    """Return every row's residue key from its author fields alone, or None.

    A key is ("auth", pdb_id, model, auth_chain, auth_seq, insertion_code,
    res_name), and a row without an author chain or an author sequence
    number has none. These keys match atoms across independently produced
    files, such as an RCSB entry and its PDB-Redo re-refinement. The residue
    name is part of the key, so a numbering difference cannot pair two
    different residues that share atom names. A join within one file uses
    identities.residue_keys instead.
    """
    keys = _author_residue_keys(
        identities.pdb_id,
        identities.model,
        identities.auth_chain,
        identities.auth_seq,
        identities.insertion_code,
        identities.res_name,
    )
    return pd.Series(keys, index=identities.residue_keys.index, dtype=object)


__all__ = [
    "RESIDUE_NUMBERING_AUTHOR",
    "RESIDUE_NUMBERING_COLUMN",
    "RESIDUE_NUMBERING_LABEL",
    "AtomSiteResidueIdentity",
    "ResidueIdentities",
    "ResidueKey",
    "ResidueNumbering",
    "atom_site_residue_identity",
    "atom_site_residue_names",
    "atom_site_residue_numbering",
    "author_chain_and_number",
    "build_author_residue_keys",
    "build_residue_identities",
    "clean_string_series",
    "coerce_bool_series",
    "first_nonempty_string_series",
    "first_numeric_series",
    "text_series",
]
