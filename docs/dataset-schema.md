# Dataset Schema

This reference explains output shapes, column profiles, nullable values, and
curated dataset columns.

[Project README](../README.md)

## Dataset shape: flat vs. star_schema

Every enrichment engine feeds into the same fully-computed atom frame.
`dataset_shape` only controls how that frame is written out.

- **`"flat"`** (default). It writes one CSV, one row per atom, with every
  residue/chain/structure-level value (B-factor stats, resolution, r_work, bnet,
  citation metadata, …) broadcast onto every atom row that shares it.
- **`"star_schema"`**, the same data split into five linked tables. The atoms
  table is published as the primary file (`pdb_atom_site_dataset.csv`, or the
  `--output` name), and each other table as `<stem>_<table>.csv` beside it, as
  in `pdb_atom_site_dataset_residues.csv`.
  - The atoms table: truly per-atom columns (coordinates, element,
    bdamage_pdb_redo, `atom_sasa_protein`, `atom_sasa_bound`, …) plus a
    `residue_key` foreign key.
  - The residues table: columns constant within a residue (both residue
    SASA/RASA contexts, eligibility masks, B-factor statistics, AlphaFold
    Database provenance/pLDDT, …), keyed by `residue_key`, with a `chain_key`
    foreign key.
  - The chains table: columns constant within a chain, keyed by `chain_key`: each
    chain's identity and, with `--include-uniprot`, the RCSB sequence-cluster
    identifiers of its polymer entity (`sequence_cluster_30` through
    `sequence_cluster_100`).
  - The structures table: scientific/experimental columns constant for the whole
    structure (resolution, r_work, bnet_pdb_redo, radiation/beam conditions,
    crystallographic quality, …), keyed by `pdb_id`.
  - The citations table: non-scientific publication/authorship columns, also
    constant per structure (title, journal, year, DOI, PubMed ID, citation
    authors, deposit authors), split out of the structures table so a researcher
    who only wants scientific data isn't carrying bibliographic bookkeeping
    alongside it. Raw items of the `_citation` categories (`_citation.*`,
    `_citation_author.*`, and any other category whose name starts with
    `_citation`) and of `_audit_author` land here too, while every other raw
    mmCIF item outside `_atom_site` goes to the structures table.

Use star_schema mode when you want to analyze residue- or structure-level trends
without carrying (and re-aggregating) duplicated values across every atom row. A
typical Bnet-vs-resolution scatter plot, for instance, only needs one row per
structure, not one per atom.

The star-schema keys are built from `pdb_id`, `chain_id`, `res_num`, `res_name`,
`model_num`, and the canonical mmCIF identity fields
(`_atom_site.label_asym_id`, `_atom_site.label_entity_id`,
`_atom_site.label_seq_id`, and `_atom_site.pdbx_PDB_ins_code`), so in
star_schema mode all nine are always retained, even when a profile or custom
column filter omits them. A polymer residue with both a label chain and a label
sequence number gets the `residue_key` `pdb_id + model + label_asym_id +
label_seq_id`. Every other residue gets `pdb_id + model + author chain + author
residue number + insertion code + residue name`, and that includes every
non-polymer residue (a ligand, an ion, a water) even when its label identity is
complete. Whether a residue belongs to a polymer is read from the entity
metadata the parser attaches (`_pdbai_polymer_type`, `_pdbai_entity_type`)
wherever it has any, and only otherwise from the chemical component type, the
presence of a label sequence number, and the ATOM or HETATM record. A missing
model reads as model 1. `chain_key` is `pdb_id + model + label_asym_id`, with
the author chain in place of a missing label chain. A row with neither a polymer
label position nor an author chain and residue number gets a `residue_key` of
its own, a deterministic surrogate built from the entry, the model, and the
row's position, and a row without any chain gets a surrogate `chain_key` the
same way. The keys do not depend on `residue_numbering`. For a structure
numbered by its label identifiers, the author parts of a key are read from
`_atom_site.auth_asym_id` and `_atom_site.auth_seq_id` alone, never from
`chain_id` and `res_num`, so a residue outside a polymer that has no author
residue number gets a surrogate key there too. One key shared by all such rows
would make one residue of unrelated atoms, and dropping them would lose atoms
the flat output keeps. A warning gives the number of rows without a residue
identity. SASA aggregation and star-schema normalization share this label-first
key, preventing distinct chain copies, insertion-coded residues, or structural
models from collapsing.

Each `residues`, `chains`, `structures`, and `citations` row holds, in each
column, the first value its atoms have present, so a value missing on the first
atom is taken from a later one. A column that should be constant within its
residue, chain, or structure but holds different present values there is logged
as a warning, since it needs a finer granularity, and the first present value is
kept.



## Engine tables

An engine can record relationships between atoms, such as a metal and its
ligands or the two cysteines of a disulfide. These records do not fit one row
per atom, so the engine declares its own tables (`TableSpec`) instead of adding
atom columns. None of the current built-in engines declares a table, but the
rules below apply to every engine table an engine adds.
Engine tables are
separate files in both output shapes, named after the primary table's file, as
in `pdb_atom_site_dataset_<table>.csv` next to `pdb_atom_site_dataset.csv`,
whether that is the flat file or the first of the five star-schema tables. An
engine can write tables and atom columns at the same time. Atoms then carry at
most a link column back to their site row, such as `<table>_site_id`.

Every engine table follows the same rules:

- **One row per record, fixed columns.** Every structure and technique gets the
  same columns in the declared order. Cells never hold lists. One-to-many data
  goes in a child table linked by a key, and the parent row carries fixed-size
  summaries of it (counts, minimum, mean, maximum).
- **Declared types.** Each column is `string`, `boolean` (nullable, as for
  [Boolean columns](#boolean-columns)), `Int64` (nullable integer), or
  `float64`. Units are part of the name (for example `distance_A` and
  `b_factor_A2`). Category columns list their allowed values, and a
  value outside the list fails the structure.
- **Missing, never zero.** A value that was not measured is empty in the CSV,
  not `0` or `False`.
- **Unique keys.** Each table declares key columns that are never empty and
  unique within the table.
- **Links.** Link columns join a row to the atom, residue, chain, or structure
  level of the dataset, or to another table of the same engine (for example,
  a bonds table's `site_id` → its sites table's `site_id`). Where names
  differ, the link names the target columns. A second atom whose key columns
  carry a prefix, such as `partner_`, joins through
  `partner_chain_id` → `chain_id`, `partner_ins_code` →
  `_atom_site.pdbx_PDB_ins_code`, `partner_altloc` → `_atom_site.label_alt_id`,
  and so on.
- **Header-only when empty.** A structure with no records still writes a
  header-only partial, so every table has a file and one header.
- **Not filtered.** Profiles and column filters apply only to the base tables.
  An engine table is written in full whenever its engine runs.

Atom sites are identified by a stable `site_id` built from the entry, model,
chain (`chain_id`), residue number (`res_num`), insertion code, atom name, and
alternate conformation, joined by colons, as in `1ABC:1:A:301::FE:` for the Fe
of a heme at residue 301 with no insertion code or alternate conformation.
Reruns give the same identifier. The chain and the residue number are the author
ones unless the structure's `residue_numbering` is `label`, as
[Identifiers](#identifiers) explains. Engines build it, and the matching
`pdb_id`, `model_num`, `chain_id`, `res_num`, `ins_code`, `res_name`,
`atom_name`, and `altloc` join columns, with `site_key_dataframe` and
`site_key_columns` in `pdbai.engines.core`, and `site_atom_link` declares the
join back to the atom level. A second atom in the same row uses a prefix, such
as `partner_site_id` and `partner_chain_id`.

`manifest.json` lists each engine table under `engine_tables` with its owning
engine, description, key, links, and every column's type, description, and
categories. `run_report.csv` adds a `<table>_row_count` column per structure,
and `run_summary.md` has an **Engine Tables** section plus a data dictionary
entry for each table.

When reading a table back, pass the declared types, which the manifest lists,
with `keep_default_na=False`, `na_values=[""]`, and
`float_precision="round_trip"`.
`keep_default_na=False` keeps codes such as the element `NA` from reading as
missing, and `float_precision="round_trip"` reads every number back as the value
that was written, where the default parser can change the last digit.


## Column profiles

Profiles are named column presets that filter the output to a curated subset.
Apply one with `--profile <id>` on the CLI or via the web UI dropdown. Selecting
a profile whose columns depend on an engine you haven't enabled turns that
engine on automatically (see `toggles` in the `/api/profiles` response). The
exception is an engine the run's technique rules out, which stays off and is
reported rather than failing the run (see [Technique
selection](cli.md#technique-selection)).

| ID | Name | Purpose | Requires engines |
|----|------|---------|-------------------|
| `all` | All Columns | No filtering. The base tables keep every column the run writes, namely each `_atom_site` item, each item of a one-row mmCIF category repeated on every atom, the parser's derived and metadata columns, and the columns of every engine the run enables. It switches no engine on. | none |
| `research_ready` | Research-Ready | The default for analysis. Keeps atom identity, B factors and anisotropic displacement parameters, experimental metadata, RABDAM's BDamage, packing density, and Bnet scores, solvent accessibility, the UniProt mapping, AlphaFold pLDDT, and wwPDB validation metrics. Of the raw mmCIF fields, only the identity items and the B factor remain. | RABDAM, SASA, AlphaFold, UniProt, validation |
| `radiation_sensitivity` | Radiation Sensitivity | Features for models of radiation-damage susceptibility. Keeps atom identity, B factors and anisotropic displacement parameters, RABDAM's BDamage, packing density, and Bnet scores, solvent accessibility, electron density at atoms, data-quality and wwPDB validation metrics, radiation dose and collection conditions, and the UniProt mapping. | RABDAM, SASA, density, UniProt, validation |
| `ml_minimal` | ML Minimal | The smallest complete feature set for training and inference. Keeps atom identity, B factors and their anisotropy summaries, resolution and R factors, the technique and radiation source, RABDAM's damage scores, relative solvent accessibility, and the real-space fit and Ramachandran class from wwPDB validation. Of the raw mmCIF fields, only the identity items and the B factor remain. | RABDAM, SASA, validation |
| `structure_quality` | Structure Quality | Experimental metadata and quality indicators for choosing structures before training. Keeps atom identity and B factors, resolution, R factors, data-collection statistics, radiation dose, crystallization conditions, the wwPDB clashscore and geometry outliers, RABDAM's structure-level Bnet and coverage, and the UniProt mapping. | RABDAM, UniProt, validation |

Profiles name RABDAM columns by wildcard patterns, such as `bnet*` and
`rabdam_alignment*`, so each profile resolves to whichever [column
set](scientific-methods.md#rabdam-column-names) the run produced. Under a
technique that rules out one of a profile's engines, the profile leaves that
engine off and reports it, and its column filter leaves that engine's columns
out, so they are not reported missing. A column filter you name yourself is
applied as given.

Custom filtering is available with `--column-filter-mode` and
`--column-filter-list`, or through the web column-filter tool. A `*` in a name
matches any run of characters, case-insensitively, and every other character in
such a pattern, brackets included, is literal. For example, `*factor*` matches
each column whose name contains `factor`. A name without `*` matches the column
spelled the same, or else the one column that matches it when case and
punctuation are ignored, so `_cell_length_a` finds `_cell.length_a`. A name that
could mean several columns is skipped and reported.

If an include filter matches nothing, PDB-AI reports the unmatched request. It
retains only the available structure identity column instead of returning the
unfiltered dataset.


## Dataset columns

Every row in the flat output CSV corresponds to one atom site (in star_schema
mode, the same columns are split across
`atoms`/`residues`/`chains`/`structures`/`citations` per [Dataset
shape](dataset-schema.md#dataset-shape-flat-vs-star_schema)). Raw mmCIF fields
(`_atom_site.*`) are also present unless filtered. Columns marked **(toggle)**
are entirely absent when their engine is disabled. When an enabled engine fails
for a structure, its columns are present and empty for that structure, and the
run report records the failure under the engine's own report fields.

### Boolean columns

Every column the schema declares boolean is a pandas **nullable boolean**,
carrying three distinct states: `True`, `False`, and missing (`<NA>`). In CSV
they serialize as `True`, `False`, and an empty field, never `1.0`/`0.0`.

The distinction is load-bearing. For a flag such as `bdamage_matches`, "the two
sources disagree" (`False`) and "there was nothing to compare" (`<NA>`) are
different scientific facts, and collapsing them loses information a researcher
needs. Numeric encoding is actively unsafe here, because `float('nan')` is
*truthy* in Python, so `if row.flag:` would treat an unmeasured comparison as
agreement. `pd.NA` raises instead of guessing.

This is enforced centrally once the annotation engines have run. A producer's
`True`, `False`, and missing values are kept, and the numbers 1 and 0 of any
numeric type (such as `1.0` and `0.0`) are converted to `True` and `False`. Any
other value, such as the text `"false"` or the number 2, fails that structure,
recorded as an output error in the run report, rather than being guessed at.
When reading a dataset back, `pd.read_csv(path)` infers a column with missing
values as object dtype. Pass `dtype={"is_hetatm": "boolean"}` (or
`.astype("boolean")`) to restore the three-state semantics.

### Identifiers

| Column | Source | Description |
|--------|--------|-------------|
| `pdb_id` | derived | Four-character PDB accession code (uppercase) for a downloaded entry. For an uploaded file, the entry identifier taken from the file name (uppercase, with a suffix such as `_2` when two files share a name). |
| `chain_id` | `_atom_site.auth_asym_id`, or `_atom_site.label_asym_id` | Author chain identifier. The label chain identifier for a structure whose `residue_numbering` is `label`. |
| `res_num` | `_atom_site.auth_seq_id`, or `_atom_site.label_seq_id` | Author residue sequence number (string). The label sequence number for a structure whose `residue_numbering` is `label`, where it is empty for a residue outside a polymer. The insertion code is the separate `_atom_site.pdbx_PDB_ins_code` column. |
| `res_name` | `_atom_site.auth_comp_id`, or `_atom_site.label_comp_id` | Three-letter residue name. The author name, or the label name on an atom whose author name is missing. |
| `residue_numbering` | derived | Which identifiers `chain_id` and `res_num` hold for the structure, `author` or `label`. A structure-level column, in every curated profile. The paragraph below the table gives the rule. |
| `atom_name` | `_atom_site.label_atom_id` | PDB standard atom name (CA, SG, OD1, …). |
| `x` / `y` / `z` | `_atom_site.Cartn_x/y/z` | Cartesian coordinates (Å). |
| `element` | `_atom_site.type_symbol` | Chemical element symbol. |
| `is_hetatm` | `_atom_site.group_PDB` | `True` for HETATM records (ligands, solvent, modified residues). |
| `occupancy` | `_atom_site.occupancy` | Crystallographic occupancy. One mean-highest-occupancy alternate ID is retained coherently per residue, together with shared blank-ID atoms. Ties go to the identifier that appears first, and an alternate atom without a finite occupancy (missing, non-numeric, or infinite) fails the parse. An occupancy written with a standard uncertainty, such as `0.50(2)`, is read without it. |
| `occupancy_all_conformers` | `_atom_site.occupancy` | The atom's occupancy summed over every alternate conformer the file models it in, the dropped ones included; equal to `occupancy` for an atom without alternates. A value below one marks an occupancy refined below full rather than one shared between conformers. |
| `conformer_count` | `_atom_site.label_alt_id` | Number of alternate conformers the file models the atom in, the dropped ones included; 1 for an atom without alternates. |
| `model_num` | `_atom_site.pdbx_PDB_model_num` | NMR model index (1 for X-ray structures). The star schema builds its keys from this column, and the raw `_atom_site.pdbx_PDB_model_num` it repeats stays an ordinary atom column and is in no curated profile. |
| `_atom_site.id` | mmCIF | Original atom-site serial number, from RCSB's mmCIF (not comparable to PDB-Redo's own numbering, see [Radiation damage (RABDAM)](scientific-methods.md#radiation-damage-rabdam)). |
| `_atom_site.label_asym_id` / `_atom_site.label_entity_id` / `_atom_site.label_seq_id` | mmCIF | Canonical label chain, entity, and polymer sequence identifiers. The shared residue key prefers the chain and sequence identifiers. |
| `_atom_site.pdbx_PDB_ins_code` | mmCIF | Insertion code used by the author-identity fallback key. `.`, `?`, blank, and null are normalized consistently. |

`chain_id` and `res_num` hold the author identifiers, since papers and the PDB
format number residues that way. A file that gives no author chain on any atom,
or no author residue number on any atom, is numbered by its label identifiers
instead. Both columns then come from `_atom_site.label_asym_id` and
`_atom_site.label_seq_id`, `residue_numbering` is `label`, and the run's
warnings name the structure. The two numberings are never mixed within a
structure, since a label number is unique only within a label chain. So an atom
that alone lacks an author number keeps an empty `res_num` in a structure
numbered by author. Under label numbering a residue outside a polymer, such as a
ligand or a water, has an empty `res_num`, since `_atom_site.label_seq_id`
numbers polymer residues only. `res_name` falls back on its own, atom by atom,
to `_atom_site.label_comp_id`. Of the 343 integrative entries with atoms that
were scanned, 93 give no author residue name, and 30 of those give no author
residue number either, 9A18, 9A03, 9A04, and 9A0B among them. A file written by
gemmi's default writer gives no author residue name.

### Atom classification

| Column | Description |
|--------|-------------|
| `is_backbone` | `True` for non-HETATM atoms named N, CA, C, O, or OXT. `False` for every other atom, including HETATM. The atom is not checked to be protein. |
| `is_sidechain` | `True` for non-HETATM atoms that are not backbone, which includes every nucleic-acid atom. `False` for backbone and HETATM. |

### B-factor columns

| Column | Description |
|--------|-------------|
| `_atom_site.B_iso_or_equiv` | Raw isotropic B-factor from RCSB's mmCIF (Å²). |
| `residue_mean_bfactor` | Mean B-factor of the residue's retained atoms that report one. Missing when none does, and for a row without a canonical residue identity. |

### Anisotropic displacement parameters

Present only for atoms with an ANISOU record, so a structure deposited without
anisotropic displacement parameters has these columns missing throughout. See
[Anisotropic displacement
parameters](scientific-methods.md#anisotropic-displacement-parameters) for the
method. `anisotropic_atom_fraction`, under [Refinement](#refinement), gives the
share of a structure's atoms that have such a record.

| Column | Description |
|--------|-------------|
| `u11` | U tensor element (Å²), the mean-square displacement along the Cartesian X axis of the coordinates. |
| `u22` | U tensor element (Å²), the mean-square displacement along Cartesian Y. |
| `u33` | U tensor element (Å²), the mean-square displacement along Cartesian Z. |
| `u12` | Off-diagonal element (Å²), the covariance of displacements along X and Y. |
| `u13` | Off-diagonal element (Å²), the covariance of displacements along X and Z. |
| `u23` | Off-diagonal element (Å²), the covariance of displacements along Y and Z. |
| `u_equivalent` | (U11 + U22 + U33) / 3, the isotropic-equivalent displacement (Å²). Equals B / 8π². |
| `b_anisotropy` | Eigenvalue ratio λ_max / λ_min of the U tensor, the reciprocal of Merritt's anisotropy. 1.0 means perfectly isotropic. Missing where any component is missing or the smallest eigenvalue is not positive. |

### Radiation and experimental conditions

Except for `experimental_methods`, `is_multi_method`, and `radiation_sources`,
these columns come from the RCSB entry record and are always missing for
uploads. Those three come from the RCSB entry record too, but when the record
lists no method or no source, the file's own `_exptl` or `_diffrn_source`
category is used instead. That is always the case for an upload. The same holds
for the experimental data quality columns other than `resolution`,
`unit_cell_volume`, and `solvent_content`.

| Column | Description |
|--------|-------------|
| `radiation_dose_MGy` | Absorbed X-ray dose in megagray (MGy), read from the first number followed by "MGy", "kGy", or "Gy" in the entry title, a value in kGy divided by 1,000 and one in Gy by 1,000,000. A number may be written in scientific notation, as in "7.88x10^3 Gy". A number that only continues a longer one, such as the 5 of "1,5 MGy" or the 6 of "10^6 Gy", is not taken, and neither is a dose rate, such as "40 kGy/s". Missing when the title states none, as for most entries, and always for uploads. |
| `is_high_radiation_dose` | `True` when `radiation_dose_MGy > 5.0`. |
| `title` | Title of the entry, `_struct.title`. It is where depositors state a dose, a time delay, or a resting or intermediate state, none of which the archive records in a field of its own. From the RCSB entry record, so missing for an uploaded structure. |
| `keywords` | Free-text keywords of the entry, `_struct_keywords.text`, as the depositors wrote them. From the RCSB entry record. |
| `mentions_time_resolved` | `True` when the title or the keywords name a time-resolved or pump-probe experiment, in any spelling of those two words. This is a text match, not a classification. A resting-state structure from a time-resolved series is `True` as well, and the title says which state the model shows. Missing when the record has neither a title nor keywords, and always for an uploaded structure. |
| `temperature_K` | Data-collection temperature (K). |
| `experimental_methods` | Every experimental method the entry reports (`_exptl.method`), each listed once, semicolon-separated in deposition order. A joint X-ray/neutron refinement lists both. From the RCSB entry record, or from the file's own `_exptl` category when the record lists none, as for every upload. Missing when neither reports one. Every entry in a run matches the run's `--technique`, so this column distinguishes joint refinements and companion methods within a cohort rather than techniques across one. |
| `is_multi_method` | Nullable boolean, `True` when the entry reports more than one method (its model was refined against several kinds of data), `False` for a single method, missing when none was reported. |
| `first_radiation_source` | The first reported radiation source (`_diffrn_source.source`), as written by the depositors without normalization (e.g. `SYNCHROTRON`, `ROTATING ANODE`, `FREE ELECTRON LASER`). |
| `radiation_sources` | Every radiation source the entry reports (`_diffrn_source.source`), each listed once, semicolon-separated in deposition order and normalized to the archive's canonical spellings (`SYNCHROTRON`, `FREE ELECTRON LASER`, `ROTATING ANODE`, `SEALED TUBE`, `LIQUID ANODE`, and for a joint X-ray/neutron entry its `NUCLEAR REACTOR` or `SPALLATION SOURCE`). XFEL entries are X-ray diffraction by method, so this column, with the matching `--xray-source` intake filter, is how to select them. Missing when no source is reported. |
| `synchrotron_site` | Synchrotron facility name (e.g. `Diamond`, `ESRF`, `APS`). |
| `beamline` | Synchrotron beamline (`_diffrn_source.pdbx_synchrotron_beamline`), falling back to the source's make and model (`_diffrn_source.type`) when no beamline is reported. |
| `radiation_wavelength` | Primary reported diffraction wavelength (Å), the first in the first source's `_diffrn_source.pdbx_wavelength_list`, or its `_diffrn_source.pdbx_wavelength` when the list is absent. Missing for a wavelength of zero or below. |
| `num_crystals` | Number of crystals used for the experiment. |
| `is_serial_crystallography` | `True` when the first diffraction record says that several crystals contributed to each data set, `_diffrn.pdbx_serial_crystal_experiment`; `False` when it says one crystal did. Missing when the record does not say, which many serial entries deposited before the field existed leave out, and always for an uploaded structure. |
| `sample_delivery_method` | How a serial-crystallography sample reached the beam, such as `injection` or `fixed target`, from the first `_pdbx_serial_crystallography_sample_delivery` record. Missing when none is reported, as for a single-crystal experiment. |
| `xfel_pulse_duration_fs` | Average X-ray pulse duration in femtoseconds, from the first `_pdbx_serial_crystallography_measurement` record. Missing when none is reported, which is the case for most free-electron laser entries. |
| `detector_type` | Detector technology (e.g. `PIXEL`, `CCD`). |
| `collection_year` | Year of diffraction data collection, from `_diffrn_detector.pdbx_collection_date` as reported to RCSB. Missing when no collection date was deposited. |
| `deposition_year` | Year the entry was deposited with RCSB. |
| `crystal_pH` | Crystallization pH. Rarely populated. |
| `crystallization_method` | Crystallization method description. |

### Experimental data quality

| Column | Description |
|--------|-------------|
| `resolution` | Resolution of the refined model (Å), read from `_refine.ls_d_res_high`, else `_refine_hist.d_res_high`, else `_reflns.d_resolution_high`, else the smallest `_reflns_shell.d_res_high`, else the RCSB record's value. A resolution of zero or less counts as not reported, so the next source is tried. Missing when none reports one. A joint X-ray/neutron entry has a refinement record for each method, and this is the first one's, of whichever method the file lists first. `resolution_xray` and `resolution_neutron` hold each method's own. |
| `resolution_xray` | Resolution (Å) of the refinement against the X-ray data, the `_refine.ls_d_res_high` of the record that `_refine.pdbx_refine_id` names `X-RAY DIFFRACTION`. In a joint X-ray/neutron entry it is the X-ray data's resolution whichever record comes first. A single refinement record that names no method is taken to belong to the entry's method when the entry reports exactly one. Missing when the file has no such record, as for a neutron-only entry. |
| `resolution_neutron` | Resolution (Å) of the refinement against the neutron data, read the same way from the record that names `NEUTRON DIFFRACTION`. Missing when the file has no such record, as for an X-ray-only entry. |
| `r_work_xray`, `r_free_xray` | R-work and R-free of the refinement against the X-ray data, from the `_refine` record that `_refine.pdbx_refine_id` names `X-RAY DIFFRACTION`, whichever record comes first. `r_work` and `r_free` hold the first record's. Missing when the file has no such record or the record states none. |
| `r_work_neutron`, `r_free_neutron` | The same for the refinement against the neutron data, from the record that names `NEUTRON DIFFRACTION`. |
| `r_work` | Model R-factor on the working set. |
| `r_free` | Model R-factor on the free set (R-free). |
| `reflns_completeness` | Data completeness (% of unique reflections). |
| `cc_half` | CC½ split-half data correlation. |
| `i_over_sigma` | Mean I/σ(I). |
| `wilson_b` | Wilson B-factor (Å²) estimated from the diffraction data falloff. |
| `multiplicity` | Data redundancy (average number of measurements per unique reflection). |
| `space_group` | Hermann-Mauguin space group symbol. |
| `unit_cell_volume` | Unit-cell volume (Å³), from the file's `_cell` lengths and angles. Missing when the file reports no complete cell, when the cell cannot close (a length that is not positive, an angle outside (0, 180), angles summing to 360° or more, or one angle at least the sum of the other two), and for the wwPDB placeholder cell of 1 Å edges and right angles written for NMR and electron-microscopy entries. |
| `solvent_content` | Crystal solvent-volume fraction (0–1). Read from the deposited `_exptl_crystal.density_percent_sol`. When that is absent, it is converted from the deposited Matthews coefficient (1 − 1.23/Vm). When both are absent, it is estimated from unit-cell volume, the space-group symmetry-operation count, and 110 Da per polymer residue of the first model, HETATM-flagged modified residues included. Each source is used only when it gives a fraction inside (0, 1), otherwise the next is tried. Missing when none does. |

### Refinement

How the entry was refined. These columns are read from the structure file, so an
uploaded structure has them too. A B-factor depends on the program that refined
it, on whether TLS groups were used, and on whether it was refined as one
isotropic value or as an anisotropic tensor, so a comparison of B-factors
between entries can be split by these columns. See [How an entry was
refined](scientific-methods.md#how-an-entry-was-refined) for the rules and for
how the columns read together.

| Column | Description |
|--------|-------------|
| `refinement_program` | Name of the program the model was refined with, the `_software.name` of the file's first `_software` record classified as `refinement`, as the file writes it (e.g. `PHENIX`, `REFMAC`, `BUSTER`). An entry that names several refinement programs has its first here and all of them in `refinement_programs`. Missing when the file names none. |
| `refinement_program_version` | Version of `refinement_program`, the `_software.version` of the same record, as the depositors wrote it and in no fixed format (e.g. `1.19.2_4158`, `5.8.0238`, `(phenix.refine: 1.8.1_1168)`). Missing when that record states none. The versions of an entry's other refinement programs are not kept. |
| `refinement_programs` | Every refinement program the entry names, each listed once, semicolon-separated in file order (e.g. `REFMAC; PHENIX`). It equals `refinement_program` for an entry that names one, so comparing the two columns finds the entries that name several. Names are as the file writes them, so `SHELXL` and `SHELXL-97`, or `BUSTER` and `BUSTER-TNT`, are listed apart. The file does not say which of several programs wrote the deposited B-factors. Missing when the file names none. |
| `tls_group_count` | Number of TLS groups the refinement used, the number of `_pdbx_refine_tls` records in the file, a whole number. `0` for a file that reports a refinement (a `_refine` category) and no TLS group. Missing for a file that reports no refinement, which says nothing of TLS. |
| `tls_residual_adp_flag` | The archive's note on the B-factors of an entry refined with TLS, the first `_refine.pdbx_TLS_residual_ADP_flag` the file states, as written. `LIKELY RESIDUAL` marks an entry whose `_atom_site.B_iso_or_equiv` probably leaves out the TLS contribution, so its B-factors do not compare with those of other entries. Missing when the file states none, as for most entries with TLS groups, so missing does not say the B-factors are whole. |
| `anisotropic_atom_fraction` | Fraction (0–1) of the file's atom sites that have an anisotropic displacement record (`_atom_site_anisotrop`), over every alternate conformer and model, with atoms of element H, D, or T left out. `0` for a file without such records, which is how an isotropic refinement is deposited. A refinement with TLS groups can write a record for every atom of a group, so read it beside `tls_group_count`. Missing only for a file without a non-hydrogen atom. |

### Citation and publication metadata

Present in flat mode, and in its own `citations` table in star_schema mode.
Sourced from RCSB's `rcsb_primary_citation` and `audit_author` fields. Many
depositions (unpublished/in-press structures) have no primary citation yet. In
that case the citation-specific columns are `NaN` rather than the row being
dropped.

| Column | Description |
|--------|-------------|
| `citation_title` | Title of the primary publication. |
| `citation_journal_abbrev` | Abbreviated journal name. |
| `citation_year` | Publication year. |
| `citation_doi` | DOI of the primary publication. |
| `citation_pubmed_id` | PubMed ID of the primary publication. Missing for an ID of zero or below, such as the -1 the archive writes for a citation PubMed does not index. |
| `citation_authors` | Semicolon-joined list of citation authors (`"Last, F."` form), in RCSB's given order. |
| `deposit_authors` | Semicolon-joined list of structure depositors (`audit_author`), ordered by `pdbx_ordinal`. Distinct from `citation_authors`, since depositors and paper authors don't always match. |

### Radiation damage (RABDAM)

Requires `--include-rabdam` (off by default). Column names carry the suffix of
the structure RABDAM scored (`_pdb_redo`, `_rcsb`, or none for uploads), chosen
by `--rabdam-source`. See [RABDAM column
names](scientific-methods.md#rabdam-column-names), which also describes the
structure-level QA columns (`bdamage_coverage_pdb_redo` and the alignment
counts). The table lists the downloaded-build names.

| Column | Description |
|--------|-------------|
| `packing_density_pdb_redo` / `packing_density_rcsb` | Number of protein heavy atoms within 7 Å (packing density index), scored on the named source. |
| `bdamage_pdb_redo` / `bdamage_rcsb` | BDamage score, the atom's B / sliding-window mean B of atoms with similar packing density. Values >1 indicate B-factor inflation relative to packing context, the primary radiation-damage signal. |
| `bnet_pdb_redo` / `bnet_rcsb` | Bnet score, the area of the kernel density estimate of Asp and Glu carboxyl-oxygen BDamage above the median BDamage of all scored atoms, divided by the area below it. Always positive, and values above 1 indicate net elevation of those atoms. Structure-level metric broadcast to all atoms. |
| `bnet_percentile_pdb_redo` / `bnet_percentile_rcsb` | Percentile rank of Bnet from rabdam-custom's bundled 124,948-entry, resolution-matched reference database, calibrated against PDB-Redo B-factors. The `_rcsb` percentile is still looked up against this same table, since there is no RCSB-calibrated equivalent. |
| `bnet_site_count_pdb_redo` / `bnet_site_count_rcsb` | Number of Asp OD1/OD2 and Glu OE1/OE2 atoms whose BDamage values fed the Bnet estimate. Bnet needs only two such atoms to compute, is biased upward and noisy when the count is small, and the reference database was not filtered on it either. So a tiny peptide can legitimately score a percentile of 100. Filter on this column in the analysis before using Bnet or its percentile as a target. |
| `bdamage_scored_atom_count_pdb_redo` / `bdamage_scored_atom_count_rcsb` | Number of atoms rabdam-custom scored on that source. |
| `bnet_percentile_resolution_in_range_pdb_redo` / `bnet_percentile_resolution_in_range_rcsb` | Nullable boolean. `True` when the structure's resolution lies inside the resolution window of the reference structures its Bnet was ranked against. `False` when it lies outside, so the percentile extrapolates from the nearest edge of the 0.59–3.5 Å reference (typically entries worse than 3.5 Å). Missing when no percentile was computed. |
| `bdamage_matches`, `packing_density_matches`, `bnet_matches`, `bnet_percentile_matches` | Only present with `--rabdam-source both`. `True`/`False` for whether the `_pdb_redo` and `_rcsb` values agree within an absolute tolerance of 1e-3. Missing (`<NA>`) when either side has no value for that atom or structure (e.g. the composite-key alignment didn't match it), never silently `False`. |

Residue- or site-level summaries of BDamage are left to analysis, since every
atom row carries its BDamage and the columns that identify its residue. For
example, the mean BDamage over a residue's commonly cited damage-site atoms:

```python
import pandas as pd

atoms = pd.read_csv("pdb_atom_site_dataset.csv")
sites = {
    ("ASP", "OD1"),
    ("ASP", "OD2"),
    ("GLU", "OE1"),
    ("GLU", "OE2"),
    ("CYS", "SG"),
    ("MET", "SD"),
    ("TYR", "OH"),
}
is_site = [
    (res, atom) in sites for res, atom in zip(atoms["res_name"], atoms["atom_name"])
]
residue = ["pdb_id", "model_num", "chain_id", "res_num", "_atom_site.pdbx_PDB_ins_code"]
site_means = atoms[is_site].groupby(residue, dropna=False)["bdamage_pdb_redo"].mean()
```

Bnet, by contrast, is defined over Asp OD1/OD2 and Glu OE1/OE2 only, the
carboxyl oxygens whose decarboxylation rabdam-custom's reference database is
calibrated on.

`_rcsb`-suffixed columns are merged directly onto the atom frame by
`_atom_site.id`. `_pdb_redo`-suffixed columns are merged by author
chain/residue/atom identity, since RCSB's and PDB-Redo's mmCIF are independently
numbered. See [Cross-file atom
alignment](scientific-methods.md#cross-file-atom-alignment).

### PDB-Redo comparison

Requires `--include-pdb-redo` (off by default).

| Column | Description |
|--------|-------------|
| `pdb_redo_x`, `pdb_redo_y`, `pdb_redo_z` | PDB-Redo's actual re-refined Cartesian coordinates (Å), not a delta. Present when the PDB-Redo comparison engine is enabled. |
| `pdb_redo_x_delta`, `pdb_redo_y_delta`, `pdb_redo_z_delta` | PDB-Redo coordinate minus RCSB coordinate (Å), per axis. |
| `pdb_redo_bfactor_delta` | PDB-Redo B-factor minus RCSB B-factor. |
| `pdb_redo_occupancy_delta` | PDB-Redo occupancy minus RCSB occupancy. |
| `pdb_redo_u_equivalent_delta` | PDB-Redo ANISOU U-equivalent minus RCSB U-equivalent. |
| `pdb_redo_revised` | `True` if any of the deltas above exceeds its tolerance (1e-3) for that atom. |
| `pdb_redo_alignment_match_rate` | Structure-level fraction of protein heavy RCSB atoms matched one-to-one to PDB-Redo by model-aware author residue identity (author chain, author sequence number, insertion code, residue name) and atom name. An entry with no protein heavy atoms is measured over all its heavy atoms instead, and one with no heavy atoms over all its atoms. Label identifiers are not used because PDB-Redo renumbers them. Reported as 0 when the mis-pairing guard rejected the alignment because matched atoms were a median of more than 1 Å apart. |
| `pdb_redo_alignment_rcsb_ambiguous_atom_count` / `pdb_redo_alignment_pdb_redo_ambiguous_atom_count` | Structure-level counts of atoms excluded on each side because their identity was duplicated and could not be paired defensibly. |

### Solvent accessibility (SASA)

Requires `--include-sasa` (off by default). The method is described under
[Solvent accessibility](scientific-methods.md#solvent-accessibility-sasa).

| Column | Description |
|--------|-------------|
| `sasa_component_category` | Biological/mmCIF-derived atom category (`protein_polymer`, `nonprotein_polymer`, `water`, `ligand_or_cofactor`, `metal`, `experimental_additive`, or `unknown_nonpolymer`). Polymer identity is independent of `ATOM`/`HETATM`. |
| `atom_sasa_protein` / `atom_sasa_bound` | Per-protein-polymer-heavy-atom SASA (Å²). The protein context has protein-polymer occluders only. The bound context additionally has supported non-protein polymers and configured ligand/cofactor/metal/unknown-nonpolymer occluders. Models are calculated independently. |
| `residue_sasa_protein` / `residue_sasa_bound` | Canonical protein-residue sums of the corresponding atom areas. A partial calculable sum is retained and identified by `residue_sasa_complete`. |
| `rasa_protein` / `rasa_bound` | Corresponding residue SASA divided by the Tien theoretical ALLOWED maximum using the paired DSSP-compatible method. Emitted only for complete, non-terminal, occupancy-unambiguous supported residues. Values above 1 are retained. |
| `is_buried_protein` / `is_buried_bound` | Nullable booleans, `True` when the corresponding RASA is below 0.25. |
| `is_exposed_protein` / `is_exposed_bound` | Nullable booleans, `True` when the corresponding RASA is at least 0.40. |
| `nonprotein_occluded_sasa` / `nonprotein_occluded_fraction` | Protein-minus-bound residue area (Å²), and that difference divided by `residue_sasa_protein`. The fraction is missing for a zero/unavailable denominator. This is the area the residue loses to the bound context's non-protein occluders. |
| `has_bound_nonprotein` | Nullable residue boolean, `True` when `nonprotein_occluded_sasa` is above zero. |
| `residue_sasa_complete` / `rasa_eligible` / `rasa_ineligible_reason` | Residue completeness, normalization eligibility, and semicolon-separated structured reason(s) when RASA is withheld. |
| `is_terminal_residue` / `is_occupancy_ambiguous` | Nullable residue-quality flags. Either condition makes Tien-normalized RASA unavailable while calculable raw SASA is retained. |
| `has_experimental_additive` / `has_unknown_nonpolymer` | Structure-level component-presence flags. Additives are excluded by default. Unknown non-polymers are included by default but surfaced as ambiguous. |
| `bound_context_complete` / `bound_context_ambiguous` | Structure-wide worst-case flags, for whether every requested bound occluder in every retained model was usable, and whether an included unknown/partial-occupancy component makes the biological interpretation ambiguous. |
| `sasa_status` / `sasa_ok` | Structure status (`ok`, `degraded`, `failed`) and nullable success boolean. Only *degrading* warnings lower the status (see [Warnings](scientific-methods.md#warnings)). Absent when SASA is disabled, like every other SASA column. |
| `sasa_warning_codes` / `sasa_warning_messages` | Stable diagnostic codes and human-readable messages. Every warning is listed here, advisory ones included, regardless of status. |
| `sasa_normalization` / `sasa_protein_radius_scheme` / `sasa_nonprotein_radius_scheme` | Scientific method provenance (`Tien2013_ALLOWED_DSSP`, DSSP protein-role radii, and Alvarez2013 non-protein element radii). |
| `sasa_probe_radius` / `sasa_n_sphere_points` / `sasa_accuracy_mode` | Numerical configuration recorded with every result. |
| `sasa_structure_context` | Coordinate-set context (`asymmetric_unit` for standard downloads, `as_provided` for uploads). No transforms are silently generated. |
| `sasa_include_experimental_additives` / `sasa_include_unknown_nonpolymers` | Recorded bound-context policy flags so differently configured datasets are not mixed silently. |

### Electron density at atoms

Requires `--include-density` (off by default, X-ray diffraction downloads only).
The engine fetches an entry's map coefficients, makes the 2mFo−DFc and mFo−DFc
maps, and reads each at every atom. `--density-source` chooses the maps:
PDB-Redo's, calculated about its re-refined model (`pdb_redo`, the default), the
wwPDB validation maps, calculated about the deposited model (`wwpdb`), or
`both`. A run writes the columns of each source it reads. From PDB-Redo's file
the engine also makes the map of the model's calculated structure factors, and
writes the calculated density and the difference density at every atom on one
scale. The wwPDB validation files hold no calculated structure factors, so
their columns have neither. Each source also gives the gradient of its
difference map at the atoms of the model it was calculated about, and
PDB-Redo's maps the shift of each atom that the gradient implies. How the maps are made and read is described under
[Electron density at atoms](scientific-methods.md#electron-density-at-atoms).
The values are measurements and are not thresholded.

The columns of PDB-Redo's maps:

| Column | Description |
|--------|-------------|
| `density_2fofc_sigma` | Numeric, atom level. Density of PDB-Redo's 2mFo−DFc map at the atom's deposited position, in units of the map's root-mean-square density over the unit cell. It depends on resolution, so compare it within bins of resolution. |
| `density_fofc_sigma` | Numeric, atom level. Density of PDB-Redo's mFo−DFc difference map at the atom's deposited position, in the same units. Negative where PDB-Redo's model places more density than the data support. |
| `density_2fofc_sigma_pdb_redo`, `density_fofc_sigma_pdb_redo` | Numeric, atom level. The same two maps at the position PDB-Redo's model gives the atom, which is the model the maps were calculated about. Missing for an atom with no partner in PDB-Redo's model. |
| `density_fc_eA3` | Numeric, atom level. Density of the map of PDB-Redo's calculated structure factors (`FC_ALL`, `PHIC_ALL`) at the atom's deposited position, on the scale of the map coefficients (nominally e/Å³), less the map's mean. It is what PDB-Redo's whole model places at the point at the resolution of the data. It is the density above the mean of the unit cell, not above zero, it includes what neighbouring atoms add, and it can be near zero or negative (see [Calculated density](scientific-methods.md#calculated-density)). Missing when the reflection file holds no usable calculated structure factors, which `density_message` in the run report then says. |
| `density_fofc_eA3` | Numeric, atom level. Difference density at the atom's deposited position on the scale of `density_fc_eA3`: the 2mFo−DFc density less the calculated density at the point, which is the density of a map of 2(mFo−DFc). `density_fofc_eA3 / density_fc_eA3` is the share of the calculated density that the data do not support, negative where the model places too much. The engine writes no such ratio, since it is unstable where the calculated density is small. Equal to `density_fofc_sigma × density_fofc_rms` for an untwinned entry, and twice that for a twinned one. Missing when `density_fc_eA3` is. |
| `density_fc_eA3_pdb_redo`, `density_fofc_eA3_pdb_redo` | Numeric, atom level. The same two values at the position PDB-Redo's model gives the atom. Their ratio is the share of the density PDB-Redo's model places at its own atom that the data do not support, which, unlike a value in sigma, does not grow with the number of electrons of the element. Missing for an atom with no partner in PDB-Redo's model. |
| `density_fofc_gradient_x_sigma_per_A_pdb_redo`, `_y_`, `_z_` | Numeric, atom level. Gradient of PDB-Redo's mFo−DFc map along the x, y, and z axes of the model's frame at the position PDB-Redo's model gives the atom, in sigma per Å. The vector points where the difference density rises, which is where the data would move the atom, so projected on a bond it says whether the data want the bond longer or shorter. Like a value in sigma it grows with the atom's electrons, so compare it within one element (see [Gradient and shift](scientific-methods.md#gradient-and-shift)). Missing for an atom with no partner in PDB-Redo's model. |
| `density_fofc_shift_x_A_pdb_redo`, `_y_`, `_z_` | Numeric, atom level. Shift of the atom in Å, along the same axes, that would level the slope of the difference density to first order: minus the inverse curvature of the calculated map times the gradient of the 2(mFo−DFc) density, on one scale. Overstated along a bond at coarse resolution, about twice at 2 Å for a disulfide sulfur, and large where the curvature is nearly flat, so summarize it by medians. Missing where the calculated map does not curve downward in every direction, as at most carbonyl carbons, and when `density_fc_eA3_pdb_redo` is. |
| `density_status` | Text, atom level. `ok`, `no_pdb_redo_atom` when only the deposited position was read, `no_coordinates` when the deposited position is not a finite point, or the entry's `density_map_status` when the entry has no maps. |
| `density_2fofc_rms`, `density_fofc_rms` | Numeric, structure level. Each map's root-mean-square deviation over the unit cell, on the scale of the map coefficients (nominally e/Å³). Multiply a sigma value by it to put the value back on that scale. The difference coefficients of a twinned refinement are half those of an untwinned one, so use `density_fofc_eA3` to compare a difference density between entries. |
| `density_map_resolution_A` | Numeric, structure level. High-resolution limit of the reflections the maps were made from. PDB-Redo may use more of the data than the depositor did, so this can be finer than `resolution`. |
| `density_map_sample_rate` | Numeric, structure level. Grid points per high-resolution limit, 4 unless the unit cell is too large, and never below 2.5. |
| `density_map_status` | Text, structure level. `ok`, or why the entry has no values: `no_pdb_redo_entry`, `no_map_coefficients`, `map_too_large`, `cell_mismatch`, or `models_not_paired`. |

The columns of the wwPDB validation maps end in `_wwpdb`. There is no
`density_fc_eA3_wwpdb` and no `density_fofc_eA3_wwpdb`, since those files hold
the two sets of map coefficients and no calculated structure factors:

| Column | Description |
|--------|-------------|
| `density_2fofc_sigma_wwpdb`, `density_fofc_sigma_wwpdb` | Numeric, atom level. Density of the wwPDB validation 2mFo−DFc and mFo−DFc maps at the atom's deposited position, in units of each map's root-mean-square density. The maps were calculated about the deposited model, the one the atom table holds. |
| `density_fofc_gradient_x_sigma_per_A_wwpdb`, `_y_`, `_z_` | Numeric, atom level. Gradient of the wwPDB validation mFo−DFc map at the atom's deposited position, in sigma per Å, read as the PDB-Redo gradient is. The map was calculated about the deposited model, so the vector points where the data would move the depositor's atom. There is no shift, since the files hold no calculated structure factors. |
| `density_status_wwpdb` | Text, atom level. `ok`, `no_coordinates`, or the entry's `density_map_status_wwpdb` when the entry has no such maps. |
| `density_2fofc_rms_wwpdb`, `density_fofc_rms_wwpdb` | Numeric, structure level. Each map's root-mean-square deviation over the unit cell, on the scale of the map coefficients. |
| `density_map_resolution_A_wwpdb`, `density_map_sample_rate_wwpdb` | Numeric, structure level. The high-resolution limit of the map files and the rate they were sampled at. |
| `density_map_status_wwpdb` | Text, structure level. `ok`, or why the entry has no values: `no_validation_maps`, `no_map_coefficients`, `map_too_large`, or `cell_mismatch`. |

The atom table keeps one conformer of each residue, so the values describe that
conformer. An atom that has moved between the two models is told apart by the
[PDB-Redo comparison](#pdb-redo-comparison) columns, which
`--include-pdb-redo` adds.

### wwPDB validation

Requires `--include-validation` (off by default). Fetched from the RCSB wwPDB
validation XML report and merged at the residue level. Provides direct
crystallographic evidence of model quality and, for susceptible residues, of
radiation damage (a poorly fitting Asp/Glu carboxyl is a hallmark of
decarboxylation). The numeric columns are float64 and the class columns
(`rama_class`, `rotamer_class`) text. A value the report leaves out, or one of
the wrong type, is missing.

| Column | Description |
|--------|-------------|
| `eds_rscc` | Real-Space Correlation Coefficient (0–1), the correlation between the model's calculated electron density and the experimental map within the residue envelope. Values < 0.9 indicate poor fit, and < 0.7 strongly suggests a problem. Per-residue, broadcast to all atoms in the residue. |
| `eds_rsr` | Real-Space R-factor (0–1), the average absolute difference between observed and calculated density, normalised by their sum. Lower is better, and outliers are typically > 0.4. Per-residue. |
| `rama_class` | Ramachandran plot region for this residue, one of `Favored`, `Allowed`, or `OUTLIER`. Missing for any residue the report does not classify, such as ligands, waters, and chain termini. A modified polymer residue written as HETATM, such as MSE, carries the report's class like any other. |
| `rotamer_class` | Side-chain rotamer assignment as written in the validation report's `rota` attribute. It is the MolProbity rotamer name for residues that fit a known rotamer (for example `mt`, `t80`, `m-80`, `tp`, or `Cg_endo`/`Cg_exo` for proline), or `OUTLIER` for a side chain that fits none. Roughly 230 distinct names occur across the library. The column does not collapse them to favored/allowed classes, so treat any value other than `OUTLIER` as an acceptable rotamer. NaN for residues with no rotatable sidechain (Ala, Gly) and for ligands and waters. A modified polymer residue such as MSE carries its rotamer. |
| `clashscore` | Structure-level all-atom clashscore, the number of serious steric clashes per 1,000 atoms. Broadcast to all atoms. Lower is better. Below 10 is typical for good structures, and above 40 is a warning sign. |
| `percent_rama_outliers` | Percentage of residues in unfavoured Ramachandran regions for the whole structure. Broadcast to all atoms. |
| `percent_rotamer_outliers` | Percentage of residues with outlier rotamer conformations for the whole structure. Broadcast to all atoms. |

### Protein identity (UniProt)

Requires `--include-uniprot` (off by default).

| Column | Description |
|--------|-------------|
| `uniprot_id` | UniProt accession aligned to this residue. Every RCSB polymer entity and aligned segment is considered, so multi-entity structures and chimeric chains can carry different accessions. If RCSB supplies only an entity-level identifier with no alignment, that accession is retained as an entity-wide fallback. AlphaFold Database also computes this mapping internally. Set `--include-uniprot` if you want the column retained when AlphaFold output is disabled. |
| `sequence_cluster_30` … `sequence_cluster_100` | Identifier of the RCSB sequence cluster at 30, 50, 70, 90, 95, and 100 % identity that the chain's polymer entity belongs to, from the entity's `rcsb_polymer_entity_group_membership`, such as `391_95`. Chains with the same identifier have sequences at least that similar, so one column splits a dataset by protein family and another removes near-duplicates. Written whenever the UniProt mapping is, for every polymer entity, protein or not. Missing for an entity RCSB places in no cluster, and always for an uploaded structure. |

### AlphaFold Database

Requires `--include-alphafold` (off by default). Model availability, provenance,
pLDDT coverage, and per-residue confidence from the EBI AlphaFold Database.
RCSB's entity-to-UniProt aligned regions are used rather than assuming PDB
residue numbers equal UniProt positions. Multi-entity structures, alignment
offsets/gaps, chimeric entities, modified polymer residues, and fragmented AFDB
predictions are handled explicitly. This is a general structural annotation
engine. It is not enabled by the radiation-focused presets.

| Column | Description |
|--------|-------------|
| `afdb_status` | Explicit AFDB state for the residue's mapped accession, one of `available`, `partial`, `no_model_available` (no usable model, and no single failure status shared by every model), `no_uniprot_mapping`, `model_not_found`, `lookup_failed`, `download_failed`, `parse_failed`, or `not_available_for_upload`. `partial` always means at least one usable model was obtained. |
| `afdb_model_id` | Exact versioned AFDB ModelCIF or fragment identifier that supplied this residue's pLDDT (for example, the CIF filename without `.cif`). Null when no model supplied a score. |
| `afdb_plddt_coverage` | Fraction of unique observed protein residues in the entity that received a pLDDT value. A residue whose amino acid differs from the model's counts as unscored. Repeated on that entity's atom rows. |
| `afdb_plddt` | Predicted local distance difference test score (0–100). Higher means greater local confidence. Null when no unambiguous mapped score is available, and when the deposited amino acid differs from the model's at the mapped position (a modified residue counts as its parent, so MSE matches MET). |
| `afdb_residue_has_plddt` | Nullable boolean, `True` when pLDDT was attached, `False` for an eligible observed protein residue without a score, and null where the engine is not applicable. |
