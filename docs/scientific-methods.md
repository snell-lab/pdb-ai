# Scientific Methods

This page records the algorithms, assumptions, thresholds, and references behind
PDB-AI's scientific fields.

[Project README](../README.md)

## Anisotropic displacement parameters

Many structures carry anisotropic displacement parameters (ADPs), stored in the
mmCIF `_atom_site_anisotrop` block (equivalent to PDB ANISOU records). They
describe the full 3D thermal motion ellipsoid of each atom rather than just the
isotropic average B-factor. High-resolution structures (typically < 1.5 Å) are
often refined with individual anisotropic ADPs. Structures refined with TLS at
ordinary resolutions can also carry ANISOU records, whose anisotropy then
describes the rigid-group TLS model rather than each atom. `tls_group_count`
and `anisotropic_atom_fraction` tell the two apart as far as the file allows
(see [How an entry was refined](#how-an-entry-was-refined)). Always on,
independent of every toggle above.

PDB-AI extracts the six independent tensor components for atoms with ANISOU
records. It then computes two summary metrics:

- **`u_equivalent`** = (U11 + U22 + U33) / 3, the isotropic-equivalent
  mean-square displacement in Å². Related to B-factor by B = 8π²·U_eq.
- **`b_anisotropy`** = λ_max / λ_min, the ratio of the largest to smallest
  eigenvalue of the U tensor, computed via `np.linalg.eigvalsh` on batched (N,
  3, 3) matrices. It is the reciprocal of Merritt's (1999) anisotropy A = λ_min
  / λ_max. A value of 1.0 indicates perfectly isotropic motion, and larger
  values indicate a more elongated ellipsoid. It is NaN when the smallest
  eigenvalue is not positive.

For atoms without an ANISOU record these columns are NaN, so a structure
deposited without ADPs has them NaN throughout.

The columns are listed under [Anisotropic displacement
parameters](dataset-schema.md#anisotropic-displacement-parameters) in the
dataset schema.

## How an entry was refined

A B-factor depends on how it was refined. Each program restrains the B-factors
of bonded atoms in its own way. A TLS model moves part of every atom's
displacement into the motion of a rigid group. An anisotropic refinement gives
an atom six displacement parameters where an isotropic one gives it one. A
measure built on B-factors, such as the B-factor of a side-chain group relative
to the atom it hangs from, can therefore differ between two entries because
they were refined differently. Six structure-level columns record these
choices, so that such a measure can be compared among entries refined alike.
Always on, independent of every toggle above.

All six are read from the structure file. The RCSB entry record lists the
software as the file does, but it holds neither the TLS groups nor the
anisotropic records, and an uploaded structure has no record. Reading the file
gives the columns one source and gives an upload the same columns. The counts
below are of the 17,725 structure files cached for the radiation-damage study,
each of which has a `_refine` category. For every one of them the programs read
from the file are those the record lists.

**Programs.** `_software` has one record per program, classified by the step
it was used for. The refinement programs are the records classified as
`refinement`. `refinement_program` is the name in the first of them, and
`refinement_program_version` is the version that record states, as the
depositors wrote it. 17,682 entries (99.8 %) name a refinement program, and
15,096 (85.2 %) state its version. The commonest are `PHENIX` (8,035), `REFMAC`
(5,734), `BUSTER` (1,155), `CNS` (1,110), `X-PLOR` (688), and `SHELXL-97`
(514).

An entry can name several refinement programs, as one refined with REFMAC and
then with PHENIX does, and the file does not say which of them wrote the
deposited B-factors. `refinement_programs` therefore lists every name once, in
file order, joined with `; `, as `experimental_methods` is. 523 entries (3.0 %)
name more than one, and for them `refinement_program` is only the first listed.
They are the entries where the two columns differ.

A name is kept as the file writes it, with its whitespace collapsed. The
archive writes one family of programs under several names, such as `SHELXL`,
`SHELXL-97`, and `SHELX`, or `BUSTER` and `BUSTER-TNT`. They are not merged,
since some of those names state a version, and which of them count as one
program is for a study to decide. Two entries give the name `NONE`, which is
kept as well.

**TLS groups.** `_pdbx_refine_tls` has one record per TLS group, and
`tls_group_count` is the number of its records. The archive writes the TLS
groups of a refinement into that category, so a file that has a `_refine`
category and no TLS group gives 0. A file without a `_refine` category reports
no refinement, which says nothing of TLS, so its count is missing. 3,932 entries
(22.2 %) list at least one group. A count of 0 means that none was deposited.
Three entries, 2F3C, 3D4S, and 5ANU, say in their refinement details that TLS
was used and list no group.

**Residual B-factors.** A refinement with TLS can write, as each atom's
B-factor, only what is left once the motion of its group is taken out (Winn et
al., 2001). Such a residual B-factor is smaller than the whole one by an amount
that varies over the group, so it does not compare with the B-factors of other
entries, and a ratio of two of them is not the ratio of the whole ones. The
archive marks the entries it judged to hold such values in
`_refine.pdbx_TLS_residual_ADP_flag`, and `tls_residual_adp_flag` holds that
text, `LIKELY RESIDUAL` for all 167 entries that state one. Each of them lists
TLS groups. The archive marks no entry as holding whole B-factors, so a missing
flag does not say that it does.

**Anisotropic records.** `anisotropic_atom_fraction` is the share of the
file's atom sites that have a record in `_atom_site_anisotrop`. Atoms of
element H, D, or T are left out of both counts, since a refinement that gives every
other atom a tensor often still lets its hydrogens ride on their parents with
one isotropic value. The atom sites are those of the file, every alternate
conformer and every model, so the fraction does not depend on the conformer the
atom table keeps and can differ slightly from the share of the table's rows
that have `u11`. A record is joined to its atom as the `u11` to `u23` columns
join it, so a record id given to two rows counts for no atom. 9,227 entries
(52.1 %) have no anisotropic record and a fraction of 0, which is how an
isotropic refinement is deposited. 6,610 (37.3 %) have a fraction of 0.9 or
more.

The fraction does not by itself show that each atom was refined
anisotropically. A program can write, for every atom of a TLS group, the
tensor the group's motion gives it. Of the 3,932 entries with TLS groups, 2,549
have a fraction of 0.9 or more. So read the two columns together. A fraction
near 1 with no TLS group is what an individual anisotropic refinement gives, a
fraction of 0 with no TLS group what an isotropic one gives, and a fraction
above 0 with TLS groups leaves open which the atoms had.

PDB-AI writes no column that names an entry's B-factor model. Where to cut the
fraction, and what to make of a mixed entry, are choices of the analysis.
`_refine.pdbx_isotropic_thermal_model`, the depositors' own description, is
free text that about 7 % of the entries state, so it is not made a column. It
stays among the raw `_refine` items of the `all` profile for an entry with one
refinement record.

The columns are listed under [Refinement](dataset-schema.md#refinement) in the
dataset schema.

## Radiation dose

The RCSB entry record has no dose field. `radiation_dose_MGy` is therefore the
first value given in MGy, kGy, or Gy in the entry title, as dose-series entries
often state it there, and is missing for most entries. A value in kGy is divided
by 1,000 and one in Gy by 1,000,000, a value may be written in scientific
notation ("7.88x10^3 Gy"), and a dose rate, such as "40 kGy/s", is not read.
Room-temperature and serial entries mostly state their dose in kGy. `is_high_radiation_dose` is true
above 5 MGy, a cut-off PDB-AI chose.

## Radiation damage (RABDAM)

PDB-AI computes `bdamage`, `packing_density`, `bnet`, and `bnet_percentile`
in-process. It uses
[snell-lab/rabdam-custom](https://github.com/snell-lab/rabdam-custom) 1.0.0 from
its `main` branch. This package is an accuracy-focused rewrite of RABDAM 2
(Shelley and Garman, 2022).

RABDAM is off by default. Use `--include-rabdam` to enable it. When enabled,
PDB-AI raises `RabdamCustomNotInstalledError` if the package is unavailable.
There is no fallback implementation.

Bnet percentile uses rabdam-custom's resolution-matched reference database. The
database was calibrated against **PDB-Redo re-refined** B-factors. RCSB scoring
uses the same reference because an RCSB-calibrated equivalent is unavailable.

The reference database is stored at
`pdbai/engines/analyses/rabdam/rabdam_custom_bnet_reference.csv`. Its source and
SHA-256 digest are recorded in the adjacent manifest and in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

### Scoring

Each eligible atom has B > 0 and 0 < occupancy ≤ 1, and is a protein atom other
than hydrogen. For each one:

1. **Packing density.** Count the non-hydrogen atoms within 7 Å, whether
   protein, solvent, or ligand, provided they pass the same occupancy and
   B-factor checks. The count runs over the crystal environment, built from the
   mmCIF file's symmetry operators and the neighbouring unit cells (±1 cell
   along each axis).
2. **BDamage.** Atoms are ranked by packing density. A sliding window (2% of
   eligible atoms, minimum 11, odd-sized) gives each atom a local reference
   B-factor, and `bdamage = atom_B / window_reference_B`.
3. **Bnet.** A Gaussian kernel-density estimate is fitted to the BDamage values
   of the Asp/Glu carboxyl oxygens (Asp OD1/OD2, Glu OE1/OE2). Bnet is the area
   under that estimate at or above the median BDamage of all scored atoms,
   divided by the area below it.
4. **Bnet percentile.** Bnet is ranked at the resolution of the RCSB file, for
   both sources, so the two percentiles of an entry are comparable. PDB-Redo can
   report a slightly different resolution (for example 1.50 rather than 1.55 Å),
   so a `_pdb_redo` percentile can differ from that of a stock rabdam-custom run
   on the PDB-Redo file. It is ranked against the 1,000 nearest-resolution
   structures (plus boundary ties) from rabdam-custom's bundled 124,948-entry
   cryo reference database, which spans 0.59–3.5 Å. There is no hard resolution
   cut-off. A structure outside that span is ranked against the nearest edge of
   the reference, and `bnet_percentile_resolution_in_range*` records when that
   happened. Bnet itself needs only two Asp/Glu carboxyl oxygens with non-zero
   BDamage variance, and `bnet_site_count*` records how many fed the estimate.
   Neither the reference nor the pipeline applies a minimum site count, so that
   filter belongs in the analysis.

This same algorithm runs once per requested source (twice in `both` mode). The
only difference between `_pdb_redo` and `_rcsb` columns is which mmCIF file's
coordinates and B-factors were fed into it.

### Choosing what RABDAM scores: `--rabdam-source`

`--rabdam-source` (`pdb_redo` default / `rcsb` / `both`) controls which
structure source RABDAM actually scores.

- **`pdb_redo`** (default). RABDAM scores PDB-Redo's re-refined mmCIF, matching
  the calibration of its Bnet-percentile reference table. This triggers a
  PDB-Redo download even if you never touch `--include-pdb-redo`. A run with
  RABDAM on (it is off by default, like every engine) and PDB-Redo comparison
  off still fetches PDB-Redo's file internally, just to score it. This is
  intentional and always surfaced, by a caption in the web UI, a note in the CLI
  `--help` text, and separate `pdb_redo_structure_parsed`,
  `pdb_redo_comparison_success`, `rabdam_status`, `rabdam_delivered_pdb_redo`,
  and `rabdam_delivered_rcsb` columns in `run_report.csv`.
- **`rcsb`**. RABDAM scores RCSB's own standard mmCIF directly instead. No
  PDB-Redo download occurs for RABDAM's sake (though `--include-pdb-redo` can
  still trigger one separately, for the comparison columns). Useful when you
  specifically want damage metrics computed against the deposited,
  non-re-refined coordinates.
- **`both`**. RABDAM scores both files and produces both column sets, letting
  you directly compare how PDB-Redo's re-refinement changes the radiation-damage
  signal. If only one source can be scored, `rabdam_status` is `partial`, and
  the delivered source is identified by its `rabdam_delivered_pdb_redo` or
  `rabdam_delivered_rcsb` column. It becomes `available` only when both
  requested sources are delivered. (`partial` is not exclusive to this mode, as
  described below.)

### What `rabdam_status` means

A source is delivered when its BDamage reached at least one atom row and its
Bnet and percentile exist. `rabdam_status` is `available` when every requested
source is delivered, `failed` when no requested source could be scored, and
`partial` when at least one was scored but not every one was delivered. Four
situations leave a source undelivered. Its structure file may be missing, as
when PDB-Redo has no entry. A `both` run where that happens to one source and
the other is scored is `partial`, while a single-source run with no file is
`failed`. The scorer may also fail on the file, which likewise gives `partial`
in `both` mode when the other source is scored. A scored source is undelivered
when BDamage exists but Bnet or its percentile could not be produced. The causes
are too few Asp/Glu carboxyl oxygens, zero variance among them, or no positive
resolution to match the reference against. The subtler case is a scored source
whose per-atom scores reach no atom row. BDamage and packing density are
per-atom and must be matched onto the atom frame, but Bnet and its percentile
are structure-level scalars that need no matching. So if the RCSB↔PDB-Redo atom
bridge is rejected or pairs no scored atom, or no RCSB-scored serial number is
found in the atom table, `bdamage_*`/`packing_density_*` come back empty while
`bnet_*`/`bnet_percentile_*` remain valid. Each of the last two cases is
`partial` even with a *single* source, not `failed`. The source's own delivered
column (`rabdam_delivered_pdb_redo` or `rabdam_delivered_rcsb`) is false
whenever that source was not delivered, and the `rabdam_message` column of
`run_report.csv` and the manifest entry carry the reason, along with every other
RABDAM warning for the structure.

### RABDAM column names

Every RABDAM column name says which structure was scored. In a build from
downloaded entries, each column carries the suffix of its source, `_pdb_redo` or
`_rcsb`, as in `bdamage_pdb_redo`/`bdamage_rcsb`, and likewise for
`packing_density`, `bnet`, `bnet_percentile`, `bnet_site_count`,
`bdamage_scored_atom_count`, and `bnet_percentile_resolution_in_range`.
`pdb_redo` produces only the `_pdb_redo` family, `rcsb` only the `_rcsb` family,
and `both` produces both. An uploaded mmCIF file has no RCSB/PDB-Redo split, so
an upload build scores the file as given and writes the same family without a
suffix (`bdamage`, `packing_density`, `bnet`, …). Column profiles select these
columns with wildcards such as `bdamage*`, so one profile fits every naming.

PDB-Redo scoring also emits structure-level QA columns:
`bdamage_coverage_pdb_redo`, `rabdam_alignment_rcsb_ambiguous_atom_count`, and
`rabdam_alignment_pdb_redo_ambiguous_atom_count`, the last two also in
`run_report.csv` beside `rabdam_alignment_match_rate`, the share of protein
heavy RCSB atoms the bridge matched. Coverage is the fraction of protein heavy
atoms that received a BDamage score. It is below 1 when RABDAM excluded atoms
for zero occupancy or non-positive B, when the bridge could not pair them, or
when rabdam-custom scored another alternate conformer than the atom table keeps,
and 0 when no score could be bridged. It is missing when PDB-Redo's model was
not scored. Coverage is a filtering feature, not a fault, so a low value raises
no warning of its own. rabdam-custom keeps the highest-occupancy conformer of
each atom, as its reference database was computed, while the atom table keeps
one conformer per residue. Where the two choose differently, as at a site with
two residue types, the scored atom has no row and its row has no score.
`rabdam_message` counts, for each source, the scored atoms that reached no row,
which also include atoms PDB-Redo built that the deposited model lacks. In
`both` mode, four additional boolean columns (`bdamage_matches`,
`packing_density_matches`, `bnet_matches`, `bnet_percentile_matches`) flag
whether the two sources agree within an absolute tolerance of 1e-3 for that atom
or structure. They are blank (a pandas `<NA>`) where either side has no
comparable value, e.g. where the atom didn't survive the composite-key alignment
on one side. Like every declared boolean column, these are pandas nullable
booleans and serialize as `True`/`False`/empty. See [Boolean
columns](dataset-schema.md#boolean-columns).

### Cross-file atom alignment

Because RCSB's mmCIF and PDB-Redo's mmCIF are independently numbered files
(different alt-loc/water/hydrogen treatment between the two refinements),
RABDAM's PDB-Redo-scored results, keyed by PDB-Redo's own `_atom_site.id`, can't
be merged onto the RCSB-sourced atom frame by raw serial number.
`pdbai/structure/alignment.py` bridges the two via a composite natural key built
from the entry, the model, and the *author* identity (author chain, author
sequence number, insertion code, residue name, and atom name) instead. Label
identity is deliberately not used across files. PDB-Redo renumbers
`label_seq_id` consecutively from the first observed residue of each chain and
reassigns `label_asym_id` sequentially, so a label-based key pairs atoms with
the wrong residue or chain whenever a chain has unobserved N-terminal residues,
internal gaps, or non-alphabetical chain ordering. Author identifiers are
preserved verbatim by PDB-Redo. A structure numbered by its label identifiers
(`residue_numbering` is `label`) still pairs by author identity alone. The
author chain and number are then read from `_atom_site.auth_asym_id` and
`_atom_site.auth_seq_id`, never from `chain_id` and `res_num`, and an atom
without them is left unpaired. As a second, physical check, the aligner measures
the distance between every matched RCSB/PDB-Redo atom pair and rejects the whole
alignment when the median exceeds 1 Å. Re-refinement moves atoms by hundredths
to tenths of an ångström, so a larger median can only mean the identity match
paired different atoms. A rejected structure gets a match rate of 0, empty
PDB-Redo-derived columns (deltas, BDamage, packing density),
`pdb_redo_comparison_success` false, and an explicit warning in the run report.
An alignment that pairs no atom likewise gives `pdb_redo_comparison_success`
false and a warning. The rejection warning names the median shift. When no
protein heavy atom pairs on author chain IDs, the RCSB label chain IDs are tried
in their place, because PDB-Redo writes those as author chains for some entries
with multi-letter chains. The distance check still applies, and the entry gets a
warning in the run report saying so. The PDB-Redo comparison also logs a warning
when fewer than 95% of *protein heavy atoms* find a partner (the RABDAM bridge
does not, since it aligns a deliberate subset). The rate is deliberately
measured over protein heavy atoms only. PDB-Redo strips deposited hydrogens and
remodels waters, so counting those as misses would report modelling choices
rather than alignment quality (entries deposited with riding hydrogens would
show rates near 50% despite perfect protein alignment). An entry with no protein
heavy atoms, such as a nucleic-acid entry, is measured over all its heavy atoms
instead, and one with no heavy atoms over all its atoms. Match rates and
ambiguous-atom counts are retained separately for the PDB-Redo comparison and
RABDAM bridge in the dataset, run manifest, `run_report.csv`, and
`run_summary.md`, rather than existing only in transient logs. RCSB-scored
results need no such bridge. They are merged directly by `_atom_site.id` onto
the same file that produced them.

## Solvent accessibility (SASA)

Solvent-accessible surface area (SASA) is optional and off by default. Enable it
with `--include-sasa`. PDB-AI computes the rolling-probe accessible surface of
Lee and Richards (1971) in-process, with the Shrake–Rupley (1973) test-point
method. FreeSASA is not a runtime dependency. Comparisons against it serve only
as regression references.

### Algorithm

Each coordinate model is calculated independently, so NMR/ensemble models
neither occlude one another nor share residue totals. For each target
protein-polymer heavy atom *i*:

1. Place *N* deterministic Fibonacci test points on the probe-inflated sphere
   (radius r_i + 1.4 Å).
2. A test point is *buried* if it lies within the probe-inflated radius of any
   neighboring heavy atom *j*.
3. `ASA(i) = 4π(r_i + 1.4)² × (unburied points / N)`.

The named modes use 92 points (`fast`, for high-throughput work), 1,000
(`default`), or 5,000 (`high`). `--sasa-points` supplies a custom positive
value. The chosen mode and point count are emitted with every result. Neighbour
candidates use `scipy.spatial.cKDTree`, while immutable point grids are cached
by density.

### Radii and normalization

Standard protein atoms use the DSSP accessibility convention, with backbone N
1.65 Å, Cα 1.87 Å, carbonyl C 1.76 Å, backbone O 1.40 Å, and side-chain heavy
atoms 1.80 Å. `CA` here means alpha carbon, not a calcium ion. Eligible residue
RASA is:

```text
RASA = calculated residue SASA / Tien2013 theoretical ALLOWED maximum
```

Tien's maxima were generated with DSSP, so the numerator uses DSSP's radii and
1.4 Å probe, and `Tien2013_ALLOWED_DSSP` names that pairing rather than implying
a universal scale. The areas themselves come from PDB-AI's Shrake–Rupley
calculation, not from DSSP. Tien took the largest area of each residue X in a
Gly–X–Gly tripeptide over its biophysically allowed backbone and side-chain
conformations, rather than in one extended conformation. MSE uses the MET
maximum, while SEC and CSE use the CYS maximum. RASA is not clipped to 1.

A residue is buried when its RASA is below 0.25 and exposed when it is at least
0.40, and a residue between the two is neither. Tien gives no cut-offs, so these
are PDB-AI's choice. A finite point grid is an approximation, so treat a residue
very close to either cut-off as a continuous RASA value rather than a firm
label.

### Protein and bound contexts

The engine calculates the same protein target atoms twice:

- `protein`: protein-polymer atoms occlude protein-polymer atoms.
- `bound`: supported non-protein polymers, biological ligands, cofactors,
  metals, and configured unknown non-polymers additionally occlude the protein.

### Warnings

SASA warnings come in two kinds. *Degrading* warnings mean the calculation
itself fell short, as with a model that produced no areas, an occluder that had
to be dropped, or a residue whose identity could not be resolved. They set
`sasa_status` to `degraded`. *Advisory* warnings
(`PARTIAL_OCCUPANCY_NONPROTEIN`, `UNKNOWN_NONPOLYMER_INCLUDED`) describe the
surroundings the calculation ran in. A partial-occupancy or unclassified
occluder makes the *biological* interpretation of the bound context ambiguous,
but leaves every protein-context area exact. Those two are reported in
`sasa_warning_codes` and flagged in `bound_context_ambiguous` without lowering
`sasa_status`, so `sasa_ok` tracks whether the numbers are sound rather than
whether the ligand environment was tidy.

### Partner atoms and residue eligibility

Non-protein partner occluders use element-specific Alvarez (2013) radii. A
requested partner atom with an unknown element, a missing coordinate, or a
missing or non-finite occupancy never silently receives a fallback. It is left
out, the structure-wide bound context becomes incomplete and degraded, and the
protein-only result remains usable. The exception is a table with no occupancy
column at all, whose atoms are all read as fully occupied. Water and recognized
bulk solvent occlude neither context. Experimental additives and cryoprotectants
are excluded unless `--sasa-include-experimental-additives` is set. Unknown
non-polymers are included by default to retain possible ligand information. Any
requested unknown non-polymer atom whose occupancy is not zero sets the
structure-wide ambiguity flag, whether or not its element has a radius.
`--sasa-exclude-unknown-nonpolymers` opts out.

Polymer membership comes from mmCIF polymer/entity and peptide-linking component
identity, not `ATOM` versus `HETATM`, so supported modified polymer residues
remain protein targets. Explicit non-polymer entity metadata also prevents an
unusual `ATOM` record from becoming a protein target. Zero-occupancy atoms
neither contribute nor occlude. Alternate identifiers may be alphabetic,
numeric, or multi-character. Shared blank-ID atoms are retained, while the
nonblank ID with the highest mean occupancy is selected coherently across each
model-aware residue. Source order breaks ties, and means equal to nine decimal
places tie. An atom with an alternate identifier must have a finite occupancy,
and a file where one does not fails to parse rather than having it read as zero.
An occupancy written with a standard uncertainty, such as `0.50(2)`, is read
without it. This prevents atom-wise selection from assembling hybrid A/B
residues. An expected heavy atom with an occupancy other than 1, a missing
expected atom, or a conservative observed-chain terminus makes normalized RASA
and flags null while calculable raw SASA remains available. So does a residue
with no Tien maximum, which covers every non-standard residue other than MSE,
SEC, and CSE. Its atoms keep their SASA. Observed-chain termini include the
first/last modeled residue and both sides of a sequence/coordinate break, so an
unresolved internal fragment is deliberately excluded rather than treated as a
complete peptide context. Burial/exposure fields use pandas nullable booleans
rather than numeric 0/1.

### Coordinate context

No crystallographic or biological transform is generated implicitly. Standard
downloaded files are labeled `asymmetric_unit`, and uploads and transformed
inputs are labeled `as_provided`. An asymmetric-unit surface can differ
substantially from a biological-assembly interface. An assembly calculation
would require explicit assembly operators and an assembly identifier.
Crystallographic symmetry metadata alone is insufficient.

## Disulfides

The disulfide engine is optional and off by default. Enable it with
`--include-disulfides`. It applies to every technique. It writes one row per
candidate disulfide in each model to the `disulfides` table, whose columns are
listed under [Disulfides](dataset-schema.md#disulfides) in the dataset schema. A
candidate is a pair of cysteines that a deposited record joins, or whose SG
atoms lie at most 3.0 Å apart. A row does not assert a bond, since a record can
be wrong and two sulfurs can lie that close without one, as at a metal site (see
[Metal sites](#metal-sites) and [Limitations](#limitations)).

X-rays damage disulfide bonds specifically, lengthening and breaking them (Weik
et al., 2000; Ravelli and McSweeney, 2000). An electron captured into the bond
first forms a disulfide radical anion (Carpentier et al., 2010; Sutton et al.,
2013), whose S–S distance is about 2.8 Å against 2.02 Å for the intact bond
(Bhattacharyya et al., 2020). The table records the evidence a deposited model
can hold: the S–S distances, the SG occupancy and alternate conformers, and the
SG B-factor relative to CB.

The counts in this section come from a run of the engine on 2,140 PDB entries,
in 18 cohorts chosen to vary in technique, resolution, size, and content. The
run gave 24,827 rows in 1,635 entries, of which 24,414 were found by a record
and by distance, 236 by distance alone, and 177 by a record alone. The engine
failed on no entry. The run was made while the distance search still had a lower
limit of 1.5 Å, since removed (see [Search limit](#search-limit)), and before
the table had the `metal_bridged` column, so the counts of metal sites come from
a separate reading of the same files.

### Every conformer

The atom table keeps one conformer of each residue (see `occupancy` under
[Identifiers](dataset-schema.md#identifiers)), so a disulfide modelled as an
intact conformer A and a broken conformer B shows only one of them there. The
engine therefore reads the structure file again, the download or the upload, and
takes every cysteine SG and CB row of every conformer and model. A row is a
cysteine's when its residue name is `CYS`, in any case, and its atom name
(`_atom_site.label_atom_id`) is `SG` or `CB`, so hydrogen and deuterium atoms
are never read, and a cysteine recorded as HETATM is read like any other. The
residue name is the author one (`_atom_site.auth_comp_id`), or the label one
(`_atom_site.label_comp_id`) where the author one is missing, row by row, which
is the rule the parser applies to `res_name`.

Each row gets the `site_id` the atom table would give it. The rows the atom
table kept are found again by model number and `_atom_site.id`, and must carry
the same `site_id` there. When one does not, or when a cysteine SG or CB of the
atom table is missing from the file, the engine fails for that structure rather
than write rows that disagree with the atom table. It also fails when two rows
share a `site_id`, since the table could not tell them apart.

The key columns of a row hold what the atom table holds for the same atom, since
they are the join columns. The engine names its rows by the parser's own rule,
so a file without `_atom_site.auth_comp_id` has the label residue name in
`res_name`, and a file without author residue numbers has its label chain and
label number in `chain_id` and `res_num`, as
[Identifiers](dataset-schema.md#identifiers) explains.

Two conformers can coexist when their alternate location labels are equal or
either has none, so a conformer A is never measured against a conformer B. A
conformer is usable when it has finite coordinates and an occupancy that is not
zero or below. A missing occupancy leaves it usable, since unknown is not zero.

### Finding pairs

Pairs are found model by model from two sources, and `detection_method` says
which found each one.

**Deposited records.** The engine reads the file's `_struct_conn` category. A
record counts when its `conn_type_id` is `disulf`, in any case, and both
partners may be cysteine SG atoms. A partner's component is its author one, or
its label one when that is missing, and must be `CYS` or missing. Its atom is
its label one, or its author one when that is missing, and must be `SG` or
missing. A record naming another residue or atom, such as a cysteine joined to
mercaptoethanol or glutathione, or an oxidized cysteine, is skipped without a
count.

A record joins two atoms of one molecule when both partners take the same
symmetry operator, compared without underscores, with a missing operator read as
`1_555`. One operator applied to both leaves their distance unchanged. A record
whose partners take different operators joins the molecule to a symmetry mate.
It gives no row and no warning and is counted in the run report's
`disulfide_symmetry_record_count`. This check comes first, since a disulfide
across a two-fold axis names one residue twice under two operators.

A partner is found by its author chain, residue number, and insertion code when
the author chain and residue number are both given. Its label chain and sequence
number are used only when they are not, never after the author fields fail to
match. In a structure whose `residue_numbering` is `label` the rows are named by
label identifiers, so a partner is found there by its label chain and sequence
number alone, whether or not the record gives author fields. `_struct_conn`
names no model, so each record is tried in every model and gives a pair in each
model where its partners are two distinct cysteines with SG rows. A record that
gives a pair in no model is unmatched. It is counted in
`disulfide_unmatched_record_count`, a warning names it and the reason (the file
holds no SG or CB atom of that cysteine, the cysteine has no SG atom, or it is
named twice), and the structure is `degraded`. Reversed records, records
repeated for each conformer combination, and exact copies give one pair, kept
once. Files such as 3DZP and 7LND carry up to three records per pair this way. A
record's alternate location and stated distance are not read, since a row
summarizes every conformer and its distances are measured from the coordinates.

**Distance.** In each model, the engine pairs usable SG conformers of different
cysteines that can coexist and lie at most 3.0 Å apart, the limit included.
There is no lower limit (see [Search limit](#search-limit)). The distance is
rounded to 6 decimals before it is compared with the limit, so coordinates
written to three decimals fall on the side of the limit their decimal values put
them. The search never crosses models.

**Found by a record alone.** A pair that a record names but the distance search
does not find keeps `struct_conn` as its only source and reports whatever
distances could be measured. Such a row is not evidence of a cleaved bond. In
the 2,140 entries checked, none of the 177 rows found by a record alone was one.
In 133 a whole residue was deposited at zero occupancy, so no distance was
measured (see `no_usable_sg` under [Measures and
statuses](#measures-and-statuses)). In 30 the SG atoms lay 0.867–1.498 Å apart,
below the lower limit the search then had, and 27 of those were in NMR
ensembles. The other 14 lay 3.21–4.42 Å apart, all in the NMR ensembles 5LG7 and
6QES, where a record is applied to every model. No diffraction entry gave such a
row with a distance above 3.0 Å. Without the lower limit, the 30 short pairs are
found by distance as well.

### Search limit

The limit of 3.0 Å is a convention, not a boundary the data show. An intact S–S
bond is 2.02 Å and the radical anion about 2.8 Å (Bhattacharyya et al., 2020),
and restraint dictionaries put the S–S target near 2.03 Å (Engh and Huber,
2001). Two sulfur atoms in van der Waals contact lie about 3.6 Å apart by
Bondi's (1964) radius of 1.80 Å, or 3.8 Å by Alvarez's (2013) 1.89 Å. Two
cysteines on one well-determined tetrahedral zinc lie about 3.8 Å apart, since
Zn–S is about 2.33 Å (Harding, 2006). Phenix uses the same limit, as its
`pdb_interpretation.disulfide_distance_cutoff` defaults to 3 Å (phenix.refine
documentation, phenix-online.org).

In the 27 entries checked while building the engine (lysozyme, BPTI, rubredoxin,
and dose series), every deposited disulfide measured 2.02–2.11 Å and no SG–SG
pair fell between 2.3 and 3.0 Å. The nearest non-bonded pairs were the Fe–Cys4
site of rubredoxin (1IRO, 3.57–3.79 Å), neighbouring disulfides in 5MCC (3.50 Å
or more), and one BPTI NMR model (1PIT, 4.30 Å). That describes those 27 entries
and no more.

The 2,140 entries showed no such gap. In them, 582 rows in 93 entries had a
shortest distance of 2.3–3.0 Å (X-ray 273, NMR 212, electron microscopy 91,
MicroED 3, and neutron 3), and 1,161 pairs of cysteines whose SG atoms lay
3.0–3.5 Å apart were in no row. Few of either were partially broken bonds. Of
the 582 rows, 177 were at metal sites, 168 were other rows of NMR ensembles, 88
were other rows of electron microscopy or MicroED models, 148 were X-ray
disulfides that a record names and that were refined long, 55 of them in the one
entry 3BJW, and one was a neutron row. About 15 were plausibly a real elongation
or partial reduction. Of the 1,161 pairs left out, 444 were bridged by a metal,
474 were contacts between the cysteines of two neighbouring disulfides, which
begin near 3.15 Å, and 66 were X-ray pairs with the geometry of a disulfide and
no record. A larger limit would therefore add more pairs that are not bonds than
pairs that are.

A band empty on each side of 3.0 Å was found only in the neutron cohort and in
X-ray entries at 1.5 Å resolution or better, apart from nickel sites. In the
neutron cohort, the rows without a metal ended at 2.357 Å and the next pair
without a metal lay above 4.1 Å. In X-ray entries at 1.5 Å or better, 5 rows
without a metal lay at 2.3–3.0 Å, no pair left out had the geometry of a
disulfide up to 3.5 Å, and the closest plain contact was at 3.27 Å. Elsewhere
the distances run on across the limit. In the six general X-ray cohorts, the
rows without a metal numbered 13, 17, 31, 26, 15, 15, and 20 per 0.1 Å from 2.3
to 3.0 Å, and the pairs left out that had the geometry of a disulfide numbered
10, 12, 12, 14, and 10 per 0.1 Å from 3.0 to 3.5 Å. One disulfide can fall on
both sides. Cys 538 and Cys 564 of myeloperoxidase are a row in chain E of 6WXZ,
at 2.997 Å, and in no row in chain B, at 3.253 Å. The limit therefore leaves out
the most separated conformers of damaged bonds, which
[Limitations](#limitations) describes.

The limit decides which pairs become rows, not which distances a row reports.
Once a pair is found, every coexisting combination of its SG conformers is
measured. In 1HJ8, for example, partially broken disulfides are modelled as a
major bonded conformer at 2.03–2.11 Å and a minor conformer whose SG lies
3.21–3.44 Å from its partner. Such a pair is found through its bonded conformer,
and its `ss_distance_max_A` can then report the broken one.

The search has no lower limit. It once left out pairs closer than 1.5 Å, on the
reasoning that two SG atoms that close are one atom modelled twice. The 2,140
entries held 65 pairs of usable SG conformers of different cysteines that can
coexist and lie below 1.5 Å, and none was one atom modelled twice, since the two
CB atoms lay 2.81–4.88 Å apart in every one. A deposited record names 30 of
them, which were rows found by the record alone. Seven were disulfides of
distorted geometry that no record names, in the X-ray and electron microscopy
entries 8QCM, 3UTZ, 4LAJ, 8TOC, and 5DQO, and they were in no row. The rest were
23 pairs of zinc ligands in the NMR ensemble 5LG7, 3 contacts between
neighbouring bonds in the NMR ensemble 1MPZ, one collapsed cluster of cysteines
in 5TY4, and one clash between conformers in 6SRY.

An implausibly short distance is therefore reported as measured, with no warning
and no status of its own. It marks an error in the model, not the length of a
bond. In the 2,140 entries, 260 rows had a shortest distance below 1.9 Å (191
NMR, 50 X-ray, 16 electron microscopy or MicroED, and 3 neutron), and 31 of them
below 1.5 Å. Of the 14,145 X-ray rows with a distance, 50 lay below 1.9 Å and 4
below 1.5 Å. Without the lower limit, the 34 pairs below 1.5 Å that no record
names would be rows as well, 8 in X-ray, electron microscopy, or MicroED entries
and 26 in NMR models.

The limit and the metal contact distance of [Metal sites](#metal-sites) are
written into the table's description, which the manifest and run summary hold,
and into the run summary's line for `include_disulfides`, so every run records
the ones it used.

### Metal sites

The engine reads no `metalc` record and leaves no pair out because of a metal.
Two cysteines bound to one metal atom are a row when a record joins them or
their SG atoms lie within the search limit, and the column `metal_bridged` marks
such a row. It is `True` when a usable SG conformer of each residue lies within
2.8 Å of the same metal atom of the same model, the limit included, where the
two SG conformers can coexist with each other and each can coexist with the
metal atom. It is `False` otherwise and never missing. The distance is rounded
to 6 decimals before it is compared, as in the pair search.

A metal atom is any `_atom_site` row, of any residue and either record type,
whose `_atom_site.type_symbol` names a metallic element, with finite coordinates
and an occupancy that is not zero or below. The metallic elements are the 84
the metal-site engine counts (see [Metal atoms](#metal-atoms)), which include
sodium, potassium, magnesium, and calcium. Case and a charge suffix do not
matter, so `ZN`, `Zn`, and `Zn2+`
all name zinc. A missing or unreadable symbol names no metal, so the flag is
`False` on every row of a file that names no elements. Only the atoms of the
file are read. No symmetry mates are generated, so a metal of a symmetry mate is
not seen.

Bonds from a metal to a cysteine sulfur are about 2.1–2.4 Å long, and Zn–S is
about 2.33 Å (Harding, 2006), so 2.8 Å allows for poor geometry. The flag is
information and never a filter. It adds, removes, and changes no row, and
`detection_method`, `pair_status`, and every measure are the same with it as
without.

Cysteines bound to one metal do fall within the search limit. In the 2,140
entries checked, 259 of 24,827 rows (1.04 %), in 43 entries, had both sulfurs
within 2.8 Å of one metal atom. The count barely moves with the distance, as it
was 245 at 2.6 Å and 263 at 3.0 Å. The metal was zinc in 203 rows, nickel in 37,
iron in 18, and copper in 1. The 259 rows were of six kinds:

- **Low-resolution zinc sites, 110 rows in 26 entries.** These are X-ray or
  electron microscopy models of ribosomes and RNA polymerases at 2.9–4.3 Å
  resolution. Deposited disulf records name 84 of the rows, and 63 measure 2.3 Å
  or less. Some are sets of three or four sulfurs all paired with each other,
  which is not chemistry, as in chain D4 of 4V6F, where all six pairs of Cys 16,
  18, 36, and 39 measure 2.02–2.05 Å.
- **One NMR ensemble, 85 rows.** 5LG7 has two zinc sites in 31 models, with S–S
  distances of 1.51–3.82 Å.
- **[NiFe] hydrogenase sites of ordinary geometry, 24 rows in 8 entries.**
  Conformers at full occupancy bind nickel or iron at 2.1–2.4 Å and lie
  2.80–2.99 Å apart, at 1.04–2.30 Å resolution. B:78 and B:492 of 6Z8J lie 2.872
  Å apart at 1.09 Å resolution. These are neither disulfides nor damage.
- **Distorted nickel, iron, or copper sites, 24 rows in 4 entries.** They are
  models at 2.2–3.0 Å resolution, and 21 of the rows are in 9E15.
- **Alternate redox states, 14 rows in 5 entries.** A minor conformer is an S–S
  bond near 2.03 Å, and the other conformer is bound to the metal with the
  sulfurs 3.6–4.0 Å apart (4UEQ, 8W6X, 5XLG, 8OSW, and 8COL). Only these 14 rows
  hold a real S–S conformer.
- **Zinc at zero occupancy, 2 rows.** Both are in 7I9F. The engine leaves out a
  metal atom at zero occupancy, so that zinc flags no row.

A well-determined zinc site keeps its cysteines apart. At 2.0 Å resolution or
better, none of 587 pairs of cysteines on one zinc lay within 3.0 Å, and their
shortest distance was 3.33 Å and their median 3.82 Å. Two cysteines on different
atoms of one iron–sulfur cluster were never a row, since the closest such pair
lay 3.61 Å apart. Over all cohorts, 2,694 of the 2,953 pairs of cysteines on one
metal atom were in no row. The cysteines on the nickel of a [NiFe] hydrogenase
lie 2.8–3.1 Å apart, so the limit cuts through the ordinary geometry of that
site, with 33 rows at 2.7–3.0 Å and 17 pairs in no row at 3.0–3.1 Å.

Neither `detection_method` nor a deposited record tells these rows from
disulfides. Of the 259, 135 were found by distance alone, 118 by a record and by
distance, and 6 by a record alone, and they were 135 of the 236 rows the whole
run found by distance alone. Exclude the rows where `metal_bridged` is `True`,
or keep them apart, before reading any measure (see [Reading the
measures](#reading-the-measures)).

The flag cannot mark a site whose metal is not in the model. In the run, 170
rows sat in sites read as zinc sites that had no zinc within 3.2 Å, most of them
in ribosomes, where three or four sulfurs were deposited as bonded to each
other, and 28 more rows had only one SG within 2.8 Å of a metal. A cysteine in
more than one row marks most of these (see [Limitations](#limitations)).

The count of 259 used symmetry images and metals at zero occupancy, which the
engine does not, so the engine can flag slightly fewer rows. In the run, 2 of
4,329 contacts between an SG and a metal used a symmetry image, and neither was
in a row. A recount that required the metal to have an occupancy above zero gave
257 rows in 42 entries.

### Representative conformers

Each residue of a row is named by its representative SG, the SG row the atom
table keeps, and that atom's row of the atom table holds the same identifier in
`disulfide_site_id`. When the kept conformer has no SG, as when conformer A
holds CB and SG at 0.45 and conformer B holds CB alone at 0.55, the
representative is the SG of highest occupancy, a missing occupancy ranking
lowest and file order breaking a tie. No atom then links to that residue, a
warning names it, and the row's `pair_status` is `sg_not_in_atom_table` unless
an earlier gap applies. The representative CB is the CB that can coexist with
the representative SG, preferring the one the atom table keeps and otherwise
ranked the same way. The B-factor ratios use the two representatives, so on an
`sg_not_in_atom_table` row they come from the fallback SG's conformer rather
than the one the atom table keeps.

The atom table keeps the conformer whose labelled atoms have the highest mean
occupancy, hydrogen and deuterium included, so in a neutron model the
occupancies of H and D atoms modelled as alternate conformers can decide which
SG is the representative and so which conformer gives the ratio.

### Measures and statuses

The distances use every usable pair of SG conformers, one from each residue,
that can coexist. The occupancy columns use every SG row the file holds for the
two residues, including rows at zero occupancy or without coordinates, and the
deficit also every CB row. Numbers written with a standard uncertainty, such as
`2.040(3)`, are read as the parser reads them, without it. A residue's SG/CB
ratio is missing when it has no representative CB, or when either B-factor is
missing, not finite, or zero or below. It is also missing when either atom has
an occupancy of zero or below, since such an atom was placed without being seen
in the data and its B-factor is not a measurement. Non-diffraction models may
carry values that are not B-factors, which are reported as measured, so filter
by technique at analysis time. The two ratio columns are the smaller and larger
of the two residues' ratios, so they do not depend on which residue comes first.
Only the occupancy sums and the deficit are rounded. `metal_bridged` is
described under [Metal sites](#metal-sites).

Each row's `pair_status` is `ok`, or the first of these that applies:

1. `no_usable_sg`: a residue has no usable SG conformer, so the distances are
   missing. Only a pair found from a record can have it. The status does not
   mean that a sulfur was left out of the model or lost to damage. In the 2,140
   entries checked, all 133 rows with it were whole residues deposited at zero
   occupancy with their SG coordinates present. Of these, 120 were in 4UDF, an
   electron microscopy model with every atom at zero occupancy, and 13 in
   zero-occupancy segments of the low-resolution X-ray models 1C9T, 1IVO, 3L5N,
   and 2WIN.
2. `no_compatible_conformers`: both residues have usable SG conformers but none
   can coexist, such as an A-only SG against a B-only one, so the distances are
   missing. Only a pair found from a record can have it.
3. `occupancy_missing`: an SG or CB occupancy of either residue is missing. A
   missing SG occupancy leaves both SG occupancy columns and the deficit
   missing, and a missing CB occupancy the deficit.
4. `cb_missing`: a residue has no representative CB, so the ratios are missing.
5. `b_factor_unavailable`: a representative SG or CB B-factor is missing, not
   finite, or zero or below, or its atom is at zero occupancy, so the ratios
   are missing.
6. `sg_not_in_atom_table`: every value is present, but a residue's
   representative SG is a conformer the atom table dropped, so no atom links to
   it.

Only the first gap is named, so a row can lack a value that a later status would
explain.

The run report gives each structure a `disulfide_status`. It is `ok` when every
deposited disulfide within one molecule matched a pair, even when the table has
no rows for the structure or its rows name gaps. A row's gaps stay on the row,
so selecting structures whose status is `ok` does not drop those with ordinary
modelling gaps, such as a truncated side chain or a placeholder B-factor. It is
`degraded` when a record within one molecule gave no pair in any model, and
`failed` when the engine raised, which leaves the structure with no rows and an
empty `disulfide_site_id`. `disulfide_message` holds every warning, each once,
or the failure message. `disulfide_cysteine_without_sg_count` counts the
cysteines, once per model, that have a CB but no SG in any conformer, which no
row holds. It says how many cysteines the table cannot see and is not a count of
damaged bonds (see [Limitations](#limitations)).

### Reading the measures

Figure 3 of the [radiation-damage study
plan](plans/radiation-damage-study.md#figure-3-disulfides) reads these measures
as follows. The table stores the measures, and each rule below is applied in the
analysis.

- **Metal sites first.** Exclude the rows where `metal_bridged` is `True`, or
  keep them as a class of their own, before any count, fraction, or distribution
  below (see [Metal sites](#metal-sites)). In the 2,140 entries checked, rows at
  metal sites were 136 of the 582 with a shortest distance of 2.3–3.0 Å, 12 of
  the 90 that the partially broken rule selects, and 23 of the 350 with
  alternate conformers, and 19 entries had no other rows.
- **Partially broken.** A disulfide modelled as partly intact and partly broken
  has alternate conformers, a short minimum, and a long maximum, for example
  `sg_has_alternate_conformers & (ss_distance_min_A <= 2.3) &
  (ss_distance_max_A > 3.0)`. The rule finds two states and does not say which
  came first. In the 2,140 entries it selected 90 rows in 51 entries. Twelve
  were metal sites in two redox states, which `metal_bridged` sets apart. One
  was a bond being formed, Cys 117 and Cys 145 of the oxidized protease 7PZQ,
  whose reduced conformer puts the sulfurs 8.93 Å apart. Check a selected row
  against the technique, the dose, and what the disulfide does in the protein.
- **Alternate conformers.** The fraction of rows with
  `sg_has_alternate_conformers` mostly reflects resolution and how finely the
  model was built, not damage. Of the 350 such rows in the 2,140 entries, 224
  held two bonded conformers, with a longest distance of 2.3 Å or less, and 90
  the partially broken pattern. In X-ray rows the fraction fell from 0.205 at
  1.0 Å resolution or better to 0.001 above 3.0 Å. Compare it within bins of
  resolution.
- **SG occupancy below 1.** Use `sg_occupancy_deficit_max > 0`, which measures
  each SG against its own CB. Neither `sg_conformer_occupancy_min`, which is
  below 1 for any split SG, nor `sg_total_occupancy_min` measures SG loss on its
  own. The usual reason for a total below 1 without a deficit is a residue or
  chain modelled at partial or zero occupancy as a whole. In the 2,140 entries
  checked, 166 rows had a total below 1 and a deficit of 0. The whole residue
  was at zero occupancy in 133 of them and at 0.40–0.96 in 31, and in 2 the
  occupancies, deposited to two decimals, summed to 0.98 or 0.99. Such sums are
  rare, since 4 of the 24,827 rows had a total from 0.975 to below 1. Entries
  outside that run show them at Cys 159 of 3DZP, 3DZR, 4EK0, and 5KW4, on CB and
  SG alike. The deficit is 0 in all these cases and 0.25 for an SG at 0.75 on a
  CB at 1.00, as in Cys 22 of 1HJ8. Sums are kept as deposited, never clamped.
- **SG B-factor relative to CB.** An intact residue's ratio is typically above
  1, not 1, so compare ratios between groups rather than against 1. Grouped or
  constant-B refinement gives exactly 1 for reasons unrelated to damage, and
  such rows are common. In the 2,140 entries checked, both ratios were exactly 1
  on 2,947 of the 19,120 rows with a ratio (15 %), and on more than half of them
  in three cohorts, which were powder (80 of 86), electron microscopy (1,476 of
  2,034), and large electron microscopy assemblies (743 of 1,131). Among X-ray
  rows the share was 3.4 %. Leave the rows with both ratios at exactly 1 out of
  a comparison, since a median of exactly 1 says how the model was refined. The
  ratio is that of the conformer the atom table keeps, which may be a broken
  conformer of higher occupancy, except on an `sg_not_in_atom_table` row (see
  [Representative conformers](#representative-conformers)). Neutron B-factors
  are refined against nuclear density, so neutron and X-ray ratios are not
  directly comparable.
- **S–S distance against resolution.** Join `ss_distance_min_A` to `resolution`
  on `pdb_id`, from the structures table in star-schema mode. A distance of
  2.3–3.0 Å is not a sign of damage on its own. In the 2,140 entries checked,
  most X-ray rows there without a metal were bonds that a record names, refined
  long at 2–3.5 Å resolution and often many to an entry, such as 55 of the 56
  disulfides of 3BJW. Read the distance with the occupancy, conformer, and
  B-factor measures. A distance below about 1.9 Å is an error in the model (see
  [Search limit](#search-limit)).

### Limitations

- **The most separated conformers are left out.** A pair whose SG atoms lie more
  than 3.0 Å apart in every conformer is in no row unless a record names it, and
  in the 2,140 entries checked no record named such a pair in a diffraction
  entry. The rows therefore leave out the most separated conformers of damaged
  bonds. In the XFEL entry 6SRL, one disulfide is a row at 2.75 Å and seven that
  no record names lie 3.06–3.17 Å apart and are in no row. Over the run, 66
  X-ray pairs in 31 entries had the geometry of a disulfide at 3.0–3.5 Å, no
  record, and no row. Cleavage fractions from the rows are lower bounds that
  favour intact bonds most where damage is greatest.
- **Cysteines without an SG are not seen.** A cysteine with no SG atom in the
  file is in no row, and a pair that no record names is not found when an SG is
  at zero occupancy. `disulfide_cysteine_without_sg_count` counts the cysteines
  with a CB and no SG. It says how many cysteines the table cannot see and is
  not a count of damaged bonds, since a side chain is cut back for many reasons.
  The TcAChE dose series of Weik et al. (2000) models Cys 67, 94, 254, 265, 402,
  and 521 to CB with no SG and no disulf record, and the count is 6 in each of
  its nine entries (1QID to 1QIK, and 1QIM), the lowest dose included, so there
  it reflects a choice made for the whole series. The 2,140 entries held 161
  such cysteines in 41 entries, and 102 of them were side chains cut back in
  electron microscopy models. A cysteine cut back past its CB is not counted.
- **A cysteine in more than one row is rarely bonded twice.** An SG that lies
  within the limit of two others is in two rows. In the 2,140 entries checked,
  409 cysteines were in more than one row (322 in two, 85 in three, and 2 in
  four), which made 587 rows in 45 entries. Of the 409, 241 were ligands of a
  metal site (221 at zinc sites and 20 at [NiFe] sites), 135 were in NMR
  ensembles, where a third sulfur comes within 3.0 Å of a bond in some models,
  30 belonged to two neighbouring disulfides modelled in contact, and 3 to one
  disulfide of 1H8D whose second SG position was deposited as a separate
  residue. At most 297 of the 587 rows could be bonds if each sulfur has one
  partner. Count the rows each `site_id` appears in, as `site_id` or as
  `partner_site_id`, and treat a cysteine in more than one as a metal site or a
  contact.
- **Records are annotations.** A deposited record does not prove a bond. In the
  2,140 entries checked, records named 124 of the 259 rows at metal sites, and
  all 43 rows of the neighbouring disulfides modelled in contact, where one to
  three records join the wrong cysteines beside the two real bonds.
- **Overlapping chain copies can pair.** When a whole molecule's disorder is
  modelled as overlapping chain copies, a cysteine of one copy can pair by
  distance with its own copy in another chain, or with another cysteine's copy.
  Such pairs become `distance` rows with `pair_status` `ok`. A `distance`-only
  row between chains whose residues each have a copy at the same residue number
  and insertion code in another chain is likely such an artefact. None was found
  in the 2,140 entries checked.
- **Restraints shorten the distance.** Refinement restraints pull the modelled
  S–S distance toward the ideal value of about 2.03 Å (Engh and Huber, 2001), so
  partial reduction can show as difference-density peaks rather than a longer
  distance (Bhattacharyya et al., 2020). Read the distance together with the
  occupancy, conformer, and B-factor measures, and against resolution.
- **Symmetry mates are left out.** A disulfide between a molecule and its
  symmetry mate is only counted, since the file's atoms alone cannot show its
  geometry, and neither the distance search nor the metal flag generates
  symmetry mates. Such bonds are not rare. Of the 63,596 archive entries with a
  deposited disulfide record, 1,829 (2.9 %) have at least one that joins a
  symmetry mate, so give `disulfide_symmetry_record_count` beside any count of
  disulfides per entry.
- **Modified cysteines are left out.** Only residues named `CYS` are read, so a
  cysteine deposited under another component name, such as an oxidized one, has
  no row.
- **Cysteinylation is included.** A free cysteine ligand bonded to a polymer
  cysteine is a `CYS` residue and gets a row. Its atoms are HETATM, so
  `is_hetatm` on its linked atom row separates such pairs.

## Electron density at atoms

The density engine is optional and off by default. Enable it with
`--include-density`. It applies to X-ray diffraction downloads, since both of
its map sources hold maps of X-ray entries only. Its columns are listed under
[Electron density at atoms](dataset-schema.md#electron-density-at-atoms) in the
dataset schema.

A deposited model records what its depositor chose to model. A disulfide that
broke during data collection but was modelled intact looks intact in the
coordinates, and its damage shows only in the data, as a hole at a sulfur in
the difference map. The engine records what the data show at each atom and
leaves the reading to the analysis.

### Maps

The 2mFo−DFc map shows how much density supports an atom, and the mFo−DFc
difference map shows where the data want more or less density than the model
placed. `--density-source` chooses whose maps are read.

- **PDB-Redo** (`pdb_redo`, the default). PDB-Redo's `{id}_final.mtz` holds the
  map coefficients of its re-refined model, `FWT` and `PHWT` for the 2mFo−DFc
  map and `DELFWT` and `PHDELWT` for the difference map. Every entry was
  re-refined by one protocol, which makes entries comparable with one another.
  The maps describe PDB-Redo's model, not the deposited one. The file also
  holds the calculated structure factors of that model, `FC_ALL` and
  `PHIC_ALL`, from which the engine makes a third map (see [Calculated
  density](#calculated-density)).
- **wwPDB validation** (`wwpdb`). The wwPDB validation pipeline calculates both
  maps about the deposited model, with no further refinement, and the archive
  holds their coefficients as `pdbx_FWT`, `pdbx_PHWT`, `pdbx_DELFWT`, and
  `pdbx_DELPHWT`. The maps describe the model the atom table holds, as each
  depositor refined it. The files hold no calculated structure factors, so
  this source gives no calculated density.
- **Both** (`both`) reads the two side by side, so that a value can be checked
  against the other source.

The engine transforms each pair of coefficients to a grid over the whole unit
cell with gemmi, one map at a time, and uses no external crystallographic
program.

A map is sampled at 4 grid points per high-resolution limit, so at 2 Å the
points lie 0.5 Å apart. A unit cell that would need more than 150 million grid
points is sampled more coarsely, down to 2.5 points per limit, and one that
still needs more gets no values (`map_too_large`).
`density_map_sample_rate` records the rate used.

### Reading a value

A value is read at a position by cubic interpolation of the grid. It is the
density there less the map's mean, divided by the map's root-mean-square
deviation over the unit cell, the scale map viewers call sigma. The two
root-mean-square densities are structure columns, so a value can be put back on
the scale of the map coefficients.

The interpolation and the sampling were chosen together. On one entry at
1.76 Å, values read at atoms were compared with those of a map sampled three
times as finely. Linear interpolation at the common rate of 3 was off by 0.64
sigma in root mean square for the 2mFo−DFc map, and by 2 sigma at worst, since
it flattens every peak. Cubic interpolation at a rate of 4 was off by 0.05
sigma for the 2mFo−DFc map and 0.02 for the difference map. Values read from
gemmi's map and from a map made by CCP4 `fft` agreed with a correlation of
0.998 for the 2mFo−DFc map and 0.999 for the difference map.

### Two positions

Each of PDB-Redo's maps is read twice for every atom. The first position is the atom's own, in
the deposited model, which is the model the atom table describes. The second is
the position PDB-Redo's re-refined model gives the same atom, from the PDB-Redo
comparison's pairing of the two models (see [Cross-file atom
alignment](#cross-file-atom-alignment)).

The maps were calculated about PDB-Redo's model, so the second position is the
one the difference map belongs to: a negative value there says PDB-Redo's
model places more density at its own atom than the data support. The first
says what the same data show where the depositor put the atom. The two models
lie within a few hundredths of an ångström at most atoms, and the values then
agree. Where PDB-Redo rebuilt a side chain they do not, and the comparison
columns show the difference.

Three checks keep a value from being read in the wrong place. A reflection file
whose unit cell is not the deposited one gives no values (`cell_mismatch`).
Neither does an entry in which no atom paired with PDB-Redo's model
(`models_not_paired`), since the pairing is the only evidence that the two
models share a frame. An atom without finite coordinates gives none at that
position.

The wwPDB validation maps were calculated about the deposited model, so they
are read at the deposited position alone and need no pairing. The cell check
applies to them as well.

### Calculated density

A value in sigma is not a share of an atom. The hole a lost atom leaves in the
difference map is as deep as the atom's own density was high, so a heavy atom
that has lost a tenth of itself shows a deeper hole than a light one that has
lost the same tenth. In one entry at 1.82 Å, the iodine of a ligand read −13.7
sigma in the difference map and the sulfur atoms −2.2 at the median. Those
were 14% and 6% of the calculated density at the same atoms.

For PDB-Redo's maps the engine therefore makes a third map, that of the
model's calculated structure factors `FC_ALL` and `PHIC_ALL`, and reads it at
the same two positions, on the same grid, by the same interpolation. It writes
two values for each position, both on the scale of the map coefficients and
not in sigma.

- `density_fc_eA3` is the density of the calculated map at the point, less the
  map's mean. It is what PDB-Redo's whole model, bulk solvent included, places
  there at the resolution of the data.
- `density_fofc_eA3` is the density of the 2mFo−DFc map less that of the
  calculated map at the same point.

`density_fc_eA3_pdb_redo` and `density_fofc_eA3_pdb_redo` are the same at the
position PDB-Redo's model gives the atom, which is the pair to read for an atom
of the model the maps belong to.

The three sets of coefficients are on one scale, which is why one reading can
be taken from another. In each of 2,134 `{id}_final.mtz` files that were
checked, `FWT` less `FC_ALL` equals `DELFWT` at every reflection, to within 2
parts in 10 million of the root-mean-square `FC_ALL`. In the 60 of them that
PDB-Redo refined as twinned it equals twice `DELFWT` to the same precision. So
`FC_ALL` is the DFc of the map coefficients, and `density_fofc_eA3` is the
density of a map of 2(mFo−DFc) in every entry. It equals
`density_fofc_sigma × density_fofc_rms` for an untwinned entry, to within
4 × 10⁻⁶ e/Å³ at each of the 682,311 atoms of one run of 134 entries, and
twice that product for a twinned one. Taking the difference of two readings,
and not the file's own difference map, is what puts twinned and untwinned
entries on one scale.

The coefficients are doubled for a reason. With phases taken from the model, a
map of mFo−DFc shows an atom the model holds in excess at about half its
depth for the acentric reflections, and 2(mFo−DFc) shows it at about the whole
(Read, 1986). For the centric reflections, a minority, the doubled
coefficients overstate it.

#### The share the data do not support

`density_fofc_eA3_pdb_redo / density_fc_eA3_pdb_redo` is the share of the
density the model places at its own atom that the data do not support. It is
the 2mFo−DFc density over the calculated density, less one. A value of 0 says
the two agree. A value of −0.5 says the data support half of what the model
placed. A value of −1 says the 2mFo−DFc density at the atom is at the mean of
the unit cell, as where the atom is not there at all. A positive value says
the data want more than the model placed.

The numbers that follow are from one run of the 134 entries of 23 dose
series, from 0.92 to 2.4 Å, none of them twinned. Among the carbons of the
better-ordered half of each protein, at full occupancy and in one
conformation, the ratio had a mean between −0.06 and 0.05 and a standard
deviation between 0.03 and 0.13 in each entry. Over all 163,745 of those
carbons its mean was 0.002 and its standard deviation 0.07. In the six entries
of one lysozyme series at 2.0 Å, 2YBH to 2YBN, the median over the eight
cysteine sulfurs fell from 0.00 at 2.3 MGy through −0.02 at 12.3 MGy to −0.07
at 28.6 MGy.

The engine writes the two densities and no ratio. A ratio is unstable where
the calculated density is small, and it changes sign where that density is
negative. Over every carbon of the run it ran from −37,000 to 1,000. The
uncertainty of a ratio is about the root-mean-square density of the
2(mFo−DFc) map, which is `density_fofc_rms` for an untwinned entry and twice
it for a twinned one, divided by `density_fc_eA3`. Across the entries of the
run, that quotient at the median carbon followed the standard deviation above
with a correlation of 0.86, at 0.07 against 0.06 in the median entry. How
small a calculated density is too small depends on the comparison being made,
so a floor would be a threshold, and the dataset keeps both measurements for
every atom. The calculated density was below 0.3 e/Å³ at 0.3% of the atoms
that are neither water nor hydrogen in the median entry and at 19% in the
entry with the most, and at or below zero at 0.6% at most.

#### What the calculated density is

- **Above the mean, not above zero.** A map made from structure factors has no
  F(000) term, so `density_fc_eA3` is the density above the mean of the unit
  cell. The mean lies between the density of water, 0.33 e/Å³, and that of
  protein, 0.43 e/Å³, and the file does not hold it.
- **Not the atom's own share.** Every atom near a point adds to the density
  there, its bonded neighbours most, and the mean that is taken away lowers
  it. At most atoms the two nearly cancel. The calculated density at an atom
  was compared with the atom's own peak, which is its scattering factor with
  its B-factor, cut at the resolution of the data and summed over the sphere
  of reflections. In eight untwinned entries from 1.2 to 2.4 Å, with median
  B-factors from 12 to 56 Å², the median of the first over the second was
  0.90 to 1.14 for carbon, 0.91 to 1.05 for nitrogen, 0.74 to 0.97 for
  oxygen, and 0.72 to 0.97 for sulfur. Within an entry it varies more. For
  carbon the 10th and 90th percentiles were 0.86 and 1.26 in one entry at
  2.0 Å, and 0.7 and 1.7 in one at 2.4 Å with a median B-factor of 56 Å². So
  at the median the ratio above is the share of an atom to within about a
  quarter, and less surely at any one atom. It is a little more negative for
  an oxygen or a sulfur than for a carbon that lost the same share.
- **Lower at high B-factor and coarse resolution.** Over the carbons at full
  occupancy of the run above, the median was 2.77 e/Å³ in the entries finer
  than 1.3 Å, 1.65 from 1.3 to 1.7 Å, 1.03 from 1.7 to 2.1 Å, and 0.95 from 2.1
  to 2.4 Å. In one entry at 2.4 Å with a median B-factor of 56 Å² it was 0.5.
  The value rises with the number of electrons: 2.2 for carbon, 2.8 for
  nitrogen, 2.9 for oxygen, 7.2 for chlorine, and 8.2 for sulfur in one entry
  at 1.2 Å, and 19.5 for bromine in another at 1.26 Å.
- **Nominally e/Å³.** In five untwinned entries, the mean amplitude of
  `FC_ALL` was 0.93 to 1.11 times that of structure factors calculated from
  PDB-Redo's model on the absolute scale, in the shells between 2 and 4 Å. It
  was lower in the coarsest shell, where the bulk solvent counts, and in the
  finest shells of some entries. In two twinned entries it was about 1.5 and
  2 times in the shells finer than 4.5 Å. Compare `density_fc_eA3` between
  entries only for untwinned ones. The ratio does not depend on the scale.
- **PDB-Redo's model at either position.** At the deposited position the
  value is what PDB-Redo's model places where the depositor put the atom. The
  two positions give the same value unless PDB-Redo moved the atom.

A file without `FC_ALL` or `PHIC_ALL` keeps the values of its two maps, and
its four calculated and difference densities are missing, which
`density_message` in the run report says. Each of the 2,134 files that were
checked holds both columns. The third map adds about half to the time the
maps take, 0.02 s for a small cell and 1.4 s for one at the size limit, and
nothing to the peak memory, since one map is held at a time.

### Gradient and shift

The value at an atom says whether the data support the density the model
places there. It cannot say whether the data would put the atom somewhere
else. Refinement restrains a disulfide near 2.03 Å whatever the data say, so
a bond that radiation has lengthened can be modelled at its restrained
length. Each sulfur then sits between a negative lobe of the difference map
on the side of its partner and a positive lobe on the far side, and the
value at the sulfur can be near zero. The slope of the map at the sulfur is
not.

For each source the engine therefore writes the gradient of the mFo−DFc map
at the atoms of the model the map was calculated about:
`density_fofc_gradient_x_sigma_per_A_pdb_redo` and its `_y` and `_z`
partners at PDB-Redo's position, and the same three ending in `_wwpdb` at
the deposited position. They are in sigma per Å, the map's root-mean-square
density over the unit cell per ångström, along the axes of the model's
frame. The vector points where the difference density rises. Projected on
the unit vector of a bond, from the bonded atom toward the atom read, it is
positive where the data would lengthen the bond and negative where they
would shorten it. The engine projects on no bond, since which bonds matter
is the analysis's choice. The gradient is not read at the deposited
position in PDB-Redo's maps: those maps were calculated about PDB-Redo's
model, and their slope at another point mixes the distance between the two
models' atoms into it.

Like a value in sigma, a gradient grows with the atom's electrons for the
same displacement, and with the sharpness of the map. Compare it within one
element and within bins of resolution.

In the six entries of one lysozyme dose series at 2.0 Å, 2YBH to 2YBN, the
gradient at the eight disulfide sulfurs projected on the S–S axis, outward,
had a mean of 0.31 sigma per Å at 2.3 MGy and 1.54 at 28.6 MGy, and rose at
each step but one. The modelled S–S length went from 2.034 to 2.049 Å.

#### How it is read

A gradient is read from a map of the derivative itself. Each coefficient,
expanded over the whole of reciprocal space by the space group's symmetry,
is multiplied by −2πi times its reciprocal vector's component along the
axis, and the map of the products is the exact derivative of the density at
the grid points. It is read between them by the same cubic interpolation as
the values. The slope of the interpolated density map itself would be
simpler, but it is not accurate enough: against a map sampled three times
as finely, at protein atoms of two entries at 1.68 and 2.0 Å, it was off by
7% in root mean square for the difference map and 11 to 15% for the
calculated map, where the derivative maps were off by 4 to 5%, and its
curvature by 11 to 22%, where theirs was off by 3%. Expanding the
coefficients by symmetry before they are differentiated is what makes this
hold in any space group: in five entries in C 2, P 61 2 2, H 3, P 21, and
C 2 2 21, the gradients at every seventh atom correlated with central
differences of a map sampled at a rate of 10 at 0.999 or better, and were
off by 2.6 to 5.0% in root mean square. Each derivative map is
made, read, and let go before the next, so the peak memory is that of one
map, as before. The derivative maps add one to two seconds for an entry of
a few thousand atoms and about 40 seconds for one of 300,000.

No gradient of the 2mFo−DFc map is written. At an atom it is the gradient
of the calculated map, which comes from the model's neighbours and says
nothing about the data, plus that of the difference map.

#### The shift

From PDB-Redo's maps the engine also writes `density_fofc_shift_x_A_pdb_redo`
and its `_y` and `_z` partners: the shift of the atom, in Å, that would
level the slope of the difference density to first order. If the data put
an atom a small distance δ from where the model has it, the difference
density at the model's atom has a slope of minus the curvature of the atom's
density times δ. The shift is therefore minus the inverse of the calculated
map's curvature at the point times the gradient of the 2(mFo−DFc) density,
the step of Newton's method. Both are on the scale of the map coefficients,
so the shift does not grow with the atom's electrons and does not depend on
the scale of a twinned refinement. The gradient of the 2(mFo−DFc) density is
the difference map's own times the factor between the file's difference
coefficients and FWT less FC_ALL, 1 or 2, which the values read at the atoms
give (see [Calculated density](#calculated-density)). The curvature is read
from six maps of second derivatives made as the gradient's are.

- **Missing where nothing levels the slope.** Where the calculated map does
  not curve downward in every direction at the point, no nearby point has a
  level slope and the shift is missing. Over the atoms of seven residue
  types in the 134 entries of 23 dose series, it was missing at 11%: 22% of
  carbonyl carbons, 17% of alpha carbons, 12% of beta carbons, 13% of Asp CG,
  16% of Glu CD, 5% of backbone nitrogens, 1.6% of disulfide sulfurs, and
  0.7% of carbonyl oxygens. A carbon's density at these resolutions is
  often a shoulder of its neighbours'.
- **Large where the curvature is nearly flat.** Where the calculated map
  curves downward only slightly in one direction, the shift along it is
  large. At the alpha carbons of the same entries its length had a median
  of 0.09 Å and a 99th percentile of 8.6 Å, and at the disulfide sulfurs
  0.05 and 1.5 Å. Summarize shifts by medians, or read them at atoms whose
  density is a clear peak, such as sulfur and oxygen.
- **Overstated along a bond at coarse resolution.** The curvature divided by
  is that of PDB-Redo's whole model, so a bonded neighbour flattens it along
  the bond, and a neighbour that has moved too adds its own slope. Both grow
  as the resolution coarsens. Data without noise were made from PDB-Redo's
  models of the 64 entries of the ten dose series with disulfides, with
  each disulfide sulfur moved 0.05 Å outward along its S–S axis, scaled
  shell by shell to the amplitudes of the entry's own `FC_ALL`, and read by
  the engine. The median outward shift was 1.00 to 1.08 times the true one
  in the 30 entries at 1.25 Å or finer, 0.77 to 1.41 times in the six from
  1.25 to 1.75 Å, and 1.35 to 2.58 times in the 28 coarser ones, up to
  2.3 Å. Compare shifts within bins of resolution.
- **A slope is not a loss.** The same data were made with each disulfide
  sulfur's occupancy lowered by a fifth, in place. That left a hole of
  −0.27 to −2.8 e/Å³ at the sulfurs and little slope along the S–S axis, of
  either sign: as much as moving the sulfurs by −0.010 to +0.019 Å would
  give in the same entry. A neighbour's hole does slope the map at an atom,
  though: the beta carbons beside the depleted sulfurs had a slope pointing
  away from them in 47 of the 64 entries.
- **An estimate, not a refined position.** It is a first-order step from one
  point, it ignores the weighting of the coefficients, and a real
  displacement comes with a loss of density that the step does not model.

### Reading the values

- **Sigma depends on resolution.** A sharper map has higher peaks, and a hole of
  a given depth stands further out of a quieter difference map. Compare values
  within bins of resolution, or on the scale of the map coefficients through
  `density_fofc_rms`.
- **Sigma depends on the element.** A heavier atom has a higher peak and
  leaves a deeper hole for the same share lost. Compare one element with
  another through `density_fofc_eA3_pdb_redo` over `density_fc_eA3_pdb_redo`
  (see [Calculated density](#calculated-density)), and not through a count of
  atoms below a value in sigma.
- **PDB-Redo's maps describe PDB-Redo's model.** PDB-Redo re-refines B-factors
  and may rebuild side chains, flip peptides, and change waters. A
  difference-map value says how that model fits the data, which is not always
  how the deposited model would. The wwPDB validation maps say how the
  deposited model fits, and the two sources read together show whether a value
  depends on the re-refinement.
- **Refinement absorbs part of the signal.** An atom that is partly absent is
  given a higher B-factor by refinement, which flattens its hole. A negative
  value is therefore a lower bound on what was lost.
- **One conformer.** The atom table keeps one conformer of each residue, so a
  second conformer of an atom has no row and no value.
- **Coverage.** An entry a source has no maps for has no values from it. Both
  sources need deposited structure factors, so older entries are covered less
  often, and each covers some entries the other does not.
- **Twinned entries.** PDB-Redo's map coefficients for a twinned refinement
  follow another convention. Its difference coefficients are mFo−DFc at every
  reflection, half those of an untwinned refinement, and all of its
  coefficients are on a scale of their own. The engine transforms them as
  given. A value in sigma is unchanged by either, but `density_fofc_rms` and
  `density_fc_eA3` of such an entry are not comparable with those of others.
  `density_fofc_eA3` over `density_fc_eA3` is.

### What is not computed

Residue-level statistics such as EDSTATS' RSZD are not computed. They are taken
over every grid point within the atoms' radii with a normalization of their
own, so values read at atom centres do not give them, and an approximation
under the same name would mislead. wwPDB's residue-level RSCC and RSR are
available through [wwPDB validation](#wwpdb-validation). The engine calculates
no map coefficients of its own. It reads those PDB-Redo and the wwPDB
validation pipeline calculated.

No ratio of the difference density to the calculated density is written, for
the reason given under [The share the data do not
support](#the-share-the-data-do-not-support). The atom's own peak is not
calculated either. It would need the element, B-factor, and occupancy of each
atom of PDB-Redo's model and the scale Refmac applied, which is a model of the
density and not a reading of it.

The wwPDB validation maps give no calculated density. Their files hold the two
sets of map coefficients alone. Taking one from the other would give the
calculated map only if the multiple of the difference coefficients were known,
and the files do not record it. Against structure factors calculated from the
deposited model, four of six entries fitted `pdbx_FWT` less `pdbx_DELFWT` and
two fitted `pdbx_FWT` less twice `pdbx_DELFWT` better.

## Metal coordination

The metal-site engine is optional and off by default. Enable it with
`--include-metal-sites`. It applies to every technique. It writes one row per
metal atom to the `metal_sites` table, one row per bond to the
`metal_ligand_bonds` table, and one row per pair of nearby metal atoms to the
`metal_metal_distances` table, whose columns are listed under [Metal
sites](dataset-schema.md#metal-sites) in the dataset schema.

X-rays reduce redox-active metal centres early, at doses far below those that
damage the rest of a protein, and reduction lengthens metal–ligand bonds and
can remove bound small ligands. The tables record each bond's length beside a
literature distance, so that cohorts can be compared, and make no judgment of a
site.

The method is that of the coordination module of
[Alchemy](https://github.com/snell-lab/alchemy), carried over without its
classifications and scores. Two things differ. The engine reads the deposited
model, where Alchemy reads PDB-Redo's, so that neutron and X-ray entries are
measured by the same code. And it takes the precision index from the deposited
file's own refinement statistics.

### Metal atoms

A metal atom is any atom whose element is one of 84 metallic elements, with
coordinates and an occupancy that is not zero. Each is a site, in the conformer
the atom table keeps. Its `parent_type` is `heme` or `cluster` when the bundled
catalog of metal-containing components lists its component as such, `ion` when
the metal is the component's only non-hydrogen atom, and `other` otherwise.

### Finding donors

Model by model, the atoms are searched with gemmi's neighbour search, which
also finds atoms of symmetry mates. Around each metal the search collects the
nitrogen, oxygen, and sulfur atoms of water and of the 20 standard amino acids
within 4.0 Å, outside the metal's own component. Hydrogen and deuterium atoms
are never donors, so a neutron model is read as an X-ray model is. Atoms at
zero occupancy are not donors either. Two images of one atom within 0.8 Å of
each other are one atom on a special position, and the nearer is kept.

A candidate becomes a bond in two ways, and `detection_method` says which.

- **By distance.** The distance search infers a bond when the atom is of a kind
  that commonly binds metals and lies within the first coordination sphere, the
  target distance for the pair plus 0.75 Å (Harding, 2004). The kinds are water
  oxygen, the backbone carbonyl oxygen, the side-chain donors of Asn (OD1), Asp,
  Cys, Gln (OE1), Glu, His, Lys, Met, Ser, Thr, and Tyr, the backbone nitrogen
  of the first residue of a polymer sequence, and a terminal oxygen. Amide,
  guanidinium, indole, and peptide nitrogens are left out.
- **By record.** A deposited `_struct_conn` record of type `metalc` makes a
  bond of the atom it names, whatever its element, residue, or distance, since
  the depositor asserted it. A record whose partners take different symmetry
  operators is matched to the nearest image of the donor. A record between two
  metals, or within one component, names no donor and is passed over.

### Reference distances

The bundled table gives a literature distance and its spread for a metal and a
donor, from Harding (2006) for every metal but nickel and from Zheng et al.
(2008) for nickel. It covers ten metals, Na, Mg, K, Ca, Mn, Fe, Co, Ni, Cu, and
Zn, and water oxygen, carboxylate oxygen, backbone carbonyl oxygen, histidine
nitrogen, and cysteine sulfur, with hydroxyl oxygens added from statements in
Harding (2006).

A bond's target distance is the table's own row for the pair (`exact`), or else
the longest row for the same metal and donor element (`element_fallback`),
which decides first-sphere membership only. A pair with neither has no target
(`missing`), so the distance search infers nothing for it. A metal outside the
ten therefore gets bond rows from deposited records only.

A bond with an exact reference gets `distance_deviation_A`, its distance less
the reference distance, and `distance_z`:

```text
z = (d − d_ref) / sqrt(DPI² + sd_ref²)
```

DPI is the diffraction-component precision index of the entry, by equation 7 of
Blow (2002):

```text
DPI = 1.28 · Ni^(1/2) · Va^(1/3) · nobs^(−5/6) · Rfree
```

Ni is the occupancy-weighted count of non-hydrogen atoms of the first model,
with a kept alternate conformer counted whole. Va is the unit-cell volume
divided by the number of space-group operations. nobs and Rfree are the first
`_refine.ls_number_reflns_obs` and `_refine.ls_R_factor_R_free` the file gives
together. One index stands for the donor's positional uncertainty, with the
metal taken as well ordered. An entry without one of the terms has no index,
and its bonds keep their deviations without a z (`no_dpi`).

### Reading the tables

- **The reference is per element and donor, not per oxidation state.** A
  reduced site reads as a deviation, not as "reduced". Compare the same kind of
  bond across cohorts, with redox-inactive metals as the control.
- **Refinement restraints pull distances toward ideal values**, more so at
  lower resolution, and link records in the deposited file may have been used
  as restraints. Compare within bins of resolution.
- **The coordination number is partial for cofactors.** Atoms of a metal's own
  component are never donors, so the four pyrrole nitrogens of a heme are not
  counted, and neither are the sulfides of a cluster.
- **Small ligands are found only when declared.** Dioxygen, peroxide, carbon
  monoxide, and other non-protein ligands become bond rows only through a
  deposited record, and their loss is not measured.
- **`no_donors_found` is not an uncoordinated metal.** It says that no
  supported donor was found and no record named one.
- **Distances within a cluster are in `metal_metal_distances`.** Every pair of
  metal atoms of one model within 6 Å is a row there, among the atoms of the
  file, with no bond asserted. The metals of a manganese or iron–sulfur cluster
  move apart when the cluster is reduced, and no bond row shows it, since the
  atoms of a metal's own component are never donors.
- **A cluster can be deposited twice over.** Some entries model a cluster as
  two overlapping components, each at part occupancy, to describe two states.
  A copy of every metal then lies within a fraction of an ångström of its
  counterpart, so `nearest_metal_distance_A` is the distance to the atom's own
  copy. Keep the rows of `metal_metal_distances` whose `same_component` is true
  to read each copy by itself.
- **Strict non-crystallographic symmetry is not applied.** Operators a file
  declares without giving the copies are ignored, so contacts between such
  copies, mostly in virus capsids, are missing.
- **A neutron B-factor** is refined against nuclear density and is not directly
  comparable with an X-ray one.

## Residue sites

The residue-site engine is optional and off by default. Enable it with
`--include-residue-sites`. It applies to every technique. It writes one row per
radiation-sensitive group of a residue, and per control group, to the
`residue_sites` table, whose columns are listed under [Residue
sites](dataset-schema.md#residue-sites) in the dataset schema.

After disulfides, the side chains that X-rays damage first are the carboxylates,
which leave as carbon dioxide, then the sulfur of methionine and the hydroxyl of
tyrosine (Burmeister, 2000; Weik et al., 2000). The sulfur of every cysteine is
read as well, in a disulfide or not. RABDAM's Bnet is computed over
the carboxyl oxygens for that reason. Those are the sites of cryo
crystallography. In water at room temperature, and at the low doses of a
solution experiment, damage is done mostly by the hydroxyl radical and the
hydrated electron of water radiolysis, so the indole of tryptophan and the
imidazole of histidine, two of their fastest targets, are read too. The table records what the model holds at
each such group. What the data show at the same atoms is in the atom table,
where the [density engine](#electron-density-at-atoms) writes the map density of
every atom, and `residue_site_id` joins the two.

### Groups and controls

A group is the set of atoms that the damage removes, and its reference atoms are
those it is bonded to, which stay.

| `site_class` | Residue | Site atoms | Reference atoms | Control for |
|---|---|---|---|---|
| `asp_carboxylate` | Asp | CG, OD1, OD2 | CB | |
| `glu_carboxylate` | Glu | CD, OE1, OE2 | CG | |
| `c_terminal_carboxylate` | any standard amino acid with an OXT | C, O, OXT | CA | |
| `met_thioether` | Met | SD | CG, CE | |
| `tyr_hydroxyl` | Tyr | OH | CZ | |
| `cys_sulfur` | Cys | SG | CB | |
| `trp_indole` | Trp | CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2 | CG | |
| `his_imidazole` | His | ND1, CD2, CE1, NE2 | CG | |
| `cys_sulfenic_acid` | CSO | OD | SG | |
| `cys_sulfinic_acid` | CSD | OD1, OD2 | SG | |
| `cys_sulfonic_acid` | OCS | OD1, OD2, OD3 | SG | |
| `met_sulfoxide` | SME, MHO | OE; OD1 | S; SD | |
| `met_sulfone` | OMT | OD1, OD2 | SD | |
| `trp_hydroxylated` | TRO, 4HT, HRP, 4PQ, TRX, 0AF, TOQ, RE0, RE3 | the added oxygen or oxygens | the ring carbon each is on | |
| `trp_hydroperoxide` | TOX | O1, O2 | NE1 | |
| `trp_kynurenine` | KYN | O2 | C1 | |
| `his_oxo` | OHI, HOO | O12; OAG, OAM | CE1; CE1, CD2 | |
| `tyr_dopa` | DAH | OE2 | CE2 | |
| `phe_hydroxylated` | MTY | OH | CE2 | |
| `asn_amide` | Asn | CG, OD1, ND2 | CB | `asp_carboxylate` |
| `gln_amide` | Gln | CD, OE1, NE2 | CG | `glu_carboxylate` |
| `phe_para_carbon` | Phe | CZ | CE1, CE2 | `tyr_hydroxyl` |
| `phe_ring_carbons` | Phe | CD1, CD2, CE1, CE2 | CG | `his_imidazole` |
| `leu_delta_carbons` | Leu | CD1, CD2 | CG | `met_thioether` |
| `ile_delta_carbon` | Ile | CD1 | CG1 | `met_thioether` |
| `thr_hydroxyl` | Thr | OG1 | CB | `tyr_hydroxyl` |
| `ser_hydroxyl` | Ser | OG | CB | `cys_sulfur` |

Decarboxylation removes the carboxyl carbon with its two oxygens, so the
carboxyl carbon is a site atom and the bond that attaches the group is the one
that breaks. The amides of Asn and Gln have the same shape and do not
decarboxylate. No standard residue has the shape of Tyr or Met without the
sensitive atom, so their controls are approximate: the ring carbon of Phe where
Tyr carries its oxygen, and the delta carbons of Leu and Ile, which sit where
the sulfur of Met does.

A control of another element is a weak one. The same error of a model leaves a
deeper hole in the density at a heavier atom, so a sulfur set against a carbon
differs from it in any map, damaged or not. Tyrosine therefore has a second
control of its own element, the hydroxyl of Thr, an oxygen on a carbon as its
own is, though on a chain and not a ring. Whether tyrosine loses its hydroxyl at
all is disputed (Bury et al., 2017), and whether a hydroxyl on a chain is
untouched by X-rays is not established, so read Thr and Ser as comparisons and
not as a zero. No such control exists for sulfur.
The hydroxyl of Ser, which is Cys with an oxygen in place of the sulfur, has the
shape of the cysteine sulfur and not its element.

The rings of Trp and His do not leave. The hydroxyl radical adds to them,
one-electron oxidation takes an electron from them, and the products keep the
ring or open it: an oxygen added at CD1 or at a carbon of the benzene ring of
Trp, a hydroperoxide on NE1, the pyrrole opened to N'-formylkynurenine and
kynurenine, and an oxygen added at CE1 of His. No one atom carries the change,
so each group is every ring atom beyond CG, the atom that joins the ring to the
chain. The control for His is the four ring carbons of Phe beyond CG, an
aromatic ring on the same atom with carbon where His has its nitrogens. CZ
stays the control for Tyr, in a row of its own. Trp has no control. No standard
residue has a fused aromatic ring, and the other aromatic rings are targets
themselves. Its carbons can be set beside the ring carbons of Phe, and its
nitrogen beside another nitrogen, from the atom table. His is the commonest
protein ligand of a metal, so `metal_bound` matters most for it.

Phe is not a group. The hydroxyl radical adds to its ring a little faster than
to that of His (6.5 × 10⁹ against 5.0 × 10⁹ per molar per second), but the
hydrated electron reacts with it at half the rate of Trp and Tyr and a
hundredth that of Cys, and cryo crystallography reports no damage to it. It
stays a control, and that makes it the residue that tells the two chemistries
apart: what hydroxyl radicals do should reach Phe as it reaches His, and what
electrons do should reach neither much. The rate constants are in a reference
table of their own (see [Radiolysis rate constants](#radiolysis-rate-constants)).

A cysteine is a row whether or not it is in a disulfide. The `disulfides` table
says which, and it measures a bond over every conformer of its two cysteines,
where the row here sets the sulfur beside its CB in the one conformer the atom
table keeps.

A row is named by its key atom, the first reference atom. A residue cut back to
that atom is therefore still a row, with no site atoms modelled. A residue
without it has no row and is counted in the run report's
`residue_site_without_key_atom_count`.

A cysteine oxidized at its sulfur is deposited as a residue of its own, with
one, two, or three oxygens on the sulfur. Its oxygens are the group and its
sulfur is what they hang from. X-rays are reported both to reduce a sulfenic
acid and to oxidize a cysteine, so these rows are read in either direction: how
often a cohort models them, and how well the oxygens hold where it does. They
are modified residues, deposited as HETATM records with a place in the polymer
sequence, and are read as such. They have no control.

A methionine, tryptophan, histidine, tyrosine, or phenylalanine deposited
oxidized is read the same way: the oxygens it gained are the group, and the
atom each hangs from is what they are attached to. A kind of product deposited
under more than one code has one class for all of them, and the row's key atom
says where on the ring the oxygen was added, so a `trp_hydroxylated` row keyed
on CZ2 is 7-hydroxytryptophan. Methionine sulfoxide is deposited as SME, with
its sulfur named S, and as MHO. Kynurenine is named afresh in the dictionary:
its C1 is the carbon that was CG of the tryptophan, and O2 the oxygen it gained
when the pyrrole opened. meta-Tyrosine, which the dictionary gives Tyr as its
parent, is Phe with an oxygen on CE2. The codes and atom names are those of the
wwPDB Chemical Component Dictionary.

Enzymes make some of the same products, 7-hydroxytryptophan and DOPA among
them on the way to protein cofactors, and they are deposited under the same
codes, so a row says that a model holds the product and not how it formed.
N'-formylkynurenine is in the dictionary only as a free molecule, NFK, and is
not read. 3-hydroxyhistidine, HSK, carries its oxygen on a ring nitrogen, which
radiolysis is not reported to give, and is not read either.

### Every atom against its parent

The engine also writes one atom column apart from the table,
`b_factor_ratio_to_parent`: each atom's B-factor divided by the mean B-factor
of its parents, the atoms it is bonded to one bond nearer the CA. It is given
for every atom of the 20 standard amino acids but the CA, from a fixed table of
their bonds. The N, the C, and the CB hang from the CA, the O and the OXT from
the C, and each side-chain atom from the one before it. An atom that closes a
ring has two parents, as the CZ of Phe has CE1 and CE2.

It is the ratio the table takes for its named groups, taken for every atom, so
that an analysis can ask which atoms move with dose without naming any in
advance. A B-factor rises along any side chain, so a ratio above one means
little by itself. Compare one atom type across structures or doses. The ratio
depends on how B-factors were refined, so compare within one refinement program
and one B-factor model where the dataset records them.

### Radiolysis rate constants

The engine ships a reference table,
`pdbai/engines/analyses/residue_sites/radiolysis_rate_constants.csv`, of the
rate constants of each standard amino acid in water with the hydroxyl radical
(`OH`) and the hydrated electron (`e_aq`), the two radicals of water radiolysis
that do most of the damage in solution. Every value was read from the page
images of Buxton et al. (1988), the standard compilation, with its pH, its
method, the form of the amino acid it was measured on, and the table, entry,
and page it is printed on. A residue has a row for each value printed, and one
row per amino acid and radical is selected: Buxton's average for the form that
dominates at pH 7 where he prints one, and otherwise the value nearest pH 7.
Upper limits and approximate values keep their qualifier in `k_qualifier`.
Cystine is listed under CYS and is never the selected row. The source has no
value for the hydrated electron with Gln or Ile, and none is filled in.

The table is not a set of columns on `residue_sites`. A rate constant belongs
to a free amino acid in a stated solution, not to a group in a crystal, it is
one value per residue type, which a join on `res_name` gives any table, and it
carries a pH and a method no row could hold. Its manifest,
`reference_data.manifest.json`, records the source, how it was read, what was
left out, and the file's SHA-256 digest, and every run that executes the engine
records a receipt of the table in its manifest. Read the values with
`pdbai.engines.analyses.residue_sites.reference.selected_rate_constants()`.

Read them as solution chemistry. They describe a free amino acid, its amino and
carboxyl groups charged, at room temperature, and the hydroxyl radical reacts
with the whole molecule, its chain as well as the group the `residue_sites`
table names. The rate constant of His with the hydrated electron rises
fortyfold when the imidazole is protonated, below pH 6.

### Reading the residue sites

- **A row is read in the conformer the atom table keeps.** A group modelled in
  two conformers shows the one of higher occupancy, and
  `has_alternate_conformer` says that there was another.
- **Missing atoms are not proof of damage.** A side chain is cut back for
  disorder as often as for damage, and surface carboxylates are often
  disordered. Compare a sensitive class with its control, which is cut back for
  disorder alone.
- **A C-terminus is seen only where the file models an OXT.** The last modelled
  residue of a chain with a disordered end is not a terminus, and a C-terminal
  carboxylate modelled without its second oxygen is not found.
- **The amide controls have a baseline of their own.** An Asn or Gln modelled
  with its oxygen and nitrogen exchanged puts difference density at both atoms.
- **`metal_bound` sets a subgroup apart.** A carboxylate bound to a metal may be
  protected. Metal atoms of a symmetry mate are not seen.
- **Bond lengths are restrained in refinement**, so `attachment_distance_min_A`
  is read with the other measures and within bins of resolution.
- **Only residues deposited as ATOM records are read**, which leaves out a free
  amino acid bound as a ligand, and a modified residue under another name. The
  oxidation products named above are the exception.
- **A ring is a group of mixed elements.** The lowest difference density over
  the eight atoms of an indole and over the four of a ring of Phe are not on
  one footing, in number of atoms or in electrons. Compare the atoms of one
  element, a ring carbon with a ring carbon, through `residue_site_id` and the
  atom table.
- **A neutron B-factor** is refined against nuclear density and is not directly
  comparable with an X-ray one.

## Selenium and halogens

The selenium and halogen engine is optional and off by default. Enable it with
`--include-selenium-halogens`. It applies to every technique. It writes one row
per selenium or halogen atom to the `selenium_halogen_sites` table, whose
columns are listed under [Selenium and
halogens](dataset-schema.md#selenium-and-halogens) in the dataset schema.

These atoms absorb X-rays far more strongly than the light atoms of a protein,
and their bonds to carbon are among the first that radiation breaks: C–Br in
brominated nucleotides (Ennifar et al., 2002), C–Br and C–I in halogenated
ligands (Rodrigues et al., 2024), and Se–C in selenomethionine (Holton, 2007).
A structure collected with femtosecond X-ray pulses may show electronic damage
at the same atoms. Mercury, platinum, gold, and the other heavy metals are rows
of the [`metal_sites`](#metal-coordination) table instead.

Chlorine and fluorine are read with them, although neither is heavy or absorbs
strongly. The bond to carbon weakens from fluorine to iodine, so the four
halogens form one series, and the carbon–fluorine bond, the strongest, is the
comparison the others are set against. Chlorine is also far more common than
bromine or iodine, on ligands and as the chloride of a buffer.

### Sites and bonded atoms

A selenium or halogen atom is a row when it has coordinates, whatever its
occupancy, since an
atom deposited at zero occupancy is one a depositor modelled as lost. Its
`parent_type` is `ion` when it is its component's only non-hydrogen atom, as in
a chloride of the buffer or the bromide or iodide of a soak, `selenomethionine` in MSE, `polymer_residue` in any
other component with a place in a polymer sequence, and `ligand` otherwise.

Two atoms are bonded when their distance is at most the sum of their covalent
radii plus 0.45 Å. The radii are gemmi's, those of Cordero et al. (2008), which
put C–F at 1.30 Å, C–Cl at 1.75 Å, C–Se and C–Br at 1.93 Å, and C–I at 2.12 Å,
and the tolerance is the one Open
Babel uses to perceive bonds from distances (O'Boyle et al., 2011). A bonded
atom is not hydrogen, deuterium, or a metal, and has an occupancy that is not
zero. Bonds are found among the atoms of the file, across components as well as
within one.

The row sets the atom beside two references within its own structure, the
atoms it is bonded to and the other atoms of its component that are not
themselves selenium or a halogen.

### Reading the selenium and halogen sites

- **A lost atom shows as occupancy or as B-factor.** A depositor who saw
  the loss lowered the occupancy below `component_occupancy_mean`. One who did
  not left a B-factor raised above its neighbours', or a hole in the difference
  map, which the density columns of the atom table hold.
- **Selenomethionine compares with methionine.** `b_factor_ratio_to_bonded` of
  an MSE row and `b_factor_ratio_to_reference` of a `met_thioether` row of
  `residue_sites` set the selenium and the sulfur against the same two carbons.
- **An ion has no bonded atom**, so its row says `no_bonded_atom` and its
  component references are missing. Its occupancy is often a fraction for
  reasons other than damage, since soaked halides bind weakly, and
  `on_special_position` marks the ones a symmetry element halves.
- **A bond refined longer than the limit is not found.** The row then says
  `no_bonded_atom` although the atom belongs to a larger component, which
  `parent_type` and `component_atom_count` show.
- **A heavier atom shows a deeper hole for the same error.** Compare the
  halogens after allowing for their electrons, or each against the rest of its
  own ligand, and not by a count of holes alone.
- **Most rows are chloride ions.** Chloride is in most crystallization
  buffers. Read the atoms bonded to carbon by `bonded_element`, and set ions
  apart by `parent_type`.
- **A halogen bound only to a metal has no bonded atom**, as the chlorides of a
  platinum complex have none, since a metal is never counted as bonded.

## Cofactors

The cofactor engine is optional and off by default. Enable it with
`--include-cofactors`. It applies to every technique. It writes one row per
recognized cofactor to the `cofactor_sites` table, whose columns are listed
under [Cofactors](dataset-schema.md#cofactors) in the dataset schema.

A cofactor absorbs the electrons that X-rays free in a crystal, so it is
reduced, and a chromophore can be changed as light would change it. The table
records the geometry where that shows, with the occupancy and B-factors of the
same atoms, and makes no judgment of a site.

### Recognizing a cofactor

A component is of a class when it holds every atom name of the class's
signature, the names the wwPDB chemical component dictionary gives the atoms of
its reactive group. An analogue or an adduct deposited under a component name of
its own is therefore read like the common one. Two classes are limited to one
component, where the names alone would not tell the cofactor from other
molecules. The 20 standard amino acids are never tried.

| `cofactor_class` | Recognized by | Key atom | Reactive group |
|---|---|---|---|
| `flavin` | N5, N10, C9A, C10, C7M, C8M | N5 | The 14 atoms of the isoalloxazine ring |
| `tetrapyrrole` | NA, NB, NC, ND, CHA, CHB, CHC, CHD | NA | The 24 atoms of the macrocycle |
| `fp_chromophore` | C1, N2, N3, C2, CA2, CB2, CG2, in a polymer residue | CB2 | The imidazolinone ring and the bridge, those 7 atoms |
| `retinal` | Component RET | C15 | The chain from C5 to C15 |
| `coumaric_acid` | Component HC4 | C3 | The 9 carbons |

Among 17,725 archive entries read while the engine was written, the flavins
found this way were FAD, FMN, FDA, riboflavin, and four analogues, and the
tetrapyrroles were the hemes, the chlorophylls and bacteriochlorophylls, the
pheophytins, siroheme, coenzyme F430, and their analogues. FAD names two ring
carbons C4X and C5X, since C4A and C5A are atoms of its adenine, and FMN names
them C4A and C5A. The engine reads either.

### Measures

- **Flavin.** `flavin_butterfly_angle_deg` is the angle between the
  least-squares planes of the pyrimidine ring (N1, C2, N3, C4, C4A, C10) and the
  benzene ring (C5A, C6, C7, C8, C9, C9A). An oxidized flavin is flat, and a
  reduced one bends about its N5–N10 axis (Røhr et al., 2010).
- **Tetrapyrrole.** `metal_out_of_plane_A` is the distance of the central metal
  from the least-squares plane of the four pyrrole nitrogens, which changes
  with the spin and oxidation state of a heme iron (Perutz, 1970).
  `central_metal_site_id` names that metal, so the row joins the metal's row of
  `metal_sites`, which holds its bonds to the protein. A pheophytin has no
  metal, and its metal columns are missing.
- **Fluorescent-protein chromophore.** `chromophore_tau_deg` and
  `chromophore_phi_deg` are the torsions N2–CA2–CB2–CG2 and CA2–CB2–CG2–CD1
  about the two bonds of the bridge between the rings, with CD2 where a
  component has no CD1. X-rays take the bridge carbon out of the plane (Adam et
  al., 2009).
- **Retinal.** The torsions about the C11=C12 and C13=C14 bonds, which light
  isomerizes in visual and in microbial rhodopsins, and the distance from C15
  to the lysine NZ of the Schiff base. X-rays change the same region
  (Borshchevskiy et al., 2014).
- **p-Coumaric acid.** The torsion about the C2=C3 bond, which light isomerizes
  in photoactive yellow protein (Genick et al., 1998), and the distance from C1
  to the cysteine SG of the thioester.
- **Every class.** `group_planarity_rmsd_A`, and the occupancy and B-factors of
  the group beside the rest of the component.

### Reading the cofactors

- **Planarity is restrained in refinement.** A flavin refined against a flat
  dictionary stays flat whatever its state, more so at lower resolution, so
  compare within bins of resolution and prefer high-resolution entries.
- **Light and X-rays act on the same atoms.** A time-resolved entry holds an
  intermediate on purpose. Set such entries apart with `mentions_time_resolved`
  before reading a torsion as damage.
- **Hemes and chlorophylls of every kind are one class.** `res_name` and
  `central_metal_element` tell them apart.
- **A torsion's sign follows atom names.** `chromophore_phi_deg` differs by 180°
  between a ring modelled one way and flipped, and between a component with a
  CD1 and one with only a CD2.
- **The link distance asserts no bond.** It is the nearest lysine NZ or cysteine
  SG within 4 Å among the atoms of the file, as measured.
- **Bilins, quinones, nicotinamides, and iron–sulfur clusters are not read.**
  The metals of a cluster are rows of `metal_sites`.

## Site environment

The site-environment engine is optional and off by default. Enable it with
`--include-site-environment`. It applies to every technique. It writes one row
per group of the [residue-site](#residue-sites) engine to the
`site_environment` table, whose columns are listed under [Site
environment](dataset-schema.md#site-environment) in the dataset schema, and
the depth of every protein atom to the atom table.

Two groups of one residue type in one protein can lose density at different
rates. What sets them apart is what surrounds them, and a model that predicts
damage from a structure needs those surroundings as features. The engine finds
its groups with the residue-site engine's own function, so its rows carry the
same `site_id`.

### What is measured

Each model is read on its own, in the conformer the atom table keeps. A group's
atoms are its site atoms with coordinates and an occupancy that is not zero.
Atoms of the group's own residue are never its partners.

- **Protein atoms** are the heavy atoms of standard amino acids and of the
  modified residues of a polymer, at an occupancy that is not zero. They alone
  give the packing counts, the solvent-accessible area, and the depth, so those
  are the same for a deposited model and for a predicted model of the same
  coordinates.
- **Hydrogen bonds** are counted by distance alone: a nitrogen or oxygen of a
  partner within 3.5 Å of a nitrogen or oxygen of the group (Baker & Hubbard,
  1984), or within 4.0 Å of a sulfur, whose hydrogen bonds are longer (Zhou et
  al., 2009). Without hydrogen positions or angles, the count holds every
  possible hydrogen bond and some that are not.
- **Salt bridges** of a carboxylate are the Lys, Arg, and His residues with a
  charged-group nitrogen within 4.0 Å of one of its oxygens (Barlow & Thornton,
  1983; Kumar & Nussinov, 2002). Whether a His is charged is not judged.
- **Packing** is the count of protein atoms within 6 and 10 Å of the group's
  centroid. The larger radius is about what half-sphere exposure counts
  (Hamelryck, 2005).
- **Solvent-accessible area** is computed with the SASA engine's Shrake–Rupley
  routine, radii, and probe, at 200 test points per atom, among the protein
  atoms of the model. The relative area divides it by the area of the same
  atoms with their own residue alone.
- **Depth** of an atom is its distance to the nearest protein atom with at least
  1 Å² of accessible area, zero at the surface. A group's depth is the mean over
  its atoms.
- **Crystal contacts** come from the images of the model's non-water atoms
  under every operation of the space group and every lattice translation that
  can bring them within 15 Å of the model, the model itself left out. They are
  missing for an entry without a usable cell and space group, and
  `crystal_contacts_status` says so.

### Reading the site environment

- **Missing means absent.** A distance to a metal is missing when the model has
  no metal atom, and a water count when it has no water. A model of a protein
  alone, such as a predicted one, has no ligand, water, or crystal contact.
- **A symmetry mate is any image.** The images that build the biological
  assembly are mates too, so a contact may be an interface of the assembly as
  well as one of the crystal.
- **The asymmetric unit is the model.** Packing, area, and depth see the chains
  of the file, not the assembly, and a chain whose partner in the assembly is a
  symmetry image is read as more exposed than it is.
- **How many waters a depositor models** depends on the resolution and the
  depositor, so the water columns compare groups within one structure or
  between structures of like resolution.

 wwPDB-format validation XML reports. PDB-AI can parse these
reports into residue-level density statistics and geometry classes, and
structure-level geometry metrics. Enable acquisition and output with
`--include-validation`.

Each residue gets its real-space correlation coefficient (`eds_rscc`) and
real-space R factor (`eds_rsr`), together with the report's Ramachandran class
(`rama_class`) and side-chain rotamer (`rotamer_class`, a rotamer name or
`OUTLIER`). Each structure gets the MolProbity clashscore and the percentages of
Ramachandran and rotamer outliers.

### Fetch and caching

The validation XML is downloaded from:

```
https://files.rcsb.org/pub/pdb/validation_reports/{mid}/{pdb_id}/{pdb_id}_validation.xml.gz
```
where `{mid}` is the 2nd and 3rd characters of the lower-case PDB ID (e.g. `mc`
for `5mcc`). The gzip-compressed XML is decompressed in memory and parsed with
`xml.etree.ElementTree`. Parsed results are cached in a
`{pdb_id}_validation.json` envelope in the `mmcif/` directory for 30 days.
`run_report.csv`, `run_summary.md`, and the run manifest expose the acquisition
outcome as `available`, `cached`, `not_available`, `fetch_failed`, or
`parse_failed`. The CSV and manifest also retain the accompanying failure
message.

### Residue merging

Per-residue statistics are stored with the model-aware key
`"{model}:{chain}:{resnum}{icode}"` and merged to the atom-site DataFrame using
the coordinate model, the author chain, the author residue number, and the
insertion code, the same identifiers as the PDB validation report. These are
`chain_id` and `res_num`, except for a structure whose `residue_numbering` is
`label`. There they are read from `_atom_site.auth_asym_id` and
`_atom_site.auth_seq_id` alone, and an atom without them takes no value. A
residue the report lists once per alternate conformer is also stored under a
per-conformer key, `"{model}:{chain}:{resnum}{icode}:{altcode}"`. Its atoms take
the values of the conformer the structure parser kept, found by its alternate
location identifier. When the report does not list that conformer, they take the
residue key's values, which come from the conformer whose altcode is blank or A,
or else the first one listed.

### Interpreting density statistics

RSCC and RSR quantify how well the refined model matches the experimental
electron density within each residue's envelope. Radiation damage causes
specific chemical changes (disulfide bond cleavage, decarboxylation of Asp/Glu,
C–S bond cleavage in Cys/Met) that are invisible in the model but apparent in
the maps. Low RSCC at susceptible residues, especially when BDamage is also
elevated, is consistent with unmodeled damage. Mobility and disorder lower RSCC
as well, so neither value alone identifies damage.

## AlphaFold Database

Toggleable via `--include-alphafold` (off by default). The engine integrates
published predictions from AlphaFold Database. It does not run AlphaFold locally
and is intentionally independent of any one model-generation version.

Because the accession is required to interpret model availability,
`--include-alphafold` retains `uniprot_id` as provenance even when
`--include-uniprot` is not separately set. The standalone UniProt toggle remains
useful when you want aligned accessions without downloading AFDB models. Such a
run fetches no model, so the run report gives a mapped entry the `afdb_status`
`mapping_only`.

### Discovery, fetch, and caching

1. The RCSB entry endpoint is used to enumerate every polymer entity. Each
   entity endpoint supplies UniProt accessions plus `aligned_regions` containing
   the PDB entity start, UniProt reference start, and aligned length.
2. The AFDB prediction API is queried once per distinct mapped accession:
   `https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}`.
3. A coherent canonical `AF-{accession}-F<n>` prediction family is selected.
   Fragmented predictions are merged in deterministic fragment order, while
   unrelated isoforms, alternate datasets, and complexes are not mixed into one
   synthetic sequence. A prediction flagged `isComplex` is never selected. When
   the database has no canonical family for the accession, the single best
   prediction recorded for exactly that accession is used instead. Best means an
   identifier that contains the accession, then the widest span, then the
   earliest start.
4. ModelCIF is parsed with `gemmi`. A local metric in `_ma_qa_metric_local`
   counts as pLDDT when its `_ma_qa_metric` name, type, or description says so.
   When the file declares no metric at all, every local metric value is read as
   pLDDT. CA `B_iso_or_equiv` is used only when the local metrics give no score.
   Scores outside 0–100 and non-finite values are skipped. A file that gives one
   residue two different scores is rejected whole, because its numbering cannot
   be trusted. A fragment's scores stay in place when every position lies within
   its declared UniProt range, and are moved by the declared start when the
   fragment is numbered from 1. A fragment that fits neither is rejected as
   `parse_failed`, and the rest of its family is kept as `partial`.
5. Successful entity mappings and completely successful model results are cached
   under `mmcif/afdb/entities/` and `mmcif/afdb/models/`, then refreshed after
   30 days so newer AFDB releases can be discovered. Empty mapping results,
   network failures, partial results, missing models, and parse failures are not
   made permanent, so a later run retries them.

### Residue mapping and provenance

Mapping is keyed by `_atom_site.label_entity_id` and `_atom_site.label_seq_id`,
then transformed through each RCSB aligned region. This preserves offsets, gaps,
repeated sequence numbers in different entities, chimeric segments, and modified
polymer residues recorded as HETATM. Ambiguous overlapping mappings are left
unmapped instead of choosing silently. A residue gets a pLDDT only when the
AlphaFold model has the same amino acid at the mapped position. A modified
residue counts as its parent amino acid, so MSE matches MET. The parent comes
from Gemmi's residue table, corrected from the wwPDB Chemical Component
Dictionary where the table is wrong or silent, so pyroglutamate (PCA) matches
GLN. A residue without a name matches nothing, and UNK matches only UNK. An
engineered mutation, or a residue the model numbers differently because it was
built on another version of the sequence, therefore gets no score and lowers the
entity's pLDDT coverage (`afdb_plddt_coverage`). For fragmented predictions,
`afdb_model_id` records the versioned ModelCIF fragment that supplied each
residue.

The engine reports two separate facts: whether an AFDB model could be obtained
(`afdb_status`) and whether a particular observed residue received a score
(`afdb_residue_has_plddt`). The structure-level run report uses `available` only
when every discovered protein entity is mapped and every queried model is
usable. `partial` guarantees at least one usable model, while
`no_model_available` means no model gave a score and the models did not share
one failure status. A blank pLDDT therefore does not conflate “no UniProt
mapping,” “no AFDB model,” “network failure,” “parse failure,” and “residue
outside the aligned/modeled region.” Per-entity pLDDT coverage provides an
additional quality-control feature for ML filtering.

Direct mmCIF uploads do not have an RCSB entry lookup, so they receive the
explicit `not_available_for_upload` status instead of a misleading missing-model
result.

## References

- Gerstel, M., Deane, C. M. & Garman, E. F. (2015). **Identifying and
  quantifying radiation damage at the atomic level.** *Journal of Synchrotron
  Radiation* 22(2), 201–212. https://doi.org/10.1107/S1600577515002131. The
  BDamage metric.

- Shelley, K. L., Dixon, T. P. E., Brooks-Bartlett, J. C. & Garman, E. F.
  (2018). **RABDAM: quantifying specific radiation damage in individual protein
  crystal structures.** *Journal of Applied Crystallography* 51(2), 552–559.
  https://doi.org/10.1107/S1600576718002509. The Bnet metric and the RABDAM
  program.

- Shelley, K. L. & Garman, E. F. (2022). **Quantifying and comparing radiation
  damage in the Protein Data Bank.** *Nature Communications* 13, 1314.
  https://doi.org/10.1038/s41467-022-28934-0. The Bnet percentile, its
  resolution-matched reference, and RABDAM 2, which rabdam-custom rewrites.

- Shrake, A. & Rupley, J. A. (1973). **Environment and exposure to solvent of
  protein atoms.** *Journal of Molecular Biology* 79(2), 351–371.
  https://doi.org/10.1016/0022-2836(73)90011-9. Test-point SASA algorithm.

- Lee, B. & Richards, F. M. (1971). **The interpretation of protein structures:
  estimation of static accessibility.** *Journal of Molecular Biology* 55(3),
  379–400. https://doi.org/10.1016/0022-2836(71)90324-X. Rolling-probe
  definition of the solvent-accessible surface.

- Tien, M. Z., Meyer, A. G., Sydykova, D. K., Spielman, S. J. & Wilke, C. O.
  (2013). **Maximum allowed solvent accessibilities of residues in proteins.**
  *PLOS ONE* 8(11), e80635. https://doi.org/10.1371/journal.pone.0080635.
  Theoretical ALLOWED maximum ASA values used by the `Tien2013_ALLOWED_DSSP`
  RASA method.

- Alvarez, S. (2013). **A cartography of the van der Waals territories.**
  *Dalton Transactions* 42, 8617. https://doi.org/10.1039/C3DT50599E.
  Element-specific radii used for every non-protein occluder: ligands,
  cofactors, metals, non-protein polymers, and other non-polymers. Its sulfur
  radius of 1.89 Å is one basis of the disulfide search limit.

- Kabsch, W. & Sander, C. (1983). **Dictionary of protein secondary structure.**
  *Biopolymers* 22(12), 2577–2637. https://doi.org/10.1002/bip.360221211.
  Protein-accessibility radius convention paired with Tien normalization in the
  SASA engine.

- Weik, M., Ravelli, R. B. G., Kryger, G., McSweeney, S., Raves, M. L., Harel,
  M., Gros, P., Silman, I., Kroon, J. & Sussman, J. L. (2000). **Specific
  chemical and structural damage to proteins produced by synchrotron
  radiation.** *Proceedings of the National Academy of Sciences USA* 97(2),
  623–628. https://doi.org/10.1073/pnas.97.2.623. Disulfide-bridge breakage and
  carboxylate decarboxylation as specific radiation damage, in
  acetylcholinesterase.

- Ravelli, R. B. G. & McSweeney, S. M. (2000). **The 'fingerprint' that X-rays
  can leave on structures.** *Structure* 8(3), 315–328.
  https://doi.org/10.1016/S0969-2126(00)00109-X. Specific damage, including
  disulfide elongation and cleavage, in a dose series.

- Carpentier, P., Royant, A., Weik, M. & Bourgeois, D. (2010). **Raman-assisted
  crystallography suggests a mechanism of X-ray-induced disulfide radical
  formation and reparation.** *Structure* 18(11), 1410–1419.
  https://doi.org/10.1016/j.str.2010.09.010. The disulfide radical anion as the
  first product of electron capture, and its repair.

- Sutton, K. A., Black, P. J., Mercer, K. R., Garman, E. F., Owen, R. L., Snell,
  E. H. & Bernhard, W. A. (2013). **Insights into the mechanism of X-ray-induced
  disulfide-bond cleavage in lysozyme crystals based on EPR, optical absorption
  and X-ray diffraction studies.** *Acta Crystallographica Section D* 69(12),
  2381–2394. https://doi.org/10.1107/S0907444913022117. Disulfide radicalization
  followed by EPR, UV–visible spectroscopy, and diffraction in lysozyme.

- Bhattacharyya, R., Dhar, J., Ghosh Dastidar, S., Chakrabarti, P. & Weiss, M.
  S. (2020). **The susceptibility of disulfide bonds towards radiation damage
  may be explained by S⋯O interactions.** *IUCrJ* 7(5), 825–834.
  https://doi.org/10.1107/S2052252520008520. The intact S–S length of 2.02 Å,
  the radical anion's of about 2.8 Å, and the restraints that pull a modelled
  S–S distance back toward the ideal value.

- Engh, R. A. & Huber, R. (2001). **Structure quality and target parameters.**
  In *International Tables for Crystallography Volume F*, chapter 18.3, pp.
  382–392. IUCr / Kluwer. Restraint targets, which put the S–S distance near
  2.03 Å.

- Bondi, A. (1964). **van der Waals Volumes and Radii.** *Journal of Physical
  Chemistry* 68(3), 441–451. https://doi.org/10.1021/j100785a001. The sulfur van
  der Waals radius of 1.80 Å, so two sulfur atoms in contact lie about 3.6 Å
  apart.

- Harding, M. M. (2006). **Small revisions to predicted distances around metal
  sites in proteins.** *Acta Crystallographica Section D* 62(6), 678–682.
  https://doi.org/10.1107/S0907444906014594. Metal–ligand target distances, with
  Zn–S(Cys) about 2.33 Å, so two cysteines on one tetrahedral zinc lie about 3.8
  Å apart.

- Read, R. J. (1986). **Improved Fourier coefficients for maps using phases
  from partial structures with errors.** *Acta Crystallographica Section A*
  42(3), 140–149. https://doi.org/10.1107/S0108767386099622. The 2mFo−DFc
  coefficients, and why a difference map made with model phases shows a
  feature at about half its height for acentric reflections.

- Merritt, E. A. (1999). **Expanding the model: anisotropic displacement
  parameters in protein structure refinement.** *Acta Crystallographica Section
  D* 55(6), 1109–1117. https://doi.org/10.1107/S0907444999003789. The anisotropy
  A = λ_min / λ_max, the reciprocal of `b_anisotropy`.

- Joosten, R. P., Long, F., Murshudov, G. N. & Perrakis, A. (2014). **The
  PDB_REDO server for macromolecular structure model optimization.** *IUCrJ*
  1(4), 213–220. https://doi.org/10.1107/S2052252514009324. PDB-Redo re-refined
  mmCIF files, an optional comparison source and RABDAM's internal scoring
  input.

- Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O.,
  … Hassabis, D. (2021). **Highly accurate protein structure prediction with
  AlphaFold.** *Nature* 596, 583–589.
  https://doi.org/10.1038/s41586-021-03819-2. AlphaFold2 predicted structures,
  whose pLDDT confidence scores are fetched from the EBI AlphaFold Database.

- Varadi, M., Anyango, S., Deshpande, M., Nair, S., Natassia, C., Yordanova, G.,
  … Velankar, S. (2022). **AlphaFold Protein Structure Database: massively
  expanding the structural coverage of protein-sequence space with high-accuracy
  models.** *Nucleic Acids Research* 50(D1), D439–D444.
  https://doi.org/10.1093/nar/gkab1061. EBI AlphaFold Database API used to
  retrieve per-residue pLDDT scores.

- Williams, C. J., Headd, J. J., Moriarty, N. W., Prisant, M. G., Videau, L. L.,
  Deis, L. N., … Richardson, D. C. (2018). **MolProbity: more and better
  reference data for improved all-atom structure validation.** *Protein Science*
  27(1), 293–315. https://doi.org/10.1002/pro.3330. Ramachandran / rotamer /
  clashscore validation metrics distributed via the wwPDB validation XML
  reports.

- Blow, D. M. (2002). **Rearrangement of Cruickshank's formulae for the
  diffraction-component precision index.** *Acta Crystallographica Section D*
  58(5), 792–797. https://doi.org/10.1107/S0907444902003931. The precision
  index of the metal-site tables.

- Harding, M. M. (2004). **The architecture of metal coordination groups in
  proteins.** *Acta Crystallographica Section D* 60(5), 849–859.
  https://doi.org/10.1107/S0907444904004081. The first-sphere tolerance.

- Harding, M. M. (2006). **Small revisions to predicted distances around metal
  sites in proteins.** *Acta Crystallographica Section D* 62(6), 678–682.
  https://doi.org/10.1107/S0907444906014594. The reference metal–donor
  distances.

- Zheng, H., Chruszcz, M., Lasota, P., Lebioda, L. & Minor, W. (2008). **Data
  mining of metal ion environments present in protein structures.** *Journal of
  Inorganic Biochemistry* 102(9), 1765–1776.
  https://doi.org/10.1016/j.jinorgbio.2008.05.006. The reference distances for
  nickel.

- Burmeister, W. P. (2000). **Structural changes in a cryo-cooled protein
  crystal owing to radiation damage.** *Acta Crystallographica Section D* 56(3),
  328–341. https://doi.org/10.1107/S0907444999016261. Decarboxylation, and
  damage to methionine and tyrosine.

- Bury, C. S., Carmichael, I. & Garman, E. F. (2017). **OH cleavage from
  tyrosine: debunking a myth.** *Journal of Synchrotron Radiation* 24(1), 7–18.
  https://doi.org/10.1107/S1600577516016775. Whether tyrosine loses its
  hydroxyl.

- Buxton, G. V., Greenstock, C. L., Helman, W. P. & Ross, A. B. (1988).
  **Critical review of rate constants for reactions of hydrated electrons,
  hydrogen atoms and hydroxyl radicals (·OH/·O⁻) in aqueous solution.**
  *Journal of Physical and Chemical Reference Data* 17(2), 513–886.
  https://doi.org/10.1063/1.555805. The rate constants of the amino acids with
  the hydroxyl radical and the hydrated electron, read from its Tables 6 and 8.

- Ennifar, E., Carpentier, P., Ferrer, J.-L., Walter, P. & Dumas, P. (2002).
  **X-ray-induced debromination of nucleic acids at the Br K absorption edge and
  implications for MAD phasing.** *Acta Crystallographica Section D* 58(8),
  1262–1268. https://doi.org/10.1107/S0907444902009526. Cleavage of C–Br bonds.

- Holton, J. M. (2007). **XANES measurements of the rate of radiation damage to
  selenomethionine side chains.** *Journal of Synchrotron Radiation* 14(1),
  51–72. https://doi.org/10.1107/S0909049506048898. Cleavage of Se–C bonds.

- Rodrigues, M. J., Cabry, M., Collie, G., Carter, M., McAndrew, C., Owen, R.
  L., Bellenie, B. R., Le Bihan, Y.-V. & van Montfort, R. L. M. (2024).
  **Specific radiation damage to halogenated inhibitors and ligands in
  protein–ligand crystal structures.** *Journal of Applied Crystallography*
  57(6), 1951–1965. https://doi.org/10.1107/S1600576724010549. Cleavage of
  carbon–halogen bonds in ligands.

- Cordero, B., Gómez, V., Platero-Prats, A. E., Revés, M., Echeverría, J.,
  Cremades, E., Barragán, F. & Alvarez, S. (2008). **Covalent radii revisited.**
  *Dalton Transactions* (21), 2832–2838. https://doi.org/10.1039/b801115j. The
  covalent radii of the selenium and halogen bond search.

- O'Boyle, N. M., Banck, M., James, C. A., Morley, C., Vandermeersch, T. &
  Hutchison, G. R. (2011). **Open Babel: An open chemical toolbox.** *Journal of
  Cheminformatics* 3, 33. https://doi.org/10.1186/1758-2946-3-33. The bond
  tolerance of the selenium and halogen bond search.

- Røhr, Å. K., Hersleth, H.-P. & Andersson, K. K. (2010). **Tracking flavin
  conformations in protein crystal structures with Raman spectroscopy and QM/MM
  calculations.** *Angewandte Chemie International Edition* 49(13), 2324–2327.
  https://doi.org/10.1002/anie.200907143. The butterfly bend of a reduced
  flavin.

- Perutz, M. F. (1970). **Stereochemistry of cooperative effects in haemoglobin:
  haem–haem interaction and the problem of allostery.** *Nature* 228, 726–734.
  https://doi.org/10.1038/228726a0. The heme iron's distance from the plane of
  its nitrogens.

- Adam, V., Carpentier, P., Violot, S., Lelimousin, M., Darnault, C., Nienhaus,
  G. U. & Bourgeois, D. (2009). **Structural basis of X-ray-induced transient
  photobleaching in a photoactivatable green fluorescent protein.** *Journal of
  the American Chemical Society* 131(50), 18063–18065.
  https://doi.org/10.1021/ja907296v. X-ray damage to a chromophore's bridge.

- Borshchevskiy, V., Round, E., Erofeev, I., Weik, M., Ishchenko, A., Gushchin,
  I., Mishin, A., Willbold, D., Büldt, G. & Gordeliy, V. (2014). **Low-dose
  X-ray radiation induces structural alterations in proteins.** *Acta
  Crystallographica Section D* 70(10), 2675–2685.
  https://doi.org/10.1107/S1399004714017295. X-ray damage to the retinal region
  of bacteriorhodopsin.

- Genick, U. K., Soltis, S. M., Kuhn, P., Canestrelli, I. L. & Getzoff, E. D.
  (1998). **Structure at 0.85 Å resolution of an early protein photocycle
  intermediate.** *Nature* 392(6672), 206–209. https://doi.org/10.1038/32462.
  Isomerization of p-coumaric acid.

- Joosten, R. P., Long, F., Murshudov, G. N. & Perrakis, A. (2014). **The
  PDB_REDO server for macromolecular structure model optimization.** *IUCrJ*
  1(4), 213–220. https://doi.org/10.1107/S2052252514009324. The source of the
  map coefficients.

- Winn, M. D., Isupov, M. N. & Murshudov, G. N. (2001). **Use of TLS parameters
  to model anisotropic displacements in macromolecular refinement.** *Acta
  Crystallographica Section D* 57(1), 122–133.
  https://doi.org/10.1107/S0907444900014736. TLS refinement, and the residual
  B-factor it leaves on each atom.

- Baker, E. N. & Hubbard, R. E. (1984). **Hydrogen bonding in globular
  proteins.** *Progress in Biophysics and Molecular Biology* 44(2), 97–179.
  The distance limit of a hydrogen bond between N and O atoms.

- Barlow, D. J. & Thornton, J. M. (1983). **Ion-pairs in proteins.** *Journal
  of Molecular Biology* 168(4), 867–885. The 4 Å limit of a salt bridge.

- Hamelryck, T. (2005). **An amino acid has two sides: a new 2D measure
  provides a different view of solvent exposure.** *Proteins* 59(1), 38–48.
  Half-sphere exposure.

- Kumar, S. & Nussinov, R. (2002). **Close-range electrostatic interactions in
  proteins.** *ChemBioChem* 3(7), 604–617. The 4 Å limit of a salt bridge.

- Zhou, P., Tian, F., Lv, F. & Shang, Z. (2009). **Geometric characteristics of
  hydrogen bonds involving sulfur atoms in proteins.** *Proteins* 76(1),
  151–163. The longer hydrogen bonds of sulfur.
