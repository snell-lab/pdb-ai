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

**Anisotropic records.** `anisotropic_atom_fraction` is the share of the file's
atom sites that have a record in `_atom_site_anisotrop`. Atoms of element H, D,
or T are left out of both counts, since a refinement that gives every other atom
a tensor often still lets its hydrogens ride on their parents with one isotropic
value. The atom sites are those of the file, every alternate conformer and every
model, so the fraction does not depend on the conformer the atom table keeps and
can differ slightly from the share of the table's rows that have `u11`. A record
is joined to its atom as the `u11` to `u23` columns join it, so a record id
given to two rows counts for no atom. 9,227 entries (52.1 %) have no anisotropic
record and a fraction of 0, which is how an isotropic refinement is deposited.
6,610 (37.3 %) have a fraction of 0.9 or more.

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
Room-temperature and serial entries mostly state their dose in kGy.
`is_high_radiation_dose` is true above 5 MGy, a cut-off PDB-AI chose.

## Radiation damage (RABDAM)

PDB-AI computes `bdamage`, `packing_density`, `bnet`, and `bnet_percentile`
in-process. It uses
[snell-lab/rabdam-custom](https://github.com/snell-lab/rabdam-custom), pinned to
its v1.0.0 release. This package is an accuracy-focused rewrite of RABDAM 2
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
`rabdam_message` counts, for each source, the scored atoms that reached no row.
For PDB-Redo's model these also include atoms PDB-Redo built that the deposited
model lacks. In `both` mode, four additional boolean columns (`bdamage_matches`,
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

Each of PDB-Redo's maps is read twice for every atom. The first position is the
atom's own, in the deposited model, which is the model the atom table describes.
The second is the position PDB-Redo's re-refined model gives the same atom, from
the PDB-Redo comparison's pairing of the two models (see
[Cross-file atom alignment](#cross-file-atom-alignment)).

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

## wwPDB validation

The wwPDB validates every entry and publishes the results as
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

```text
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
  cofactors, metals, non-protein polymers, and other non-polymers.

- Kabsch, W. & Sander, C. (1983). **Dictionary of protein secondary structure.**
  *Biopolymers* 22(12), 2577–2637. https://doi.org/10.1002/bip.360221211.
  Protein-accessibility radius convention paired with Tien normalization in the
  SASA engine.

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

- Joosten, R. P., Long, F., Murshudov, G. N. & Perrakis, A. (2014). **The
  PDB_REDO server for macromolecular structure model optimization.** *IUCrJ*
  1(4), 213–220. https://doi.org/10.1107/S2052252514009324. The source of the
  map coefficients.

- Winn, M. D., Isupov, M. N. & Murshudov, G. N. (2001). **Use of TLS parameters
  to model anisotropic displacements in macromolecular refinement.** *Acta
  Crystallographica Section D* 57(1), 122–133.
  https://doi.org/10.1107/S0907444900014736. TLS refinement, and the residual
  B-factor it leaves on each atom.
