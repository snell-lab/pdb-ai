# Study plan: radiation-sensitive sites across neutron, XFEL, and synchrotron structures

This plan covers the case study for the PDB-AI paper in *Database: The Journal
of Biological Databases and Curation*. The paper describes PDB-AI as a tool and
data resource, and the case study shows what it makes possible: a high-level,
archive-wide overview of radiation-sensitive sites in structures collected by
methods that deposit very different amounts of radiation damage.

The case study is descriptive. Mechanistic analysis, statistical modeling, and
machine learning (ML) belong to later studies (see [Follow-up
studies](#follow-up-studies)).

## Aims

1. Show that PDB-AI can build reproducible cohorts from the whole archive by
   technique, X-ray source, and temperature.
2. Describe what the archive contains for each cohort: size, growth,
   temperature, resolution, and overlap in proteins.
3. Report how completely the PDB records the metadata that radiation-damage
   studies need, such as dose, exposure time, and temperature.
4. Compare radiation-sensitive sites across cohorts with a small set of clear
   figures.
5. Release the dataset, cohort lists, and run manifests so others can reuse
   them.

## Background

Most synchrotron structures carry some radiation damage, and the dose behind
each one is usually unknown. X-ray free-electron laser (XFEL) structures are
collected with femtosecond pulses that finish diffracting before most damage
develops. Neutron structures receive almost no absorbed dose. Room-temperature
serial synchrotron crystallography (SSX) spreads the dose over many crystals and
fills part of the range between them.

At low dose, damage is limited to radiation-sensitive sites, so the case study
focuses on those sites and uses similar non-sensitive sites as controls.

Single-protein studies have compared these methods directly (Ebrahim et al.
2019; Moreno-Chicano et al. 2022), and Bnet has been computed across the PDB
(Shelley & Garman 2022). An archive-wide, site-level comparison across all four
cohorts has not been found so far. A full literature review is still needed
before submission.

## Cohorts

One PDB-AI run processes one technique, and X-ray runs can also be narrowed to
one X-ray source and by temperature. Each cohort is its own run.

| Cohort | Selection | Expected damage |
|---|---|---|
| Neutron | `--select --technique neutron_diffraction` | None |
| XFEL | `--select --xray-source free_electron_laser` | Electronic damage within the pulse only, mainly at metals and heavy atoms |
| Room-temperature synchrotron (including SSX) | `--select --xray-source synchrotron --min-temperature 273` | Low to moderate, with radical chemistry active |
| Cryo synchrotron | `--select --xray-source synchrotron --max-temperature 120`, narrowed as [Cryo cohort filtering](#cryo-cohort-filtering) says | Ordinary synchrotron damage, with radicals mostly immobile |

The temperature cut-offs are placeholders. The temperature filters are strict,
so an entry with data sets on both sides of a cut-off is in neither cohort, and
an entry with no recorded temperature is in neither. Entries between the
cut-offs can be listed with both flags and are reported but left out of the
cohort figures.

### What PDB-AI does and what the analysis does

PDB-AI builds the data and decides which entries to build. The analysis decides
what the data means. The division is:

- **Selection filters**, in PDB-AI, exist to save downloads. Technique, source,
  temperature, resolution, and matching by UniProt accession each cut a cohort
  from hundreds of thousands of entries to thousands before anything is
  fetched. A decision that can be made from columns after the build is not a
  selection filter.
- **Columns**, in PDB-AI, record facts about an entry that any study could use.
  A derived flag is one only when its rule is fixed, documented, and sits
  beside its evidence, as `is_multi_method`, `metal_bridged`, and
  `mentions_time_resolved` do.
- **The analysis** makes this study's choices. Cohort labels, the
  one-cohort-per-entry rule, quality thresholds, matching and sampling, pairing
  controls across cohorts, and anything that needs a person to look.

### Cohort rules

- **Neutron:** keep neutron-only entries in the main cohort. Show joint
  X-ray/neutron entries (`experimental_methods` lists both) as a separate
  subgroup, since their X-ray data may have been collected at a synchrotron.
  `is_multi_method` alone is not enough, since it is true for any pair of
  methods.
- **XFEL:** flag time-resolved pump-probe models, which show light-activated or
  intermediate states. Show dark or ground-state models in the main figures.
- **SSX:** within the room-temperature synchrotron cohort, mark entries as
  serial when `num_crystals` is above 1, and check the marked entries by hand.
- **One cohort per entry:** the source filter admits an entry that lists the
  chosen source among others, and a joint X-ray/neutron entry matches both
  techniques. An entry that lists a free-electron laser belongs to the XFEL
  cohort only. A joint X-ray/neutron entry belongs to the neutron subgroup only.
  Both are left out of the synchrotron cohorts.
- **Temperature:** `temperature_K` is read from the first data-collection record
  only. Check entries with several data sets, such as joint X-ray/neutron
  entries, by hand before assigning them to a temperature group.
- **All cohorts:** record resolution, R-free, completeness, and deposition year
  for every entry. A joint X-ray/neutron entry has a resolution for each data
  set, in `resolution_xray` and `resolution_neutron`. `resolution` holds
  whichever the file lists first, so it is not used to compare joint entries.

### Scale

The neutron, XFEL, and room-temperature synchrotron cohorts are small enough to
cover in full. The cryo synchrotron cohort (around 200,000 entries) is reduced
as described in [Cryo cohort filtering](#cryo-cohort-filtering). Shelley &
Garman (2022) already describe Bnet across the whole cryo archive, and the paper
can cite them for that.

### Cryo cohort filtering

1. **The same quality filters for every cohort.** For example, resolution of 2.5
   Å or better, an R-free cut-off, and basic validation checks such as
   `clashscore` and `percent_rama_outliers`. These apply to all cohorts, not
   only cryo.
2. **Matched cryo.** Cryo entries that share a UniProt accession with a neutron,
   XFEL, or room-temperature entry. This subset supplies the matched examples.
   It is selected directly: the distinct `uniprot_id` values of the other
   cohorts' datasets go to a file, and `--uniprot-accessions` with the cryo
   temperature filter returns the matching cryo entries.
3. **Resolution-matched background.** A sample of the remaining cryo entries
   whose resolution distribution matches the other cohorts, for example the same
   share in each 0.25 Å bin.
4. **Redundancy removal.** One representative per RCSB sequence cluster, for
   example at 95 % identity.

Do not reduce the cryo cohort by high resolution alone. XFEL and neutron
structures are usually around 1.5–3 Å. High-resolution cryo structures skew
toward small, well-diffracting proteins, often need a higher dose, and are
refined with anisotropic B-factors and looser restraints.

Panels that need high resolution in every cohort, such as partially broken
disulfides, alternate SG conformations, and precise metal–ligand distances, use
a stricter cut-off (for example 1.8 Å or better) applied to all cohorts.

### Feasibility check (September 2026)

Counts from `--select --list-only` and a UniProt partner search against RCSB.
Partner counts use any shared UniProt accession on a protein chain, with no
quality filter, so they are upper bounds on true matched pairs.

| | XFEL | Neutron |
|---|---|---|
| Entries | 1,041 | 262 |
| Distinct UniProt accessions | 341 | 105 |
| Entries with a cryo synchrotron partner | 984 (95 %) | 240 (92 %) |
| Entries with 5 or more cryo partners | 874 | 217 |
| Entries with a room-temperature synchrotron partner | 661 (63 %) | 176 (67 %) |
| Entries collected at 120 K or below | 52 | 36 |
| Entries with no protein UniProt mapping | 8 | 13 |

- 21 UniProt accessions appear in both the XFEL and neutron cohorts. Of these,
  18 also have cryo and room-temperature synchrotron entries.
- Proteins in all four cohorts include hen egg-white lysozyme, SARS-CoV-2 main
  protease, photoactive yellow protein, T4 lysozyme, carbonic anhydrase II,
  xylose isomerase, haemoglobin, sperm whale myoglobin, cytochrome c peroxidase,
  and HRas.
- Only 123 of the 341 XFEL proteins have a room-temperature synchrotron partner,
  against 303 with a cryo partner.
- Lysozyme accounts for about 10 % of XFEL entries (101). Photosystem II appears
  as 55 multi-subunit entries.
- Synchrotron entries with no recorded temperature were not counted as partners.

Site types present in each cohort, from ligand codes and sequences. These are
rough counts, not site detection. Two or more Cys in a sequence does not mean a
disulfide.

| Site type | Neutron (262) | XFEL (1,041) |
|---|---|---|
| Any metal | 136 | 552 |
| Redox-active metal | 66 | 316 |
| Redox-inactive metal (controls) | 78 | 301 |
| Heme | 16 | 242 |
| Fe–S or Mn cluster | 3 | 93 |
| Selenomethionine | 0 | 3 |
| Br or I ligand | 6 | 11 |
| Hg, Pt, or Au | 0 | 2 |
| Proteins with 2 or more Cys | 141 | 793 |
| Flavin | 4 | 91 |
| Retinal | 0 | 90 |
| Chlorophyll | 0 | 107 |

- Metal sites and disulfides are well covered in the neutron cohort.
- Metal clusters, cofactors, heavy atoms, and halogens are rare or absent in the
  neutron cohort, and heavy atoms and halogens are mostly a synchrotron feature.
  XFEL is the baseline for cofactors. Metal clusters, heavy atoms, and halogens
  have no damage-free baseline, since these are the sites where XFEL electronic
  damage is expected. XFEL is treated as a separate damage regime for them,
  compared against literature geometry or low-dose series.

## Radiation-sensitive sites

Sites are grouped by type in roughly the order damage appears as dose rises. The
dose ranges are approximate cryo values from the literature. At room
temperature, global damage sets in at much lower doses and can mask
site-specific damage (Gotthard et al. 2019). Local environment changes the
order.

### 1. Metal centres (earliest, from about tens of kGy)

Redox-active metals are reduced by photoelectrons, which changes their ligand
distances and coordination.

| Site | Examples | What changes |
|---|---|---|
| Heme Fe | Peroxidases, catalases, cytochromes, myoglobin, P450s | Fe–axial ligand distances (His NE2, Cys SG, Met SD); Fe–O distance for oxo, aqua, and peroxo states |
| Metal-bound small ligands | O₂, peroxo, oxo, NO, CO on Fe or Cu | Lengthening or loss of the bond; reduced intermediates |
| Fe–S clusters | Ferredoxins, [2Fe–2S], [4Fe–4S] | Cluster geometry, Fe–S distances, occupancy |
| Non-heme Fe | Dioxygenases, ribonucleotide reductase | Fe–ligand distances, coordination number |
| Cu | Blue copper (Cys, His, Met), copper nitrite reductase, CuA | Cu–ligand distances, loss of the water ligand |
| Mn | The Mn₄CaO₅ cluster in photosystem II | Mn–Mn and Mn–O distances; reduced very quickly |
| Others | Ni (hydrogenase), Mo/W cofactors, Co (B12 Co–C bond) | Metal–ligand distances, bond cleavage |

**Control:** redox-inactive metals (Zn, Mg, Ca, Na, K). They are not
photoreduced, so their ligand distances should not change between cohorts.

### 2. Heavy atoms and halogens

Heavy atoms absorb X-rays strongly and lose electrons within the pulse, so XFEL
electronic damage is most likely here.

| Site | What changes |
|---|---|
| Selenomethionine (MSE), Se–C bond | Se occupancy and B-factor; C–Se bond cleavage |
| C–Br and C–I bonds in ligands (for example brominated drugs, 5-bromouracil) | Halogen occupancy loss; very early cleavage |
| Heavy-atom derivatives (Hg–Cys, Pt, Au, iodide or bromide soaks) | Occupancy loss, B-factor increase |

**Controls:** Met for selenomethionine, and the non-halogen atoms of the same
ligand.

### 3. Disulfides (from about 1 MGy)

Electrons are captured into the S–S bond, which then lengthens or breaks.

| Measure | Notes |
|---|---|
| S–S distance | Restrained during refinement, so read it together with the measures below |
| SG occupancy below 1, alternate conformations | Direct signs of a broken bond |
| SG B-factor relative to CB | Local increase at the damaged bond |
| Torsion angles (χ3) | Strained disulfides may be more susceptible |

**Controls:** Ser, which is Cys with O in place of S; unbonded Cys (imperfect,
since free thiols can also oxidize); SG compared with CB in the same residue.

### 4. Carboxylates (a few MGy)

Asp and Glu lose CO₂ (decarboxylation). BDamage and Bnet detect this.

| Site | Atoms |
|---|---|
| Asp | OD1, OD2 |
| Glu | OE1, OE2 |
| C-terminal carboxylate | O, OXT |
| Special cases | Glu222 in GFP-like proteins; active-site carboxylates |

**Controls:** Asn and Gln, which have the same shape but do not decarboxylate.
Carboxylates that bind metals or form strong hydrogen bonds may be protected and
are tagged as a subgroup.

### 5. Met and Tyr (later)

| Site | What changes |
|---|---|
| Met SD (SD–CG and SD–CE bonds) | C–S bond cleavage, SD B-factor and occupancy |
| Tyr OH | Loss of the hydroxyl group (C–O cleavage) |

**Controls:** Phe for Tyr, since it is Tyr without the OH. Leu or Ile are
approximate controls for Met.

### 6. Cofactors and chromophores

| Site | What changes |
|---|---|
| Flavins (FAD, FMN) | Photoreduction; bending of the ring system |
| Retinal (bacteriorhodopsin, rhodopsins) | Schiff-base and chromophore geometry |
| Fluorescent-protein and photoactive yellow protein chromophores | Chromophore geometry, decarboxylation of nearby Glu |
| Chlorophylls and bacteriochlorophylls | Mg coordination, ring geometry |

### 7. Exploratory

Trp and His (B-factor or occupancy changes at the side-chain rings) belong in
the exploratory residue-wide scan, not the main registry.

Nucleic acids are out of scope for now. Known damage there includes C3′–O3′
backbone cleavage and loss of Br and I from modified bases, and protein-bound
DNA is less sensitive.

### Local-environment features

Recorded for every site, since they affect susceptibility and become ML features
later:

- Solvent accessibility (from the SASA engine).
- Hydrogen bonds and salt bridges.
- Distance to the nearest metal, aromatic ring, and positively charged group.
- Whether the site is in an active site or binds a ligand.
- Strain and torsion angles, for disulfides.
- The site's B-factor relative to its neighbours.

### Current coverage

PDB-AI has no damage-site list of its own. RABDAM reports BDamage for every atom
and Bnet over Asp and Glu carboxyl oxygens. Site-level summaries are computed at
analysis time from those atom values until the site registry below exists.

The disulfide engine covers section 3 and the metal-site engine section 1, each
with a reduced scope (see [Already available](#already-available) and items 3
and 5 of [To build](#to-build)). The selenium and halogen engine covers the
selenium and the halogens of section 2, the residue-site engine sections 4 and 5, and
the cofactor engine section 6, each also with a reduced scope (items 4, 6, and
7). The density engine serves every section, since it reads PDB-Redo's maps at
every atom, and each site table links its atoms to the atom table.

## Figures

Each cohort phase ([Phases 3–5](#phases)) first describes its cohort on its own
with the same site measures. The figures below compare cohorts and are made only
after all cohorts are characterized.

### Figure 1: What the archive contains

- Entries per cohort by deposition year, showing XFEL and SSX growth.
- Temperature and resolution distributions by cohort.
- An UpSet plot of UniProt accessions shared across cohorts.

### Figure 2: Metadata completeness

- A heatmap showing how often each cohort reports dose, dose rate, exposure
  time, temperature, beamline, and crystal size.

### Figure 3: Disulfides

- The fraction of disulfides modeled as partially broken, with alternate
  conformations, or with SG occupancy below 1, by cohort.
- SG B-factor relative to CB, by cohort.
- S–S distance against resolution, by cohort.

A run of the disulfide engine on 2,140 varied entries showed three things these
panels must allow for (see [Reading the
measures](../scientific-methods.md#reading-the-measures)). Rows with
`metal_bridged` set are metal sites, 259 of the 24,827 rows of that run, and are
left out or plotted apart before any quantity is read. An S–S distance of
2.3–3.0 Å is not read as damage without the occupancy, conformer, and B-factor
measures, since most such X-ray rows away from metals were bonds refined long.
The fraction with alternate conformations is confounded by resolution, since 224
of the 350 rows with alternates held two bonded conformers, so it is compared
within bins of resolution.

### Figure 4: Carboxylates and Bnet

- Damage-site mean BDamage for Asp/Glu next to Asn/Gln, by cohort.
- Bnet by cohort. X-ray cohorts only.

### Figure 5: Metal sites, heavy atoms, and halogens

- Metal–ligand distances for the most common site types, by cohort, with
  redox-inactive metals (Zn, Mg, Ca) as the control panel.
- Se occupancy in selenomethionine and halogen occupancy in C–Br and C–I
  ligands, by cohort. These are the sites most likely to show XFEL electronic
  damage.

### Figure 6: Matched examples

- One or two well-sampled proteins, such as lysozyme, shown across every cohort
  they appear in. Candidates come from the 18 proteins found in all four
  cohorts. Metalloproteins and heme proteins among them, such as carbonic
  anhydrase II, myoglobin, and cytochrome c peroxidase, also suit the metal-site
  panels.

### Figure 7: Same-temperature comparison

- Cryo XFEL (52 entries) and cryo neutron (36 entries) against cryo synchrotron,
  for the sensitive-site measures. At a fixed temperature, the differences
  reflect method rather than temperature.

### Rules for every figure

- **Split by temperature.** Never pool room-temperature and cryo entries.
  Comparing XFEL with all synchrotron entries would mostly show temperature.
- **Weight by protein as well as by entry.** A few proteins, such as lysozyme,
  thaumatin, and proteinase K, may dominate some cohorts. Show per-entry values
  and one value per UniProt accession, or say which is used. The
  `sequence_cluster_95` column groups near-identical sequences for the same
  purpose, and `sequence_cluster_30` groups protein families.
- **Allow for refinement restraints.** Refinement pulls bond lengths toward
  ideal values, especially at lower resolution. Plot geometry against
  resolution, or restrict it to high-resolution entries.
- **Allow for how B-factors were refined.** A B-factor measure, such as that
  of a side-chain group relative to the atom it hangs from, depends on the
  refinement program, on TLS, and on isotropic or anisotropic refinement.
  Repeat such a panel within one `refinement_program`, with and without TLS
  groups (`tls_group_count`), and for isotropic and anisotropic entries
  (`anisotropic_atom_fraction`). Leave out the entries `tls_residual_adp_flag`
  marks, whose B-factors lack the TLS contribution.
- **Mark missing neutron panels.** RABDAM and PDB-REDO run on X-ray entries
  only, so neutron entries have no BDamage or Bnet. Geometry panels include
  them.
- **Use RCSB models for cross-technique panels.** PDB-REDO re-refines X-ray
  entries only, so mixing its models with deposited neutron models would add a
  refinement difference between cohorts. Use PDB-REDO values as a sensitivity
  check within the X-ray cohorts, and state which source each panel uses.
- **Show counts.** Give n per cohort in every panel.

### Statistics

Report medians, interquartile ranges, and counts, with simple effect sizes where
useful. Significance testing and modeling are left to the follow-up studies.

## PDB-AI work required

### Engine design

Site engines are defined by structure, not by technique. One engine per site
class runs the same way on neutron, XFEL, and X-ray entries, so differences
between cohorts cannot come from different code. Engines read the deposited RCSB
model, which is the model the atom table is built from. Technique-specific
handling is limited to places where the physics differs, and each case is
documented:

- Neutron models contain hydrogen and deuterium atoms, which the engines skip or
  handle explicitly.
- Neutron B-factors are refined against nuclear density. B-factor measures are
  reported for neutron entries but flagged as not directly comparable to X-ray
  values.
- Joint X-ray/neutron entries are marked so their results can be separated.

Every engine reads its site definitions from one shared site registry and
declares support for both techniques, `x_ray_diffraction` and
`neutron_diffraction`. XFEL is an X-ray source, not a separate technique.

### Output design

#### Current outputs

- Every engine adds columns to one atom table, with one row per atom. An engine
  returns columns aligned with the atom rows
  ([coordinator.py](../../pdbai/engines/core/coordinator.py)). Residue- and
  structure-level values are repeated on every atom they belong to.
- Each column declares its level: atom, residue, chain, structure, or citation.
- Each finished structure is saved as partial files
  ([partials.py](../../pdbai/runs/partials.py)). Flat mode writes the atom table
  to `partials/flat/<PDB>.csv`. Star-schema mode splits it by column level into
  atoms, residues, chains, structures, and citations
  ([dataset_shapes.py](../../pdbai/dataset_shapes.py)) and writes
  `partials/<table>/<PDB>.csv`.
- At the end of a run, [assembly.py](../../pdbai/runs/assembly.py) concatenates
  the partials for each table into the run's dataset files, and
  [publish.py](../../pdbai/runs/publish.py) copies them into the working
  directory as `pdb_atom_site_dataset.csv` or the five star-schema tables.
- Relationships between atoms, such as a metal and its ligands, a disulfide
  between two Cys residues, or an Fe–S cluster, go in engine tables (done, see
  [Framework changes](#framework-changes)). The disulfide engine's `disulfides`
  table and the metal-site engine's `metal_sites` and `metal_ligand_bonds`
  tables are built, and the other site engines below are planned users.

#### Hybrid design

- **Properties of one atom, residue, or structure stay as columns**, as now.
  Existing engines do not change.
- **Relationships between atoms go in engine tables.** An engine declares a
  table only when it has one-to-many or multi-atom data. An engine can do both.
- **Atoms get only a link.** At most a `site_id` and a `radiation_site_type`
  flag, so site rows can be joined back to their atoms, residues, and structure.
  Putting all site data on atoms would make the atom table very wide and mostly
  empty.
- **Engine tables are separate files in both output modes.** Flat mode writes
  `pdb_atom_site_dataset.csv` plus one file per engine table. Star-schema mode
  writes its five tables plus the engine tables.

A database format (SQLite or DuckDB) was considered and rejected. It is a large
change for little gain, since per-table files load anywhere and DuckDB can query
Parquet files directly.

#### Framework changes

**Done.** Items 1–6 are built: `TableSpec` in
[contracts.py](../../pdbai/engines/core/contracts.py), validation and the shared
`site_id` in [tables.py](../../pdbai/engines/core/tables.py), tested in
`tests/engines/test_engine_tables.py` and
`tests/integration/test_engine_tables_integration.py`, and documented in [Engine
tables](../dataset-schema.md#engine-tables).

1. **Table specs.** `EngineSpec` gains a `tables` field. Each table spec gives a
   name, key columns, link columns back to atoms or residues, and column specs
   (type, description, and allowed categories). Units go in column names.
2. **Table results.** `EngineResult` gains a `tables` field. The coordinator
   checks each returned table against its spec: no undeclared columns, unique
   keys, link columns present.
3. **Partials and resume.** Each structure writes `partials/<table>/<PDB>.csv`
   for every enabled engine table. A structure with no sites still gets a
   header-only file, so columns stay consistent. A structure counts as complete
   only when all its declared tables are written.
4. **Assembly.** The existing concatenation code is reused. The table list is
   the base tables for the output mode plus the tables of each enabled engine.
5. **Records.** The run manifest, run report, run summary, and data dictionary
   list each engine table with its description, keys, and columns.
6. **Filtering.** Column filtering and profiles apply to the base tables only at
   first. Engine tables are written in full when their engine is enabled, and
   their key columns are never dropped.

#### ML compatibility

1. **One row per site, fixed schema.** Every table has the same columns for
   every structure and technique. No lists or nested values in cells.
   One-to-many data goes in child tables linked by a key, and the parent row
   carries fixed-size summaries of its children (counts, minimum, mean, and
   maximum), since models need a fixed-length input per site.
2. **Stable keys.** `site_id` is built from entry, model, chain, residue,
   insertion code, atom, and alternate conformation, so reruns give the same ID.
   Join keys link rows to the atoms, residues, and structures tables. Grouping
   keys for protein-level splits (UniProt accession, sequence cluster) are on
   the row or one join away.
3. **Measurements only.** Engines store raw measurements. Cohort labels and
   cross-cohort differences are added in the comparison step, which keeps the
   engines unbiased and stops label information leaking into features.
   Dataset-wide normalization happens at training time, fitted on the training
   set only. Within-structure comparisons, such as SG B-factor relative to CB,
   can be stored.
4. **Controls in the same table**, marked with `is_control` and `control_for`.
5. **Explicit missing values.** Missing (NaN), never 0, when something was not
   measured. A status column says why (`ok`, `no_ligands_found`,
   `hydrogen_skipped`, `not_applicable`), following the existing `sasa_status`
   and `sasa_ok` pattern. Nullable booleans keep "unknown" separate from
   "false".
6. **Types, units, and categories.** Units in column names (`distance_A`,
   `b_factor_A2`, `angle_deg`). Every column's type declared in the spec. Fixed,
   documented values for category columns (`site_type`, `metal_element`,
   `ligand_atom`, `detection_method`), so one-hot encoding stays stable across
   runs.
7. **One shared environment table.** Local-environment features come from one
   shared calculation, written to `site_environment` keyed by `site_id`, so
   every site type has identical features.
8. **Parquet output.** An output-format option (CSV, Parquet, or both) for all
   tables. Parquet keeps column types, is compressed, and loads much faster at
   archive scale. This is a separate change from the engine tables.
9. **Provenance.** The run manifest records every cut-off used.

A training set is then built by starting from a site table, joining
`site_environment` on `site_id` and the structures table on `pdb_id`, and adding
labels in the comparison step. The result is one row per site with a fixed set
of columns. Bond and disulfide tables are already edge lists, so they also suit
graph models later.

#### Example: metal tables

This sketch preceded the engine. The tables as built are listed under [Metal
sites](../dataset-schema.md#metal-sites). They differ from the sketch in four
ways. Controls are not marked, since which metal is a control is the study's
choice. There is no `cluster_id` or `site_type`, and `parent_type` says whether
the metal sits in a heme or a cluster. Each bond carries a literature reference
distance and its deviation from it. And density is not in the tables, since the
atom table holds it for the metal and for each donor.

`metal_sites`, one row per metal atom:

| Column | Example or note |
|---|---|
| `site_id` | `1ABC:1:A:301::FE:` |
| `pdb_id`, `model_num`, `chain_id`, `res_num`, `ins_code`, `res_name`, `atom_name`, `altloc` | Join keys |
| `site_type` | `heme_fe` |
| `metal_element` | `FE` |
| `is_redox_active`, `is_control`, `control_for` | Control flags |
| `cluster_id` | Shared by every metal in one Fe–S or Mn₄CaO₅ cluster |
| `coordination_number` | 6 |
| `ligand_distance_mean_A`, `ligand_distance_min_A`, `ligand_distance_max_A` | 2.05, 1.98, 2.21 |
| `n_ligand_N`, `n_ligand_O`, `n_ligand_S`, `n_ligand_water` | Ligand counts |
| `occupancy`, `b_factor_A2`, `b_factor_rel_neighbours` | |
| `metal_site_status` | `ok` |

`metal_ligand_bonds`, one row per metal–ligand bond: `site_id`, the ligand
atom's keys, `ligand_atom`, `distance_A`, `is_symmetry_mate`, `detection_method`
(`struct_conn` or `distance`), and `bond_status`.

Neither table has cohort, technique, or temperature columns. Those come from the
structures table at analysis time.

### Already available

- Archive selection by technique and X-ray source, narrowed by strict ranges of
  data-collection temperature and resolution (`--min-temperature`,
  `--max-temperature`, `--max-resolution`) and by a list of UniProt accessions
  (`--uniprot-accessions`), recorded in the run manifest. Every value an entry
  records must be in range, so a joint X-ray/neutron entry must pass a
  resolution range on both data sets.
- The entry `title` and `keywords`, and `mentions_time_resolved`, a text match
  on them. `is_serial_crystallography`, `sample_delivery_method`, and
  `xfel_pulse_duration_fs` from the serial crystallography records.
- `resolution_xray`, `resolution_neutron`, and the per-method `r_work` and
  `r_free` columns for joint X-ray/neutron entries.
- How each entry was refined, read from its structure file:
  `refinement_program`, its version, and `refinement_programs`,
  `tls_group_count`, `tls_residual_adp_flag`, and `anisotropic_atom_fraction`.
- RCSB sequence cluster membership, `sequence_cluster_30` to
  `sequence_cluster_100`, written with the UniProt mapping.
- RABDAM BDamage, Bnet, and Bnet percentile (X-ray only). Damage-site means are
  computed at analysis time from atom BDamage.
- `temperature_K`, `radiation_dose_MGy`, `beamline`, `radiation_sources`,
  `experimental_methods`, `is_multi_method`, `num_crystals`, `collection_year`,
  `synchrotron_site`, `detector_type`, and data-quality columns.
  `radiation_dose_MGy` is read from the entry title, so it shows only doses that
  titles state. `beamline` falls back to the source's make and model when no
  beamline is reported.
- wwPDB validation (`--include-validation`, X-ray and neutron): per-residue
  `eds_rscc` and `eds_rsr`, which show poorly fitting carboxylates, and
  structure-level `clashscore`, `percent_rama_outliers`, and
  `percent_rotamer_outliers` for the quality filters.
- UniProt mapping, PDB-REDO comparison (X-ray only), SASA, and AlphaFold
  Database pLDDT.
- Electron density at atoms (`--include-density`, X-ray only): the density of
  the 2mFo−DFc and mFo−DFc maps at every atom, in units of each map's
  root-mean-square density. `--density-source` reads PDB-Redo's maps, at the
  deposited position and at PDB-Redo's, the wwPDB validation maps of the
  deposited model, or both. It reads what the data show at a site whether or
  not the depositor modelled the damage (see [Electron density at
  atoms](../scientific-methods.md#electron-density-at-atoms)). A hole in sigma
  is deeper at a heavier atom for the same share lost, so from PDB-Redo's maps
  the engine also writes `density_fc_eA3` and `density_fofc_eA3`, the
  calculated density and the difference density on one scale, whose ratio
  compares one element with another. The value at an atom cannot show a bond
  that restraints hold short of where the data put its atoms, so the engine
  also writes the gradient of the difference map at each atom, in sigma per
  Å, and from PDB-Redo's maps the shift that would level it, in Å. Projected
  on a bond, they say whether the data want it longer.
- Metal sites (`--include-metal-sites`, every technique): the `metal_sites` and
  `metal_ligand_bonds` tables, with each metal atom's donors from deposited
  records and first-sphere distances, symmetry mates included, and each bond's
  deviation from a literature distance, and the `metal_metal_distances` table,
  with every pair of metal atoms within 6 Å, which holds the distances within
  a cluster (see [Metal
  coordination](../scientific-methods.md#metal-coordination)).
- Disulfides (`--include-disulfides`, every technique): the `disulfides` table,
  one row per candidate pair of cysteines and model, found from `_struct_conn`
  disulf records and from SG–SG distances of at most 3.0 Å, with no lower limit,
  over every alternate conformer. It holds `ss_distance_min_A` and
  `ss_distance_max_A`, SG occupancy, alternate conformations, and SG/CB B-factor
  ratios for Figure 3, and `metal_bridged` marks the pairs whose two sulfurs are
  bound to one metal atom. A row is a candidate, not a proven bond. Atoms link
  to it through `disulfide_site_id` (see
  [Disulfides](../scientific-methods.md#disulfides)).
- Residue sites (`--include-residue-sites`, every technique): the
  `residue_sites` table, one row per carboxylate of Asp, Glu, and a C-terminus,
  per Met and Cys sulfur, per Tyr hydroxyl, and per ring of Trp and His, per
  control group of Asn, Gln, Phe, Leu, Ile, Thr, and Ser, and per residue
  deposited oxidized: cysteine, methionine sulfoxide and sulfone, hydroxylated
  tryptophan, the tryptophan hydroperoxide, kynurenine, 2-oxohistidine, DOPA,
  and meta-tyrosine. It ships a reference table of the rate constants of each
  amino acid with the hydroxyl radical and the hydrated electron (Buxton et
  al., 1988), which each run records a receipt of. The engine also
  writes `b_factor_ratio_to_parent` on every atom of a standard amino acid. It holds how many of the group's atoms the model has, their
  occupancy, their B-factors against the atom the group is attached to, and
  `metal_bound`. Site atoms link to it through `residue_site_id` (see [Residue
  sites](../scientific-methods.md#residue-sites)).
- Site environment (`--include-site-environment`, every technique): the
  `site_environment` table, one row per group of `residue_sites` with the same
  `site_id`, with possible hydrogen bonds to protein and water, salt bridges,
  the distance to the nearest cation, disulfide, metal, aromatic ring, and
  ligand, packing within 6 and 10 Å, solvent-accessible area and depth, and the
  nearest atom of a symmetry mate, and `atom_depth_A` on every protein atom.
  Everything but the waters and crystal contacts comes from the protein atoms
  alone, so it can be read from a predicted model (see [Site
  environment](../scientific-methods.md#site-environment)).
- Selenium and halogens (`--include-selenium-halogens`, every technique): the
  `selenium_halogen_sites` table, one row per selenium or halogen atom,
  fluorine and chlorine included as the lighter halogens the others are
  compared with, with the atoms it is bonded to, and its occupancy and B-factor
  beside theirs and beside the rest of its component (see [Selenium and
  halogens](../scientific-methods.md#selenium-and-halogens)).
- Cofactors (`--include-cofactors`, every technique): the `cofactor_sites`
  table, one row per flavin, tetrapyrrole, fluorescent-protein chromophore,
  retinal, or p-coumaric acid, with the flavin butterfly angle, the central
  metal's distance from the plane of a heme or chlorophyll, and the torsions
  about the bonds that isomerize (see
  [Cofactors](../scientific-methods.md#cofactors)).

### To build

**Engine output tables**

1. **Engine output tables, done:** the framework changes in [Output
   design](#output-design), namely table specs, table results with validation,
   partials and resume, assembly, and records. Built and tested with a small
   test engine before any site engine uses it.

**Site registry**

2. **Radiation-sensitive site registry:** one shared definition of every site in
   [Radiation-sensitive sites](#radiation-sensitive-sites) and its controls,
   plus the local-environment features recorded for each site, computed once and
   written to the shared `site_environment` table.

**Site engines**

Engines are built in the order the cohorts need them (see [Phases](#phases)).
The list below may grow as the XFEL and X-ray phases turn up new site types.

Each engine follows the [hybrid design](#hybrid-design): site data in its own
tables, one row per site with its controls, and only a `site_id` link and site
flag on the atom table. All engines declare support for both X-ray and neutron
diffraction.

3. **Metal-site engine, built with a reduced scope** (sections 1 and the
   heavy-atom derivatives in 2): metal–ligand distances, coordination number,
   occupancy, and B-factors, in the `metal_sites` and `metal_ligand_bonds`
   tables, and the distances between nearby metal atoms in
   `metal_metal_distances` (see [Already available](#already-available)).
   Deferred: bound small ligands, which are rows only when a deposited record
   names them, donors within the metal's own component, a shared identifier for
   the metals of one cluster, for which `same_component` of
   `metal_metal_distances` stands in, strict non-crystallographic symmetry, and
   the shared `site_environment` features. The design decisions it followed:
   - **Ligand detection:** `struct_conn` records of type `metalc`, which are
     incomplete in many entries, plus distance cut-offs that depend on the metal
     and ligand atom (roughly 1.9–2.8 Å). Each bond records which method found
     it.
   - **Symmetry mates:** ligands from neighbouring molecules in the crystal,
     found with gemmi's neighbour search.
   - **Clusters:** Fe–S and Mn₄CaO₅ metals share a `cluster_id`.
   - **Alternate conformations and models:** one row each.
   - **Neutron models:** hydrogen and deuterium atoms are not counted as
     ligands.
   - **Oxidation state:** not in the PDB, so the engine reports geometry only.
4. **Heavy-atom and halogen engine, built as the selenium and halogen engine
   with a reduced scope** (section 2):
   selenomethionine Se–C bonds and C–Br and C–I bonds, with occupancy and
   B-factors, against the bonded atoms and the ligand's other atoms, in the
   `selenium_halogen_sites` table (see [Already available](#already-available)).
   Halide ions are rows too. Met is the control and is a `met_thioether` row of
   `residue_sites`. Deferred: bonds to a symmetry mate, every alternate
   conformer of the atom, and the shared `site_environment` features. The
   heavy metals of section 2 are rows of `metal_sites`.
5. **Disulfide engine, built with a reduced scope** (section 3): S–S distance,
   SG occupancy, alternate conformations, SG vs CB B-factor, and a metal-site
   flag, in the `disulfides` table of candidate pairs (see [Already
   available](#already-available)). Pairs whose SG atoms lie more than 3.0 Å
   apart in every conformer are in the table only when a deposited record names
   them. Deferred:
   - χ3 and other strain measures.
   - The Ser and unbonded-Cys controls, which the `residue_sites` table now
     holds as its `ser_hydroxyl` and `cys_sulfur` rows, in the one conformer
     the atom table keeps.
   - Disulfides to symmetry mates, which are counted but not measured.
   - The neutron check of the thiol hydrogen or deuterium on SG.
   - The shared `site_environment` features.
6. **Carboxylate, Met, and Tyr engine, built with a reduced scope** (sections 4
   and 5): modelled atoms, occupancy, B-factors, and the attaching bond at the
   sensitive atoms, including C-terminal carboxylates, against Asn, Gln, Phe,
   Leu, and Ile, in the `residue_sites` table (see [Already
   available](#already-available)). Deferred: every alternate conformer of a
   group, C-termini modelled without an OXT, hydrogen bonds and salt bridges,
   which belong to the shared `site_environment` features, and the special
   cases of section 4, which are rows like any other and are picked out at
   analysis time.
7. **Cofactor engine, built with a reduced scope** (section 6): geometry,
   occupancy, and B-factors at the reactive group of flavins, retinal,
   fluorescent-protein and photoactive yellow protein chromophores, and
   chlorophylls and hemes, in the `cofactor_sites` table (see [Already
   available](#already-available)). Deferred: bilins, quinones, and
   nicotinamides, retinal deposited within the LYR residue, and the shared
   `site_environment` features.

**Columns and output**

8. **Collection metadata:** extract dose rate, exposure time, and crystal size
   where the entry deposits them, as structure-level columns.
9. **Parquet output:** an output-format option (CSV, Parquet, or both) for
   every table. Adds pyarrow as a dependency.

**Done since the plan was written.** The density engine and the metal-site
engine, with a local copy of PDB-Redo read before the network. The
residue-site, selenium and halogen, and cofactor engines. Strict
temperature and resolution filters
and matching by UniProt accession on the archive search. The entry title,
keywords, and the `mentions_time_resolved` text match, with the serial
crystallography flag, sample delivery method, and pulse duration, as columns.
Per-method resolution and R factors for joint X-ray/neutron entries. RCSB
sequence cluster membership at six identity levels, written with the UniProt
mapping. The refinement program, the number of TLS groups, the archive's flag
for residual B-factors, and the share of atoms with an anisotropic record, as
columns.

**Analysis, not PDB-AI.** Under the division above, these are scripts in the
study's analysis, built on the columns:

- **Cohort labeling** from technique, source, and temperature, with the
  one-cohort-per-entry rule applied from `radiation_sources` and
  `experimental_methods`.
- **The time-resolved review.** `mentions_time_resolved` finds the entries of
  time-resolved series, and a pass over their titles separates resting-state
  models from intermediates. The archive records no field for that.
- **Cross-cohort protein overlap** by `uniprot_id` and `sequence_cluster_95`,
  for the UpSet plot and the matched examples.
- **The metadata-completeness report**, a per-cohort view of the "% Null" that
  the run summary's data dictionary already gives.
- **Resolution-matched sampling and one-per-cluster reduction** of the cryo
  cohort.

The paper describes PDB-AI by its data and its selection filters. The analysis
scripts are released with the dataset.

## Phases

Phase 2 builds the shared framework. The cohort phases (3–5) then build and
check the engines on one cohort at a time. No cross-cohort comparisons are made
until all cohorts are characterized.

**New engines added during a later phase.** When the XFEL or X-ray phase needs a
new engine or engine option, it is built the same technique-neutral way and then
run on every cohort, including the earlier ones. Every engine therefore covers
every technique before comparisons start.

### Phase 1: Feasibility

- Done: list the XFEL and neutron cohorts, count proteins shared with
  synchrotron entries, and tabulate temperature (see [Feasibility
  check](#feasibility-check-september-2026)).
- List and count the room-temperature synchrotron cohort.
- Tabulate the XFEL and neutron cohorts by resolution, time-resolved status, and
  multi-method status.
- Fix the temperature and quality cut-offs.

### Phase 2: Engine output tables and site registry

- Done: build the engine output tables in this order: table specs and
  validation; partials, resume, and assembly; then manifest, report, summary,
  and data dictionary entries. Test each step with a small test engine.
- Build the shared site registry and its controls. Partly done: the
  `site_environment` table, for the groups of `residue_sites`. Disulfides,
  metal sites, the selenium and halogen atoms, and cofactors do not have its
  features yet.
- Review the site list against the radiation-damage literature.

### Phase 3: Neutron

The neutron cohort comes first because it is small (262 entries), has no
radiation damage, and is the hardest case technically.

- Done: the disulfide engine, with the reduced scope of [To build](#to-build)
  item 5, and the metal-site engine, with that of item 3.
- Done: the residue-site engine for carboxylates, Met, and Tyr, with the
  reduced scope of item 6, and the selenium and halogen and the cofactor
  engines, with those
  of items 4 and 7. Heavy atoms, halogens, metal clusters, and cofactors are
  rare or absent in neutron entries (see [Feasibility
  check](#feasibility-check-september-2026)), so those two tables are nearly
  empty for this cohort.
- Check each engine by hand on a few well-known neutron structures.
- Run the engines on the full neutron cohort.
- Describe the neutron cohort on its own: site geometry with no radiation
  damage. This is the baseline for every later comparison of the site types it
  contains. For cofactors, the XFEL cohort is the baseline. Metal clusters,
  heavy atoms, and halogens have no damage-free baseline (see [Feasibility
  check](#feasibility-check-september-2026)).

### Phase 4: XFEL

- Run the same engines on the full XFEL cohort, plus RABDAM and PDB-REDO, fixing
  any bugs in the shared code rather than adding XFEL-specific paths.
- Add engines or engine options for site types that appear here, such as metal
  clusters and cofactors, following the rule for new engines above.
- Build and check the time-resolved flag.
- Check the engines by hand on a few well-known XFEL structures.
- Describe the XFEL cohort on its own, including its cryo subset.

### Phase 5: X-ray

- Build the room-temperature synchrotron cohort and the filtered cryo cohort
  (see [Cryo cohort filtering](#cryo-cohort-filtering)).
- Run the same engines, plus RABDAM and PDB-REDO.
- Add engines or engine options for site types that appear here, such as
  selenomethionine, heavy-atom derivatives, and halogenated ligands.
- Check the engines by hand on a few well-known X-ray structures.
- Describe each X-ray cohort on its own.

### Phase 6: Dataset

- Add the collection-metadata columns (dose rate, exposure time, and crystal
  size).
- Build the final cohorts in the star-schema shape.
- Keep every run manifest.
- Tag a PDB-AI release and freeze the dataset.

### Phase 7: Comparisons and figures

- Write the analysis scripts: cohort labeling, the cross-cohort protein
  overlap, and the metadata-completeness report.
- Produce Figures 1–7 following the rules above.
- Write short descriptive results for each figure.

### Phase 8: Submission

- Deposit the dataset, cohort lists, and run manifests with a DOI, for example
  on Zenodo.
- Give the repository URL and dataset DOI in the abstract.
- Consider a small hosted demo or ready-to-download dataset, since the web
  application runs locally only.

## Paper outline

1. **Introduction:** the need for analysis-ready structural datasets, and why
   radiation damage is a good test case.
2. **PDB-AI:** architecture, sources, engines, output shapes, provenance, and
   reproducible cohort selection.
3. **Curation:** how PDB-AI handles inconsistent PDB metadata, and what the
   metadata-completeness figure shows.
4. **Site engines:** the shared site registry and the structure-specific
   engines, with their checks on each technique.
5. **Case study:** each cohort described on its own, then Figures 1–7.
6. **Availability:** repository, licence, release, and dataset DOI.
7. **Outlook:** the follow-up studies below.

## Scope and framing

Crystallographic "low dose" means kGy to MGy. Radiation protection works in mGy
to Gy, at least three orders of magnitude lower. The case study describes which
sites differ between cohorts. It does not claim to measure dose-response at
radiation-protection doses.

## Follow-up studies

- **Mechanism study:** a paired comparison by protein, mixed-effects models,
  validation of Bnet on room-temperature and XFEL data, and a comparison of
  neutron geometry with XFEL geometry.
- **ML study:** prediction of per-site susceptibility from local environment.
  See [ML study notes](#ml-study-notes).
- **External check:** comparison of site susceptibility with hydroxyl-radical
  footprinting mass-spectrometry data, which is collected in solution at room
  temperature and low dose.

## ML study notes

The aim is a useful first model, not a perfect one. This case study comes first
because it checks that the cohorts differ at sensitive sites, builds the
features the model needs, and shows how much low-damage data exists.

### Role of each data source

- **XFEL and neutron structures** fix the zero-damage point. Their value comes
  from matched synchrotron partners. The site-by-site difference within a pair
  is the training label.
- **Dose-series depositions and SSX structures** cover the low-dose range
  between zero and ordinary synchrotron doses.
- **Cryo synchrotron structures** cover the high-damage end. Comparing sensitive
  sites with similar non-sensitive sites inside each structure also gives a
  signal without a partner, which could be used for pre-training later.

### Scale

About 425 distinct proteins across the XFEL and neutron cohorts (341 + 105, less
the 21 shared) give thousands of labeled sites, but the protein count limits how
well a model generalizes to new protein families. The cryo XFEL and cryo neutron
entries give the cleanest labels, since they compare method at a fixed
temperature.

### First model

- Match pairs by UniProt accession plus a resolution window.
- Use gradient boosting on site-level features: residue type, solvent
  accessibility, B-factor relative to neighbours, resolution, and temperature.
  Cohort is not a feature, since the label is the difference between cohorts.
- Leave out metal sites or flag them. XFEL labels are noisy there because of
  electronic damage within the pulse.
- Defer pre-training on the cryo archive and dose-series calibration to later
  versions.

### Feature-table helper

A small helper joins a site table, `site_environment`, and the structures table
into one feature table, so every ML study builds its training data the same way.
It belongs to the ML study, not this phase.

### Requirements, even for the first model

- **Split by sequence cluster.** The same protein, or a close homolog, must
  never appear in both training and test sets. Split by RCSB sequence cluster
  rather than UniProt accession, using the `sequence_cluster_30` column or a
  stricter level, and set test clusters aside before training.
- **Handle temperature.** Include it as a feature or split by it, so the model
  does not learn room temperature vs cryo and call it damage.

## Open questions

- Which temperature cut-offs should define the room-temperature and cryo
  cohorts?
- Which resolution, R-free, and sequence-identity thresholds should the cryo
  filtering use?
- Should joint X-ray/neutron entries appear in the figures or only in a
  supplement?
- Which proteins make the best matched examples?
- Is a hosted demo worth building before submission?

## Key references

Check these before citing them.

- Ebrahim, A. et al. (2019). Dose-resolved serial synchrotron and XFEL
  structures of radiation-sensitive metalloproteins. *IUCrJ*.
- Gerstel, M., Deane, C. M. & Garman, E. F. (2015). Introduces BDamage. *Journal
  of Synchrotron Radiation*.
- Gotthard, G. et al. (2019). Specific radiation damage is a lesser concern at
  room temperature. *IUCrJ*.
- Henderson, R. (1990). Proposes the 20 MGy dose limit for cryo crystals.
  *Proceedings of the Royal Society B*.
- Moreno-Chicano, T. et al. (2022). Complementarity of neutron, XFEL and
  synchrotron crystallography for defining the structures of metalloenzymes at
  room temperature. *IUCrJ*.
- Nass, K. et al. (2015). Reports damage at Fe–S clusters with XFEL pulses.
  *Journal of Synchrotron Radiation* 22, 225–238.
- Owen, R. L., Rudiño-Piñera, E. & Garman, E. F. (2006). Proposes the 30 MGy
  dose limit. *PNAS*.
- Shelley, K. L. & Garman, E. F. (2022). Quantifying and comparing radiation
  damage in the Protein Data Bank. *Nature Communications*.
