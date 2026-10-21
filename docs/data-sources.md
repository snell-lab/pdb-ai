# Data Sources and Limitations

This page describes remote services, cache behavior, coverage limits, and
interpretation risks.

[Project README](../README.md)

## External data sources

| Source | URL pattern | Purpose | Cached as | Fetched when |
|--------|-------------|---------|-----------|---------------|
| RCSB standard mmCIF | `https://files.rcsb.org/download/{pdb_id}.cif` | Primary structure file, with coordinates, B-factors, occupancy, and ANISOU. Also the source of how the entry was refined, namely its refinement programs and TLS groups, which the RCSB Core Entry record holds only in part. | `mmcif/rcsb/{pdb_id}.cif` | always |
| PDB-Redo re-refined mmCIF | `https://pdb-redo.eu/db/{pdb_id}/{pdb_id}_final.cif` | Re-refined structure. It is the source for the PDB-Redo comparison columns (deltas and real x/y/z) and RABDAM's `pdb_redo`/`both` scoring input. | `mmcif/pdb_redo/{pdb_id}.cif` | `--include-pdb-redo`, or `--include-rabdam` (off by default, like every engine) with `--rabdam-source pdb_redo` (the default) or `both`; X-ray diffraction runs only |
| PDB-Redo reflection file | `https://pdb-redo.eu/db/{pdb_id}/{pdb_id}_final.mtz` | The map coefficients of PDB-Redo's re-refined model (`FWT`, `PHWT`, `DELFWT`, `PHDELWT`), from which the density engine makes the 2mFo−DFc and mFo−DFc maps, and its calculated structure factors (`FC_ALL`, `PHIC_ALL`), from which it makes the calculated map. | `mmcif/pdb_redo/{pdb_id}_final.mtz` | `--include-density` with `--density-source pdb_redo` (the default) or `both`, which fetches the re-refined mmCIF as well; X-ray diffraction runs only |
| wwPDB validation map coefficients | `https://files.rcsb.org/pub/pdb/validation_reports/{mid}/{pdb_id}/{pdb_id}_validation_2fo-fc_map_coef.cif.gz` and `..._validation_fo-fc_map_coef.cif.gz` | The coefficients of the 2mFo−DFc and mFo−DFc maps the wwPDB validation pipeline calculated about the deposited model (`pdbx_FWT`, `pdbx_PHWT`, `pdbx_DELFWT`, `pdbx_DELPHWT`). The files hold no calculated structure factors. | `mmcif/validation_maps/{pdb_id}_2fofc_map_coef.cif.gz` and `{pdb_id}_fofc_map_coef.cif.gz` | `--include-density` with `--density-source wwpdb` or `both`; X-ray diffraction runs only |
| RCSB Search API | `https://search.rcsb.org/rcsbsearch/v2/query` | Archive-wide entry selection. It returns every current experimental entry matching the run's technique and X-ray source, narrowed by any temperature or resolution range filter, as a sorted identifier list. | not cached; the query, time, and result are recorded in the run manifest under `selection` | `--select` (CLI) or "Find matching entries in the PDB" (web) |
| RCSB Core Entry | `https://data.rcsb.org/rest/v1/core/entry/{pdb_id}` | Experimental metadata, such as resolution, R-factors, beamline, temperature, detector, space group, reflection statistics, Wilson B, multiplicity, deposit date, and citation and author information. The record has no dose field, so the dose is read from the entry title. | `mmcif/{pdb_id}_metadata.json` | always (whenever metadata is fetched at all, that is, for any PDB ID-based build) |
| RCSB Polymer Entity | `https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}` | UniProt accessions and aligned regions for every polymer entity. | `mmcif/afdb/entities/{pdb_id}.json` | `--include-uniprot` or `--include-alphafold` |
| RCSB wwPDB Validation XML | `https://files.rcsb.org/pub/pdb/validation_reports/{mid}/{pdb_id}/{pdb_id}_validation.xml.gz` | Per-residue electron density statistics (RSCC, RSR) and geometry quality (Ramachandran class, rotamer class), with structure-level clashscore and outlier percentages. | `mmcif/{pdb_id}_validation.json` | `--include-validation` |
| AlphaFold Database Prediction API | `https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}` | Discovers the canonical prediction family, fragments, model IDs, sequence bounds, and CIF URLs for an accession. | successful result in `mmcif/afdb/models/{uniprot_id}.json` | `--include-alphafold` |
| AlphaFold Database ModelCIF | URL returned by the prediction API | Validated per-residue pLDDT and exact model and fragment provenance. | same model JSON cache | `--include-alphafold` |

`{mid}` is the 2nd and 3rd characters of the lower-case PDB ID. The `minimal`
profile turns on engines, and that triggers the same fetches as their flags.

### Cache behavior

Downloaded files and successful lookups are cached under the working directory's
`mmcif/` tree; search results and failures are not cached. The tree has `rcsb/`
and `pdb_redo/` subdirectories for structure files. A cached file is reused for
30 days (`CACHE_TTL_SECONDS` in `pdbai/engines/sources/http_cache.py`) and then
fetched again, because entries are revised after deposition. To re-fetch one
sooner, delete it. A response that is not mmCIF never replaces a cached
structure file, and one that does not start as an MTZ or gzip file never
replaces a cached reflection file. Cache file names use the lower-cased PDB ID
or UniProt accession, as in `1abc_metadata.json` or `p69905.json`. Paths below
are relative to `mmcif/`.

| Cache file | Controls |
|------------|----------|
| `rcsb/{pdb_id}.cif` | RCSB structure (coordinates, B-factors, ANISOU) |
| `pdb_redo/{pdb_id}.cif` | PDB-Redo structure (comparison source and RABDAM input) |
| `pdb_redo/{pdb_id}_final.mtz` | PDB-Redo reflection file (density engine input) |
| `validation_maps/{pdb_id}_2fofc_map_coef.cif.gz` | wwPDB validation 2mFo−DFc map coefficients (density engine input) |
| `validation_maps/{pdb_id}_fofc_map_coef.cif.gz` | wwPDB validation mFo−DFc map coefficients (density engine input) |
| `pdb_redo_mirror/{mid}/{pdb_id}/` | A local copy of PDB-Redo's databank, read before the network and never expired (see below) |
| `{pdb_id}_metadata.json` | The RCSB Core Entry record, as RCSB sent it |
| `{pdb_id}_validation.json` | Parsed validation report |
| `afdb/entities/{pdb_id}.json` | Complete RCSB entity-to-UniProt alignment discovery |
| `afdb/models/{uniprot_id}.json` | Completely successful AFDB pLDDT and per-position model provenance. Failures are not cached, so they are retried. |

### A local copy of PDB-Redo

PDB-Redo offers its databank by rsync as well as file by file, and a run over
thousands of entries is better served by a copy made once. Both PDB-Redo source
engines read a copy placed under `mmcif/pdb_redo_mirror/` before they make a
request. Its layout is PDB-Redo's own, the directory of an entry beneath the two
middle characters of its identifier, so this command writes it for the whole
databank:

```bash
rsync -av --exclude=attic rsync://rsync.pdb-redo.eu/pdb-redo/ \
    <workdir>/mmcif/pdb_redo_mirror/
```

To copy only the entries of a run, give rsync a list of their files with
`--files-from`, three lines per entry in the form `cb/1cbs/1cbs_final.cif`,
`cb/1cbs/1cbs_final.mtz`. A file of the copy is used as it stands and does not
expire, since whoever made the copy decides when to renew it. It must still pass
the check a download would. An mmCIF file must open a data block and a
reflection file must start as an MTZ file, or the engine fails for the entry.
The manifest records it by a `file://` URL. An entry the copy lacks is requested
from PDB-Redo as it would be without a copy, so a partial copy is safe.

## Known issues

- **Missing experimental metadata.** Many RCSB fields are sparsely populated
  (e.g. `radiation_dose_MGy`, `wilson_b`, `crystal_pH`). These columns will
  contain NaN for most structures.
- **PDB-Redo coverage.** PDB-Redo does not cover every entry (e.g. `1CRN` and
  `4HHB` have no PDB-Redo entry). For those, the PDB-Redo comparison and
  PDB-Redo-scored RABDAM columns are empty and the run report records why, while
  the rest of the entry is built. `--rabdam-source rcsb` scores the deposited
  model instead. See [Choosing what RABDAM
  scores](scientific-methods.md#choosing-what-rabdam-scores---rabdam-source).
- **Density coverage.** Each map source covers the entries it has maps for. An
  entry PDB-Redo has not re-refined has empty PDB-Redo density columns, and
  `density_map_status` says `no_pdb_redo_entry`. An entry the wwPDB archive has
  no validation map coefficients for has empty `_wwpdb` columns, and
  `density_map_status_wwpdb` says `no_validation_maps`. Both need deposited
  structure factors, so older entries are covered less often. PDB-Redo's maps
  describe its re-refined model, and the wwPDB maps the deposited one (see
  [Electron density at
  atoms](scientific-methods.md#electron-density-at-atoms)).
- **Validation XML availability.** A small fraction of PDB entries (very recent
  depositions, some NMR structures) lack validation XML reports. `eds_rscc`,
  `eds_rsr`, `rama_class`, `rotamer_class`, and structure-level quality columns
  will be NaN for those structures. Inspect `validation_status` and
  `validation_message` in `run_report.csv` to distinguish an unavailable report
  from a fetch or parse failure.
- **AlphaFold Database coverage.** Not every PDB protein entity has a UniProt
  alignment or a corresponding AFDB prediction. Inspect `afdb_status`,
  `afdb_residue_has_plddt`, and `afdb_plddt_coverage` rather than treating every
  null `afdb_plddt` as the same kind of missing value.

Caveats of the computed columns, such as RCSB/PDB-Redo atom alignment, the SASA
coordinate context and termini, and ambiguous UniProt mappings, are described
with their methods in [Scientific methods](scientific-methods.md).
