# Third-Party Notices

PDB-AI is distributed under the Apache License 2.0. This file covers
separately licensed software that PDB-AI installs from outside PyPI or
redistributes in this repository. `python -m pdbai.licenses` lists the license
of every runtime dependency.

## rabdam-custom

PDB-AI depends on `rabdam-custom`, an independent implementation of the
RABDAM 2 scientific method. It is installed as the `rabdam` package:

- Repository: <https://github.com/snell-lab/rabdam-custom>
- Package version checked for this notice: 1.0.0
- Dependency ref: `main`, so a new install takes the current `main`
- Copyright: Copyright (c) 2026 Nicolas Rolfe
- License: MIT

MIT License

Copyright (c) 2026 Nicolas Rolfe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Bundled RABDAM Bnet reference database

PDB-AI redistributes the canonical `rabdam-custom` 1.0.0 Bnet reference
artifact as
`pdbai/engines/analyses/rabdam/rabdam_custom_bnet_reference.csv`.

- Upstream path: `database/reference/database.csv`
- Upstream project version: 1.0.0
- SHA-256: `56f15160ca8e1c5cc598efcc5e229e02d9ef3fcde15eb5c23758f3e0154d8ed3`
- Entries: 124,948
- Database identifier: `pdb_redo_8_20_202606_strict_cryo_bnet`
- PDB-REDO snapshot: 8.20, downloaded June 2026

The reference database is derived from PDB-REDO databank entries. Credit
PDB-REDO and the original authors of the structural data, and follow the
[PDB-REDO usage policy](https://pdb-redo.eu/license). Where applicable, the
usage policy of the parent PDB archive entries also applies.

Machine-readable provenance is stored alongside the CSV in
`rabdam_custom_bnet_reference.manifest.json`.

## Alchemy

The metal-site engine (`pdbai/engines/analyses/metal_sites/`) carries over the
method of the coordination module of Alchemy: the search radius, the
first-sphere rule, the donor tables, the list of metal elements, the handling of
symmetry images and special positions, and the precision-scaled distance
comparison. Alchemy's classifications and scores are not carried over.

- Repository: <https://github.com/snell-lab/alchemy>
- Version the engine was written against: 1.0.0
- Copyright: Copyright (c) 2026 Alchemy contributors
- License: MIT, with the same text as the notice above apart from the copyright
  line

## Bundled metal-site reference tables

PDB-AI redistributes two reference tables of Alchemy 1.0.0, unchanged, beside
the metal-site engine. Machine-readable provenance is stored with them in
`reference_data.manifest.json`, and every run that executes the engine records a
receipt of each.

`pdbai/engines/analyses/metal_sites/metal_donor_distances.txt`

- Upstream path: `src/coordination/metal_distances/metal_distances_info.txt`
- SHA-256: `7f6e3a53084505dfb95b760c77daa29561c7a6d374fc53e801dd50817ebc00b3`
- Reference distances: 79
- Sources: Harding, M. M. (2006). Small revisions to predicted distances around
  metal sites in proteins. *Acta Cryst.* D62, 678-682,
  [doi:10.1107/S0907444906014594](https://doi.org/10.1107/S0907444906014594),
  for every metal except nickel, and Zheng, H., Chruszcz, M., Lasota, P.,
  Lebioda, L. & Minor, W. (2008). Data mining of metal ion environments present
  in protein structures. *J. Inorg. Biochem.* 102, 1765-1776,
  [doi:10.1016/j.jinorgbio.2008.05.006](https://doi.org/10.1016/j.jinorgbio.2008.05.006),
  for nickel.

`pdbai/engines/analyses/metal_sites/metallocofactors.txt`

- Upstream path: `src/metallocofactors/metallocofactors_id.txt`
- SHA-256: `7a5bb0aca90db06b92cbbdd1853be0c823d44c8bf6bfbaeff5f4c1663e53cd4b`
- Entries: 937, of which 70 clusters and 60 hemes
- Derived from the wwPDB Chemical Component Dictionary,
  <https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz>, as
  modified on 18 July 2026. The Chemical Component Dictionary is in the public
  domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
