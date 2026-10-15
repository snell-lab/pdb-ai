"""wwPDB validation reports as residue and structure quality columns.

parse.py reads a validation XML report. It keeps each residue's density fit
and its Ramachandran and rotamer classes, for the residue and for each of
its alternate conformers. It also keeps the entry's clashscore and outlier
percentages.

fetch.py downloads an entry's report from the RCSB archive, parses it, and
caches the parsed values. Every outcome is a result with a status, never an
exception.

artifacts.py names the artifact that carries the parsed report from the
source engine to the annotation engine.

engine.py holds the two engines. The source engine obtains the report. The
annotation engine finds each atom's residue by its author identifiers and
copies the residue's values onto the atom.

maps.py holds a third engine, which downloads the coefficients of the two
maps the validation pipeline calculated about the deposited model. It runs
only for the density engine, when that engine reads the wwPDB maps.

A residue modelled in alternate conformers is listed in the report once per
conformer, but the structure parser keeps only one conformer of each
residue. parse_validation_xml therefore keeps each conformer under a key of
its own, beside the residue's key, which holds one conformer chosen by the
rule its docstring gives. The annotation engine gives a residue's atoms the
values of the conformer the structure parser kept, found by the first
alternate location identifier among them. Atoms without an identifier
belong to that conformer too. When the residue has no identifier, or the
report does not list its conformer, the atoms take the residue key's values.

A missing, unreachable, or unreadable report never fails a structure. The
validation columns stay empty, and validation_status in the run report says
which of these it was.
"""
