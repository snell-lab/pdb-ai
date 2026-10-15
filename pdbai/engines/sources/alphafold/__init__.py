"""AlphaFold Database confidence for the residues of a PDB entry.

models.py holds the records the other modules pass between them, the
statuses they report, and the paths of their cache files.

references.py asks RCSB which UniProt sequences each polymer entity of an
entry aligns to, and over which residues.

model_fetch.py fetches the AlphaFold predictions of one UniProt accession,
chooses the family of them that describes one protein, and reads the pLDDT
and the name of each residue out of the model files.

fetch.py runs both for an entry and sums the outcome up in one status and
one failure reason.

artifacts.py names the entry result the acquisition engine hands to the
annotation engine.

annotation.py maps the entry result onto the atom table, residue by residue.

engine.py declares the two engines, their settings, and their columns.

The package reads published predictions and never runs AlphaFold. A residue
receives a score only when RCSB aligns it to exactly one UniProt position
and the model has the same amino acid there. Only complete lookups are
cached, so the next run retries every failure.
"""

__all__: list[str] = []
