"""The engines that fetch an entry's records from remote databases.

scalars.py reads numbers and text out of responses. A value that is not a
plain decimal number reads as None, so one malformed field leaves one cell
empty.

http_cache.py keeps every response and downloaded file on disk and decides
when one is too old to reuse. Its module docstring sets out that policy.

rcsb downloads the entry's mmCIF file, which every run on downloaded entries
requires. It also fetches the entry's Core Entry record, which the
orchestration reads before any engine runs.

pdb_redo fetches and parses PDB-Redo's re-refined model and compares it with
the RCSB model atom by atom.

validation fetches the entry's wwPDB validation report and adds its residue
and structure quality metrics.

alphafold maps each residue to its UniProt sequence through RCSB's polymer
entities. It also adds the confidence of the AlphaFold Database model of
that sequence.

Each acquisition engine runs only for an entry downloaded by its PDB ID.
It names every file it asked for, with the file's URL and whether the cache
served it. The run manifest records these, so each value can be traced to a
source.
"""

__all__: list[str] = []
