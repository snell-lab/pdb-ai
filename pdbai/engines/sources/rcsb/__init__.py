"""RCSB's structure file and record of an entry, where every download starts.

extraction.py reads the dataset's curated fields out of a Core Entry record.
That record is RCSB's JSON description of an entry, served by its Data API.

metadata.py fetches the record through the cache and returns the fields that
extraction.py reads from it. The orchestration calls it for each entry before
any engine runs. The record names the experimental method, and the method
decides whether the entry is wanted.

engine.py registers the engine every run on downloaded entries requires. It
downloads the entry's mmCIF file, which the atom rows are parsed from.

A failed record fetch is reported, and the entry goes on without its
metadata. A failed structure download fails the entry, since every other
engine needs the atom rows.
"""

from .metadata import fetch_rcsb_metadata, rcsb_entry_url

__all__ = ["fetch_rcsb_metadata", "rcsb_entry_url"]
