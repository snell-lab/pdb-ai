"""The engines that fetch and compute the parts of a dataset.

An engine is one object with a spec and a run method. The spec declares the
engine's columns, tables, settings, report fields, and techniques. It also
declares the artifacts the engine provides, the typed values engines hand to
one another. The run method returns the engine's result for one structure.

core is the engine system itself. It holds the contracts an engine is
written against, the vocabulary of experimental techniques, the registry
that checks and orders engines, and the coordinator that runs them on a
structure. It names no engine.

sources holds the engines that fetch an entry's records from remote
databases, and analyses the engines that compute new values from a
structure.

catalog.py lists the built-in engines in display order and builds the one
registry every run draws its engines from.

compatibility.py checks a run's settings against what the registered engines
declare, and keeps a run from enabling an engine for a technique the engine
does not support.

Everything PDB-AI shows of an engine is generated from its spec, from its
command-line flags to its columns in the data dictionary. So a new engine
touches nothing outside its own package except one line in the catalog.
"""
