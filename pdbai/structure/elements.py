"""The element symbols that every reader of atom rows treats alike.

Every reader that leaves hydrogens out, the structural readers and the
engines alike, does so by this one set. A neutron model's deuterium, or a
tritium label, is then skipped wherever a hydrogen is.
"""

from __future__ import annotations

# Hydrogen and its isotopes deuterium and tritium, by upper-case element
# symbol. An atom without a symbol is none of them.
HYDROGEN_ELEMENTS: frozenset[str] = frozenset({"H", "D", "T"})

__all__ = ["HYDROGEN_ELEMENTS"]
