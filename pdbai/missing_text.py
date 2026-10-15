"""The texts that stand for a missing value wherever text is read.

mmCIF writes "?" for a value that is unknown and "." for one that does not
apply. A column converted to text before it is read turns a missing value
into "nan", "none", or "<na>". A quoted empty value reads as the empty text.
The readers of identity columns, of flags, and of engine table keys all take
these texts as missing, so that no two of them disagree about whether a
value is there. The texts are lower-case, and a reader strips and
lower-cases a value before looking it up.
"""

from __future__ import annotations

MISSING_TEXTS = frozenset({"", ".", "?", "nan", "none", "<na>"})

__all__ = ["MISSING_TEXTS"]
