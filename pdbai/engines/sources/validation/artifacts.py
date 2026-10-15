"""The value the wwPDB validation source hands to other engines.

VALIDATION_REPORT holds the entry's parsed validation report and the status
that says how it was obtained. The source engine publishes it whenever it
runs to the end. It does so even when the report is missing or broken, so
the status can say why. The value is absent only when the source engine
itself failed.

The validation annotation engine reads it and turns it into columns.
"""

from __future__ import annotations

from ...core import ArtifactKey
from .fetch import ValidationReportResult

VALIDATION_REPORT = ArtifactKey(
    "validation_report",
    ValidationReportResult,
    "Parsed wwPDB validation metrics for the entry and how they were obtained.",
)

__all__ = ["VALIDATION_REPORT"]
