"""The two reports a run writes when it settles.

run_report.py writes run_report.csv. It has one row for each entry the run
set out to process, in order. An entry refused at intake or never reached has
a row too. A row gives what the host did with the entry, what each engine
reported, and the entry's row count in each engine table.

column_statistics.py computes the per-column statistics of one table. It
reads the table one chunk of rows at a time, and it stops counting the
distinct values of a column after MAX_DISTINCT_COUNT. Memory therefore stays
bounded for a dataset of any size.

summary.py writes run_summary.md. It records the run's settings and
reproducibility log, the status of each entry, the engine tables, and a data
dictionary of every assembled table, which shows those statistics.

Both reports are written even when no table was assembled, so a run that
produced no data still says why. Each report replaces its file atomically.
"""

from .run_report import write_run_report
from .summary import write_run_summary

__all__ = [
    "write_run_report",
    "write_run_summary",
]
