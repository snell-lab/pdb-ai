"""What each persisted run has done, and the tables it leaves behind.

Each run has a directory of its own, runs/<run ID>/ in the working directory.

storage.py names the files in a run directory and hashes a run's settings.
It also checks run IDs and output filenames, so that neither can name a path
outside the working directory.

state.py keeps the run's record: its manifest, the log of entry updates, and
its progress events. It also holds the lock that lets only one process at a
time work on the run. open_run returns the RunState of a run. A run resumes
only under the settings and the column schema that wrote its partials.

partials.py writes the tables of each finished entry as partial CSV files.

assembly.py joins the partials of each table into one CSV file, one partial at
a time.

publish.py copies the assembled tables into the working directory.

settlement.py ends a run. Its settle_run assembles the tables, publishes them
when asked, writes the run report and summary, records the status the run is
left in, and returns the build result.

listing.py summarizes every run for the Runs panel of the web app. Each poll
reads only what changed since the poll before.

Partials, tables, reports, and the manifest are each written to a scratch file
and moved into place, so a reader never sees one half-written. The entry and
event logs are appended to in place instead, and their readers skip a line
that is not yet complete.
"""

from .listing import RunSummary, list_run_summaries
from .partials import write_partial_csvs
from .publish import published_table_paths
from .settlement import PREVIEW_ROW_COUNT, settle_run
from .state import (
    ProgressCallback,
    ProgressStage,
    RunLockedError,
    RunNotResumableError,
    RunState,
    error_message,
    open_run,
    run_table_names,
)
from .storage import (
    RunDirectory,
    new_run_id,
    validate_output_filename,
    validate_run_id,
)

__all__ = [
    "PREVIEW_ROW_COUNT",
    "ProgressCallback",
    "ProgressStage",
    "RunDirectory",
    "RunLockedError",
    "RunNotResumableError",
    "RunState",
    "RunSummary",
    "error_message",
    "list_run_summaries",
    "new_run_id",
    "open_run",
    "published_table_paths",
    "run_table_names",
    "settle_run",
    "validate_output_filename",
    "validate_run_id",
    "write_partial_csvs",
]
