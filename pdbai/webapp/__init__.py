"""The local web interface, a page and a JSON API that build and manage runs.

route_helpers.py checks request bodies and run IDs, finds existing runs,
makes the API's camelCase keys, and deletes stale scratch files.

options.py reads a request's fields into validated run settings, by one rule
for blanks, case, and types that holds for every field.

jobs.py runs each build on a background thread and keeps the job's state
until it is written to the run's job.json.

build_routes.py starts a build from PDB identifiers, with the provenance
record of the archive search that found them when there was one.

select_routes.py searches the archive for the entries that match a technique
and X-ray source.

upload_routes.py starts a build from uploaded mmCIF files, and writes the
column-filtered copy of an uploaded CSV.

run_routes.py reports a run's status, lists the runs, serves their files and
archives, and pauses, resumes, and deletes them.

factory.py builds the Flask application, guards every request's host and
origin, and serves the page with the configuration its script reads.

__main__.py runs the development server, configured from the environment.

A build request is checked in full before anything is written, so a refused
one leaves no run directory, upload, or job state behind. A run is built by
one worker at a time across processes, and a build from PDB identifiers or a
resume naming a run that another process holds is refused with 409 before a
job starts. Every error reaches the client as a JSON {"error": message}, and
local paths go to the log, never into a response.
"""

from .factory import create_app

__all__ = ["create_app"]
