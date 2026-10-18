# Web Application

The Flask application provides local dataset builds, uploads, filtering,
progress tracking, and downloads.

[Project README](../README.md)

## Quick start

```bash
python -m pdbai.webapp
```

Then open <http://localhost:8001>. See [Running the server](#running-the-server)
to change the address or port.

## Local use only

The server has no authentication. It answers only to the loopback host names
`localhost` and `127.0.0.1` (`TRUSTED_HOSTS`), which stops a DNS-rebinding page
from reaching it under its own name, and it refuses a write that a browser sends
from a page on another origin. Request bodies are capped at 512 MiB
(`MAX_CONTENT_LENGTH`). Those are hardening measures, not a substitute for
putting it behind a real WSGI server and an authenticating proxy if you expose
it.

## Features

### Choosing entries

- **PDB ID input.** Paste a list of identifiers and submit. Processing runs in a
  background thread with live progress updates. Identifiers are separated by
  commas, semicolons, or whitespace (spaces, tabs, line breaks), and the page
  counts them the way the server will.
- **Find matching entries in the PDB.** A button under the identifier box fills
  it with every current PDB entry matching the technique and X-ray source
  selected below, via one query to the RCSB Search API (`POST /api/select`).
  The "Filters" button beside it opens a checklist of the optional [range
  filters](cli.md#range-filters), grouped by quantity: minimum and maximum
  temperature in kelvin, and maximum resolution in ångströms, and under Protein
  a list of UniProt accessions, as text in any separation (see [Matching by
  UniProt accession](cli.md#matching-by-uniprot-accession)). Ticking a filter
  shows its field under the buttons, and the × beside a field removes it again,
  so the form stays short however many filters exist. The button counts the
  filters in use. A ticked filter needs a value, and the search says so rather
  than run without it. A range is strict, so an entry is kept only when every
  value it records is in range, and an entry that records none is left out.
  Pressing Enter in a filter field runs the search. The
  search runs while the request waits. Each connection attempt and each wait for
  the answer gives up after 20 seconds, so an unreachable archive shows an error
  rather than a button that never comes back. The hint reports how many entries
  matched and when. Every match is loaded. Only if a deployment sets
  `MAX_PDB_IDS` is the selection capped the same way as a build request, and the
  hint then says how many entries matched and how many of them it loaded. When
  the box still holds exactly the list the search filled in, the build records
  the search (criteria, query, time, counts) in the run manifest and the Runs
  panel labels the run "PDB search". Editing the list, or changing the technique
  or X-ray source afterwards, drops the record, since it would no longer
  describe the input, and the hint says so when a change of technique or source
  drops it. Changing a range field afterwards does not drop it, since the record
  still describes the search that filled the box.
- **Experimental technique.** A dropdown of the PDB's own method vocabulary
  (X-ray diffraction by default). A run processes one technique. Entries
  recorded under any other method are refused at intake, before anything is
  downloaded, and appear as *refused* in the run report and the Runs panel's
  progress counts. Choosing a technique switches off and locks every engine that
  does not apply to it (RABDAM and PDB-Redo outside X-ray, and wwPDB validation
  outside the methods it reports on), with a note saying so. "Select All" skips
  locked engines, and a column profile that needs a locked engine leaves it off.
  The same rule is enforced server-side, so an API request enabling a locked
  engine is rejected with `400`. `/api/techniques` lists the vocabulary and the
  per-option compatibility map.
- **X-ray source.** A second dropdown, shown only while the technique is X-ray
  diffraction, filters on the reported radiation source (all sources by default,
  or one of synchrotron, X-ray free-electron laser, rotating anode, sealed tube,
  or liquid anode). The free-electron laser option is how to select XFEL
  entries. A specific source refuses entries reporting a different source, and
  entries recording no source, with the reason in the run report. The field is
  hidden and reset for any other technique, and the server rejects the
  combination if sent anyway.
- **Direct mmCIF upload.** Upload one or more `.cif`, `.mmcif`, `.cif.gz`, or
  `.mmcif.gz` files (gzipped files are expanded on upload) when you have local
  structures. Uploads skip the RCSB and PDB-Redo downloads entirely. RABDAM, if
  enabled, scores the uploaded file itself and writes [unsuffixed column
  names](scientific-methods.md#rabdam-column-names). Each file is saved under
  its base name with every run of characters other than ASCII letters, digits,
  `.`, `_`, and `-` made one `_` (`my file.cif` becomes `my_file.cif`). A name
  with no ASCII letter or digit before the extension is refused. Two files of
  one name both become entries, numbered as the command line numbers them. The
  expanded files together may not exceed `MAX_CONTENT_LENGTH`, so a small `.gz`
  that expands past it is refused with `413` and nothing is kept. The files are
  readable only by the server's user and are deleted when the build ends. Files
  a stopped server left behind are deleted when it next starts.

### Engines and columns

- **Data sources & engines panel.** It is generated from the settings each
  engine declares, the same declarations the command line and the JSON API use.
  It holds a checkbox per engine (PDB-Redo comparison, RABDAM, SASA, electron
  density at atoms, AlphaFold Database, UniProt, and wwPDB validation) and the
  dataset-shape
  dropdown. The settings that refine an engine (RABDAM's source, and SASA's grid
  mode, point count, and bound-context policy, and the density engine's map
  source) are nested beneath its checkbox
  and disabled while the setting they refine is off. Each checkbox starts at the
  default its engine declares. Every built-in engine is off by default, so
  builds are fully opt-in. A "Select All" / "Deselect All" button flips every
  available engine at once. Captions explain fetches that a toggle combination
  performs internally (for example, RABDAM with its default source fetches
  PDB-Redo's mmCIF, while AlphaFold Database performs entity-aware UniProt
  mapping).
- **Column profiles.** Choose a pre-defined column preset from the dropdown
  (`all`, `standard`, or `minimal`). The minimal profile switches on the
  engines its columns need (except those the technique rules out) and leaves
  every other switch as it is. Every profile fills in its column filter. The
  server applies the same rule, so an API request naming only a `profileId`
  gets the profile's engines and filter, with the columns the run does not
  write left out: those of engines the technique rules out, and for `standard`
  those of engines left off. An explicit
  `columnFilterMode`/`columnFilterList` in the request wins over the profile's.
  While the profile still applies, the page sends only the `profileId`, so the
  server's filter is the one used. Editing the filter the profile filled in, or
  unticking one of its engines, takes precedence, as the page then sends the
  form as it stands without the profile. An unknown `profileId` or
  `columnFilterMode` is rejected with `400`.
- **Column filter tool.** Upload an existing dataset CSV (UTF-8, with a header
  row), name columns to include or exclude (separated by commas, semicolons, or
  line breaks, where `*` matches any run of characters, as in the build form),
  and download the filtered copy with a 25-row preview. The copy is written
  under a new name, `<stem>_filtered_<32 hex digits>.csv`, and can be downloaded
  for at least an hour. The first filter upload after that removes it. A CSV
  that is not UTF-8, cannot be parsed, has no header, names a column twice, or
  has a row with more or fewer fields than the header is refused with `400`
  naming the problem. Blank lines are skipped. An exclude filter that would
  remove every column keeps `pdb_id`, as a build does, and an include filter
  that matches nothing leaves the table unchanged with a warning.

### Runs and downloads

- **Dataset download.** A finished run's files are served straight from its run
  directory (`/api/runs/<run_id>/files/<file>`, saved as `<run_id>_<file>`),
  with a preview of the first 25 rows shown inline in a scrolling table whose
  column names stay in view. Nothing is copied elsewhere. "Download All (.zip)",
  shown when a run has more than one table, bundles every output table with the
  run report, readable run summary, and machine-readable `manifest.json`. The
  Runs panel's Download link fetches the same archive for any run with a
  dataset. Individual download links are also available for each table, the run
  report, and the run summary. Tables are written to a scratch file and moved
  into place, so a download never catches one half-written. On Windows, avoid
  downloading a run's files while that run is being resumed, since an open
  download can make the run's final write fail.
- **Runs panel.** Runs are listed by when their status last changed. A run
  recorded as running that no process is building, as after its server or
  command stopped, is shown as *stale*, even when every entry had been
  processed, and a page still polling it stops and says so. A run built from PDB
  IDs that is paused, errored, or stale can be resumed from the panel. One built
  from uploaded files is started again by uploading them, since the panel has no
  files to send. A build that fails is shown as *error* with the reason under
  the badge. A run is held by one worker at a time across processes. A run the
  command line or another server is building is shown as running with "Being
  built by another process." and no Pause, Resume, or Delete, and Pause reaches
  only a build this server started. Deleting a run asks for confirmation and
  holds the run for the whole deletion. Errors from the listing and from the
  panel's actions appear in a status line under its heading. The panel refreshes
  every 1.5 s while a run is live and every 8 s otherwise, never while the tab
  is hidden. Unticking "Auto-refresh while running" stops it refreshing on its
  own, and Refresh still works.

## Running the server

`python -m pdbai.webapp` reads these environment variables:

| Variable | Accepted values | Default |
|---|---|---|
| `PDBAI_HOST` | The address to bind. `0.0.0.0` or `::` listens on every interface. | `127.0.0.1` |
| `PDBAI_PORT` | A whole number from 1 to 65535. | `8001` |
| `PDBAI_DEBUG` | `true` or `false`, in any case. Unset or empty means `false`. | `false` |
| `PDBAI_TRUSTED_HOSTS` | More host names the app answers to, separated by commas. | none |

A bad `PDBAI_PORT` or `PDBAI_DEBUG` (`1`, `yes`, and `on` included) stops the
server with one `error:` line and exit status 2, before any workspace directory
is created. A specific `PDBAI_HOST` other than loopback is added to the trusted
host names. A wildcard adds none, so a request naming the machine from elsewhere
is refused with `400` until `PDBAI_TRUSTED_HOSTS` lists the names or addresses
clients use, and the server logs a warning when it does not. For example, run
`PDBAI_HOST=0.0.0.0 PDBAI_TRUSTED_HOSTS=lab-box.local,192.168.1.5 python -m
pdbai.webapp`. A name starting with a dot, such as `.example.org`, trusts every
subdomain. IPv6 address literals are never matched as host names, because
Werkzeug compares a trusted name only up to its first colon, so a client on IPv6
loopback browses to `localhost`. Binding an IPv6 address still works.

`PDBAI_DEBUG=true` turns on Flask's debug mode, which reloads the page template
when it changes. Every error is still a JSON response and the detail goes to the
server log. Debug mode also enables Werkzeug's PIN-protected interactive
debugger console, which the app's host and origin checks do not cover, so keep
debug mode to loopback use. The code reloader is never enabled, because a
restart would stop running builds and leave their runs stale.

Stopping the server does not pause running builds. A build running when the
process ends is left stale, and a resume continues it from its last recorded
entry.

### Configuration

A WSGI deployment or a library caller passes settings to `create_app`, for
example `app = create_app({"WORKING_DIRECTORY": "/srv/pdbai", "MAX_PDB_IDS":
5000})`. `python -m pdbai.webapp` sets only `TRUSTED_HOSTS`. Serve
`create_app(...)` from your own entry point to set the others.

| Key | Meaning | Default |
|---|---|---|
| `WORKING_DIRECTORY` | The workspace, which holds runs, caches, uploads, and scratch archives. | `./workspace` |
| `DATASET_DIRECTORY` | Where the column filter tool writes its CSVs, served at `/downloads/<name>`. | `<WORKING_DIRECTORY>/datasets` |
| `MAX_PDB_IDS` | The most distinct PDB IDs one build, or one archive search, may name. | no cap |
| `MAX_UPLOAD_FILES` | The most files one upload may hold. | no cap |
| `TRUSTED_HOSTS` | The host names the app answers to. An empty list trusts every host. | `["localhost", "127.0.0.1"]` |
| `ALLOWED_ORIGINS` | Origins besides the app's own that may send writes, such as `"https://ui.example"`. | none |
| `MAX_CONTENT_LENGTH` | The largest request body, in bytes. | 512 MiB |

Both directories are created. The directories and caps are read once, when the
app is created, so changing `app.config` afterwards has no effect on them. A
directory that is not a `str` or `Path`, a cap that is neither `None` nor a
positive whole number, or an `ALLOWED_ORIGINS` that is not a list of strings
raises `ValueError`, and the app does not start. Flask also caps a multipart
form at 1,000 parts (`MAX_FORM_PARTS`), a limit the form's other fields share,
so one upload holds fewer than 1,000 files whatever `MAX_UPLOAD_FILES` says, and
a larger form is refused with `413`. Behind a TLS-terminating proxy, wrap the
app in Werkzeug's `ProxyFix` so it sees the https origin, or list that origin in
`ALLOWED_ORIGINS`, and list the public host name in `TRUSTED_HOSTS`. Otherwise
every write from the browser is refused with `403`.

## HTTP API

Every error is a JSON object, `{"error": "<sentence>"}`, with the status code,
and never names a local path. An unexpected failure is `500` `{"error":
"Internal server error."}` with the detail in the server log. A Host header
naming no trusted host is `400`. A write (POST, PUT, PATCH, or DELETE) whose
`Origin` header is neither the app's own origin nor in `ALLOWED_ORIGINS` is
`403`, `Origin: null` included. A request without `Origin`, as from curl or a
script, is allowed. `/api/build` and `/api/select` require `Content-Type:
application/json` (`415` otherwise) and a body that is a JSON object (`400`
otherwise). Every route that names a run answers a malformed run ID with `400`,
and every one but pause answers an ID that names no run directory with `404`
"Run not found.".

**Request fields** of `/api/build` (JSON) and `/api/build-upload` (form):

- `technique`, `xraySource`, `datasetShape`, `profileId`, `columnFilterMode`,
  and each engine setting under its camelCase name (`includeSasa`,
  `rabdamSource`, `sasaAccuracyMode`, and so on). A missing, null, or blank
  value takes the default, a choice ignores case and surrounding spaces, and a
  flag is a JSON boolean or the text `true`/`false` in any case.
  `columnFilterList` is text of names separated by commas, semicolons, or line
  breaks, or a list of such text, and is given together with `columnFilterMode`.
  A value of the wrong type is `400` naming the field.
- `runId`: a run ID string. Missing, null, or `""` starts a new run under a
  fresh ID. Any other value that is not a valid run ID is `400`.
- `resume`: a flag, false by default. It requires `runId`.

**Endpoints:**

- `POST /api/build` builds from PDB IDs. `pdbIds` is text split on whitespace
  (tab, line break, space), commas, and semicolons, or a list with one
  identifier per item. Case and surrounding spaces are ignored, and a repeated
  ID is built once. `selection`, optional, is the record `/api/select` returned,
  sent back unchanged. It must list exactly the request's distinct IDs, sorted,
  and carry the same technique and X-ray source, and it is stored in the
  manifest as the cohort's provenance. `200` `{"runId": runId}`. Poll
  `/api/runs/<runId>/status`. `400` for no or an invalid PDB ID, more distinct
  IDs than `MAX_PDB_IDS`, invalid settings, a malformed selection or one
  describing other IDs or settings, or `resume` without `runId`. `404` for
  `resume` of a run that does not exist. `409` for a `runId` naming an existing
  run without `resume`, or a run another build or a deletion holds. A refused
  request creates no run directory and writes no job state.
- `POST /api/build-upload` builds from uploaded files, sent as the multipart
  field `mmcifFiles` with the request fields above. A part without a file name
  is ignored. `200` `{"runId": runId}`. `400` for no files, more than
  `MAX_UPLOAD_FILES`, invalid settings, `resume` without `runId`, a name not
  ending in `.cif` or `.mmcif` (optionally `.gz`) or with no ASCII letter or
  digit, or an unreadable gzip file. `404` for `resume` of a run that does not
  exist. `409` for a `runId` naming an existing run without `resume`, or a run
  that a job of this server or another process holds, or that a deletion holds.
  These run checks are the same as for `/api/build` and come before any file is
  saved. `413` when the request, or its files once expanded, pass
  `MAX_CONTENT_LENGTH`, or the form has more than `MAX_FORM_PARTS` parts. With
  `resume`, `runId` names an upload run to continue and the same files must be
  sent again, since the run records each file's name, size, and SHA-256 and
  refuses others. The Runs panel cannot do this.
- `POST /api/select` searches the archive. `technique` and `xraySource` as
  above, and `limit`, a positive whole number or its text (missing, null, or
  blank means none). The [range filters](cli.md#range-filters)
  `minTemperatureK`, `maxTemperatureK`, and `maxResolutionA` are each a positive
  number or its plain decimal text, and `uniprotAccessions` is text or a list of
  text holding UniProt accessions in any separation (missing, null, or blank
  means the filter is not set). `MAX_PDB_IDS`, when set, caps the search too.
  `200` with
  `selection`, the snake_case provenance record whose `pdb_ids` is the only copy
  of the matching identifiers (sorted and distinct), beside `description`,
  `totalCount`, `returnedCount`, `truncated`, `retrievedAtIso`, `warnings`, and
  `maxPdbIds` (null when uncapped). To build from it, post `selection.pdb_ids`
  as `pdbIds` with the same technique and source and the record as `selection`.
  `400` for an invalid field, or for a minimum above its maximum, `502` when the
  archive cannot be searched.
- `POST /api/filter-upload` filters a CSV, sent as the multipart field `file`
  with optional `columnFilterMode` (`include` or `exclude`) and
  `columnFilterList`. `200` `{"csvUrl": "/downloads/<name>", "summary":
  {"rowCount", "columnCount"}, "warnings", "preview", "columnFilter"}`, where
  `preview` holds the first 25 rows as objects and `columnFilter` is null
  without a filter. `400` for no file, a mode without a list or a list without a
  mode, or a CSV refused as above.
- `GET /api/runs/<runId>/status` returns `{runId, status, progress, result,
  error}`. `status` is `running`, `paused`, `completed`, `error`, `stale`, or
  `unknown`. For a run this server no longer holds in memory it is the
  manifest's, as in `/api/runs`. `progress` is `{stage, current, total,
  subject}` or null, where `subject` names what the stage is working on: the
  entry's ID while building, a partial table's file name or `dataset` while
  assembling, and `dataset` or the published file's name while publishing.
  `result` is null until the build ends, then holds `csvUrl`,
  `additionalCsvUrls` (table name to link), `runReportUrl`, and `runSummaryUrl`,
  the download links, beside `summary` (`pdbCount` entries in the dataset,
  `requestedCount` entries the run names, `atomCount`, `columnCount`),
  `warnings`, `preview`, and `columnFilter`. Each link carries the path prefix
  the app was served under (its `SCRIPT_NAME`, as behind a proxy that mounts it
  under a path) when the request that started the build came in. `error` is the
  build's own message, which may name local paths as the manifest does. `404`
  when neither the manifest nor `job.json` records a status.
- `GET /api/runs` returns `{"runs": [...]}`, the most recently changed first,
  each with `runId`, `status`, `createdAt` and `updatedAt` (Unix seconds),
  `settings`, `pdbCount`, `selection`, `counts` (`done`, `failed`, `refused`,
  `pending`, `total`), `lastProgress`, `hasDataset`, `error`, and `live`, which
  is true while a job of this server builds the run. A run another process
  builds is `status: "running"` with `live: false`.
- `GET /api/runs/<runId>/files/<file>` downloads one of the tables the run's
  manifest names, `run_report.csv`, or `run_summary.md`, as `<runId>_<file>`.
  Any other name, a missing file, or a symbolic link is `404` "File not found.".
- `GET /api/runs/<runId>/download-all` downloads `pdb_dataset_<runId>.zip`.
  `404` when no table is assembled yet, `409` when a file changes while the
  archive is built.
- `POST /api/runs/<runId>/pause` returns `{"status": "pausing", "runId"}`. The
  build stops after the entry in hand and settles as paused. `409` when this
  server is not building the run.
- `POST /api/runs/<runId>/resume` returns `{"status": "resumed", "runId"}`.
  `409` when the run is held here or by another process, was built from uploaded
  files, records no PDB IDs, or has stored settings that no longer validate.
  `404` when its manifest is missing or unreadable.
- `DELETE /api/runs/<runId>` returns `{"status": "deleted", "runId"}`. `409`
  when the run is held here or by another process, `500` when the directory
  cannot be removed.
- `GET /api/techniques` and `GET /api/profiles` return the technique vocabulary
  with its compatibility map and the column profiles, the same data the page
  embeds. `GET /downloads/<name>` serves a column-filtered CSV, named
  `<stem>_filtered_<32 hex digits>.csv`, as an attachment. Any other name is
  `404`.
