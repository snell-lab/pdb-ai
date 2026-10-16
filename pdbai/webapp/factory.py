"""The Flask application factory, its request guards, and the page routes.

create_app reads its configuration once, before it registers a route. A bad
value therefore stops the app at startup, not when a request needs it. The
routes here serve the page, the technique and profile vocabularies, and the
column-filtered tables. The API routes are registered from their own modules.

Every request passes two guards before a route sees it. The Host header must
name a trusted host, which stops a DNS-rebinding page from reaching the
server under its own name. A write that carries an Origin header must come
from the app's own origin or from one the deployment allows. Every HTTP
error and every unhandled exception becomes a JSON {"error": message}
response, and that response never names a local path.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask.typing import ResponseReturnValue
from werkzeug.exceptions import HTTPException

from ..column_filtering import COLUMN_FILTER_MODES
from ..engines.catalog import get_engine_registry
from ..engines.compatibility import toggle_compatibility
from ..engines.core import (
    ALL_XRAY_SOURCES,
    DATASET_SHAPES,
    DEFAULT_DATASET_SHAPE,
    DEFAULT_TECHNIQUE,
    TECHNIQUES,
    XRAY_SOURCE_TECHNIQUE,
    XRAY_SOURCES,
)
from ..positive_numbers import parse_positive_int
from ..profiles import PROFILES
from ..runs import PREVIEW_ROW_COUNT, ProgressStage
from ..selection_filters import SEARCH_FILTERS, SEARCH_QUANTITIES, UNIPROT_ACCESSIONS
from .build_routes import register_build_routes
from .jobs import JobService
from .route_helpers import camel_keys
from .run_routes import register_run_routes
from .select_routes import register_select_routes
from .upload_routes import (
    ACCEPTED_UPLOAD_SUFFIXES,
    is_filtered_table_name,
    register_upload_routes,
)

# A request body past this size is refused before it is read, so one upload
# cannot fill the disk.
_DEFAULT_MAX_CONTENT_LENGTH = 512 * 1024 * 1024
# These methods only read, so a foreign page gains nothing by sending them.
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_JOB_SERVICE_KEY = "pdbai_jobs"

# Werkzeug compares a trusted name only up to its first colon, so it never
# matches an IPv6 literal such as [::1]. A client on IPv6 loopback uses
# localhost instead.
LOOPBACK_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1")

# These are the page's words for each build stage. The command line keeps its
# own words, which finish a sentence that names the entry. Intake reports a
# refusal, not work under way, so app.js shows its label without "in
# progress".
_STAGE_LABELS: dict[ProgressStage, str] = {
    "download": "Downloading data",
    "intake": "Refused at technique intake",
    "parse": "Parsing atom-site data",
    "assemble": "Assembling the dataset",
    "finalize": "Writing the dataset",
}
# These are the page's words for the star-schema tables it links for download.
# The shape declares only the tables' names, so every table needs a label here.
_TABLE_LABELS = {
    "atoms": "atom-level data",
    "residues": "residue-level data",
    "chains": "chain-level data",
    "structures": "structure-level data",
    "citations": "citation & publication metadata",
}


def job_service(app: Flask) -> JobService:
    """Return the background-job service ``create_app`` attached to ``app``."""
    return cast(JobService, app.extensions[_JOB_SERVICE_KEY])


def _configured_path(config: Mapping[str, object], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{key} must be a path; got {value!r}.")
    return Path(value)


def _configured_cap(config: Mapping[str, object], key: str) -> int | None:
    # None means no cap. Any other value must be a positive whole number.
    value = config.get(key)
    return None if value is None else parse_positive_int(value, key)


def _allowed_origins(config: Mapping[str, object]) -> frozenset[str]:
    # A string is refused, because iterating it would allow each character.
    value = config.get("ALLOWED_ORIGINS")
    if value is None:
        return frozenset()
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError(f"ALLOWED_ORIGINS must be a list of origins; got {value!r}.")
    origins = cast(Iterable[object], value)
    for origin in origins:
        if not isinstance(origin, str):
            raise ValueError(f"ALLOWED_ORIGINS must hold only text; got {origin!r}.")
    return frozenset(origin.rstrip("/") for origin in cast(Iterable[str], origins))


def _is_downloadable(dataset_directory: Path, filename: str) -> bool:
    # Only a name the filter route gives a table passes. Such a name never
    # starts with a dot, so the scratch copy the filter route writes while it
    # works is never served.
    if not is_filtered_table_name(filename):
        return False
    path = dataset_directory / filename
    return not path.is_symlink() and path.is_file()


def _technique_payload() -> dict[str, object]:
    # The page embeds this payload, and /api/techniques returns it. app.js
    # reads its camelCase keys.
    return camel_keys(
        {
            "default": DEFAULT_TECHNIQUE,
            "techniques": [camel_keys(technique.to_dict()) for technique in TECHNIQUES],
            "compatibility": [
                camel_keys(record.to_dict()) for record in toggle_compatibility()
            ],
            "xray_sources": {
                "all": ALL_XRAY_SOURCES,
                "technique": XRAY_SOURCE_TECHNIQUE,
                "sources": [camel_keys(source.to_dict()) for source in XRAY_SOURCES],
            },
        }
    )


def _profile_payloads() -> list[dict[str, object]]:
    # The page embeds this payload, and /api/profiles returns it.
    return [camel_keys(profile.to_dict()) for profile in PROFILES]


def _page_config() -> dict[str, object]:
    """Return the settings the page embeds as JSON for app.js to read at start.

    The URLs come from the app's own rules, so they follow the prefix the app
    is served under. Each run URL holds a RUN_ID placeholder for app.js to
    replace. Everything else is a vocabulary or a limit the server owns, and
    the page never repeats one. Calling this outside a request raises
    RuntimeError.
    """
    return {
        "urls": {
            "build": url_for("build"),
            "buildUpload": url_for("build_upload"),
            "select": url_for("select"),
            "filterUpload": url_for("filter_upload"),
            "runs": url_for("list_runs"),
            "runStatus": url_for("run_status", run_id="RUN_ID"),
            "runDownloadAll": url_for("download_run_archive", run_id="RUN_ID"),
            "runPause": url_for("pause_run", run_id="RUN_ID"),
            "runResume": url_for("resume_run", run_id="RUN_ID"),
            "runDelete": url_for("delete_run", run_id="RUN_ID"),
        },
        "techniques": _technique_payload(),
        "profiles": _profile_payloads(),
        "stageLabels": dict(_STAGE_LABELS),
        "tableLabels": {
            table: _TABLE_LABELS[table]
            for table in DATASET_SHAPES["star_schema"].tables
        },
        "previewRows": PREVIEW_ROW_COUNT,
        "acceptedSuffixes": list(ACCEPTED_UPLOAD_SUFFIXES),
    }


def create_app(config_overrides: Mapping[str, object] | None = None) -> Flask:
    """Return a configured Flask application for the web interface.

    config_overrides is applied to the configuration before any of it is
    read. WORKING_DIRECTORY defaults to ``workspace`` in the current
    directory, and DATASET_DIRECTORY to ``datasets`` inside it. Both are
    created. TRUSTED_HOSTS lists the host names Werkzeug lets through, and
    defaults to the loopback names. ALLOWED_ORIGINS lists the origins besides
    the app's own that may send writes. MAX_PDB_IDS and MAX_UPLOAD_FILES cap
    the entries of one request, and there is no cap by default.
    MAX_CONTENT_LENGTH caps a request body in bytes, at 512 MiB by default.

    A directory that is not a path raises ValueError. So does a cap that is
    neither None nor a positive whole number, and an origin list that is not
    a list of strings.
    """
    webapp_root = Path(__file__).parent
    app = Flask(
        __name__,
        template_folder=str(webapp_root / "templates"),
        static_folder=str(webapp_root / "static"),
    )

    config = cast(dict[str, object], app.config)
    if config_overrides is not None:
        config.update(config_overrides)
    config.setdefault("WORKING_DIRECTORY", Path.cwd() / "workspace")
    working_directory = _configured_path(config, "WORKING_DIRECTORY")
    config.setdefault("DATASET_DIRECTORY", working_directory / "datasets")
    # Flask fills in these two keys with None, so setdefault would leave them
    # unset.
    if config.get("MAX_CONTENT_LENGTH") is None:
        config["MAX_CONTENT_LENGTH"] = _DEFAULT_MAX_CONTENT_LENGTH
    # A DNS-rebinding page reaches this server under the page's own host name.
    # The Origin check below cannot tell its writes from same-origin ones.
    # Werkzeug therefore refuses any Host header that names no trusted host.
    if config.get("TRUSTED_HOSTS") is None:
        config["TRUSTED_HOSTS"] = list(LOOPBACK_HOSTS)
    allowed_origins = _allowed_origins(config)

    service = JobService(
        working_directory=working_directory,
        dataset_directory=_configured_path(config, "DATASET_DIRECTORY"),
        max_pdb_ids=_configured_cap(config, "MAX_PDB_IDS"),
        max_upload_files=_configured_cap(config, "MAX_UPLOAD_FILES"),
    )
    service.working_directory.mkdir(parents=True, exist_ok=True)
    service.dataset_directory.mkdir(parents=True, exist_ok=True)
    app.extensions[_JOB_SERVICE_KEY] = service

    @app.before_request
    def _refuse_cross_origin_writes() -> None:
        # A request without an Origin header is let through, for scripts and
        # command-line clients. A browser attaches Origin to every
        # cross-origin write, so no browser write can skip this check.
        if request.method in _SAFE_METHODS:
            return
        origin = request.headers.get("Origin")
        if not origin:
            return
        if origin.rstrip("/") not in {request.host_url.rstrip("/")} | allowed_origins:
            abort(403, f"Cross-origin requests are not allowed; got {origin!r}.")

    @app.errorhandler(HTTPException)
    def _http_error_as_json(error: HTTPException) -> ResponseReturnValue:
        # The exception's own headers are kept, such as Allow on a 405, and
        # only its HTML Content-Type is replaced. Flask handles an exception
        # without a code before any handler is called, so the 500 fallback
        # only satisfies the int | None type.
        headers = [
            (name, value)
            for name, value in error.get_headers()
            if name.lower() != "content-type"
        ]
        return jsonify({"error": error.description}), error.code or 500, headers

    @app.errorhandler(Exception)
    def _unhandled_error_as_json(error: Exception) -> ResponseReturnValue:
        # The detail is logged, not returned, because it may name local paths.
        app.logger.exception("Unhandled error serving %s", request.path, exc_info=error)
        return jsonify({"error": "Internal server error."}), 500

    @app.route("/")
    def index() -> str:
        """Render the page, with the configuration app.js reads embedded."""
        return render_template(
            "index.html",
            page_config=_page_config(),
            engine_options=get_engine_registry().option_specs(),
            techniques=TECHNIQUES,
            default_technique=DEFAULT_TECHNIQUE,
            xray_sources=XRAY_SOURCES,
            all_xray_sources=ALL_XRAY_SOURCES,
            search_filters=SEARCH_FILTERS,
            search_quantities=SEARCH_QUANTITIES,
            accession_filter=UNIPROT_ACCESSIONS,
            profiles=PROFILES,
            column_filter_modes=COLUMN_FILTER_MODES,
            accepted_suffixes=ACCEPTED_UPLOAD_SUFFIXES,
            dataset_shapes=DATASET_SHAPES.values(),
            default_dataset_shape=DEFAULT_DATASET_SHAPE,
        )

    @app.route("/favicon.ico")
    def favicon() -> ResponseReturnValue:
        """Redirect to the page's SVG icon."""
        # Browsers ask for this path before they read the page's icon link.
        return redirect(url_for("static", filename="favicon.svg"))

    @app.route("/api/profiles")
    def list_profiles() -> ResponseReturnValue:
        """Return the column profiles, as the page embeds them."""
        return jsonify(_profile_payloads())

    @app.route("/api/techniques")
    def list_techniques() -> ResponseReturnValue:
        """Return the technique vocabulary and its compatibility map."""
        return jsonify(_technique_payload())

    @app.route("/downloads/<filename>")
    def download_filtered_table(filename: str) -> ResponseReturnValue:
        """Send a filtered table as an attachment.

        Any other name is a 404.
        """
        if not _is_downloadable(service.dataset_directory, filename):
            abort(404, "File not found.")
        return send_from_directory(
            service.dataset_directory, filename, as_attachment=True
        )

    register_build_routes(app, service)
    register_select_routes(app, service)
    register_upload_routes(app, service)
    register_run_routes(app, service)
    return app


__all__ = ["LOOPBACK_HOSTS", "create_app", "job_service"]
