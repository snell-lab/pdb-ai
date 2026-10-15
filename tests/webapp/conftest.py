"""Helpers shared by the web application tests.

The test modules import these by name rather than as fixtures. A client
runs with the working directory set to the test's temporary directory,
because create_app keeps its workspace in the working directory.
"""

from __future__ import annotations

import time
from pathlib import Path

from flask.testing import FlaskClient
from pytest import MonkeyPatch

from pdbai.engines.core import EngineOptions
from pdbai.webapp import create_app
from pdbai.webapp.jobs import JobService
from pdbai.webapp.options import settings_from_request

__all__ = ["_client", "_options", "_wait_for_job"]


def _client(tmp_path: Path, monkeypatch: MonkeyPatch) -> FlaskClient:
    """Return a test client of an app whose workspace is under ``tmp_path``."""
    monkeypatch.chdir(tmp_path)
    app = create_app()
    app.testing = True
    return app.test_client()


def _options(payload: dict[str, object]) -> EngineOptions:
    """Return the engine options a download request with ``payload`` gives."""
    return settings_from_request(payload, source="download").options


def _wait_for_job(service: JobService, run_id: str) -> None:
    """Wait up to 10 seconds for the run's job to end, or raise AssertionError.

    A job stays live until its final state is recorded and its claim is
    released.
    """
    deadline = time.monotonic() + 10.0
    while run_id in service.live_run_ids():
        if time.monotonic() > deadline:
            raise AssertionError(f"The job of run {run_id!r} did not finish.")
        time.sleep(0.01)
