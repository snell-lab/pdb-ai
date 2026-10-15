from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pdbai.network import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_RETRY_AFTER_SECONDS,
    create_retry_session,
)


def _retry(**options: tuple[str, ...]) -> Retry:
    with create_retry_session(**options) as session:
        https_adapter = session.get_adapter("https://example.org")
        http_adapter = session.get_adapter("http://example.org")
    assert isinstance(https_adapter, HTTPAdapter)
    assert http_adapter is https_adapter
    return https_adapter.max_retries


@contextmanager
def _serve(answers: list[tuple[int, dict[str, str]]]) -> Generator[str]:
    """Serve each status and headers in turn, one per request, on localhost.

    Yields the server's base URL, and on leaving checks every answer was used.
    """
    pending = list(answers)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            status, headers = pending.pop(0)
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert pending == []


def test_retry_session_defaults_to_get_on_server_errors() -> None:
    retry = _retry()

    assert retry.total == 3
    assert retry.backoff_factor == 0.5
    assert retry.status_forcelist == (500, 502, 503, 504)
    assert retry.allowed_methods == ("GET",)
    assert retry.raise_on_status is False


def test_retry_session_retries_only_listed_statuses_with_retry_after() -> None:
    retry = _retry()

    assert retry.is_retry("GET", 503, has_retry_after=True)
    # urllib3 alone would retry these because of the header.
    for status in (413, 429):
        assert not retry.is_retry("GET", status, has_retry_after=True)


def test_retry_session_retries_the_methods_given() -> None:
    retry = _retry(allowed_methods=("GET", "POST"))

    assert retry.allowed_methods == ("GET", "POST")
    assert retry.is_retry("POST", 503)
    assert not _retry().is_retry("POST", 503)


def test_retry_session_caps_the_wait_retry_after_asks_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    # urllib3 waits by calling time.sleep, so replacing it records the wait
    # without taking it.
    monkeypatch.setattr(time, "sleep", sleeps.append)

    with (
        _serve([(503, {"Retry-After": "3600"}), (200, {})]) as url,
        create_retry_session() as session,
    ):
        response = session.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)

    assert response.status_code == 200
    assert sleeps == [MAX_RETRY_AFTER_SECONDS]


def test_retry_session_spends_no_retries_on_redirects() -> None:
    # Five redirects would exhaust the three retries if urllib3 counted them.
    redirects = [(302, {"Location": f"/{hop}"}) for hop in range(5)]

    with (
        _serve([*redirects, (200, {})]) as url,
        create_retry_session() as session,
    ):
        response = session.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)

    assert response.status_code == 200
    assert len(response.history) == 5
