"""Retrying HTTP sessions, their default timeout, and the network logger.

Every download of a build and the archive search go through a session made
by create_retry_session, so a transient failure is retried by the same rule
wherever it happens.

DEFAULT_TIMEOUT_SECONDS bounds each connection attempt and each wait for the
next part of an answer. It does not bound the whole request, so it stops a
stalled service without cutting off a slow download. A session has no
timeout of its own, so every request must pass one. A caller that makes a
session owns it and closes it. Code handed a session, through EngineServices
or search_archive, uses it as given and leaves it open for its owner.

The network logger, pdbai.network, carries progress messages such as
searches, requests, and cache hits. This module installs no handler for it.
The command line and the web app print it through console_logging, and
tests capture it by name.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if TYPE_CHECKING:
    # BaseHTTPResponse is new in urllib3 2, and only annotations need it.
    from urllib3.response import BaseHTTPResponse

NETWORK_LOGGER_NAME = "pdbai.network"
network_logger = logging.getLogger(NETWORK_LOGGER_NAME)

DEFAULT_TIMEOUT_SECONDS = 30.0
_RETRY_TOTAL = 3
_BACKOFF_FACTOR = 0.5
_RETRY_STATUSES = (500, 502, 503, 504)
# The request timeout does not cover the sleep a Retry-After header asks for.
# Unbounded, that sleep lets one answer stall a download worker for as long as
# the server says, an hour for "Retry-After: 3600". A minute covers the brief
# overloads the header usually reports. Three retries then wait at most three
# minutes in all.
MAX_RETRY_AFTER_SECONDS = 60.0


class _BoundedRetry(Retry):
    """A Retry that caps Retry-After and retries only the listed statuses."""

    # By default urllib3 retries a 413, 429, or 503 that carries Retry-After,
    # even when status_forcelist leaves it out. With this set empty, the list
    # is the whole rule. A listed status still obeys the header.
    RETRY_AFTER_STATUS_CODES = frozenset[int]()

    def get_retry_after(self, response: BaseHTTPResponse) -> float | None:
        """Return the wait Retry-After asks for, at most the cap above."""
        retry_after = super().get_retry_after(response)
        if retry_after is None:
            return None
        return min(retry_after, MAX_RETRY_AFTER_SECONDS)


def create_retry_session(
    *, allowed_methods: tuple[str, ...] = ("GET",)
) -> requests.Session:
    """Create a requests session that retries transient failures.

    A request that cannot connect is retried whatever its method, since
    nothing reached the server. A read timeout, a connection dropped before
    the answer begins, and a 500, 502, 503, or 504 are retried only for
    ``allowed_methods``. That is GET alone by default, since a POST may
    already have taken effect. No other status is retried, whatever
    Retry-After says. Three retries are allowed, four attempts in all. The
    first retry is immediate and the next two wait 1 and 2 seconds. A
    Retry-After header from the server is obeyed instead, for up to
    MAX_RETRY_AFTER_SECONDS. When retries run out on a status, the last
    response is returned for the caller's own status check. A failure partway
    through the body is not retried. Redirects use none of the retries.
    requests follows them itself, up to 30, and gives each hop the full count.
    """
    retry = _BoundedRetry(
        total=_RETRY_TOTAL,
        backoff_factor=_BACKOFF_FACTOR,
        status_forcelist=_RETRY_STATUSES,
        allowed_methods=allowed_methods,
        # Otherwise urllib3 raises MaxRetryError when retries run out on a
        # status, and the caller never sees the last response.
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_RETRY_AFTER_SECONDS",
    "NETWORK_LOGGER_NAME",
    "create_retry_session",
    "network_logger",
]
