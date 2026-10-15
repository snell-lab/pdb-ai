"""Scripted stand-ins for ``requests.Session`` and its responses.

The network tests pass a ``FakeSession`` to the code under test in place of
a real session, so no test reaches RCSB, PDB-Redo, wwPDB, or AlphaFold. The
session answers from a script the test writes up front. It also records
every request, so a test can check which URLs the code asked for and with
which timeout.

Only ``get``, ``post``, and the parts of a response the code reads are
imitated. The fake is cast to ``requests.Session`` rather than derived from
it, so a method it lacks raises AttributeError instead of quietly reaching
the network. A request the script did not foresee fails the test rather
than getting a default answer, so an extra fetch cannot go unnoticed.
"""

from __future__ import annotations

import json
from typing import cast

import requests


class FakeResponse:
    """A canned answer to one request, with a fixed status, payload, and body.

    ``json()`` returns ``payload`` when one is given and otherwise parses
    ``text``, so a malformed body can be simulated. ``raise_for_status``
    raises ``requests.HTTPError`` from status 400 up, as the real one does.
    """

    def __init__(
        self,
        payload: object = None,
        *,
        status_code: int = 200,
        content: bytes = b"",
        text: str = "",
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.content = content
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> object:
        if self._payload is None:
            return json.loads(self.text)
        return self._payload


class FakeSession:
    """A session that answers requests from a script, in order.

    Each request takes the next entry of the script. An exception there is
    raised instead of answering, and a request past the end of the script
    raises AssertionError. ``calls`` records the URL and timeout of every
    request, and ``json_bodies`` the body of every POST.
    """

    def __init__(self, *responses: FakeResponse | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, float]] = []
        self.json_bodies: list[object] = []

    def get(self, url: str, *, timeout: float) -> FakeResponse:
        self.calls.append((url, timeout))
        return self._next()

    def post(self, url: str, *, json: object, timeout: float) -> FakeResponse:
        self.calls.append((url, timeout))
        self.json_bodies.append(json)
        return self._next()

    def as_session(self) -> requests.Session:
        return cast(requests.Session, self)

    def _next(self) -> FakeResponse:
        if not self.responses:
            raise AssertionError("Unexpected network request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
