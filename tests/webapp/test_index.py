from __future__ import annotations

import contextlib
import dataclasses
import html as html_module
import json
from collections import Counter
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from flask import Flask
from pytest import MonkeyPatch
from werkzeug.exceptions import MethodNotAllowed

from pdbai.column_filtering import COLUMN_FILTER_MODES
from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.core import (
    XRAY_SOURCE_IDS,
    EngineContext,
    EnginePhase,
    EngineResult,
    EngineSpec,
    OptionSpec,
)
from pdbai.profiles import PROFILES
from pdbai.selection_filters import (
    SEARCH_FILTERS,
    SEARCH_QUANTITIES,
    UNIPROT_ACCESSIONS,
)
from pdbai.webapp import create_app, factory


class _Page(HTMLParser):
    """The elements, select options, and script text of a rendered page."""

    def __init__(self, html: str) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.options: dict[str, list[str]] = {}
        self.scripts: dict[str, str] = {}
        self._select: str | None = None
        self._script: str | None = None
        self.feed(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        self.elements.append((tag, attributes))
        if tag == "select":
            self._select = attributes.get("id")
            self.options[self._select or ""] = []
        elif tag == "option" and self._select is not None:
            self.options[self._select].append(attributes.get("value") or "")
        elif tag == "script":
            self._script = attributes.get("id")

    def handle_endtag(self, tag: str) -> None:
        if tag == "select":
            self._select = None
        elif tag == "script":
            self._script = None

    def handle_data(self, data: str) -> None:
        if self._script is not None:
            self.scripts[self._script] = self.scripts.get(self._script, "") + data

    def by_id(self, element_id: str) -> dict[str, str | None]:
        return next(
            attrs for _, attrs in self.elements if attrs.get("id") == element_id
        )

    def config(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self.scripts["pdbaiConfig"]))


def _app(tmp_path: Path, monkeypatch: MonkeyPatch) -> Flask:
    monkeypatch.chdir(tmp_path)
    return create_app()


def _page(app: Flask) -> _Page:
    return _Page(app.test_client().get("/").get_data(as_text=True))


def test_every_id_is_unique_and_every_label_names_a_control(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    page = _page(_app(tmp_path, monkeypatch))
    ids = Counter(attrs["id"] for _, attrs in page.elements if attrs.get("id"))

    assert [element_id for element_id, count in ids.items() if count > 1] == []
    for tag, attrs in page.elements:
        if tag == "label" and attrs.get("for"):
            assert attrs["for"] in ids, attrs["for"]
        for described_by in (attrs.get("aria-describedby") or "").split():
            assert described_by in ids, described_by


def test_the_search_has_a_labelled_field_for_every_declared_filter(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    html = _app(tmp_path, monkeypatch).test_client().get("/").get_data(as_text=True)
    page = _Page(html)
    fields = [attrs for _, attrs in page.elements if "data-search-filter" in attrs]
    labelled = {attrs.get("for") for tag, attrs in page.elements if tag == "label"}
    keys = [search_filter.json_key for search_filter in SEARCH_FILTERS]
    keys.append(UNIPROT_ACCESSIONS.json_key)

    # Each field is named by the key /api/select reads, so app.js sends it as
    # it is. The accession list is a text area, the ranges number inputs.
    assert [attrs["name"] for attrs in fields] == keys
    assert [attrs.get("type") for attrs in fields] == ["number"] * len(
        SEARCH_FILTERS
    ) + [None]
    assert fields[-1]["data-search-filter-kind"] == "accessions"
    assert all(attrs["id"] in labelled for attrs in fields)
    # The note holds an apostrophe, which the template escapes.
    assert UNIPROT_ACCESSIONS.label in html
    assert UNIPROT_ACCESSIONS.note in html_module.unescape(html)
    # The menu has one unticked checkbox for each filter, and app.js pairs it
    # with the field of the same key. So does the button that removes a filter.
    toggles = [
        attrs for _, attrs in page.elements if "data-search-filter-toggle" in attrs
    ]
    assert [attrs["data-search-filter-toggle"] for attrs in toggles] == keys
    assert all(attrs["type"] == "checkbox" for attrs in toggles)
    assert not any("checked" in attrs for attrs in toggles)
    assert [
        attrs["data-search-filter-remove"]
        for _, attrs in page.elements
        if "data-search-filter-remove" in attrs
    ] == keys
    for search_filter in SEARCH_FILTERS:
        assert search_filter.label in html
    for quantity in SEARCH_QUANTITIES:
        assert quantity.note in html


def test_a_filter_field_and_its_note_stay_hidden_until_the_filter_is_ticked(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    page = _page(_app(tmp_path, monkeypatch))
    wrappers = [attrs for _, attrs in page.elements if "data-search-quantity" in attrs]
    notes = [
        attrs for _, attrs in page.elements if "data-search-quantity-note" in attrs
    ]

    # The form therefore stays the same length however many filters exist.
    assert "hidden" in page.by_id("searchFilters")
    assert len(wrappers) == len(SEARCH_FILTERS) + 1
    assert all("hidden" in attrs for attrs in wrappers)
    assert [attrs["data-search-quantity"] for attrs in wrappers] == [
        *(search_filter.quantity.id for search_filter in SEARCH_FILTERS),
        UNIPROT_ACCESSIONS.id,
    ]
    assert [attrs["data-search-quantity-note"] for attrs in notes] == [
        *(quantity.id for quantity in SEARCH_QUANTITIES),
        UNIPROT_ACCESSIONS.id,
    ]
    assert all("hidden" in attrs for attrs in notes)


def test_the_page_config_holds_the_api_vocabularies_and_routes_that_exist(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch)
    client = app.test_client()
    config = _page(app).config()

    assert config["techniques"] == client.get("/api/techniques").get_json()
    assert config["profiles"] == client.get("/api/profiles").get_json()
    adapter = app.url_map.bind("localhost")
    urls = cast(dict[str, str], config["urls"])
    for url in urls.values():
        path = url.replace("RUN_ID", "run-a")
        # MethodNotAllowed means a rule matched the path under another
        # method. Only NotFound, for a path no rule matches, fails the test.
        with contextlib.suppress(MethodNotAllowed):
            adapter.match(path)


def test_a_label_holding_a_script_end_tag_stays_inside_the_config(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    label = "</script><script>alert(1)</script>"
    profiles = (
        dataclasses.replace(PROFILES[0], name=label),
        *PROFILES[1:],
    )
    monkeypatch.setattr(factory, "PROFILES", profiles)

    config = _page(_app(tmp_path, monkeypatch)).config()

    first = cast(list[dict[str, object]], config["profiles"])[0]
    assert first["name"] == label


class _NestedEngine:
    spec = EngineSpec(
        name="stub_nested",
        display_name="Stub nested",
        phase=EnginePhase.ANNOTATION,
        toggle="include_stub_nested",
        options=(
            OptionSpec(
                "include_stub_nested", label="Stub", help="Run the stub.", default=True
            ),
            OptionSpec(
                "stub_detail",
                label="Stub detail",
                help="Refine the stub.",
                refines="include_stub_nested",
            ),
            OptionSpec(
                "stub_depth",
                label="Stub depth",
                help="Refine the refinement.",
                kind="count",
                default=None,
                refines="stub_detail",
            ),
        ),
    )

    def run(self, context: EngineContext) -> EngineResult:
        return EngineResult(columns=pd.DataFrame(index=context.atom_dataframe.index))


def test_engine_controls_start_at_their_defaults_at_every_depth(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    install_engines: Callable[..., None],
) -> None:
    install_engines(_NestedEngine())

    page = _page(_app(tmp_path, monkeypatch))

    for option in get_engine_registry().option_specs():
        control = page.by_id(f"option-{option.name}")
        assert control.get("name") == option.json_key
        if option.kind == "flag":
            assert ("checked" in control) is option.default, option.name
    assert page.by_id("option-stub_depth")["data-refines"] == "stub_detail"


@pytest.mark.parametrize(
    ("select_id", "expected"),
    [
        # The source select has no blank option, because XRAY_SOURCE_IDS
        # starts with the value that means every source.
        ("xraySource", list(XRAY_SOURCE_IDS)),
        ("columnMode", ["", *COLUMN_FILTER_MODES]),
        ("filterMode", ["", *COLUMN_FILTER_MODES]),
    ],
)
def test_select_options_come_from_the_server_vocabularies(
    tmp_path: Path, monkeypatch: MonkeyPatch, select_id: str, expected: list[str]
) -> None:
    page = _page(_app(tmp_path, monkeypatch))

    assert page.options[select_id] == expected
