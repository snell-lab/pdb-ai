from __future__ import annotations

import logging
from pathlib import Path

import pytest
from flask import Flask
from pytest import CaptureFixture, LogCaptureFixture, MonkeyPatch

from pdbai.webapp.__main__ import main
from pdbai.webapp.factory import LOOPBACK_HOSTS

_VARIABLES = ("PDBAI_HOST", "PDBAI_PORT", "PDBAI_DEBUG", "PDBAI_TRUSTED_HOSTS")


def _capture_server_runs(
    tmp_path: Path, monkeypatch: MonkeyPatch, **environ: str
) -> list[tuple[Flask, dict[str, object]]]:
    """Stub out Flask.run and return the list that records each call.

    Each item is the application and the keyword arguments of one call.
    Of the variables in _VARIABLES, only those in ``environ`` are set.
    """
    runs: list[tuple[Flask, dict[str, object]]] = []

    def run(application: Flask, **kwargs: object) -> None:
        runs.append((application, kwargs))

    monkeypatch.setattr(Flask, "run", run)
    monkeypatch.chdir(tmp_path)
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    return runs


def test_main_serves_loopback_on_port_8001_without_the_reloader(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    runs = _capture_server_runs(tmp_path, monkeypatch)

    assert main() == 0
    [(application, kwargs)] = runs
    assert kwargs == {
        "host": "127.0.0.1",
        "port": 8001,
        "debug": False,
        "use_reloader": False,
    }
    assert application.config["TRUSTED_HOSTS"] == list(LOOPBACK_HOSTS)


@pytest.mark.parametrize(
    ("name", "value"),
    [("PDBAI_PORT", "abc"), ("PDBAI_PORT", "70000"), ("PDBAI_DEBUG", "1")],
)
def test_a_bad_setting_returns_2_before_creating_a_workspace(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
    name: str,
    value: str,
) -> None:
    runs = _capture_server_runs(tmp_path, monkeypatch, **{name: value})

    assert main() == 2
    assert capsys.readouterr().err.startswith(f"error: {name}")
    assert runs == []
    assert not (tmp_path / "workspace").exists()


def test_debug_is_read_as_a_strict_flag(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    runs = _capture_server_runs(tmp_path, monkeypatch, PDBAI_DEBUG=" TRUE ")

    assert main() == 0
    assert runs[0][1]["debug"] is True


def test_trusted_hosts_add_the_named_hosts_and_the_bind_address(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    runs = _capture_server_runs(
        tmp_path,
        monkeypatch,
        PDBAI_HOST=" 192.168.1.5 ",
        PDBAI_TRUSTED_HOSTS="lab.example, ,x.example,",
    )

    assert main() == 0
    [(application, kwargs)] = runs
    assert kwargs["host"] == "192.168.1.5"
    assert application.config["TRUSTED_HOSTS"] == [
        *LOOPBACK_HOSTS,
        "lab.example",
        "x.example",
        "192.168.1.5",
    ]


def test_a_wildcard_host_without_trusted_names_logs_a_warning(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    runs = _capture_server_runs(tmp_path, monkeypatch, PDBAI_HOST="0.0.0.0")

    with caplog.at_level(logging.WARNING, logger="pdbai.webapp"):
        assert main() == 0

    assert runs[0][0].config["TRUSTED_HOSTS"] == list(LOOPBACK_HOSTS)
    assert "PDBAI_TRUSTED_HOSTS" in caplog.text
