from __future__ import annotations

import json
from collections.abc import Iterable
from email.message import Message
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from pdbai import licenses


class _FakeDistribution:
    def __init__(
        self,
        name: str,
        *,
        version: str = "1.0.0",
        license_expression: str | None = None,
        license_text: str | None = None,
        classifiers: tuple[str, ...] = (),
        license_files: tuple[str, ...] = (),
        project_urls: tuple[str, ...] = (),
        requires: tuple[str, ...] = (),
        files: tuple[str, ...] | None = None,
    ) -> None:
        self.name = name
        self.metadata = Message()
        self.metadata["Name"] = name
        self.metadata["Summary"] = f"{name} summary"
        if license_expression is not None:
            self.metadata["License-Expression"] = license_expression
        if license_text is not None:
            self.metadata["License"] = license_text
        for classifier in classifiers:
            self.metadata["Classifier"] = classifier
        for license_file in license_files:
            self.metadata["License-File"] = license_file
        for project_url in project_urls:
            self.metadata["Project-URL"] = project_url
        self.version = version
        self.requires = list(requires)
        self.files = (
            [metadata.PackagePath(file) for file in files]
            if files is not None
            else None
        )


def _install(
    monkeypatch: pytest.MonkeyPatch, *distributions: _FakeDistribution
) -> None:
    installed = {canonicalize_name(dist.name): dist for dist in distributions}

    def distribution(name: str) -> _FakeDistribution:
        try:
            return installed[canonicalize_name(name)]
        except KeyError:
            raise metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(metadata, "distribution", distribution)


def test_parse_requirements_supports_comments_and_direct_references(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "# Runtime dependencies\n"
        "demo[feature]>=2 ; python_version >= '3.10'  # inline comment\n"
        "\n"
        "rabdam @ git+https://github.com/snell-lab/rabdam-custom.git@main\n"
        "nested @ https://example.test/archive.zip#subdirectory=pkg\t# tab comment\n",
        encoding="utf-8",
    )

    parsed = licenses.parse_requirements(requirements)

    assert [requirement.name for requirement in parsed] == ["demo", "rabdam", "nested"]
    assert parsed[0].extras == {"feature"}
    assert parsed[1].url == "git+https://github.com/snell-lab/rabdam-custom.git@main"
    assert parsed[2].url == "https://example.test/archive.zip#subdirectory=pkg"


def test_parse_requirements_reads_the_project_requirements() -> None:
    runtime = licenses.REQUIREMENTS_PATH

    for path in (runtime, runtime.with_name("requirements-dev.txt")):
        assert licenses.parse_requirements(path)


def test_parse_requirements_rejects_a_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(ValueError, match="Requirements file not found"):
        licenses.parse_requirements(missing_path)


def test_collect_dependency_licenses_walks_markers_extras_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        _FakeDistribution(
            "Root",
            license_expression="MIT",
            project_urls=("Homepage, https://example.test/root",),
            requires=(
                "child>=1",
                "feature-dependency; extra == 'feature'",
                "skipped-dependency; python_version < '1'",
            ),
        ),
        _FakeDistribution(
            "Child",
            license_text="Full license body\nwith many lines",
            classifiers=("License :: OSI Approved :: BSD License",),
        ),
        _FakeDistribution(
            "Feature-Dependency",
            license_files=("LICENSE.txt",),
        ),
    )

    entries = licenses.collect_dependency_licenses([Requirement("root[feature]")])
    by_name = {entry.name: entry for entry in entries}

    assert set(by_name) == {"Root", "Child", "Feature-Dependency"}
    assert by_name["Root"].license == "MIT"
    assert by_name["Root"].homepage == "https://example.test/root"
    assert by_name["Root"].direct is True
    assert by_name["Child"].license == "OSI Approved :: BSD License"
    assert by_name["Child"].direct is False
    assert by_name["Feature-Dependency"].license == "Review license file: LICENSE.txt"
    assert by_name["Feature-Dependency"].status == "review"


def test_a_package_requested_again_with_a_new_extra_brings_in_its_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        _FakeDistribution("First", license_expression="MIT", requires=("shared",)),
        _FakeDistribution(
            "Second", license_expression="MIT", requires=("shared[extra]",)
        ),
        _FakeDistribution(
            "Shared",
            license_expression="MIT",
            requires=("extra-dependency; extra == 'extra'",),
        ),
        _FakeDistribution("Extra-Dependency", license_expression="MIT"),
    )

    entries = licenses.collect_dependency_licenses(
        [Requirement("first"), Requirement("second")]
    )

    assert {entry.name: entry.direct for entry in entries} == {
        "Extra-Dependency": False,
        "First": True,
        "Second": True,
        "Shared": False,
    }


def test_a_root_is_direct_even_when_required_and_left_out_when_its_marker_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        _FakeDistribution("Root", license_expression="MIT", requires=("child",)),
        _FakeDistribution("Child", license_expression="MIT"),
    )

    entries = licenses.collect_dependency_licenses(
        [
            Requirement("root"),
            Requirement("child"),
            Requirement("absent; python_version < '1'"),
        ]
    )

    assert {entry.name: entry.direct for entry in entries} == {
        "Root": True,
        "Child": True,
    }


# UNKNOWN is the placeholder older setuptools wrote when no license was
# given. A value that runs over several lines is taken for pasted license
# text.
@pytest.mark.parametrize(
    ("license_text", "expected"),
    [
        ("BSD-3-Clause", "BSD-3-Clause"),
        ("UNKNOWN", "OSI Approved :: BSD License"),
        ("BSD\nsee LICENSE.txt", "OSI Approved :: BSD License"),
    ],
)
def test_a_license_name_ranks_above_classifiers_and_a_pasted_text_below(
    monkeypatch: pytest.MonkeyPatch, license_text: str, expected: str
) -> None:
    _install(
        monkeypatch,
        _FakeDistribution(
            "Package",
            license_text=license_text,
            classifiers=("License :: OSI Approved :: BSD License",),
        ),
    )

    entry = licenses.collect_dependency_licenses([Requirement("package")])[0]

    assert entry.license == expected
    assert entry.status == "ok"


def test_a_pasted_license_text_without_classifiers_is_cut_to_its_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pasted_text = "Copyright (c) 2026 Example\n\n" + "Permission is granted. " * 20
    _install(monkeypatch, _FakeDistribution("Package", license_text=pasted_text))

    entry = licenses.collect_dependency_licenses([Requirement("package")])[0]

    assert entry.license is not None
    assert entry.license.startswith("Copyright (c) 2026 Example Permission is granted.")
    assert entry.license.endswith("…")
    assert len(entry.license) <= 200
    assert entry.status == "ok"


@pytest.mark.parametrize(
    ("license_text", "classifiers", "expected"),
    [
        (
            "Dual License",
            (
                "License :: OSI Approved :: BSD License",
                "License :: OSI Approved :: Apache Software License",
            ),
            "OSI Approved :: BSD License, OSI Approved :: Apache Software License",
        ),
        (
            "see LICENSE.txt",
            ("License :: OSI Approved :: MIT License",),
            "OSI Approved :: MIT License",
        ),
        (
            None,
            ("License :: OSI Approved", "License :: OSI Approved :: MIT License"),
            "OSI Approved :: MIT License",
        ),
    ],
)
def test_a_value_that_names_no_license_falls_through_to_the_classifiers(
    monkeypatch: pytest.MonkeyPatch,
    license_text: str | None,
    classifiers: tuple[str, ...],
    expected: str,
) -> None:
    _install(
        monkeypatch,
        _FakeDistribution(
            "Package", license_text=license_text, classifiers=classifiers
        ),
    )

    entry = licenses.collect_dependency_licenses([Requirement("package")])[0]

    assert entry.license == expected
    assert entry.status == "ok"


@pytest.mark.parametrize(
    ("license_text", "classifiers"),
    [
        ("Dual License", ()),
        ("See LICENSE", ()),
        (None, ("License :: OSI Approved",)),
        (None, ("License :: DFSG approved",)),
        ("Other/Proprietary License", ("License :: Other/Proprietary License",)),
    ],
)
def test_metadata_that_names_no_license_is_not_a_stated_license(
    monkeypatch: pytest.MonkeyPatch,
    license_text: str | None,
    classifiers: tuple[str, ...],
) -> None:
    _install(
        monkeypatch,
        _FakeDistribution(
            "Without-Files", license_text=license_text, classifiers=classifiers
        ),
        _FakeDistribution(
            "With-Files",
            license_text=license_text,
            classifiers=classifiers,
            license_files=("LICENSE",),
        ),
    )

    entries = licenses.collect_dependency_licenses(
        [Requirement("without-files"), Requirement("with-files")]
    )

    assert {entry.name: (entry.license, entry.status) for entry in entries} == {
        "Without-Files": (None, "unknown"),
        "With-Files": ("Review license file: LICENSE", "review"),
    }


@pytest.mark.parametrize(
    ("license_expression", "license_files"),
    [(None, ()), (None, ("LICENSE",)), ("MIT", ("LICENSE",))],
)
def test_metadata_only_distribution_does_not_require_a_software_license(
    monkeypatch: pytest.MonkeyPatch,
    license_expression: str | None,
    license_files: tuple[str, ...],
) -> None:
    _install(
        monkeypatch,
        _FakeDistribution(
            "Meta-Package",
            files=(
                "meta_package-1.0.0.dist-info/METADATA",
                "meta_package-1.0.0.dist-info/RECORD",
            ),
            license_expression=license_expression,
            license_files=license_files,
        ),
    )

    entry = licenses.collect_dependency_licenses([Requirement("meta-package")])[0]

    assert entry.license == "Not applicable (metadata-only package)"
    assert entry.status == "ok"


def test_code_bearing_distribution_without_license_remains_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        _FakeDistribution(
            "Unlicensed-Package",
            classifiers=("License :: UNKNOWN",),
            files=(
                "unlicensed_package/__init__.py",
                "unlicensed_package-1.0.0.dist-info/METADATA",
            ),
        ),
    )

    entry = licenses.collect_dependency_licenses([Requirement("unlicensed-package")])[0]

    assert entry.license is None
    assert entry.status == "unknown"


def test_missing_direct_reference_uses_its_distribution_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)

    entries = licenses.collect_dependency_licenses(
        [
            Requirement(
                "rabdam @ git+https://github.com/snell-lab/rabdam-custom.git@main"
            )
        ]
    )

    assert len(entries) == 1
    assert entries[0].name == "rabdam"
    assert entries[0].version is None
    assert entries[0].status == "missing"
    assert entries[0].direct is True


def test_report_fails_and_says_a_missing_package_is_not_installed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("absent\n", encoding="utf-8")
    _install(monkeypatch)

    table_exit_code = licenses.main(["--requirements", str(requirements)])
    table_row = capsys.readouterr().out.splitlines()[1]
    json_exit_code = licenses.main(
        ["--requirements", str(requirements), "--format", "json"]
    )
    report = json.loads(capsys.readouterr().out)

    assert table_exit_code == json_exit_code == 1
    assert [cell.strip() for cell in table_row.split("|")] == [
        "absent",
        "not installed",
        "unknown",
        "direct",
        "missing",
    ]
    assert report == [
        {
            "name": "absent",
            "version": None,
            "license": None,
            "homepage": None,
            "summary": None,
            "status": "missing",
            "direct": True,
        }
    ]


def test_markdown_output_is_single_line_escaped_and_html_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    entry = licenses.PackageLicense(
        name="demo<script>",
        version="1.0",
        license="MIT | BSD\nadditional text",
        homepage=None,
        summary=None,
        status="ok",
        direct=True,
    )
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("demo\n", encoding="utf-8")

    def fake_collect_dependency_licenses(
        requirements: Iterable[Requirement],
    ) -> list[licenses.PackageLicense]:
        del requirements
        return [entry]

    monkeypatch.setattr(
        licenses,
        "collect_dependency_licenses",
        fake_collect_dependency_licenses,
    )

    exit_code = licenses.main(
        ["--requirements", str(requirements_path), "--format", "markdown"]
    )
    rendered = capsys.readouterr().out.rstrip("\n")

    assert exit_code == 0
    assert len(rendered.splitlines()) == 3
    assert "demo&lt;script&gt;" in rendered
    assert "MIT \\| BSD additional text" in rendered
