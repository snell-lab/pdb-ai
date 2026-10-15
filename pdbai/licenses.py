"""An audit of the license metadata of the packages the project depends on.

python -m pdbai.licenses reads requirements.txt, or the file that
--requirements names. It prints one line for each package in the dependency
closure of those requirements, as a table, Markdown, or JSON. CI runs it
after installing the requirements, so the build fails when a dependency is
missing or its license can no longer be read from its metadata. provenance
calls collect_dependency_licenses too, to record which packages a run had
installed.

The audit checks that every license is stated, not that it is acceptable.
Which licenses the project can depend on, and the notices it must carry in
THIRD_PARTY_NOTICES.md, are a person's call. A license is read from package
metadata alone. A package whose metadata only names its license files is
therefore left for a person to review rather than guessed at. The closure
follows the requirements as installed metadata records them, so a package
that is installed but not required is not audited. A metapackage installs
no code of its own, so it passes whatever its metadata says. The packages it
requires are audited in their own right.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Literal

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import NormalizedName, canonicalize_name

REQUIREMENTS_PATH = Path(__file__).resolve().parent.parent / "requirements.txt"

# As in pip, "#" starts a comment at the start of a line or after whitespace,
# so a URL fragment such as #subdirectory= stays part of the requirement.
_COMMENT = re.compile(r"(^|\s)#.*")

# Values that mean a metadata field was left unset. distutils and older
# setuptools wrote UNKNOWN for fields such as Summary, Home-page, and License
# that a project did not fill in. Authors write the others by hand.
_PLACEHOLDERS = frozenset({"n/a", "na", "none", "unknown", "unknown license"})

_LICENSE_CLASSIFIER_PREFIX = "License :: "

# License values that speak of a license without naming it, casefolded. The
# first two appear in a License field when a project names its licenses
# elsewhere. python-dateutil, for one, writes "Dual License" and names its two
# licenses in classifiers. The rest are trove classifiers that name only a
# category of license, or none.
_UNNAMED_LICENSES = frozenset(
    {
        "dual license",
        "dual licensed",
        "osi approved",
        "dfsg approved",
        "other/proprietary license",
    }
)

# A License field such as "see LICENSE" points at a file rather than naming
# the license, and License-File already lists the files a package ships.
_FILE_POINTER = re.compile(r"see\b", re.IGNORECASE)

# A License field longer than this, or over several lines, holds a pasted
# license text rather than the name of a license. The report keeps this much
# of such a text.
_MAX_LICENSE_NAME_LENGTH = 200

_REPORT_COLUMNS = ("Package", "Version", "License", "Dependency", "Status")
_MAX_CELL_LENGTH = 120

# How far the report vouches for a package's license. The audit passes only
# when every package is "ok". A package is "ok" when its metadata gives the
# license, or when it is a metapackage with no code to license. It is "review"
# when its metadata only names license files, which a person must read. It is
# "unknown" when its metadata names neither a license nor a license file. It
# is "missing" when the package is not installed.
type LicenseStatus = Literal["ok", "review", "unknown", "missing"]


@dataclass(frozen=True)
class PackageLicense:
    """One package's line in the license report.

    ``version`` is None when the package is not installed, and ``license``,
    ``homepage``, and ``summary`` are None when its metadata gives none.
    ``direct`` is True for a package the audited requirements name, even when
    another package also requires it, and False for one they only bring in.
    """

    name: str
    version: str | None
    license: str | None
    homepage: str | None
    summary: str | None
    status: LicenseStatus
    direct: bool


def _parse_requirement(text: str, *, source: str) -> Requirement:
    """Parse a requirement, raising ValueError that says where it came from."""
    try:
        return Requirement(text)
    except InvalidRequirement as exception:
        raise ValueError(f"Invalid requirement in {source}: {text!r}") from exception


def parse_requirements(requirements_path: Path) -> list[Requirement]:
    """Return the requirements a requirements file lists.

    Only blank lines, comments, and requirement specifiers are supported. The
    project's requirements files use no pip options or nested includes. A
    file that does not exist, or a line that is not a valid requirement,
    raises ValueError.
    """
    if not requirements_path.exists():
        raise ValueError(f"Requirements file not found: {requirements_path}")
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    return [
        _parse_requirement(text, source=f"{requirements_path}:{line_number}")
        for line_number, line in enumerate(lines, start=1)
        if (text := _COMMENT.sub("", line).strip())
    ]


# Building one package's report entry from its installed metadata.
def _one_line(text: str) -> str:
    return " ".join(text.split())


def _shorten(text: str, limit: int) -> str:
    """Cut text to ``limit`` characters, ending a cut text with an ellipsis."""
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}…"


def _metadata_text(value: str | None) -> str | None:
    """Return a value on one line, or None when blank or a placeholder."""
    if value is None:
        return None
    text = _one_line(value)
    if not text or text.casefold() in _PLACEHOLDERS:
        return None
    return text


def _license_name(value: str | None) -> str | None:
    """Return a license value on one line, or None when it names no license."""
    text = _metadata_text(value)
    if text is None:
        return None
    if text.casefold() in _UNNAMED_LICENSES or _FILE_POINTER.match(text):
        return None
    return text


def _is_metapackage(distribution: metadata.Distribution) -> bool:
    """Return whether the distribution installs nothing but its own metadata.

    The answer is False when the installer recorded no file list.
    """
    files = distribution.files
    if not files:
        return False
    return all(file.parts[0].endswith((".dist-info", ".egg-info")) for file in files)


def _license_and_status(
    distribution: metadata.Distribution,
) -> tuple[str | None, LicenseStatus]:
    """Return the distribution's license and the status it earns.

    A metapackage is marked not applicable and passes. For any other package,
    the license is the first that its metadata provides, in this order. First
    comes the SPDX License-Expression. Next is a License field holding a short
    one-line name. Next are the License classifiers. Last is a License field
    holding a pasted license text, cut to its opening. Classifiers rank above
    a pasted text because they name the license, and the text often does not.
    When none of these gives a license, the License-File names are returned
    with status "review". With no license file either, the status is
    "unknown". A placeholder, a value in _UNNAMED_LICENSES, or a License field
    that points at a file counts as unset.
    """
    if _is_metapackage(distribution):
        return "Not applicable (metadata-only package)", "ok"

    fields = distribution.metadata
    if expression := _metadata_text(fields.get("License-Expression")):
        return expression, "ok"

    declared = fields.get("License") or ""
    is_pasted_text = "\n" in declared or len(declared) > _MAX_LICENSE_NAME_LENGTH
    if not is_pasted_text and (license_name := _license_name(declared)):
        return license_name, "ok"

    classifier_licenses = [
        name
        for classifier in fields.get_all("Classifier") or []
        if classifier.startswith(_LICENSE_CLASSIFIER_PREFIX)
        and (name := _license_name(classifier.removeprefix(_LICENSE_CLASSIFIER_PREFIX)))
    ]
    if classifier_licenses:
        return ", ".join(classifier_licenses), "ok"
    if is_pasted_text and (license_text := _metadata_text(declared)):
        return _shorten(license_text, _MAX_LICENSE_NAME_LENGTH), "ok"

    license_files = [
        name
        for value in fields.get_all("License-File") or []
        if (name := _metadata_text(value))
    ]
    if license_files:
        return f"Review license file: {', '.join(license_files)}", "review"
    return None, "unknown"


def _homepage(fields: metadata.PackageMetadata) -> str | None:
    """Return the project's home page, or None when its metadata gives no URL.

    The Home-page field comes first. Failing that, the Project-URL labelled
    as the home page is used, then the one labelled as the source repository,
    then the first one listed.
    """
    if homepage := _metadata_text(fields.get("Home-page")):
        return homepage
    urls_by_label: dict[str, str] = {}
    for project_url in fields.get_all("Project-URL") or []:
        label, _, value = project_url.partition(",")
        if url := _metadata_text(value):
            urls_by_label[label.strip().casefold()] = url
    for label in ("homepage", "repository", "source", "source code"):
        if label in urls_by_label:
            return urls_by_label[label]
    return next(iter(urls_by_label.values()), None)


def _package_license(
    requested_name: str,
    distribution: metadata.Distribution | None,
    *,
    direct: bool,
) -> PackageLicense:
    """Return a package's report entry.

    ``distribution`` is None when the package is not installed, and the entry
    then carries the name it was requested by.
    """
    if distribution is None:
        return PackageLicense(
            name=requested_name,
            version=None,
            license=None,
            homepage=None,
            summary=None,
            status="missing",
            direct=direct,
        )
    license_text, status = _license_and_status(distribution)
    fields = distribution.metadata
    return PackageLicense(
        name=distribution.name,
        version=distribution.version,
        license=license_text,
        homepage=_homepage(fields),
        summary=_metadata_text(fields.get("Summary")),
        status=status,
        direct=direct,
    )


# Walking the requirements that installed distributions record.
def _installed_distribution(name: str) -> metadata.Distribution | None:
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None


def _requirement_applies(requirement: Requirement, *, extras: set[str]) -> bool:
    """Return whether the marker holds here under any of the extras."""
    if requirement.marker is None:
        return True
    return any(
        requirement.marker.evaluate({"extra": extra}) for extra in extras or {""}
    )


def _applicable_requirements(
    distribution: metadata.Distribution, *, extras: set[str]
) -> Iterator[Requirement]:
    source = f"metadata for {distribution.name}"
    for text in distribution.requires or []:
        requirement = _parse_requirement(text, source=source)
        if _requirement_applies(requirement, extras=extras):
            yield requirement


def collect_dependency_licenses(
    requirements: Iterable[Requirement],
) -> list[PackageLicense]:
    """Return the report entries for the requirements' closure, sorted by name.

    A requirement whose marker does not hold in this environment is left out.
    The entries cover the listed packages and everything they require,
    directly or not. The walk follows installed metadata, under the extras
    each package was requested with. A package that is not installed gets an
    entry with status "missing", and nothing it would require is followed.
    An invalid requirement in installed metadata raises ValueError.
    """
    roots = [
        requirement
        for requirement in requirements
        if _requirement_applies(requirement, extras=set())
    ]
    direct_names = {canonicalize_name(requirement.name) for requirement in roots}

    # The three maps are keyed by canonical package name, and a distribution
    # is None when its package is not installed. The requirements a package
    # brings in depend on its extras, so a package requested again with a new
    # extra is walked again for that extra alone.
    entries: dict[NormalizedName, PackageLicense] = {}
    distributions: dict[NormalizedName, metadata.Distribution | None] = {}
    requested_extras: dict[NormalizedName, set[str]] = {}
    queue = deque(roots)
    while queue:
        requirement = queue.popleft()
        name = canonicalize_name(requirement.name)
        if name in entries:
            new_extras = requirement.extras - requested_extras[name]
            if not new_extras:
                continue
        else:
            new_extras = requirement.extras
            distributions[name] = _installed_distribution(requirement.name)
            entries[name] = _package_license(
                requirement.name, distributions[name], direct=name in direct_names
            )
        requested_extras.setdefault(name, set()).update(new_extras)
        distribution = distributions[name]
        if distribution is not None:
            queue.extend(_applicable_requirements(distribution, extras=new_extras))
    return sorted(entries.values(), key=lambda entry: entry.name.casefold())


# Rendering the report. The table and Markdown formats show the same cells.
def _report_row(entry: PackageLicense) -> tuple[str, ...]:
    """Return the entry's cells under _REPORT_COLUMNS, each one short line."""
    cells = (
        entry.name,
        entry.version or "not installed",
        entry.license or "unknown",
        "direct" if entry.direct else "transitive",
        entry.status,
    )
    return tuple(_shorten(_one_line(cell), _MAX_CELL_LENGTH) for cell in cells)


def _format_table(entries: Sequence[PackageLicense]) -> str:
    rows = [_REPORT_COLUMNS, *(_report_row(entry) for entry in entries)]
    widths = [max(map(len, column)) for column in zip(*rows, strict=True)]
    return "\n".join(
        " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in rows
    )


def _markdown_row(cells: Iterable[str]) -> str:
    """Return a Markdown table row of the cells.

    Package metadata is third-party text. HTML in it is escaped rather than
    rendered, and a pipe is escaped so that it cannot end its cell early.
    """
    escaped = (html.escape(cell, quote=False).replace("|", "\\|") for cell in cells)
    return f"| {' | '.join(escaped)} |"


def _format_markdown(entries: Sequence[PackageLicense]) -> str:
    return "\n".join(
        [
            _markdown_row(_REPORT_COLUMNS),
            "|---" * len(_REPORT_COLUMNS) + "|",
            *(_markdown_row(_report_row(entry)) for entry in entries),
        ]
    )


def _format_json(entries: Sequence[PackageLicense]) -> str:
    return json.dumps([asdict(entry) for entry in entries], indent=2)


_REPORT_FORMATS: dict[str, Callable[[Sequence[PackageLicense]], str]] = {
    "table": _format_table,
    "markdown": _format_markdown,
    "json": _format_json,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Print the license report and return the exit status.

    The status is 1 when any package's status is not "ok", and 0 otherwise.
    An invalid argument, a requirements file that does not exist, and an
    invalid requirement in it or in installed metadata raise SystemExit with
    status 2, through argparse.
    """
    parser = argparse.ArgumentParser(
        description="Report license metadata for the dependency closure of a requirements file."
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=REQUIREMENTS_PATH,
        help="Path to a requirements.txt file.",
    )
    parser.add_argument(
        "--format",
        choices=tuple(_REPORT_FORMATS),
        default="table",
        help="Output format for the license report.",
    )
    args = parser.parse_args(argv)

    try:
        entries = collect_dependency_licenses(parse_requirements(args.requirements))
    except ValueError as exception:
        parser.error(str(exception))

    print(_REPORT_FORMATS[args.format](entries))
    return 0 if all(entry.status == "ok" for entry in entries) else 1


__all__ = [
    "REQUIREMENTS_PATH",
    "LicenseStatus",
    "PackageLicense",
    "collect_dependency_licenses",
    "main",
    "parse_requirements",
]


if __name__ == "__main__":
    raise SystemExit(main())
