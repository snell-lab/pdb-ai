"""The options a run is configured with, and how engines declare settings.

EngineOptions holds one run's settings. The output shape, the technique, and
the X-ray source belong to the run itself. They are checked as soon as the
options are made, since intake and output need them before any engine is
consulted. The other settings belong to engines, which the core does not
know. They are kept as plain values under the names the engines declare, and
they are checked against those declarations elsewhere. A run stores its
options in its manifest as a JSON object and resumes only under the same
options. So the options convert to that object and back without loss.

An engine declares each setting it reads as an OptionSpec, once, so adding
a setting touches no code outside the engine. A setting is a flag, a choice
among declared strings, or a count.

The dataset shapes are declared here, with the base tables each writes,
because the shape is a run option and no engine table may take a base
table's name.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

from ...positive_numbers import parse_positive_int, positive_int
from .techniques import (
    ALL_XRAY_SOURCES,
    DEFAULT_TECHNIQUE,
    XRAY_SOURCE_TECHNIQUE,
    get_technique,
    get_xray_source,
)

type OptionValue = bool | str | int | None
type OptionKind = Literal["flag", "choice", "count"]
type DatasetShapeName = Literal["flat", "star_schema"]


@dataclass(frozen=True)
class DatasetShape:
    """One output shape and the base tables it writes."""

    name: DatasetShapeName
    label: str
    # The shape's base tables, primary table first.
    tables: tuple[str, ...]
    # Completes "output as ..." in the run summary.
    description: str


DATASET_SHAPES: dict[DatasetShapeName, DatasetShape] = {
    shape.name: shape
    for shape in (
        DatasetShape(
            "flat",
            "Flat",
            ("flat",),
            "a single denormalized CSV (one row per atom).",
        ),
        DatasetShape(
            "star_schema",
            "Star Schema",
            ("atoms", "residues", "chains", "structures", "citations"),
            "a 5-table star schema (atoms/residues/chains/structures/citations).",
        ),
    )
}
DEFAULT_DATASET_SHAPE: DatasetShapeName = "flat"


def parse_dataset_shape(value: object) -> DatasetShapeName:
    """Return the dataset shape ``value`` names, or raise ValueError."""
    for name in DATASET_SHAPES:
        if value == name:
            return name
    raise ValueError(
        f"dataset_shape must be one of {', '.join(DATASET_SHAPES)}; got {value!r}"
    )


@dataclass(frozen=True)
class EngineOptions:
    """The settings of one run, the engines' included.

    An unknown dataset shape, technique, or X-ray source raises ValueError, and
    so does an X-ray source under any technique but X-ray diffraction. Engine
    settings live in ``values``, keyed by the option names the engines
    declare, and are not checked here. validate_options in
    pdbai.engines.compatibility checks them against the declarations.
    EngineRegistry.with_defaults fills in the declared default of each one a
    run leaves out.
    """

    dataset_shape: DatasetShapeName = DEFAULT_DATASET_SHAPE
    # Intake refuses an entry of any other technique, and a run that enables an
    # engine for a technique it does not support is refused before it starts.
    technique: str = DEFAULT_TECHNIQUE
    # A second filter within X-ray diffraction. A specific source refuses at
    # intake every entry that does not report it, including one that reports
    # no source at all.
    xray_source: str = ALL_XRAY_SOURCES
    values: Mapping[str, OptionValue] = field(
        default_factory=dict[str, OptionValue], hash=False
    )

    def __post_init__(self) -> None:
        """Check the run-level settings and copy the caller's values."""
        parse_dataset_shape(self.dataset_shape)
        technique = get_technique(self.technique)
        if self.xray_source != ALL_XRAY_SOURCES:
            source = get_xray_source(self.xray_source)
            if technique.id != XRAY_SOURCE_TECHNIQUE:
                raise ValueError(
                    f"xray_source {source.id!r} applies only to technique "
                    f"'{XRAY_SOURCE_TECHNIQUE}'; got technique {technique.id!r}"
                )
        # A copy, so that a mapping the caller changes later cannot change
        # options that are meant to be frozen.
        object.__setattr__(self, "values", dict(self.values))

    def value(self, name: str) -> OptionValue:
        """Return an engine setting by name, or None when it is unset."""
        return self.values.get(name)

    def flag(self, name: str) -> bool:
        """Return whether a flag setting is on, which only True counts as."""
        return self.value(name) is True

    def choice(self, name: str) -> str:
        """Return a choice setting.

        Raises KeyError when the setting is unset or holds no choice.
        """
        value = self.value(name)
        if not isinstance(value, str):
            raise KeyError(f"Engine setting {name!r} holds no choice; got {value!r}")
        return value

    def count(self, name: str) -> int | None:
        """Return a count setting, or None when it is unset.

        Raises KeyError when the setting holds something other than a count.
        """
        value = self.value(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise KeyError(f"Engine setting {name!r} holds no count; got {value!r}")
        return value

    def with_values(self, values: Mapping[str, OptionValue]) -> EngineOptions:
        """Return a copy with the given engine settings added or replaced.

        The new values are not checked against their declarations.
        """
        return dataclasses.replace(self, values={**self.values, **values})

    def as_settings_dict(self) -> dict[str, object]:
        """Return these options as the JSON object a manifest stores.

        Settings are sorted, so equal options give equal objects and so the
        same settings hash.
        """
        return {
            "dataset_shape": self.dataset_shape,
            # Left from a removed option that switched engines on by name. It
            # is always empty, and is still written so that runs stored while
            # it was an option keep their settings hash and can be resumed.
            "enabled_engines": [],
            "technique": self.technique,
            "xray_source": self.xray_source,
            "engine_settings": dict(sorted(self.values.items())),
        }

    @classmethod
    def from_settings_dict(cls, settings: Mapping[str, object]) -> EngineOptions:
        """Return the options a settings object from a manifest describes.

        The object may come from a file anyone could have edited, so nothing
        in it is coerced. A missing key raises ValueError, and so does a value
        the constructor refuses. A run-level value that is not a string raises
        TypeError. So does an engine setting that is not a Boolean, a string,
        an integer, or None. Keys it does not know are ignored. Whatever the
        engine list holds is ignored too, so that a run stored while the list
        could name engines still lists. Engine settings are not checked
        against their declarations, which is the job of validate_options.
        """
        missing = sorted(cls().as_settings_dict().keys() - settings.keys())
        if missing:
            raise ValueError(
                "Engine settings are incomplete; missing: " + ", ".join(missing)
            )
        raw_values = settings["engine_settings"]
        if not isinstance(raw_values, Mapping):
            raise TypeError("Engine setting 'engine_settings' must be an object")
        values: dict[str, OptionValue] = {}
        for key, value in cast(Mapping[object, object], raw_values).items():
            if not isinstance(key, str) or not (
                value is None or isinstance(value, (bool, str, int))
            ):
                raise TypeError(f"Engine setting {key!r} is not a plain value")
            values[key] = value
        return cls(
            dataset_shape=parse_dataset_shape(_text_setting(settings, "dataset_shape")),
            technique=_text_setting(settings, "technique"),
            xray_source=_text_setting(settings, "xray_source"),
            values=values,
        )


def _text_setting(settings: Mapping[str, object], name: str) -> str:
    value = settings[name]
    if not isinstance(value, str):
        raise TypeError(f"Engine setting {name!r} must be a string; got {value!r}")
    return value


def camel_case(name: str) -> str:
    """Return a snake_case name as the camelCase key the web API uses."""
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


@dataclass(frozen=True)
class OptionSpec:
    """One setting an engine reads, declared once for every place it appears.

    The command line, the web form and its JSON API, persisted settings, the
    technique compatibility map, and the run summary are all generated from
    it. A default that does not suit the kind raises ValueError when the
    declaration is made. A flag needs a Boolean default, a choice one of its
    choices, and a count a positive integer or None. A choice setting with no
    choices, or with labels that do not match them, raises ValueError too.
    """

    name: str
    label: str
    help: str
    kind: OptionKind = "flag"
    # A short explanation for the web form, which has no command-line flags
    # to refer to. The help text serves when there is none.
    caption: str = ""
    # The value a run that leaves the setting out holds. None leaves a count
    # unset.
    default: OptionValue = False
    choices: tuple[str, ...] = ()
    # Human-readable text for each choice, in the order of the choices, for
    # the web form.
    choice_labels: tuple[str, ...] = ()
    # The flag this setting refines. The web form disables the setting while
    # that flag is off, and the run summary nests it beneath the flag.
    refines: str | None = None
    # The command-line spelling. None means the name with dashes, after "--".
    cli_flag: str | None = None
    # Whether giving the flag stores False, for a setting that defaults on.
    cli_negated: bool = False
    # What the run summary says about a flag that is on or off, and about each
    # value of a choice. ``summary_override`` replaces the whole summary line when it
    # returns text, for a setting whose meaning depends on other settings.
    enabled_text: str = ""
    disabled_text: str = ""
    value_texts: tuple[tuple[str, str], ...] = ()
    summary_override: Callable[[EngineOptions], str | None] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Check the default and the choices against the kind."""
        if self.kind == "flag" and not isinstance(self.default, bool):
            raise ValueError(f"Flag option {self.name!r} needs a Boolean default")
        if self.kind == "choice":
            if not self.choices:
                raise ValueError(f"Choice option {self.name!r} declares no choices")
            if self.default not in self.choices:
                raise ValueError(
                    f"Choice option {self.name!r} default {self.default!r} is not "
                    "one of its choices"
                )
            if self.choice_labels and len(self.choice_labels) != len(self.choices):
                raise ValueError(
                    f"Choice option {self.name!r} labels do not match its choices"
                )
        # check refuses a count that is not positive, so such a default would
        # make every run that leaves the setting out fail validation.
        if (
            self.kind == "count"
            and self.default is not None
            and (
                isinstance(self.default, bool)
                or not isinstance(self.default, int)
                or self.default <= 0
            )
        ):
            raise ValueError(
                f"Count option {self.name!r} needs a positive integer default or "
                f"None; got {self.default!r}"
            )

    @property
    def form_caption(self) -> str:
        """Return the text the web form shows beneath this setting."""
        return self.caption or self.help

    @property
    def json_key(self) -> str:
        """Return the camelCase key the web API uses for this setting."""
        return camel_case(self.name)

    @property
    def resolved_cli_flag(self) -> str:
        """Return the command-line spelling of this setting."""
        return self.cli_flag or "--" + self.name.replace("_", "-")

    def check(self, value: object) -> OptionValue:
        """Return ``value`` if this setting can hold it, or raise ValueError.

        A flag takes only a Boolean, a choice only one of its choices, and a
        count only a positive integer or None. No text is parsed, since the
        value comes from code or from storage rather than from a person.
        """
        if self.kind == "flag":
            if not isinstance(value, bool):
                raise ValueError(f"{self.name} must be true or false; got {value!r}")
            return value
        if self.kind == "choice":
            if value not in self.choices:
                raise ValueError(
                    f"{self.name} must be one of {', '.join(self.choices)}; "
                    f"got {value!r}"
                )
            return str(value)
        return None if value is None else positive_int(value, self.name)

    def parse(self, raw: object) -> OptionValue:
        """Return the value that text or decoded JSON gives this setting.

        The command line reads a count setting's text through it. None or
        blank text gives the declared default. A flag takes a Boolean, or the
        text true or false in any case. Anything else is refused rather than
        read by its truthiness, so persisted settings record what was given.
        A choice takes text naming one of its choices, with surrounding
        whitespace ignored, and nothing else is converted to text. A count
        takes a positive whole number or its text. Every refusal raises
        ValueError naming the setting as check does.
        """
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return self.default
        text = raw.strip() if isinstance(raw, str) else raw
        if self.kind == "flag":
            if isinstance(text, str) and text.lower() in {"true", "false"}:
                return text.lower() == "true"
            return self.check(text)
        if self.kind == "choice":
            return self.check(text)
        return parse_positive_int(raw, self.name)

    def summary_line(self, options: EngineOptions) -> str:
        """Return the run summary's Markdown line for this setting's value.

        The line names the setting in bold and gives its value as code, then
        the declared text for that value, if any. A flag that is off and has no
        text says its columns are absent. ``summary_override`` replaces the whole line
        when it returns text.
        """
        if self.summary_override is not None:
            custom = self.summary_override(options)
            if custom is not None:
                return custom
        value = options.value(self.name)
        if self.kind != "flag":
            text = dict(self.value_texts).get(str(value), "")
        elif value:
            text = self.enabled_text
        else:
            text = (
                self.disabled_text
                or "disabled, so its columns are absent from the output."
            )
        return f"**{self.name}** (`{value}`)" + (f": {text}" if text else "")


__all__ = [
    "DATASET_SHAPES",
    "DEFAULT_DATASET_SHAPE",
    "DatasetShape",
    "DatasetShapeName",
    "EngineOptions",
    "OptionKind",
    "OptionSpec",
    "OptionValue",
    "camel_case",
    "parse_dataset_shape",
]
