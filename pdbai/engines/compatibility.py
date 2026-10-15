"""Run settings checked against what the registered engines declare.

validate_options refuses settings that cannot be run as given.
technique_conflicts names each engine a run would execute that the run's
experimental technique rules out, and require_compatible_options refuses a
run that has any. Every build calls require_compatible_options before it
starts, so no engine computes values for a technique its science does not
apply to.

toggle_compatibility and compatible_toggles tell the web app and the column
profiles which flags apply to a technique.

Every function checks against the shared engine registry, so an engine
registered from outside PDB-AI is checked like a built-in one. What applies
to a technique is read from the engines' declarations rather than listed
here, so a new engine needs no change to this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import get_engine_registry
from .core import DatasetSource, EngineOptions, get_technique


@dataclass(frozen=True)
class ToggleCompatibility:
    """The techniques one flag applies to, and its label."""

    toggle: str
    label: str
    # None means the flag applies to every technique.
    techniques: frozenset[str] | None

    def supports(self, technique: str) -> bool:
        """Return whether the flag applies to ``technique``."""
        return self.techniques is None or technique in self.techniques

    def to_dict(self) -> dict[str, object]:
        """Return the record as a dict JSON can hold, keyed by field name.

        The techniques are a sorted list, or None when the flag applies to
        every technique.
        """
        return {
            "toggle": self.toggle,
            "label": self.label,
            "techniques": (
                None if self.techniques is None else sorted(self.techniques)
            ),
        }


def _intersect(
    left: frozenset[str] | None, right: frozenset[str] | None
) -> frozenset[str] | None:
    if left is None:
        return right
    if right is None:
        return left
    return left & right


def toggle_compatibility() -> tuple[ToggleCompatibility, ...]:
    """Return the techniques each flag applies to, sorted by flag name.

    A flag that switches on several engines, such as an acquisition and
    annotation pair, applies only to the techniques all of them support. A
    flag that switches on only some of an engine's columns limits no
    technique. A flag that refines another applies only where that one
    applies too.
    """
    registry = get_engine_registry()
    supported: dict[str, frozenset[str] | None] = {}
    for engine in registry.engines_to_run():
        toggles = {
            engine.spec.toggle,
            *(column.toggle for column in engine.spec.columns),
        }
        for toggle in toggles:
            if toggle is None:
                continue
            restriction = (
                engine.spec.techniques if toggle == engine.spec.toggle else None
            )
            supported[toggle] = _intersect(supported.get(toggle), restriction)
    refines = {option.name: option.refines for option in registry.option_specs()}

    def applicable(name: str) -> frozenset[str] | None:
        # A refining flag is offered only while the flag it refines is on,
        # so it is of no use where that flag is ruled out.
        techniques = supported.get(name)
        seen = {name}
        refined = refines.get(name)
        while refined is not None and refined not in seen:
            seen.add(refined)
            techniques = _intersect(techniques, supported.get(refined))
            refined = refines.get(refined)
        return techniques

    return tuple(
        ToggleCompatibility(
            toggle=option.name,
            label=option.label,
            techniques=applicable(option.name),
        )
        for option in sorted(registry.option_specs(), key=lambda option: option.name)
        if option.kind == "flag"
    )


def technique_conflicts(options: EngineOptions) -> list[str]:
    """Return a message for each engine the run executes outside its technique.

    Every engine the run would execute and that does not support the run's
    technique is covered, whatever the dataset source. An engine the run
    switches on through its own toggle is named by that flag, once for all
    the engines of one toggle. Any other engine is named by its own name,
    such as one switched on in enabled_engines or one that a supported
    engine depends on. An engine that runs only because unsupported engines
    depend on it gets no message, since theirs already rule the run out.
    """
    registry = get_engine_registry()
    technique = get_technique(options.technique)
    running = registry.engines_to_run(options=options)
    unsupported = {
        engine.spec.name
        for engine in running
        if not engine.spec.supports_technique(technique.id)
    }
    needed_by_supported = {
        dependency
        for engine in running
        if engine.spec.name not in unsupported
        for dependency in engine.spec.dependencies_under(options)
    }
    messages: list[str] = []
    reported_toggles: set[str] = set()

    for engine in running:
        spec = engine.spec
        if spec.name not in unsupported:
            continue
        switched_on = spec.is_enabled(options)
        if not switched_on and spec.name not in needed_by_supported:
            # PDB-Redo runs for RABDAM in this way, and the message about
            # RABDAM tells the user what to change.
            continue
        toggle = spec.toggle
        if switched_on and toggle is not None and options.flag(toggle):
            if toggle in reported_toggles:
                continue
            reported_toggles.add(toggle)
            declared = registry.option_spec(toggle)
            label = declared.label if declared is not None else spec.display_name
            messages.append(
                f"{label} ({toggle}) is not available for {technique.label}"
            )
        else:
            messages.append(
                f"Engine {spec.name!r} is not available for {technique.label}"
            )
    return messages


def validate_options(
    options: EngineOptions, *, source: DatasetSource = "download"
) -> None:
    """Raise ValueError when a run's settings cannot be run as given.

    A setting no engine declares is refused, and so is a value its
    declaration does not allow and an engine name that is not registered.
    Every artifact an enabled engine reads must come from an engine that runs
    before it on a dataset from ``source``. The technique is not checked.
    """
    registry = get_engine_registry()
    declared = {option.name: option for option in registry.option_specs()}
    unknown = sorted(set(options.values) - set(declared))
    if unknown:
        raise ValueError(f"No engine declares the settings {unknown}")
    for name, value in options.values.items():
        declared[name].check(value)
    unknown_engines = sorted(
        name for name in options.enabled_engines if name not in registry
    )
    if unknown_engines:
        raise ValueError(f"Unknown engines requested: {unknown_engines}")
    registry.check_artifacts(options, source)


def require_compatible_options(
    options: EngineOptions, *, source: DatasetSource = "download"
) -> None:
    """Raise ValueError as validate_options does, or for unsupported engines.

    The message joins every message technique_conflicts returns for the
    run.
    """
    validate_options(options, source=source)
    messages = technique_conflicts(options)
    if messages:
        raise ValueError("; ".join(messages) + ".")


def compatible_toggles(technique: str) -> frozenset[str]:
    """Return the flags that apply to ``technique``.

    An unknown technique raises ValueError, rather than getting only the
    flags that apply to every technique.
    """
    get_technique(technique)
    return frozenset(
        record.toggle for record in toggle_compatibility() if record.supports(technique)
    )


__all__ = [
    "ToggleCompatibility",
    "compatible_toggles",
    "require_compatible_options",
    "technique_conflicts",
    "toggle_compatibility",
    "validate_options",
]
