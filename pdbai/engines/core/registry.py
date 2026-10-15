"""The engines a process knows, and the order in which a run executes them.

EngineRegistry holds every registered engine by name. Registration checks
what can be checked of one engine alone. validate checks what spans engines,
once all of them are registered. Each column, table, setting, and report
field has exactly one owning engine, since everything shown about it is
generated from that one declaration.

A run executes the engines it switches on and every engine they depend on.
Acquisition engines run before annotation engines, so an engine may depend
only on engines of its own phase or an earlier one. Within a phase, each
engine follows its dependencies. Engines with no order between them keep the
order they were registered in, which is also the order they are displayed
in. So the running order is the same every time.

The registry also derives what the rest of PDB-AI needs from the engines'
declarations. The tables, report fields, preflight checks, and reference
data receipts follow the engines a run executes. The settings cover every
registered engine, so a run stores a value for each of them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .artifacts import HOST_ARTIFACTS, ArtifactKey
from .contracts import (
    DatasetSource,
    Engine,
    EnginePhase,
    EngineSpec,
    ReportField,
    TableSpec,
)
from .options import EngineOptions, OptionSpec

_PHASE_ORDER = (EnginePhase.ACQUISITION, EnginePhase.ANNOTATION)


def _cycle(path: Sequence[str], name: str) -> str:
    """Return the cycle that ``name`` closes on ``path``, as arrows."""
    return " -> ".join((*path[path.index(name) :], name))


class EngineRegistry:
    """The engines a process can run, by name."""

    def __init__(self, engines: Iterable[Engine] = ()) -> None:
        """Register each of ``engines`` in order."""
        self._engines: dict[str, Engine] = {}
        for engine in engines:
            self.register(engine)

    def register(self, engine: Engine) -> None:
        """Add an engine after checking it against those already registered.

        An empty or duplicate name raises ValueError. So does a column, table,
        setting, or report field that another engine already declares. The
        schema catalog refuses a column that takes a base column's name, when
        the catalog is next built. The registry cannot, since it does not know
        the base columns.
        """
        spec = engine.spec
        name = spec.name
        if not name:
            raise ValueError("Engine names must not be empty")
        if name in self._engines:
            raise ValueError(f"Engine {name!r} is already registered")
        for other in self._engines.values():
            for label, mine, theirs in (
                ("Column", spec.columns, other.spec.columns),
                ("Table", spec.tables, other.spec.tables),
                ("Option", spec.options, other.spec.options),
                ("Report field", spec.report_fields, other.spec.report_fields),
            ):
                clash = {item.name for item in mine} & {item.name for item in theirs}
                if clash:
                    raise ValueError(
                        f"{label} {sorted(clash)[0]!r} is declared by both engines "
                        f"{other.spec.name!r} and {name!r}"
                    )
        self._engines[name] = engine

    def validate(self) -> None:
        """Check the declarations that span several engines.

        A declaration that names another engine or setting can be checked
        only once every engine is registered. Each problem raises ValueError.
        The toggle of an engine or of a column must name a flag that some
        engine declares, and so must the flag a setting refines. No setting
        may refine itself, directly or through the settings it refines. Each
        engine named as a dependency or a conditional dependency must be
        registered, and must not run in a later phase, since it could then
        never run first. Last, no engines may depend on one another in a
        cycle. Every conditional dependency counts toward a cycle, whatever
        settings would bring it in.
        """
        flags = set(self.flag_names())
        refines = {option.name: option.refines for option in self.option_specs()}
        for name, refined in refines.items():
            if refined is not None and refined not in flags:
                raise ValueError(
                    f"Option {name!r} refines {refined!r}, which no engine "
                    "declares as a flag"
                )
        for name in refines:
            # The web form disables a setting while the flag it refines is
            # off, so settings that refine one another could never be on.
            chain = [name]
            refined = refines[name]
            while refined is not None:
                if refined in chain:
                    raise ValueError(
                        "Options refine one another in a cycle: "
                        + _cycle(chain, refined)
                    )
                chain.append(refined)
                refined = refines[refined]
        for spec in self.registered_specs:
            gates = [
                (spec.toggle, ""),
                *(
                    (column.toggle, f" through column {column.name!r}")
                    for column in spec.columns
                ),
            ]
            for toggle, where in gates:
                if toggle is not None and toggle not in flags:
                    raise ValueError(
                        f"Engine {spec.name!r} is gated by {toggle!r}{where}, which "
                        "no engine declares as a flag option"
                    )
            for dependency in (*spec.dependencies, *spec.conditional_dependency_names):
                if dependency not in self._engines:
                    raise ValueError(
                        f"Engine {spec.name!r} refers to unknown engine {dependency!r}"
                    )
                other_phase = self._engines[dependency].spec.phase
                if _PHASE_ORDER.index(other_phase) > _PHASE_ORDER.index(spec.phase):
                    raise ValueError(
                        f"Engine {spec.name!r} runs in the {spec.phase.value} "
                        f"phase but needs engine {dependency!r} first, which "
                        f"runs in the later {other_phase.value} phase"
                    )
        # Ordering every engine without settings follows every dependency,
        # conditional or not, so it meets any cycle.
        self.engines_to_run()

    @property
    def registered_specs(self) -> tuple[EngineSpec, ...]:
        """Return every registered spec in registration order.

        Registration order is display order, the order in which the schema
        catalog lists column groups.
        """
        return tuple(engine.spec for engine in self._engines.values())

    def engines_to_run(
        self,
        *,
        phase: EnginePhase | None = None,
        options: EngineOptions | None = None,
        source: DatasetSource | None = None,
    ) -> tuple[Engine, ...]:
        """Return the engines a run executes, in the order it runs them.

        Without options, every registered engine is returned. With a source,
        engines that do not apply to it are left out, and so are dependencies
        on them. Without a phase, both phases are returned. A dependency cycle
        and a dependency that names no engine raise ValueError.
        """

        def applies(name: str) -> bool:
            return source is None or source in self._engines[name].spec.sources

        roots = [
            name
            for name, engine in self._engines.items()
            if applies(name) and (options is None or engine.spec.is_enabled(options))
        ]
        # A dependency runs even when the run does not switch it on, since the
        # engine that needs it cannot run without it.
        included: set[str] = set()
        resolving: list[str] = []

        def include(name: str) -> None:
            if name in included:
                return
            if name in resolving:
                raise ValueError(f"Engine dependency cycle: {_cycle(resolving, name)}")
            resolving.append(name)
            engine = self._engines[name]
            for dependency in engine.spec.dependencies_under(options):
                if dependency not in self._engines:
                    raise ValueError(
                        f"Engine {name!r} depends on unknown engine {dependency!r}"
                    )
                if applies(dependency):
                    include(dependency)
            resolving.pop()
            included.add(name)

        for engine_name in roots:
            include(engine_name)

        phases = _PHASE_ORDER if phase is None else (phase,)
        ordered: list[Engine] = []
        for current_phase in phases:
            candidates = {
                name: engine
                for name, engine in self._engines.items()
                if name in included and engine.spec.phase == current_phase
            }
            ordered.extend(self._in_dependency_order(candidates, options))
        return tuple(ordered)

    def _in_dependency_order(
        self,
        candidates: dict[str, Engine],
        options: EngineOptions | None,
    ) -> list[Engine]:
        # engines_to_run has already refused any cycle among the candidates.
        ordered: list[Engine] = []
        visited: set[str] = set()

        # A depth-first visit in registration order, so that engines with no
        # order between them keep that order.
        def visit(name: str) -> None:
            # A name that is not a candidate is skipped. It is in an earlier
            # phase, which has already run, or it is not in the run. validate
            # refuses one in a later phase.
            if name in visited or name not in candidates:
                return
            visited.add(name)
            engine = candidates[name]
            for dependency in engine.spec.dependencies_under(options):
                visit(dependency)
            ordered.append(engine)

        for engine_name in candidates:
            visit(engine_name)
        return ordered

    def check_artifacts(
        self, options: EngineOptions, source: DatasetSource = "download"
    ) -> None:
        """Check that each artifact a run's engines read is produced earlier.

        The run has ``options`` and builds a dataset from ``source``. The
        host's own artifacts count as produced before any engine runs. An
        artifact that no earlier engine provides raises ValueError naming the
        engine that reads it. So does one that an earlier producer provides
        as a type the reader's key does not accept, that is, a type that is
        not the key's type or a subclass of it. The check runs before a run
        starts, rather than failing partway through a structure.
        """
        # Every type a name may hold when an engine reads it. A later
        # producer replaces an earlier one's value only when it succeeds, so
        # a reader may meet any of them.
        available: dict[str, list[type[Any]]] = {}

        def provide(keys: Iterable[ArtifactKey[Any]]) -> None:
            for key in keys:
                types = available.setdefault(key.name, [])
                if key.type not in types:
                    types.append(key.type)

        provide(HOST_ARTIFACTS)
        for engine in self.engines_to_run(options=options, source=source):
            spec = engine.spec
            missing = [key.name for key in spec.consumes if key.name not in available]
            if missing:
                raise ValueError(
                    f"Engine {spec.name!r} reads {missing}, which no engine that "
                    "runs before it provides"
                )
            for key in spec.consumes:
                for provided in available[key.name]:
                    if not issubclass(provided, key.type):
                        raise ValueError(
                            f"Engine {spec.name!r} reads {key.name!r} as type "
                            f"{key.type.__name__}, but it is provided as type "
                            f"{provided.__name__}"
                        )
            provide(spec.provides)

    def preflight(
        self, options: EngineOptions, source: DatasetSource = "download"
    ) -> None:
        """Run the preflight check of every engine a run executes.

        The first check that raises stops the rest, and its error propagates.
        """
        for engine in self.engines_to_run(options=options, source=source):
            if engine.spec.preflight is not None:
                engine.spec.preflight()

    def reference_data(
        self, options: EngineOptions, source: DatasetSource = "download"
    ) -> dict[str, object]:
        """Return receipts for the data files that a run's engines ship with.

        An engine whose receipts cannot be produced gets an entry under its
        own name that records the error. Provenance collection reports its own
        failure rather than stopping the run.
        """
        receipts: dict[str, object] = {}
        for engine in self.engines_to_run(options=options, source=source):
            hook = engine.spec.reference_data
            if hook is None:
                continue
            try:
                receipts.update(hook())
            except Exception as exception:
                receipts[engine.spec.name] = {
                    "available": False,
                    "error": str(exception),
                }
        return receipts

    def option_specs(self) -> tuple[OptionSpec, ...]:
        """Return every declared setting, in registration order."""
        return tuple(
            option
            for engine in self._engines.values()
            for option in engine.spec.options
        )

    def with_defaults(self, options: EngineOptions) -> EngineOptions:
        """Return ``options`` with each setting it leaves out at its default.

        When it leaves none out, ``options`` itself is returned.
        """
        missing = {
            option.name: option.default
            for option in self.option_specs()
            if option.name not in options.values
        }
        return options.with_values(missing) if missing else options

    def option_spec(self, name: str) -> OptionSpec | None:
        """Return the declaration of the setting ``name``, or None."""
        return next(
            (option for option in self.option_specs() if option.name == name), None
        )

    def flag_names(self) -> tuple[str, ...]:
        """Return the names of the declared flag settings, in registration order."""
        return tuple(
            option.name for option in self.option_specs() if option.kind == "flag"
        )

    def profile_patterns(self) -> dict[str, tuple[str, ...]]:
        """Return the glob selectors engines declare, by column profile.

        A selector that several engines declare appears once.
        """
        patterns: dict[str, list[str]] = {}
        for spec in self.registered_specs:
            for profile_id, selectors in spec.profile_patterns:
                declared = patterns.setdefault(profile_id, [])
                for selector in selectors:
                    if selector not in declared:
                        declared.append(selector)
        return {
            profile_id: tuple(selectors) for profile_id, selectors in patterns.items()
        }

    def enabled_table_specs(
        self, options: EngineOptions, source: DatasetSource = "download"
    ) -> tuple[TableSpec, ...]:
        """Return the engine tables a run writes, in running order.

        The run has ``options`` and builds a dataset from ``source``.
        """
        return tuple(
            table
            for engine in self.engines_to_run(
                phase=EnginePhase.ANNOTATION, options=options, source=source
            )
            for table in engine.spec.tables
        )

    def table_owner(self, name: str) -> str | None:
        """Return the name of the engine that declares table ``name``.

        A table that no engine declares gives None.
        """
        return next(
            (
                spec.name
                for spec in self.registered_specs
                if any(table.name == name for table in spec.tables)
            ),
            None,
        )

    def report_fields(
        self, options: EngineOptions, source: DatasetSource | None = None
    ) -> tuple[ReportField, ...]:
        """Return the report fields of the engines a run executes, in order.

        With a ``source``, engines that do not apply to it are left out.
        """
        return tuple(
            report_field
            for engine in self.engines_to_run(options=options, source=source)
            for report_field in engine.spec.report_fields
        )

    def get(self, name: str) -> Engine:
        """Return the engine registered as ``name``, or raise KeyError."""
        try:
            return self._engines[name]
        except KeyError as exception:
            raise KeyError(f"Unknown engine {name!r}") from exception

    def __contains__(self, name: str) -> bool:
        """Return whether an engine is registered as ``name``."""
        return name in self._engines


__all__ = ["EngineRegistry"]
