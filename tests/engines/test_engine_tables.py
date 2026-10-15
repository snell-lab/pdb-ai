from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from pdbai.engines.core import (
    ColumnSpec,
    EngineContext,
    EngineExecutionError,
    EngineOptions,
    EnginePhase,
    EngineRegistry,
    EngineResult,
    EngineSpec,
    TableColumnSpec,
    TableLink,
    TableSpec,
    run_engine_phase,
    site_atom_link,
    site_key_columns,
    site_key_dataframe,
)


def _sites_table(name: str = "test_sites") -> TableSpec:
    return TableSpec(
        name=name,
        description="One row per test site.",
        key=("site_id",),
        columns=(
            TableColumnSpec("site_id"),
            TableColumnSpec("pdb_id"),
            TableColumnSpec(
                "site_type", categories=("metal", "control"), description="Kind."
            ),
            TableColumnSpec("distance_A", dtype="float64"),
            TableColumnSpec("n_ligands", dtype="Int64"),
            TableColumnSpec("is_control", dtype="boolean"),
        ),
        links=(TableLink(target="structure", columns=("pdb_id",)),),
    )


def _bonds_table() -> TableSpec:
    return TableSpec(
        name="test_bonds",
        description="One row per test bond.",
        key=("site_id", "partner"),
        columns=(TableColumnSpec("site_id"), TableColumnSpec("partner")),
        links=(TableLink(target="test_sites", columns=("site_id",)),),
    )


def _spec(
    name: str = "sites",
    *,
    tables: tuple[TableSpec, ...] = (),
    columns: tuple[ColumnSpec, ...] = (),
    phase: EnginePhase = EnginePhase.ANNOTATION,
    required: bool = True,
) -> EngineSpec:
    return EngineSpec(
        name=name,
        display_name=name.title(),
        phase=phase,
        columns=columns,
        tables=tables,
        default_enabled=True,
        required=required,
    )


@dataclass
class TableEngine:
    spec: EngineSpec
    tables: dict[str, pd.DataFrame] = field(default_factory=dict[str, pd.DataFrame])
    columns: pd.DataFrame | None = None

    def run(self, context: EngineContext) -> EngineResult:
        return EngineResult(columns=self.columns, tables=self.tables)


def _context() -> EngineContext:
    return EngineContext(
        pdb_id="1ABC",
        atom_dataframe=pd.DataFrame({"atom_name": ["N", "CA"]}, index=[10, 20]),
        options=EngineOptions(),
    )


def _run(engine: TableEngine) -> tuple[EngineContext, dict[str, pd.DataFrame]]:
    context = _context()
    report = run_engine_phase(
        context, EnginePhase.ANNOTATION, registry=EngineRegistry([engine])
    )
    return context, report.tables


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "Metal-Sites"}, "snake_case"),
        ({"name": "atoms"}, "reserved"),
        ({"key": ()}, "must declare key"),
        ({"key": ("missing",)}, "key columns are not declared"),
        (
            {"links": (TableLink(target="atom", columns=("missing",)),)},
            "undeclared columns",
        ),
        (
            {
                "links": (
                    TableLink(
                        target="atom",
                        columns=("site_id",),
                        target_columns=("a", "b"),
                    ),
                )
            },
            "pairs 1 column",
        ),
        (
            {"columns": (TableColumnSpec("site_id"), TableColumnSpec("site_id"))},
            "twice",
        ),
    ],
)
def test_table_spec_rejects_unusable_declarations(
    kwargs: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "name": "sites",
        "description": "",
        "key": ("site_id",),
        "columns": (TableColumnSpec("site_id"),),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        TableSpec(**arguments)  # type: ignore[arg-type]


def test_table_columns_reject_unknown_types_and_non_string_categories() -> None:
    with pytest.raises(ValueError, match="dtype 'object'"):
        TableColumnSpec("x", dtype="object")
    with pytest.raises(ValueError, match="only to string"):
        TableColumnSpec("x", dtype="Int64", categories=("1",))
    with pytest.raises(ValueError, match="duplicate categories"):
        TableColumnSpec("x", categories=("a", "a"))


def test_engine_spec_checks_table_phase_names_and_links() -> None:
    with pytest.raises(ValueError, match="only annotation engines"):
        _spec(tables=(_sites_table(),), phase=EnginePhase.ACQUISITION)
    with pytest.raises(ValueError, match="declares a table twice"):
        _spec(tables=(_sites_table(), _sites_table()))
    with pytest.raises(ValueError, match="nor a table of engine"):
        _spec(tables=(_bonds_table(),))
    orphan_link = TableSpec(
        name="test_bonds",
        description="",
        key=("site_id",),
        columns=(TableColumnSpec("site_id"),),
        links=(
            TableLink(
                target="test_sites", columns=("site_id",), target_columns=("nope",)
            ),
        ),
    )
    with pytest.raises(ValueError, match="does not declare"):
        _spec(tables=(_sites_table(), orphan_link))

    spec = _spec(tables=(_sites_table(), _bonds_table()))
    assert [table.name for table in spec.tables] == ["test_sites", "test_bonds"]


def test_registry_rejects_one_table_declared_by_two_engines() -> None:
    registry = EngineRegistry([TableEngine(_spec("first", tables=(_sites_table(),)))])
    with pytest.raises(ValueError, match="declared by both engines 'first'"):
        registry.register(TableEngine(_spec("second", tables=(_sites_table(),))))


def test_registry_lists_tables_of_engines_the_run_executes() -> None:
    enabled = TableEngine(_spec("enabled", tables=(_sites_table(),)))
    disabled = TableEngine(
        EngineSpec(
            name="disabled",
            display_name="Disabled",
            phase=EnginePhase.ANNOTATION,
            tables=(_sites_table("other_sites"),),
        )
    )
    registry = EngineRegistry([enabled, disabled])

    assert [t.name for t in registry.enabled_table_specs(EngineOptions())] == [
        "test_sites"
    ]
    assert registry.table_owner("other_sites") == "disabled"
    assert registry.table_owner("unknown_sites") is None


def test_coordinator_returns_tables_in_declared_order_and_types() -> None:
    engine = TableEngine(
        _spec(tables=(_sites_table(), _bonds_table())),
        tables={
            "test_sites": pd.DataFrame(
                {
                    "n_ligands": [4, None],
                    "site_id": ["A", "B"],
                    "distance_A": [2.05, None],
                    "pdb_id": ["1ABC", "1ABC"],
                    "is_control": [False, None],
                },
                index=[7, 3],
            )
        },
    )

    _context_after, tables = _run(engine)

    sites = tables["test_sites"]
    assert list(sites.columns) == list(_sites_table().column_names)
    assert sites.index.tolist() == [0, 1]
    assert str(sites["site_id"].dtype) == "string"
    assert str(sites["n_ligands"].dtype) == "Int64"
    assert str(sites["distance_A"].dtype) == "float64"
    assert str(sites["is_control"].dtype) == "boolean"
    # Declared columns the engine left out are missing, never 0 or False.
    assert sites["site_type"].isna().all()
    assert sites["is_control"].isna().tolist() == [False, True]
    # A declared table the engine did not return comes back header-only.
    bonds = tables["test_bonds"]
    assert bonds.empty
    assert list(bonds.columns) == ["site_id", "partner"]


@pytest.mark.parametrize(
    ("tables", "message"),
    [
        ({"unknown": pd.DataFrame({"site_id": ["A"]})}, "undeclared tables"),
        (
            {"test_sites": pd.DataFrame({"site_id": ["A"], "pdb_id": ["1"], "x": [1]})},
            "undeclared columns",
        ),
        ({"test_sites": pd.DataFrame({"site_id": ["A"]})}, "missing key or link"),
        (
            {"test_sites": pd.DataFrame({"site_id": ["A", "A"], "pdb_id": ["1", "1"]})},
            "duplicate rows",
        ),
        (
            {"test_sites": pd.DataFrame({"site_id": [None], "pdb_id": ["1"]})},
            "missing key values",
        ),
        (
            {
                "test_sites": pd.DataFrame(
                    {"site_id": ["A"], "pdb_id": ["1"], "site_type": ["other"]}
                )
            },
            "undeclared categories",
        ),
        (
            {
                "test_sites": pd.DataFrame(
                    {"site_id": ["A"], "pdb_id": ["1"], "n_ligands": ["four"]}
                )
            },
            "cannot be read as Int64",
        ),
        (
            {
                "test_sites": pd.DataFrame(
                    {"site_id": ["A"], "pdb_id": ["1"], "is_control": [2]}
                )
            },
            "cannot be read as boolean",
        ),
        # A fraction is refused, whether it comes as text or as a nullable
        # float, rather than cut down to a whole number.
        (
            {
                "test_sites": pd.DataFrame(
                    {
                        "site_id": ["A"],
                        "pdb_id": ["1"],
                        "n_ligands": pd.array(["4.5"], dtype="string"),
                    }
                )
            },
            "'n_ligands' cannot be read as Int64: 4.5 is not a whole number",
        ),
        (
            {
                "test_sites": pd.DataFrame(
                    {
                        "site_id": ["A"],
                        "pdb_id": ["1"],
                        "n_ligands": pd.array([4.5], dtype="Float64"),
                    }
                )
            },
            "4.5 is not a whole number",
        ),
        # A Boolean is refused in a number column rather than read as 1 or 0.
        (
            {
                "test_sites": pd.DataFrame(
                    {"site_id": ["A"], "pdb_id": ["1"], "distance_A": [True]}
                )
            },
            "'distance_A' cannot be read as float64: it holds a Boolean",
        ),
        (
            {
                "test_sites": pd.DataFrame(
                    {
                        "site_id": ["A", "B"],
                        "pdb_id": ["1", "1"],
                        "n_ligands": pd.Series([2, True], dtype=object),
                    }
                )
            },
            "'n_ligands' cannot be read as Int64: it holds a Boolean",
        ),
        (
            {
                "test_sites": pd.DataFrame(
                    [["A", "1", "1"]], columns=["site_id", "pdb_id", "pdb_id"]
                )
            },
            r"returned columns more than once: \['pdb_id'\]",
        ),
    ],
)
def test_coordinator_rejects_invalid_tables_without_touching_atoms(
    tables: dict[str, pd.DataFrame], message: str
) -> None:
    context = _context()
    engine = TableEngine(
        _spec(
            tables=(_sites_table(),),
            columns=(ColumnSpec("site_flag", group="Test Fields"),),
        ),
        tables=tables,
        columns=pd.DataFrame({"site_flag": [True, False]}, index=[10, 20]),
    )
    original = context.atom_dataframe.copy()

    with pytest.raises(EngineExecutionError, match=message):
        run_engine_phase(
            context, EnginePhase.ANNOTATION, registry=EngineRegistry([engine])
        )
    pd.testing.assert_frame_equal(context.atom_dataframe, original)


def test_an_optional_engine_with_invalid_tables_leaves_them_header_only() -> None:
    context = _context()
    engine = TableEngine(
        _spec(
            tables=(_sites_table(),),
            columns=(ColumnSpec("site_flag", group="Test Fields"),),
            required=False,
        ),
        tables={"surprise": pd.DataFrame({"x": [1]})},
        columns=pd.DataFrame({"site_flag": [True, False]}, index=[10, 20]),
    )

    report = run_engine_phase(
        context, EnginePhase.ANNOTATION, registry=EngineRegistry([engine])
    )

    assert "undeclared tables" in context.failures["sites"]
    assert report.tables["test_sites"].empty
    assert list(report.tables["test_sites"].columns) == list(
        _sites_table().column_names
    )
    assert context.atom_dataframe["site_flag"].isna().all()


def test_site_key_dataframe_builds_stable_ids_from_raw_atom_columns() -> None:
    # The raw columns mix floats, integers, and strings, as parsed atom tables
    # can. They also use the mmCIF markers ? and . for missing values.
    atoms = pd.DataFrame(
        {
            "pdb_id": ["1ABC", "1ABC"],
            "model_num": [1.0, 2],
            "chain_id": ["A", "A"],
            "res_num": ["301", 45],
            "_atom_site.pdbx_PDB_ins_code": ["?", "B"],
            "res_name": ["HEM", "CYS"],
            "atom_name": ["FE", "SG"],
            "_atom_site.label_alt_id": [".", "A"],
        },
        index=[5, 9],
    )

    keys = site_key_dataframe(atoms)
    ligand_keys = site_key_dataframe(atoms, prefix="ligand_")

    assert keys["site_id"].tolist() == ["1ABC:1:A:301::FE:", "1ABC:2:A:45:B:SG:A"]
    assert keys.index.tolist() == [5, 9]
    assert keys["model_num"].tolist() == [1, 2]
    assert keys["ins_code"].tolist() == ["", "B"]
    assert list(keys.columns) == [column.name for column in site_key_columns()]
    assert list(ligand_keys.columns) == [
        column.name for column in site_key_columns("ligand_")
    ]

    link = site_atom_link("ligand_", role="ligand")
    assert link.target == "atom"
    assert link.columns[0] == "ligand_pdb_id"
    assert "_atom_site.label_alt_id" in link.resolved_target_columns
    assert "ligand_site_id" not in link.columns


def test_an_integer_column_keeps_whole_numbers_given_as_text_or_floats() -> None:
    engine = TableEngine(
        _spec(tables=(_sites_table(),)),
        tables={
            "test_sites": pd.DataFrame(
                {
                    "site_id": ["A", "B", "C"],
                    "pdb_id": ["1ABC"] * 3,
                    "n_ligands": pd.array(["4", "6.0", None], dtype="string"),
                    "distance_A": [2, 2.5, None],
                }
            )
        },
    )

    _context_after, tables = _run(engine)

    sites = tables["test_sites"]
    assert sites["n_ligands"].tolist() == [4, 6, pd.NA]
    assert sites["distance_A"].tolist()[:2] == [2.0, 2.5]


def test_site_key_dataframe_leaves_a_fractional_model_number_missing() -> None:
    # A model number of 1.5 is left missing rather than read as model 1.
    atoms = pd.DataFrame({"pdb_id": ["1ABC", "1ABC"], "model_num": ["1.5", "2"]})

    keys = site_key_dataframe(atoms)

    assert keys["model_num"].tolist() == [pd.NA, 2]
    assert keys["site_id"].tolist() == ["1ABC:1.5:::::", "1ABC:2:::::"]


def test_site_key_columns_validate_as_a_table() -> None:
    table = TableSpec(
        name="metal_ligand_bonds",
        description="",
        key=("site_id", "ligand_site_id"),
        columns=(TableColumnSpec("site_id"), *site_key_columns("ligand_")),
        links=(site_atom_link("ligand_", role="ligand"),),
    )
    atoms = pd.DataFrame({"pdb_id": ["1ABC"], "atom_name": ["SG"], "res_num": [4]})
    rows = site_key_dataframe(atoms, prefix="ligand_").assign(site_id="1ABC:1:A:1::ZN:")

    engine = TableEngine(_spec(tables=(table,)), tables={table.name: rows})
    _context_after, tables = _run(engine)

    assert tables["metal_ligand_bonds"]["ligand_site_id"].tolist() == ["1ABC:::4::SG:"]
