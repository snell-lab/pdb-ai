"""Shared fixtures for the test suite.

pytest loads this module for every test under tests/, so it holds only the
fixtures that tests in several packages need. Together they let a test run
without the network and leave nothing behind for the next one. They provide
engine options under which nothing fetches, a registry swapped in for one
test, a logger level set for one test, and an entry written to a temporary
directory.

``install_engines`` swaps the catalog's registry through ``monkeypatch``
rather than registering into the shared one, so the built-in registry comes
back when the test ends, even when the test fails. It validates the new
registry before installing it, so an engine whose declarations do not fit
the built-ins fails where it is installed rather than partway through a run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from pdbai.engines import catalog
from pdbai.engines.core import Engine, EngineOptions, EngineRegistry
from pdbai.network import NETWORK_LOGGER_NAME


@pytest.fixture
def offline_options() -> EngineOptions:
    """Return options under which no engine reaches the network.

    Every network-backed engine is off by default, RABDAM included. Override
    run-level fields with ``dataclasses.replace`` and engine settings with
    ``with_values``.
    """
    return EngineOptions(values={"include_rabdam": False})


@pytest.fixture
def install_engines(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Return a function that registers engines after the built-in ones."""

    def install(*engines: Engine) -> None:
        registry = EngineRegistry((*catalog.BUILTIN_ENGINES, *engines))
        registry.validate()
        monkeypatch.setattr(catalog, "_registry", registry)

    return install


@pytest.fixture
def network_logs(
    caplog: pytest.LogCaptureFixture,
) -> pytest.LogCaptureFixture:
    """Return ``caplog`` recording the network logger from INFO up.

    Cache hits are logged at INFO. Without this fixture they reach caplog
    only when an earlier CLI test has set the package logger to INFO, so a
    test that reads them would depend on test order.
    """
    caplog.set_level(logging.INFO, logger=NETWORK_LOGGER_NAME)
    return caplog


@pytest.fixture
def minimal_cif(tmp_path: Path) -> Path:
    """Write entry 1ABC, one glycine and a zinc ion, and return its path.

    The CA atom has alternate conformers A at occupancy 0.40 and B at 0.80,
    so the parser keeps B and drops atom 2. Author numbering (chain X,
    residue 7) differs from label numbering (chain A, residue 1), so a test
    can tell which of the two a column was read from. The cell is 10 by 20 by
    30 angstroms at right angles, a volume of 6000 cubic angstroms. Several
    tests edit this text by exact match, so a changed line here must be
    changed there too.
    """
    path = tmp_path / "1abc.cif"
    path.write_text(
        """data_1abc
_cell.length_a 10
_cell.length_b 20
_cell.length_c 30
_cell.angle_alpha 90
_cell.angle_beta 90
_cell.angle_gamma 90
_refine.ls_d_res_high 1.50

loop_
_entity.id
_entity.type
1 polymer
2 non-polymer

loop_
_entity_poly.entity_id
_entity_poly.type
1 'polypeptide(L)'

loop_
_chem_comp.id
_chem_comp.type
GLY 'L-peptide linking'
ZN non-polymer

loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . GLY A 1 1 ? 0 0 0 1.00 10 7 GLY X N 1
ATOM 2 C CA A GLY A 1 1 ? 1 0 0 0.40 20 7 GLY X CA 1
ATOM 3 C CA B GLY A 1 1 ? 2 0 0 0.80 30 7 GLY X CA 1
ATOM 4 C C . GLY A 1 1 ? 3 0 0 1.00 50 7 GLY X C 1
HETATM 5 ZN ZN . ZN B 2 . ? 4 0 0 1.00 60 8 ZN X ZN 1
""",
        encoding="utf-8",
    )
    return path
