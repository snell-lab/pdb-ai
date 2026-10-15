from __future__ import annotations

import pytest

from pdbai.engines.core import HOST_REPORT_COLUMNS, ReportField


@pytest.mark.parametrize("name", HOST_REPORT_COLUMNS)
def test_report_field_cannot_take_a_host_column_name(name: str) -> None:
    with pytest.raises(ValueError, match="reserved for a run report column"):
        ReportField(name, "Clashes with the host.")


def test_report_field_cannot_take_a_row_count_column_name() -> None:
    with pytest.raises(ValueError, match="engine table row counts"):
        ReportField("stub_sites_row_count", "Clashes with a table's row count.")


def test_report_field_may_mention_a_host_column_inside_its_name() -> None:
    report_field = ReportField("stub_state", "Outcome.")

    assert report_field.name == "stub_state"
