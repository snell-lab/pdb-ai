from __future__ import annotations

import pytest

from pdbai.json_values import object_dict, object_list, optional_object_dict, str_list


def test_dict_helpers_return_a_new_dict_for_a_mapping() -> None:
    document: dict[str, object] = {"state": "done"}
    for narrowed in (optional_object_dict(document), object_dict(document)):
        assert narrowed == {"state": "done"}
        assert narrowed is not document


@pytest.mark.parametrize("value", [None, ["state"], "state", 1])
def test_dict_helpers_differ_on_a_value_that_is_not_a_mapping(value: object) -> None:
    assert optional_object_dict(value) is None
    assert object_dict(value) == {}


def test_object_list_returns_a_new_list() -> None:
    document: list[object] = ["1abc", None]
    narrowed = object_list(document)
    assert narrowed == ["1abc", None]
    assert narrowed is not document


@pytest.mark.parametrize("value", [None, {"pdb_ids": []}, "1abc", ("1abc",)])
def test_list_helpers_return_an_empty_list_for_a_value_that_is_not_a_list(
    value: object,
) -> None:
    assert object_list(value) == []
    assert str_list(value) == []


def test_str_list_leaves_out_items_that_are_not_strings() -> None:
    methods = [
        "X-RAY DIFFRACTION",
        None,
        1,
        ["NEUTRON DIFFRACTION"],
        "ELECTRON CRYSTALLOGRAPHY",
    ]
    assert str_list(methods) == ["X-RAY DIFFRACTION", "ELECTRON CRYSTALLOGRAPHY"]
