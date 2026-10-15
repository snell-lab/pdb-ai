from __future__ import annotations

import math

import numpy as np
import pytest

from pdbai.engines.analyses.sasa import sphere
from pdbai.engines.analyses.sasa.sphere import (
    compute_sasa,
    get_sphere_points,
    resolve_sphere_point_count,
)


def _cap_free_area(radius: float, other: float, distance: float) -> float:
    # The exact area of a sphere of radius ``radius`` left outside a sphere of
    # radius ``other`` whose centre lies ``distance`` away.
    cut = (distance**2 + radius**2 - other**2) / (2.0 * distance)
    return 4.0 * math.pi * radius**2 - 2.0 * math.pi * radius * (radius - cut)


def test_sphere_points_lie_on_the_unit_sphere_and_are_shared_read_only() -> None:
    points = get_sphere_points(500)

    assert points.shape == (500, 3)
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0)
    # An even spread has its centre of mass at the sphere's centre.
    assert np.allclose(points.mean(axis=0), 0.0, atol=1e-2)
    assert get_sphere_points(500) is points
    with pytest.raises(ValueError, match="read-only"):
        points[0, 0] = 2.0


def test_a_cached_point_count_does_not_admit_its_float_or_boolean_twin() -> None:
    get_sphere_points(92)
    get_sphere_points(1)

    with pytest.raises(ValueError, match="sphere point count must be a positive"):
        get_sphere_points(92.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sphere point count must be a positive"):
        get_sphere_points(True)


def test_an_isolated_atom_keeps_the_whole_enlarged_sphere() -> None:
    (area,) = compute_sasa(np.zeros((1, 3)), np.array([1.8]), n_sphere_points=92)

    assert area == pytest.approx(4.0 * math.pi * (1.8 + 1.4) ** 2, rel=1e-12)


@pytest.mark.parametrize("point_count", [1_000, 5_000])
def test_two_overlapping_atoms_match_the_exact_area(point_count: int) -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    radii = np.array([1.87, 1.65])

    areas = compute_sasa(coordinates, radii, n_sphere_points=point_count)

    assert areas[0] == pytest.approx(_cap_free_area(3.27, 3.05, 2.0), rel=5e-3)
    assert areas[1] == pytest.approx(_cap_free_area(3.05, 3.27, 2.0), rel=5e-3)


def test_areas_do_not_depend_on_where_the_structure_sits() -> None:
    # Far from the origin, a squared distance expanded from the absolute
    # coordinates would lose enough precision to move test points across a
    # neighbour's surface.
    rng = np.random.default_rng(3)
    coordinates = rng.uniform(0.0, 12.0, size=(60, 3))
    radii = np.full(60, 1.8)

    here = compute_sasa(coordinates, radii, n_sphere_points=1_000)
    far = compute_sasa(coordinates + 1.0e7, radii, n_sphere_points=1_000)

    assert np.array_equal(here, far)


def test_targets_are_returned_in_order_and_chunking_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(7)
    coordinates = rng.uniform(0.0, 10.0, size=(40, 3))
    radii = rng.choice([1.4, 1.65, 1.8], size=40)
    targets = np.array([5, 0, 39, 5])

    whole = compute_sasa(coordinates, radii, n_sphere_points=92)
    chosen = compute_sasa(
        coordinates, radii, n_sphere_points=92, target_indices=targets
    )
    monkeypatch.setattr(sphere, "_NEIGHBOR_CHUNK_SIZE", 1)
    one_at_a_time = compute_sasa(coordinates, radii, n_sphere_points=92)

    assert np.array_equal(chosen, whole[targets])
    assert np.array_equal(one_at_a_time, whole)


def test_no_atoms_and_no_targets_give_no_areas() -> None:
    assert compute_sasa(np.empty((0, 3)), np.empty(0)).shape == (0,)
    assert compute_sasa(
        np.zeros((1, 3)), np.ones(1), target_indices=np.empty(0, dtype=int)
    ).shape == (0,)


@pytest.mark.parametrize(
    ("coordinates", "radii", "options", "error", "message"),
    [
        (np.zeros((2, 2)), np.ones(2), {}, ValueError, r"shape \(N, 3\)"),
        (np.zeros((2, 3)), np.ones(3), {}, ValueError, r"radii must have shape"),
        (
            np.array([[0.0, 0.0, np.nan]]),
            np.ones(1),
            {},
            ValueError,
            "coordinates must be finite",
        ),
        (np.zeros((1, 3)), np.zeros(1), {}, ValueError, "finite and positive"),
        (
            np.zeros((1, 3)),
            np.ones(1),
            {"n_sphere_points": 0},
            ValueError,
            "sphere point count must be a positive integer",
        ),
        (
            np.zeros((1, 3)),
            np.ones(1),
            {"target_indices": np.array([[0]])},
            ValueError,
            "one-dimensional",
        ),
        (
            np.zeros((2, 3)),
            np.ones(2),
            {"target_indices": np.array([2])},
            IndexError,
            "0 to 1; got 2 to 2",
        ),
    ],
)
def test_compute_sasa_refuses_bad_input(
    coordinates: np.ndarray,
    radii: np.ndarray,
    options: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        compute_sasa(coordinates, radii, **options)  # type: ignore[arg-type]


def test_accuracy_modes_resolve_to_their_point_counts() -> None:
    assert resolve_sphere_point_count("fast") == ("fast", 92)
    assert resolve_sphere_point_count("default") == ("default", 1_000)
    assert resolve_sphere_point_count("high") == ("high", 5_000)
    # An explicit count wins, and the mode it replaces is not checked.
    assert resolve_sphere_point_count("nonsense", 123) == ("custom", 123)
    with pytest.raises(ValueError, match="Unknown SASA accuracy mode 'exact'"):
        resolve_sphere_point_count("exact")
    with pytest.raises(ValueError, match="positive integer"):
        resolve_sphere_point_count("fast", 0)
