"""The Shrake-Rupley calculation of the solvent-accessible area of atoms.

The method is that of Shrake and Rupley (1973). Each atom's sphere is
enlarged by the probe radius, which gives the surface the probe's centre
can reach. That surface is sampled with a fixed set of test points. A test
point is buried when it lies inside the enlarged sphere of another atom. The
atom's area is the area of its enlarged sphere times the fraction of test
points left unburied.

The test points form a Fibonacci lattice. This is a spiral that spreads any
number of points almost evenly over a sphere, so the user can choose any
point count. The lattice is the same for every atom and every run, and it is
never rotated. At a given point count, the same coordinates therefore always
give the same areas.

The areas carry a sampling error, which shrinks as the point count grows. A
residue whose relative area lies close to a burial threshold can change class
from one point count to another.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

from ....positive_numbers import positive_int
from .constants import (
    SASA_ACCURACY_POINT_COUNTS,
    SASA_DEFAULT_N_SPHERE_POINTS,
    SASA_PROBE_RADIUS,
)

# How many neighbours of one atom are tested against its points at once. It
# bounds the (points, neighbours) array in memory and does not change areas.
_NEIGHBOR_CHUNK_SIZE = 256


def resolve_sphere_point_count(
    accuracy_mode: str, n_sphere_points: int | None = None
) -> tuple[str, int]:
    """Return the accuracy mode and the sphere-point count it stands for.

    An explicit ``n_sphere_points`` overrides ``accuracy_mode``. The mode is
    then reported as "custom" and is not checked.

    Raises ValueError for a mode not in SASA_ACCURACY_POINT_COUNTS or a point
    count that is not a positive integer.
    """
    if n_sphere_points is not None:
        return "custom", positive_int(n_sphere_points, "sphere point count")
    if accuracy_mode not in SASA_ACCURACY_POINT_COUNTS:
        allowed = ", ".join(sorted(SASA_ACCURACY_POINT_COUNTS))
        raise ValueError(
            f"Unknown SASA accuracy mode {accuracy_mode!r}; expected {allowed}."
        )
    return accuracy_mode, SASA_ACCURACY_POINT_COUNTS[accuracy_mode]


def get_sphere_points(n: int) -> np.ndarray:
    """Return ``n`` points spread almost evenly over the unit sphere.

    The points form an (n, 3) array, which is cached and shared by every
    caller, so it is read-only. Raises ValueError when ``n`` is not a
    positive integer.
    """
    # The count is checked before the cache is consulted. The cache matches
    # its keys by equality, so 92.0 or True would otherwise find the array of
    # 92 or 1 points whenever an earlier call had cached it.
    return _sphere_points(positive_int(n, "sphere point count"))


@lru_cache(maxsize=16)
def _sphere_points(point_count: int) -> np.ndarray:
    # Each point sits in the middle of one of n bands of equal area, and
    # each turns from the one before by the golden angle. No two points then
    # line up along a meridian.
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    i = np.arange(point_count, dtype=float)
    theta = np.arccos(1.0 - 2.0 * (i + 0.5) / point_count)
    phi = 2.0 * np.pi * i / golden
    points = np.column_stack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        )
    )
    points.setflags(write=False)
    return points


def compute_sasa(
    coordinates: np.ndarray,
    radii: np.ndarray,
    n_sphere_points: int = SASA_DEFAULT_N_SPHERE_POINTS,
    target_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return the solvent-accessible area in Å² of each target atom.

    ``coordinates`` is an (N, 3) array in Å, and ``radii`` holds the N atomic
    radii in Å, which are enlarged here by SASA_PROBE_RADIUS.
    ``n_sphere_points`` is the number of test points per atom. Every atom can
    bury the test points of the others. Only the atoms at ``target_indices``
    get an area, returned in that order, and by default every atom is a
    target.

    Raises ValueError for a wrong shape, a coordinate or radius that is not
    finite, a radius that is not positive, or a point count that is not a
    positive integer. Raises IndexError for a target index out of range.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
        raise ValueError(
            f"coordinates must have shape (N, 3); got {coordinates.shape!r}."
        )
    if radii.shape != (len(coordinates),):
        raise ValueError(
            f"radii must have shape ({len(coordinates)},); got {radii.shape!r}."
        )
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite.")
    if not np.isfinite(radii).all() or (radii <= 0).any():
        raise ValueError("radii must be finite and positive.")
    sphere_points = get_sphere_points(n_sphere_points)
    if len(coordinates) == 0:
        return np.empty(0, dtype=float)

    if target_indices is None:
        target_indices = np.arange(len(coordinates), dtype=int)
    else:
        target_indices = np.asarray(target_indices, dtype=int)
    if target_indices.ndim != 1:
        raise ValueError(
            "target_indices must be one-dimensional; "
            f"got shape {target_indices.shape!r}."
        )
    if len(target_indices) == 0:
        return np.empty(0, dtype=float)
    if target_indices.min() < 0 or target_indices.max() >= len(coordinates):
        raise IndexError(
            f"target_indices must lie in 0 to {len(coordinates) - 1}; got "
            f"{int(target_indices.min())} to {int(target_indices.max())}."
        )

    effective_radii = radii + SASA_PROBE_RADIUS
    max_effective_radius = float(effective_radii.max())
    target_coordinates = coordinates[target_indices]
    target_effective_radii = effective_radii[target_indices]

    # A neighbour can bury a test point only if the two centres are closer
    # than the sum of the two enlarged radii. Searching out to the target's
    # enlarged radius plus the largest one present finds every such neighbour.
    all_neighbors = cKDTree(coordinates).query_ball_point(
        target_coordinates, r=target_effective_radii + max_effective_radius
    )

    areas = np.empty(len(target_indices), dtype=float)
    for output_position, atom_index in enumerate(target_indices):
        radius = float(effective_radii[atom_index])
        sphere_area = 4.0 * np.pi * radius * radius
        neighbors = np.asarray(all_neighbors[output_position], dtype=int)
        neighbors = neighbors[neighbors != atom_index]
        if len(neighbors) == 0:
            areas[output_position] = sphere_area
            continue

        # Distances are measured from the atom's own centre. Expanding the
        # squared distance between two points far from the origin subtracts
        # large numbers that nearly cancel, and the rounding would then
        # depend on where the structure sits.
        offsets = coordinates[neighbors] - coordinates[atom_index]
        offset_squared_norms = np.einsum("ij,ij->i", offsets, offsets)
        test_points = radius * sphere_points
        test_point_squared_norms = np.einsum("ij,ij->i", test_points, test_points)
        buried = np.zeros(len(test_points), dtype=bool)
        for start in range(0, len(neighbors), _NEIGHBOR_CHUNK_SIZE):
            stop = start + _NEIGHBOR_CHUNK_SIZE
            # Expanding the squared distance avoids building an array of
            # shape (points, neighbours, 3).
            distance_squared = (
                test_point_squared_norms[:, None]
                + offset_squared_norms[None, start:stop]
                - 2.0 * test_points @ offsets[start:stop].T
            )
            buried |= (
                distance_squared
                < np.square(effective_radii[neighbors[start:stop]])[None, :]
            ).any(axis=1)
            if buried.all():
                break
        areas[output_position] = (
            sphere_area * float((~buried).sum()) / len(sphere_points)
        )
    return areas


__all__ = [
    "compute_sasa",
    "get_sphere_points",
    "resolve_sphere_point_count",
]
