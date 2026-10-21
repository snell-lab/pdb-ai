"""The map coefficients, the sampling, and the vocabularies of the density engine.

Every number the engine uses is here beside its basis. The sampling is written
into the option's summary text, so every run records it.
"""

from __future__ import annotations

# The columns of PDB-Redo's reflection file that hold the map coefficients.
# Refmac writes them, and they are the ones Coot and EDSTATS read.
MAP_2FOFC_LABELS = ("FWT", "PHWT")
MAP_FOFC_LABELS = ("DELFWT", "PHDELWT")

# The columns of the same file that hold PDB-Redo's calculated structure
# factors (DFc), from its whole model with bulk solvent. Their map is on the
# same scale as the two above, so it gives the density the model places at an
# atom, and the 2mFo−DFc map minus this map gives the difference density on
# that scale. DELFWT is not used for it, because Refmac halves DELFWT in a
# twinned refinement.
MAP_FC_LABELS = ("FC_ALL", "PHIC_ALL")

# The items of the two wwPDB validation reflection files that hold the map
# coefficients, calculated about the deposited model. Those files hold no
# calculated structure factors, so no calculated map is read from them.
WWPDB_2FOFC_LABELS = ("pdbx_FWT", "pdbx_PHWT")
WWPDB_FOFC_LABELS = ("pdbx_DELFWT", "pdbx_DELPHWT")

# How finely a map is sampled, as the number of grid points per high-resolution
# limit. A rate of 4 puts grid points a quarter of the resolution apart, and
# with cubic interpolation the value read at an atom is then close to the
# value a much finer map gives. Linear interpolation at the usual rate of 3
# flattens every peak, so the engine uses neither.
MAP_SAMPLE_RATE = 4.0

# The lowest rate a large map may be sampled at, the coarsest at which the
# value read at an atom stays close to the value of a finer map.
MAP_MIN_SAMPLE_RATE = 2.5

# The most grid points one map may hold. A map of this size takes 0.6 GB as
# single-precision numbers, and the engine holds one map at a time. A unit
# cell that needs more at MAP_SAMPLE_RATE is sampled more coarsely, down to
# MAP_MIN_SAMPLE_RATE, and one that still needs more gets no values.
MAP_MAX_GRID_POINTS = 150_000_000

# The largest relative difference in a cell length, and the largest difference
# in a cell angle in degrees, between the reflection file and the deposited
# model at which the two still describe the same crystal. PDB-Redo keeps the
# deposited cell, so a larger difference means a different setting or
# indexing, and a deposited position would then be read at the wrong place.
CELL_LENGTH_TOLERANCE = 0.01
CELL_ANGLE_TOLERANCE_DEG = 1.0

# The status of a structure's maps:
#   ok                   the maps were read.
#   no_pdb_redo_entry    PDB-Redo has no reflection file for the entry.
#   no_validation_maps   the wwPDB archive has no validation map coefficients
#                        for the entry.
#   no_map_coefficients  the file lacks the coefficients or any usable
#                        reflection.
#   map_too_large        the unit cell exceeds the size limit even at the
#                        lowest sample rate.
#   cell_mismatch        the reflection file's cell is not the deposited one.
#   models_not_paired    no atom of PDB-Redo's model paired with a deposited
#                        atom, so nothing shows the two share a frame.
#   failed               the engine raised.
MAP_STATUS_OK = "ok"
MAP_STATUS_NO_ENTRY = "no_pdb_redo_entry"
MAP_STATUS_NO_VALIDATION_MAPS = "no_validation_maps"
MAP_STATUS_NO_COEFFICIENTS = "no_map_coefficients"
MAP_STATUS_TOO_LARGE = "map_too_large"
MAP_STATUS_CELL_MISMATCH = "cell_mismatch"
MAP_STATUS_NOT_PAIRED = "models_not_paired"
MAP_STATUS_FAILED = "failed"
MAP_STATUSES = (
    MAP_STATUS_OK,
    MAP_STATUS_NO_ENTRY,
    MAP_STATUS_NO_VALIDATION_MAPS,
    MAP_STATUS_NO_COEFFICIENTS,
    MAP_STATUS_TOO_LARGE,
    MAP_STATUS_CELL_MISMATCH,
    MAP_STATUS_NOT_PAIRED,
    MAP_STATUS_FAILED,
)

# The status of one atom's values. An atom of a structure without maps takes
# the structure's status. With maps, it is ok when both positions were read,
# no_pdb_redo_atom when PDB-Redo's model has no partner for the atom, so only
# the deposited position was read, and no_coordinates when the deposited
# position is not a finite point, so nothing was read.
ATOM_STATUS_OK = "ok"
ATOM_STATUS_NO_PDB_REDO_ATOM = "no_pdb_redo_atom"
ATOM_STATUS_NO_COORDINATES = "no_coordinates"

__all__ = [
    "ATOM_STATUS_NO_COORDINATES",
    "ATOM_STATUS_NO_PDB_REDO_ATOM",
    "ATOM_STATUS_OK",
    "CELL_ANGLE_TOLERANCE_DEG",
    "CELL_LENGTH_TOLERANCE",
    "MAP_2FOFC_LABELS",
    "MAP_FC_LABELS",
    "MAP_FOFC_LABELS",
    "MAP_MAX_GRID_POINTS",
    "MAP_MIN_SAMPLE_RATE",
    "MAP_SAMPLE_RATE",
    "MAP_STATUSES",
    "MAP_STATUS_CELL_MISMATCH",
    "MAP_STATUS_FAILED",
    "MAP_STATUS_NOT_PAIRED",
    "MAP_STATUS_NO_COEFFICIENTS",
    "MAP_STATUS_NO_ENTRY",
    "MAP_STATUS_NO_VALIDATION_MAPS",
    "MAP_STATUS_OK",
    "MAP_STATUS_TOO_LARGE",
    "WWPDB_2FOFC_LABELS",
    "WWPDB_FOFC_LABELS",
]
