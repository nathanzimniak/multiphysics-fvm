"""
File   : boundary_conditions.py
Author : Nathan ZIMNIAK
Date   : 2026-03-16
-----------------
Boundary conditions for the finite-volume solver.
Available boundary conditions: neumann, periodic, reflecting.
"""

import numpy as np


def neumann(
    field : np.ndarray,
    side  : str
    ) -> None:
    """
    Apply a homogeneous Neumann boundary condition on a single domain face.

    Arguments
    ----------
    field : Field including ghost cells.
    side  : Boundary selector. One of {"left", "right", "bottom", "top", "front", "back"}.

    Returns
    -------
    None
    """

    # Zero normal gradient: ghost cell = adjacent interior cell.
    if side == "left"     : field[0, 1:-1, 1:-1]  = field[1, 1:-1, 1:-1]
    elif side == "right"  : field[-1, 1:-1, 1:-1] = field[-2, 1:-1, 1:-1]
    elif side == "bottom" : field[1:-1, 0, 1:-1]  = field[1:-1, 1, 1:-1]
    elif side == "top"    : field[1:-1, -1, 1:-1] = field[1:-1, -2, 1:-1]
    elif side == "front"  : field[1:-1, 1:-1, 0]  = field[1:-1, 1:-1, 1]
    elif side == "back"   : field[1:-1, 1:-1, -1] = field[1:-1, 1:-1, -2]
    else: raise ValueError(f"Unknown boundary side: '{side}'. Expected one of: left, right, bottom, top, front, back.")


def periodic(
    field : np.ndarray,
    side  : str
    ) -> None:
    """
    Apply a periodic boundary condition on a single domain face.

    Arguments
    ----------
    field : Field including ghost cells.
    side  : Boundary selector. One of {"left", "right", "bottom", "top", "front", "back"}.

    Returns
    -------
    None
    """

    # Periodicity: ghost cell = opposite interior cell.
    if side == "left"     : field[0, 1:-1, 1:-1]  = field[-2, 1:-1, 1:-1]
    elif side == "right"  : field[-1, 1:-1, 1:-1] = field[1, 1:-1, 1:-1]
    elif side == "bottom" : field[1:-1, 0, 1:-1]  = field[1:-1, -2, 1:-1]
    elif side == "top"    : field[1:-1, -1, 1:-1] = field[1:-1, 1, 1:-1]
    elif side == "front"  : field[1:-1, 1:-1, 0]  = field[1:-1, 1:-1, -2]
    elif side == "back"   : field[1:-1, 1:-1, -1] = field[1:-1, 1:-1, 1]
    else: raise ValueError(f"Unknown boundary side: '{side}'. Expected one of: left, right, bottom, top, front, back.")


def reflecting(
    field : np.ndarray,
    side  : str
    ) -> None:
    """
    Apply a reflecting boundary condition on a single domain face.

    Arguments
    ----------
    field : Field including ghost cells.
    side  : Boundary selector. One of {"left", "right", "bottom", "top", "front", "back"}.

    Returns
    -------
    None
    """

    # Odd symmetry with respect to the boundary face: ghost cell = opposite of adjacent interior cell.
    if side == "left"     : field[0, 1:-1, 1:-1]  = -field[1, 1:-1, 1:-1]
    elif side == "right"  : field[-1, 1:-1, 1:-1] = -field[-2, 1:-1, 1:-1]
    elif side == "bottom" : field[1:-1, 0, 1:-1]  = -field[1:-1, 1, 1:-1]
    elif side == "top"    : field[1:-1, -1, 1:-1] = -field[1:-1, -2, 1:-1]
    elif side == "front"  : field[1:-1, 1:-1, 0]  = -field[1:-1, 1:-1, 1]
    elif side == "back"   : field[1:-1, 1:-1, -1] = -field[1:-1, 1:-1, -2]
    else: raise ValueError(f"Unknown boundary side: '{side}'. Expected one of: left, right, bottom, top, front, back.")
