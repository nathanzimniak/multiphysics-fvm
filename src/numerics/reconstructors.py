"""
File   : reconstructors.py
Author : Nathan ZIMNIAK
Date   : 2026-03-14
-----------------
Interface state reconstructors for the finite-volume solver.
Available reconstructors: piecewise_constant, muscl.
"""

import numpy as np


def piecewise_constant(
    fields        : dict,
    L             : dict,
    slope_limiter : callable
    ) -> tuple:
    """
    Construct left and right interface values using piecewise-constant reconstruction (first-order Godunov).

    Arguments
    ---------
    fields        : Dictionary of cell-centered arrays (with ghost cells).
    grid_geometry : Cell geometry arrays (unused, kept for API consistency with other reconstructors).
    slope_limiter : Slope limiter function (unused, kept for API consistency with other reconstructors).

    Returns
    -------
    fields_L : Dictionary of left interface values with keys "x1", "x2", "x3".
    fields_R : Dictionary of right interface values with keys "x1", "x2", "x3".
    """

    # Indices for cell interface indexing: ipf = i-1/2, inf = i+1/2.
    # For xn-oriented faces (with n=1,2,3):
    # - In the longitudinal direction (xn), left cells include all cells except the last one (None,-1), right cells include all cells except the first one (1,None).
    # - In the transverse direction (xm, with m≠n) ghost cells are excluded (1,-1).
    ipf = (slice(None, -1), slice(1, -1), slice(1, -1))
    inf = (slice(1, None),  slice(1, -1), slice(1, -1))
    jpf = (slice(1, -1), slice(None, -1), slice(1, -1))
    jnf = (slice(1, -1), slice(1, None),  slice(1, -1))
    kpf = (slice(1, -1), slice(1, -1), slice(None, -1))
    knf = (slice(1, -1), slice(1, -1), slice(1, None))

    # Left/Right values at x1-oriented faces, shape (nx1+1, nx2, nx3).
    fields_L_x1 = {k: fields[k][ipf] for k in fields}
    fields_R_x1 = {k: fields[k][inf] for k in fields}

    # Left/Right values at x2-oriented faces, shape (nx1, nx2+1, nx3).
    fields_L_x2 = {k: fields[k][jpf] for k in fields}
    fields_R_x2 = {k: fields[k][jnf] for k in fields}

    # Left/Right values at x3-oriented faces, shape (nx1, nx2, nx3+1).
    fields_L_x3 = {k: fields[k][kpf] for k in fields}
    fields_R_x3 = {k: fields[k][knf] for k in fields}

    fields_L = {"x1" : fields_L_x1,
                "x2" : fields_L_x2,
                "x3" : fields_L_x3}

    fields_R = {"x1" : fields_R_x1,
                "x2" : fields_R_x2,
                "x3" : fields_R_x3}

    return fields_L, fields_R


def muscl(
    fields        : dict,
    L             : dict,
    slope_limiter : callable
    ) -> tuple:
    """
    Construct left and right interface values using MUSCL reconstruction.

    Arguments
    ---------
    fields        : Dictionary of cell-centered arrays (with ghost cells).
    grid_geometry : Cell geometry arrays.
    slope_limiter : Slope limiter function.

    Returns
    -------
    fields_L : Dictionary of left interface values with keys "x1", "x2", "x3".
    fields_R : Dictionary of right interface values with keys "x1", "x2", "x3".
    """

    # Unpack inputs.
    Lx1 = L["x1"]
    Lx2 = L["x2"]
    Lx3 = L["x3"]

    # Indices for cell indexing: ipc = i-1, inc = i+1.
    # For xn-oriented faces (with n=1,2,3):
    # - In the longitudinal direction (xn), previous cells include all cells except the two last ones (None,-2), next cells include all cells except the two first ones (2,None).
    # - In the transverse direction (xm, with m≠n) ghost cells are excluded (1,-1).
    ic  = (slice(1, -1),  slice(1, -1),  slice(1, -1))
    ipc = (slice(None,-2), slice(1, -1),  slice(1, -1))
    inc = (slice(2, None), slice(1, -1),  slice(1, -1))
    jpc = (slice(1, -1),  slice(None,-2), slice(1, -1))
    jnc = (slice(1, -1),  slice(2, None), slice(1, -1))
    kpc = (slice(1, -1),  slice(1, -1),  slice(None,-2))
    knc = (slice(1, -1),  slice(1, -1),  slice(2, None))

    # Indices for cell interface indexing: ipf = i-1/2, inf = i+1/2.
    # For xn-oriented faces (with n=1,2,3):
    # - In the longitudinal direction (xn), left cells include all cells except the last one (None,-1), right cells include all cells except the first one (1,None).
    # - In the transverse direction (xm, with m≠n) ghost cells are excluded (1,-1).
    ipf = (slice(None, -1), slice(1, -1),  slice(1, -1))
    inf = (slice(1, None),  slice(1, -1),  slice(1, -1))
    jpf = (slice(1, -1),  slice(None, -1), slice(1, -1))
    jnf = (slice(1, -1),  slice(1, None),  slice(1, -1))
    kpf = (slice(1, -1),  slice(1, -1),  slice(None, -1))
    knf = (slice(1, -1),  slice(1, -1),  slice(1, None))

    # Backward and forward slopes, and limited slopes for each field along each direction.
    slopes_x1 = {}
    slopes_x2 = {}
    slopes_x3 = {}
    for k, f in fields.items():
        back_x1  = (f[ic] - f[ipc])/Lx1
        front_x1 = (f[inc] - f[ic])/Lx1
        back_x2  = (f[ic] - f[jpc])/Lx2
        front_x2 = (f[jnc] - f[ic])/Lx2
        back_x3  = (f[ic] - f[kpc])/Lx3
        front_x3 = (f[knc] - f[ic])/Lx3

        slope_g_x1 = np.zeros_like(f)
        slope_g_x2 = np.zeros_like(f)
        slope_g_x3 = np.zeros_like(f)
        slope_g_x1[ic] = slope_limiter(back_x1, front_x1)*Lx1
        slope_g_x2[ic] = slope_limiter(back_x2, front_x2)*Lx2
        slope_g_x3[ic] = slope_limiter(back_x3, front_x3)*Lx3

        slopes_x1[k] = slope_g_x1
        slopes_x2[k] = slope_g_x2
        slopes_x3[k] = slope_g_x3

    # Left/Right values at x1-oriented faces, shape (nx1+1, nx2, nx3).
    fields_L_x1 = {k: fields[k][ipf] + slopes_x1[k][ipf]/2.0 for k in fields}
    fields_R_x1 = {k: fields[k][inf] - slopes_x1[k][inf]/2.0 for k in fields}

    # Left/Right values at x2-oriented faces, shape (nx1, nx2+1, nx3).
    fields_L_x2 = {k: fields[k][jpf] + slopes_x2[k][jpf]/2.0 for k in fields}
    fields_R_x2 = {k: fields[k][jnf] - slopes_x2[k][jnf]/2.0 for k in fields}

    # Left/Right values at x3-oriented faces, shape (nx1, nx2, nx3+1).
    fields_L_x3 = {k: fields[k][kpf] + slopes_x3[k][kpf]/2.0 for k in fields}
    fields_R_x3 = {k: fields[k][knf] - slopes_x3[k][knf]/2.0 for k in fields}

    fields_L = {"x1" : fields_L_x1,
                "x2" : fields_L_x2,
                "x3" : fields_L_x3}

    fields_R = {"x1" : fields_R_x1,
                "x2" : fields_R_x2,
                "x3" : fields_R_x3}

    return fields_L, fields_R

