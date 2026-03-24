"""
File   : geometry.py
Author : Nathan ZIMNIAK
Date   : 2026-03-10
-----------------
Grid geometry for the finite-volume solver.
Computes cell volumes, face surfaces, cell lengths, and Lamé coefficients
for cartesian, cylindrical, and spherical coordinate systems.
"""

import numpy as np


def build_grid(
    coordinate_system : str,
    grid_config       : dict
    ) -> dict:
    """
    Build the discrete grid for the finite-volume solver.

    Arguments
    ---------
    coordinate_system : Coordinate system. One of {"cartesian", "cylindrical", "spherical"}.
    grid_config       : Grid configuration (x1_min, x1_max, nx1, ...).

    Returns
    -------
    grid : Discrete grid quantities: coordinates, cell geometry, and Lamé coefficients.
    """

    # Unpack grid configuration.
    x1_min, x1_max, nx1 = grid_config["x1_min"], grid_config["x1_max"], grid_config["nx1"]
    x2_min, x2_max, nx2 = grid_config["x2_min"], grid_config["x2_max"], grid_config["nx2"]
    x3_min, x3_max, nx3 = grid_config["x3_min"], grid_config["x3_max"], grid_config["nx3"]

    # Coordinate spacings.
    dx1 = (x1_max-x1_min)/nx1
    dx2 = (x2_max-x2_min)/nx2
    dx3 = (x3_max-x3_min)/nx3

    # 1D arrays of cell-center coordinates along each direction (without ghost cells).
    x1_1d = x1_min + (np.arange(nx1)+0.5)*dx1
    x2_1d = x2_min + (np.arange(nx2)+0.5)*dx2
    x3_1d = x3_min + (np.arange(nx3)+0.5)*dx3

    # 1D arrays of face coordinates along each direction (without ghost cells).
    x1f_1d = x1_min + np.arange(nx1+1)*dx1
    x2f_1d = x2_min + np.arange(nx2+1)*dx2
    x3f_1d = x3_min + np.arange(nx3+1)*dx3

    # 3D arrays of cell-center coordinates along each direction (without ghost cells).
    x1, x2, x3 = np.meshgrid(x1_1d, x2_1d, x3_1d, indexing="ij")

    # Slices for face indexing: i_m = i-1/2, i_p = i+1/2.
    ip = (slice(1, None), slice(None),  slice(None) )  # [1:, :, :]
    im = (slice(None,-1), slice(None),  slice(None) )  # [:-1, :, :]
    jp = (slice(None),  slice(1, None), slice(None) )  # [:, 1:, :]
    jm = (slice(None),  slice(None,-1), slice(None) )  # [:, :-1, :]
    kp = (slice(None),  slice(None),  slice(1, None))  # [:, :, 1:]
    km = (slice(None),  slice(None),  slice(None,-1))  # [:, :, :-1]

    # Slices for broadcasting 1D face arrays to 3D.
    si = (slice(None), np.newaxis,  np.newaxis )  # [:, np.newaxis, np.newaxis]
    sj = (np.newaxis,  slice(None), np.newaxis )  # [np.newaxis, :, np.newaxis]
    sk = (np.newaxis,  np.newaxis,  slice(None))  # [np.newaxis, np.newaxis, :]

    # Shape without ghost cells.
    sng = (nx1, nx2, nx3)

    # Faces shapes.
    sx1f = (nx1+1, nx2, nx3)
    sx2f = (nx1, nx2+1, nx3)
    sx3f = (nx1, nx2, nx3+1)

    # 3D face coordinate arrays, each varying only along its own direction.
    x1f = x1f_1d[si]*np.ones(sx1f)
    x2f = x2f_1d[sj]*np.ones(sx2f)
    x3f = x3f_1d[sk]*np.ones(sx3f)

    if coordinate_system == "cartesian":

        # Cell volume.
        V = (x1f[ip]-x1f[im]) * (x2f[jp]-x2f[jm]) * (x3f[kp]-x3f[km])

        # Face areas.
        Sx1 = np.full(sx1f,1.0) * (x2f_1d[sj][jp]-x2f_1d[sj][jm]) * (x3f_1d[sk][kp]-x3f_1d[sk][km])
        Sx2 = (x1f_1d[si][ip]-x1f_1d[si][im]) * np.full(sx2f,1.0) * (x3f_1d[sk][kp]-x3f_1d[sk][km])
        Sx3 = (x1f_1d[si][ip]-x1f_1d[si][im]) * (x2f_1d[sj][jp]-x2f_1d[sj][jm]) * np.full(sx3f,1.0)

        # Physical cell lengths along each direction.
        Lx1 = x1f[ip]-x1f[im]
        Lx2 = x2f[jp]-x2f[jm]
        Lx3 = x3f[kp]-x3f[km]

        # Lamé coefficients.
        hx1 = np.full(sng, 1.0)
        hx2 = np.full(sng, 1.0)
        hx3 = np.full(sng, 1.0)

        # Logarithmic derivatives of Lamé coefficients.
        dx2_ln_hx1       = np.full(sng, 0.0)
        dx3_ln_hx1       = np.full(sng, 0.0)
        dx1_ln_hx2       = np.full(sng, 0.0)
        dx3_ln_hx2       = np.full(sng, 0.0)
        dx1_ln_hx3       = np.full(sng, 0.0)
        dx2_ln_hx3       = np.full(sng, 0.0)
        dx1_ln_hx1hx2hx3 = np.full(sng, 0.0)
        dx2_ln_hx1hx2hx3 = np.full(sng, 0.0)
        dx3_ln_hx1hx2hx3 = np.full(sng, 0.0)

    elif coordinate_system == "cylindrical":
        
        # Cell volume.
        V = ((x1f[ip]**2-x1f[im]**2)/2.0) * (x2f[jp]-x2f[jm]) * (x3f[kp]-x3f[km])

        # Face areas.
        Sx1 = (x1f_1d[si]) * (x2f_1d[sj][jp]-x2f_1d[sj][jm]) * (x3f_1d[sk][kp]-x3f_1d[sk][km])
        Sx2 = (x1f_1d[si][ip]-x1f_1d[si][im]) * np.full(sx2f,1.0) * (x3f_1d[sk][kp]-x3f_1d[sk][km])
        Sx3 = ((x1f_1d[si][ip]**2-x1f_1d[si][im]**2)/2.0) * (x2f_1d[sj][jp]-x2f_1d[sj][jm]) * np.full(sx3f,1.0)

        # Physical cell lengths along each direction.
        Lx1 = x1f[ip]-x1f[im]
        Lx2 = x1*(x2f[jp]-x2f[jm])
        Lx3 = x3f[kp]-x3f[km]

        # Lamé coefficients.
        hx1 = np.full(sng, 1.0)
        hx2 = x1
        hx3 = np.full(sng, 1.0)

        # Logarithmic derivatives of Lamé coefficients.
        dx2_ln_hx1       = np.full(sng, 0.0)
        dx3_ln_hx1       = np.full(sng, 0.0)
        dx1_ln_hx2       = 1.0/x1
        dx3_ln_hx2       = np.full(sng, 0.0)
        dx1_ln_hx3       = np.full(sng, 0.0)
        dx2_ln_hx3       = np.full(sng, 0.0)
        dx1_ln_hx1hx2hx3 = 1.0/x1
        dx2_ln_hx1hx2hx3 = np.full(sng, 0.0)
        dx3_ln_hx1hx2hx3 = np.full(sng, 0.0)

    elif coordinate_system == "spherical":

        # Cell volume.
        V = ((x1f[ip]**3-x1f[im]**3)/3.0) * (np.cos(x2f[jm])-np.cos(x2f[jp])) * (x3f[kp]-x3f[km])

        # Face areas.
        Sx1 = (x1f_1d[si]**2) * (np.cos(x2f_1d[sj][jm])-np.cos(x2f_1d[sj][jp])) * (x3f_1d[sk][kp]-x3f_1d[sk][km])
        Sx2 = ((x1f_1d[si][ip]**2-x1f_1d[si][im]**2)/2.0) * (np.sin(x2f_1d[sj])) * (x3f_1d[sk][kp]-x3f_1d[sk][km])
        Sx3 = ((x1f_1d[si][ip]**2-x1f_1d[si][im]**2)/2.0) * (x2f_1d[sj][jp]-x2f_1d[sj][jm]) * np.full(sx3f,1.0)

        # Physical cell lengths along each direction.
        Lx1 = x1f[ip] - x1f[im]
        Lx2 = x1*(x2f[jp]-x2f[jm])
        Lx3 = x1*np.sin(x2)*(x3f[kp]-x3f[km])

        # Lamé coefficients.
        hx1 = np.full(sng, 1.0)
        hx2 = x1
        hx3 = x1*np.sin(x2)

        # Logarithmic derivatives of Lamé coefficients.
        dx2_ln_hx1       = np.full(sng, 0.0)
        dx3_ln_hx1       = np.full(sng, 0.0)
        dx1_ln_hx2       = 1.0/x1
        dx3_ln_hx2       = np.full(sng, 0.0)
        dx1_ln_hx3       = 1.0/x1
        dx2_ln_hx3       = 1.0/np.tan(x2)
        dx1_ln_hx1hx2hx3 = 2.0/x1
        dx2_ln_hx1hx2hx3 = 1.0/np.tan(x2)
        dx3_ln_hx1hx2hx3 = np.full(sng, 0.0)

    else: raise ValueError(f"Unknown coordinate system: '{coordinate_system}'. Expected one of: cartesian, cylindrical, spherical.")

    # Unit normal vectors pointing in the x1, x2, and x3 directions.
    nx1 = np.array([1.0, 0.0, 0.0])
    nx2 = np.array([0.0, 1.0, 0.0])
    nx3 = np.array([0.0, 0.0, 1.0])

    # Assemble and return grid dictionary.
    grid = {"config" : grid_config,

            "cell"   : {"coordinates": {"x1": x1,  "x2": x2,  "x3": x3},
                        "normal"     : {"x1": nx1, "x2": nx2, "x3": nx3},
                        "geometry"   : {"volume" : V,
                                        "surface": {"x1": Sx1, "x2": Sx2, "x3": Sx3},
                                        "length" : {"x1": Lx1, "x2": Lx2, "x3": Lx3}}},

            "lame"   : {"coefficients": {"x1": hx1, "x2": hx2, "x3": hx3}, 
                        "derivatives" : {"x1": {"ln_hx1hx2hx3": dx1_ln_hx1hx2hx3, "ln_hx2": dx1_ln_hx2, "ln_hx3": dx1_ln_hx3},
                                         "x2": {"ln_hx1hx2hx3": dx2_ln_hx1hx2hx3, "ln_hx1": dx2_ln_hx1, "ln_hx3": dx2_ln_hx3},
                                         "x3": {"ln_hx1hx2hx3": dx3_ln_hx1hx2hx3, "ln_hx1": dx3_ln_hx1, "ln_hx2": dx3_ln_hx2}}}}

    return grid
