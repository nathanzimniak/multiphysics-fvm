"""
File   : riemann_solvers.py
Author : Nathan ZIMNIAK
Date   : 2026-03-18
-----------------
Riemann solvers for the finite-volume solver.
Available solvers: rusanov, hll.
"""

import numpy as np


def rusanov(
    Fc_L     : dict,
    Fc_R     : dict,
    U_L      : dict,
    U_R      : dict,
    lambda_L : dict,
    lambda_R : dict
    ) -> dict:
    """
    Compute the Rusanov (local Lax-Friedrichs) numerical flux at a cell interface.

    Arguments
    ---------
    Fc_L : Left convective flux.
    Fc_R : Right convective flux.
    U_L  : Left interface state.
    U_R  : Right interface state.
    lambda_L : Left characteristic wave velocities in the lab frame.
    lambda_R : Right characteristic wave velocities in the lab frame.
    
    Returns
    -------
    Fc : Rusanov numerical flux for each conservative variable.
    """

    # Maximum absolute characteristic velocities on each left/right interface | shape (nx1+1, nx2, nx3) etc.
    max_lambda_L = np.maximum.reduce([np.abs(l) for l in lambda_L.values()])
    max_lambda_R = np.maximum.reduce([np.abs(l) for l in lambda_R.values()])

    # Maximum signal speed at the interface | shape (nx1+1, nx2, nx3) etc.
    vs = np.maximum(max_lambda_L, max_lambda_R)

    # Rusanov flux.
    Fc = {k: (Fc_L[k] + Fc_R[k])/2.0 - vs*(U_R[k] - U_L[k])/2.0 for k in Fc_L}

    return Fc


def hll(
    Fc_L     : dict,
    Fc_R     : dict,
    U_L      : dict,
    U_R      : dict,
    lambda_L : dict,
    lambda_R : dict
    ) -> dict:
    """
    Compute the HLL numerical flux at a cell interface.

    Arguments
    ---------
    Fc_L : Left convective flux.
    Fc_R : Right convective flux.
    U_L  : Left interface state.
    U_R  : Right interface state.
    lambda_L : Left characteristic wave velocities in the lab frame.
    lambda_R : Right characteristic wave velocities in the lab frame.
    
    Returns
    -------
    Fc : HLL numerical flux for each conservative variable.
    """

    # Extremal characteristic velocities on each side.
    min_lambda_L = np.minimum.reduce([l for l in lambda_L.values()])
    max_lambda_L = np.maximum.reduce([l for l in lambda_L.values()])
    min_lambda_R = np.minimum.reduce([l for l in lambda_R.values()])
    max_lambda_R = np.maximum.reduce([l for l in lambda_R.values()])

    # Minimum/Maximum signal speed at the interface | shape (nx1+1, nx2, nx3) etc.
    vs_min = np.minimum(min_lambda_L, min_lambda_R)
    vs_max = np.maximum(max_lambda_L, max_lambda_R)

    # HLL flux.
    Fc = {k: np.where(vs_min >= 0.0, Fc_L[k], np.where(vs_max <= 0.0, Fc_R[k],(vs_max*Fc_L[k]-vs_min*Fc_R[k]+vs_min*vs_max*(U_R[k]-U_L[k]))/(vs_max-vs_min))) for k in Fc_L}

    return Fc

