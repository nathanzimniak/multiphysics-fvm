"""
File   : diffusive_solvers.py
Author : Nathan ZIMNIAK
Date   : 2026-03-12
-----------------
Diffusion solvers for the finite-volume solver.
Available solvers: centered.
"""


def centered(
    Fd_l : dict,
    Fd_r : dict
    ) -> dict:
    """
    Compute the centered numerical flux at a cell interface.

    Arguments
    ---------
    Fd_l : Left diffusive flux.
    Fd_r : Right diffusive flux.

    Returns
    -------
    Fd : Centered numerical flux for each conservative variable.
    """

    Fd = {k: (Fd_l[k] + Fd_r[k])/2.0 for k in Fd_l}

    return Fd
