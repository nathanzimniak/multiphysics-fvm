"""
File   : time_integrators.py
Author : Nathan ZIMNIAK
Date   : 2026-03-19
-----------------
Time integrators for the finite-volume solver.
Available integrators: euler_explicit, rk3, rk3_ssp, rk4.
"""

def euler_explicit(
    rhs           : callable,
    U             : dict,
    dt            : float,
    grid          : dict,
    space_schemes : dict,
    params        : dict,
    bcs           : dict,
    apply_bcs     : callable
    ) -> None:
    """
    Advance the simulation state by one explicit Euler time step.

    Arguments
    ----------
    rhs           : Right-hand side of the governing equations.
    U             : Conserved variable arrays (with ghost cells).
    dt            : Time-step size.
    grid          : Discrete grid quantities, coordinates, and domain config.
    space_schemes : Spatial discretization schemes.
    params        : Physical parameters.
    bcs           : Boundary condition configuration.
    apply_bcs     : Boundary condition application function.

    Returns
    -------
    None
    """

    # Physical domain (no ghosts).
    ing = (slice(1,-1), slice(1,-1), slice(1,-1))
    U_phys = {k: U[k][ing] for k in U}

    # Forward-Euler update.
    apply_bcs(U, bcs)
    k1 = rhs(U, grid, params, space_schemes)
    for k in U_phys:
        U_phys[k][:] += dt*k1[k]


def rk3(
    rhs           : callable,
    U             : dict,
    dt            : float,
    grid          : dict,
    space_schemes : dict,
    params        : dict,
    bcs           : dict,
    apply_bcs     : callable
    ) -> None:
    """
    Advance the simulation state by one explicit RK3 time step.

    Arguments
    ----------
    rhs           : Right-hand side of the governing equations.
    U             : Conserved variable arrays (with ghost cells).
    dt            : Time-step size.
    grid          : Discrete grid quantities, coordinates, and domain config.
    space_schemes : Spatial discretization schemes.
    params        : Physical parameters.
    bcs           : Boundary condition configuration.
    apply_bcs     : Boundary condition application function.

    Returns
    -------
    None
    """


    # Physical domain (no ghosts).
    ing    = (slice(1,-1), slice(1,-1), slice(1,-1))
    U_phys = {k: U[k][ing] for k in U}

    # Save base state.
    U0 = {k: U_phys[k].copy() for k in U_phys}

    # Stage 1: slope at the beginning of the interval.
    apply_bcs(U, bcs)
    k1 = rhs(U, grid, params, space_schemes)
    for k in U_phys:
        U_phys[k][:] = U0[k] + 0.5*dt*k1[k]

    # Stage 2: slope at the midpoint, using k1.
    apply_bcs(U, bcs)
    k2 = rhs(U, grid, params, space_schemes)
    for k in U_phys:
        U_phys[k][:] = U0[k] + dt*k2[k]

    # Stage 3: slope at the end of the interval, using k2.
    apply_bcs(U, bcs)
    k3 = rhs(U, grid, params, space_schemes)
    for k in U_phys:
        U_phys[k][:] = U0[k] + (dt/6.0)*(k1[k] + 4.0*k2[k] + k3[k])


def rk3_ssp(
    rhs           : callable,
    U             : dict,
    dt            : float,
    grid          : dict,
    space_schemes : dict,
    params        : dict,
    bcs           : dict,
    apply_bcs     : callable
    ) -> None:
    """
    Advance the simulation state by one explicit SSPRK3 time step.

    Arguments
    ----------
    rhs           : Right-hand side of the governing equations.
    U             : Conserved variable arrays (with ghost cells).
    dt            : Time-step size.
    grid          : Discrete grid quantities, coordinates, and domain config.
    space_schemes : Spatial discretization schemes.
    params        : Physical parameters.
    bcs           : Boundary condition configuration.
    apply_bcs     : Boundary condition application function.

    Returns
    -------
    None
    """


    # Physical domain (no ghosts).
    ing    = (slice(1,-1), slice(1,-1), slice(1,-1))
    U_phys = {k: U[k][ing] for k in U}

    # Save base state.
    U0 = {k: U_phys[k].copy() for k in U_phys}

    # Stage 1: Forward-Euler step from the base state.
    apply_bcs(U, bcs)
    k1 = rhs(U, grid, params, space_schemes)
    for k in U_phys:
        U_phys[k][:] = U0[k] + dt*k1[k]

    # Stage 2: Combination of the base state and a Forward-Euler step from stage 1.
    apply_bcs(U, bcs)
    k2 = rhs(U, grid, params, space_schemes)
    for k in U_phys:
        U_phys[k][:] = 0.75*U0[k] + 0.25*(U_phys[k] + dt*k2[k])

    # Stage 3: Combination of the base state and a Forward-Euler step from stage 2.
    apply_bcs(U, bcs)
    k3 = rhs(U, grid, params, space_schemes)
    for k in U_phys:
        U_phys[k][:] = (1.0/3.0)*U0[k] + (2.0/3.0)*(U_phys[k] + dt*k3[k])


def rk4(
    rhs           : callable,
    U             : dict,
    dt            : float,
    grid          : dict,
    space_schemes : dict,
    params        : dict,
    bcs           : dict,
    apply_bcs     : callable
    ) -> None:
    """
    Advance the simulation state by one explicit RK4 time step.

    Arguments
    ----------
    rhs           : Right-hand side of the governing equations.
    U             : Conserved variable arrays (with ghost cells).
    dt            : Time-step size.
    grid          : Discrete grid quantities, coordinates, and domain config.
    space_schemes : Spatial discretization schemes.
    params        : Physical parameters.
    bcs           : Boundary condition configuration.
    apply_bcs     : Boundary condition application function.

    Returns
    -------
    None
    """


    # Physical domain (no ghosts).
    ing    = (slice(1,-1), slice(1,-1), slice(1,-1))
    U_phys = {k: U[k][ing] for k in U}

    # Save base state.
    U0 = {k: U_phys[k].copy() for k in U_phys}

    # Stage 1: slope at the beginning of the interval.
    apply_bcs(U, bcs)
    k1 = rhs(U, grid, params, space_schemes)
    for k in U_phys:
        U_phys[k][:] = U0[k] + 0.5*dt*k1[k]

    # Stage 2: slope at the midpoint, using k1.
    apply_bcs(U, bcs)
    k2 = rhs(U, grid, params, space_schemes)
    for k in U_phys:
        U_phys[k][:] = U0[k] + 0.5*dt*k2[k]

    # Stage 3: slope at the midpoint, using k2.
    apply_bcs(U, bcs)
    k3 = rhs(U, grid, params, space_schemes)
    for k in U_phys:
        U_phys[k][:] = U0[k] + dt*k3[k]

    # Stage 4: slope at the end of the interval, using k3.
    apply_bcs(U, bcs)
    k4 = rhs(U, grid, params, space_schemes)
    for k in U_phys:
        U_phys[k][:] = U0[k] + (dt/6.0)*(k1[k] + 2.0*k2[k] + 2.0*k3[k] + k4[k])
