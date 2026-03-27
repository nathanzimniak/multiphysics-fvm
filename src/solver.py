"""
File   : solver.py
Author : Nathan ZIMNIAK
Date   : 2026-03-21
-----------------
Orchestrates the simulation: state initialization, initial/boundary
conditions, and the main time loop.
"""

from src.grid import geometry


def apply_bcs(
    U   : dict,
    bcs : dict
    ) -> None:
    """
    Apply boundary conditions to all conserved variables.

    Arguments
    ---------
    U   : Conserved variable arrays (with ghost cells).
    bcs : Boundary condition configuration per variable and side.

    Returns
    -------
    None
    """

    for var_name in bcs:

        field = U[var_name]
        bc_by_side = bcs[var_name]

        # Apply boundary conditions on the six faces of the domain.
        for side in ("left", "right", "bottom", "top", "front", "back"):
            bc_by_side[side](field, side)

        # Fill ghost-cell edges by averaging the two adjacent ghost-cell face values.
        field[0, 0, 1:-1]   = 0.5*(field[1, 0, 1:-1]   + field[0, 1, 1:-1])
        field[0, -1, 1:-1]  = 0.5*(field[1, -1, 1:-1]  + field[0, -2, 1:-1])
        field[-1, 0, 1:-1]  = 0.5*(field[-2, 0, 1:-1]  + field[-1, 1, 1:-1])
        field[-1, -1, 1:-1] = 0.5*(field[-2, -1, 1:-1] + field[-1, -2, 1:-1])
        field[0, 1:-1, 0]   = 0.5*(field[1, 1:-1, 0]   + field[0, 1:-1, 1])
        field[0, 1:-1, -1]  = 0.5*(field[1, 1:-1, -1]  + field[0, 1:-1, -2])
        field[-1, 1:-1, 0]  = 0.5*(field[-2, 1:-1, 0]  + field[-1, 1:-1, 1])
        field[-1, 1:-1, -1] = 0.5*(field[-2, 1:-1, -1] + field[-1, 1:-1, -2])
        field[1:-1, 0, 0]   = 0.5*(field[1:-1, 1, 0]   + field[1:-1, 0, 1])
        field[1:-1, 0, -1]  = 0.5*(field[1:-1, 1, -1]  + field[1:-1, 0, -2])
        field[1:-1, -1, 0]  = 0.5*(field[1:-1, -2, 0]  + field[1:-1, -1, 1])
        field[1:-1, -1, -1] = 0.5*(field[1:-1, -2, -1] + field[1:-1, -1, -2])

        # Fill ghost-cell corners by averaging the three adjacent ghost-cell face values meeting at each corner.
        field[0,0,0]    = (field[1,0,0]    + field[0,1,0]    + field[0,0,1])    / 3.0
        field[0,0,-1]   = (field[1,0,-1]   + field[0,1,-1]   + field[0,0,-2])   / 3.0
        field[0,-1,0]   = (field[1,-1,0]   + field[0,-2,0]   + field[0,-1,1])   / 3.0
        field[0,-1,-1]  = (field[1,-1,-1]  + field[0,-2,-1]  + field[0,-1,-2])  / 3.0
        field[-1,0,0]   = (field[-2,0,0]   + field[-1,1,0]   + field[-1,0,1])   / 3.0
        field[-1,0,-1]  = (field[-2,0,-1]  + field[-1,1,-1]  + field[-1,0,-2])  / 3.0
        field[-1,-1,0]  = (field[-2,-1,0]  + field[-1,-2,0]  + field[-1,-1,1])  / 3.0
        field[-1,-1,-1] = (field[-2,-1,-1] + field[-1,-2,-1] + field[-1,-1,-2]) / 3.0


def apply_ics(
    U      : dict,
    params : dict,
    grid   : dict,
    ics    : dict
    ) -> None:
    """
    Apply the configured initial condition function to all state variables.
    
    Arguments
    ---------
    U      : Conserved variable arrays (with ghost cells).
    params : Physical parameters.
    grid   : Discrete grid quantities, coordinates, and domain config.
    ics    : Initial condition configuration (initializer + parameters).
    
    Returns
    -------
    None
    """

    ics["initializer"](U, params, grid, ics["parameters"])


def run(
    setup : dict
    ) -> None:
    """
    Run the simulation: initialize, apply ICs/BCs, and advance in time.

    Arguments
    ---------
    setup : Full setup configuration.

    Returns
    -------
    None
    """

    # Unpack setup.
    physics           = setup["physics"]
    params            = setup["physical_parameters"]
    t_start           = setup["time_config"]["t_start"]
    t_end             = setup["time_config"]["t_end"]
    cfl               = setup["time_config"]["CFL"]
    time_integrator   = setup["time_integrator"]
    space_schemes     = setup["space_schemes"]
    ics               = setup["initial_conditions"]
    bcs               = setup["boundary_conditions"]
    saver             = setup["save_config"]["saver"]
    output_freq       = setup["save_config"]["output_frequency"]
    save_dir          = setup["save_config"]["directory"]
    coordinate_system = setup["coordinate_system"]
    grid_config       = setup["grid_config"]

    allocate_u = physics.allocate_u
    rhs        = physics.compute_rhs
    compute_dt = physics.compute_dt
    build_grid = geometry.build_grid

    # Create grid.
    grid  = build_grid(coordinate_system, grid_config)

    # Create conserved arrays.
    U = allocate_u(grid_config)

    # Initial and boundary conditions.
    apply_ics(U, params, grid, ics)
    apply_bcs(U, bcs)

    # Time loop.
    t = t_start
    n = 0
    while t <= t_end:

        # Compute time step.
        dt = compute_dt(U, params, grid, cfl)

        # Advance solution by one time step.
        time_integrator(rhs, U, dt, params, grid, space_schemes, bcs, apply_bcs)

        # Save snapshot and print progress.
        if n % output_freq == 0:
            print(f"n = {n:6d} | t = {t:.2f}/{t_end:.2f} | dt = {dt:.2e} | {100*t/t_end:.1f}%")
            saver(U, setup, t, n, save_dir)

        t += dt
        n += 1
