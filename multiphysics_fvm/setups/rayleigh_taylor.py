"""
File   : rayleigh_taylor.py
Author : Nathan ZIMNIAK
Date   : 2026-03-10
-----------------
Rayleigh-Taylor instability in Cartesian coordinates.
Heavy fluid on top of light fluid with a sinusoidal interface perturbation.
"""

import numpy as np
from src.numerics import riemann_solvers, diffusive_solvers, reconstructors, slope_limiters, time_integrators
from src.grid     import boundary_conditions
from src.physics  import navier_stokes
from src          import io


def initializer(
    U          : dict,
    grid       : dict,
    params     : dict,
    ics_params : dict
    ) -> None:
    """
    Initialize a Rayleigh-Taylor instability from a hydrostatic equilibrium
    with a plane density interface and a vertical velocity perturbation.

    Arguments
    ----------
    U          : Conserved variable arrays (with ghost cells).
    grid       : Discrete grid quantities, coordinates, and domain config.
    params     : Physical parameters.
    ics_params : Initial condition parameters.
    
    Returns
    -------
    None
    """

    # Unpack conserved variables.
    rho    = U["mass_density"]
    rhovx1 = U["momentum_x1"]
    rhovx2 = U["momentum_x2"]
    rhovx3 = U["momentum_x3"]
    rhoE   = U["energy_density"]

    # Unpack grid configuration.
    x   = grid["cell"]["coordinates"]["x1"]
    y   = grid["cell"]["coordinates"]["x2"]
    nx1 = grid["config"]["nx1"]
    nx2 = grid["config"]["nx2"]
    nx3 = grid["config"]["nx3"]

    # Unpack physical parameters.
    gamma = params["heat_capacity_ratio"]
    g     = params["gravitational_acceleration"]
    gy    = g[1]

    # Unpack initial condition parameters.
    rho_top = ics_params["mass_density_top"]
    rho_bot = ics_params["mass_density_bottom"]
    P0      = ics_params["background_pressure"]
    A       = ics_params["perturbation_amplitude"]
    ky      = ics_params["wavenumber"]
    delta   = ics_params["interface_thickness"]
    y_int   = ics_params["interface_position"]
    sigma   = ics_params["perturbation_width"]

    # Remove ghost cells from conserved variables.
    ing    = (slice(1, -1), slice(1, -1), slice(1, -1))
    rho    = rho[ing]
    rhovx1 = rhovx1[ing]
    rhovx2 = rhovx2[ing]
    rhovx3 = rhovx3[ing]
    rhoE   = rhoE[ing]

    # Plane density interface.
    s = np.tanh((y - y_int) / delta)

    # Initial conditions.
    rho_init = rho_bot + 0.5*(rho_top - rho_bot)*(1.0 + s)
    vx1_init = np.zeros((nx1, nx2, nx3))
    vx2_init = A*np.cos(2.0*np.pi*ky*x)*np.exp(-((y - y_int)/sigma)**2)
    vx3_init = np.zeros((nx1, nx2, nx3))

    # Hydrostatic pressure profile.
    #P_init = np.zeros_like(rho_init)
    #P_init[:, 0, :] = P0
    dy = y[:, 1:, :] - y[:, :-1, :]
    #rho_avg = 0.5*(rho_init[:, 1:, :] + rho_init[:, :-1, :])
    P_init = np.zeros_like(rho_init)
    P_init[:, 0, :] = P0
    for j in range(1, nx2):
        rho_mid = 0.5*(rho_init[:, j, :] + rho_init[:, j-1, :])
        P_init[:, j, :] = P_init[:, j-1, :] - rho_mid * np.abs(gy) * dy[:, j-1, :]

    # Total energy density.
    rhoE_init = P_init/(gamma-1.0) + 0.5*rho_init*(vx1_init**2 + vx2_init**2 + vx3_init**2)

    # Fill conserved variable arrays.
    rho[:]    = rho_init
    rhovx1[:] = rho_init*vx1_init
    rhovx2[:] = rho_init*vx2_init
    rhovx3[:] = rho_init*vx3_init
    rhoE[:]   = rhoE_init


def get_setup(
    ) -> dict:
    """
    Return the full configuration dictionary for the Rayleigh-Taylor setup.

    Returns
    -------
    setup : Full setup configuration.
    """

    # Physical module.
    physics = navier_stokes

    # Physical parameters.
    params = {"heat_capacity_ratio"        : 1.4,
              "gravitational_acceleration" : [0.0, -2.0, 0.0],
              "dynamic_viscosity"          : 0.0001}

    # Coordinate system.
    coordinate_system = "cartesian"

    # Grid configuration.
    grid_config = {"x1_min": 0.0, "x1_max": 1.0, "nx1": 256,
                   "x2_min": 0.0, "x2_max": 2.0, "nx2": 512,
                   "x3_min": 0.0, "x3_max": 1.0, "nx3": 1}

    # Time configuration.
    time_config = {"t_start": 0.0, "t_end": 10.0, "CFL": 0.2}

    # Time integrator.
    time_integrator = time_integrators.rk3_ssp

    # Space schemes.
    space_schemes = {"riemann_solver"   : riemann_solvers.hll,
                     "reconstructor"    : reconstructors.muscl,
                     "slope_limiter"    : slope_limiters.van_leer,
                     "diffusive_solver" : diffusive_solvers.centered}

    # Boundary conditions.
    bcs = {"mass_density"   : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.neumann,    "top": boundary_conditions.neumann,    "front": boundary_conditions.neumann, "back": boundary_conditions.neumann},
           "momentum_x1"    : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.neumann,    "top": boundary_conditions.neumann,    "front": boundary_conditions.neumann, "back": boundary_conditions.neumann},
           "momentum_x2"    : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.reflecting, "top": boundary_conditions.reflecting, "front": boundary_conditions.neumann, "back": boundary_conditions.neumann},
           "momentum_x3"    : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.neumann,    "top": boundary_conditions.neumann,    "front": boundary_conditions.neumann, "back": boundary_conditions.neumann},
           "energy_density" : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.neumann,    "top": boundary_conditions.neumann,    "front": boundary_conditions.neumann, "back": boundary_conditions.neumann}}

    # Initial conditions.
    ics = {"initializer" : initializer,
           "parameters"  : {"mass_density_top"       : 5.0,
                            "mass_density_bottom"    : 1.0,
                            "background_pressure"    : 15.0,
                            "perturbation_amplitude" : 0.005,
                            "wavenumber"             : 1,
                            "interface_thickness"    : 0.025,
                            "interface_position"     : 1.0,
                            "perturbation_width"     : 0.1}}

    # Save configuration.
    save_config = {"saver"            : io.save_snapshot,
                   "output_frequency" : 50,
                   "directory"        : "./outputs/rayleigh_taylor"}

    setup = {"physics"             : physics,
             "physical_parameters" : params,
             "coordinate_system"   : coordinate_system,
             "grid_config"         : grid_config,
             "time_config"         : time_config,
             "time_integrator"     : time_integrator,
             "space_schemes"       : space_schemes,
             "boundary_conditions" : bcs,
             "initial_conditions"  : ics,
             "save_config"         : save_config}

    return setup
