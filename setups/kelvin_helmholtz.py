"""
File   : kelvin_helmholtz.py
Author : Nathan ZIMNIAK
Date   : 2026-03-10
-----------------
Kelvin-Helmholtz instability.
"""

import numpy as np
from src.numerics import riemann_solvers, diffusive_solvers, reconstructors, slope_limiters, time_integrators
from src.grid     import boundary_conditions
from src.physics  import navier_stokes
from src          import io


def initializer(
    U          : dict,
    params     : dict,
    grid       : dict,
    ics_params : dict
    ) -> None:
    """
    Initialize a Kelvin-Helmholtz instability with two shear layers.

    Arguments
    ----------
    U          : Conserved variable arrays (with ghost cells).
    params     : Physical parameters.
    grid       : Discrete grid quantities, coordinates, and domain config.
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

    # Unpack initial condition parameters.
    rho_in  = ics_params["mass_density_inner"]
    rho_out = ics_params["mass_density_outer"]
    U0      = ics_params["shear_velocity"]
    P0      = ics_params["background_pressure"]
    A       = ics_params["perturbation_amplitude"]
    delta   = ics_params["shear_layer_thickness"]
    kx      = ics_params["wavenumber"]

    # Remove ghost cells from conserved variables.
    ing    = (slice(1, -1), slice(1, -1), slice(1, -1))
    rho    = rho[ing]
    rhovx1 = rhovx1[ing]
    rhovx2 = rhovx2[ing]
    rhovx3 = rhovx3[ing]
    rhoE   = rhoE[ing]

    # Shear layer profiles.
    y1 = 0.25
    y2 = 0.75
    s1 = np.tanh((y-y1)/delta)
    s2 = np.tanh((y2-y)/delta)

    # Initial conditions.
    rho_init  = rho_out + 0.5*(rho_in-rho_out)*(s1+s2)
    vx1_init  = U0*(s1+s2-1.0)
    vx2_init  = A*np.sin(2.0*np.pi*kx*x)
    vx3_init  = np.zeros((nx1, nx2, nx3))
    rhoE_init = P0/(gamma-1.0) + 0.5*rho_init*(vx1_init**2+vx2_init**2+vx3_init**2)

    # Fill conserved variable arrays.
    rho[:]    = rho_init
    rhovx1[:] = rho_init*vx1_init
    rhovx2[:] = rho_init*vx2_init
    rhovx3[:] = rho_init*vx3_init
    rhoE[:]   = rhoE_init


def get_setup(
    ) -> dict:
    """
    Compute the full configuration dictionary for the Kelvin-Helmholtz setup.

    Returns
    -------
    setup : Full setup configuration.
    """

    # Physical module.
    physics = navier_stokes

    # Physical parameters.
    params = {"heat_capacity_ratio"        : 1.4,             #gamma
              "dynamic_viscosity"          : 0.0001,          #mu
              "gravitational_acceleration" : [0.0, 0.0, 0.0]} #g

    # Coordinate system.
    coordinate_system = "cartesian"

    # Grid configuration.
    grid_config = {"x1_min": 0.0, "x1_max": 1.0, "nx1": 256,
                   "x2_min": 0.0, "x2_max": 1.0, "nx2": 256,
                   "x3_min": 0.0, "x3_max": 1.0, "nx3": 1}

    # Time configuration.
    time_config = {"t_start" : 0.0,
                   "t_end"   : 5.0,
                   "CFL"     : 0.2}

    # Time integrator.
    time_integrator = time_integrators.rk3_ssp

    # Space schemes.
    space_schemes = {"riemann_solver"   : riemann_solvers.hll,
                     "reconstructor"    : reconstructors.muscl,
                     "slope_limiter"    : slope_limiters.van_leer,
                     "diffusive_solver" : diffusive_solvers.centered}

    # Boundary conditions.
    bcs = {"mass_density"   : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.periodic, "top": boundary_conditions.periodic, "front": boundary_conditions.periodic, "back": boundary_conditions.periodic},
           "momentum_x1"    : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.periodic, "top": boundary_conditions.periodic, "front": boundary_conditions.periodic, "back": boundary_conditions.periodic},
           "momentum_x2"    : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.periodic, "top": boundary_conditions.periodic, "front": boundary_conditions.periodic, "back": boundary_conditions.periodic},
           "momentum_x3"    : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.periodic, "top": boundary_conditions.periodic, "front": boundary_conditions.periodic, "back": boundary_conditions.periodic},
           "energy_density" : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.periodic, "top": boundary_conditions.periodic, "front": boundary_conditions.periodic, "back": boundary_conditions.periodic}}

    # Initial conditions.
    ics = {"initializer" : initializer,
           "parameters"  : {"mass_density_inner"     : 2.0,
                            "mass_density_outer"     : 1.0,
                            "shear_velocity"         : 0.5,
                            "background_pressure"    : 2.5,
                            "perturbation_amplitude" : 0.01,
                            "shear_layer_thickness"  : 0.025,
                            "wavenumber"             : 2}}

    # Save configuration.
    save_config = {"saver"            : io.save_snapshot,
                   "output_frequency" : 50,
                   "directory"        : "./outputs/kelvin_helmholtz"}

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
