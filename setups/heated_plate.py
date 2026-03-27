"""
File   : heated_plate.py
Author : Nathan ZIMNIAK
Date   : 2026-03-25
-----------------
Gaussian hot spot diffusing in a plate.
"""

import numpy as np
from src.numerics import diffusive_solvers, reconstructors, slope_limiters, time_integrators
from src.grid     import boundary_conditions
from src.physics  import heat
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
    rhocpT = U["internal_energy_density"]

    # Unpack grid configuration.
    x = grid["cell"]["coordinates"]["x1"]
    y = grid["cell"]["coordinates"]["x2"]

    # Unpack physical parameters.
    rho = params["mass_density"]
    cp  = params["specific_heat_capacity"]

    # Unpack initial condition parameters.
    T0     = ics_params["background_temperature"]
    A      = ics_params["hot_spot_amplitude"]
    x0     = ics_params["hot_spot_center_x1"]
    y0     = ics_params["hot_spot_center_x2"]
    sigma  = ics_params["hot_spot_width"]

    # Remove ghost cells from conserved variables.
    ing    = (slice(1, -1), slice(1, -1), slice(1, -1))
    rhocpT = rhocpT[ing]

    # Build temperature field.
    T_init = T0 + A*np.exp(-((x-x0)**2+(y-y0)**2)/(2.0*sigma**2))
    rhocpT_init = rho*cp*T_init

    # Fill conserved variable arrays.
    rhocpT[:] = rhocpT_init


def get_setup(
    ) -> dict:
    """
    Compute the full configuration dictionary for the heat diffusion plate setup.

    Returns
    -------
    setup : Full setup configuration.
    """

    # Physics module.
    physics = heat

    # Physical parameters (example values).
    params = {"mass_density"            : 10000.0, #rho
              "specific_heat_capacity"  : 200.0,   #cp
              "thermal_conductivity"    : 800.0,   #k
              "volumetric_heat_source"  : 0.0}     #q

    # Coordinate system.
    coordinate_system = "cartesian"

    # Grid configuration.
    grid_config = {"x1_min": 0.0, "x1_max": 1.0, "nx1": 256,
                   "x2_min": 0.0, "x2_max": 1.0, "nx2": 256,
                   "x3_min": 0.0, "x3_max": 1.0, "nx3": 1}

    # Time configuration.
    time_config = {"t_start": 0.0, "t_end": 5.0, "CFL": 0.2}

    # Time integrator.
    time_integrator = time_integrators.rk3_ssp

    # Space schemes.
    space_schemes = {"reconstructor"    : reconstructors.muscl,
                     "slope_limiter"    : slope_limiters.van_leer,
                     "diffusive_solver" : diffusive_solvers.centered}

    # Boundary conditions.
    bcs = {"internal_energy_density": {"left": boundary_conditions.neumann, "right": boundary_conditions.neumann, "bottom": boundary_conditions.neumann, "top": boundary_conditions.neumann, "front": boundary_conditions.neumann, "back": boundary_conditions.neumann}}

    # Initial conditions.
    ics = {"initializer": initializer,
           "parameters": {"background_temperature": 300.0,
                          "hot_spot_amplitude"    : 1000.0,
                          "hot_spot_center_x1"    : 0.5,
                          "hot_spot_center_x2"    : 0.5,
                          "hot_spot_width"        : 0.08}}

    # Save configuration.
    save_config = {"saver"            : io.save_snapshot,
                   "output_frequency" : 50,
                   "directory"        : "./outputs/heated_plate"}

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