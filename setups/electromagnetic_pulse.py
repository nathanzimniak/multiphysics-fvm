"""
File   : electromagnetic_pulse.py
Author : Nathan ZIMNIAK
Date   : 2026-03-27
-----------------
Gaussian electromagnetic pulse.
"""

import numpy as np
from src.numerics import riemann_solvers, reconstructors, slope_limiters, time_integrators
from src.grid     import boundary_conditions
from src.physics  import maxwell
from src          import io


def initializer(
    U          : dict,
    params     : dict,
    grid       : dict,
    ics_params : dict
    ) -> None:
    """
    Initialize a centered Gaussian pulse on Ez and convert to Dz.
    """

    # Unpack conserved variables.
    Bx1 = U["magnetic_field_x1"]
    Bx2 = U["magnetic_field_x2"]
    Bx3 = U["magnetic_field_x3"]
    Dx1 = U["electric_d_field_x1"]
    Dx2 = U["electric_d_field_x2"]
    Dx3 = U["electric_d_field_x3"]

    # Unpack grid configuration.
    x1 = grid["cell"]["coordinates"]["x1"]
    x2 = grid["cell"]["coordinates"]["x2"]

    # Unpack physical parameters.
    ep = params["electric_permittivity"]

    # Unpack initial condition parameters.
    E0    = ics_params["electric_field_amplitude"]
    x0    = ics_params["pulse_center_x1"]
    y0    = ics_params["pulse_center_x2"]
    sigma = ics_params["pulse_width"]

    # Remove ghost cells from conserved variables.
    ing = (slice(1, -1), slice(1, -1), slice(1, -1))
    Bx1 = Bx1[ing]
    Bx2 = Bx2[ing]
    Bx3 = Bx3[ing]
    Dx1 = Dx1[ing]
    Dx2 = Dx2[ing]
    Dx3 = Dx3[ing]

    # Initial conditions.
    Ex3_init = E0*np.exp(-((x1-x0)**2+(x2-y0)**2)/(2.0*sigma**2))
    Bx1_init = 0.0
    Bx2_init = 0.0
    Bx3_init = 0.0
    Dx1_init = 0.0
    Dx2_init = 0.0
    Dx3_init = ep*Ex3_init

    # Fill conserved variable arrays.
    Bx1[:] = Bx1_init
    Bx2[:] = Bx2_init
    Bx3[:] = Bx3_init
    Dx1[:] = Dx1_init
    Dx2[:] = Dx2_init
    Dx3[:] = Dx3_init


def get_setup(
    ) -> dict:
    """
    Compute the full configuration dictionary for the Gaussian pulse Maxwell test.

    Returns
    -------
    setup : Full setup configuration.
    """

    # Physics module.
    physics = maxwell

    # Physical parameters.
    params = {"magnetic_permeability"      : 1.0, #mu
              "electric_permittivity"      : 1.0, #ep
              "electrical_conductivity"    : 0.0, #sigma
              "charge_density"             : 0.0} #rho

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
    space_schemes = {"riemann_solver" : riemann_solvers.hll,
                     "reconstructor"  : reconstructors.muscl,
                     "slope_limiter"  : slope_limiters.van_leer}

    # Periodic BC for all fields.
    bcs = {"magnetic_field_x1"   : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.periodic, "top": boundary_conditions.periodic, "front": boundary_conditions.periodic, "back": boundary_conditions.periodic},
           "magnetic_field_x2"   : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.periodic, "top": boundary_conditions.periodic, "front": boundary_conditions.periodic, "back": boundary_conditions.periodic},
           "magnetic_field_x3"   : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.periodic, "top": boundary_conditions.periodic, "front": boundary_conditions.periodic, "back": boundary_conditions.periodic},
           "electric_d_field_x1" : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.periodic, "top": boundary_conditions.periodic, "front": boundary_conditions.periodic, "back": boundary_conditions.periodic},
           "electric_d_field_x2" : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.periodic, "top": boundary_conditions.periodic, "front": boundary_conditions.periodic, "back": boundary_conditions.periodic},
           "electric_d_field_x3" : {"left": boundary_conditions.periodic, "right": boundary_conditions.periodic, "bottom": boundary_conditions.periodic, "top": boundary_conditions.periodic, "front": boundary_conditions.periodic, "back": boundary_conditions.periodic}}

    # Initial conditions.
    ics = {"initializer": initializer,
           "parameters": {"electric_field_amplitude": 1.0,
                          "pulse_center_x1": 0.5,
                          "pulse_center_x2": 0.5,
                          "pulse_width": 0.04}}

    # Save configuration.
    save_config = {"saver"            : io.save_snapshot,
                   "output_frequency" : 50,
                   "directory"        : "./outputs/electromagnetic_pulse"}

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