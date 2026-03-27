"""
File   : navier_stokes.py
Author : Nathan ZIMNIAK
Date   : 2026-03-22
-----------------
Right-hand side and CFL time step for the 3D compressible Navier-Stokes equations.
"""


import numpy as np


def allocate_u(
    grid_config : dict
    ) -> dict:
    """
    Allocate conserved variables arrays with one layer of ghost cells on each boundary.

    Arguments
    ---------
    grid_config : Grid configuration.

    Returns
    -------
    U : Zero-initialized conserved variables arrays with ghost cells.
    """

    # Unpack inputs.
    nx1 = grid_config["nx1"]
    nx2 = grid_config["nx2"]
    nx3 = grid_config["nx3"]

    # State arrays including one layer of ghost cells on each boundary.
    sg = (nx1+2, nx2+2, nx3+2)
    U = {"mass_density"   : np.zeros(sg),
         "momentum_x1"    : np.zeros(sg),
         "momentum_x2"    : np.zeros(sg),
         "momentum_x3"    : np.zeros(sg),
         "energy_density" : np.zeros(sg)}

    return U


def compute_closure(
    U      : dict,
    params : dict,
    grid   : dict
    ) -> dict:
    """
    Compute closure variables from the conserved variables.

    Arguments
    ---------
    U      : Conserved variable arrays (with ghost cells).
    grid   : Discrete grid quantities, coordinates, and domain config.
    params : Physical parameters.

    Returns
    -------
    C : Closure variables (pressure and viscous stress tensor) with ghost cells.
    """

    # Unpack conserved variables.
    rho    = U["mass_density"]
    rhovx1 = U["momentum_x1"]
    rhovx2 = U["momentum_x2"]
    rhovx3 = U["momentum_x3"]
    rhoE   = U["energy_density"]

    # Unpack physical parameters.
    ga = params["heat_capacity_ratio"]
    mu = params["dynamic_viscosity"]

    # Unpack physical lengths.
    L = grid["cell"]["geometry"]["length"]
    Lx1 = L["x1"]
    Lx2 = L["x2"]
    Lx3 = L["x3"]

    # Unpack Lamé coefficients.
    hx1 = grid["lame"]["coefficients"]["x1"]
    hx2 = grid["lame"]["coefficients"]["x2"]
    hx3 = grid["lame"]["coefficients"]["x3"]

    # Unpack Lamé coefficients derivatives.
    dx1_ln_hx2 = grid["lame"]["derivatives"]["x1"]["ln_hx2"]
    dx1_ln_hx3 = grid["lame"]["derivatives"]["x1"]["ln_hx3"]
    dx2_ln_hx1 = grid["lame"]["derivatives"]["x2"]["ln_hx1"]
    dx2_ln_hx3 = grid["lame"]["derivatives"]["x2"]["ln_hx3"]
    dx3_ln_hx1 = grid["lame"]["derivatives"]["x3"]["ln_hx1"]
    dx3_ln_hx2 = grid["lame"]["derivatives"]["x3"]["ln_hx2"]

    # Primitive variables.
    vx1 = rhovx1/rho
    vx2 = rhovx2/rho
    vx3 = rhovx3/rho
    E   = rhoE/rho

    # Pressure from the ideal gas equation of state.
    P = (ga-1.0)*rho*(E - (vx1**2+vx2**2+vx3**2)/2.0)

    # Slices of neighboring cell centers for centered derivatives: i_m = i-1, i_p = i+1.
    im = (slice(None, -2), slice(1, -1), slice(1, -1)) # [:-2, 1:-1, 1:-1]
    ip = (slice(2, None), slice(1, -1), slice(1, -1))  # [2:, 1:-1, 1:-1]
    jm = (slice(1, -1), slice(None, -2), slice(1, -1)) # [1:-1, :-2, 1:-1]
    jp = (slice(1, -1), slice(2, None), slice(1, -1))  # [1:-1, 2:, 1:-1]
    km = (slice(1, -1), slice(1, -1), slice(None, -2)) # [1:-1,1:-1,:-2]
    kp = (slice(1, -1), slice(1, -1), slice(2, None))  # [1:-1,1:-1,2:]

    # Velocity gradient using centered finite differences.
    dvx1_dx1 = (vx1[ip]-vx1[im])/(2*Lx1)
    dvx2_dx1 = (vx2[ip]-vx2[im])/(2*Lx1)
    dvx3_dx1 = (vx3[ip]-vx3[im])/(2*Lx1)
    dvx1_dx2 = (vx1[jp]-vx1[jm])/(2*Lx2)
    dvx2_dx2 = (vx2[jp]-vx2[jm])/(2*Lx2)
    dvx3_dx2 = (vx3[jp]-vx3[jm])/(2*Lx2)
    dvx1_dx3 = (vx1[kp]-vx1[km])/(2*Lx3)
    dvx2_dx3 = (vx2[kp]-vx2[km])/(2*Lx3)
    dvx3_dx3 = (vx3[kp]-vx3[km])/(2*Lx3)

    # Remove ghost cells from primitive variables.
    ing = (slice(1,-1), slice(1,-1), slice(1,-1))
    vx1   = vx1[ing]
    vx2   = vx2[ing]
    vx3   = vx3[ing]

    # Velocity divergence.
    divv = (dvx1_dx1 + dvx2_dx2 + dvx3_dx3 + vx1*(dx1_ln_hx2+dx1_ln_hx3)/hx1 + vx2*(dx2_ln_hx1+dx2_ln_hx3)/hx2 + vx3*(dx3_ln_hx1+dx3_ln_hx2)/hx3)

    # Viscous tensor from Stokes hypothesis.
    Tx1x1 = mu*(2*dvx1_dx1 - (2.0/3.0)*divv)
    Tx2x2 = mu*(2*(dvx2_dx2 + vx1*dx1_ln_hx2/hx1) - (2.0/3.0)*divv)
    Tx3x3 = mu*(2*(dvx3_dx3 + vx1*dx1_ln_hx3/hx1 + vx2*dx2_ln_hx3/hx2) - (2.0/3.0)*divv)
    Tx1x2 = mu*(dvx1_dx2 + dvx2_dx1 - vx2*dx1_ln_hx2/hx1 + vx1*dx2_ln_hx1/hx2)
    Tx1x3 = mu*(dvx1_dx3 + dvx3_dx1 - vx3*dx1_ln_hx3/hx1 + vx1*dx3_ln_hx1/hx3)
    Tx2x3 = mu*(dvx2_dx3 + dvx3_dx2 - vx3*dx2_ln_hx3/hx2 + vx2*dx3_ln_hx2/hx3)
    Tx2x1 = Tx1x2
    Tx3x1 = Tx1x3
    Tx3x2 = Tx2x3

    # Embed viscous tensor components with ghost cells.
    Tx1x1, Tx2x2, Tx3x3, Tx1x2, Tx1x3, Tx2x3, Tx2x1, Tx3x1, Tx3x2 = [np.pad(T, 1) for T in [Tx1x1, Tx2x2, Tx3x3, Tx1x2, Tx1x3, Tx2x3, Tx2x1, Tx3x1, Tx3x2]]

    # Closure variables.
    C = {"P"     : P,
         "Tx1x1" : Tx1x1,
         "Tx2x2" : Tx2x2,
         "Tx3x3" : Tx3x3,
         "Tx1x2" : Tx1x2,
         "Tx1x3" : Tx1x3,
         "Tx2x3" : Tx2x3,
         "Tx2x1" : Tx2x1,
         "Tx3x1" : Tx3x1,
         "Tx3x2" : Tx3x2}
        
    return C


def compute_convective_flux(
    U      : dict,
    C      : dict,
    params : dict,
    n      : np.array
    ) -> dict:
    """
    Compute the convective (hyperbolic) flux normal to an interface.

    Arguments
    ---------
    U : Conserved variable arrays.
    C : Closure variable arrays.
    n : Unit normal vector to the interface.

    Returns
    -------
    Fc_n : Normal convective flux for each conserved variable.
    """

    # Unpack conserved variables.
    rho    = U["mass_density"]
    rhovx1 = U["momentum_x1"]
    rhovx2 = U["momentum_x2"]
    rhovx3 = U["momentum_x3"]
    rhoE   = U["energy_density"]

    # Unpack closure variables.
    P = C["P"]

    # Unpack normal vector components.
    nx1 = n[0]
    nx2 = n[1]
    nx3 = n[2]

    # Primitive variables.
    vx1 = rhovx1/rho
    vx2 = rhovx2/rho
    vx3 = rhovx3/rho
    E   = rhoE/rho

    # Convective fluxes.
    Fc = {"mass_density"   : {"x1" : rho*vx1,       "x2" : rho*vx2,       "x3" : rho*vx3},
          "momentum_x1"    : {"x1" : rho*vx1*vx1+P, "x2" : rho*vx1*vx2,   "x3" : rho*vx1*vx3},
          "momentum_x2"    : {"x1" : rho*vx2*vx1,   "x2" : rho*vx2*vx2+P, "x3" : rho*vx2*vx3},
          "momentum_x3"    : {"x1" : rho*vx3*vx1,   "x2" : rho*vx3*vx2,   "x3" : rho*vx3*vx3+P},
          "energy_density" : {"x1" : (rho*E+P)*vx1, "x2" : (rho*E+P)*vx2, "x3" : (rho*E+P)*vx3}}

    # Normal convective fluxes.
    Fc_n = {"mass_density"   : nx1*Fc["mass_density"]["x1"]   + nx2*Fc["mass_density"]["x2"]   + nx3*Fc["mass_density"]["x3"],
            "momentum_x1"    : nx1*Fc["momentum_x1"]["x1"]    + nx2*Fc["momentum_x1"]["x2"]    + nx3*Fc["momentum_x1"]["x3"],
            "momentum_x2"    : nx1*Fc["momentum_x2"]["x1"]    + nx2*Fc["momentum_x2"]["x2"]    + nx3*Fc["momentum_x2"]["x3"],
            "momentum_x3"    : nx1*Fc["momentum_x3"]["x1"]    + nx2*Fc["momentum_x3"]["x2"]    + nx3*Fc["momentum_x3"]["x3"],
            "energy_density" : nx1*Fc["energy_density"]["x1"] + nx2*Fc["energy_density"]["x2"] + nx3*Fc["energy_density"]["x3"]}

    return Fc_n


def compute_diffusive_flux(
    U : dict,
    C : dict,
    n : np.ndarray
    ) -> dict:
    """
    Compute the diffusive (parabolic) flux normal to an interface.

    Arguments
    ---------
    U : Conserved variable arrays.
    C : Closure variable arrays.
    n : Unit normal vector to the interface.

    Returns
    -------
    Fd_n : Normal diffusive flux for each conserved variable.
    """

    # Unpack conserved variables.
    rho    = U["mass_density"]
    rhovx1 = U["momentum_x1"]
    rhovx2 = U["momentum_x2"]
    rhovx3 = U["momentum_x3"]

    # Unpack closure variables.
    Tx1x1 = C["Tx1x1"]
    Tx2x2 = C["Tx2x2"]
    Tx3x3 = C["Tx3x3"]
    Tx1x2 = C["Tx1x2"]
    Tx1x3 = C["Tx1x3"]
    Tx2x3 = C["Tx2x3"]
    Tx2x1 = C["Tx2x1"]
    Tx3x1 = C["Tx3x1"]
    Tx3x2 = C["Tx3x2"]

    # Unpack normal vector components.
    nx1 = n[0]
    nx2 = n[1]
    nx3 = n[2]

    # Primitive variables.
    vx1 = rhovx1/rho
    vx2 = rhovx2/rho
    vx3 = rhovx3/rho

    # Diffusive fluxes.
    Fd = {"mass_density"   : {"x1" : 0.0, "x2" : 0.0, "x3" : 0.0},
          "momentum_x1"    : {"x1" : -Tx1x1, "x2" : -Tx1x2, "x3" : -Tx1x3},
          "momentum_x2"    : {"x1" : -Tx2x1, "x2" : -Tx2x2, "x3" : -Tx2x3},
          "momentum_x3"    : {"x1" : -Tx3x1, "x2" : -Tx3x2, "x3" : -Tx3x3},
          "energy_density" : {"x1" : -Tx1x1*vx1-Tx1x2*vx2-Tx1x3*vx3, "x2" : -Tx2x1*vx1-Tx2x2*vx2-Tx2x3*vx3, "x3" : -Tx3x1*vx1-Tx3x2*vx2-Tx3x3*vx3}}

    # Normal diffusive fluxes.
    Fd_n = {"mass_density"   : nx1*Fd["mass_density"]["x1"]   + nx2*Fd["mass_density"]["x2"]   + nx3*Fd["mass_density"]["x3"],
            "momentum_x1"    : nx1*Fd["momentum_x1"]["x1"]    + nx2*Fd["momentum_x1"]["x2"]    + nx3*Fd["momentum_x1"]["x3"],
            "momentum_x2"    : nx1*Fd["momentum_x2"]["x1"]    + nx2*Fd["momentum_x2"]["x2"]    + nx3*Fd["momentum_x2"]["x3"],
            "momentum_x3"    : nx1*Fd["momentum_x3"]["x1"]    + nx2*Fd["momentum_x3"]["x2"]    + nx3*Fd["momentum_x3"]["x3"],
            "energy_density" : nx1*Fd["energy_density"]["x1"] + nx2*Fd["energy_density"]["x2"] + nx3*Fd["energy_density"]["x3"]}

    return Fd_n


def compute_characteristic_velocity(
    U      : dict,
    C      : dict,
    params : dict
    ) -> dict:
    """
    Compute the characteristic wave velocities in the moving frame.
    
    Arguments
    ---------
    U      : Conserved variable arrays.
    C      : Closure variable arrays.
    params : Physical parameters.
    
    Returns
    -------
    kappa : Dictionary of characteristic wave velocities in the moving frame.
    """

    # Unpack conserved variables.
    rho = U["mass_density"]

    # Unpack closure variables.
    P = C["P"]

    # Unpack physical parameters
    ga = params["heat_capacity_ratio"]

    # Sound speed.
    cs = np.sqrt(ga*P/rho)

    # Characteristic wave velocities in the moving frame.
    kappa = {"contact"        : np.full(rho.shape,0),
             "acoustic_minus" : -cs,
             "acoustic_plus"  : cs}

    return kappa


def compute_normal_velocity(
    U : dict,
    n : np.ndarray
    ) -> np.ndarray:
    """
    Compute the normal velocity at an interface.

    Arguments
    ---------
    U : Conserved variable arrays.
    n : Unit normal vector to the interface.

    Returns
    -------
    v_n : Normal velocity.
    """

    # Unpack conserved variables.
    rho    = U["mass_density"]
    rhovx1 = U["momentum_x1"]
    rhovx2 = U["momentum_x2"]
    rhovx3 = U["momentum_x3"]

    # Unpack normal vector components.
    nx1 = n[0]
    nx2 = n[1]
    nx3 = n[2]

    # Primitive variables.
    vx1 = rhovx1/rho
    vx2 = rhovx2/rho
    vx3 = rhovx3/rho

    # Normal velocity.
    v_n = nx1*vx1 + nx2*vx2 + nx3*vx3

    return v_n


def compute_geometric_source(
    U      : dict,
    C      : dict,
    params : dict,
    grid   : dict
    ) -> dict:
    """
    Compute the convective and diffusive geometric source terms for curvilinear coordinates.

    Arguments
    ---------
    U      : Conserved variable arrays (with ghost cells).
    C      : Closure variable arrays (with ghost cells).
    params : Physical parameters.
    grid   : Discrete grid quantities, coordinates, and domain config.

    Returns
    -------
    sg : Geometric source terms on the physical domain.
    """

    # Unpack conserved variables.
    rho    = U["mass_density"]
    rhovx1 = U["momentum_x1"]
    rhovx2 = U["momentum_x2"]
    rhovx3 = U["momentum_x3"]

    # Unpack closure variables.
    P     = C["P"]
    Tx1x1 = C["Tx1x1"]
    Tx2x2 = C["Tx2x2"]
    Tx3x3 = C["Tx3x3"]
    Tx1x2 = C["Tx1x2"]
    Tx1x3 = C["Tx1x3"]
    Tx2x3 = C["Tx2x3"]
    Tx2x1 = C["Tx2x1"]
    Tx3x1 = C["Tx3x1"]
    Tx3x2 = C["Tx3x2"]

    # Unpack Lamé coefficients.
    hx1 = grid["lame"]["coefficients"]["x1"]
    hx2 = grid["lame"]["coefficients"]["x2"]
    hx3 = grid["lame"]["coefficients"]["x3"]

    # Unpack Lamé coefficients derivatives.
    dx1_ln_hx1hx2hx3 = grid["lame"]["derivatives"]["x1"]["ln_hx1hx2hx3"]
    dx1_ln_hx2       = grid["lame"]["derivatives"]["x1"]["ln_hx2"]
    dx1_ln_hx3       = grid["lame"]["derivatives"]["x1"]["ln_hx3"]
    dx2_ln_hx1hx2hx3 = grid["lame"]["derivatives"]["x2"]["ln_hx1hx2hx3"]
    dx2_ln_hx1       = grid["lame"]["derivatives"]["x2"]["ln_hx1"]
    dx2_ln_hx3       = grid["lame"]["derivatives"]["x2"]["ln_hx3"]
    dx3_ln_hx1hx2hx3 = grid["lame"]["derivatives"]["x3"]["ln_hx1hx2hx3"]
    dx3_ln_hx1       = grid["lame"]["derivatives"]["x3"]["ln_hx1"]
    dx3_ln_hx2       = grid["lame"]["derivatives"]["x3"]["ln_hx2"]

    # Primitive variables.
    vx1 = rhovx1/rho
    vx2 = rhovx2/rho
    vx3 = rhovx3/rho

    # Remove ghost cells from conserved and primitive variables.
    ing = (slice(1,-1), slice(1,-1), slice(1,-1))
    rho   = rho[ing]
    vx1   = vx1[ing]
    vx2   = vx2[ing]
    vx3   = vx3[ing]

    # Remove ghost cells from closure variables.
    P = P[ing]
    Tx1x1 = Tx1x1[ing]
    Tx2x2 = Tx2x2[ing]
    Tx3x3 = Tx3x3[ing]
    Tx1x2 = Tx1x2[ing]
    Tx1x3 = Tx1x3[ing]
    Tx2x3 = Tx2x3[ing]
    Tx2x1 = Tx2x1[ing]
    Tx3x1 = Tx3x1[ing]
    Tx3x2 = Tx3x2[ing]

    # Convective (hyperbolic) geometric source terms.
    sgc = {"mass_density"   : 0.0,
           "momentum_x1"    : P*dx1_ln_hx1hx2hx3/hx1 + rho*vx2*vx2*dx1_ln_hx2/hx1 + rho*vx3*vx3*dx1_ln_hx3/hx1 - rho*vx1*vx2*dx2_ln_hx1/hx2 - rho*vx1*vx3*dx3_ln_hx1/hx3,
           "momentum_x2"    : P*dx2_ln_hx1hx2hx3/hx2 + rho*vx1*vx1*dx2_ln_hx1/hx2 + rho*vx3*vx3*dx2_ln_hx3/hx2 - rho*vx2*vx1*dx1_ln_hx2/hx1 - rho*vx2*vx3*dx3_ln_hx2/hx3,
           "momentum_x3"    : P*dx3_ln_hx1hx2hx3/hx3 + rho*vx1*vx1*dx3_ln_hx1/hx3 + rho*vx2*vx2*dx3_ln_hx2/hx3 - rho*vx3*vx1*dx1_ln_hx3/hx1 - rho*vx3*vx2*dx2_ln_hx3/hx2,
           "energy_density" : 0.0}
    
    # Diffusive (parabolic) geometric source terms.
    sgd = {"mass_density"   : 0.0,
           "momentum_x1"    : Tx1x2*dx2_ln_hx1/hx2 + Tx1x3*dx3_ln_hx1/hx3 - Tx2x2*dx1_ln_hx2/hx1 - Tx3x3*dx1_ln_hx3/hx1,
           "momentum_x2"    : Tx2x1*dx1_ln_hx2/hx1 + Tx2x3*dx3_ln_hx2/hx3 - Tx1x1*dx2_ln_hx1/hx2 - Tx3x3*dx2_ln_hx3/hx2,
           "momentum_x3"    : Tx3x1*dx1_ln_hx3/hx1 + Tx3x2*dx2_ln_hx3/hx2 - Tx1x1*dx3_ln_hx1/hx3 - Tx2x2*dx3_ln_hx2/hx3,
           "energy_density" : 0.0}

    # Total geometric source terms.
    sg = {k: sgc[k] + sgd[k] for k in sgc}

    return sg


def compute_physical_source(
    U      : dict,
    C      : dict,
    params : dict
    ) -> dict:
    """
    Compute the physical source terms.

    Arguments
    ---------
    U      : Conserved variable arrays (with ghost cells).
    params : Physical parameters.

    Returns
    -------
    sp : Physical source terms on the physical domain.
    """

    # Unpack conserved variables.
    rho    = U["mass_density"]
    rhovx1 = U["momentum_x1"]
    rhovx2 = U["momentum_x2"]
    rhovx3 = U["momentum_x3"]

    # Unpack physical parameters
    g = params["gravitational_acceleration"]
    gx1 = g[0]
    gx2 = g[1]
    gx3 = g[2]

    # Primitive variables.
    vx1 = rhovx1/rho
    vx2 = rhovx2/rho
    vx3 = rhovx3/rho

    # Remove ghost cells from conserved and primitive variables.
    ing = (slice(1,-1), slice(1,-1), slice(1,-1))
    rho   = rho[ing]
    vx1   = vx1[ing]
    vx2   = vx2[ing]
    vx3   = vx3[ing]

    # Gravity source terms.
    spg = {"mass_density"   : 0.0,
           "momentum_x1"    : rho*gx1,
           "momentum_x2"    : rho*gx2,
           "momentum_x3"    : rho*gx3,
           "energy_density" : rho*(gx1*vx1 + gx2*vx2 + gx3*vx3)}

    # Total physical source terms.
    sp = spg

    return sp


def compute_dt(
    U      : dict,
    params : dict,
    grid   : dict,
    cfl    : float
    ) -> float:
    """
    Compute a CFL-limited time step from the current simulation state.

    Arguments
    ---------
    U      : Conserved variable arrays (with ghost cells).
    params : Physical parameters.
    grid   : Discrete grid quantities, coordinates, and domain config.
    cfl    : Courant–Friedrichs–Lewy number.

    Returns
    -------
    dt : CFL-limited time-step.
    """

    # Unpack conserved variables.
    rho    = U["mass_density"]
    rhovx1 = U["momentum_x1"]
    rhovx2 = U["momentum_x2"]
    rhovx3 = U["momentum_x3"]

    # Unpack physical lengths.
    L = grid["cell"]["geometry"]["length"]
    Lx1 = L["x1"]
    Lx2 = L["x2"]
    Lx3 = L["x3"]

    # Unpack physical parameters.
    mu = params["dynamic_viscosity"]

    # Primitive variables.
    vx1 = rhovx1/rho
    vx2 = rhovx2/rho
    vx3 = rhovx3/rho

    v = {"x1": vx1,
         "x2": vx2,
         "x3": vx3}

    # Closure variables | shape (nx1+2, nx2+2, nx3+2).
    C = compute_closure(U, params, grid)

    # Characteristic wave velocities in the moving frame | shape (nx1+2, nx2+2, nx3+2).
    kappa = compute_characteristic_velocity(U, C, params)

    # Characteristic wave velocities in the fixed frame (lambda_k = u_n + mu_k) along each direction | shape (nx1, nx2, nx3).
    ing = (slice(1,-1), slice(1,-1), slice(1,-1))
    lambda_ = {d: {w: v[d][ing] + kappa[w][ing] for w in kappa} for d in v}

    # Maximum absolute characteristic velocities along each direction | shape (nx1, nx2, nx3).
    max_lambda = {d: np.maximum.reduce([np.abs(l) for l in lambda_[d].values()]) for d in lambda_}

    # Characteristic diffusivity.
    nu = np.max(mu/rho[ing])

    # Maximum local CFL rate | shape (nx1, nx2, nx3) etc.
    cfl_rate_loc_c = np.maximum.reduce([max_lambda["x1"]/Lx1, max_lambda["x2"]/Lx2, max_lambda["x3"]/Lx3])
    cfl_rate_loc_d = 2*nu*(1/Lx1**2 + 1/Lx2**2 + 1/Lx3**2)

    # Maximum domain-wide CFL rate.
    cfl_rate_c = np.max(cfl_rate_loc_c)
    cfl_rate_d = np.max(cfl_rate_loc_d)
    cfl_rate   = cfl_rate_c + cfl_rate_d

    # CFL time step based on the fastest signal speed in the domain.
    dt = cfl/cfl_rate

    return dt


def compute_rhs(
    U             : dict,
    grid          : dict,
    params        : dict,
    space_schemes : dict,
    ) -> dict:
    """
    Compute the right-hand side of the 3D compressible Navier-Stokes equations.

    Arguments
    ---------
    U             : Conserved variable arrays (with ghost cells).
    grid          : Discrete grid quantities, coordinates, and domain config.
    params        : Physical parameters.
    space_schemes : Spatial discretization schemes.

    Returns
    -------
    rhs : Right-hand-side terms on the physical domain.
    """

    # Unpack cell configuration.
    V = grid["cell"]["geometry"]["volume"]
    S = grid["cell"]["geometry"]["surface"]
    L = grid["cell"]["geometry"]["length"]
    n = grid["cell"]["normal"]

    # Unpack numerical schemes.
    slope_limiter    = space_schemes["slope_limiter"]
    riemann_solver   = space_schemes["riemann_solver"]
    diffusive_solver = space_schemes["diffusive_solver"]

    # Closure variables | shape (nx1+2, nx2+2, nx3+2).
    C = compute_closure(U, params, grid)

    # Conserved variables on each left/right interface along each direction | shape (nx1+1, nx2, nx3) etc.
    U_L, U_R = space_schemes["reconstructor"](U, L, slope_limiter)

    # Closure variables on each left/right interface for along direction | shape (nx1+1, nx2, nx3) etc.
    C_L, C_R = space_schemes["reconstructor"](C, L, slope_limiter)

    # Convective (hyperbolic) fluxes on each left/right interface along each direction | shape (nx1+1, nx2, nx3) etc.
    Fc_L = {"x1": compute_convective_flux(U_L["x1"], C_L["x1"], params, n["x1"]),
            "x2": compute_convective_flux(U_L["x2"], C_L["x2"], params, n["x2"]),
            "x3": compute_convective_flux(U_L["x3"], C_L["x3"], params, n["x3"])}

    Fc_R = {"x1": compute_convective_flux(U_R["x1"], C_R["x1"], params, n["x1"]),
            "x2": compute_convective_flux(U_R["x2"], C_R["x2"], params, n["x2"]),
            "x3": compute_convective_flux(U_R["x3"], C_R["x3"], params, n["x3"])}

    # Diffusive (parabolic) fluxes on each left/right interface along each direction | shape (nx1+1, nx2, nx3) etc.
    Fd_L = {"x1": compute_diffusive_flux(U_L["x1"], C_L["x1"], n["x1"]),
            "x2": compute_diffusive_flux(U_L["x2"], C_L["x2"], n["x2"]),
            "x3": compute_diffusive_flux(U_L["x3"], C_L["x3"], n["x3"])}

    Fd_R = {"x1": compute_diffusive_flux(U_R["x1"], C_R["x1"], n["x1"]),
            "x2": compute_diffusive_flux(U_R["x2"], C_R["x2"], n["x2"]),
            "x3": compute_diffusive_flux(U_R["x3"], C_R["x3"], n["x3"])}

    # Characteristic wave velocities in the moving frame on each left/right interface along each direction | shape (nx1+1, nx2, nx3) etc.
    kappa_L  = {"x1": compute_characteristic_velocity(U_L["x1"], C_L["x1"], params),
                "x2": compute_characteristic_velocity(U_L["x2"], C_L["x2"], params),
                "x3": compute_characteristic_velocity(U_L["x3"], C_L["x3"], params)}

    kappa_R  = {"x1": compute_characteristic_velocity(U_R["x1"], C_R["x1"], params),
                "x2": compute_characteristic_velocity(U_R["x2"], C_R["x2"], params),
                "x3": compute_characteristic_velocity(U_R["x3"], C_R["x3"], params)}

    # Normal velocities on each left/right interface along each direction | shape (nx1+1, nx2, nx3) etc.
    vn_L = {"x1": compute_normal_velocity(U_L["x1"], n["x1"]),
            "x2": compute_normal_velocity(U_L["x2"], n["x2"]),
            "x3": compute_normal_velocity(U_L["x3"], n["x3"])}

    vn_R = {"x1": compute_normal_velocity(U_R["x1"], n["x1"]),
            "x2": compute_normal_velocity(U_R["x2"], n["x2"]),
            "x3": compute_normal_velocity(U_R["x3"], n["x3"])}

    # Characteristic wave velocities in the fixed frame (lambda_k = u_n + kappa_k) on each left/right interface along each direction | shape (nx1+1, nx2, nx3) etc.
    lambda_L = {d: {w: vn_L[d] + kappa_L[d][w] for w in kappa_L[d]} for d in kappa_L}
    lambda_R = {d: {w: vn_R[d] + kappa_R[d][w] for w in kappa_R[d]} for d in kappa_R}

    # Convective (hyperbolic) fluxes along each direction | shape (nx1+1, nx2, nx3) etc.
    Fc = {"x1": riemann_solver(Fc_L["x1"], Fc_R["x1"], U_L["x1"], U_R["x1"], lambda_L["x1"], lambda_R["x1"]),
          "x2": riemann_solver(Fc_L["x2"], Fc_R["x2"], U_L["x2"], U_R["x2"], lambda_L["x2"], lambda_R["x2"]),
          "x3": riemann_solver(Fc_L["x3"], Fc_R["x3"], U_L["x3"], U_R["x3"], lambda_L["x3"], lambda_R["x3"])}

    # Diffusive (parabolic) fluxes along each direction | shape (nx1+1, nx2, nx3) etc.
    Fd = {"x1": diffusive_solver(Fd_L["x1"], Fd_R["x1"]),
          "x2": diffusive_solver(Fd_L["x2"], Fd_R["x2"]),
          "x3": diffusive_solver(Fd_L["x3"], Fd_R["x3"])}

    # Total fluxes | shape (nx1+1, nx2, nx3) etc.
    F = {"x1": {k: Fc["x1"][k] + Fd["x1"][k] for k in Fc["x1"]},
         "x2": {k: Fc["x2"][k] + Fd["x2"][k] for k in Fc["x2"]},
         "x3": {k: Fc["x3"][k] + Fd["x3"][k] for k in Fc["x3"]}}

    # Geometric source terms | shape (nx1, nx2, nx3).
    sg = compute_geometric_source(U, C, params, grid)
    
    # Physical source terms | shape (nx1, nx2, nx3).
    sp = compute_physical_source(U, C, params)

    # Total source terms | shape (nx1, nx2, nx3).
    src = {k: sg[k] + sp[k] for k in sg}

    # Indices for cell interface indexing: ipf = i-1/2, inf = i+1/2.
    # For xn-oriented faces (with n=1,2,3):
    # - In the longitudinal direction (xn), left cells include all cells except the last one (None,-1), right cells include all cells except the first one (1,None).
    # - In the transverse direction (xm, with m≠n) ghost cells are excluded (1,-1).
    ipf = (slice(None, -1), slice(None), slice(None)) # [:-1,:,:]
    inf = (slice(1, None), slice(None), slice(None))  # [1:,:,:]
    jpf = (slice(None), slice(None, -1), slice(None)) # [:,:-1,:]
    jnf = (slice(None), slice(1, None), slice(None))  # [:,1:,:]
    kpf = (slice(None), slice(None), slice(None, -1)) # [:,:,:-1]
    knf = (slice(None), slice(None), slice(1, None))  # [:,:,1:]

    # Finite-volume RHS: divergence of fluxes plus geometric sources | shape (nx1, nx2, nx3).
    rhs = {"mass_density"   : -((F["x1"]["mass_density"]*S["x1"])[inf]   - (F["x1"]["mass_density"]*S["x1"])[ipf]   + (F["x2"]["mass_density"]*S["x2"])[jnf]   - (F["x2"]["mass_density"]*S["x2"])[jpf]   + (F["x3"]["mass_density"]*S["x3"])[knf]   - (F["x3"]["mass_density"]*S["x3"])[kpf])/V   + src["mass_density"],
           "momentum_x1"    : -((F["x1"]["momentum_x1"]*S["x1"])[inf]    - (F["x1"]["momentum_x1"]*S["x1"])[ipf]    + (F["x2"]["momentum_x1"]*S["x2"])[jnf]    - (F["x2"]["momentum_x1"]*S["x2"])[jpf]    + (F["x3"]["momentum_x1"]*S["x3"])[knf]    - (F["x3"]["momentum_x1"]*S["x3"])[kpf] )/V   + src["momentum_x1"],
           "momentum_x2"    : -((F["x1"]["momentum_x2"]*S["x1"])[inf]    - (F["x1"]["momentum_x2"]*S["x1"])[ipf]    + (F["x2"]["momentum_x2"]*S["x2"])[jnf]    - (F["x2"]["momentum_x2"]*S["x2"])[jpf]    + (F["x3"]["momentum_x2"]*S["x3"])[knf]    - (F["x3"]["momentum_x2"]*S["x3"])[kpf] )/V   + src["momentum_x2"],
           "momentum_x3"    : -((F["x1"]["momentum_x3"]*S["x1"])[inf]    - (F["x1"]["momentum_x3"]*S["x1"])[ipf]    + (F["x2"]["momentum_x3"]*S["x2"])[jnf]    - (F["x2"]["momentum_x3"]*S["x2"])[jpf]    + (F["x3"]["momentum_x3"]*S["x3"])[knf]    - (F["x3"]["momentum_x3"]*S["x3"])[kpf] )/V   + src["momentum_x3"],
           "energy_density" : -((F["x1"]["energy_density"]*S["x1"])[inf] - (F["x1"]["energy_density"]*S["x1"])[ipf] + (F["x2"]["energy_density"]*S["x2"])[jnf] - (F["x2"]["energy_density"]*S["x2"])[jpf] + (F["x3"]["energy_density"]*S["x3"])[knf] - (F["x3"]["energy_density"]*S["x3"])[kpf])/V + src["energy_density"]}

    return rhs
