"""
File   : maxwell.py
Author : Nathan ZIMNIAK
Date   : 2026-03-26
-----------------
Right-hand side and CFL time step for the 3D Maxwell's equations.
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
    U = {"magnetic_field_x1"   : np.zeros(sg),
         "magnetic_field_x2"   : np.zeros(sg),
         "magnetic_field_x3"   : np.zeros(sg),
         "electric_d_field_x1" : np.zeros(sg),
         "electric_d_field_x2" : np.zeros(sg),
         "electric_d_field_x3" : np.zeros(sg)}

    return U


def compute_convective_flux(
    U      : dict,
    params : dict,
    n      : np.array
    ) -> dict:
    """
    Compute the convective (hyperbolic) flux normal to an interface.

    Arguments
    ---------
    U      : Conserved variable arrays.
    params : Physical parameters.
    n      : Unit normal vector to the interface.

    Returns
    -------
    Fc_n : Normal convective flux for each conserved variable.
    """

    # Unpack conserved variables.
    Bx1 = U["magnetic_field_x1"]
    Bx2 = U["magnetic_field_x2"]
    Bx3 = U["magnetic_field_x3"]
    Dx1 = U["electric_d_field_x1"]
    Dx2 = U["electric_d_field_x2"]
    Dx3 = U["electric_d_field_x3"]

    # Unpack normal vector components.
    nx1 = n[0]
    nx2 = n[1]
    nx3 = n[2]

    # Unpack physical parameters
    mu = params["magnetic_permeability"]
    ep = params["electric_permittivity"]

    # Primitive variables.
    Hx1 = Bx1/mu
    Hx2 = Bx2/mu
    Hx3 = Bx3/mu
    Ex1 = Dx1/ep
    Ex2 = Dx2/ep
    Ex3 = Dx3/ep

    # Magnetic flux tensor.
    FBx1x1 = 0.0
    FBx2x2 = 0.0
    FBx3x3 = 0.0
    FBx1x2 = Ex3
    FBx1x3 = -Ex2
    FBx2x3 = Ex1
    FBx2x1 = -FBx1x2
    FBx3x1 = -FBx1x3
    FBx3x2 = -FBx2x3

    # Electric displacement flux tensor.
    FDx1x1 = 0.0
    FDx2x2 = 0.0
    FDx3x3 = 0.0
    FDx1x2 = -Hx3
    FDx1x3 = Hx2
    FDx2x3 = -Hx1
    FDx2x1 = -FDx1x2
    FDx3x1 = -FDx1x3
    FDx3x2 = -FDx2x3

    # Convective fluxes.
    Fc = {"magnetic_field_x1"   : {"x1" : FBx1x1, "x2" : FBx1x2, "x3" : FBx1x3},
          "magnetic_field_x2"   : {"x1" : FBx2x1, "x2" : FBx2x2, "x3" : FBx2x3},
          "magnetic_field_x3"   : {"x1" : FBx3x1, "x2" : FBx3x2, "x3" : FBx3x3},
          "electric_d_field_x1" : {"x1" : FDx1x1, "x2" : FDx1x2, "x3" : FDx1x3},
          "electric_d_field_x2" : {"x1" : FDx2x1, "x2" : FDx2x2, "x3" : FDx2x3},
          "electric_d_field_x3" : {"x1" : FDx3x1, "x2" : FDx3x2, "x3" : FDx3x3}}

    # Normal convective fluxes.
    Fc_n = {"magnetic_field_x1"   : nx1*Fc["magnetic_field_x1"]["x1"]   + nx2*Fc["magnetic_field_x1"]["x2"]   + nx3*Fc["magnetic_field_x1"]["x3"],
            "magnetic_field_x2"   : nx1*Fc["magnetic_field_x2"]["x1"]   + nx2*Fc["magnetic_field_x2"]["x2"]   + nx3*Fc["magnetic_field_x2"]["x3"],
            "magnetic_field_x3"   : nx1*Fc["magnetic_field_x3"]["x1"]   + nx2*Fc["magnetic_field_x3"]["x2"]   + nx3*Fc["magnetic_field_x3"]["x3"],
            "electric_d_field_x1" : nx1*Fc["electric_d_field_x1"]["x1"] + nx2*Fc["electric_d_field_x1"]["x2"] + nx3*Fc["electric_d_field_x1"]["x3"],
            "electric_d_field_x2" : nx1*Fc["electric_d_field_x2"]["x1"] + nx2*Fc["electric_d_field_x2"]["x2"] + nx3*Fc["electric_d_field_x2"]["x3"],
            "electric_d_field_x3" : nx1*Fc["electric_d_field_x3"]["x1"] + nx2*Fc["electric_d_field_x3"]["x2"] + nx3*Fc["electric_d_field_x3"]["x3"]}

    return Fc_n


def compute_characteristic_velocity(
    U      : dict,
    params : dict
    ) -> dict:
    """
    Compute the characteristic wave velocities in the fluid frame.
    
    Arguments
    ---------
    U      : Conserved variable arrays.
    params : Physical parameters.
    
    Returns
    -------
    kappa : Dictionary of characteristic wave velocities in the fluid frame.
    """

    # Unpack conserved variables.
    Bx1 = U["magnetic_field_x1"]

    # Unpack physical parameters
    mu = params["magnetic_permeability"]
    ep = params["electric_permittivity"]

    # Sound speed.
    c = 1/np.sqrt(mu*ep)

    # Characteristic wave velocities in the fluid frame.
    kappa = {"light_minus" : np.full(Bx1.shape,-c),
             "light_plus"  : np.full(Bx1.shape,+c)}

    return kappa


def compute_geometric_source(
    U      : dict,
    params : dict,
    grid   : dict
    ) -> dict:
    """
    Compute the convective and diffusive geometric source terms for curvilinear coordinates.

    Arguments
    ---------
    U      : Conserved variable arrays (with ghost cells).
    params : Physical parameters.
    grid   : Discrete grid quantities, coordinates, and domain config.

    Returns
    -------
    sg : Geometric source terms on the physical domain.
    """

    # Unpack conserved variables.
    Bx1 = U["magnetic_field_x1"]
    Bx2 = U["magnetic_field_x2"]
    Bx3 = U["magnetic_field_x3"]
    Dx1 = U["electric_d_field_x1"]
    Dx2 = U["electric_d_field_x2"]
    Dx3 = U["electric_d_field_x3"]

    # Unpack physical parameters
    mu = params["magnetic_permeability"]
    ep = params["electric_permittivity"]

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
    Hx1 = Bx1/mu
    Hx2 = Bx2/mu
    Hx3 = Bx3/mu
    Ex1 = Dx1/ep
    Ex2 = Dx2/ep
    Ex3 = Dx3/ep

    # Magnetic flux tensor.
    FBx1x1 = 0.0
    FBx2x2 = 0.0
    FBx3x3 = 0.0
    FBx1x2 = Ex3
    FBx1x3 = -Ex2
    FBx2x3 = Ex1
    FBx2x1 = -FBx1x2
    FBx3x1 = -FBx1x3
    FBx3x2 = -FBx2x3

    # Electric displacement flux tensor.
    FDx1x1 = 0.0
    FDx2x2 = 0.0
    FDx3x3 = 0.0
    FDx1x2 = -Hx3
    FDx1x3 = Hx2
    FDx2x3 = -Hx1
    FDx2x1 = -FDx1x2
    FDx3x1 = -FDx1x3
    FDx3x2 = -FDx2x3

    # Remove ghost cells from fluxes tensors.
    ing = (slice(1,-1), slice(1,-1), slice(1,-1))
    #FDx1x1 = FDx1x1[ing]
    #FDx2x2 = FDx2x2[ing]
    #FDx3x3 = FDx3x3[ing]
    FDx1x2 = FDx1x2[ing]
    FDx1x3 = FDx1x3[ing]
    FDx2x3 = FDx2x3[ing]
    FDx2x1 = FDx2x1[ing]
    FDx3x1 = FDx3x1[ing]
    FDx3x2 = FDx3x2[ing]
    #FBx1x1 = FBx1x1[ing]
    #FBx2x2 = FBx2x2[ing]
    #FBx3x3 = FBx3x3[ing]
    FBx1x2 = FBx1x2[ing]
    FBx1x3 = FBx1x3[ing]
    FBx2x3 = FBx2x3[ing]
    FBx2x1 = FBx2x1[ing]
    FBx3x1 = FBx3x1[ing]
    FBx3x2 = FBx3x2[ing]

    # Convective (hyperbolic) geometric source terms.
    sgc = {"magnetic_field_x1"   : FBx1x2*dx2_ln_hx1/hx2 + FBx1x3*dx3_ln_hx1/hx3 - FBx2x2*dx1_ln_hx2/hx1 - FBx3x3*dx1_ln_hx3/hx1,
           "magnetic_field_x2"   : FBx2x1*dx1_ln_hx2/hx1 + FBx2x3*dx3_ln_hx2/hx3 - FBx1x1*dx2_ln_hx1/hx2 - FBx3x3*dx2_ln_hx3/hx2,
           "magnetic_field_x3"   : FBx3x1*dx1_ln_hx3/hx1 + FBx3x2*dx2_ln_hx3/hx2 - FBx1x1*dx3_ln_hx1/hx3 - FBx2x2*dx3_ln_hx2/hx3,
           "electric_d_field_x1" : FDx1x2*dx2_ln_hx1/hx2 + FDx1x3*dx3_ln_hx1/hx3 - FDx2x2*dx1_ln_hx2/hx1 - FDx3x3*dx1_ln_hx3/hx1,
           "electric_d_field_x2" : FDx2x1*dx1_ln_hx2/hx1 + FDx2x3*dx3_ln_hx2/hx3 - FDx1x1*dx2_ln_hx1/hx2 - FDx3x3*dx2_ln_hx3/hx2,
           "electric_d_field_x3" : FDx3x1*dx1_ln_hx3/hx1 + FDx3x2*dx2_ln_hx3/hx2 - FDx1x1*dx3_ln_hx1/hx3 - FDx2x2*dx3_ln_hx2/hx3}

    # Total geometric source terms.
    sg = {k: sgc[k] for k in sgc}

    return sg


def compute_physical_source(
    U      : dict,
    params : dict
    ) -> dict:
    """
    Compute the current density source terms.

    Arguments
    ---------
    U      : Conserved variable arrays (with ghost cells).
    params : Physical parameters.

    Returns
    -------
    sp : Physical source terms on the physical domain.
    """

    # Unpack conserved variables.
    Dx1 = U["electric_d_field_x1"]
    Dx2 = U["electric_d_field_x2"]
    Dx3 = U["electric_d_field_x3"]

    # Unpack physical parameters
    ep    = params["electric_permittivity"]
    sigma = params["electrical_conductivity"]

    # Primitive variables.
    Ex1 = Dx1/ep
    Ex2 = Dx2/ep
    Ex3 = Dx3/ep

    # Remove ghost cells from conserved and primitive variables.
    ing = (slice(1,-1), slice(1,-1), slice(1,-1))
    Ex1   = Ex1[ing]
    Ex2   = Ex2[ing]
    Ex3   = Ex3[ing]

    # Current density source terms.
    spg = {"magnetic_field_x1"   : 0.0,
           "magnetic_field_x2"   : 0.0,
           "magnetic_field_x3"   : 0.0,
           "electric_d_field_x1" : -sigma*Ex1,
           "electric_d_field_x2" : -sigma*Ex2,
           "electric_d_field_x3" : -sigma*Ex3,}

    # Total physical source terms.
    sp = spg #{k: spg[k] + sp?[k] for k in spg} # For future implementation.

    return sp


def compute_constraint_cleaning_source(
    U      : dict,
    grid   : dict,
    params : dict
    ) -> dict:
    """
    Compute divergence-cleaning source terms for Maxwell constraints.

    Arguments
    ---------
    U      : Conserved variable arrays (with ghost cells).
    grid   : Discrete grid quantities, coordinates, and domain config.
    params : Physical parameters.

    Returns
    -------
    sc : Constraint-cleaning source terms on the physical domain.
    """

    # Unpack conserved variables.
    Bx1 = U["magnetic_field_x1"]
    Bx2 = U["magnetic_field_x2"]
    Bx3 = U["magnetic_field_x3"]
    Dx1 = U["electric_d_field_x1"]
    Dx2 = U["electric_d_field_x2"]
    Dx3 = U["electric_d_field_x3"]

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
    dx1_lnhx2 = grid["lame"]["derivatives"]["x1"]["ln_hx2"]
    dx1_lnhx3 = grid["lame"]["derivatives"]["x1"]["ln_hx3"]
    dx2_lnhx1 = grid["lame"]["derivatives"]["x2"]["ln_hx1"]
    dx2_lnhx3 = grid["lame"]["derivatives"]["x2"]["ln_hx3"]
    dx3_lnhx1 = grid["lame"]["derivatives"]["x3"]["ln_hx1"]
    dx3_lnhx2 = grid["lame"]["derivatives"]["x3"]["ln_hx2"]

    # Unpack physical parameters.
    rho = params["charge_density"]

    # Cleaning coefficients.
    chi_B = 0.0
    chi_D = 0.0

    # Physical-domain slice.
    ing = (slice(1, -1), slice(1, -1), slice(1, -1))

    # Slices of neighboring cell centers for centered derivatives: i_m = i-1, i_p = i+1.
    im = (slice(None, -2), slice(1, -1), slice(1, -1))
    ip = (slice(2, None),  slice(1, -1), slice(1, -1))
    jm = (slice(1, -1), slice(None, -2), slice(1, -1))
    jp = (slice(1, -1), slice(2, None),  slice(1, -1))
    km = (slice(1, -1), slice(1, -1), slice(None, -2))
    kp = (slice(1, -1), slice(1, -1), slice(2, None))

    # Magnetic field/Electric displacement field gradient using centered finite differences.
    dBx1_dx1 = (Bx1[ip]-Bx1[im])/(2.0*Lx1)
    dBx2_dx2 = (Bx2[jp]-Bx2[jm])/(2.0*Lx2)
    dBx3_dx3 = (Bx3[kp]-Bx3[km])/(2.0*Lx3)
    dDx1_dx1 = (Dx1[ip]-Dx1[im])/(2.0*Lx1)
    dDx2_dx2 = (Dx2[jp]-Dx2[jm])/(2.0*Lx2)
    dDx3_dx3 = (Dx3[kp]-Dx3[km])/(2.0*Lx3)

    # Remove ghost cells from conserved variables.
    Bx1c, Bx2c, Bx3c = Bx1[ing], Bx2[ing], Bx3[ing]
    Dx1c, Dx2c, Dx3c = Dx1[ing], Dx2[ing], Dx3[ing]

    # Magnetic field/Electric displacement field divergence.
    divB  = (dBx1_dx1 + dBx2_dx2 + dBx3_dx3 + Bx1c*(dx1_lnhx2+dx1_lnhx3)/hx1 + Bx2c*(dx2_lnhx1+dx2_lnhx3)/hx2 + Bx3c*(dx3_lnhx1+dx3_lnhx2)/hx3)
    divD  = (dDx1_dx1 + dDx2_dx2 + dDx3_dx3 + Dx1c*(dx1_lnhx2+dx1_lnhx3)/hx1 + Dx2c*(dx2_lnhx1+dx2_lnhx3)/hx2 + Dx3c*(dx3_lnhx1+dx3_lnhx2)/hx3)

    # Residuals to clean.
    rB = divB
    rD = divD - rho

    # Gradients of residuals (component form), using padded residual for centered stencils.
    rB_g = np.pad(rB, 1, mode="edge")
    rD_g = np.pad(rD, 1, mode="edge")

    grBx1 = (rB_g[ip]-rB_g[im])/(2.0*Lx1)
    grBx2 = (rB_g[jp]-rB_g[jm])/(2.0*Lx2)
    grBx3 = (rB_g[kp]-rB_g[km])/(2.0*Lx3)

    grDx1 = (rD_g[ip]-rD_g[im])/(2.0*Lx1)
    grDx2 = (rD_g[jp]-rD_g[jm])/(2.0*Lx2)
    grDx3 = (rD_g[kp]-rD_g[km])/(2.0*Lx3)

    # Cleaning source terms.
    sc = {"magnetic_field_x1"   : -chi_B*grBx1,
          "magnetic_field_x2"   : -chi_B*grBx2,
          "magnetic_field_x3"   : -chi_B*grBx3,
          "electric_d_field_x1" : -chi_D*grDx1,
          "electric_d_field_x2" : -chi_D*grDx2,
          "electric_d_field_x3" : -chi_D*grDx3}

    return sc


def compute_dt(
    U      : dict,
    grid   : dict,
    params : dict,
    cfl    : float
    ) -> float:
    """
    Compute a CFL-limited time step from the current simulation state.

    Arguments
    ---------
    U      : Conserved variable arrays (with ghost cells).
    grid   : Discrete grid quantities, coordinates, and domain config.
    params : Physical parameters.
    cfl    : Courant–Friedrichs–Lewy number.

    Returns
    -------
    dt : CFL-limited time-step.
    """

    # Unpack conserved variables.
    Bx1 = U["magnetic_field_x1"]
    Bx2 = U["magnetic_field_x2"]
    Bx3 = U["magnetic_field_x3"]

    # Unpack physical lengths.
    L = grid["cell"]["geometry"]["length"]
    Lx1 = L["x1"]
    Lx2 = L["x2"]
    Lx3 = L["x3"]

    B = {"x1": Bx1,
         "x2": Bx2,
         "x3": Bx3}

    # Characteristic wave velocities in the fluid frame | shape (nx1+2, nx2+2, nx3+2).
    kappa = compute_characteristic_velocity(U, params)

    # Characteristic wave velocities in the lab frame (lambda_k = mu_k) along each direction | shape (nx1, nx2, nx3).
    ing = (slice(1,-1), slice(1,-1), slice(1,-1))
    lambda_ = {d: {w: kappa[w][ing] for w in kappa} for d in B}

    # Maximum absolute characteristic velocities along each direction | shape (nx1, nx2, nx3).
    max_lambda = {d: np.maximum.reduce([np.abs(l) for l in lambda_[d].values()]) for d in lambda_}

    # Maximum local CFL rate | shape (nx1, nx2, nx3) etc.
    cfl_rate_loc_c = np.maximum.reduce([max_lambda["x1"]/Lx1, max_lambda["x2"]/Lx2, max_lambda["x3"]/Lx3])

    # Maximum domain-wide CFL rate.
    cfl_rate_c = np.max(cfl_rate_loc_c)
    cfl_rate   = cfl_rate_c

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
    Compute the right-hand side of the 3D Maxwell's equations.

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
    slope_limiter  = space_schemes["slope_limiter"]
    riemann_solver = space_schemes["riemann_solver"]

    # Conserved variables on each left/right interface along each direction | shape (nx1+1, nx2, nx3) etc.
    U_L, U_R = space_schemes["reconstructor"](U, L, slope_limiter)

    # Convective (hyperbolic) fluxes on each left/right interface along each direction | shape (nx1+1, nx2, nx3) etc.
    Fc_L = {"x1": compute_convective_flux(U_L["x1"], params, n["x1"]),
            "x2": compute_convective_flux(U_L["x2"], params, n["x2"]),
            "x3": compute_convective_flux(U_L["x3"], params, n["x3"])}

    Fc_R = {"x1": compute_convective_flux(U_R["x1"], params, n["x1"]),
            "x2": compute_convective_flux(U_R["x2"], params, n["x2"]),
            "x3": compute_convective_flux(U_R["x3"], params, n["x3"])}

    # Characteristic wave velocities in the fluid frame on each left/right interface along each direction | shape (nx1+1, nx2, nx3) etc.
    mu_L  = {"x1": compute_characteristic_velocity(U_L["x1"], params),
             "x2": compute_characteristic_velocity(U_L["x2"], params),
             "x3": compute_characteristic_velocity(U_L["x3"], params)}

    mu_R  = {"x1": compute_characteristic_velocity(U_R["x1"], params),
             "x2": compute_characteristic_velocity(U_R["x2"], params),
             "x3": compute_characteristic_velocity(U_R["x3"], params)}

    # Characteristic wave velocities in the lab frame (lambda_k = mu_k) on each left/right interface along each direction | shape (nx1+1, nx2, nx3) etc.
    lambda_L = {d: {w: mu_L[d][w] for w in mu_L[d]} for d in mu_L}
    lambda_R = {d: {w: mu_R[d][w] for w in mu_R[d]} for d in mu_R}

    # Convective (hyperbolic) fluxes along each direction | shape (nx1+1, nx2, nx3) etc.
    Fc = {"x1": riemann_solver(Fc_L["x1"], Fc_R["x1"], U_L["x1"], U_R["x1"], lambda_L["x1"], lambda_R["x1"]),
          "x2": riemann_solver(Fc_L["x2"], Fc_R["x2"], U_L["x2"], U_R["x2"], lambda_L["x2"], lambda_R["x2"]),
          "x3": riemann_solver(Fc_L["x3"], Fc_R["x3"], U_L["x3"], U_R["x3"], lambda_L["x3"], lambda_R["x3"])}

    # Total fluxes | shape (nx1+1, nx2, nx3) etc.
    F = {"x1": {k: Fc["x1"][k] for k in Fc["x1"]},
         "x2": {k: Fc["x2"][k] for k in Fc["x2"]},
         "x3": {k: Fc["x3"][k] for k in Fc["x3"]}}

    # Geometric source terms | shape (nx1, nx2, nx3).
    sg = compute_geometric_source(U, params, grid)
    
    # Physical source terms | shape (nx1, nx2, nx3).
    sp = compute_physical_source(U, params)

    # Cleaning source termes for Maxwell-Gauss and Maxwell-Thomson | shape (nx1, nx2, nx3).
    sc = compute_constraint_cleaning_source(U, grid, params)

    # Total source terms | shape (nx1, nx2, nx3).
    src = {k: sg[k] + sp[k] + sc[k] for k in sg}

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
    rhs = {"magnetic_field_x1"   : -((F["x1"]["magnetic_field_x1"]*S["x1"])[inf]   - (F["x1"]["magnetic_field_x1"]*S["x1"])[ipf]   + (F["x2"]["magnetic_field_x1"]*S["x2"])[jnf]   - (F["x2"]["magnetic_field_x1"]*S["x2"])[jpf]   + (F["x3"]["magnetic_field_x1"]*S["x3"])[knf]   - (F["x3"]["magnetic_field_x1"]*S["x3"])[kpf] )/V   + src["magnetic_field_x1"],
           "magnetic_field_x2"   : -((F["x1"]["magnetic_field_x2"]*S["x1"])[inf]   - (F["x1"]["magnetic_field_x2"]*S["x1"])[ipf]   + (F["x2"]["magnetic_field_x2"]*S["x2"])[jnf]   - (F["x2"]["magnetic_field_x2"]*S["x2"])[jpf]   + (F["x3"]["magnetic_field_x2"]*S["x3"])[knf]   - (F["x3"]["magnetic_field_x2"]*S["x3"])[kpf] )/V   + src["magnetic_field_x2"],
           "magnetic_field_x3"   : -((F["x1"]["magnetic_field_x3"]*S["x1"])[inf]   - (F["x1"]["magnetic_field_x3"]*S["x1"])[ipf]   + (F["x2"]["magnetic_field_x3"]*S["x2"])[jnf]   - (F["x2"]["magnetic_field_x3"]*S["x2"])[jpf]   + (F["x3"]["magnetic_field_x3"]*S["x3"])[knf]   - (F["x3"]["magnetic_field_x3"]*S["x3"])[kpf] )/V   + src["magnetic_field_x3"],
           "electric_d_field_x1" : -((F["x1"]["electric_d_field_x1"]*S["x1"])[inf] - (F["x1"]["electric_d_field_x1"]*S["x1"])[ipf] + (F["x2"]["electric_d_field_x1"]*S["x2"])[jnf] - (F["x2"]["electric_d_field_x1"]*S["x2"])[jpf] + (F["x3"]["electric_d_field_x1"]*S["x3"])[knf] - (F["x3"]["electric_d_field_x1"]*S["x3"])[kpf] )/V + src["electric_d_field_x1"],
           "electric_d_field_x2" : -((F["x1"]["electric_d_field_x2"]*S["x1"])[inf] - (F["x1"]["electric_d_field_x2"]*S["x1"])[ipf] + (F["x2"]["electric_d_field_x2"]*S["x2"])[jnf] - (F["x2"]["electric_d_field_x2"]*S["x2"])[jpf] + (F["x3"]["electric_d_field_x2"]*S["x3"])[knf] - (F["x3"]["electric_d_field_x2"]*S["x3"])[kpf] )/V + src["electric_d_field_x2"],
           "electric_d_field_x3" : -((F["x1"]["electric_d_field_x3"]*S["x1"])[inf] - (F["x1"]["electric_d_field_x3"]*S["x1"])[ipf] + (F["x2"]["electric_d_field_x3"]*S["x2"])[jnf] - (F["x2"]["electric_d_field_x3"]*S["x2"])[jpf] + (F["x3"]["electric_d_field_x3"]*S["x3"])[knf] - (F["x3"]["electric_d_field_x3"]*S["x3"])[kpf] )/V + src["electric_d_field_x3"]}

    return rhs
