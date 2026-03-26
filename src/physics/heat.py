"""
File   : heat.py
Author : Nathan ZIMNIAK
Date   : 2026-03-25
-----------------
Right-hand side and CFL time step for the 3D heat equation.
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
    U = {"internal_energy_density" : np.zeros(sg)}

    return U


def compute_closure(
    U      : dict,
    grid   : dict,
    params : dict
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
    C : Closure variables (heat flux) with ghost cells.
    """

    # Unpack conserved variables.
    rhocpT = U["internal_energy_density"]

    # Unpack physical lengths.
    L = grid["cell"]["geometry"]["length"]
    Lx1 = L["x1"]
    Lx2 = L["x2"]
    Lx3 = L["x3"]
    
    # Unpack physical parameters.
    cp  = params["specific_heat_capacity"]
    k   = params["thermal_conductivity"]
    rho = params["mass_density"]

    # Primitive variables.
    T = rhocpT/(rho*cp)

    # Slices of neighboring cell centers for centered derivatives: i_m = i-1, i_p = i+1.
    im = (slice(None, -2), slice(1, -1), slice(1, -1)) # [:-2, 1:-1, 1:-1]
    ip = (slice(2, None), slice(1, -1), slice(1, -1))  # [2:, 1:-1, 1:-1]
    jm = (slice(1, -1), slice(None, -2), slice(1, -1)) # [1:-1, :-2, 1:-1]
    jp = (slice(1, -1), slice(2, None), slice(1, -1))  # [1:-1, 2:, 1:-1]
    km = (slice(1, -1), slice(1, -1), slice(None, -2)) # [1:-1,1:-1,:-2]
    kp = (slice(1, -1), slice(1, -1), slice(2, None))  # [1:-1,1:-1,2:]

    # Temperature gradient using centered finite differences.
    dT_dx1 = (T[ip]-T[im])/(2*Lx1)
    dT_dx2 = (T[jp]-T[jm])/(2*Lx2)
    dT_dx3 = (T[kp]-T[km])/(2*Lx3)

    # Heat flux vector from Fourier's law.
    phix1 = -k*dT_dx1
    phix2 = -k*dT_dx2
    phix3 = -k*dT_dx3

    # Embed heat flux vector components with ghost cells.
    phix1, phix2, phix3 = [np.pad(phi, 1) for phi in [phix1, phix2, phix3]]

    # Closure variables.
    C = {"phix1" : phix1,
         "phix2" : phix2,
         "phix3" : phix3}
        
    return C


def compute_diffusive_flux(
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

    # Unpack closure variables.
    phix1 = C["phix1"]
    phix2 = C["phix2"]
    phix3 = C["phix3"]

    # Unpack normal vector components.
    nx1 = n[0]
    nx2 = n[1]
    nx3 = n[2]

    # Diffusive fluxes.
    Fd = {"internal_energy_density" : {"x1" : phix1, "x2" : phix2, "x3" : phix3}}

    # Normal diffusive fluxes.
    Fd_n = {"internal_energy_density" : nx1*Fd["internal_energy_density"]["x1"] + nx2*Fd["internal_energy_density"]["x2"] + nx3*Fd["internal_energy_density"]["x3"]}

    return Fd_n


def compute_physical_source(
    params : dict
    ) -> dict:
    """
    Compute the volumetric heat source terms.

    Arguments
    ---------
    params : Physical parameters.

    Returns
    -------
    sp : Physical source terms on the physical domain.
    """

    # Unpack physical parameters
    q = params["volumetric_heat_source"]

    # Volumetric heat source terms.
    spq = {"internal_energy_density" : q}

    # Total physical source terms.
    sp = spq #{k: spq[k] + sp?[k] for k in spq} # For future implementation.

    return sp


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

    # Unpack physical lengths.
    L = grid["cell"]["geometry"]["length"]
    Lx1 = L["x1"]
    Lx2 = L["x2"]
    Lx3 = L["x3"]
    
    # Unpack physical parameters.
    cp  = params["specific_heat_capacity"]
    k   = params["thermal_conductivity"]
    rho = params["mass_density"]

    # Characteristic diffusivity.
    nu = k/(rho*cp)

    # Maximum local CFL rate | shape (nx1, nx2, nx3) etc.
    cfl_rate_loc_d = 2*nu*(1/Lx1**2 + 1/Lx2**2 + 1/Lx3**2)

    # Maximum domain-wide CFL rate.
    cfl_rate_d = np.max(cfl_rate_loc_d)
    cfl_rate   = cfl_rate_d

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
    Compute the right-hand side of the 3D heat equation.

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
    diffusive_solver = space_schemes["diffusive_solver"]

    # Closure variables | shape (nx1+2, nx2+2, nx3+2).
    C = compute_closure(U, grid, params)

    # Closure variables on each left/right interface for along direction | shape (nx1+1, nx2, nx3) etc.
    C_L, C_R = space_schemes["reconstructor"](C, L, slope_limiter)

    # Diffusive (parabolic) fluxes on each left/right interface along each direction | shape (nx1+1, nx2, nx3) etc.
    Fd_L = {"x1": compute_diffusive_flux(C_L["x1"], n["x1"]),
            "x2": compute_diffusive_flux(C_L["x2"], n["x2"]),
            "x3": compute_diffusive_flux(C_L["x3"], n["x3"])}

    Fd_R = {"x1": compute_diffusive_flux(C_R["x1"], n["x1"]),
            "x2": compute_diffusive_flux(C_R["x2"], n["x2"]),
            "x3": compute_diffusive_flux(C_R["x3"], n["x3"])}

    # Diffusive (parabolic) fluxes along each direction | shape (nx1+1, nx2, nx3) etc.
    Fd = {"x1": diffusive_solver(Fd_L["x1"], Fd_R["x1"]),
          "x2": diffusive_solver(Fd_L["x2"], Fd_R["x2"]),
          "x3": diffusive_solver(Fd_L["x3"], Fd_R["x3"])}

    # Total fluxes | shape (nx1+1, nx2, nx3) etc.
    F = {"x1": {k: Fd["x1"][k] for k in Fd["x1"]},
         "x2": {k: Fd["x2"][k] for k in Fd["x2"]},
         "x3": {k: Fd["x3"][k] for k in Fd["x3"]}}
    
    # Physical source terms | shape (nx1, nx2, nx3).
    sp = compute_physical_source(params)

    # Total source terms | shape (nx1, nx2, nx3).
    src = {k: sp[k] for k in sp}

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
    rhs = {"internal_energy_density"   : -((F["x1"]["internal_energy_density"]*S["x1"])[inf]   - (F["x1"]["internal_energy_density"]*S["x1"])[ipf]   + (F["x2"]["internal_energy_density"]*S["x2"])[jnf]   - (F["x2"]["internal_energy_density"]*S["x2"])[jpf]   + (F["x3"]["internal_energy_density"]*S["x3"])[knf]   - (F["x3"]["internal_energy_density"]*S["x3"])[kpf])/V   + src["internal_energy_density"]}

    return rhs
