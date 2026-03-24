"""
File   : io.py
Author : Nathan ZIMNIAK
Date   : 2026-03-10
-----------------
Snapshot saving utilities for the finite-volume solver.
Saves conserved variables and setup parameters to HDF5 format.
"""

import h5py
import os


def save_snapshot(
    U         : dict,
    setup     : dict,
    time      : float,
    iteration : int,
    save_dir  : str
    ) -> None:
    """
    Save a simulation snapshot to an HDF5 file.

    Arguments
    ---------
    U         : Conserved variable arrays (with ghost cells).
    setup     : Full setup configuration.
    time      : Current simulation time.
    iteration : Current iteration number.
    save_dir  : Output directory path.

    Returns
    -------
    None
    """

    # Create output directory if it does not exist.
    os.makedirs(save_dir, exist_ok=True)

    with h5py.File(f"{save_dir}/snapshot_{iteration:06d}.h5", "w") as f:

        # Create HDF5 groups.
        grp_U = f.create_group("U")
        grp_setup  = f.create_group("setup")

        # Coordinate system.
        grp_setup.attrs["coordinate_system"] = setup["coordinate_system"]

        # Grid configuration.
        for key, val in setup["grid_config"].items():
            grp_setup.attrs[key] = val
        
        # Physical parameters.
        for key, val in setup["physical_parameters"].items():
            grp_setup.attrs[key] = val

        # Time.
        f.attrs["time"]      = time
        f.attrs["iteration"] = iteration

        # Conserved variables arrays.
        for var_name, array in U.items():
            grp_U.create_dataset(var_name, data=array)


def load_snapshot(
    filepath : str
    ) -> tuple:
    """
    Load a simulation snapshot from an HDF5 file.

    Arguments
    ---------
    filepath : Path to the HDF5 snapshot file.

    Returns
    -------
    U         : Conserved variable arrays (with ghost cells).
    setup     : Full setup configuration.
    time      : Simulation time at snapshot.
    iteration : Iteration number at snapshot.
    """

    with h5py.File(filepath, "r") as f:

        # Setup parameters (coordinate system, grid configuration, physical parameters).
        setup = {key : val for key, val in f["setup"].attrs.items()}

        # Time.
        time      = f.attrs["time"]
        iteration = f.attrs["iteration"]

        # Conserved variables arrays.
        U = {var_name : f["U"][var_name][:] for var_name in f["U"]}

    return U, setup, time, iteration
