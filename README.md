# Multiphysics FVM

A modular, extensible 3D finite-volume solver designed for multiphysics simulations on structured grids, supporting cartesian, cylindrical, and spherical coordinate systems.

Built around a general finite-volume framework, the code can accommodate a wide range of physical models without modifying the core infrastructure. Each physics module is fully self-contained and interchangeable, making it straightforward to implement and switch between different physical models.

Currently shipped with a compressible Navier-Stokes module, available numerical schemes include the Rusanov (local Lax-Friedrichs) Riemann solver, a centered diffusive solver, MUSCL reconstruction with the minmod slope limiter, and explicit Euler and RK4 time integrators. Reference setups include acoustic wave propagation in spherical coordinates, Kelvin-Helmholtz instability, and Rayleigh-Taylor instability.

Written in pure Python with NumPy, the codebase prioritizes readability, maintainability, and extensibility over raw performance.

## Installation

Clone the repository and install the dependencies:
```bash
git clone https://github.com/username/multiphysics_fvm.git
cd multiphysics_fvm
pip install -r requirements.txt
```

## Running a simulation

Simulations are launched from the command line using `main.py` and the name of a setup file located in the `setups/` directory:
```bash
python main.py --setup acoustic_spherical
```

## Contributing

If you find a bug or have a suggestion, feel free to open an issue.
