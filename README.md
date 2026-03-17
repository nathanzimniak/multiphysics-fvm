## Multiphysics FVM

<br>

A modular and extensible **3D finite-volume solver** designed for **multiphysics simulations** on structured grids, supporting cartesian, cylindrical, and spherical coordinate systems.

The solver is built around a general finite-volume framework, allowing the integration of multiple physical models without modifying the core infrastructure. Each physics module is self-contained, interchangeable, and easy to extend. Written in pure Python with NumPy, the codebase prioritizes readability, maintainability, and extensibility over raw performance.

Currently implemented physics modules: ```navier_stokes```.

---

### Numerical Methods

The solver uses a finite-volume framework. Convective (hyperbolic) terms can be treated using Godunov-type methods (Riemann solvers), while diffusive (parabolic) terms are handled with centered schemes. The numerical pipeline is decomposed into modular building blocks, each of which can be independently selected and extended. The currently implemented methods are summarized below:

<table>
  <tbody>
    <tr>
      <td><b>Riemann solver</b></td>
      <td><code>Rusanov (local Lax-Friedrichs)</code></td>
    </tr>
    <tr>
      <td><b>Diffusive solver</b></td>
      <td><code>Centered</code></td>
    </tr>
    <tr>
      <td><b>Reconstruction</b></td>
      <td><code>Piecewise constant</code>, <code>MUSCL</code></td>
    </tr>
    <tr>
      <td><b>Slope limiter</b></td>
      <td><code>Minmod</code></td>
    </tr>
    <tr>
      <td><b>Time integration</b></td>
      <td><code>Explicit Euler</code>, <code>RK4</code></td>
    </tr>
  </tbody>
</table>

---

### Getting Started

#### Installation

Clone the repository:

```
git clone https://github.com/nathanzimniak/multiphysics-fvm.git
```

Install dependencies:

```
pip install -r requirements.txt
```

#### Running a Simulation

Simulations are launched via the command line using ```main.py``` and a setup file from the ```setups/``` directory:

```
python main.py --setup acoustic_spherical
```

You can create new simulations by adding configuration files in the ```setups/``` folder.

---

### Contributing

Contributions are welcome. Feel free to open an issue or submit a pull request if you find a bug, want to add a numerical method, or want to implement a new physical module.
