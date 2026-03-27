<h2 align="center">Multiphysics FVM</h2>

<br>

<p align="center">
    A modular and extensible <strong>3D finite-volume framework</strong> for structured grids (cartesian, cylindrical, spherical).
</p>

<br>

<p align="center">
    <img src="https://github.com/nathanzimniak/multiphysics-fvm/blob/main/img/navier_stokes.png" width="800">
    <img src="https://github.com/nathanzimniak/multiphysics-fvm/blob/main/img/heat.png" width="800">
    <img src="https://github.com/nathanzimniak/multiphysics-fvm/blob/main/img/maxwell.png" width="800">
</p>

<br>

Written in **Python** with the *NumPy* library, the codebase prioritizes readability, and simplicity over raw performance. It is designed to support **multiple physics modules** through a common solver interface. At the current stage, implemented modules are primarily run one at a time (uncoupled).

Currently implemented physics module: `navier_stokes`, `heat`, `maxwell`.

---

### Numerical Methods

The discretization follows a finite-volume approach. Convective (*hyperbolic*) terms are treated using Godunov-type finite-volume schemes with approximate Riemann solvers, while diffusive (*parabolic*) terms are handled with centered schemes. The numerical pipeline is decomposed into modular building blocks, each of which can be independently selected and extended. The currently implemented methods are summarized below:

<table>
  <tbody>
    <tr>
      <td>Riemann solvers</td>
      <td><code>Rusanov</code>, <code>HLL</code></td>
    </tr>
    <tr>
      <td>Diffusive solvers</td>
      <td><code>Central</code></td>
    </tr>
    <tr>
      <td>Reconstructors</td>
      <td><code>Piecewise constant</code>, <code>MUSCL</code></td>
    </tr>
    <tr>
      <td>Slope limiters</td>
      <td><code>Minmod</code>, <code>Monotonized central</code>, <code>van Leer</code></td>
    </tr>
    <tr>
      <td>Time integrators</td>
      <td><code>Euler</code>, <code>RK3</code>, <code>SSPRK3</code>, <code>RK4</code></td>
    </tr>
  </tbody>
</table>

---

### Getting Started

Clone the repository:

```
git clone https://github.com/nathanzimniak/multiphysics-fvm.git
```

Install dependencies:

```
pip install -r requirements.txt
```

Simulations are launched via the command line using ```main.py``` and a setup file from the ```setups/``` directory:

```
python main.py --setup kelvin_helmholtz
```

Outputs are saved in **HDF5** format. You can create new simulations by adding configuration files in the ```setups/``` folder.

---

### Contributing

Contributions are welcome. Feel free to open an issue or submit a pull request if you find a bug, want to add a numerical method, or implement a new physical module.
