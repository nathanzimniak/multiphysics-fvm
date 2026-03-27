## Multiphysics FVM

<br>

A modular and extensible **3D finite-volume framework** for structured grids (cartesian, cylindrical, spherical).

It is designed to support **multiple physics modules** through a common solver interface. At the current stage, implemented modules are primarily run one at a time (uncoupled). However, the architecture is coupling-ready: tightly coupled multiphysics can be added without rewriting the solver core. Written in pure Python with NumPy, the codebase prioritizes readability, maintainability, and extensibility over raw performance.

Currently implemented physics module: `navier_stokes`, `heat`, `maxwell`.

<p align="center">
  <img src="https://github.com/nathanzimniak/multiphysics-fvm/blob/main/img/navier_stokes.png">
  <img src="https://github.com/nathanzimniak/multiphysics-fvm/blob/main/img/heat.png">
  <img src="https://github.com/nathanzimniak/multiphysics-fvm/blob/main/img/maxwell.png">
</p>

---

### Numerical Methods

The discretization follows a finite-volume approach. Convective (hyperbolic) terms are treated using Godunov-type finite-volume schemes with approximate Riemann solvers, while diffusive (parabolic) terms are handled with centered schemes. The numerical pipeline is decomposed into modular building blocks, each of which can be independently selected and extended. The currently implemented methods are summarized below:

<table>
  <tbody>
    <tr>
      <td>Riemann solver</td>
      <td><code>Rusanov</code>, <code>HLL</code></td>
    </tr>
    <tr>
      <td>Diffusive solver</td>
      <td><code>Central</code></td>
    </tr>
    <tr>
      <td>Reconstruction</td>
      <td><code>Piecewise constant</code>, <code>MUSCL</code></td>
    </tr>
    <tr>
      <td>Slope limiter</td>
      <td><code>Minmod</code>, <code>Monotonized central</code>, <code>van Leer</code></td>
    </tr>
    <tr>
      <td>Time integration</td>
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

You can create new simulations by adding configuration files in the ```setups/``` folder.

---

### Contributing

Contributions are welcome. Feel free to open an issue or submit a pull request if you find a bug, want to add a numerical method, or implement a new physical module.
