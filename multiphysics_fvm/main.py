"""
File   : main.py
Author : Nathan ZIMNIAK
Date   : 2026-03-10
-----------------
Entry point for the finite-volume solver.
Loads the setup, builds the grid, initializes the state, and runs the simulation.
"""

import argparse
import importlib
from src import solver


# Parse command-line arguments.
parser = argparse.ArgumentParser(description="Run a finite-volume simulation setup.")
parser.add_argument("--setup", required=True, help="Setup name (e.g. acoustic_spherical).")
args = parser.parse_args()

# Load the setup from user input.
setup  = importlib.import_module(f"setups.{args.setup}").get_setup()

# Run the simulation.
solver.run(setup)