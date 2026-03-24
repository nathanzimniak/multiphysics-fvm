"""
File   : slope_limiters.py
Author : Nathan ZIMNIAK
Date   : 2026-03-19
-----------------
Slope-limiters for the finite-volume solver.
Available limiters: minmod, monotonized_central, van_leer.
"""

import numpy as np


def minmod(
    bwd_slope : np.ndarray,
    fwd_slope : np.ndarray
    ) -> np.ndarray:
    """
    Minmod slope limiter: returns the smallest-magnitude admissible slope if the backward and forward slopes have the same sign, zero otherwise.

    Arguments
    ---------
    bwd_slope : Backward slope (from cell i-1 to i).
    fwd_slope : Forward slope (from cell i to i+1).

    Returns
    -------
    limited_slope : Limited slope.
    """

    # Detect whether the two slopes have the same sign: +1 if both positive, -1 if both negative, 0 if opposite signs. 
    same_sign = (np.sign(bwd_slope) + np.sign(fwd_slope))/2.0 

    # Magnitude of the smallest slope (minmod selects the least steep admissible one).
    smaller_slope = np.minimum(np.abs(bwd_slope), np.abs(fwd_slope))

    # Apply the sign test: if slopes have different signs the result becomes zero.
    limited_slope = same_sign*smaller_slope

    return limited_slope


def monotonized_central(
    bwd_slope : np.ndarray,
    fwd_slope : np.ndarray
    ) -> np.ndarray:
    """
    Monotonized Central slope limiter: returns the smallest-magnitude admissible slope if all candidates have the same sign, zero otherwise.

    Arguments
    ---------
    bwd_slope : Backward slope (from cell i-1 to i).
    fwd_slope : Forward slope (from cell i to i+1).

    Returns
    -------
    limited_slope : Limited slope.
    """

    # Candidate slopes.
    slope_centered = 0.5*(bwd_slope + fwd_slope)
    slope_bwd      = 2.0*bwd_slope
    slope_fwd      = 2.0*fwd_slope

    # Detect whether the three slopes have the same sign: +1 if all three are positive, -1 if all three are negative, 0 if opposite signs. 
    same_sign = (np.sign(slope_centered) + np.sign(slope_bwd) + np.sign(slope_fwd)) / 3.0

    # Magnitude of the smallest slope (MC selects the least steep admissible one).
    smaller_slope = np.minimum(np.abs(slope_centered), np.minimum(np.abs(slope_bwd), np.abs(slope_fwd)))

    # Apply the sign test: if slopes have different signs the result becomes zero.
    limited_slope = same_sign*smaller_slope

    return limited_slope


def van_leer(
    bwd_slope : np.ndarray,
    fwd_slope : np.ndarray
    ) -> np.ndarray:
    """
    Van Leer slope limiter: returns the harmonic mean of the two slopes if both have the same sign, zero otherwise.

    Arguments
    ---------
    bwd_slope : Backward slope (from cell i-1 to i).
    fwd_slope : Forward slope (from cell i to i+1).

    Returns
    -------
    limited_slope : Limited slope.
    """

    # Detect whether the two slopes have the same sign: +1 if both positive, -1 if both negative, 0 if opposite signs.
    same_sign = (np.sign(bwd_slope) + np.sign(fwd_slope))/2.0

    # Harmonic mean of the two slopes (van Leer selects a smooth limited slope).
    #denom = np.abs(bwd_slope) + np.abs(fwd_slope)
    #harmonic_slope = np.where((np.abs(bwd_slope) + np.abs(fwd_slope)) > 0.0, 2.0*bwd_slope*fwd_slope/(np.abs(bwd_slope) + np.abs(fwd_slope)), 0.0)

    # Safe division only where denom > 0.
    denom = np.abs(bwd_slope) + np.abs(fwd_slope)
    harmonic_slope = np.zeros_like(bwd_slope)
    mask = denom > 0.0
    harmonic_slope[mask] = 2.0*bwd_slope[mask]*fwd_slope[mask] / denom[mask]

    # Apply the sign test: if slopes have opposite signs the result becomes zero.
    limited_slope = same_sign*harmonic_slope

    return limited_slope
