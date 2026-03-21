# --------------------------------------------------------------
#  file: sbin/run_helpers_simple.py
# --------------------------------------------------------------
from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

# Import the class and its result container that live in your package.
# Adjust the relative import if you place the helper in a different
# sub‑package.
from .SuppressionSimulator import SuppressionSimulator, SuppressionResult


def run_trials(
    simulator: SuppressionSimulator,
    n_trials: int,
    max_a_draw: float = 300.,
) -> Tuple[SuppressionResult, Optional[List[SuppressionResult]]]:
    """
    Run `n_trials` independent realizations of the same ``SuppressionSimulator``
    and return the *mean* of the scalar outputs.

    Parameters
    ----------
    simulator : SuppressionSimulator
        A fully configured simulator (catalog, suppression function, etc.).
        The same instance is reused for all trials; its RNG will keep moving
        forward.

    n_trials : int
        Number of Monte‑Carlo draws to take.

    Returns
    -------
    mean_result : SuppressionResult
        Aggregated result: the two fractions are the arithmetic means across
        all trials, and the attached DataFrames are the concatenation of the
        per‑trial DataFrames.

    all_results : list[SuppressionResult] | None
        The raw per‑trial results if ``return_all`` is True; otherwise ``None``.
    """
    # ------------------------------------------------------------------
    # Run the trials the lazy way
    # ------------------------------------------------------------------
    results_systems: List[SuppressionResult] = []
    results_planets: List[SuppressionResult] = []

    for _ in range(n_trials):
        simulator.run(max_a_draw=max_a_draw)
        results_systems.append(simulator.get_results(suppression_style="systems"))
        results_planets.append(simulator.get_results(suppression_style="planets"))
        
    return _stack(results_systems), _stack(results_planets)


def _stack(results):
    
    
    # ------------------------------------------------------------------
    # Compute the means of the scalar quantities
    # ------------------------------------------------------------------
    #mean_frac_super = np.mean([r.frac_super_earths for r in results])
    #mean_frac_multi  = np.mean([r.frac_multiplanet   for r in results])

    # ------------------------------------------------------------------
    # Concatenate the DataFrames so the return type matches the class
    # ------------------------------------------------------------------
    all_planets = pd.concat(
        [r.survived_planets for r in results],
        ignore_index=True,
    )
    all_counts = pd.concat(
        [r.survived_systems for r in results],
        ignore_index=True,
    )

    ssp_by_planet = [r.survived_periods for r in results]

    # ------------------------------------------------------------------
    # Build trial-by-trial results
    # ------------------------------------------------------------------
    #ssm_by_planet = [r.survived_semimajor for r in results]
    #spr_by_planet = [r.survived_radii for r in results]
    #spp_by_planet = [r.survived_periods for r in results]
    
    # ------------------------------------------------------------------
    # Build the aggregated result object
    # ------------------------------------------------------------------
    return SuppressionResult(
        survived_planets=all_planets, # concatenation of all planets from all trials
        survived_systems=all_counts, # concatenation of all systems from all trials
        survived_periods=ssp_by_planet # concatenation of all periods from all trials
    )
