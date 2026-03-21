# ----------------------------------------------------------------------
#  suppression_simulation.py
# ----------------------------------------------------------------------

import dataclasses
from typing import Callable, Iterable, Optional, List, Union

import numpy as np
import pandas as pd

from .suppression_utilities import model_binary_separations

# ----------------------------------------------------------------------
#  Helper: a tiny container for the results


# ----------------------------------------------------------------------
ArrayOrList = Union[np.ndarray, List[np.ndarray]]

@dataclasses.dataclass(frozen=True) # so it can't be modified once created
class SuppressionResult:
    """All objects produced by a single run of :class:`SuppressionSimulator`."""
    survived_planets: pd.DataFrame          # rows = planets that survive
    survived_systems: pd.DataFrame  # KOI, a_values, n_planets
    survived_periods: ArrayOrList # planet orbital period (d)



# ----------------------------------------------------------------------
#  Main class
# ----------------------------------------------------------------------
class SuppressionSimulator:
    """
    Simulate the (probabilistic) suppression of planet formation in binary systems.

    Parameters
    ----------
    planets_cat : pd.DataFrame
        Table of *single‑star* planets. Must contain:
        * a column named ``join_on`` (defaults to ``'KOI'``)
        * a column with planet radii (defaults to ``'koi_prad'``).

    separations : Optional[Iterable[float]], default ``None``
        If supplied, a list/array of binary separations (AU) from which a
        random value will be drawn for each host star.  If ``None`` the
        ``model_binary_separations`` routine is used.

    sup_function : Callable[[pd.Series], pd.Series], default ``suppression_factor``
        Function that maps a binary separation to a “survival probability”.

    join_col : str, default ``'KOI'``
        Column name used to merge the planet catalogue with the binary‑star catalogue.

    prad_col : str, default ``'koi_prad'``
        Column name that stores the planet radius (in Earth radii).

    teff_col : str, default ``'koi_steff'``
        Column name that stores the stellar temperature (in K).

    random_state : Optional[int], default ``None``
        Seed for random number generator
    """

    # ------------------------------------------------------------------
    #  Constructor – just store the arguments, do *no* heavy work yet
    # ------------------------------------------------------------------
    def __init__(
        self,
        planets_cat: pd.DataFrame,
        separations: Optional[Iterable[float]] = None,
        sup_function: Callable[[pd.Series], pd.Series] = None,
        join_col: str = "KOI",
        prad_col: str = "koi_prad",
        teff_col: str = "koi_steff",
        period_col: str = "koi_period",
        random_state: Optional[int] = None,
    ) -> None:
        
        # --------------------------------------------------------------
        #  Publicly visible configuration
        # --------------------------------------------------------------
        self.planets_cat = planets_cat.copy()
        self.separations = np.asarray(separations) if separations is not None else None
        self.sup_function = sup_function or suppression_factor   # returns the first true value
        self.join_col = join_col
        self.prad_col = prad_col
        self.teff_col = teff_col
        self.period_col = period_col
        self.radius_valley = 1.8

        # --------------------------------------------------------------
        #  RNG – keep it in the box
        # --------------------------------------------------------------
        self._rng = np.random.default_rng(random_state)

        # --------------------------------------------------------------
        #  Containers that will be filled during ``run()``
        # --------------------------------------------------------------
        self._suppression_cat: pd.DataFrame | None = None
        self._realization: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    #  Public entry point ------------------------------------------------
    # ------------------------------------------------------------------
    def run(self, max_a_draw=300.) -> SuppressionResult:
        """Execute the full simulation and return a :class:`SuppressionResult`."""
        self._init_suppression_catalog()
        self._draw_separations(max_a_draw)
        self._compute_factors()
        self._pick_surviving_systems()
        self._merge_catalogs()
        self._pick_surviving_planets()

    # ------------------------------------------------------------------
    #  Private helpers – one for each logical block of the original function
    # ------------------------------------------------------------------

    def _init_suppression_catalog(self) -> None:
        """Create a DataFrame with one row per *unique* host star."""
        tmp = (
            self.planets_cat
            .drop_duplicates(subset=self.join_col, keep='first')
            .reset_index(drop=True)
        )
        self._suppression_cat = tmp[[self.join_col, self.teff_col]]

    def _draw_separations(self, max_a_draw=300.) -> None:
        """Assign a binary separation to every host star (with optional jitter)."""
        assert self._suppression_cat is not None, "Catalog not initialised"

        n_stars = len(self._suppression_cat)

        if self.separations is not None:
            # Random draw from the user‑supplied list
            raw = self._rng.choice(self.separations, size=n_stars, replace=True)

            # Add 10 % Gaussian jitter (the same recipe that was in the original code)
            jitter_std = 0.1 * raw
            jitter = self._rng.normal(loc=0.0, scale=jitter_std)
            drawn = raw + jitter
        else:
            # Fall back to the default model (you must have imported it)
            drawn = model_binary_separations(
                n_stars, trunc_low=1., trunc_high=max_a_draw, rng=self._rng
            )

        self._suppression_cat["a_values"] = drawn

    def _compute_factors(self) -> None:
        """Apply the user‑supplied suppression function to the separations."""
        assert self._suppression_cat is not None, "Catalog not initialised"
        self._suppression_cat["sup_factor"] = self.sup_function(self._suppression_cat["a_values"], 
            star_teff = self._suppression_cat[self.teff_col]
        )

    def _pick_surviving_systems(self) -> None:
        """Draw a uniform random number for each system and decide whether the whole system survives."""
        assert self._suppression_cat is not None, "Catalog not initialised"
        n_stars = len(self._suppression_cat)
        rand = self._rng.random(n_stars)                     # uniform in [0, 1)
        self._suppression_cat["system_exists"] = rand < self._suppression_cat["sup_factor"]

    def _merge_catalogs(self) -> None:
        """Attach the binary‑star information to every planet row."""
        assert self._suppression_cat is not None, "Catalog not initialised"
        self._realization = self.planets_cat.merge(
            self._suppression_cat, on=self.join_col, how="left"
        )
        # Initialise the planet‑existence column
        self._realization["planet_exists"] = True

    def _pick_surviving_planets(self) -> None:
        """Apply the radius‑dependent suppression logic."""
        assert self._realization is not None, "Realisation not built yet"

        n_planets = len(self.planets_cat)
        rand = self._rng.random(n_planets)
        prob = self._realization["sup_factor"]   
        self._realization["planet_exists"] = rand < prob.values

    def get_results(self, suppression_style='systems', verbose=False) -> SuppressionResult:
        """Collect the final data frames and the derived statistics."""
        assert self._realization is not None, "Realisation not built yet"

        # --------------------------------------------------------------
        #  Choose what we keep depending on sup_type
        # --------------------------------------------------------------
        if suppression_style == "planets":
            obs = self._realization[self._realization["planet_exists"]].copy()
        elif suppression_style == "systems":
            obs = self._realization[self._realization["system_exists"]].copy()
        else:
            raise
            
        # --------------------------------------------------------------
        #  Properties of the survivors
        # --------------------------------------------------------------
        planet_radius = obs[self.prad_col]
        planet_period = obs[[self.join_col,self.period_col]]

        # --------------------------------------------------------------
        #  Fraction of “super‑Earths” (R < radius_valley) among the survivors
        # --------------------------------------------------------------
        n_super = (planet_radius < self.radius_valley).sum()
        n_total = len(planet_radius)
        frac_super = n_super / n_total if n_total > 0 else np.nan

        # --------------------------------------------------------------
        #  Build the per‑system counts
        # --------------------------------------------------------------
        planet_counts = (
            obs.groupby([self.join_col, "a_values"])
            .size()
            .reset_index(name="n_planets")
        )

        # --------------------------------------------------------------
        #  Multiplanet / single‑planet fractions (only makes sense when we keep
        #  whole systems; if we kept planets we still report the same numbers)
        # --------------------------------------------------------------
        n_multi = (planet_counts["n_planets"] > 1).sum()
        n_single = (planet_counts["n_planets"] == 1).sum()
        frac_multi = n_multi / (n_multi + n_single) if (n_multi + n_single) > 0 else np.nan
        if verbose:
            print("Number of surviving planets :", len(obs))
            print("Fraction that are super‑Earths :", frac_super)
            print("Fraction of multiplanet systems :", frac_multi)

        # --------------------------------------------------------------
        #  Pack everything in a tidy dataclass
        # --------------------------------------------------------------
        return SuppressionResult(
            survived_planets=obs,
            survived_systems=planet_counts,
            survived_periods=planet_period
        )
