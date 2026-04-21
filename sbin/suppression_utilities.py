# sbin/sbin.py

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from typing import Optional

def model_binary_separations(
    n_stars: int,
    trunc_low: float = 0.01,
    trunc_high: float = 50000.,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return `n` random binary separations drawn from a log‑uniform distribution."""
    
    rng = np.random.default_rng(rng)

    # set up parameters for the binary star distribution of a
    logmu = np.log10(40) # AU
    logsigma = 1.5 # log10(AU)
    trunc_low_loga = np.log10(trunc_low) # 0.01 is min shown
    trunc_high_loga = np.log10(trunc_high) # 5*10^4 AU is max a shown in Offner plot
    trunc_high_sig = (trunc_high_loga-logmu)/logsigma # truncates at no. of sigma from loc
    trunc_low_sig= (trunc_low_loga-logmu)/logsigma # truncates at no. of sigma from loc

    a_values = truncnorm.rvs(trunc_low_sig, trunc_high_sig,
                                    loc=logmu, scale=logsigma,
                                    size=n_stars)

    return 10.**a_values


def a_to_snow(a_values):
    fit = np.poly1d([0.16, 0.])
    return fit(a_values)

def suppression_factor_snow(a_values, **kwargs):

    star_teff = kwargs.get("star_teff")
    if star_teff is None:
        raise ValueError("star_teff required for the snow‑line suppression factor")
        
    diskAU = a_values*0.41 # median of the TB sample
    snowAU = 0.00465*0.5*(star_teff/170.)**2 # for T_eq = 170K
    
    a_inner_true = a_to_snow(10)  # AU (suppression 100%)
    a_outer_true = a_to_snow(200)  # AU (suppression 0%)
    results = (np.log10(diskAU/snowAU) - np.log10(a_inner_true)) / (np.log10(a_outer_true) - np.log10(a_inner_true))

    return np.clip(results, a_min=0, a_max=1)

def suppression_factor_50(a_values, **kwargs):
    results = np.ones_like(a_values)*0.5 # 50% for everything
    results[a_values > 200] = 1. # but only <200 au
    return results

def suppression_factor_simple(a_values, **kwargs):
    a_inner_true = 10  # AU (suppression 100%)
    a_outer_true = 100  # AU (suppression 0%)
    results = (np.log10(a_values) - np.log10(a_inner_true)) / (np.log10(a_outer_true) - np.log10(a_inner_true))
    return np.clip(results, a_min=0, a_max=1)

def add_radius_suppression(S_values, radius, radius_valley):
    #new_S_values = np.copy(S_values)
    #new_S_values[radius >= radius_valley] = S_values[radius >= radius_valley]/2.
    #new_S_values[radius < radius_valley] = S_values[radius < radius_valley]*2.
    
    rmin=0.1
    rmax=4.
    #results = (np.log10(radius) - np.log10(rmin)) / (np.log10(rmax) - np.log10(rmin))
    results = 1 - (radius - rmin) / (rmax - rmin)

    new_S_values = S_values * results
    
    return np.clip(new_S_values, a_min=0, a_max=1)

def suppression_factor_mk21(a_values, **kwargs):
    """
    Return the suppression factor S for a set of orbital semi-major axes.
    
    S is defined as a piece-wise linear function (linear in log space of 
    semi-major axis) by three break points as specified in Moe \& Kratter 
    (2021).

    Parameters
    ----------
    a_values : array‑like
        One‑dimensional collection of semi‑major axes (in astronomical units).
 
    Returns
    -------
    np.ndarray
        An array of the same shape as ``a_values`` containing the suppression
        factor for each input value, expressed as a fraction from [0,1] where
        0 means complete suppression and 1 means no suppression.
    """
    
    # Change to log space for semi-major axis
    log_a = np.log10(np.asarray(a_values)) # Semi-major axes for which to calculate S
    
    # Define the function
    log_points = np.array([np.log10(1), np.log10(10), np.log10(200)]) # break points in log(au)
    S_points = np.array([0, 0.15, 1])  # suppression value at break points
    
    # Apply the formula
    S_val = np.interp(log_a, log_points, S_points)
    
    return np.clip(S_val, a_min=0, a_max=1)


