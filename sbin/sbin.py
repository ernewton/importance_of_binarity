# sbin/sbin.py

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from sbin import parameters

def model_binary_separations(n_stars, trunc_low=0.01, trunc_high=50000.):
    
    # set up parameters for the binary star distribution of a
    logmu = np.log10(40) # AU
    logsigma = 1.5 # log10(AU)
    trunc_low_loga = np.log10(trunc_low) # 0.01 is min shown
    trunc_high_loga = np.log10(trunc_high) # 5*10^4 AU is max a shown in Offner plot
    trunc_high = (trunc_high_loga-logmu)/logsigma # truncates at no. of sigma from loc
    trunc_low = (trunc_low_loga-logmu)/logsigma # truncates at no. of sigma from loc

    a_values = truncnorm.rvs(trunc_low, trunc_high,
                                    loc=logmu, scale=logsigma,
                                    size=n_stars)
    
    return 10.**a_values


def suppression_factor_50(a_values):
    results = np.ones_like(a_values)*0.5 # 50% for everything
    results[a_values > 200] = 1. # but only <200 au
    return results

def suppression_factor_simple(a_values):
    a_inner_true = 15  # AU (suppression 100%)
    a_outer_true = 100  # AU (suppression 0%)
    results = (np.log10(a_values) - np.log10(a_inner_true)) / (np.log10(a_outer_true) - np.log10(a_inner_true))
    return np.clip(results, a_min=0, a_max=1)

def suppression_factor(a_values):
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



def suppression_simulation(planets_cat, separations=None, 
                           sup_function = suppression_factor,
                           sup_type = 'planets',
                           join_on='KOI', prad_col='koi_prad'):
    """
    Simulate the suppression of planet formation by stellar binaries. Return 
    the original (single-star host) planet population from ``planets_cat'',
    suppressed as if all the host stars are binaries with semi-major axes 
    drawn from the distribution defined by ``suppression_cat''.

    Parameters
    ----------
    planets_cat : pandas.DataFrame
        Table of single‑star host planets (must contain the radius column 
        given by ``prad_col`` and the join column given by ``join_on``).
    separations : np.array, sort of optional
        Array-like list of binary star separations from which to draw, in 
        units of astronomical units.
    sup_function : function name
        Function to do the planet suppressing
    sup_type : str, optional
        Suppress planets or planetary systems? (default ``'planets'``).
    join_on : str, optional
        Column name used to merge the two catalogs (default ``'KOI'``).
    prad_col : str, optional
        Column name for planet radius in ``planets_cat`` (default ``'Rp'``).

    Returns
    -------
    planet_radius : np.ndarray
        Radii of planets that survive the suppression process.
    planet_counts : pandas.DataFrame
        Number of surviving planets per KOI (columns ``'KOI'`` and 
        ``'n_planets'``).
    frac_super_earths :
        Fraction of surviving planets with radius < ``rad_valley``.
    frac_multiplanet : float
        Fraction of surviving KOIs that are multiplanet systems.
    """
    
    # ----------------------------------------------
    # Set-up
    # ----------------------------------------------

    # Number of UNIQUE stellar hosts
    d = {'KOI': planets_cat['KOI'].unique()}
    suppression_cat = pd.DataFrame(data=d)

    n_stars = len(suppression_cat)
    n_planets = len(planets_cat)

    # ----------------------------------------------
    # Assign a binary star to each planetary system
    # ----------------------------------------------   
    
    # Randomly draw a binary separation for each stellar host
    if separations is not None:
        random_separations = np.random.choice(separations, 
                                          size=n_stars, replace=True)     
        # Add the jitter to the sampled values
        error_std = 0.1 * random_separations
        random_error = np.random.normal(loc=0.0, scale=error_std)
        random_separations = random_separations + random_error

    else:
        random_separations = model_binary_separations(n_stars,
                                          trunc_low = 0.1, trunc_high = 100.)
        

    # Suppress planet formation (per STAR)
    suppression_cat['a_values'] = random_separations
    suppression_cat['my_factor'] = sup_function(suppression_cat['a_values'])
    
    # ----------------------------------------------
    # Suppress planetary SYSTEM
    # ----------------------------------------------   
    random_vals = np.random.rand(n_stars)  # uniform random [0,1) to compare to my_factor
    suppression_cat['system_exists'] = random_vals < suppression_cat['my_factor']
  
    # Match STAR suppression to each PLANET
    # ``realization'' will hold the outcome of this simulation
    realization = planets_cat.merge(suppression_cat, on=join_on)
    realization['planet_exists'] = np.ones(n_planets, dtype=bool) 

    # ----------------------------------------------
    # Suppress planets
    # ----------------------------------------------   

    # Condition 1: If Rp < 1.8, planet formation is not suppressed
    realization.loc[realization[prad_col] <= parameters.radius_valley, 'planet_exists'] = True

    # Condition 2: If Rp >= 1.8, planet formation probabilistically suppressed
    mask = (realization[prad_col] > parameters.radius_valley)
    random_vals = np.random.rand(n_planets)  # uniform random [0,1) to compare to my_factor
    realization.loc[mask, 'planet_exists'] = random_vals[mask] < realization['my_factor'][mask]

    # ----------------------------------------------
    # Determine the properties of the suppressed population
    # ----------------------------------------------   
    
    if sup_type=='planets':
        # Only the planets that still exist
        obs = realization.loc[realization['planet_exists'] == 1].copy()
    else:
        # Only the systems that still exist
        obs = realization.loc[realization['system_exists'] == 1].copy()
        
    planet_radius = obs[prad_col]
   
    n_super_earths_after = float(len(planet_radius[planet_radius < 1.8]))
    n_planets_after = float(len(planet_radius))
    
    planet_counts = obs.groupby(['KOI','a_values']).size().reset_index(name='n_planets')    
    mtps = float(len(planet_counts.loc[planet_counts['n_planets']>1]))
    stps = float(len(planet_counts.loc[planet_counts['n_planets']==1]))
    
    #print(realization[['KOI','system_exists','planet_exists']])
    #print(n_super_earths_after, n_planets_after, mtps, stps)
    
    return(obs, planet_counts, 
           n_super_earths_after/n_planets_after, 
           mtps/(mtps+stps))