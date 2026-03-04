# ----------------------------------------------------------------------
# Public API 
# ----------------------------------------------------------------------
__all__ = ["create_complexity_df","gap_complexity"]


import numpy as np

def create_complexity_df(db):
    
    complexity_df = (db
             .groupby('KOI')
             .apply(lambda g: gap_complexity(g['koi_period'].values) 
                              if len(g) >= 3 else np.nan)
             .reset_index(name='gap_complexity')
             .assign(n_planets=lambda df: db.groupby('KOI').size().values)
            )
    
    return complexity_df
    
    
def _get_cmax(ngaps: int) -> float:
    """
    Return the cmax value that corresponds to the integer N.

    Parameters
    ----------
    ngaps : int
        ngaps (N planets is N = ngaps+1)

    Returns
    -------
    float
        The exact cmax from the table.

    Raises
    ------
    KeyError
        If N is not present in the table.
    TypeError
        If N is not an integer.
    """

    C_MAX = {
        2: 0.106,
        3: 0.212,
        4: 0.291,
        5: 0.350,
        6: 0.398,
        7: 0.437,
        8: 0.469,
        9: 0.497,
    }
    
    if not isinstance(ngaps, int):
        raise TypeError(f"N must be an integer, got {type(N)!r}")

    return C_MAX[ngaps]
 

def gap_complexity(periods):
    """Return C_gap for a Kepler multi‑planet system."""
    periods = np.sort(periods)               # ensure ordering
    N = len(periods)

    if N < 3:                                # need at least two gaps
        raise ValueError("C_gap is defined for N >= 3 planets")
    
    # Log‑period ratios between adjacent planets
    log_ratios = np.log(periods[1:] / periods[:-1])

    # Normalise to obtain p*
    p_star = log_ratios / np.log(periods[-1]/periods[0])

    # Shannon entropy
    entropy = np.sum(p_star * np.log(p_star))

    # Normalise by the maximum possible entropy, ln(N‑1)
    disequilibrium = np.sum( (p_star  - 1/(N-1))**2 )
    
    K = 1./_get_cmax(N-1)

    return -K * entropy * disequilibrium


