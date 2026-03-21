import numpy as np
from scipy import stats

def one_multiplicity_arr(counts, max_bin=3):
    """
    Convert a 1‑D array of planet counts per star into a frequency vector.
    (max_bin inclusive, everything above is lumped).
    """
    clipped = np.minimum(counts, max_bin)  
    hist, _ = np.histogram(clipped,
                           bins=np.arange(0.5, max_bin + 1.5, 1))
    return hist.astype(float)

def chisquare_multiplicity(obs_arr, ref_arr):
    obs_counts    = one_multiplicity_arr(obs_arr['n_planets'])               
    ref_counts    = one_multiplicity_arr(ref_arr['n_planets']) 

    # Scale the expected frequencies to the size of the observed sample
    ref_scaled = ref_counts * sum(obs_counts) / sum(ref_counts)

    chi2, p_chi2 = stats.chisquare(f_obs=obs_counts, f_exp=ref_scaled)
    #print(f"χ² = {chi2:.2f},  p‑value = {p_chi2:.4f}")
    return p_chi2

def fishers_multiplicity(obs_arr, ref_arr):
    obs_single = np.sum(obs_arr['n_planets'] == 1)
    obs_multi  = np.sum(obs_arr['n_planets'] >= 2)
    ref_single = np.sum(ref_arr['n_planets'] == 1)
    ref_multi  = np.sum(ref_arr['n_planets'] >= 2)

    stat_fisher, p_fisher = stats.fisher_exact([[obs_single, obs_multi], [ref_single, ref_multi]], alternative='two-sided')
    #print(f"Fisher's exact p-value = {p_fisher:.4f}")
    return stat_fisher, p_fisher

