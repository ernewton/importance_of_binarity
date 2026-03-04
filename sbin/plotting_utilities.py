import numpy as np
import matplotlib.pyplot as plt


def plot_hist_confidence(ax, trials_to_plot, bins=None, 
                         log_flag=True, density_flag=True,
                         weights_flag=False,
                         ci_low=16, ci_high=84):
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])           # for plotting
    
    hists = []
    for x in trials_to_plot:
        if log_flag:
            x = np.log10(x)
        if weights_flag:
            weights=1./np.full(len(x), len(x), dtype=float)
        else:
            weights=None
        count, _ = np.histogram(x, bins=bins, density=density_flag, weights=weights)
        hists.append(count)
 
    # simulation shenanigans
    hists = np.asarray(hists)              
    median_h  = np.median(hists, axis=0)               # median density
    lower_h   = np.percentile(hists, ci_low, axis=0)       # 16 th percentile
    upper_h   = np.percentile(hists, ci_high, axis=0)       # 84 th percentile
    lower_step = np.r_[lower_h, lower_h[-1]]
    upper_step = np.r_[upper_h,  upper_h[-1]]
    median_step = np.r_[median_h,  median_h[-1]]

    #plt.hist(np.log10(median_h), bins=bins, density=True, alpha=0.2)
    # 68 % confidence band
    ax.step(bins, median_step, where='post',
            color='gray', lw=4, linestyle=':',
            label='Simulated KOIs')
    ax.fill_between(bins, lower_step, upper_step, step='post',
                    color='gray', alpha=0.5)



def ecdf_grid(x):
    """
    Return (xs, ys) that define the ECDF of a 1‑D array x.
    xs  – sorted x values
    ys  – cumulative probabilities  (0 < y ≤ 1)
    """
    xs = np.sort(x)
    n  = xs.size
    # y runs from 1/n to 1  (step after each data point)
    ys = np.arange(1, n + 1) / n
    return xs, ys




def ecdf_on_grid(sample, grid):
    
    return np.searchsorted(np.sort(sample), grid, side='right') / sample.size


def ecdf_confidence(samples):  
    
    # ---------- ECDF on a common grid ----------
    x_grid = np.sort(np.unique(np.concatenate(samples)))
    ecdf_matrix = np.vstack([ecdf_on_grid(trial, x_grid) for trial in samples])

    # ---------- Percentiles ----------
    lower = np.percentile(ecdf_matrix, 16, axis=0)
    upper = np.percentile(ecdf_matrix, 84, axis=0)
    median = np.percentile(ecdf_matrix, 50, axis=0)
    
    return [x_grid, lower, median, upper]


def plot_ecdf_confidence(ax, samples, log_flag=False):
    
    x_grid, lower, median, upper = ecdf_confidence(samples)
    if log_flag:
        x_grid=np.log10(x_grid)
    
    # ---------- Plot ----------
    ax.step(x_grid, median, where='post',
            color='gray', lw=3, linestyle=':',
            label='Simulated KOIs')
    ax.fill_between(x_grid,
                    lower, upper,
                    step='post', color='gray', alpha=0.4)

