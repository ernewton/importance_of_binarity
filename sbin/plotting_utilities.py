import numpy as np
import matplotlib.pyplot as plt


def plot_hist_confidence(ax, trials_to_plot, bins=None, 
                         log_flag=True, density_flag=True,
                         weights_flag=False):
    
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
    lower_h   = np.percentile(hists, 16, axis=0)       # 16 th percentile
    upper_h   = np.percentile(hists, 84, axis=0)       # 84 th percentile
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

def plot_semimajor_ecdf(ax, a_values_sim,
                  log_min=0.5, log_max=2.0,
                  n_grid=500,
                  ci_low=16, ci_high=84):
    """
    Plot the median ECDF of many Monte‑Carlo realisations together with a
    (ci_low, ci_high) confidence band.

    Parameters
    ----------
    a_values_sim : list or ndarray of shape (n_real, n_samples)
        Each element is an array of simulated semi‑major axes (in AU).
    log_min, log_max : float, optional
        Limits of the x‑axis in log10 space.
    n_grid : int, optional
        Number of points on the common interpolation grid.
    ci_low, ci_high : int, optional
        Percentiles for the confidence band (default 16 % / 84 % → ≈68 %).
    """
    # ------------------------------------------------------------------ #
    # 1.  Build a common grid in log10‑space on which we will evaluate
    #     every ECDF.  A regular grid works well for visualisation.
    # ------------------------------------------------------------------ #
    x_grid = np.linspace(log_min, log_max, n_grid)

    # ------------------------------------------------------------------ #
    # 2.  Compute the ECDF for each realisation and interpolate it onto
    #     the common grid.
    # ------------------------------------------------------------------ #
    interpolated_ecdfs = []                        # will become (n_real, n_grid)

    for sim in a_values_sim:
        # ----- (a) work in log10 space the same way you did for the histograms
        log_a = np.log10(sim)

        # ----- (b) get the raw ECDF (sorted x, cumulative y)
        xs, ys = ecdf_grid(log_a)

        # ----- (c) interpolate onto the fixed grid.
        #     Outside the data range the ECDF is 0 (left) or 1 (right).
        #     np.interp does exactly this with the `left`/`right` arguments.
        y_on_grid = np.interp(x_grid, xs, ys,
                              left=0.0,   # no data → prob = 0
                              right=1.0)  # beyond max → prob = 1
        interpolated_ecdfs.append(y_on_grid)

    # Convert to a 2‑D array: shape = (n_real, n_grid)
    interpolated_ecdfs = np.asarray(interpolated_ecdfs)

    # ------------------------------------------------------------------ #
    # 3.  From the stacked ECDFs obtain median and confidence limits.
    # ------------------------------------------------------------------ #
    median_ecdf = np.median(interpolated_ecdfs, axis=0)
    lower_ecdf  = np.percentile(interpolated_ecdfs, ci_low, axis=0)
    upper_ecdf  = np.percentile(interpolated_ecdfs, ci_high, axis=0)

    # For convenient stepping we append the last value once more so that
    # `where='post'` draws a horizontal line to the right edge of the plot.
    median_step = np.r_[median_ecdf, median_ecdf[-1]]
    lower_step  = np.r_[lower_ecdf,  lower_ecdf[-1]]
    upper_step  = np.r_[upper_ecdf,  upper_ecdf[-1]]
    step_x      = np.r_[x_grid,         x_grid[-1] + (x_grid[1]-x_grid[0])]

    # ------------------------------------------------------------------ #
    # 4.  Plot
    # ------------------------------------------------------------------ #
    ax.step(step_x, median_step, where='post',
            color='gray', lw=3, linestyle=':',
            label='Median simulated ECDF')
    ax.fill_between(step_x,
                    lower_step, upper_step,
                    step='post', color='gray', alpha=0.4,
                    label=r'68 % confidence band')

    ax.set_xlabel(r'$\log_{10}(a\;[\mathrm{au}])$')
    ax.set_ylabel('Cumulative probability')
