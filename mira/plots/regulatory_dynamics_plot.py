import numpy as np
from scipy import sparse
from lisa.core.genome_tools import Region
import warnings
from matplotlib import patches
from tqdm.notebook import trange

def _residual_transform(X, pi_j_hat, n_i):
    
    assert(isinstance(X, np.ndarray))
    assert(isinstance(pi_j_hat, np.ndarray))
    assert(isinstance(n_i, np.ndarray))
    pi_j_hat = np.squeeze(pi_j_hat)[np.newaxis, :]
    n_i = np.squeeze(n_i)[:, np.newaxis]

    mu_ij_hat = n_i * pi_j_hat

    count_dif = n_i - X
    expected_count_dif = n_i - mu_ij_hat

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        r_ij = np.multiply(
            np.sign(X - mu_ij_hat), 
            np.sqrt(
            np.where(X > 0, 2 * np.multiply(X, np.log(X / mu_ij_hat)), 0) + \
            2 * np.multiply(count_dif, np.log(count_dif / expected_count_dif))
            )
        )

    return np.clip(np.nan_to_num(r_ij), -10, 10)

def _get_pi(X):
    return np.array(X.sum(0)).reshape(-1)/X.sum()

def _get_n(X):
    return np.array(X.sum(-1)).reshape(-1)


def plot_fragment_heatmap(ax,*, accessibility, chrom, start, end, peaks, time, height = 1):

    accessibility.data = np.ones_like(accessibility.data)
    n_i = _get_n(accessibility)
    p_i = _get_pi(accessibility)

    interval = Region(chrom, start, end)

    overlapped_peaks = np.array([
        Region(*peak).overlaps(interval)
        for peak in peaks
    ])
    
    order = time.argsort()

    X = accessibility[:, overlapped_peaks].toarray()[order, :]

    residuals = _residual_transform(X, p_i[overlapped_peaks], n_i)

    ax.set(xlim = (start, end), ylim = (0, len(residuals)))
    t_ = iter(trange((residuals > 0).sum(), desc = 'Plotting fragments'))

    for alpha, position in zip(residuals.T, peaks[overlapped_peaks]):
        
        start, end = (int(position[1]), int(position[2]))
        min_, max_ = alpha.min(), alpha.max()
        transparency = (alpha - min_)/(max_ - min_)
        
        for j, a in enumerate(alpha):
            if a > 0:
                ax.add_patch(
                    patches.Rectangle((start, j), (end - start), height, color = 'black', alpha = transparency[j])
                )
                next(t_)

    ax.invert_yaxis()