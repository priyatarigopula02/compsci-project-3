import numpy as np
import matplotlib as mpl


def detpep10exp(x1, x2, x3):
    """Evaluates the Dette & Pepelyshev (2010) exponential function.

    References
    ----------
    https://www.sfu.ca/~ssurjano/detpep10exp.html

    """
    term1 = np.exp(-2/(x1**1.75))
    term2 = np.exp(-2/(x2**1.5))
    term3 = np.exp(-2/(x3**1.25))
    y = 100 * (term1 + term2 + term3)
    return y


def load_detpep10exp_dataset(n_samples, sample_range=(0, 1)):
    """Returns a dataset derived from the detpep10exp function.

    Parameters
    ----------
    n_samples: int
        The number of samples in each dimension. Final dataset will be of size n_samples^3.
    sample_range: tuple
        The range in each dimension in which datapoints will be sampled.

    """
    x1 = np.linspace(*sample_range, n_samples)
    x2 = np.linspace(*sample_range, n_samples)
    x3 = np.linspace(*sample_range, n_samples)
    xx1, xx2, xx3 = np.meshgrid(x1, x2, x3)
    y = detpep10exp(xx1.flatten(), xx2.flatten(), xx3.flatten())
    return np.stack([xx1.flatten(), xx2.flatten(), xx3.flatten()], axis=-1), y


def plot_init():
    """Settings for matplotlib.
    """
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
