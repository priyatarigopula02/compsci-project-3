import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm


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


def load_detpep10exp_dataset(n_samples,
                             sample_range=(0, 1)):
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


def plot_3d(x,
            y,
            figsize=(10, 10),
            vmin=None,
            vmax=None):
    """Creates a 3d plot for 3 features (explanatory variables) and 1 response variable.

    Parameters
    ----------
    x: ndarray of shape (n_samples, 3)
        Coordinates of samples.
    y: ndarray of shape (n_samples,)
        Values corresponding to coordinates.
    figsize: tuple
        Size of figure.
    vmin: int, optional
        Lower limit of color scale.
    vmax: int, optional
        Upper limit of color scale.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap=cm.jet)
    if vmin is not None and vmax is not None:
        im.set_clim(vmin=vmin, vmax=vmax)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.colorbar(im, fraction=0.026, pad=0.04)
    plt.show()
