"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape

    knd_array = X[np.newaxis, :, :] - mixture.mu[:, np.newaxis, :]
    kn_array = -(1/2)*np.square(np.linalg.norm(knd_array, axis=-1))
    kn_array = np.divide(kn_array, mixture.var[:, np.newaxis])
    kn_array = np.exp(kn_array)
    kn_array = np.divide(kn_array, np.power(2*np.pi*mixture.var, d/2)[:, np.newaxis])
    kn_array = np.multiply(kn_array, mixture.p[:, np.newaxis])
    n_array = np.sum(kn_array, axis=0)
    post = np.transpose(np.divide(kn_array, n_array[np.newaxis, :]))

    LL = np.sum(np.log(n_array))[0]
    return post, LL


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    sum = np.sum(post, axis=0)
    p = sum/n
    mu = np.transpose(post)@X
    mu = np.divide(mu, sum[:, np.newaxis])
    knd_array = X[np.newaxis, :, :] - mu[:, np.newaxis, :]
    nk_array = np.transpose(np.square(np.linalg.norm(knd_array, axis=-1)))*post
    k_array = np.sum(nk_array, axis=0)
    sum = d*sum
    var = np.divide(k_array, sum)

    return GaussianMixture(mu, var, p)




def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_cost = None
    cost = None
    while prev_cost is None or cost - prev_cost > 1e-6*np.abs(cost):
        prev_cost = cost
        post, cost = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, cost
