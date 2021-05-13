"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros([n, K])
    log_likelihood = 0
    for u in range(n):
        x = X[u, :]
        C_u = (x != 0)
        x_Cu = x[C_u]
        mu_Cu = mixture.mu[:, C_u]
        kCu_array = x_Cu[np.newaxis, :] - mu_Cu
        k_array = -0.5 * np.square(np.linalg.norm(kCu_array, axis=-1))
        k_array = np.divide(k_array, mixture.var)
        k_array = k_array - (x_Cu.shape[0] / 2)*np.log(2 * np.pi * mixture.var)
        k_array = k_array + (np.log(mixture.p) + 1e-16)
        post_xCu_sum = logsumexp(k_array)
        log_likelihood += post_xCu_sum
        post[u, :] = np.transpose(np.exp(k_array - post_xCu_sum))

    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]
    sum = np.sum(post, axis=0)
    p = sum / n
    var = np.zeros(K)
    var_denominator = np.zeros(K)
    mu = np.zeros([K, d])
    mu_denominator = np.zeros([K, d])

    for u in range(n):
        x = X[u, :]
        C_u = np.nonzero(x)
        x_Cu = x[C_u]
        for l in range(d):
            mu_denominator[:, l] += delta(l, x) * post[u, :]
            mu[:, l] += delta(l, x) * post[u, :] * x[l]

    for i in range(K):
        for j in range(d):
            if mu_denominator[i, j] >= 1:
                mu[i, j] = mu[i, j] / mu_denominator[i, j]
            else:
                mu[i, j] = mixture.mu[i, j]

    for u in range(n):
        x = X[u, :]
        C_u = (x != 0)
        x_Cu = x[C_u]
        mu_Cu = mu[:, C_u]
        kCu_array = x_Cu[np.newaxis, :] - mu_Cu
        var += post[u, :] * np.square(np.linalg.norm(kCu_array, axis=-1))
        var_denominator += x_Cu.shape[0] * post[u, :]

    var = var / var_denominator
    for k in range(K):
        if var[k] < min_variance:
            var[k] = min_variance
    return GaussianMixture(mu, var, p)


def delta(l, x):
    if x[l] != 0:
        return 1
    else:
        return 0


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
        mixture = mstep(X, post, mixture)

    return mixture, post, cost


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros([n, K])
    for u in range(n):
        x = X[u, :]
        C_u = (x != 0)
        x_Cu = x[C_u]
        mu_Cu = mixture.mu[:, C_u]
        kCu_array = x_Cu[np.newaxis, :] - mu_Cu
        k_array = -0.5 * np.square(np.linalg.norm(kCu_array, axis=-1))
        k_array = np.divide(k_array, mixture.var)
        k_array = k_array - (x_Cu.shape[0] / 2)*np.log(2 * np.pi * mixture.var)
        k_array = k_array + (np.log(mixture.p) + 1e-16)
        post_xCu_sum = logsumexp(k_array)
        post[u, :] = np.transpose(np.exp(k_array - post_xCu_sum))

    X_new = X
    for u in range(n):
        for j in range(d):
            if X[u, j] == 0:
                X_new[u, j] = np.sum(mixture.mu[:, j]*post[u, :])

    return X_new

