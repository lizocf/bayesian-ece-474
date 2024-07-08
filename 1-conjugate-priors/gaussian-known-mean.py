import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm, invgamma

np.random.seed(111)

# Gaussian Distribution with Known Variance -> Conj. Prior: Gaussian
def generate_gaussian_data(mu, sigma, size):
    return np.random.normal(mu, sigma, size)

def gaussian_known_variance_mle(data):
    return np.mean(data)

def gaussian_known_variance_conjugate(data, mu_0, tau_squared, sigma_squared):
    n = len(data)
    x_bar = np.mean(data)
    mu_n = (mu_0 / tau_squared + n * x_bar / sigma_squared) / (1 / tau_squared + n / sigma_squared)
    sigma_n_squared = 1 / (1 / tau_squared + n / sigma_squared)
    return mu_n, sigma_n_squared


mu_true = 5.0
sigma_true = 2.0
n_samples = 100

gaussian_data = np.random.normal(mu_true, sigma_true, n_samples)

