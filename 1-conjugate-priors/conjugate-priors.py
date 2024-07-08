import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm, invgamma

np.random.seed(111)

# Binomial Distribution -> Conj. Prior: Beta Distribution
def generate_binomial_data(n, p, size):
    return np.random.binomial(n, p, size)

def binomial_mle(data, n):
    return np.mean(data) / n

def binomial_conjugate(data, n, alpha, beta):
    k = np.sum(data)
    return (alpha + k) / (alpha + beta + len(data) * n)

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

# Gaussian Distribution with Known Mean -> Conj. Prior: Wishart
def gaussian_known_mean_mle(data, mu):
    n = len(data)
    return np.sum((data - mu)**2) / n

def gaussian_known_mean_conjugate(data, mu, alpha, beta):
    n = len(data)
    sum_squared_diff = np.sum((data - mu)**2)
    alpha_n = alpha + n / 2
    beta_n = beta + sum_squared_diff / 2
    return alpha_n, beta_n

# Plot Mean Squared Error
def plot_mse(estimates, true_value, label):
    mse = [(est - true_value)**2 for est in estimates]
    plt.plot(mse, label=label)

# Plot Posterior Density
def plot_posterior_density(prior, posterior, label, x_range, n_points=100):
    x = np.linspace(x_range[0], x_range[1], n_points)
    plt.plot(x, prior.pdf(x), label=f'Prior {label}')
    plt.plot(x, posterior.pdf(x), label=f'Posterior {label}')

# Binomial 
n = 10
p = 0.5
size = 100

binomial_data = generate_binomial_data(n, p, size)
binomial_mle_estimates = [binomial_mle(binomial_data[:i+1], n) for i in range(size)]
binomial_conjugate_estimates = [binomial_conjugate(binomial_data[:i+1], n, 2, 2) for i in range(size)]

plt.figure()
plot_mse(binomial_mle_estimates, p, 'Binomial MLE')
plot_mse(binomial_conjugate_estimates, p, 'Binomial Conjugate')
plt.legend()
plt.show()