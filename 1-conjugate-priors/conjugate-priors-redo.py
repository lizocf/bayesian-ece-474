import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm, invgamma

np.random.seed(111)

# Binomial 
n = 10
p = 0.5
bin_alphas = [1, 2, 10]
bin_betas = [1, 4, 20]

# Gaussian
mu_true = 2.0
sigma_true = 1.0

mu_0s = [1.5, 2.0, 2.5]
tau_0s = [1.0, 1.01, 0.99]

gauss_alphas = [1, 4, 20]
gauss_betas = [1, 4, 20]


n_samples = 100

binomial_data = np.random.binomial(n, p, n_samples)
gaussian_data = np.random.normal(mu_true, sigma_true, n_samples)


# breakpoint()

# Binomial Distribution -> Conj. Prior: Beta Distribution
def generate_binomial_data(n, p, size):
    return np.random.binomial(n, p, size)

def binomial_mle(data, n):
    return np.mean(data) / n

# Beta Mean: alpha + num_successes / alpha + beta + total_trials

def binomial_conjugate(data, n, alpha, beta):
    num_successes = np.sum(data) # num_successes
    num_trials = n * len(data)
    return (alpha + num_successes) / (alpha + beta + num_trials)

# Gaussian Distribution with Known Variance -> Conj. Prior: Gaussian
def generate_gaussian_data(mu, sigma, size):
    return np.random.normal(mu, sigma, size)

def gaussian_known_variance_mle(data):
    return np.mean(data)

# ref: https://youtu.be/R3pdTd2mVaI?si=CsBdZu7ni4ech2Gj&t=1612
def gaussian_known_variance_conjugate(data, mu_0, tau_0, sigma):
    n = len(data)
    x_bar = np.mean(data)
    mu_n = (mu_0 * sigma**2 + tau_0**2 * np.sum(data)) / (tau_0**2 + n * sigma**2)
    return mu_n

# Gaussian Distribution with Known Mean -> Conj. Prior: Inv Gamma
def gaussian_known_mean_mle(data, mu):
    n = len(data)
    return np.sum((data - mu)**2) / n

# ref: https://youtu.be/C_ZszihZzV0?si=WpElMVrvH7HaAbOu&t=1104
def gaussian_known_mean_conjugate(data, mu, alpha, beta):
    n = len(data)
    sum_squared_diff = np.sum((data - mu)**2)
    alpha_n = alpha + n / 2 - 1
    beta_n = beta + sum_squared_diff / 2
    return alpha_n / beta_n

def get_mse(estimates, true_value):
    mse = [(est - true_value)**2 for est in estimates]
    return mse

# Plot Posterior Density
def plot_posterior_density(prior, posterior, label, x_range, n_points=100):
    x = np.linspace(x_range[0], x_range[1], n_points)
    plt.plot(x, prior.pdf(x), label=f'Prior {label}')
    plt.plot(x, posterior.pdf(x), label=f'Posterior {label}')

# # Binomial 
# i, j = 0, 0
# fig, axs = plt.subplots(3,3)
# fig.suptitle('Binomial Estimates')
# for alpha in bin_alphas:
#     for beta in bin_betas:
#         binomial_mle_estimates = [binomial_mle(binomial_data[:i+1], n) for i in range(n_samples)]
#         binomial_conjugate_estimates = [binomial_conjugate(binomial_data[:i+1], n, alpha, beta) for i in range(n_samples)]
#         mle_mse = get_mse(binomial_mle_estimates, p)
#         conj_mse = get_mse(binomial_conjugate_estimates, p)
#         # plt.figure(
#         axs[i, j].plot(mle_mse, label= 'Binomial MLE')
#         axs[i, j].plot(conj_mse, label='Binomial Conjugate')
#         axs[i, j].set_title(f'$\\alpha$: {alpha}, $\\beta$: {beta}')
#         axs[i, j].legend()
#         j += 1
#     i += 1
#     j = 0

# plt.show()

# # # Gaussian + Known Variance


# i, j = 0, 0
# fig, axs = plt.subplots(3,3)
# fig.suptitle('Gaussian + Known Variance Estimates')
# for mu_0 in mu_0s:
#     for tau_0 in tau_0s:
#         gaussian_known_var_mle_estimates = [gaussian_known_variance_mle(gaussian_data[:i+1]) for i in range(n_samples)]
#         gaussian_known_var_conjugate_estimates = [gaussian_known_variance_conjugate(gaussian_data[:i+1], mu_0, tau_0, sigma_true) for i in range(n_samples)]
#         mle_mse = get_mse(gaussian_known_var_mle_estimates, mu_true)
#         conj_mse = get_mse(gaussian_known_var_conjugate_estimates, mu_true)
#         # plt.figure(
#         axs[i, j].plot(mle_mse, label= 'Gaussian MLE')
#         axs[i, j].plot(conj_mse, label='Gaussian Conjugate')
#         axs[i, j].set_title(f'$\\mu_0$: {mu_0}, $\\tau_0$: {tau_0}')
#         axs[i, j].legend()
#         j += 1
#     i += 1
#     j = 0

# plt.show()

# Gaussian + Known Mean

i, j = 0, 0
fig, axs = plt.subplots(3,3)
fig.suptitle('Gaussian + Known Mean Estimates')
for alpha in gauss_alphas:
    for beta in gauss_betas:
        gaussian_known_mean_mle_estimates = [gaussian_known_mean_mle(gaussian_data[:i+1], mu_true) for i in range(n_samples)]
        gaussian_known_mean_conjugate_estimates = [gaussian_known_mean_conjugate(gaussian_data[:i+1], mu_true, alpha, beta) for i in range(n_samples)]
        mle_mse = get_mse(gaussian_known_mean_mle_estimates, mu_true)
        conj_mse = get_mse(gaussian_known_mean_conjugate_estimates, mu_true)
        # plt.figure(
        axs[i, j].plot(mle_mse, label= 'Binomial MLE')
        axs[i, j].plot(conj_mse, label='Binomial Conjugate')
        axs[i, j].set_title(f'$\\alpha$: {alpha}, $\\beta$: {beta}')
        axs[i, j].legend()
        j += 1
    i += 1
    j = 0

plt.show()

# gaussian_known_mean_mle_estimates = [gaussian_known_mean_mle(gaussian_data[:i+1], mu_true) for i in range(n_samples)]
# gaussian_known_mean_conjugate_estimates = [gaussian_known_mean_conjugate(gaussian_data[:i+1], mu_true, alpha, beta) for i in range(n_samples)]

# plt.figure()
# plot_mse(gaussian_known_mean_mle_estimates, mu_true, 'Gaussian Known Mean MLE')
# plot_mse(gaussian_known_mean_conjugate_estimates, mu_true, 'Gaussian Known Mean Conjugate')
# plt.xlabel('Number of Samples')
# plt.ylabel('Mean Squared Error (MSE)')
# plt.legend()
# plt.title('Mean Squared Error Comparison')
# plt.show()
