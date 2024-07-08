import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm, invgamma
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def conjugate_prior_estimator_binomial(n_trials, true_p, alpha_prior, beta_prior):
    # Generate synthetic binomial data
    data = np.random.binomial(1, true_p, n_trials)
    
    # Prior parameters
    alpha = alpha_prior
    beta_p = beta_prior
    
    # Update posterior parameters
    alpha_post = alpha + np.sum(data)
    beta_post = beta_p + n_trials - np.sum(data)
    
    # Plot the prior and posterior Beta distributions
    x = np.linspace(0, 1, 100)
    prior = beta.pdf(x, alpha, beta_p)
    posterior = beta.pdf(x, alpha_post, beta_post)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, prior, label='Prior Beta Distribution')
    plt.plot(x, posterior, label='Posterior Beta Distribution')
    plt.title('Prior and Posterior Beta Distributions')
    plt.xlabel('p')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Example usage
conjugate_prior_estimator_binomial(n_trials=50, true_p=0.6, alpha_prior=2, beta_prior=2)