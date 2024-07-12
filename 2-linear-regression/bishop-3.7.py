import matplotlib.pyplot as plt
import numpy as np
from linear import LinearRegression

# create dataset
np.random.seed(42)

def generate_data(size):
    a_0 = -0.3  # linear.b
    a_1 = 0.5   # linear.W
    X = np.random.uniform(-1, 1, size)
    noise = np.random.normal(0, 0.2, size)
    y = a_0 + a_1 * X + noise
    return X, y

def plot_data(ax, X, y):
    ax.scatter(X, y, c='b', zorder=1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])


def get_samples(num_samples):
    prior_mean_w = 0
    prior_std_w = 1
    prior_mean_b = 0
    prior_std_b = 1

    # generate samples from the prior
    sampled_W = np.random.normal(prior_mean_w, prior_std_w, num_samples)
    sampled_b = np.random.normal(prior_mean_b, prior_std_b, num_samples)
    return sampled_W, sampled_b

def plot_density(X, y):
    # Plot the data
    fig, ax = plt.subplots()
    # plot_data(ax, X, y)

    # Plot the prior
    x_range = np.linspace(-1, 1, 100)
    y_range = np.linspace(-1, 1, 100)
    W, b = np.meshgrid(x_range, y_range)
    Z = np.zeros(W.shape)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            for k in range(len(X)):
                Z[i, j] += np.log(1/(2*np.pi)) + 0.5 * np.log(1) - 0.5 * (y[k] - W[i, j] * X[k] - b[i, j])**2
    ax.contour(W, b, Z, levels=10, zorder=2)
    ax.set_xlabel('W')
    ax.set_ylabel('b')
    plt.show()


def plot_samples(X, y, num_samples, sampled_W, sampled_b):

    # Plot the data
    plt.scatter(X, y, label='Observed Data')

    # Plot six samples of the model
    x_range = np.linspace(-1, 1, 100).reshape(-1, 1)
    for i in range(num_samples):
        y_sample = sampled_W[i] * x_range + sampled_b[i]
        plt.plot(x_range, y_sample, label=f'Sample {i+1}')
    
    plt.xlabel('X')
    plt.ylabel('y')
    # plt.legend()
    plt.show()



X, y = generate_data(10)

X = X.reshape(-1, 1)

sampled_W, sampled_b = get_samples(6)

plot_density(X, y)

# plot_samples(X, y, 6, sampled_W, sampled_b)




# fig, ax = plt.subplots()
# fig.x_lim = (-1, 1)
# fig.y_lim = (-1, 1)
# plot_data(ax, X, y)


# linear = LinearRegression(learning_rate=0.01, iterations=100)
# linear.fit(X, y)

# linear.predict(X)
breakpoint()