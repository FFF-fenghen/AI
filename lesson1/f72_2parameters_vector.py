import f70dataset
import numpy as np
import f72_plot_utils as plt

m = 100
X, Y = f70dataset.get_beans(m)
plt.show_scatter(X, Y)

alpha = 0.01
W = np.array([0.1, 0.1])
B = np.array([0.1])


def forward_propagation(X):
    Z = X.dot(W.T) + B
    A = 1 / (1 + np.exp(-Z))
    return A


plt.show_scatter_surface(X, Y, forward_propagation)


for _ in range(5000):
    for i in range(m):
        Xi = X[i]
        Yi = Y[i]
        A = forward_propagation(Xi)

        E = (Yi - A) ** 2
        dEdA = -2 * (Y - A)
        dAdz = A * (1 - A)
        dZdW = Xi
        dAdB = 1

        W = W - alpha * (dEdA * dAdz * dZdW)
        B = B - alpha * (dEdA * dAdz * dAdB)

plt.show_scatter_surface(X, Y, forward_propagation)
