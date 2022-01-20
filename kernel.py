import numpy as np

def polynomial_kernel(x, y, b, d):
    return (np.dot(x, y) + b)**d

def radial_kernel(x, y, var):
    return np.exp(-(-np.linalg.norm(x, y)**2)/(2 * var**2))

def tanh_kernel(x, y, a, b):
    return np.tanh(a * np.dot(x, y) + b)

def a(X, p, k, kernel, S):
    """Returns auxiliary function for Kernel-based VFC."""

    N, _ = S.shape()

    term_1 = kernel(X[p], X[p])
    num_2 = 0
    den_2 = 0

    for l in range(N):
        num_2 += S[l, k] * kernel(X[l], X[p])
        den_2 += S[l, k]

    term_2 = num_2/den_2

    num_3, den_3 = 0, 0

    for q in range(N):
        for l in range(N):
            num_3 += S[q, k] * S[l, k] * kernel(X[q], X[l])

        den_3 += S[q, k]

    term_3 = num_3 / (den_3**2)

    return term_1 - 2 * term_2 + term_3
