import numpy as np

def polynomial_kernel(x, y, b, d):
    return (np.dot(x, y) + b)**d

def radial_kernel(x, y, var):
    return np.exp(-(np.linalg.norm(x - y)**2)/(2 * var**2))

def tanh_kernel(x, y, a, b):
    return np.tanh(a * np.dot(x, y) + b)

def kernel_d(p, k, kernel_matrix, S, indep_terms):
    """Returns Kernel based distance between point and cluster center."""
    N, _ = S.shape
    term_1 = kernel_matrix[p, p]

    num_2 = 0
    den_2 = 0

    for l in range(N):
        if S[l, k] != 0:
            num_2 += S[l, k] * kernel_matrix[p, l]
            den_2 += S[l, k]

    term_2 = num_2 / den_2

    return term_1 - 2 * term_2 + indep_terms[k]

def kernel_dist_calc(X, S, K, kernel_type, kernel_args):

    """
    Returns distances of each point to each cluser. 

    Output: np array of size (N x K).
    """

    if kernel_type == 'poly':
        kernel = polynomial_kernel
    elif kernel_type == 'rad':
        kernel = radial_kernel
    elif kernel_type == 'tanh':
        kernel = tanh_kernel
    else:
        kernel = polynomial_kernel

    N = len(X)
    
    def f(i, j):
        return kernel(X[i], X[j], *kernel_args)
    
    print("Calculating kernel matrix ...")
    kernel_matrix = np.fromfunction(np.vectorize(f), (N, N), dtype=int)
 
    indep_terms = []
    for k in range(K):
        num_3, den_3 = 0, 0

        for q in range(N):
            for l in range(N):
                num_3 += S[q, k] * S[l, k] * kernel_matrix[q, l]
                
            den_3 += S[q, k]

        indep_terms.append(num_3 / (den_3**2))


    kernel_dist = np.zeros((N, K))
    for p in range(N):
        for k in range(K):
            kernel_dist[p, k] = kernel_d(p, k, kernel_matrix, S, indep_terms)
        
    return kernel_dist

def kernel_clustering_update(X, l, K, C):
    C_new = []
    for k in range(K):
        cents = X[l==k].mean(axis=0) # if k mean
        C_new.append(cents)

    return np.vstack(C_new)