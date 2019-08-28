import numpy as np


def lflatten(l):
    l = [item for sublist in l for item in sublist]
    return l


def remove_zero_rows(X):
    import numpy as np
    # X is a scipy sparse matrix. We want to remove all zero rows from it
    nonzero_row_indice, _ = X.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)
    return X[unique_nonzero_indice]

def mergsum(a, b):
    for k in b:
        if k in a:
            b[k] = b[k] + a[k]
    c = {**a, **b}
    return c


def update_in_alist(alist, key, value):
    return [[k, v] if (k != key) else (key, value) for (k, v) in alist]


def update_in_alist_inplace(alist, key, value):
    alist[:] = update_in_alist(alist, key, value)

def rms(y):
    return np.sqrt(np.mean(y ** 2))