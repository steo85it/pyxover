


def lflatten(l):
    l = [item for sublist in l for item in sublist]
    return l


def remove_zero_rows(X):
    import numpy as np
    # X is a scipy sparse matrix. We want to remove all zero rows from it
    nonzero_row_indice, _ = X.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)
    return X[unique_nonzero_indice]
