#!/usr/bin/env python3
# ----------------------------------
# utilities
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Mar-2019
#

import numpy as np

from scipy.sparse import csr_matrix, csc_matrix


def lflatten(l):
    l = [item for sublist in l for item in sublist]
    return l


def remove_zero_rows(X):
#    import numpy as np
    # X is a scipy sparse matrix. We want to remove all zero rows from it
    nonzero_row_indice, _ = X.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)
    return X[unique_nonzero_indice]

# auxiliary function for multiply_sparse_get_diag
def row_times_cols(args):
    i, a, b = args
    n = a.shape[0]
    # if i % (int(n / 1.)) == 0:
    #     print("processing ", i)
    a2 = a.getrow(i)
    if (a2.getnnz() == 0):
        tmp = 0.
    else:
        b2 = b.getcol(i)
        if (b2.getnnz() == 0):
            tmp = 0.
        else:
            res = csr_matrix.dot(a2, b2)
            if res.getnnz() > 0:
                tmp = res.data[0]
            else:
                tmp = 0.
    # print(i,tmp)
    return tmp #(i,tmp)

# Multiplies two matrices to get diagonal elements only
# @profile
def multiply_sparse_get_diag(a,b):

    a = csr_matrix(a)
    b = csc_matrix(b)
    n = a.shape[0]

    # a, b = get_random_sparse(n=n, p=p, density=0.0002)

    print('start multiply_sparse_get_diag')
    args = ((i, a, b) for i in range(n))

    if True: #parallel:
        import multiprocessing as mp

        pool = mp.Pool(processes=mp.cpu_count() - 1)
        tmp = pool.map(row_times_cols, args)  # parallel
        pool.close()
        pool.join()
    else:
        tmp = [row_times_cols(args[i]) for i in range(n)]

    print('end multiply_sparse_get_diag')

    return np.array(tmp)


def mergsum(a, b):
    import copy
    sum=copy.deepcopy(b)
    for k in b:
        if k in a:
            sum[k] = sum[k] + a[k]
    c = {**a, **sum}
    return c

def dict2np(x):
    return np.array(list(x.values()))

def update_in_alist(alist, key, value):
    return [[k, v] if (k != key) else (key, value) for (k, v) in alist]


def update_in_alist_inplace(alist, key, value):
    alist[:] = update_in_alist(alist, key, value)


def get_size(obj, seen=None):
    import sys

    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
