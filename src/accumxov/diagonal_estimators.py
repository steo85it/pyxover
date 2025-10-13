#!/usr/bin/env python3
# ----------------------------------
# diagonal_estimators.py
#
# Description: various methods to be applied to AccumXov
#
# ----------------------------------------------------
# Author: William Desprats
# Created: 13-Oct-2025

import numpy as np
from scipy.sparse.linalg import cg, lsqr, LinearOperator
from scipy.linalg import solve_triangular


def stochastic_diag_estimate_A(A, num_samples=20, tol=1e-5):
    # probably not correct
    n = A.shape[1]
    diag_est = np.zeros(n)

    for _ in range(num_samples):
        z = np.random.choice([-1, 1], size=n)
        y = lsqr(A,A @ z, atol=tol,btol=tol, iter_lim=500)[0]
        diag_est += z * y

    return diag_est / num_samples

def estimate_diag_inv_AtA(A, num_probes=50, distribution="rademacher", tol=1e-6, seed=None):
   n = A.shape[1]
   rng = np.random.default_rng(seed)
   diag_est = np.zeros(n)

   def matvec(x):
      return A.T @ (A @ x)

   AtA = LinearOperator((n, n), matvec=matvec, dtype=np.float64)

   for _ in range(num_probes):
      z = rng.integers(0, 2, size=n) * 2 - 1  # ±1

      x, info = cg(AtA, z, tol=tol)
      if info != 0:
         print(f"Warning: CG did not converge (info={info})")
      # diag_est += x * z  # elementwise
      diag_est += np.clip(x * z, 0, None)  # elementwise

   return diag_est / num_probes

def stochastic_diag_estimate_N(N, num_samples=20, tol=1e-5):
    n = N.shape[0]
    diag_est = np.zeros(n)

    for _ in range(num_samples):
        z = np.random.choice([-1, 1], size=n)
        y, info = cg(N, z, tol=tol, maxiter=500)
        if info != 0:
           print("cg did not converge")
           continue
        diag_est += z * y

    return diag_est / num_samples

def stochastic_trace_estimate_full(Ni, N, m=20):
    dim = Ni.shape[0]
    total = 0.0
    D = N.diagonal()
    M_inv = 1.0 / D
    M_precond = LinearOperator(N.shape, matvec=lambda x: M_inv * x)
    trace_estimates = []
    for _ in range(m):
        z = np.random.choice([1.0, -1.0], size=dim)     # Rademacher probe
        # w = cg(N, z)[0]                               # solve N w = z
        w = cg(N, z, M=M_precond)[0]                    # solve N w = z
        Nz = Ni.dot(w.T)                                # multiply back by N
        trace_estimates.append(z @ Nz)
    return np.mean(trace_estimates)
  
def stochastic_trace_estimate_chol(L, Ni, num_samples=20):
   n = L.shape[0]
   total = 0.0

   trace_estimates = []
   for _ in range(num_samples):
      z = np.random.choice([-1, 1], size=n)
        
      # Solve N x = z using Cholesky: L y = z, L.T x = y
      y = solve_triangular(L, z, lower=True)
      x = solve_triangular(L.T, y, lower=False)
      x = (Ni.dot(x))

      trace_estimates.append(z.dot(x.T))  # zᵀ M x

   
   return  np.mean(trace_estimates)