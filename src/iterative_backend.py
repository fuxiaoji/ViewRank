"""Matrix-free base responses for solver-agnostic posterior certificates."""

from time import perf_counter

import numpy as np
from scipy import sparse

from .viewrank import ViewRank


class FixedPointSolver:
    """Solve (I-alpha P)x=rhs to an absolute per-column L1 residual target."""

    def __init__(self, P, alpha, tol=1e-12, maxiter=10000, A=None):
        self.P, self.alpha, self.tol, self.maxiter = P, alpha, tol, maxiter
        self.A = A if A is not None else (sparse.eye(P.shape[0], format="csc")-alpha*P).tocsc()
        self.calls = 0
        self.rhs_columns = 0
        self.iterations = 0
        self.elapsed_seconds = 0.0
        self.max_returned_residual = 0.0

    def solve(self, rhs):
        started = perf_counter()
        rhs = np.asarray(rhs, dtype=float)
        x = np.zeros_like(rhs)
        columns = 1 if rhs.ndim == 1 else rhs.shape[1]
        for iteration in range(1, self.maxiter+1):
            new = self.alpha*(self.P@x)+rhs
            delta = np.sum(np.abs(new-x), axis=0)
            x = new
            # delta is the old iterate's residual. Stochastic P contracts
            # its L1 norm by alpha, so the returned iterate meets tol too.
            if np.all(delta <= self.tol):
                residual = np.sum(np.abs(rhs-self.A@x), axis=0)
                if not np.all(residual <= self.tol):
                    continue
                self.calls += 1
                self.rhs_columns += columns
                self.iterations += iteration
                self.elapsed_seconds += perf_counter()-started
                self.max_returned_residual = max(self.max_returned_residual, float(np.max(residual)))
                self.last_residuals = np.atleast_1d(residual).copy()
                return x
        raise RuntimeError(f"fixed-point response did not meet L1 residual target in {self.maxiter} iterations")


class MatrixFreeViewRank(ViewRank):
    """ViewRank interface with no LU and no graph factorization setup.

    prepare/solve remain inherited. They use approximate base responses with
    residual <=tol, which Envelope measures and absorbs in its posterior bound.
    """

    def __init__(self, P, alpha=0.85, tol=1e-12):
        started = perf_counter()
        self.P = sparse.csc_matrix(P, dtype=np.float64)
        self.alpha = float(alpha)
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        self.n = self.P.shape[0]
        self.A = (sparse.eye(self.n, format="csc")-self.alpha*self.P).tocsc()
        self.base_lu = FixedPointSolver(self.P, self.alpha, tol=tol, A=self.A)
        self.base_factor_seconds = perf_counter()-started
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def base_payload_bytes(self):
        return sum(m.data.nbytes+m.indices.nbytes+m.indptr.nbytes for m in [self.P, self.A])
