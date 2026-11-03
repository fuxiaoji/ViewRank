"""Exact personalized diffusion over local graph views via shared responses."""

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu


def _sparse_bytes(matrix):
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


def _canonical_update(D, sources, n):
    """Merge source columns, including updates which cancel each other."""
    D = sparse.csc_matrix(D, dtype=np.float64)
    sources = np.asarray(sources, dtype=np.int64)
    if D.shape != (n, len(sources)):
        raise ValueError("D must have shape (number of nodes, number of sources)")
    if sources.size and (sources.min() < 0 or sources.max() >= n):
        raise ValueError("source node index is outside the graph")
    unique, inverse = np.unique(sources, return_inverse=True)
    groups = sparse.csc_matrix(
        (np.ones(len(sources)), (np.arange(len(sources)), inverse)),
        shape=(len(sources), len(unique)),
    )
    D = (D @ groups).tocsc()
    D.sum_duplicates()
    D.eliminate_zeros()
    keep = np.flatnonzero(np.diff(D.indptr))
    D = D[:, keep]
    D.sort_indices()
    return D, unique[keep]


@dataclass
class PreparedView:
    D: sparse.csc_matrix
    sources: np.ndarray
    Z: np.ndarray
    K: np.ndarray
    small_lu: tuple | None
    fingerprint: str
    prepare_seconds: float

    @property
    def rank(self):
        return len(self.sources)

    @property
    def nbytes(self):
        small_bytes = 0 if self.small_lu is None else sum(a.nbytes for a in self.small_lu)
        return _sparse_bytes(self.D) + self.sources.nbytes + self.Z.nbytes + self.K.nbytes + small_bytes


class ViewRank:
    """Share one base LU factorization across column-local graph updates.

    P is column stochastic, including self loops for dangling nodes. Each view
    is P + D E_sources.T. The caller supplies column-normalized view updates.
    """

    def __init__(self, P, alpha=0.85):
        self.P = sparse.csc_matrix(P, dtype=np.float64)
        self.alpha = float(alpha)
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        self.n = self.P.shape[0]
        self.A = (sparse.eye(self.n, format="csc") - self.alpha * self.P).tocsc()
        started = perf_counter()
        self.base_lu = splu(self.A)
        self.base_factor_seconds = perf_counter() - started
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def cache_bytes(self):
        return sum(prepared.nbytes for prepared in self.cache.values())

    def prepare(self, D, sources, key=None):
        started = perf_counter()
        D, sources = _canonical_update(D, sources, self.n)
        digest = sha256()
        for item in (np.asarray(D.shape, dtype=np.int64), sources, D.indptr, D.indices, D.data):
            digest.update(item.tobytes())
        fingerprint = digest.hexdigest()
        cache_key = (key, fingerprint)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        self.cache_misses += 1
        rank = len(sources)
        Z = self.base_lu.solve((-self.alpha * D).toarray()) if rank else np.empty((self.n, 0))
        K = np.eye(rank) + Z[sources, :]
        prepared = PreparedView(
            D=D, sources=sources, Z=Z, K=K,
            small_lu=lu_factor(K) if rank else None,
            fingerprint=fingerprint, prepare_seconds=0.0,
        )
        prepared.prepare_seconds = perf_counter() - started
        self.cache[cache_key] = prepared
        return prepared

    def solve(self, seed, prepared):
        y = self.base_lu.solve((1.0 - self.alpha) * np.asarray(seed, dtype=float))
        if not prepared.rank:
            return y
        coefficients = lu_solve(prepared.small_lu, y[prepared.sources])
        return y - prepared.Z @ coefficients

    def residual(self, seed, x, prepared):
        x = np.asarray(x)
        rhs = (1.0 - self.alpha) * np.asarray(seed)
        product = self.A @ x - self.alpha * (prepared.D @ x[prepared.sources])
        return float(np.linalg.norm(rhs - product, ord=1))

    def view_matrix(self, prepared):
        selection = sparse.csc_matrix(
            (np.ones(prepared.rank), (np.arange(prepared.rank), prepared.sources)),
            shape=(prepared.rank, self.n),
        )
        return (self.P + prepared.D @ selection).tocsc()


def direct_solve(P, seed, alpha=0.85):
    """Per-view direct LU baseline, including factorization."""
    P = sparse.csc_matrix(P)
    A = (sparse.eye(P.shape[0], format="csc") - alpha * P).tocsc()
    return splu(A).solve((1.0 - alpha) * np.asarray(seed, dtype=float))


def power_iteration(P, seed, alpha=0.85, tol=1e-10, x0=None, maxiter=10000,
                    return_iterations=False):
    """Cold/warm iteration; stop at L1 residual <= tol*(1-alpha)."""
    seed = np.asarray(seed, dtype=float)
    x = seed.copy() if x0 is None else np.asarray(x0, dtype=float).copy()
    teleport = (1.0 - alpha) * seed
    for iteration in range(1, maxiter + 1):
        new = alpha * (P @ x) + teleport
        delta = float(np.linalg.norm(new - x, ord=1))
        x = new
        if delta <= tol * (1.0 - alpha):
            return (x, iteration) if return_iterations else x
    raise RuntimeError(f"power iteration did not converge within {maxiter} iterations")
