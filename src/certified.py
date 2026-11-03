"""Compressed view responses with residual certificates for exact top-k sets.

The compressed object retains no dense exact responses. A failed certificate
restores response columns on demand using the shared base factorization.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import lu_factor, lu_solve

from .viewrank import _sparse_bytes


@dataclass
class CompressedView:
    D: sparse.csc_matrix
    sources: np.ndarray
    Zhat: sparse.csc_matrix
    rho: np.ndarray
    small_lu: tuple | None
    prepare_seconds: float
    fingerprint: str

    @property
    def response_bytes(self):
        return _sparse_bytes(self.Zhat)

    @property
    def nbytes(self):
        small_bytes = 0 if self.small_lu is None else sum(a.nbytes for a in self.small_lu)
        return _sparse_bytes(self.D) + self.sources.nbytes + self.rho.nbytes + self.response_bytes + small_bytes


@dataclass
class CertifiedResult:
    scores: np.ndarray
    indices: np.ndarray
    bound: float
    initial_certified: bool
    refined_columns: int
    refined_sources: tuple
    response_bytes: int
    working_bytes: int
    solve_seconds: float
    exact_fallback: bool
    score_error: float | None
    passes: int


def _topk(scores, k):
    # Stable node-ID tie breaking; the certificate guarantees set membership.
    return np.lexsort((np.arange(len(scores)), -scores))[:k]


def prepare_compressed(ranker, prepared, keep_per_column=32):
    """Keep the largest absolute responses per update column.

    Preparation uses an exact PreparedView, but the returned object does not
    reference that object or its Z. Response/object memory is exposed separately
    from any exact benchmark reference retained by the caller.
    """
    started = perf_counter()
    keep = min(max(int(keep_per_column), 0), ranker.n)
    columns = []
    for j in range(prepared.rank):
        values = prepared.Z[:, j]
        rows = np.argpartition(np.abs(values), -keep)[-keep:] if keep else np.empty(0, dtype=int)
        column = sparse.csc_matrix((values[rows], (rows, np.zeros(len(rows), dtype=int))), shape=(ranker.n, 1))
        column.eliminate_zeros()
        columns.append(column)
    Zhat = sparse.hstack(columns, format="csc") if columns else sparse.csc_matrix((ranker.n, 0))
    U = -ranker.alpha * prepared.D
    rho = np.asarray(abs(U - ranker.A @ Zhat).sum(axis=0)).ravel()
    small = np.eye(prepared.rank) + Zhat[prepared.sources, :].toarray()
    small_lu = lu_factor(small) if prepared.rank else None
    return CompressedView(
        D=prepared.D.copy(), sources=prepared.sources.copy(), Zhat=Zhat,
        rho=rho, small_lu=small_lu, prepare_seconds=perf_counter() - started,
        fingerprint=prepared.fingerprint,
    )


def certified_topk(ranker, seed, compressed, k=10, reference_scores=None, base_scores=None):
    """Return an exact top-k set, with adaptive response-column restoration.

    For stochastic P_view, ||(I-alpha P_view)^-1||_1=1/(1-alpha).
    Writing U=-alpha D, R=U-A Zhat, c=(I+E.T Zhat)^-1 E.T y,
    xhat=y-Zhat c has L1 and Linfinity error bounded by
    sum_j |c_j| ||R_j||_1/(1-alpha), up to floating point roundoff.
    The top-k set is certified whenever its boundary gap exceeds twice this
    bound. At full restoration deterministic exact top-k resolves ties.
    """
    started = perf_counter()
    if not 1 <= k <= ranker.n:
        raise ValueError("k must lie between one and the number of graph nodes")
    rank = len(compressed.sources)
    seed = np.asarray(seed, dtype=float)
    y = (ranker.base_lu.solve((1.0 - ranker.alpha) * seed)
         if base_scores is None else np.asarray(base_scores, dtype=float))
    columns = [compressed.Zhat[:, j] for j in range(rank)]
    rho = compressed.rho.copy()
    Zhat = compressed.Zhat
    restored = np.zeros(rank, dtype=bool)
    refined = []
    initial_certified = False
    exact_fallback = False
    passes = 0
    while True:
        passes += 1
        if rank:
            if passes == 1:
                coefficients = lu_solve(compressed.small_lu, y[compressed.sources])
            else:
                small = np.eye(rank) + Zhat[compressed.sources, :].toarray()
                coefficients = np.linalg.solve(small, y[compressed.sources])
            scores = y - Zhat @ coefficients
            contributions = np.abs(coefficients) * rho
            bound = float(contributions.sum() / (1.0 - ranker.alpha))
        else:
            scores = y
            contributions = np.empty(0)
            bound = 0.0
        order = _topk(scores, min(k + 1, ranker.n))
        gap = float(scores[order[k - 1]] - scores[order[k]]) if k < ranker.n else float("inf")
        if gap > 2.0 * bound:
            initial_certified = passes == 1
            break
        if restored.all():
            exact_fallback = True
            break
        contributions[restored] = -1.0
        j = int(np.argmax(contributions))
        rhs = (-ranker.alpha * compressed.D[:, j]).toarray().ravel()
        full = ranker.base_lu.solve(rhs)
        columns[j] = sparse.csc_matrix(full.reshape(-1, 1))
        rho[j] = np.linalg.norm(rhs - ranker.A @ full, ord=1)
        restored[j] = True
        refined.append(int(compressed.sources[j]))
        Zhat = sparse.hstack(columns, format="csc")
    response_bytes = _sparse_bytes(Zhat)
    score_error = None if reference_scores is None else float(np.max(np.abs(scores - np.asarray(reference_scores))))
    return CertifiedResult(
        scores=np.asarray(scores), indices=order[:k], bound=bound,
        initial_certified=initial_certified, refined_columns=len(refined),
        refined_sources=tuple(refined), response_bytes=response_bytes,
        working_bytes=compressed.nbytes + response_bytes + y.nbytes + scores.nbytes,
        solve_seconds=perf_counter() - started, exact_fallback=exact_fallback,
        score_error=score_error, passes=passes,
    )
