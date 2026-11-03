"""Boundary-exact response compression for certifiable view-diffusion top-k.

Retaining the exact r-by-r boundary coupling keeps query coefficients exact.
Only the n-by-r output response is truncated, yielding a direct tail bound.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import lu_solve

from .viewrank import _sparse_bytes
from .certified import CertifiedResult, _topk


@dataclass
class BoundaryView:
    D: sparse.csc_matrix
    sources: np.ndarray
    Zhat: sparse.csc_matrix
    tail_inf: np.ndarray
    K: np.ndarray
    small_lu: tuple | None
    prepare_seconds: float
    fingerprint: str

    @property
    def response_bytes(self):
        return _sparse_bytes(self.Zhat)

    @property
    def nbytes(self):
        small_bytes = 0 if self.small_lu is None else sum(a.nbytes for a in self.small_lu)
        return (_sparse_bytes(self.D) + self.sources.nbytes + self.tail_inf.nbytes
                + self.response_bytes + self.K.nbytes + small_bytes)


def prepare_boundary(ranker, prepared, keep_per_column=32):
    """Compress only output coordinates; retain the exact source coupling K."""
    started = perf_counter()
    keep = min(max(int(keep_per_column), 0), ranker.n)
    columns, tails = [], []
    for j in range(prepared.rank):
        values = prepared.Z[:, j]
        rows = np.argpartition(np.abs(values), -keep)[-keep:] if keep else np.empty(0, dtype=int)
        column = sparse.csc_matrix((values[rows], (rows, np.zeros(len(rows), dtype=int))), shape=(ranker.n, 1))
        column.eliminate_zeros()
        columns.append(column)
        tails.append(float(np.max(np.abs(values - column.toarray().ravel()))))
    Zhat = sparse.hstack(columns, format="csc") if columns else sparse.csc_matrix((ranker.n, 0))
    return BoundaryView(
        D=prepared.D.copy(), sources=prepared.sources.copy(), Zhat=Zhat,
        tail_inf=np.asarray(tails), K=prepared.K.copy(),
        small_lu=tuple(a.copy() for a in prepared.small_lu) if prepared.small_lu is not None else None,
        prepare_seconds=perf_counter() - started, fingerprint=prepared.fingerprint,
    )


def boundary_topk(ranker, seed, compressed, k=10, reference_scores=None, base_scores=None):
    """Certify exact top-k SETS using exact coefficients and output-tail bounds.

    c=Kexact^-1 y[sources] is unchanged by compression. Consequently
    ||x_exact - (y-Zhat c)||_infinity <= sum_j |c_j| tail_inf[j].
    An uncertified boundary restores the largest contributing response column;
    coefficients remain unchanged and scores are updated by a rank-one delta.
    """
    started = perf_counter()
    if not 1 <= k <= ranker.n:
        raise ValueError("k must lie between one and the number of graph nodes")
    y = (ranker.base_lu.solve((1.0 - ranker.alpha) * np.asarray(seed, dtype=float))
         if base_scores is None else np.asarray(base_scores, dtype=float))
    rank = len(compressed.sources)
    coefficients = (lu_solve(compressed.small_lu, y[compressed.sources])
                    if rank else np.empty(0))
    scores = np.asarray(y - compressed.Zhat @ coefficients)
    contributions = np.abs(coefficients) * compressed.tail_inf
    columns = [compressed.Zhat[:, j] for j in range(rank)]
    restored = np.zeros(rank, dtype=bool)
    refined = []
    initial_certified = False
    exact_fallback = False
    passes = 0
    while True:
        passes += 1
        bound = float(contributions.sum())
        order = _topk(scores, min(k + 1, ranker.n))
        gap = float(scores[order[k - 1]] - scores[order[k]]) if k < ranker.n else float("inf")
        if gap > 2.0 * bound:
            initial_certified = passes == 1
            break
        if restored.all():
            exact_fallback = True
            break
        priority = contributions.copy()
        priority[restored] = -1
        j = int(np.argmax(priority))
        rhs = (-ranker.alpha * compressed.D[:, j]).toarray().ravel()
        full = ranker.base_lu.solve(rhs)
        scores -= coefficients[j] * (full - columns[j].toarray().ravel())
        columns[j] = sparse.csc_matrix(full.reshape(-1, 1))
        contributions[j] = 0.0
        restored[j] = True
        refined.append(int(compressed.sources[j]))
    response_bytes = _sparse_bytes(sparse.hstack(columns, format="csc")) if rank else compressed.response_bytes
    score_error = None if reference_scores is None else float(np.max(np.abs(scores - np.asarray(reference_scores))))
    return CertifiedResult(
        scores=scores, indices=order[:k], bound=bound,
        initial_certified=initial_certified, refined_columns=len(refined),
        refined_sources=tuple(refined), response_bytes=response_bytes,
        working_bytes=compressed.nbytes + response_bytes + y.nbytes + scores.nbytes,
        solve_seconds=perf_counter() - started, exact_fallback=exact_fallback,
        score_error=score_error, passes=passes,
    )
