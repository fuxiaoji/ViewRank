"""Sparse coordinate error envelopes with exact graph-view boundary coupling."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import lu_solve

from .boundary import BoundaryView, prepare_boundary
from .certified import CertifiedResult, _topk
from .viewrank import _sparse_bytes


@dataclass
class EnvelopeView(BoundaryView):
    response_residual_l1: np.ndarray
    update_colnorm_l1: np.ndarray

    @property
    def nbytes(self):
        return super().nbytes + self.response_residual_l1.nbytes + self.update_colnorm_l1.nbytes


@dataclass
class EnvelopeResult(CertifiedResult):
    error_bounds: np.ndarray
    interval_margin: float
    numerical_allowance: float


def prepare_envelope(ranker, prepared, keep_per_column=32):
    started = perf_counter()
    base = prepare_boundary(ranker, prepared, keep_per_column)
    # Boundary scores equal c exactly in real arithmetic; no tail budget is
    # needed there, even when a boundary entry was not retained in Zhat.
    for j in range(prepared.rank):
        tail = prepared.Z[:, j] - base.Zhat[:, j].toarray().ravel()
        tail[base.sources] = 0.0
        base.tail_inf[j] = np.max(np.abs(tail))
    U = -ranker.alpha * prepared.D
    response_residual = np.asarray(abs(U - ranker.A @ prepared.Z).sum(axis=0)).ravel()
    update_norm = np.asarray(abs(U).sum(axis=0)).ravel()
    result = EnvelopeView(**vars(base), response_residual_l1=response_residual,
                          update_colnorm_l1=update_norm)
    result.prepare_seconds = perf_counter() - started
    return result


def envelope_topk(ranker, seed, compressed, k=10, reference_scores=None, base_scores=None):
    """Certify a top-k set by separating coordinate lower/upper envelopes.

    e_i=sum_j |c_j| tau_j for columns whose i-th entry was discarded, and
    e_i=0 on exact boundary coordinates. Retained entries have no tail error.
    A candidate set L is certified if min_L(xhat-e)>max_outside_L(xhat+e).

    Floating LU residuals receive a separate common allowance. With ry=b-Ay,
    RZ=U-AZ and h=y[S]-Kc, the uncompressed numerical solution has residual
    ry-RZ*c-U*h, bounded in L1 by the three corresponding absolute sums.
    Its score error is at most this norm bound/(1-alpha). Boundary overwrite
    contributes an additional ||h||infinity. Arithmetic used to evaluate these
    bounds is still finite precision; the final scalar includes a rounding pad.
    """
    started = perf_counter()
    if not 1 <= k <= ranker.n:
        raise ValueError("k must lie between one and the number of graph nodes")
    seed = np.asarray(seed, dtype=float)
    y = (ranker.base_lu.solve((1-ranker.alpha)*seed)
         if base_scores is None else np.asarray(base_scores, dtype=float))
    rank = len(compressed.sources)
    c = lu_solve(compressed.small_lu, y[compressed.sources]) if rank else np.empty(0)
    scores = np.asarray(y-compressed.Zhat@c)
    scores[compressed.sources] = c
    contributions = np.abs(c)*compressed.tail_inf
    envelope = np.full(ranker.n, contributions.sum())
    columns = [compressed.Zhat[:, j] for j in range(rank)]
    for j, column in enumerate(columns):
        envelope[column.indices] -= contributions[j]
    envelope[compressed.sources] = 0.0
    np.maximum(envelope, 0.0, out=envelope)
    ry_norm = float(np.linalg.norm((1-ranker.alpha)*seed-ranker.A@y, ord=1))
    residual_columns = compressed.response_residual_l1.copy()
    h = y[compressed.sources]-compressed.K@c
    restored = np.zeros(rank, dtype=bool)
    refined = []
    initial_certified = False
    exact_fallback = False
    passes = 0
    while True:
        passes += 1
        h_inf = float(np.max(np.abs(h))) if rank else 0.0
        numerical_allowance = (ry_norm+np.dot(np.abs(c), residual_columns)
                               +np.dot(np.abs(h), compressed.update_colnorm_l1))/(1-ranker.alpha)+h_inf
        numerical_allowance += (64*np.finfo(float).eps*(1+np.linalg.norm(y, ord=1))
                                *passes/(1-ranker.alpha))
        error_bounds = envelope+numerical_allowance
        order = _topk(scores, k)
        lower = float(np.min(scores[order]-error_bounds[order]))
        upper_values = scores+error_bounds
        upper_values[order] = -np.inf
        upper = float(np.max(upper_values))
        interval_margin = lower-upper
        if interval_margin > 0:
            initial_certified = passes == 1
            break
        if restored.all():
            exact_fallback = True
            break
        priority = contributions.copy()
        priority[restored] = -1.0
        j = int(np.argmax(priority))
        rhs = (-ranker.alpha*compressed.D[:, j]).toarray().ravel()
        full = ranker.base_lu.solve(rhs)
        scores -= c[j]*(full-columns[j].toarray().ravel())
        scores[compressed.sources] = c
        # Replace the old compressed column error by zero, using only its
        # sparse retained coordinates; no n-by-r envelope is materialized.
        envelope -= contributions[j]
        envelope[columns[j].indices] += contributions[j]
        envelope[compressed.sources] = 0.0
        np.maximum(envelope, 0.0, out=envelope)
        old_boundary = compressed.K[:, j].copy()
        old_boundary[j] -= 1.0
        h -= c[j]*(full[compressed.sources]-old_boundary)
        residual_columns[j] = np.linalg.norm(rhs-ranker.A@full, ord=1)
        columns[j] = sparse.csc_matrix(full.reshape(-1, 1))
        contributions[j] = 0.0
        restored[j] = True
        refined.append(int(compressed.sources[j]))
    response_bytes = _sparse_bytes(sparse.hstack(columns, format="csc")) if rank else compressed.response_bytes
    error = None if reference_scores is None else float(np.max(np.abs(scores-np.asarray(reference_scores))))
    return EnvelopeResult(
        scores=scores, indices=order, bound=float(error_bounds.max()),
        initial_certified=initial_certified, refined_columns=len(refined),
        refined_sources=tuple(refined), response_bytes=response_bytes,
        working_bytes=compressed.nbytes+response_bytes+3*ranker.n*8,
        solve_seconds=perf_counter()-started, exact_fallback=exact_fallback,
        score_error=error, passes=passes, error_bounds=error_bounds,
        interval_margin=interval_margin, numerical_allowance=numerical_allowance,
    )
