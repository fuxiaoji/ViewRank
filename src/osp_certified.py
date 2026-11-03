"""OSP with deterministic top-k early stopping at the same retrieval target.

The published OSP offset series is the prior algorithm. This module adds the
standard L1-tail/gap stopping rule so its baseline need not solve every score
to a uniform tolerance when only top-k membership is requested.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class CertifiedOSPResult:
    scores: np.ndarray
    indices: np.ndarray
    certified: np.ndarray
    certification_step: np.ndarray
    finish_step: np.ndarray
    final_error_bound: np.ndarray
    iterations: int
    initial_offset_l1: np.ndarray


def osp_topk(Pview, D, sources, base_scores, k=10, alpha=.85, tol=1e-10,
             base_residual_l1=None, maxiter=10000):
    """Return certified sets or a separately marked numerical fallback.

    q0=alpha D*y[S], qt=(alpha Pview)^t q0. After including qt, the
    uncomputed tail has L1 norm <= alpha*||qt||1/(1-alpha). A base LU
    residual allowance is added. Completed columns are frozen at their first
    certificate or numerical stopping step, while unfinished queries continue
    in the same sparse/dense batch. Offset construction and every top-k check
    happen inside this function.
    """
    y = np.asarray(base_scores, dtype=float)
    single = y.ndim == 1
    y = y[:, None] if single else y
    n, queries = y.shape
    if not 1 <= k <= n:
        raise ValueError("k must lie between one and the number of nodes")
    residual = np.zeros(queries) if base_residual_l1 is None else np.broadcast_to(np.asarray(base_residual_l1), (queries,))
    base_allowance = residual / (1 - alpha)
    transition = sparse.csr_matrix(Pview)
    term = alpha * (sparse.csc_matrix(D) @ y[np.asarray(sources, dtype=int)])
    scores = y + term
    initial_norms = np.sum(np.abs(term), axis=0)
    indices = np.empty((k, queries), dtype=np.int64)
    certified = np.zeros(queries, dtype=bool)
    cert_step = np.full(queries, -1, dtype=np.int64)
    finish_step = np.full(queries, -1, dtype=np.int64)
    saved_bounds = np.full(queries, np.inf)
    active = np.ones(queries, dtype=bool)
    epsilon = (1 - alpha) * tol
    iteration = 0
    while True:
        active_ids = np.flatnonzero(active)
        norms = np.sum(np.abs(term[:, active_ids]), axis=0)
        bounds = alpha / (1 - alpha) * norms + base_allowance[active_ids]
        # The theoretical bound is exact-arithmetic. Pad ordinary floating
        # summation/partial sorting without claiming interval arithmetic.
        bounds += 64 * np.finfo(float).eps * (1 + np.sum(np.abs(scores[:, active_ids]), axis=0)) / (1 - alpha)
        for local_j, query in enumerate(active_ids):
            values = scores[:, query]
            if k < n:
                candidate = np.argpartition(-values, k)[: k + 1]
                order = candidate[np.lexsort((candidate, -values[candidate]))]
                gap = float(values[order[k - 1]] - values[order[k]])
            else:
                order = np.lexsort((np.arange(n), -values))
                gap = float("inf")
            proven = gap > 2 * bounds[local_j]
            numeric_done = norms[local_j] <= epsilon
            if proven or numeric_done:
                if numeric_done and not proven:
                    # Stable complete sorting gives deterministic fallback ties.
                    order = np.lexsort((np.arange(n), -values))
                indices[:, query] = order[:k]
                certified[query] = proven
                cert_step[query] = iteration if proven else -1
                finish_step[query] = iteration
                saved_bounds[query] = bounds[local_j]
                active[query] = False
                term[:, query] = 0.0
        if not active.any():
            break
        if iteration >= maxiter:
            raise RuntimeError("OSP top-k did not finish within maxiter")
        term = alpha * (transition @ term)
        scores[:, active] += term[:, active]
        iteration += 1
    return CertifiedOSPResult(
        scores=scores[:, 0] if single else scores,
        indices=indices[:, 0] if single else indices,
        certified=certified, certification_step=cert_step,
        finish_step=finish_step, final_error_bound=saved_bounds,
        iterations=iteration, initial_offset_l1=initial_norms,
    )
