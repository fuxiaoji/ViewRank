"""Offset Score Propagation, independently implemented from Yoon et al. (2018).

Source: https://woojeongjin.github.io/papers/ospWWW18.pdf, section 3.1.
With column-stochastic P and continuation alpha, q=alpha*(Pv-P)*y,
and the correction is sum_{i>=0}(alpha*Pv)^i*q.  This implementation
includes the zeroth term stated in the paper's series (Algorithm 1's
indexing omits it). It is not the authors' original implementation.

OSP and OSP-T share this recurrence; epsilon controls truncation. For a
column-stochastic matrix, the remaining L1 tail after the included last
term is at most alpha*||term||_1/(1-alpha), also for signed offsets.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class OffsetResult:
    scores: np.ndarray
    iterations: int
    first_converged_iteration: np.ndarray
    final_term_l1: np.ndarray
    tail_bound_l1: np.ndarray
    initial_offset_l1: np.ndarray
    mode: str


def offset_seed(D, sources, base_scores, alpha=0.85):
    """Exploit column-local changes without scanning two complete matrices."""
    return alpha * (D @ np.asarray(base_scores)[np.asarray(sources, dtype=np.int64)])


def propagate_offset(Pview, base_scores, offset, alpha=0.85, epsilon=1.5e-11,
                     mode="dense", maxiter=10000):
    """Accumulate the OSP signed series, retaining all nonzero frontier entries.

    Dense mode uses batched sparse/dense products. Sparse mode uses exact CSC
    frontier products, with no threshold pruning or changed stopping rule.
    Both accept one vector or a matrix with one seed per column. A batch runs
    until every column's term norm meets epsilon; first crossing is diagnostic.
    """
    base = np.asarray(base_scores, dtype=np.float64)
    single = base.ndim == 1
    base = base[:, None] if single else base
    q = np.asarray(offset, dtype=np.float64)
    q = q[:, None] if q.ndim == 1 else q
    initial_norms = np.sum(np.abs(q), axis=0)
    norms = initial_norms.copy()
    first = np.full(base.shape[1], -1, dtype=np.int64)
    first[norms <= epsilon] = 0
    correction = q.copy()
    iteration = 0
    if mode == "dense":
        transition = sparse.csr_matrix(Pview)
        term = q.copy()
        while float(norms.max()) > epsilon and iteration < maxiter:
            term = alpha * (transition @ term)
            correction += term
            iteration += 1
            norms = np.sum(np.abs(term), axis=0)
            first[(first < 0) & (norms <= epsilon)] = iteration
    elif mode == "sparse":
        transition = sparse.csc_matrix(Pview)
        term = sparse.csc_matrix(q)
        while float(norms.max()) > epsilon and iteration < maxiter:
            term = alpha * (transition @ term)
            term.eliminate_zeros()
            coordinates = term.tocoo(copy=False)
            correction[coordinates.row, coordinates.col] += coordinates.data
            iteration += 1
            norms = np.asarray(abs(term).sum(axis=0)).ravel()
            first[(first < 0) & (norms <= epsilon)] = iteration
    else:
        raise ValueError("mode must be dense or sparse")
    if float(norms.max()) > epsilon:
        raise RuntimeError("OSP did not meet its term-norm stopping threshold")
    scores = base + correction
    return OffsetResult(
        scores=scores[:, 0] if single else scores,
        iterations=iteration, first_converged_iteration=first,
        final_term_l1=norms, tail_bound_l1=alpha * norms / (1 - alpha),
        initial_offset_l1=initial_norms, mode=mode,
    )


def osp_update(Pview, D, sources, base_scores, alpha=0.85, tol=1e-10,
               mode="dense"):
    """OSP-T at epsilon=(1-alpha)*tol, giving an L1 tail bound <= alpha*tol."""
    q = offset_seed(D, sources, base_scores, alpha)
    return propagate_offset(Pview, base_scores, q, alpha=alpha,
                            epsilon=(1 - alpha) * tol, mode=mode)
