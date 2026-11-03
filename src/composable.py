"""Composable compressed atomic responses for previously unseen graph views.

The resident directory never retains an n-vector response. For each atomic
normalized column update d at source s it stores z[B], sparse top-|z| entries,
a coordinate tail bound, d itself, and the measured base-LU residual norm.
Arbitrary distinct-source combinations assemble their exact small coupling
from z[B]. An uncertified output needs one solve for the merged RHS, not one
solve per missing response column.
"""
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import lu_factor, lu_solve

from .certified import _topk
from .viewrank import _sparse_bytes


@dataclass
class AtomicResponse:
    source: int
    fingerprint: str
    delta: sparse.csc_matrix
    boundary: np.ndarray
    rows: np.ndarray
    values: np.ndarray
    tail_inf: float
    residual_l1: float
    update_l1: float
    prepare_seconds: float
    solve_calls: int

    @property
    def nbytes(self):
        return (_sparse_bytes(self.delta) + self.boundary.nbytes + self.rows.nbytes
                + self.values.nbytes + 3 * 8 + 8)

    @property
    def response_bytes(self):
        return self.boundary.nbytes + self.rows.nbytes + self.values.nbytes + 3 * 8


@dataclass
class ComposedView:
    atoms: tuple
    sources: np.ndarray
    K: np.ndarray
    small_lu: tuple | None
    fingerprint: str
    new_whole_view: bool
    atom_hits: int
    atom_misses: int
    atomic_solve_calls: int
    atomic_prepare_seconds: float
    compose_seconds: float

    @property
    def rank(self):
        return len(self.sources)

    @property
    def nbytes(self):
        return (self.sources.nbytes + self.K.nbytes
                + (sum(a.nbytes for a in self.small_lu) if self.small_lu is not None else 0))


@dataclass
class ComposableResult:
    scores: np.ndarray
    indices: np.ndarray
    coefficients: np.ndarray
    initial_certified: bool
    fallback_one_solve: bool
    fallback_solve_calls: int
    bound: float
    error_bounds: np.ndarray
    numerical_allowance: float
    interval_margin: float
    solve_seconds: float
    directory_bytes: int
    working_bytes: int


class ComposableBoundaryDirectory:
    def __init__(self, ranker, boundary_sources, keep_per_atom=32):
        self.ranker = ranker
        self.B = np.asarray(boundary_sources, dtype=np.int64).copy()
        if len(set(self.B.tolist())) != len(self.B):
            raise ValueError("boundary source universe must contain distinct nodes")
        self.positions = {int(source): index for index, source in enumerate(self.B)}
        self.keep = min(max(int(keep_per_atom), 0), ranker.n)
        self.cache = {}
        self.hits = self.misses = self.atomic_solve_calls = 0
        self.atomic_prepare_seconds = 0.0
        self.whole_views = set()
        self._payload_bytes = self.B.nbytes
        self._response_bytes = self.B.nbytes

    @property
    def nbytes(self):
        """Array payload only; the shared base factor/P and Python overhead excluded."""
        return self._payload_bytes

    @property
    def response_bytes(self):
        return self._response_bytes

    def atomic(self, source, delta):
        source = int(source)
        if source not in self.positions:
            raise ValueError("atomic source lies outside the boundary universe")
        column = sparse.csc_matrix(np.asarray(delta).reshape(-1, 1)) if not sparse.issparse(delta) else sparse.csc_matrix(delta)
        column = column.copy()
        column.sum_duplicates()
        column.eliminate_zeros()
        column.sort_indices()
        if column.shape != (self.ranker.n, 1):
            raise ValueError("an atomic delta must contain one graph column")
        digest = sha256(np.asarray([self.ranker.n, source], dtype=np.int64).tobytes())
        digest.update(column.indptr.tobytes())
        digest.update(column.indices.tobytes())
        digest.update(column.data.tobytes())
        fingerprint = digest.hexdigest()
        if fingerprint in self.cache:
            self.hits += 1
            return self.cache[fingerprint]
        self.misses += 1
        started = perf_counter()
        rhs = (-self.ranker.alpha * column).toarray().ravel()
        if column.nnz:
            response = self.ranker.base_lu.solve(rhs)
            solve_calls = 1
        else:
            response = np.zeros(self.ranker.n)
            solve_calls = 0
        measured_residuals = getattr(self.ranker.base_lu, 'last_residuals', None) if solve_calls else np.array([0.])
        response_residual = (float(measured_residuals[0]) if measured_residuals is not None else
                             float(np.linalg.norm(rhs-self.ranker.A@response,ord=1)))
        rows = (np.argpartition(np.abs(response), -self.keep)[-self.keep:]
                if self.keep else np.empty(0, dtype=np.int64))
        rows = np.sort(rows)
        rows = rows[response[rows] != 0]
        omitted = np.ones(self.ranker.n, dtype=bool)
        omitted[rows] = False
        omitted[self.B] = False
        atom = AtomicResponse(
            source=source, fingerprint=fingerprint, delta=column,
            boundary=response[self.B].copy(), rows=rows.copy(), values=response[rows].copy(),
            tail_inf=float(np.max(np.abs(response[omitted]))) if omitted.any() else 0.0,
            residual_l1=response_residual,
            update_l1=float(np.linalg.norm(rhs, ord=1)),
            prepare_seconds=perf_counter() - started, solve_calls=solve_calls,
        )
        self.cache[fingerprint] = atom
        self._payload_bytes += atom.nbytes
        self._response_bytes += atom.response_bytes
        self.atomic_solve_calls += solve_calls
        self.atomic_prepare_seconds += atom.prepare_seconds
        return atom

    def compose(self, sources, deltas):
        started = perf_counter()
        sources = np.asarray(sources, dtype=np.int64)
        if len(set(sources.tolist())) != len(sources):
            raise ValueError("combine repeated-source deltas before composition")
        deltas = sparse.csc_matrix(deltas)
        if deltas.shape != (self.ranker.n, len(sources)):
            raise ValueError("deltas must contain one column per source")
        deltas.eliminate_zeros()
        effective = np.flatnonzero(np.diff(deltas.indptr))
        deltas = deltas[:, effective]
        sources = sources[effective]
        order = np.argsort(sources, kind="stable")
        sources = sources[order]
        before_hits, before_misses = self.hits, self.misses
        before_solve, before_setup = self.atomic_solve_calls, self.atomic_prepare_seconds
        atoms = tuple(self.atomic(source, deltas[:, int(j)]) for source, j in zip(sources, order))
        boundary = np.column_stack([atom.boundary for atom in atoms]) if atoms else np.empty((len(self.B), 0))
        locations = np.asarray([self.positions[int(s)] for s in sources], dtype=np.int64)
        K = np.eye(len(sources)) + boundary[locations]
        fingerprint = sha256("|".join(atom.fingerprint for atom in atoms).encode("ascii")).hexdigest()
        new_view = fingerprint not in self.whole_views
        self.whole_views.add(fingerprint)
        return ComposedView(
            atoms=atoms, sources=sources, K=K, small_lu=lu_factor(K) if atoms else None,
            fingerprint=fingerprint, new_whole_view=new_view,
            atom_hits=self.hits - before_hits, atom_misses=self.misses - before_misses,
            atomic_solve_calls=self.atomic_solve_calls - before_solve,
            atomic_prepare_seconds=self.atomic_prepare_seconds - before_setup,
            compose_seconds=perf_counter() - started,
        )

    def topk(self, seed, view, k=10, base_scores=None):
        started = perf_counter()
        ranker = self.ranker
        seed = np.asarray(seed, dtype=float)
        y = (ranker.base_lu.solve((1 - ranker.alpha) * seed)
             if base_scores is None else np.asarray(base_scores, dtype=float))
        coefficients = lu_solve(view.small_lu, y[view.sources]) if view.rank else np.empty(0)
        scores = y.copy()
        contributions = np.asarray([abs(c) * atom.tail_inf for c, atom in zip(coefficients, view.atoms)])
        envelope = np.full(ranker.n, contributions.sum())
        for j, atom in enumerate(view.atoms):
            scores[atom.rows] -= coefficients[j] * atom.values
            envelope[atom.rows] -= contributions[j]
        if view.rank:
            boundaries = np.column_stack([atom.boundary for atom in view.atoms])
            scores[self.B] = y[self.B] - boundaries @ coefficients
            scores[view.sources] = coefficients
        envelope[self.B] = 0
        np.maximum(envelope, 0, out=envelope)
        ry = float(np.linalg.norm((1 - ranker.alpha) * seed - ranker.A @ y, ord=1))
        h = y[view.sources] - view.K @ coefficients
        h_inf = float(np.max(np.abs(h))) if view.rank else 0.0
        residual = sum(abs(c) * atom.residual_l1 for c, atom in zip(coefficients, view.atoms))
        residual += sum(abs(value) * atom.update_l1 for value, atom in zip(h, view.atoms))
        padding = 64 * np.finfo(float).eps * (1 + np.linalg.norm(y, ord=1)) / (1 - ranker.alpha)
        numerical = (ry + residual) / (1 - ranker.alpha) + h_inf + padding
        error_bounds = envelope + numerical
        ids = _topk(scores, k)
        outside = scores + error_bounds
        outside[ids] = -np.inf
        margin = float(np.min(scores[ids] - error_bounds[ids]) - np.max(outside))
        initial = margin > 0
        fallback_calls = 0
        if not initial and view.rank:
            # U=-alpha D. Linearity permits one solve for U*c; no dense atomic
            # response survives, and no whole-view factorization is constructed.
            rhs = np.zeros(ranker.n)
            for c, atom in zip(coefficients, view.atoms):
                rhs[atom.delta.indices] -= ranker.alpha * c * atom.delta.data
            correction = ranker.base_lu.solve(rhs)
            fallback_calls = 1
            scores = y - correction
            scores[view.sources] = coefficients
            merged_residual = (1 - ranker.alpha) * seed - ranker.A @ scores - rhs
            numerical = float(np.linalg.norm(merged_residual, ord=1)) / (1 - ranker.alpha) + padding
            error_bounds = np.full(ranker.n, numerical)
            ids = _topk(scores, k)
            outside = scores + error_bounds
            outside[ids] = -np.inf
            margin = float(np.min(scores[ids] - error_bounds[ids]) - np.max(outside))
        elif not initial:
            scores = y.copy()
            error_bounds = np.full(ranker.n, numerical)
            ids = _topk(scores, k)
        return ComposableResult(
            scores=scores, indices=ids, coefficients=coefficients,
            initial_certified=initial, fallback_one_solve=bool(fallback_calls),
            fallback_solve_calls=fallback_calls, bound=float(error_bounds.max()),
            error_bounds=error_bounds, numerical_allowance=numerical, interval_margin=margin,
            solve_seconds=perf_counter() - started, directory_bytes=self.nbytes,
            working_bytes=self.nbytes + view.nbytes + 6 * ranker.n * 8 + len(self.B) * view.rank * 8,
        )

    def fused_topk(self, seed, view, k=10, base_scores=None):
        """Exact one-merged-solve ablation using the same atoms and exact K.

        This deliberately performs no output approximation or certificate.
        Atomic setup is identical to topk(); a production exact-only variant
        could omit the top-L sidecar entirely.
        """
        ranker = self.ranker
        y = (ranker.base_lu.solve((1 - ranker.alpha) * np.asarray(seed, dtype=float))
             if base_scores is None else np.asarray(base_scores, dtype=float))
        if view.rank:
            c = lu_solve(view.small_lu, y[view.sources])
            rhs = np.zeros(ranker.n)
            for coefficient, atom in zip(c, view.atoms):
                rhs[atom.delta.indices] -= ranker.alpha * coefficient * atom.delta.data
            scores = y - ranker.base_lu.solve(rhs)
            scores[view.sources] = c
        else:
            scores = y.copy()
        return scores, _topk(scores, k)
