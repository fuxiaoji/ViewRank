"""Execution refinements layered over the frozen V6 representation.

The deterministic selection primitive is suitable for every comparison
method. The sparse family also shares its fixed numerical pad across target
strengths and materializes each nonzero completion direction only once.
The cached float64 pad is charged as eight retained family bytes.
"""
from importlib import import_module
from time import perf_counter

import numpy as np

from .compiled_policy_cache import (CompiledDenseDirectory, CompiledDenseFamily,
    CompiledFastFamily, CompiledStencilDirectory)
from .packed_policy_cache import PackedStencilDirectory


def top_indices(scores, k):
    """Descending scores, ascending IDs at ties, without unconditional scans.

    A partition containing k+1 nodes detects whether the requested boundary
    is tied. A separated boundary needs only a local sort; a tied boundary
    uses the complete vector to choose the smallest equal-score node IDs.
    """
    n = len(scores)
    k = min(int(k), n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k == n:
        return np.lexsort((np.arange(n), -scores))
    cut = n-k-1
    chosen = np.argpartition(scores, cut)[cut:]
    chosen = chosen[np.lexsort((chosen, -scores[chosen]))]
    if scores[chosen[k-1]] != scores[chosen[k]]:
        return chosen[:k]
    cutoff = scores[chosen[k-1]]
    better = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[:k-len(better)]
    ids = np.concatenate([better, tied])
    return ids[np.lexsort((ids, -scores[ids]))]


def bind_common_selection():
    """Give every production comparison the identical selection primitive.

    Imports by value retain their own module binding; update each explicitly.
    The caller's driver namespace still needs ``top_indices = top_indices``
    from this module because it is outside this list of engine modules.
    """
    names = (
        'src.policy_family_fast', 'src.full_scan_policy_family',
        'src.compiled_policy_cache', 'src.policy_box',
        'src.baselines.certified_base', 'src.baselines.certified_native_family',
        'src.baselines.dense_family_budget', 'src.baselines.native_offset_family',
    )
    for name in names:
        import_module(name).top_indices = top_indices
    return names


def family_padding(family):
    if hasattr(family, '_base_pad'):
        return family._base_pad
    return 64*np.finfo(float).eps*(1+np.linalg.norm(family.y, 1))/(1-family.ranker.alpha)


def complete_family(family, d, c, ks, started, initial, materialize):
    """Same completion equation and bound, one CSC materialization per atom."""
    r = family.ranker
    if hasattr(family.directory, 'merge_rhs'):
        rhs = family.directory.merge_rhs(family.atoms, d)
    else:
        rhs = np.zeros(r.n)
        for value, atom in zip(d, family.atoms):
            if value:
                delta = atom.delta
                rhs[delta.indices] -= r.alpha*value*delta.data
    nonzero = bool(np.any(d))
    if nonzero:
        correction = r.base_lu.solve(rhs)
        measured = getattr(r.base_lu, 'last_residuals', None)
        residual = (float(measured[0]) if measured is not None else
                    float(np.linalg.norm(rhs-r.A@correction, 1)))
        scores = family.y-correction
    else:
        scores = family.y.copy()
        residual = 0.
    discrepancy = scores[family.sources]-c
    if len(c):
        scores[family.sources] = c
    bound = (family.base_residual+residual+(1+r.alpha)*np.linalg.norm(discrepancy, 1))/(1-r.alpha)
    bound += family_padding(family)
    order = top_indices(scores, min(max(ks)+1, r.n))
    result = dict(sets={k: order[:k] for k in ks},
        selected_scores={k: scores[order[:k]] for k in ks},
        initial_certified=initial, fallback_solve_calls=int(nonzero),
        seconds=perf_counter()-started, payload_bytes=family.nbytes,
        candidate_nodes=r.n, numerical_allowance=bound,
        completion_certified={k: (k == r.n or scores[order[k-1]]-scores[order[k]] > 2*bound) for k in ks})
    if materialize:
        result.update(scores=scores, error_bounds=np.full(r.n, bound))
    return result


class EfficientCompiledFastFamily(CompiledFastFamily):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_pad = 64*np.finfo(float).eps*(1+np.linalg.norm(self.y, 1))/(1-self.ranker.alpha)

    _complete = complete_family

    def solve(self, strengths, ks=None, always_fused=False, materialize=False):
        started = perf_counter()
        ks = self.ks if ks is None else tuple(ks)
        r = self.ranker
        a, c, d, K = self.coefficients(strengths)
        if always_fused:
            return self._complete(d, c, ks, started, {}, materialize)
        contributions = np.array([abs(v)*atom.tail_inf for v, atom in zip(d, self.atoms)])
        tail = float(contributions.sum())
        scores = self.y[self.rows].copy()
        errors = np.full(len(self.rows), tail)
        for value, atom, pos, contribution in zip(d, self.atoms, self.atom_positions, contributions):
            scores[pos] -= value*atom.values
            errors[pos] -= contribution
        if len(d):
            scores[self.boundary_positions] = self.y[self.boundary_sources]-self.boundary_matrix@d
            scores[self.source_positions] = c
        errors[self.boundary_positions] = 0
        np.maximum(errors, 0, out=errors)
        h = self.y[self.sources]-K@c
        h_inf = float(np.max(np.abs(h))) if len(h) else 0.
        residual = self.base_residual+sum(abs(v)*atom.residual_l1 for v, atom in zip(d, self.atoms))
        residual += sum(abs(v*scale)*atom.update_l1 for v, scale, atom in zip(h, a, self.atoms))
        numerical = residual/(1-r.alpha)+h_inf+self._base_pad
        errors += numerical
        order = top_indices(scores, max(ks))
        lower = np.minimum.accumulate(scores[order]-errors[order])
        upper = scores+errors
        margins = {}
        previous = 0
        for k in sorted(ks):
            upper[order[previous:k]] = -np.inf
            outside = max(float(np.max(upper)), self.outside_base+tail+numerical)
            margins[k] = float(lower[k-1]-outside)
            previous = k
        initial = {k: margin > 0 for k, margin in margins.items()}
        if not all(initial.values()) and np.any(d):
            return self._complete(d, c, ks, started, initial, materialize)
        result = dict(sets={k: self.rows[order[:k]] for k in ks},
            selected_scores={k: scores[order[:k]] for k in ks}, initial_certified=initial,
            initial_margins=margins, fallback_solve_calls=0, seconds=perf_counter()-started,
            payload_bytes=self.nbytes, candidate_nodes=len(self.rows))
        if materialize:
            full = self.y.copy()
            full[self.rows] = scores
            bounds = np.full(r.n, tail+numerical)
            bounds[self.rows] = errors
            result.update(scores=full, error_bounds=bounds)
        return result


class EfficientCompiledDenseFamily(CompiledDenseFamily):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.route != 'native':
            self._base_pad = 64*np.finfo(float).eps*(1+np.linalg.norm(self.y, 1))/(1-self.ranker.alpha)


class _CachedPaddingBudget:
    def prepare(self, *args, **kwargs):
        # The frozen constructor does not retain a pad for full-scan families.
        reserve = 0 if kwargs.get('full_scan', False) else 8
        self.budget_bytes -= reserve
        try:
            family = super().prepare(*args, **kwargs)
        finally:
            self.budget_bytes += reserve
        if hasattr(family, '_base_pad'):
            family._metadata_bytes += 8
            self._live_core_bytes += 8
            self._track()
            assert self.nbytes <= self.budget_bytes
        return family


class EfficientStencilDirectory(_CachedPaddingBudget, CompiledStencilDirectory):
    pass


class EfficientPackedStencilDirectory(_CachedPaddingBudget, PackedStencilDirectory):
    pass


class EfficientDenseDirectory(_CachedPaddingBudget, CompiledDenseDirectory):
    pass
