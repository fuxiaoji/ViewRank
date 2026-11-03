"""Request-local exact cores for complete, unprojected policy families.

The same original source set is used at every strength. A response direction
is prepared once per nondegenerate source group; the boundary is this request's
complete S, so no global source-coverage restriction is imposed. All requested
k boundaries share reconstruction, and each is certified separately.
"""
from time import perf_counter

import numpy as np
from scipy import sparse

from .composable import ComposableBoundaryDirectory
from .semantic_basis import strength_scale
from .viewrank import _sparse_bytes


def ordered(scores):
    return np.lexsort((np.arange(len(scores)), -scores))


def boundary_margins(scores, errors, order, ks):
    lower = np.minimum.accumulate((scores-errors)[order])
    upper = np.maximum.accumulate((scores+errors)[order][::-1])[::-1]
    return {int(k): float(lower[k-1]-(upper[k] if k < len(order) else -np.inf)) for k in ks}


class PolicyFamily:
    def __init__(self, ranker, seed, sources, directions, masses, keep=128, base_scores=None, directory=None, base_residual=None):
        started = perf_counter()
        self.ranker = ranker
        self.seed = np.asarray(seed, dtype=float)
        self.original_sources = np.asarray(sources, dtype=np.int64)
        self.y = (ranker.base_lu.solve((1-ranker.alpha)*self.seed)
                  if base_scores is None else np.asarray(base_scores, dtype=float))
        self.base_residual = (float(np.linalg.norm((1-ranker.alpha)*self.seed-ranker.A@self.y,1))
                              if base_residual is None else float(base_residual))
        self.directory = (ComposableBoundaryDirectory(ranker, self.original_sources, keep_per_atom=keep)
                          if directory is None else directory)
        triples = sorted(zip(self.original_sources, directions, masses), key=lambda p: p[0])
        triples = [(int(s), sparse.csc_matrix(d), float(m)) for s,d,m in triples if d.nnz]
        self.sources = np.array([s for s,_,_ in triples], dtype=np.int64)
        self.directions = tuple(d for _,d,_ in triples)
        self.masses = np.array([m for _,_,m in triples])
        self.atoms = tuple(self.directory.atomic(s,d) for s,d,_ in triples)
        if directory is None:
            self.boundary_sources=self.directory.B
            self.boundary_matrix=(np.column_stack([atom.boundary for atom in self.atoms])
                                  if self.atoms else np.empty((len(self.boundary_sources),0)))
        else:
            self.boundary_sources,self.boundary_matrix=directory.boundary_block(self.atoms,self.original_sources)
        locations={int(s):j for j,s in enumerate(self.boundary_sources)}
        self.positions=np.array([locations[int(s)] for s in self.sources],dtype=np.int64)
        self.core=self.boundary_matrix[self.positions,:].copy()
        self.prepare_seconds = perf_counter()-started

    @property
    def nbytes(self):
        return self.directory.nbytes+self.core.nbytes+self.masses.nbytes+self.sources.nbytes

    def scales(self, strengths):
        strengths = np.broadcast_to(np.asarray(strengths, dtype=float), self.masses.shape)
        return np.array([strength_scale(l,m) for l,m in zip(strengths,self.masses)])

    def target_delta(self, strengths):
        a = self.scales(strengths)
        return (sparse.hstack([d*v for d,v in zip(self.directions,a)],format='csc')
                if len(a) else sparse.csc_matrix((self.ranker.n,0)))

    def coefficients(self, strengths):
        a = self.scales(strengths)
        K = np.eye(len(a))+self.core*a[None,:]
        c = np.linalg.solve(K,self.y[self.sources]) if len(a) else np.empty(0)
        return a,c,a*c,K

    def solve(self, strengths, ks=(2,5,10), always_fused=False):
        started = perf_counter()
        ranker = self.ranker
        a,c,d,K = self.coefficients(strengths)
        if always_fused:
            rhs = np.zeros(ranker.n)
            for value,atom in zip(d,self.atoms):
                rhs[atom.delta.indices] -= ranker.alpha*value*atom.delta.data
            scores = self.y-ranker.base_lu.solve(rhs) if np.any(d) else self.y.copy()
            if len(c): scores[self.sources]=c
            order=ordered(scores)
            return dict(scores=scores,sets={k:order[:k] for k in ks},
                        fallback_solve_calls=int(np.any(d)),seconds=perf_counter()-started,
                        payload_bytes=self.nbytes)
        scores = self.y.copy()
        contributions = np.array([abs(v)*atom.tail_inf for v,atom in zip(d,self.atoms)])
        errors = np.full(ranker.n,contributions.sum())
        for value,atom,contribution in zip(d,self.atoms,contributions):
            scores[atom.rows] -= value*atom.values
            errors[atom.rows] -= contribution
        if len(d):
            scores[self.boundary_sources] = self.y[self.boundary_sources]-self.boundary_matrix@d
            scores[self.sources] = c
        errors[self.boundary_sources] = 0
        np.maximum(errors,0,out=errors)
        h = self.y[self.sources]-K@c
        h_inf = float(np.max(np.abs(h))) if len(h) else 0.
        residual = self.base_residual+sum(abs(v)*atom.residual_l1 for v,atom in zip(d,self.atoms))
        residual += sum(abs(v*scale)*atom.update_l1 for v,scale,atom in zip(h,a,self.atoms))
        padding = 64*np.finfo(float).eps*(1+np.linalg.norm(self.y,1))/(1-ranker.alpha)
        numerical = residual/(1-ranker.alpha)+h_inf+padding
        errors += numerical
        order = ordered(scores)
        margins = boundary_margins(scores,errors,order,ks)
        initial = {k: margin > 0 for k,margin in margins.items()}
        fallback = bool(len(d) and np.any(d) and (always_fused or not all(initial.values())))
        if fallback:
            rhs = np.zeros(ranker.n)
            for value,atom in zip(d,self.atoms):
                rhs[atom.delta.indices] -= ranker.alpha*value*atom.delta.data
            scores = self.y-ranker.base_lu.solve(rhs)
            scores[self.sources] = c
            residual_vector = (1-ranker.alpha)*self.seed-ranker.A@scores-rhs
            numerical = float(np.linalg.norm(residual_vector,1))/(1-ranker.alpha)+padding
            errors = np.full(ranker.n,numerical)
            order = ordered(scores)
        return dict(scores=scores,sets={k:order[:k] for k in ks},error_bounds=errors,
                    initial_certified=initial,initial_margins=margins,fallback_solve_calls=int(fallback),
                    coefficients=c,scaled_coefficients=d,numerical_allowance=numerical,
                    seconds=perf_counter()-started,payload_bytes=self.nbytes)


class DensePolicyFamily:
    """Same basis and exact core; full responses are the only representation change."""
    def __init__(self, ranker, seed, sources, directions, masses, base_scores=None, directory=None):
        started = perf_counter()
        self.ranker = ranker
        self.seed = np.asarray(seed)
        self.y = (ranker.base_lu.solve((1-ranker.alpha)*self.seed)
                  if base_scores is None else np.asarray(base_scores))
        triples = sorted(zip(sources,directions,masses),key=lambda p:p[0])
        triples = [(int(s),sparse.csc_matrix(d),float(m)) for s,d,m in triples if d.nnz]
        self.sources = np.array([s for s,_,_ in triples],dtype=np.int64)
        self.masses = np.array([m for _,_,m in triples])
        self.D = sparse.hstack([d for _,d,_ in triples],format='csc') if triples else sparse.csc_matrix((ranker.n,0))
        self.directory=directory
        self.responses = ((ranker.base_lu.solve((-ranker.alpha*self.D).toarray()) if directory is None
                           else np.column_stack([directory.atomic(s,d).response for s,d,_ in triples]))
                          if triples else np.empty((ranker.n,0)))
        self.core = self.responses[self.sources,:].copy()
        self.prepare_seconds = perf_counter()-started

    @property
    def nbytes(self):
        if self.directory is not None:return self.directory.nbytes+self.core.nbytes
        return self.responses.nbytes+self.core.nbytes+self.sources.nbytes+self.masses.nbytes+_sparse_bytes(self.D)

    def solve(self,strengths,ks=(2,5,10)):
        started=perf_counter()
        strengths=np.broadcast_to(np.asarray(strengths,dtype=float),self.masses.shape)
        a=np.array([strength_scale(l,m) for l,m in zip(strengths,self.masses)])
        K=np.eye(len(a))+self.core*a[None,:]
        c=np.linalg.solve(K,self.y[self.sources]) if len(a) else np.empty(0)
        scores=self.y-self.responses@(a*c)
        order=ordered(scores)
        return dict(scores=scores,sets={k:order[:k] for k in ks},seconds=perf_counter()-started,
                    payload_bytes=self.nbytes)
