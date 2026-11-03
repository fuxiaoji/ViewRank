"""Array-budgeted LRU controls for fixed-group policy-response atoms.

The caller plans B from the complete public input batch, without reading
evaluation annotations. Budget applies to resident atom arrays; numerical
backend, query output and preparation workspace are reported separately.
"""
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy import sparse

from .composable import ComposableBoundaryDirectory
from .viewrank import _sparse_bytes
from .policy_family import ordered,boundary_margins
from .semantic_basis import strength_scale


class BudgetedBoundaryDirectory(ComposableBoundaryDirectory):
    def __init__(self,ranker,B,keep,budget_bytes):
        super().__init__(ranker,B,keep)
        self.cache=OrderedDict();self.budget_bytes=int(budget_bytes)
        self.evictions=0;self.peak_bytes=self.nbytes

    def atomic(self,source,delta):
        atom=super().atomic(source,delta)
        self.cache.move_to_end(atom.fingerprint)
        while self.nbytes>self.budget_bytes and self.cache:
            _,old=self.cache.popitem(last=False)
            self._payload_bytes-=old.nbytes;self._response_bytes-=old.response_bytes
            self.evictions+=1
        self.peak_bytes=max(self.peak_bytes,self.nbytes)
        return atom

    def boundary_block(self,atoms,sources):
        return self.B,(np.column_stack([atom.boundary for atom in atoms]) if atoms else np.empty((len(self.B),0)))


def interaction_stencil(source_sets):
    neighbors={}
    for sources in source_sets:
        for source in sources:neighbors.setdefault(int(source),set()).update(map(int,sources))
    return {s:np.array(sorted(v),dtype=np.int64) for s,v in neighbors.items()}


@dataclass
class StencilAtom:
    delta: sparse.csc_matrix
    rows: np.ndarray
    values: np.ndarray
    tail_inf: float
    residual_l1: float
    update_l1: float

    @property
    def nbytes(self):return _sparse_bytes(self.delta)+self.rows.nbytes+self.values.nbytes+32


class BudgetedStencilDirectory:
    """Exact coupling only on pairs required by the input interaction graph.

    Every requested source pair is registered by the batch plan. Top-L and
    stencil coordinates are merged, so their values are never stored twice.
    A new combination is supported whenever all its pair interactions occur
    in this graph, even if that whole source set has not occurred before.
    """
    def __init__(self,ranker,stencil,keep,budget_bytes):
        self.ranker=ranker;self.stencil=stencil;self.keep=keep;self.budget_bytes=int(budget_bytes)
        self.cache=OrderedDict();self.hits=self.misses=self.evictions=self.atomic_solve_calls=0
        self.atomic_prepare_seconds=0.
        self.plan_bytes=sum(v.nbytes for v in stencil.values())+len(stencil)*8
        self._payload_bytes=self.plan_bytes;self.peak_bytes=self.nbytes

    @property
    def nbytes(self):return self._payload_bytes

    def atomic(self,source,delta):
        delta=sparse.csc_matrix(delta).copy();delta.sum_duplicates();delta.eliminate_zeros();delta.sort_indices()
        h=sha256(np.asarray([int(source)],dtype=np.int64).tobytes())
        for values in [delta.indptr,delta.indices,delta.data]:h.update(values.tobytes())
        key=h.hexdigest()
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);return self.cache[key]
        self.misses+=1;t=perf_counter();rhs=(-self.ranker.alpha*delta).toarray().ravel()
        response=self.ranker.base_lu.solve(rhs)
        measured_residuals=getattr(self.ranker.base_lu,'last_residuals',None)
        response_residual=(float(measured_residuals[0]) if measured_residuals is not None else
                           float(np.linalg.norm(rhs-self.ranker.A@response,1)))
        count=min(self.keep,len(response))
        top=np.argpartition(np.abs(response),-count)[-count:] if count else np.empty(0,dtype=np.int64)
        rows=np.union1d(top,self.stencil[int(source)])
        values=response[rows].copy();omitted=np.ones(self.ranker.n,dtype=bool);omitted[rows]=False
        atom=StencilAtom(delta,rows,values,float(np.max(np.abs(response[omitted]))) if omitted.any() else 0.,
                         response_residual,float(np.linalg.norm(rhs,1)))
        self.atomic_solve_calls+=1;self.atomic_prepare_seconds+=perf_counter()-t
        self.cache[key]=atom;self._payload_bytes+=atom.nbytes
        while self.nbytes>self.budget_bytes and self.cache:
            _,old=self.cache.popitem(last=False);self._payload_bytes-=old.nbytes;self.evictions+=1
        self.peak_bytes=max(self.peak_bytes,self.nbytes)
        return atom

    def boundary_block(self,atoms,sources):
        sources=np.asarray(sources,dtype=np.int64)
        columns=[]
        for atom in atoms:
            positions=np.searchsorted(atom.rows,sources)
            if np.any(positions>=len(atom.rows)) or np.any(atom.rows[positions]!=sources):
                raise ValueError('This source combination needs an unregistered interaction; execute or prepare the complete view.')
            columns.append(atom.values[positions])
        block=np.column_stack(columns) if columns else np.empty((len(sources),0))
        return sources,block


@dataclass
class DenseAtom:
    delta: sparse.csc_matrix
    response: np.ndarray

    @property
    def nbytes(self):return _sparse_bytes(self.delta)+self.response.nbytes+32


class BudgetedDenseDirectory:
    def __init__(self,ranker,budget_bytes):
        self.ranker=ranker;self.budget_bytes=int(budget_bytes);self.cache=OrderedDict()
        self.hits=self.misses=self.evictions=self.peak_bytes=0
        self._payload_bytes=0;self.atomic_prepare_seconds=0.;self.atomic_solve_calls=0

    @property
    def nbytes(self):return self._payload_bytes

    def atomic(self,source,delta):
        delta=sparse.csc_matrix(delta).copy();delta.sum_duplicates();delta.eliminate_zeros();delta.sort_indices()
        h=sha256(np.asarray([int(source)],dtype=np.int64).tobytes())
        for values in [delta.indptr,delta.indices,delta.data]:h.update(values.tobytes())
        key=h.hexdigest()
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);return self.cache[key]
        self.misses+=1;t=perf_counter()
        atom=DenseAtom(delta,self.ranker.base_lu.solve((-self.ranker.alpha*delta).toarray().ravel()))
        self.atomic_solve_calls+=1;self.atomic_prepare_seconds+=perf_counter()-t
        self.cache[key]=atom;self._payload_bytes+=atom.nbytes
        while self.nbytes>self.budget_bytes and self.cache:
            _,old=self.cache.popitem(last=False);self._payload_bytes-=old.nbytes;self.evictions+=1
        self.peak_bytes=max(self.peak_bytes,self.nbytes)
        return atom


@dataclass
class ResidualAtom:
    delta: sparse.csc_matrix
    rows: np.ndarray
    values: np.ndarray
    residual_l1: float
    update_l1: float

    @property
    def nbytes(self):return _sparse_bytes(self.delta)+self.rows.nbytes+self.values.nbytes+32


class BudgetedResidualDirectory(BudgetedDenseDirectory):
    """Same atomic deduplication, but no retained exact interaction core."""
    def __init__(self,ranker,keep,budget_bytes):
        super().__init__(ranker,budget_bytes);self.keep=keep

    def atomic(self,source,delta):
        delta=sparse.csc_matrix(delta).copy();delta.sum_duplicates();delta.eliminate_zeros();delta.sort_indices()
        h=sha256(np.asarray([int(source)],dtype=np.int64).tobytes())
        for values in [delta.indptr,delta.indices,delta.data]:h.update(values.tobytes())
        key=h.hexdigest()
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);return self.cache[key]
        self.misses+=1;t=perf_counter();rhs=(-self.ranker.alpha*delta).toarray().ravel()
        response=self.ranker.base_lu.solve(rhs)
        count=min(self.keep,len(response))
        rows=np.sort(np.argpartition(np.abs(response),-count)[-count:]) if count else np.empty(0,dtype=int)
        values=response[rows].copy()
        approximation=sparse.csc_matrix((values,(rows,np.zeros(len(rows),dtype=int))),shape=delta.shape)
        residual=(-self.ranker.alpha*delta-self.ranker.A@approximation)
        atom=ResidualAtom(delta,rows,values,float(abs(residual).sum()),float(np.linalg.norm(rhs,1)))
        self.atomic_solve_calls+=1;self.atomic_prepare_seconds+=perf_counter()-t
        self.cache[key]=atom;self._payload_bytes+=atom.nbytes
        while self.nbytes>self.budget_bytes and self.cache:
            _,old=self.cache.popitem(last=False);self._payload_bytes-=old.nbytes;self.evictions+=1
        self.peak_bytes=max(self.peak_bytes,self.nbytes)
        return atom


class ResidualPolicyFamily:
    def __init__(self,ranker,seed,sources,directions,masses,directory,base_scores,base_residual=None):
        t=perf_counter();self.ranker=ranker;self.seed=seed;self.y=base_scores;self.directory=directory
        triples=sorted([(int(s),d,float(m)) for s,d,m in zip(sources,directions,masses) if d.nnz],key=lambda p:p[0])
        self.sources=np.array([s for s,_,_ in triples],dtype=np.int64)
        self.masses=np.array([m for _,_,m in triples]);self.atoms=[directory.atomic(s,d) for s,d,_ in triples]
        self.core=np.zeros((len(triples),len(triples)))
        for j,atom in enumerate(self.atoms):
            for i,source in enumerate(self.sources):
                pos=np.searchsorted(atom.rows,source)
                if pos<len(atom.rows) and atom.rows[pos]==source:self.core[i,j]=atom.values[pos]
        self.ry=(float(np.linalg.norm((1-ranker.alpha)*seed-ranker.A@self.y,1))
                 if base_residual is None else float(base_residual))
        self.prepare_seconds=perf_counter()-t

    def solve(self,strength,ks=(2,5,10)):
        t=perf_counter();r=self.ranker
        a=np.array([strength_scale(strength,m) for m in self.masses]);K=np.eye(len(a))+self.core*a[None,:]
        singular=False
        try:c=np.linalg.solve(K,self.y[self.sources]) if len(a) else np.empty(0)
        except np.linalg.LinAlgError:singular=True;c=np.zeros(len(a))
        d=a*c;scores=self.y.copy()
        for value,atom in zip(d,self.atoms):scores[atom.rows]-=value*atom.values
        h=self.y[self.sources]-K@c
        residual=self.ry+sum(abs(v)*atom.residual_l1 for v,atom in zip(d,self.atoms))
        residual+=sum(abs(v*s)*atom.update_l1 for v,s,atom in zip(h,a,self.atoms))
        error=residual/(1-r.alpha)+64*np.finfo(float).eps*(1+np.linalg.norm(self.y,1))/(1-r.alpha)
        order=ordered(scores);margins=boundary_margins(scores,np.full(r.n,error),order,ks)
        initial={k:(margin>0 and not singular) for k,margin in margins.items()}
        fallback=not all(initial.values())
        if fallback:
            D=(sparse.hstack([atom.delta*scale for atom,scale in zip(self.atoms,a)],format='csc')
               if len(a) else sparse.csc_matrix((r.n,0)))
            graph=r.base_lu.graph
            graph.update_columns(self.sources,D)
            scores,_=graph.solve(self.seed,r.alpha,1e-12)
            graph.update_columns([],sparse.csc_matrix((r.n,0)))
            order=ordered(scores)
        return dict(scores=scores,sets={k:order[:k] for k in ks},initial_certified=initial,
                    fallback_solve_calls=int(fallback),seconds=perf_counter()-t,payload_bytes=self.directory.nbytes)
