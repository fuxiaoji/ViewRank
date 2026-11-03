"""Dense-family control whose retained basis, including live families, is capped.

This is independent of the frozen cache-only control. A family pins references
to cached response vectors; it never owns an n-by-r response copy. Budget is
unique atom arrays plus live exact cores/source/mass arrays, not process RSS.
Caller graph/input/seed/base-y/output and native solver workspace are separate.
"""
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy import sparse

from src.policy_family_fast import top_indices
from src.semantic_basis import strength_scale
from src.viewrank import _sparse_bytes


@dataclass
class _Atom:
    delta: sparse.csc_matrix
    response: np.ndarray

    @property
    def nbytes(self):
        return _sparse_bytes(self.delta)+self.response.nbytes+32


def _canonical(source,delta):
    delta=sparse.csc_matrix(delta).copy()
    delta.sum_duplicates();delta.eliminate_zeros();delta.sort_indices()
    h=sha256(np.asarray([int(source)],dtype=np.int64).tobytes())
    for values in (delta.indptr,delta.indices,delta.data):h.update(values.tobytes())
    return h.hexdigest(),delta


class BasisResidentDenseDirectory:
    """LRU full-response cache with explicit live-family pins.

    Use `with directory.prepare(..., base_scores=y) as family:` and call
    family.solve(lambda, ks) for every requested target before leaving it.
    Preparing a later family cannot evict a response pinned by an earlier one.
    Oversized families use native direct for their whole request family,
    without preparing any response atoms or dropping any source/target.
    """
    def __init__(self,ranker,budget_bytes,point_graph=None):
        self.ranker=ranker;self.budget_bytes=int(budget_bytes)
        self.point_graph=point_graph
        self.cache=OrderedDict();self._pins={};self._payload_bytes=0
        self._live_core_bytes=0
        self.hits=self.misses=self.evictions=self.atomic_solve_calls=0
        self.atomic_prepare_seconds=0.
        self.fallback_families=self.native_direct_calls=self.native_offset_calls=0
        self.peak_bytes=self.peak_live_response_bytes=0

    @property
    def nbytes(self):return self._payload_bytes+self._live_core_bytes

    @property
    def response_bytes(self):return sum(a.response.nbytes for a in self.cache.values())

    @property
    def memory(self):
        live=[self.cache[key] for key,count in self._pins.items() if count]
        return dict(cache_atom_basis_bytes=self._payload_bytes,
            cache_response_bytes=self.response_bytes,
            live_atom_basis_bytes=sum(a.nbytes for a in live),
            live_response_bytes=sum(a.response.nbytes for a in live),
            live_core_metadata_bytes=self._live_core_bytes,
            total_private_retained_basis_bytes=self.nbytes,
            peak_private_retained_basis_bytes=self.peak_bytes,
            peak_live_response_bytes=self.peak_live_response_bytes)

    def _track(self):
        self.peak_bytes=max(self.peak_bytes,self.nbytes)
        self.peak_live_response_bytes=max(self.peak_live_response_bytes,
            sum(self.cache[key].response.nbytes for key in self._pins))

    def prepare(self,seed,sources,directions,masses,base_scores=None):
        started=perf_counter()
        sources=np.asarray(sources,dtype=np.int64)
        masses=np.asarray(masses,dtype=float)
        directions=tuple(directions)
        specs=[]
        for index in np.argsort(sources,kind='stable'):
            key,delta=_canonical(sources[index],directions[index])
            if delta.nnz:
                specs.append((int(index),key,delta,
                              _sparse_bytes(delta)+self.ranker.n*8+32))
        rank=len(specs)
        metadata_bytes=8*(rank*rank+3*rank)
        required_response_bytes=rank*self.ranker.n*8
        required_basis_bytes=sum(spec[3] for spec in specs)+metadata_bytes
        # The lower bound includes other live families, counting shared atoms
        # once. No atom preparation occurs before this routing decision.
        pinned_bytes=sum(self.cache[key].nbytes for key in self._pins)
        extra_bytes=sum(size for _,key,_,size in specs if key not in self._pins)
        minimum_retained=pinned_bytes+extra_bytes+self._live_core_bytes+metadata_bytes
        if minimum_retained>self.budget_bytes:
            self.fallback_families+=1
            reason=('family_basis_exceeds_budget' if required_basis_bytes>self.budget_bytes
                    else 'other_live_family_pins')
            family=DenseBudgetFamily(self,seed,sources,directions,masses,base_scores,
                route='native',reason=reason)
        else:
            # Touch and pin every cached atom needed by this family BEFORE
            # inserting misses or evicting any older item.
            for _,key,_,_ in specs:
                if key in self.cache:
                    self.cache.move_to_end(key)
                    self._pins[key]=self._pins.get(key,0)+1
            missing_bytes=sum(size for _,key,_,size in specs if key not in self.cache)
            while self.nbytes+missing_bytes+metadata_bytes>self.budget_bytes:
                key=next(key for key in self.cache if key not in self._pins)
                old=self.cache.pop(key);self._payload_bytes-=old.nbytes;self.evictions+=1
            atoms=[]
            for _,key,delta,_ in specs:
                if key in self.cache:
                    self.hits+=1;atom=self.cache[key]
                else:
                    self.misses+=1;t=perf_counter()
                    response=self.ranker.base_lu.solve((-self.ranker.alpha*delta).toarray().ravel())
                    atom=_Atom(delta,response)
                    self.cache[key]=atom;self._payload_bytes+=atom.nbytes
                    self._pins[key]=1
                    self.atomic_solve_calls+=1;self.atomic_prepare_seconds+=perf_counter()-t
                atoms.append(atom);self._track()
            active=np.array([index for index,_,_,_ in specs],dtype=np.int64)
            family=DenseBudgetFamily(self,seed,sources,directions,masses,base_scores,
                route='dense',atoms=tuple(atoms),keys=tuple(key for _,key,_,_ in specs),
                active=active,metadata_bytes=metadata_bytes)
            self._live_core_bytes+=metadata_bytes;self._track()
        family.required_response_bytes=required_response_bytes
        family.required_family_basis_bytes=required_basis_bytes
        family.prepare_seconds=perf_counter()-started
        return family


class DenseBudgetFamily:
    """One complete source/policy family, with either pinned R or native route."""
    def __init__(self,directory,seed,sources,directions,masses,base_scores,
                 route,reason=None,atoms=(),keys=(),active=None,metadata_bytes=0):
        self.directory=directory;self.ranker=directory.ranker
        self.seed=np.asarray(seed,dtype=float)
        self.original_sources=sources;self.original_directions=directions;self.original_masses=masses
        self.y=(self.ranker.base_lu.solve((1-self.ranker.alpha)*self.seed)
                if base_scores is None else np.asarray(base_scores,dtype=float))
        self.route=route;self.route_reason=reason;self.atoms=atoms;self.keys=keys
        self._metadata_bytes=metadata_bytes;self.closed=False
        self._native_offset=None
        if route=='dense':
            self.sources=sources[active].copy();self.masses=masses[active].copy();self.active=active
            # Only r-by-r coordinates are copied. Full response vectors remain
            # the exact cache-owned arrays referenced by self.atoms.
            self.core=np.empty((len(atoms),len(atoms)))
            for j,atom in enumerate(atoms):self.core[:,j]=atom.response[self.sources]

    @property
    def nbytes(self):return self.directory.nbytes

    @property
    def memory(self):return self.directory.memory

    def solve(self,strengths,ks=(2,5,10)):
        if self.closed:raise RuntimeError('family pins have been released')
        started=perf_counter();r=self.ranker
        strengths=np.broadcast_to(np.asarray(strengths,dtype=float),self.original_masses.shape)
        all_scales=np.array([strength_scale(l,m) for l,m in zip(strengths,self.original_masses)])
        identity=not any(scale and direction.nnz for scale,direction in zip(all_scales,self.original_directions))
        native_calls=0;native_stats=None
        if identity:
            scores=self.y.copy()
        elif self.route=='native':
            # The strongest native completion is shared by Dense and Stencil
            # bypasses. No full-response atom is prepared for this route.
            from src.baselines.native_offset_family import NativeOffsetFamily
            base_check=0.
            if self._native_offset is None:
                self._native_offset=NativeOffsetFamily(r,self.seed,self.original_sources,
                    self.original_directions,self.original_masses,base_scores=self.y,
                    point_graph=self.directory.point_graph)
                base_check=self._native_offset.base_residual_seconds
            result=self._native_offset.solve(strengths,ks=ks)
            stages=result['stage_seconds'].copy();stages['residual_check']+=base_check
            elapsed=perf_counter()-started;verification=stages['residual_check']
            self.directory.native_offset_calls=getattr(self.directory,'native_offset_calls',0)+result['native_solve_calls']
            result.update(seconds=elapsed-verification,operator_seconds=elapsed-verification,
                seconds_with_verification=elapsed,stage_seconds=stages,
                payload_bytes=self.nbytes,route=self.route,route_reason=self.route_reason,
                native_direct_calls=0,native_offset_calls=result['native_solve_calls'],basis_memory=self.memory)
            return result
        else:
            a=all_scales[self.active]
            K=np.eye(len(a))+self.core*a[None,:]
            c=np.linalg.solve(K,self.y[self.sources]) if len(a) else np.empty(0)
            scores=self.y.copy()
            for value,atom in zip(a*c,self.atoms):
                if value:scores-=value*atom.response
        order=top_indices(scores,max(ks))
        return dict(scores=scores,sets={int(k):order[:k] for k in ks},
            selected_scores={int(k):scores[order[:k]].copy() for k in ks},
            seconds=perf_counter()-started,payload_bytes=self.nbytes,
            route=self.route,route_reason=self.route_reason,identity_reused=identity,
            native_direct_calls=native_calls,native_stats=native_stats,
            basis_memory=self.memory)

    def close(self):
        if self.closed:return
        for key in self.keys:
            self.directory._pins[key]-=1
            if not self.directory._pins[key]:del self.directory._pins[key]
        self.directory._live_core_bytes-=self._metadata_bytes
        self.atoms=();self.keys=()
        self._native_offset=None
        if self.route=='dense':
            self.core=None;self.sources=None;self.masses=None;self.active=None
        self.closed=True

    def __enter__(self):return self

    def __exit__(self,*args):self.close()
