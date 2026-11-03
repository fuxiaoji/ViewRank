"""Stencil execution with an enforced retained-basis and family-index budget.

Preparation may use one transient dense solve vector. The retained budget
includes cache arrays, the interaction plan and live query-specific indexes;
graph/backend, seed/base score, output and numerical-solver scratch are separate.
"""
import numpy as np
from .budgeted_family import BudgetedStencilDirectory
from .policy_family_fast import FastPolicyFamily
from .full_scan_policy_family import FullScanPolicyFamily
from .baselines.dense_family_budget import DenseBudgetFamily,_canonical
from .viewrank import _sparse_bytes


class RetainedStencilDirectory(BudgetedStencilDirectory):
    def __init__(self,ranker,stencil,keep,budget_bytes):
        super().__init__(ranker,stencil,keep,budget_bytes)
        self.full_budget=int(budget_bytes);self._live_core_bytes=0;self.peak_private_bytes=self.nbytes
        self.native_direct_calls=self.fallback_families=0;self.point_graph=None

    @property
    def memory(self):
        return dict(cache_atom_basis_bytes=self.nbytes,live_core_metadata_bytes=self._live_core_bytes,
                    total_private_retained_basis_bytes=self.nbytes+self._live_core_bytes,
                    peak_private_retained_basis_bytes=self.peak_private_bytes)

    def prepare(self,seed,sources,directions,masses,base_scores,base_residual,ks=(2,5,10),always_fused=False,full_scan=False):
        active=[(int(s),d) for s,d in zip(sources,directions) if d.nnz];r=len(active)
        retained_upper=sum(min(self.ranker.n,self.keep+len(self.stencil[s])) for s,_ in active)
        # The boundary block has |original S| rows, even when many original
        # sources have zero directions and only r response columns remain.
        metadata_upper=8*(r*r+len(sources)*r+3*r+len(sources))
        if not (always_fused or full_scan):
            candidate_upper=min(self.ranker.n,retained_upper+len(sources)+max(ks))
            metadata_upper+=8*(candidate_upper+len(sources)+r+retained_upper)
        atom_upper=sum(_sparse_bytes(d)+16*min(self.ranker.n,self.keep+len(self.stencil[s]))+32 for s,d in active)
        if self.plan_bytes+atom_upper+metadata_upper>self.full_budget:
            self.fallback_families+=1
            return DenseBudgetFamily(self,seed,np.asarray(sources),tuple(directions),np.asarray(masses),base_scores,
                                     route='native',reason='stencil_family_exceeds_retained_budget')
        self.budget_bytes=self.full_budget-metadata_upper
        for source,direction in active:
            key,_=_canonical(source,direction)
            if key in self.cache:self.cache.move_to_end(key)
        while self.nbytes>self.budget_bytes and self.cache:
            _,old=self.cache.popitem(last=False);self._payload_bytes-=old.nbytes;self.evictions+=1
        family_class=RetainedFullScanFamily if full_scan else RetainedFastFamily
        options={} if full_scan else dict(candidate_outputs=not always_fused)
        family=family_class(self.ranker,seed,sources,directions,masses,directory=self,
                            base_scores=base_scores,base_residual=base_residual,ks=ks,**options)
        arrays=[family.original_sources,family.sources,family.masses,family.positions,family.core,family.boundary_matrix]
        if not (always_fused or full_scan):
            arrays += [family.rows,family.boundary_positions,family.source_positions,*family.atom_positions]
        seen=set();metadata=0
        for array in arrays:
            if id(array) not in seen:metadata+=array.nbytes;seen.add(id(array))
        cached={id(a) for a in self.cache.values()}
        metadata+=sum(a.nbytes for a in family.atoms if id(a) not in cached)
        self._live_core_bytes=metadata
        self.peak_private_bytes=max(self.peak_private_bytes,self.nbytes+metadata)
        assert self.nbytes+metadata<=self.full_budget
        family.route='stencil';family.route_reason=None
        return family


class RetainedFastFamily(FastPolicyFamily):
    def __enter__(self):return self

    def __exit__(self,*args):self.close()

    def close(self):
        self.directory._live_core_bytes=0
        self.atoms=();self.core=None;self.boundary_matrix=None
        self.sources=None;self.masses=None;self.positions=None
        if hasattr(self,'rows'):
            self.rows=None;self.atom_positions=();self.boundary_positions=None;self.source_positions=None


class RetainedFullScanFamily(FullScanPolicyFamily):
    close=RetainedFastFamily.close

    def __enter__(self):return self

    def __exit__(self,*args):self.close()
