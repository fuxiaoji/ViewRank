"""Full-node output/certificate scan, sharing the Fast family's exact core.

The only intended ablation is output working representation: every target
allocates full score/envelope vectors. It uses the same partial top-k primitive
and the exact same one-fused-solve completion method as FastPolicyFamily.
"""
from time import perf_counter
import numpy as np

from .policy_family import PolicyFamily
from .policy_family_fast import FastPolicyFamily,top_indices


class FullScanPolicyFamily(PolicyFamily):
    _complete=FastPolicyFamily._complete

    def __init__(self,*args,ks=(2,5,10),**kwargs):
        started=perf_counter();super().__init__(*args,**kwargs)
        self.ks=tuple(ks)
        self.prepare_seconds=perf_counter()-started

    def solve(self,strengths,ks=None,always_fused=False,materialize=False):
        started=perf_counter();ks=self.ks if ks is None else tuple(ks);r=self.ranker
        a,c,d,K=self.coefficients(strengths)
        if always_fused:return self._complete(d,c,ks,started,{},materialize)
        contributions=np.array([abs(v)*atom.tail_inf for v,atom in zip(d,self.atoms)])
        scores=self.y.copy();errors=np.full(r.n,float(contributions.sum()))
        for value,atom,contribution in zip(d,self.atoms,contributions):
            scores[atom.rows]-=value*atom.values
            errors[atom.rows]-=contribution
        if len(d):
            scores[self.boundary_sources]=self.y[self.boundary_sources]-self.boundary_matrix@d
            scores[self.sources]=c
        errors[self.boundary_sources]=0;np.maximum(errors,0,out=errors)
        h=self.y[self.sources]-K@c
        h_inf=float(np.max(np.abs(h))) if len(h) else 0.
        residual=self.base_residual+sum(abs(v)*atom.residual_l1 for v,atom in zip(d,self.atoms))
        residual+=sum(abs(v*scale)*atom.update_l1 for v,scale,atom in zip(h,a,self.atoms))
        pad=64*np.finfo(float).eps*(1+np.linalg.norm(self.y,1))/(1-r.alpha)
        numerical=residual/(1-r.alpha)+h_inf+pad;errors+=numerical
        order=top_indices(scores,max(ks));margins={}
        for k in ks:
            upper=scores+errors;upper[order[:k]]=-np.inf
            margins[k]=float(np.min(scores[order[:k]]-errors[order[:k]])-np.max(upper))
        initial={k:margin>0 for k,margin in margins.items()}
        if not all(initial.values()) and np.any(d):
            return self._complete(d,c,ks,started,initial,materialize)
        result=dict(sets={k:order[:k] for k in ks},selected_scores={k:scores[order[:k]] for k in ks},
            initial_certified=initial,initial_margins=margins,fallback_solve_calls=0,
            seconds=perf_counter()-started,payload_bytes=self.nbytes,candidate_nodes=r.n)
        if materialize:result.update(scores=scores,error_bounds=errors)
        return result
