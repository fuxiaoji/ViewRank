"""Output selection for sparse family responses, with separately checked k cuts.

Unmodified approximate coordinates retain their base score and share a common
tail envelope. Only retained coordinates and the best unaffected base nodes
can enter an approximate top-k. Dense controls use the same linear-time top-k
selection primitive rather than a full sort.
"""
from time import perf_counter
import numpy as np
from .policy_family import PolicyFamily


def top_indices(scores,k):
    n=len(scores);k=min(k,n)
    if k==n:return np.lexsort((np.arange(n),-scores))
    chosen=np.argpartition(-scores,k-1)[:k]
    cutoff=float(np.min(scores[chosen]))
    better=np.flatnonzero(scores>cutoff)
    tied=np.flatnonzero(scores==cutoff)[:k-len(better)]
    ids=np.concatenate([better,tied])
    return ids[np.lexsort((ids,-scores[ids]))]


class FastPolicyFamily(PolicyFamily):
    def __init__(self,*args,ks=(2,5,10),candidate_outputs=True,**kwargs):
        started=perf_counter();super().__init__(*args,**kwargs)
        self.ks=tuple(ks);self.kmax=max(ks)
        if candidate_outputs:
            retained=np.unique(np.concatenate([atom.rows for atom in self.atoms])) if self.atoms else np.empty(0,dtype=np.int64)
            retained=np.union1d(retained,self.boundary_sources)
            mask=np.ones(self.ranker.n,dtype=bool);mask[retained]=False
            unaffected=np.flatnonzero(mask)
            best=unaffected[top_indices(self.y[unaffected],min(self.kmax+1,len(unaffected)))] if len(unaffected) else unaffected
            self.rows=np.union1d(retained,best[:self.kmax])
            self.outside_base=float(self.y[best[self.kmax]]) if len(best)>self.kmax else -np.inf
            self.atom_positions=tuple(np.searchsorted(self.rows,a.rows) for a in self.atoms)
            self.boundary_positions=np.searchsorted(self.rows,self.boundary_sources)
            self.source_positions=np.searchsorted(self.rows,self.sources)
        self.prepare_seconds=perf_counter()-started

    def solve(self,strengths,ks=None,always_fused=False,materialize=False):
        started=perf_counter();ks=self.ks if ks is None else tuple(ks);r=self.ranker
        a,c,d,K=self.coefficients(strengths)
        if always_fused:
            return self._complete(d,c,ks,started,{},materialize)
        contributions=np.array([abs(v)*atom.tail_inf for v,atom in zip(d,self.atoms)])
        tail=float(contributions.sum())
        scores=self.y[self.rows].copy();errors=np.full(len(self.rows),tail)
        for value,atom,pos,contribution in zip(d,self.atoms,self.atom_positions,contributions):
            scores[pos]-=value*atom.values;errors[pos]-=contribution
        if len(d):
            scores[self.boundary_positions]=self.y[self.boundary_sources]-self.boundary_matrix@d
            scores[self.source_positions]=c
        errors[self.boundary_positions]=0;np.maximum(errors,0,out=errors)
        h=self.y[self.sources]-K@c
        h_inf=float(np.max(np.abs(h))) if len(h) else 0.
        residual=self.base_residual+sum(abs(v)*atom.residual_l1 for v,atom in zip(d,self.atoms))
        residual+=sum(abs(v*scale)*atom.update_l1 for v,scale,atom in zip(h,a,self.atoms))
        pad=64*np.finfo(float).eps*(1+np.linalg.norm(self.y,1))/(1-r.alpha)
        numerical=residual/(1-r.alpha)+h_inf+pad;errors+=numerical
        order=top_indices(scores,max(ks));margins={}
        for k in ks:
            upper=scores+errors;upper=upper.copy();upper[order[:k]]=-np.inf
            outside=max(float(np.max(upper)),self.outside_base+tail+numerical)
            margins[k]=float(np.min(scores[order[:k]]-errors[order[:k]])-outside)
        initial={k:margin>0 for k,margin in margins.items()}
        if not all(initial.values()) and np.any(d):
            return self._complete(d,c,ks,started,initial,materialize)
        result=dict(sets={k:self.rows[order[:k]] for k in ks},
                    selected_scores={k:scores[order[:k]] for k in ks},initial_certified=initial,
                    initial_margins=margins,fallback_solve_calls=0,seconds=perf_counter()-started,
                    payload_bytes=self.nbytes,candidate_nodes=len(self.rows))
        if materialize:
            full=self.y.copy();full[self.rows]=scores
            bounds=np.full(r.n,tail+numerical);bounds[self.rows]=errors
            result.update(scores=full,error_bounds=bounds)
        return result

    def _complete(self,d,c,ks,started,initial,materialize):
        r=self.ranker;rhs=np.zeros(r.n)
        for value,atom in zip(d,self.atoms):rhs[atom.delta.indices]-=r.alpha*value*atom.delta.data
        if np.any(d):
            correction=r.base_lu.solve(rhs)
            measured=getattr(r.base_lu,'last_residuals',None)
            correction_residual=(float(measured[0]) if measured is not None else
                                 float(np.linalg.norm(rhs-r.A@correction,1)))
            scores=self.y-correction
        else:
            scores=self.y.copy();correction_residual=0.
        discrepancy=scores[self.sources]-c
        if len(c):scores[self.sources]=c
        pad=64*np.finfo(float).eps*(1+np.linalg.norm(self.y,1))/(1-r.alpha)
        numerical=(self.base_residual+correction_residual+(1+r.alpha)*np.linalg.norm(discrepancy,1))/(1-r.alpha)+pad
        order=top_indices(scores,min(max(ks)+1,r.n))
        result=dict(sets={k:order[:k] for k in ks},selected_scores={k:scores[order[:k]] for k in ks},
                    initial_certified=initial,fallback_solve_calls=int(np.any(d)),seconds=perf_counter()-started,
                    payload_bytes=self.nbytes,candidate_nodes=r.n,numerical_allowance=numerical,
                    completion_certified={k:(k==r.n or scores[order[k-1]]-scores[order[k]]>2*numerical) for k in ks})
        if materialize:
            result.update(scores=scores,error_bounds=np.full(r.n,numerical))
        return result
