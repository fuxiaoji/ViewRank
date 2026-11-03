"""Additional top-k early-stop controls; independent of the frozen V5 matrix.

NativeGap and OffsetGap use the original PRPACK kernels, coarse to fine. Every
actual temporary-view residual used for certification is charged algorithm work.
OSPGap shares one unpruned offset recurrence across all requested cutoffs.
"""
from time import perf_counter

import numpy as np
from scipy import sparse

from src.osp import offset_seed
from src.policy_family_fast import top_indices
from src.semantic_basis import strength_scale


TOLERANCES=(1e-4,1e-6,1e-8,1e-10,1e-12)


class CertifiedNativeFamily:
    """Complete-source top-k family using native_gap, offset_gap or osp_gap.

    All actual residuals, gap checks, repeated native calls, weights and restore
    are inside seconds/operator_seconds. Base construction is once per family
    (or supplied by the caller) and is reported by prepare_seconds/base_seconds.
    A final unseparated cutoff is flagged as numerical completion, not certified.
    """
    def __init__(self,ranker,seed,sources,directions,masses,base_scores=None,
                 base_residual=None,mode='native_gap',point_graph=None):
        started=perf_counter()
        if mode not in ('native_gap','offset_gap','osp_gap'):raise ValueError('unknown certificate mode')
        self.ranker=ranker;self.mode=mode;self.seed=np.asarray(seed,dtype=float)
        self.sources=np.asarray(sources,dtype=np.int64)
        self.directions=tuple(directions);self.masses=np.asarray(masses,dtype=float)
        self.graph=point_graph if point_graph is not None else getattr(ranker.base_lu,'graph',None)
        self.base_seconds=0.;self.base_residual_seconds=0.
        if base_scores is None:
            t=perf_counter();self.y=ranker.base_lu.solve((1-ranker.alpha)*self.seed)
            self.base_seconds=perf_counter()-t
            measured=getattr(ranker.base_lu,'last_residuals',None)
            if base_residual is None and measured is not None:base_residual=float(measured[0])
        else:self.y=np.asarray(base_scores,dtype=float)
        if base_residual is None:
            t=perf_counter()
            base_residual=float(np.linalg.norm((1-ranker.alpha)*self.seed-ranker.A@self.y,1))
            self.base_residual_seconds=perf_counter()-t
        self.base_residual=float(base_residual)
        self.base_pad=64*np.finfo(float).eps*(1+np.linalg.norm(self.y,1))/(1-ranker.alpha)
        self.prepare_seconds=perf_counter()-started
        self.native_solve_calls=0;self._base_outputs={}

    @property
    def nbytes(self):return 0

    def _residual(self,scores,D):
        r=self.ranker
        return float(np.linalg.norm((1-r.alpha)*self.seed-scores+
            r.alpha*(r.P@scores+D@scores[self.sources]),1))

    def solve(self,strengths,ks=(2,5,10)):
        started=perf_counter();r=self.ranker;alpha=r.alpha;ks=tuple(map(int,ks))
        stages=dict(assembly=0.,offset_rhs=0.,update=0.,native_solve=0.,restore=0.,
                    certificate_residual=0.,certificate_gap=0.,propagation=0.,output=0.)
        t=perf_counter()
        strengths=np.broadcast_to(np.asarray(strengths,dtype=float),self.masses.shape)
        scales=np.array([strength_scale(l,m) for l,m in zip(strengths,self.masses)])
        D=(sparse.hstack([d*a for d,a in zip(self.directions,scales)],format='csc')
           if len(scales) else sparse.csc_matrix((r.n,0)))
        D.eliminate_zeros();identity=not D.nnz
        stages['assembly']=perf_counter()-t
        if identity and ks in self._base_outputs:
            result=self._base_outputs[ks].copy();elapsed=perf_counter()-started
            result.update(seconds=elapsed,operator_seconds=elapsed,stage_seconds=stages,
                          base_output_reused=True)
            return result
        frozen={};levels={};certificate_bounds={};history=[];tolerances=[];native_stats=[]
        native_calls=0;iterations=0;initial={};actual_residual=self.base_residual

        def remember(scores,bound,level,tolerance=None):
            nonlocal initial
            t=perf_counter();order=top_indices(scores,min(max(ks)+1,r.n))
            gaps={k:(float('inf') if k==r.n else float(scores[order[k-1]]-scores[order[k]])) for k in ks}
            flags={k:gaps[k]>2*bound for k in ks}
            if not history:initial=flags.copy()
            for k in ks:
                if flags[k] and k not in frozen:
                    frozen[k]=order[:k].copy();levels[k]=level;certificate_bounds[k]=bound
            history.append(dict(level=level,tolerance=tolerance,error_bound=bound,gaps=gaps,certified=flags))
            stages['certificate_gap']+=perf_counter()-t
            return len(frozen)==len(ks)

        zero_offset=False
        if identity:
            scores=self.y;bound=self.base_residual/(1-alpha)+self.base_pad
            remember(scores,bound,0)
        else:
            t=perf_counter();q=offset_seed(D,self.sources,self.y,alpha);zero_offset=not np.any(q)
            stages['offset_rhs']=perf_counter()-t
            if zero_offset:
                scores=self.y
                t=perf_counter();actual_residual=self._residual(scores,D)
                stages['certificate_residual']+=perf_counter()-t
                bound=actual_residual/(1-alpha)+self.base_pad;remember(scores,bound,0)
            elif self.mode=='osp_gap':
                # Same q0 and unpruned signed recurrence as osp_certified.py;
                # one shared vector serves all three k cuts, not three runs.
                t=perf_counter();term=q;scores=self.y+term
                stages['propagation']+=perf_counter()-t
                for iterations in range(10001):
                    t=perf_counter();norm=float(np.linalg.norm(term,1))
                    pad=64*np.finfo(float).eps*(1+np.linalg.norm(scores,1))/(1-alpha)
                    bound=alpha*norm/(1-alpha)+self.base_residual/(1-alpha)+pad
                    stages['certificate_residual']+=perf_counter()-t
                    if remember(scores,bound,iterations) or norm<=(1-alpha)*TOLERANCES[-1]:break
                    if iterations==10000:raise RuntimeError('OSP gap recurrence did not finish')
                    t=perf_counter();term=alpha*(r.P@term+D@term[self.sources]);scores+=term
                    stages['propagation']+=perf_counter()-t
                t=perf_counter();actual_residual=self._residual(scores,D)
                stages['certificate_residual']+=perf_counter()-t
                bound=min(bound,actual_residual/(1-alpha)+pad)
                remember(scores,bound,iterations)
            else:
                t=perf_counter();self.graph.update_columns(self.sources,D)
                stages['update']=perf_counter()-t
                try:
                    for level,tolerance in enumerate(TOLERANCES):
                        tolerances.append(tolerance);t=perf_counter()
                        if self.mode=='native_gap':scores,stats=self.graph.solve(self.seed,alpha,tolerance)
                        else:
                            delta,stats=self.graph.solve_linear(q,alpha,tolerance);scores=self.y+delta
                        stages['native_solve']+=perf_counter()-t
                        native_calls+=1;native_stats.append(stats)
                        t=perf_counter();actual_residual=self._residual(scores,D)
                        stages['certificate_residual']+=perf_counter()-t
                        bound=actual_residual/(1-alpha)+self.base_pad
                        complete=remember(scores,bound,level,tolerance)
                        history[-1]['actual_l1_residual']=actual_residual
                        if complete:break
                finally:
                    t=perf_counter();self.graph.update_columns([],sparse.csc_matrix((r.n,0)))
                    stages['restore']=perf_counter()-t
                self.native_solve_calls+=native_calls
        t=perf_counter();order=top_indices(scores,max(ks)) if len(frozen)<len(ks) else None
        sets={};selected={}
        for k in ks:
            ids=frozen[k] if k in frozen else order[:k]
            ids=ids[np.lexsort((ids,-scores[ids]))]
            sets[k]=ids;selected[k]=scores[ids].copy()
        flags={k:k in frozen for k in ks}
        stages['output']+=perf_counter()-t;elapsed=perf_counter()-started
        result=dict(scores=scores,sets=sets,selected_scores=selected,seconds=elapsed,operator_seconds=elapsed,
            stage_seconds=stages,certificate_residual_is_algorithm_work=True,mode=self.mode,
            initial_certified=initial,completion_certified=flags,certified=flags,
            certification_level={k:levels.get(k) for k in ks},certificate_error_bounds=certificate_bounds,
            numerical_completion_ks=[k for k in ks if not flags[k]],score_error_bound=bound,
            actual_l1_residual=actual_residual,tolerances_used=tolerances,certificate_history=history,
            native_solve_calls=native_calls,native_stats=native_stats,iterations=iterations,
            identity_reused=identity,zero_offset_reused=zero_offset,base_output_reused=False,
            original_source_count=len(self.sources),payload_bytes=0)
        if identity:self._base_outputs[ks]=result.copy()
        return result


class CertifiedOSPFamily(CertifiedNativeFamily):
    """Single-propagation, three-cutoff family adapter for certified OSP."""
    def __init__(self,*args,**kwargs):
        kwargs['mode']='osp_gap';super().__init__(*args,**kwargs)
