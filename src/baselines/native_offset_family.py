"""Native OSP equation: solve one signed offset on each complete temporary view.

The original PRPACK SCC-GS kernel is used in its existing unnormalized mode.
No response basis or answer cache is retained. Native weights are restored
after every changed target, so the shared base solver remains on P0.
"""
from time import perf_counter

import numpy as np
from scipy import sparse

from src.policy_family_fast import top_indices
from src.semantic_basis import strength_scale


class NativeOffsetFamily:
    """Complete original S, general reset, scalar or per-source strengths.

    ranker supplies P/alpha/n/base_lu; its graph must expose the raw signed
    SCC `solve_linear`. An explicitly supplied point_graph can instead be used.
    Pass the shared base_scores and its captured base_residual to avoid paying
    a second base solve/check. prepare_seconds/base_seconds record local work.
    """
    def __init__(self,ranker,seed,sources,directions,masses,base_scores=None,
                 base_residual=None,point_graph=None,tol=1e-12):
        started=perf_counter();self.ranker=ranker;self.tol=float(tol)
        self.seed=np.asarray(seed,dtype=float)
        self.sources=np.asarray(sources,dtype=np.int64)
        self.directions=tuple(directions);self.masses=np.asarray(masses,dtype=float)
        self.graph=ranker.base_lu.graph if point_graph is None else point_graph
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
        self.preparation_stages=dict(base=self.base_seconds,base_residual=self.base_residual_seconds)
        self.prepare_seconds=perf_counter()-started
        self.native_solve_calls=0

    @property
    def nbytes(self):
        """Private retained response-basis bytes; shared y/input/backend separate."""
        return 0

    def solve(self,strengths,ks=(2,5,10)):
        started=perf_counter();r=self.ranker;alpha=r.alpha
        stages=dict(assembly=0.,offset_rhs=0.,update=0.,raw_solve=0.,restore=0.,
                    residual_check=0.,output=0.)
        t=perf_counter()
        strengths=np.broadcast_to(np.asarray(strengths,dtype=float),self.masses.shape)
        scales=np.array([strength_scale(l,m) for l,m in zip(strengths,self.masses)])
        identity=not any(a and direction.nnz for a,direction in zip(scales,self.directions))
        D=(sparse.hstack([direction*a for direction,a in zip(self.directions,scales)],format='csc')
           if len(scales) and not identity else sparse.csc_matrix((r.n,len(self.sources))))
        D.eliminate_zeros();identity=not D.nnz
        stages['assembly']=perf_counter()-t
        native_stats=None;native_calls=0;zero_offset=False;delta_residual=0.
        if identity:
            t=perf_counter();scores=self.y.copy();score_residual=self.base_residual
            stages['output']=perf_counter()-t
        else:
            t=perf_counter();q=alpha*(D@self.y[self.sources]);stages['offset_rhs']=perf_counter()-t
            zero_offset=not np.any(q)
            if zero_offset:
                t=perf_counter();scores=self.y.copy();score_residual=self.base_residual
                stages['output']=perf_counter()-t
            else:
                t=perf_counter();self.graph.update_columns(self.sources,D)
                stages['update']=perf_counter()-t
                try:
                    t=perf_counter();delta,native_stats=self.graph.solve_linear(q,alpha,self.tol)
                    stages['raw_solve']=perf_counter()-t
                finally:
                    t=perf_counter();self.graph.update_columns([],sparse.csc_matrix((r.n,0)))
                    stages['restore']=perf_counter()-t
                native_calls=1;self.native_solve_calls+=1
                t=perf_counter();scores=self.y+delta;stages['output']=perf_counter()-t
                # Graph.P is the immutable base even while native weights are
                # changed. Explicit P0*z+D*z[S] checks the actual target Av.
                t=perf_counter()
                delta_residual=float(np.linalg.norm(q-delta+alpha*(r.P@delta+D@delta[self.sources]),1))
                score_residual=float(np.linalg.norm((1-alpha)*self.seed-scores+
                    alpha*(r.P@scores+D@scores[self.sources]),1))
                stages['residual_check']=perf_counter()-t
        t=perf_counter();ids=top_indices(scores,max(ks))
        sets={int(k):ids[:k] for k in ks}
        selected={int(k):scores[ids[:k]].copy() for k in ks}
        stages['output']+=perf_counter()-t
        operator_stage_sum=sum(value for name,value in stages.items() if name!='residual_check')
        elapsed=perf_counter()-started
        operator=elapsed-stages['residual_check']
        return dict(scores=scores,sets=sets,selected_scores=selected,
            seconds=elapsed,operator_seconds=operator,operator_stage_sum=operator_stage_sum,stage_seconds=stages,
            family_base_seconds=self.base_seconds,family_base_residual_seconds=self.base_residual_seconds,
            identity_reused=identity,zero_offset_reused=zero_offset,
            native_solve_calls=native_calls,native_stats=native_stats,
            actual_l1_residual=score_residual,offset_l1_residual=delta_residual,
            posterior_l1_error_bound=score_residual/(1-alpha),payload_bytes=0,
            original_source_count=len(self.sources))
