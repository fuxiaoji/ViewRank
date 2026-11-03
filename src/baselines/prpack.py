"""Thin native author-PRPACK adapter with fixed-structure weight updates."""
from pathlib import Path
from time import perf_counter
import ctypes as ct

import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[2]
DLL=ROOT/"baselines/build/Release/prpack_native.dll"
_lib=None


def library():
    global _lib
    if _lib is None:
        _lib=ct.CDLL(str(DLL))
        ip=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
        dp=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
        _lib.prpack_create_csr.argtypes=[ct.c_int,ct.c_int,ip,ip,dp,ct.c_int,ct.POINTER(ct.c_double)]
        _lib.prpack_create_csr.restype=ct.c_void_p
        _lib.prpack_solve_personalized.argtypes=[ct.c_void_p,dp,ct.c_double,ct.c_double,dp,dp]
        _lib.prpack_solve_linear_scc.argtypes=[ct.c_void_p,dp,ct.c_double,ct.c_double,dp,dp]
        _lib.prpack_update_weights.argtypes=[ct.c_void_p,ct.c_int,ip,dp]
        _lib.prpack_update_weights.restype=None
        _lib.prpack_memory.argtypes=[ct.c_void_p,dp]
        _lib.prpack_memory.restype=None
        _lib.prpack_destroy.argtypes=[ct.c_void_p]
        _lib.prpack_destroy.restype=None
    return _lib


class PRPackGraph:
    """P[target,source], weighted directed normalized transition matrix.

    Graph construction copies CSR arrays. First solve builds upstream SCC-GS
    or global GS preprocessing. Positive support-preserving column updates
    reuse that structure and change mapped numeric weights; no kernel changes.
    """
    def __init__(self,P,method='sccgs'):
        t=perf_counter()
        self.P=sparse.csc_matrix(P,dtype=np.float64)
        if not self.P.has_canonical_format:
            self.P=self.P.copy();self.P.sum_duplicates();self.P.sort_indices()
        self.shared_base_csc=sparse.isspmatrix_csc(P) and all(np.shares_memory(a,b) for a,b in
            [(self.P.data,P.data),(self.P.indices,P.indices),(self.P.indptr,P.indptr)])
        self.n=self.P.shape[0]
        self.method=method
        if method not in ['sccgs','gs']:raise ValueError('method must be author sccgs or gs')
        csr=self.P.tocsr();csr.sum_duplicates();csr.sort_indices()
        indptr=np.ascontiguousarray(csr.indptr,dtype=np.int32)
        indices=np.ascontiguousarray(csr.indices,dtype=np.int32)
        values=np.ascontiguousarray(csr.data,dtype=np.float64)
        native_seconds=ct.c_double()
        self.handle=library().prpack_create_csr(self.n,csr.nnz,indptr,indices,values,int(method=='sccgs'),ct.byref(native_seconds))
        # CSR array positions mapped into source-column order once.
        ids=sparse.csr_matrix((np.arange(csr.nnz,dtype=np.int32),csr.indices,csr.indptr),shape=csr.shape).tocsc()
        self.csc_to_csr=np.ascontiguousarray(ids.data,dtype=np.int32)
        self.previous_sources=set()
        self.setup_seconds=perf_counter()-t
        self.native_copy_seconds=native_seconds.value
        self.total_preprocess_seconds=0.

    def solve(self,seed,alpha=.85,tol=1e-12):
        seed=np.ascontiguousarray(seed,dtype=np.float64)
        scores=np.empty(self.n,dtype=np.float64);stats=np.zeros(5)
        t=perf_counter()
        library().prpack_solve_personalized(self.handle,seed,alpha,tol,scores,stats)
        wall=perf_counter()-t;self.total_preprocess_seconds+=stats[1]
        return scores,dict(solve_seconds=wall,native_seconds=stats[0],preprocess_seconds=stats[1],
                           compute_seconds=stats[2],edge_map_seconds=stats[4],edges_touched=int(stats[3]))

    def update_columns(self,sources,deltas):
        """Set current view to base P+D E_S.T, restoring prior changed sources."""
        from src.viewrank import _canonical_update
        t=perf_counter();D,sources=_canonical_update(deltas,sources,self.n)
        selected={int(s):j for j,s in enumerate(sources)}
        positions=[];values=[]
        for source in sorted(self.previous_sources|set(selected)):
            lo,hi=self.P.indptr[source:source+2]
            column=self.P.data[lo:hi].copy()
            if source in selected:
                update=D[:,selected[source]]
                rows=self.P.indices[lo:hi]
                offsets=np.searchsorted(rows,update.indices)
                if np.any(offsets>=len(rows)) or np.any(rows[offsets]!=update.indices):
                    raise ValueError("native fixed-SCC path requires unchanged edge support")
                column[offsets]+=update.data
            if np.any(column<=0):
                raise ValueError("native fixed-SCC path requires positive retained weights")
            positions.append(self.csc_to_csr[lo:hi]);values.append(column)
        if positions:
            pos=np.ascontiguousarray(np.concatenate(positions),dtype=np.int32)
            val=np.ascontiguousarray(np.concatenate(values),dtype=np.float64)
            library().prpack_update_weights(self.handle,len(pos),pos,val)
        self.previous_sources=set(selected)
        return perf_counter()-t

    def solve_linear(self,rhs,alpha=.85,tol=1e-12):
        """One unnormalized author SCC-GS solve for an arbitrary signed RHS.

        This is A^-1 rhs, not a probability-normalized PageRank vector. Global
        GS's mass-based stopping rule is not used for signed RHS.
        """
        if self.method!='sccgs':raise ValueError('raw signed solve requires the author SCC-GS kernel')
        rhs=np.ascontiguousarray(rhs,dtype=np.float64)
        scores=np.empty(self.n,dtype=np.float64);stats=np.zeros(5)
        t=perf_counter()
        library().prpack_solve_linear_scc(self.handle,rhs,alpha,tol,scores,stats)
        wall=perf_counter()-t;self.total_preprocess_seconds+=stats[1]
        return scores,dict(solve_seconds=wall,native_seconds=stats[0],preprocess_seconds=stats[1],
                           compute_seconds=stats[2],edge_map_seconds=stats[4],edges_touched=int(stats[3]))

    @property
    def memory(self):
        stats=np.zeros(4);library().prpack_memory(self.handle,stats)
        return dict(native_base_graph_bytes=int(stats[0]),native_scc_arrays_bytes=int(stats[1]) if self.method=='sccgs' else 0,
                    native_gs_arrays_bytes=int(stats[1]) if self.method=='gs' else 0,
                    native_weight_map_bytes=int(stats[2]),scc_components=int(stats[3]),
                    python_base_csc_bytes=0 if self.shared_base_csc else self.P.data.nbytes+self.P.indices.nbytes+self.P.indptr.nbytes,
                    python_csr_bytes=0,
                    python_column_map_bytes=self.csc_to_csr.nbytes)

    def close(self):
        if self.handle:
            library().prpack_destroy(self.handle);self.handle=None

    def __del__(self):
        if getattr(self,"handle",None):self.close()


def personalized_pagerank(P,alpha,seed,tol=1e-12,method='sccgs'):
    graph=PRPackGraph(P,method=method)
    scores,stats=graph.solve(seed,alpha,tol)
    stats["setup_seconds"]=graph.setup_seconds;stats["memory"]=graph.memory
    graph.close();return scores,stats


class PRPackLinearSolver:
    """A^-1 RHS using one raw SCC-GS call per nonzero column by default.

    Global GS, or direct_signed=False, uses positive/negative probability
    splits and linearity. The positive PageRank point API remains unchanged.

    After solve(), last_residuals contains the already measured per-column
    ||rhs-A@output||_1; last_residual_l1 is their maximum (the same scalar for
    a vector RHS). Capture this metadata immediately alongside the returned
    response, before another solve overwrites it. No vectors are cached.
    """
    def __init__(self,P,alpha=.85,tol=1e-12,A=None,method='sccgs',direct_signed=True):
        self.graph=PRPackGraph(P,method=method);self.alpha=alpha;self.tol=tol
        self.direct_signed=bool(direct_signed)
        self.calls=self.rhs_columns=self.native_calls=0
        self.max_returned_residual=0.;self.elapsed_seconds=0.
        self.last_residuals=np.empty(0,dtype=float)
        self.last_residual_l1=None
        self.A=A if A is not None else (sparse.eye(P.shape[0],format="csc")-alpha*P).tocsc()

    def solve(self,rhs):
        t=perf_counter();rhs=np.asarray(rhs,dtype=float)
        matrix=rhs.reshape(-1,1) if rhs.ndim==1 else rhs
        output=np.zeros_like(matrix)
        for j in range(matrix.shape[1]):
            if self.direct_signed and self.graph.method=='sccgs':
                if np.any(matrix[:,j]):
                    output[:,j],_=self.graph.solve_linear(matrix[:,j],self.alpha,self.tol)
                    self.native_calls+=1
                continue
            positive=np.maximum(matrix[:,j],0);negative=np.maximum(-matrix[:,j],0)
            mass=float(positive.sum()+negative.sum())
            if mass==0:continue
            budget=self.tol*(1-self.alpha)/mass
            for sign,part in [(1.,positive),(-1.,negative)]:
                m=float(part.sum())
                if m:
                    scores,_=self.graph.solve(part/m,self.alpha,budget)
                    output[:,j]+=sign*m/(1-self.alpha)*scores
                    self.native_calls+=1
        self.last_residuals=np.sum(np.abs(matrix-self.A@output),axis=0)
        self.last_residual_l1=float(np.max(self.last_residuals))
        self.max_returned_residual=max(self.max_returned_residual,self.last_residual_l1)
        self.calls+=1;self.rhs_columns+=matrix.shape[1];self.elapsed_seconds+=perf_counter()-t
        return output[:,0] if rhs.ndim==1 else output


from src.viewrank import ViewRank, _sparse_bytes


class PRPackViewRank(ViewRank):
    """ViewRank API backed by author PRPACK SCC-GS, without base sparse LU.

    Lazy SCC construction and native edge mapping are charged to the first
    .base_lu.solve(). Signed SCC responses use the original kernel's existing
    unnormalized mode; global GS continues to split positive/negative parts.
    tol controls the original SCC residual-based stopping rule. Each returned
    response's actual L1 residual is measured for posterior certification;
    residual/(1-alpha) bounds its L1 score error. Split mode allocates the
    original personalized solver's declared L1 score-error tolerance.
    """
    def __init__(self,P,alpha=.85,tol=1e-12,method='sccgs',direct_signed=True):
        t=perf_counter()
        self.P=sparse.csc_matrix(P,dtype=np.float64)
        self.n=self.P.shape[0];self.alpha=float(alpha)
        self.A=(sparse.eye(self.n,format="csc")-self.alpha*self.P).tocsc()
        self.base_lu=PRPackLinearSolver(self.P,self.alpha,tol,A=self.A,method=method,direct_signed=direct_signed)
        self.base_factor_seconds=perf_counter()-t
        self.cache={};self.cache_hits=self.cache_misses=0

    @property
    def base_payload_bytes(self):
        memory=self.base_lu.graph.memory
        return sum(_sparse_bytes(m) for m in [self.P,self.A])+sum(
            value for name,value in memory.items() if name.endswith("_bytes"))
