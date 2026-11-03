import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from src.viewrank import ViewRank
from src.baselines.dense_family_budget import BasisResidentDenseDirectory


@pytest.fixture
def problem():
    n=8;alpha=.5;nodes=np.arange(n)
    P=sparse.csc_matrix((np.r_[np.full(n,.6),np.full(n,.4)],
        (np.r_[nodes,(nodes+1)%n],np.r_[nodes,nodes])),shape=(n,n))
    ranker=ViewRank(P,alpha);seed=np.arange(1,n+1,dtype=float);seed/=seed.sum()
    y=ranker.base_lu.solve((1-alpha)*seed)
    def direction(source):
        return sparse.csc_matrix(([.24,-.24],([source,(source+1)%n],[0,0])),shape=(n,1))
    def reference(sources,strength):
        a=(strength-1)/(1+(strength-1)*.6)
        D=sparse.hstack([direction(s)*a for s in sources],format='csc')
        E=sparse.csc_matrix((np.ones(len(sources)),(np.arange(len(sources)),sources)),shape=(len(sources),n))
        return spsolve(sparse.eye(n)-alpha*(P+D@E),(1-alpha)*seed)
    class Point:
        def __init__(self):self.current=P;self.calls=0;self.raw_calls=0;self.direct_calls=0;self.updates=[]
        def update_columns(self,sources,D):
            self.updates.append(list(sources))
            E=sparse.csc_matrix((np.ones(len(sources)),(np.arange(len(sources)),sources)),shape=(len(sources),n))
            self.current=P+D@E
        def solve(self,reset,alpha,tol):
            self.calls+=1;self.direct_calls+=1
            return spsolve(sparse.eye(n)-alpha*self.current,(1-alpha)*reset),{}
        def solve_linear(self,rhs,alpha,tol):
            self.calls+=1;self.raw_calls+=1
            return spsolve(sparse.eye(n)-alpha*self.current,rhs),{}
    return ranker,seed,y,direction,reference,Point()


def test_no_response_copy_and_hits_are_pinned_before_misses(problem):
    r,seed,y,direction,reference,point=problem
    cache=BasisResidentDenseDirectory(r,350,point_graph=point)
    with cache.prepare(seed,[0,1],[direction(0),direction(1)],[.6,.6],base_scores=y) as family:
        assert family.route=='dense' and not hasattr(family,'responses')
        assert family.memory['live_response_bytes']==2*r.n*8
        for strength in [.5,1,2,4]:
            result=family.solve(strength,ks=(2,5))
            assert np.max(abs(result['scores']-reference([0,1],strength)))<1e-12
            for k in [2,5]:
                assert np.array_equal(result['selected_scores'][k],result['scores'][result['sets'][k]])
        cached=family.atoms[1]
        assert cached.response is cache.cache[family.keys[1]].response
        assert cache.nbytes<=350
    assert not cache._pins and cache.memory['live_core_metadata_bytes']==0
    with cache.prepare(seed,[1,2],[direction(1),direction(2)],[.6,.6],base_scores=y) as family:
        assert family.route=='dense' and family.atoms[0] is cached
        assert cache.evictions==1 and cache.nbytes<=350
    assert cache.peak_bytes==336


def test_oversized_family_uses_complete_native_targets_and_restores_base(problem):
    r,seed,y,direction,reference,point=problem
    cache=BasisResidentDenseDirectory(r,350,point_graph=point)
    sources=[3,0,2]
    with cache.prepare(seed,sources,[direction(s) for s in sources],[.6]*3,base_scores=y) as family:
        assert family.route=='native' and cache.atomic_solve_calls==0
        for strength in [.5,1,2,4]:
            result=family.solve(strength,ks=(2,5))
            assert np.max(abs(result['scores']-reference(sources,strength)))<1e-12
            assert result['identity_reused']==(strength==1)
        assert point.updates==[sources,[],sources,[],sources,[]]
    assert point.calls==3 and point.raw_calls==3 and point.direct_calls==0
    assert cache.native_offset_calls==3 and (point.current-r.P).nnz==0
    tiny=BasisResidentDenseDirectory(r,100,point_graph=point)
    with tiny.prepare(seed,[0,1],[direction(0),direction(1)],[.6,.6],base_scores=y) as family:
        assert family.required_response_bytes>100 and family.route=='native'
        assert tiny.atomic_solve_calls==0


def test_other_live_families_cannot_be_evicted_and_zero_directions_need_no_basis(problem):
    r,seed,y,direction,_,point=problem
    cache=BasisResidentDenseDirectory(r,350,point_graph=point)
    with cache.prepare(seed,[1],[direction(1)],[.6],base_scores=y) as first:
        pinned=first.atoms[0]
        with cache.prepare(seed,[1,2],[direction(1),direction(2)],[.6,.6],base_scores=y) as second:
            assert second.route=='native' and second.route_reason=='other_live_family_pins'
            assert cache.cache[first.keys[0]] is pinned
    assert not cache._pins and cache.nbytes<=350
    zero=sparse.csc_matrix((r.n,1))
    with cache.prepare(seed,[0],[zero],[0.],base_scores=y) as family:
        assert family.route=='dense' and family.required_response_bytes==0
        result=family.solve(4,ks=(2,5))
        assert np.array_equal(result['scores'],y)
        assert result['identity_reused'] and result['native_direct_calls']==0
