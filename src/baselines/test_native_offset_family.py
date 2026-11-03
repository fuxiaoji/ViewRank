import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from src.baselines.native_offset_family import NativeOffsetFamily
from src.baselines.prpack import PRPackViewRank
from src.semantic_basis import strength_scale


@pytest.mark.parametrize('alpha',[.5,.85])
def test_complete_view_offset_matches_lu_and_checks_actual_target(alpha):
    rng=np.random.default_rng(9);n=9;W=np.zeros((n,n))
    for source in range(n):
        start=(source//3)*3;W[start:start+3,source]=rng.uniform(.3,1,3)
        if start<6:W[start+3,source]=.15
    P=sparse.csc_matrix(W/W.sum(axis=0));r=PRPackViewRank(P,alpha=alpha)
    seed=np.arange(1,n+1,dtype=float);seed/=seed.sum()
    y=r.base_lu.solve((1-alpha)*seed);ry=r.base_lu.last_residual_l1
    S=np.array([6,0,4]);directions=[];masses=[]
    for source in S:
        p=P[:,source];g=sparse.csc_matrix(([p[source,0]],([source],[0])),shape=(n,1))
        m=float(g.sum());directions.append(g-m*p);masses.append(m)
    family=NativeOffsetFamily(r,seed,S,directions,masses,base_scores=y,base_residual=ry)
    for strengths in [.5,1,2,4,np.array([.5,2,4])]:
        result=family.solve(strengths,ks=(2,5))
        scales=[strength_scale(l,m) for l,m in zip(np.broadcast_to(strengths,(3,)),masses)]
        D=sparse.hstack([d*a for d,a in zip(directions,scales)],format='csc')
        E=sparse.csc_matrix((np.ones(3),(np.arange(3),S)),shape=(3,n));Pv=P+D@E
        expected=spsolve(sparse.eye(n)-alpha*Pv,(1-alpha)*seed)
        assert np.max(abs(result['scores']-expected))<1e-11
        actual=float(np.linalg.norm((1-alpha)*seed-result['scores']+alpha*(Pv@result['scores']),1))
        assert abs(result['actual_l1_residual']-actual)<1e-15
        assert result['actual_l1_residual']<1e-12
        assert result['offset_l1_residual']<1e-12
        assert result['original_source_count']==3
        for k in [2,5]:
            assert np.array_equal(result['selected_scores'][k],result['scores'][result['sets'][k]])
        assert not r.base_lu.graph.previous_sources
        if np.any(scales):
            delta=result['scores']-y;q=alpha*(D@y[S])
            # A0 is deliberately the wrong residual operator for the offset.
            assert np.linalg.norm(q-r.A@delta,1)>1e-6
        else:
            assert result['identity_reused'] and result['native_solve_calls']==0
    assert np.max(abs(r.base_lu.solve((1-alpha)*seed)-y))<1e-12
    r.base_lu.graph.close()


def test_zero_directions_empty_sources_and_zero_offset_reuse_base():
    P=sparse.block_diag([np.full((3,3),1/3),np.full((3,3),1/3)],format='csc')
    r=PRPackViewRank(P,alpha=.5);seed=np.array([1.,0,0,0,0,0])
    y=r.base_lu.solve(.5*seed);ry=r.base_lu.last_residual_l1
    zero=sparse.csc_matrix((6,1))
    families=[NativeOffsetFamily(r,seed,[0],[zero],[0.],base_scores=y,base_residual=ry),
              NativeOffsetFamily(r,seed,[],[],[],base_scores=y,base_residual=ry)]
    for family in families:
        for strength in [.5,1,2,4]:
            result=family.solve(strength,ks=(2,5))
            assert result['identity_reused'] and result['native_solve_calls']==0
            assert np.array_equal(result['scores'],y)
    g=sparse.csc_matrix(([1/3],([3],[0])),shape=(6,1));d=g-P[:,3]/3
    family=NativeOffsetFamily(r,seed,[3],[d],[1/3],base_scores=y,base_residual=ry)
    result=family.solve(4,ks=(2,5))
    assert result['zero_offset_reused'] and result['native_solve_calls']==0
    assert np.array_equal(result['scores'],y)
    assert not r.base_lu.graph.previous_sources
    r.base_lu.graph.close()
