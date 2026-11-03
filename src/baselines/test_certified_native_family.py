import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from src.baselines.prpack import PRPackViewRank
from src.baselines.certified_native_family import CertifiedNativeFamily,CertifiedOSPFamily,TOLERANCES
from src.osp_certified import osp_topk
from src.policy_family_fast import top_indices
from src.semantic_basis import strength_scale


def fixture():
    rng=np.random.default_rng(18);n=17;W=rng.uniform(.1,1,(n,n))
    P=sparse.csc_matrix(W/W.sum(0));ranker=PRPackViewRank(P,alpha=.5)
    seed=np.arange(n,0,-1,dtype=float)**2;seed/=seed.sum()
    S=np.array([9,2,13]);directions=[];masses=[]
    for source in S:
        p=P[:,source].toarray().ravel();g=p.copy();g[np.arange(n)%2==0]=0
        m=float(g.sum());directions.append(sparse.csc_matrix((g-m*p)[:,None]));masses.append(m)
    y=ranker.base_lu.solve(.5*seed);ry=ranker.base_lu.last_residual_l1
    return ranker,seed,S,directions,masses,y,ry


@pytest.mark.parametrize('mode',['native_gap','offset_gap','osp_gap'])
def test_early_stop_full_sources_and_actual_view_bound(mode):
    r,seed,S,directions,masses,y,ry=fixture()
    family=CertifiedNativeFamily(r,seed,S,directions,masses,y,ry,mode=mode)
    for strength in [.5,2,4]:
        result=family.solve(strength)
        D=sparse.hstack([d*strength_scale(strength,m) for d,m in zip(directions,masses)],format='csc')
        E=sparse.csc_matrix((np.ones(len(S)),(np.arange(len(S)),S)),shape=(len(S),r.n));Pv=r.P+D@E
        exact=spsolve(sparse.eye(r.n)-.5*Pv,.5*seed)
        assert np.max(abs(result['scores']-exact))<=result['score_error_bound']+1e-14
        assert all(result['certified'].values())
        for k in [2,5,10]:assert set(result['sets'][k])==set(top_indices(exact,k))
        assert result['seconds']==result['operator_seconds']
        assert result['stage_seconds']['certificate_residual']>0
        assert 'residual_check' not in result['stage_seconds']
        assert not r.base_lu.graph.previous_sources
        if mode!='osp_gap':
            assert result['tolerances_used'][-1]>TOLERANCES[-1]
            assert result['native_solve_calls']==len(result['tolerances_used'])
        else:
            for k in [2,5,10]:
                prior=osp_topk(Pv,D,S,y,k=k,alpha=.5,tol=1e-12,base_residual_l1=ry)
                assert set(prior.indices)==set(result['sets'][k])
    assert np.max(abs(r.base_lu.solve(.5*seed)-y))<1e-12
    r.base_lu.graph.close()


@pytest.mark.parametrize('mode',['native_gap','offset_gap'])
def test_each_changed_native_target_updates_once_and_restores(mode):
    r,seed,S,directions,masses,y,ry=fixture();graph=r.base_lu.graph
    update=graph.update_columns;calls=[]
    def tracked(sources,D):calls.append(list(sources));return update(sources,D)
    graph.update_columns=tracked
    family=CertifiedNativeFamily(r,seed,S,directions,masses,y,ry,mode=mode)
    result=family.solve(4)
    assert calls==[S.tolist(),[]]
    assert result['original_source_count']==len(S)
    graph.close()


@pytest.mark.parametrize('mode',['native_gap','offset_gap','osp_gap'])
def test_ties_keep_final_numerical_flags_and_noeffect_targets(mode):
    n=12;P=sparse.csc_matrix(np.full((n,n),1/n));r=PRPackViewRank(P,alpha=.5)
    seed=np.full(n,1/n);y=r.base_lu.solve(.5*seed);ry=r.base_lu.last_residual_l1
    d=sparse.csc_matrix(([.01,-.01],([0,1],[0,0])),shape=(n,1))
    family=CertifiedNativeFamily(r,seed,[0],[d],[.5],y,ry,mode=mode)
    result=family.solve(4)
    assert result['numerical_completion_ks']
    assert not all(result['certified'].values())
    if mode!='osp_gap':assert result['tolerances_used']==list(TOLERANCES)
    zero=CertifiedNativeFamily(r,seed,[0],[sparse.csc_matrix((n,1))],[0.],y,ry,mode=mode)
    for index,strength in enumerate([.5,1,2,4]):
        result=zero.solve(strength)
        assert result['native_solve_calls']==0 and result['identity_reused']
        assert np.array_equal(result['scores'],y)
        assert result['base_output_reused']==(index>0)
    r.base_lu.graph.close()
