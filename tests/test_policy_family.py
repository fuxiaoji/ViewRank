import numpy as np
import pytest
from scipy import sparse

from src.viewrank import ViewRank
from src.iterative_backend import MatrixFreeViewRank
from src.policy_family import PolicyFamily,DensePolicyFamily


@pytest.mark.parametrize('backend',[ViewRank,MatrixFreeViewRank])
@pytest.mark.parametrize('keep',[0,2,7])
def test_complete_family_matches_independent_view(backend,keep):
    W=np.array([[2,1,0,0,0,0,1],[1,2,1,0,0,0,0],[1,0,2,1,0,0,0],
                [0,1,0,2,1,0,0],[0,0,1,0,2,1,0],[0,0,0,1,0,2,1],[1,0,0,0,1,1,2]],dtype=float)
    P=W/W.sum(axis=0)
    sources=np.array([4,1])
    groups=[];directions=[];masses=[]
    for source in sources:
        g=P[:,source].copy();g[np.arange(7)%2==0]=0
        m=g.sum();groups.append(g);directions.append(sparse.csc_matrix((g-m*P[:,source]).reshape(-1,1)));masses.append(m)
    seed=np.array([.5,0,0,.3,0,.2,0])
    ranker=backend(sparse.csc_matrix(P),alpha=.72)
    family=PolicyFamily(ranker,seed,sources,directions,masses,keep=keep)
    dense=DensePolicyFamily(ranker,seed,sources,directions,masses)
    for strengths in [.5,1,2,4,np.array([.6,3.])]:
        # Family canonicalizes sources, so assign input vector in that order.
        ordered_strengths=np.broadcast_to(strengths,(2,))
        Pv=P.copy()
        for s,l in zip(sorted(sources),ordered_strengths):
            idx=list(sources).index(s);g=groups[idx];m=masses[idx]
            Pv[:,s]=(P[:,s]+(l-1)*g)/(1+(l-1)*m)
        exact=np.linalg.solve(np.eye(7)-.72*Pv,.28*seed)
        result=family.solve(strengths,ks=(1,3,5))
        control=dense.solve(strengths,ks=(1,3,5))
        expected=np.lexsort((np.arange(7),-exact))
        for k in (1,3,5):
            assert set(result['sets'][k])==set(expected[:k])
            assert set(control['sets'][k])==set(expected[:k])
        assert np.all(np.abs(result['scores']-exact)<=result['error_bounds']+1e-12)
        assert np.max(np.abs(control['scores']-exact))<1e-11
        assert result['fallback_solve_calls']<=1
    # No caller source is omitted to fit a global boundary.
    assert set(family.original_sources)==set(sources)


def test_identity_and_degenerate_group_require_no_response():
    P=sparse.eye(5,format='csc');ranker=MatrixFreeViewRank(P,alpha=.5)
    seed=np.array([.5,.2,.15,.1,.05])
    family=PolicyFamily(ranker,seed,[3],[sparse.csc_matrix((5,1))],[1.],keep=0)
    for strength in [.5,1,4]:
        result=family.solve(strength,ks=(1,2,3))
        assert result['fallback_solve_calls']==0
        assert np.max(np.abs(result['scores']-seed))<1e-11
    assert family.directory.atomic_solve_calls==0
