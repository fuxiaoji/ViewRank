import numpy as np
from scipy import sparse
from src.iterative_backend import MatrixFreeViewRank
from src.budgeted_family import BudgetedStencilDirectory,interaction_stencil
from src.policy_family_fast import FastPolicyFamily
from src.full_scan_policy_family import FullScanPolicyFamily


def test_full_scan_matches_candidates_and_reuses_identical_completion():
    assert FullScanPolicyFamily._complete is FastPolicyFamily._complete
    n=31;W=np.eye(n)*2
    for source in range(n):
        W[(source+1)%n,source]=1;W[(source+4)%n,source]=.7;W[(source+9)%n,source]=.2
    P=sparse.csc_matrix(W/W.sum(0));ranker=MatrixFreeViewRank(P,alpha=.65,tol=1e-9)
    sources=[2,7,14];seed=np.zeros(n);seed[[2,5,8]]=[.5,.3,.2]
    directions=[];masses=[]
    for source in sources:
        p=P[:,source].toarray().ravel();group=p.copy();group[np.arange(n)%2==0]=0
        mass=group.sum();masses.append(mass)
        directions.append(sparse.csc_matrix((group-mass*p)[:,None]))
    stencil=interaction_stencil([sources,[2,5],[7,11],[14,18]])
    for keep in [0,2,8,31]:
        directory=BudgetedStencilDirectory(ranker,stencil,keep,20000)
        fast=FastPolicyFamily(ranker,seed,sources,directions,masses,directory=directory)
        full=FullScanPolicyFamily(ranker,seed,sources,directions,masses,directory=directory)
        for strength in [.5,1,2,4,np.array([.5,2,4])]:
            a=fast.solve(strength,materialize=True);b=full.solve(strength,materialize=True)
            assert a['initial_certified']==b['initial_certified']
            assert a['fallback_solve_calls']==b['fallback_solve_calls']
            assert np.max(abs(a['scores']-b['scores']))<1e-14
            assert np.max(abs(a['error_bounds']-b['error_bounds']))<1e-14
            for k in [2,5,10]:assert np.array_equal(a['sets'][k],b['sets'][k])
        a=fast.solve(4,always_fused=True,materialize=True)
        b=full.solve(4,always_fused=True,materialize=True)
        assert np.array_equal(a['scores'],b['scores'])
        assert a['completion_certified']==b['completion_certified']


def test_full_scan_with_no_active_directions():
    P=sparse.csc_matrix([[.3,.2,.4],[.5,.4,.1],[.2,.4,.5]])
    ranker=MatrixFreeViewRank(P,alpha=.7)
    family=FullScanPolicyFamily(ranker,np.array([.6,.3,.1]),[2],
        [sparse.csc_matrix((3,1))],[0.],ks=(1,2))
    for strength in [.5,1,2,4]:
        result=family.solve(strength,materialize=True)
        assert result['fallback_solve_calls']==0
        assert result['candidate_nodes']==3
        assert set(result['sets'][1])=={0}
