import numpy as np
from scipy import sparse
from src.iterative_backend import MatrixFreeViewRank
from src.policy_family import PolicyFamily,DensePolicyFamily
from src.budgeted_family import BudgetedBoundaryDirectory,BudgetedDenseDirectory
from src.budgeted_family import BudgetedStencilDirectory,interaction_stencil


def test_budgeted_cross_combination_matches_uncached_without_source_projection():
    W=np.array([[2,1,1,0,0,1],[1,2,0,1,0,0],[0,1,2,1,1,0],
                [1,0,1,2,0,1],[0,1,0,1,2,1],[1,0,1,0,1,2]],float)
    P=sparse.csc_matrix(W/W.sum(0));ranker=MatrixFreeViewRank(P,alpha=.6)
    B=np.array([0,1,2,3,4,5]);compressed=BudgetedBoundaryDirectory(ranker,B,1,700)
    dense=BudgetedDenseDirectory(ranker,350)
    for qi,sources in enumerate([[4,1],[3,1],[2,4],[4,1]]):
        seed=np.eye(6)[:,qi]
        directions=[];masses=[]
        for s in sources:
            p=P[:,s].toarray().ravel();g=p.copy();g[np.arange(6)%2==0]=0
            masses.append(g.sum());directions.append(sparse.csc_matrix((g-g.sum()*p).reshape(-1,1)))
        family=PolicyFamily(ranker,seed,sources,directions,masses,directory=compressed)
        control=DensePolicyFamily(ranker,seed,sources,directions,masses,directory=dense)
        reference=DensePolicyFamily(ranker,seed,sources,directions,masses)
        for strength in [.5,1,2,4]:
            a=family.solve(strength,ks=(1,2,3));b=control.solve(strength,ks=(1,2,3));c=reference.solve(strength,ks=(1,2,3))
            for k in (1,2,3):
                assert set(a['sets'][k])==set(c['sets'][k])
                assert set(b['sets'][k])==set(c['sets'][k])
        assert compressed.nbytes<=700 and dense.nbytes<=350
    assert compressed.hits>0 and dense.evictions>0


def test_stencil_supports_unseen_clique_without_global_boundary():
    W=np.array([[3,1,0,1,0,0],[1,3,1,0,1,0],[0,1,3,1,0,1],
                [1,0,1,3,1,0],[0,1,0,1,3,1],[1,0,1,0,1,3]],float)
    P=sparse.csc_matrix(W/W.sum(0));ranker=MatrixFreeViewRank(P,alpha=.7)
    stencil=interaction_stencil([[0,1],[1,2],[0,2],[4,5]])
    assert sum(map(len,stencil.values()))<len(stencil)**2
    directory=BudgetedStencilDirectory(ranker,stencil,1,1600)
    sources=[0,1,2];directions=[];masses=[]
    for s in sources:
        p=P[:,s].toarray().ravel();g=p.copy();g[np.arange(6)%2==0]=0
        masses.append(g.sum());directions.append(sparse.csc_matrix((g-g.sum()*p).reshape(-1,1)))
    seed=np.array([.5,.1,.1,.1,.1,.1])
    family=PolicyFamily(ranker,seed,sources,directions,masses,directory=directory)
    full=DensePolicyFamily(ranker,seed,sources,directions,masses)
    for strengths in [.5,1,4,np.array([.5,2,4])]:
        a=family.solve(strengths,ks=(1,2,3));b=full.solve(strengths,ks=(1,2,3))
        for k in [1,2,3]:assert set(a['sets'][k])==set(b['sets'][k])
        assert np.all(np.abs(a['scores']-b['scores'])<=a['error_bounds']+1e-12)
    assert directory.nbytes<=1600
