import numpy as np
from scipy import sparse
from src.iterative_backend import MatrixFreeViewRank
from src.budgeted_family import BudgetedStencilDirectory,interaction_stencil
from src.policy_family import DensePolicyFamily
from src.policy_family_fast import FastPolicyFamily,top_indices


def test_partial_topk_ties_match_full_lexicographic_sort():
    scores=np.array([1.,3.,3.,2.,2.,2.,0.,3.])
    for k in range(1,len(scores)+1):
        assert np.array_equal(top_indices(scores,k),np.lexsort((np.arange(len(scores)),-scores))[:k])


def test_sparse_candidates_and_every_k_match_dense_for_different_strengths():
    n=31;W=np.eye(n)*2
    for s in range(n):
        W[(s+1)%n,s]=1;W[(s+4)%n,s]=.7;W[(s+9)%n,s]=.2
    P=sparse.csc_matrix(W/W.sum(0));ranker=MatrixFreeViewRank(P,alpha=.65)
    sources=[2,7,14];stencil=interaction_stencil([sources,[2,5],[7,11],[14,18]])
    seed=np.zeros(n);seed[[2,5,8]]=[.5,.3,.2];ds=[];masses=[]
    for s in sources:
        p=P[:,s].toarray().ravel();g=p.copy();g[np.arange(n)%2==0]=0
        masses.append(g.sum());ds.append(sparse.csc_matrix((g-g.sum()*p).reshape(-1,1)))
    for keep in [0,2,8,31]:
        directory=BudgetedStencilDirectory(ranker,stencil,keep,20000)
        fast=FastPolicyFamily(ranker,seed,sources,ds,masses,directory=directory)
        dense=DensePolicyFamily(ranker,seed,sources,ds,masses)
        for strength in [.5,1,2,4,np.array([.5,2,4])]:
            result=fast.solve(strength,materialize=True);ref=dense.solve(strength)
            for k in [2,5,10]:assert set(result['sets'][k])==set(ref['sets'][k])
            assert np.all(np.abs(result['scores']-ref['scores'])<=result['error_bounds']+1e-12)


def test_no_active_groups_and_completion_bound_with_approximate_core():
    P=sparse.csc_matrix(np.array([[.3,.2,.4],[.5,.4,.1],[.2,.4,.5]]))
    seed=np.array([.6,.3,.1]);ranker=MatrixFreeViewRank(P,alpha=.7,tol=1e-8)
    group=P[:,0].toarray().ravel();group[1:]=0;m=group.sum()
    direction=sparse.csc_matrix((group-m*P[:,0].toarray().ravel()).reshape(-1,1))
    stencil=interaction_stencil([[0]])
    d=BudgetedStencilDirectory(ranker,stencil,0,1000)
    f=FastPolicyFamily(ranker,seed,[0],[direction],[m],directory=d,ks=(1,2))
    result=f.solve(4,ks=(1,2),always_fused=True,materialize=True)
    Pv=P.toarray();Pv[:,0]=(Pv[:,0]+3*group)/(1+3*m)
    exact=np.linalg.solve(np.eye(3)-.7*Pv,.3*seed)
    assert np.all(np.abs(result['scores']-exact)<=result['error_bounds']+1e-14)
    empty=FastPolicyFamily(ranker,seed,[2],[sparse.csc_matrix((3,1))],[0.],ks=(1,2))
    result=empty.solve(.5,materialize=True)
    assert set(result['sets'][1])=={0}
