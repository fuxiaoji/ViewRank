import numpy as np
from scipy import sparse
from src.baselines.prpack import PRPackViewRank
from src.budgeted_family import interaction_stencil
from src.retained_policy_family import RetainedStencilDirectory


def test_retained_indexes_fit_and_oversized_family_uses_complete_native_view():
    n=12;W=np.eye(n)
    for s in range(n):W[(s+1)%n,s]=2;W[(s+3)%n,s]=1
    P=sparse.csc_matrix(W/W.sum(0));ranker=PRPackViewRank(P,alpha=.5)
    seed=np.eye(n)[:,0];sources=[0,2,5];dirs=[];masses=[];groups=[]
    for s in sources:
        p=P[:,s].toarray().ravel();g=p.copy();g[np.arange(n)%2==0]=0
        groups.append(g);masses.append(g.sum());dirs.append(sparse.csc_matrix((g-g.sum()*p).reshape(-1,1)))
    y=ranker.base_lu.solve(.5*seed);residual=float(ranker.base_lu.last_residuals[0])
    for budget in [150,10000]:
        directory=RetainedStencilDirectory(ranker,interaction_stencil([sources]),2,budget)
        with directory.prepare(seed,sources,dirs,masses,base_scores=y,base_residual=residual) as family:
            for strength in [.5,1,4]:
                result=family.solve(strength,ks=(2,5,10))
                Pv=P.toarray()
                for s,g,m in zip(sources,groups,masses):Pv[:,s]=(Pv[:,s]+(strength-1)*g)/(1+(strength-1)*m)
                expected=np.linalg.solve(np.eye(n)-.5*Pv,.5*seed)
                order=np.lexsort((np.arange(n),-expected))
                for k in [2,5,10]:assert set(result['sets'][k])==set(order[:k])
                assert directory.memory['total_private_retained_basis_bytes']<=budget
        if budget==150:assert directory.atomic_solve_calls==0
        assert directory._live_core_bytes==0
    ranker.base_lu.graph.close()


def test_many_zero_groups_reserve_full_original_boundary_before_solving_atoms():
    from src.iterative_backend import MatrixFreeViewRank
    from src.viewrank import _sparse_bytes
    n=100;r=20;sources=list(range(n))
    P=sparse.csc_matrix(np.full((n,n),1/n));ranker=MatrixFreeViewRank(P,alpha=.5)
    d=sparse.csc_matrix(([.002,-.002],([0,1],[0,0])),shape=(n,1))
    zero=sparse.csc_matrix((n,1));directions=[d]*r+[zero]*(n-r);masses=[.5]*r+[0.]*(n-r)
    stencil=interaction_stencil([sources])
    plan=sum(v.nbytes for v in stencil.values())+len(stencil)*8
    atom_upper=r*(_sparse_bytes(d)+16*n+32)
    old_fused_upper=16*r*r+80*(r+len(sources))
    seed=np.full(n,1/n);y=ranker.base_lu.solve(.5*seed)
    for always_fused in [False,True]:
        directory=RetainedStencilDirectory(ranker,stencil,0,plan+atom_upper+old_fused_upper)
        with directory.prepare(seed,sources,directions,masses,y,ranker.base_lu.max_returned_residual,
                               always_fused=always_fused) as family:
            assert family.route=='native'
            assert directory.atomic_solve_calls==0
            assert directory.memory['total_private_retained_basis_bytes']<=directory.full_budget
