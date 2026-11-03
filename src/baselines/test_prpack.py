import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from src.baselines.prpack import PRPackGraph, PRPackViewRank, PRPackLinearSolver


@pytest.mark.parametrize('method',['sccgs','gs'])
def test_weighted_multicomponent_updates_and_restore(method):
    rng=np.random.default_rng(17)
    W=np.zeros((31,31))
    for source in range(31):
        # Reducible support, self edges and a dangling-repaired terminal node.
        group=source//7
        targets=np.arange(group*7,min((group+1)*7,31))
        W[targets,source]=rng.uniform(.1,1,len(targets))
        if group<4:W[min((group+1)*7,30),source]=.15
    P=sparse.csc_matrix(W/W.sum(axis=0));graph=PRPackGraph(P,method=method)
    seed=rng.uniform(0,1,31);seed/=seed.sum();alpha=.85
    graph.solve(seed,alpha)
    if method=='sccgs':assert graph.memory['scc_components']>1
    for sources in [[0,5,22],[7,19],[30],[]]:
        columns=[]
        for source in sources:
            p=P[:,source].toarray().ravel();q=p.copy()
            q[np.flatnonzero(p)[0]]*=3;q/=q.sum()
            columns.append(sparse.csc_matrix((q-p)[:,None]))
        D=sparse.hstack(columns,format='csc') if sources else sparse.csc_matrix((31,0))
        graph.update_columns(sources,D)
        Pv=P+D@sparse.csc_matrix((np.ones(len(sources)),(np.arange(len(sources)),sources)),shape=(len(sources),31))
        actual,_=graph.solve(seed,alpha)
        expected=spsolve(sparse.eye(31)-alpha*Pv,(1-alpha)*seed)
        assert np.max(np.abs(actual-expected))<1e-11
    graph.close()


def test_author_global_gs_absorbing_singleton_personalization():
    P=sparse.eye(7115,format='csc');seed=np.zeros(7115);seed[474]=1.
    graph=PRPackGraph(P,method='gs');scores,_=graph.solve(seed,.85,1e-12)
    assert np.max(np.abs(scores-seed))<1e-12
    graph.close()


def test_signed_batched_viewrank_backend():
    P=sparse.csc_matrix([[.3,.4,0],[.7,.2,.5],[0,.4,.5]])
    ranker=PRPackViewRank(P,alpha=.5)
    rhs=np.array([[.7,-.4],[0,.1],[-.3,.9]])
    actual=ranker.base_lu.solve(rhs)
    assert np.max(np.abs(actual-spsolve(ranker.A,rhs)))<1e-11
    assert np.array_equal(ranker.base_lu.last_residuals,np.sum(np.abs(rhs-ranker.A@actual),axis=0))
    assert ranker.base_lu.last_residual_l1==max(ranker.base_lu.last_residuals)
    saved_residuals=ranker.base_lu.last_residuals
    D=sparse.csc_matrix([[.1,0],[-.1,.1],[0,-.1]])
    prepared=ranker.prepare(D,[0,2]);seed=np.array([.2,.7,.1])
    scores=ranker.solve(seed,prepared)
    assert ranker.base_lu.last_residuals.shape==(1,)
    assert saved_residuals.shape==(2,)
    assert np.max(np.abs(scores-spsolve(sparse.eye(3)-.5*ranker.view_matrix(prepared),.5*seed)))<1e-11
    assert ranker.base_lu.max_returned_residual<1e-11


@pytest.mark.parametrize('alpha',[.5,.85])
@pytest.mark.parametrize('method',['sccgs','gs'])
def test_raw_signed_multiple_scc_and_legacy_split(alpha,method):
    rng=np.random.default_rng(41);n=37;W=np.zeros((n,n))
    for source in range(n-1):
        start=(source//6)*6;targets=np.arange(start,start+6)
        W[targets,source]=rng.uniform(.2,1,len(targets))
        if start<30:W[start+6,source]=.15
        else:W[36,source]=.15
    W[36,36]=1.
    P=sparse.csc_matrix(W/W.sum(axis=0));A=sparse.eye(n,format='csc')-alpha*P
    rhs=rng.normal(0,.02,(n,4));rhs[:,3]=0.
    direct=PRPackLinearSolver(P,alpha=alpha,method=method)
    legacy=PRPackLinearSolver(P,alpha=alpha,method=method,direct_signed=False)
    actual=direct.solve(rhs);split=legacy.solve(rhs);expected=spsolve(A,rhs)
    assert np.max(np.sum(np.abs(actual-expected),axis=0))<2e-12
    assert np.max(np.sum(np.abs(split-expected),axis=0))<2e-12
    assert direct.max_returned_residual<1e-12
    assert legacy.max_returned_residual<1e-12
    assert direct.native_calls==(3 if method=='sccgs' else 6)
    assert legacy.native_calls==6
    if method=='sccgs':assert direct.graph.memory['scc_components']==7
