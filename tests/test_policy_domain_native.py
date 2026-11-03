from fractions import Fraction

import numpy as np
import pytest
from scipy import sparse

from src.baselines.prpack import PRPackViewRank
from src.policy_domain_execution import DomainCompiler,DomainDirectory
from src.policy_domain_native import certify_rank1_native


def ranker_data(stable=True):
    if stable:
        P=np.array([[.5,0,0,0],[.25,1,0,0],[.25,0,1,0],[0,0,0,1.]])
        G=np.zeros_like(P);G[1,0]=.25
        b=500/797;seed=np.array([b,0,0,.594*b]);box=((1,2),(1,2))
    else:
        P=np.array([[.5,0,0],[.25,1,0],[.25,0,1.]])
        G=np.zeros_like(P);G[2,0]=.25
        seed=np.array([.3,.4,.3]);box=((1,4),(.25,1))
    ranker=PRPackViewRank(sparse.csc_matrix(P),alpha=.5)
    compiler=DomainCompiler(ranker,sparse.csc_matrix(G))
    directory=DomainDirectory(compiler,{},budget_bytes=16384,mode='dense_pruned')
    return ranker,compiler,directory,P,G,seed,box


def direct(P,G,seed,weights):
    weak=P[:,0]-G[:,0];weak=weak.copy();weak[0]=0
    raw=weights[0]*G[:,0]+weights[1]*weak;raw[0]+=P[0,0]
    view=P.copy();view[:,0]=raw/raw.sum()
    x=np.linalg.solve(np.eye(len(P))-.5*view,.5*seed)
    return x,np.lexsort((np.arange(len(P)),-x))


def clean(directory):
    assert not directory.live and not directory.pins
    assert directory._pending_bytes==directory._live_bytes==directory._live_reserved_bytes==0
    assert directory.nbytes==directory.reserved_bytes==directory.plan_bytes+directory._cache_bytes


def test_four_strict_corners_prove_real_rank_one_domain_without_base_or_atoms():
    ranker,c,d,P,G,seed,box=ranker_data()
    try:
        c.point_tolerance=7e-11
        result=certify_rank1_native(c,d,seed,[0],box,(1,))
        assert result['cuts'][1]['status']=='STABLE'
        assert result['cuts'][1]['selected_ids']==[0]
        assert result['point_checks']==4 and result['corners_visited']==4
        assert result['corner_summary_reservation_bytes']==4096
        assert c.point_tolerance==7e-11
        assert c.atom_solves==c.base_solves==ranker.base_lu.calls==0
        assert not d.cache and not ranker.base_lu.graph.previous_sources
        assert d.peak_reserved_bytes>=4096
        for m in np.linspace(1,2,7):
            for w in np.linspace(1,2,7):
                _,order=direct(P,G,seed,(m,w));assert order[0]==0
        clean(d)
    finally:ranker.base_lu.graph.close()


def test_actual_different_corners_return_verified_witness_and_stop_before_four():
    ranker,c,d,P,G,seed,box=ranker_data(False)
    try:
        result=certify_rank1_native(c,d,seed,[0],box,(1,))
        cut=result['cuts'][1]
        assert cut['status']=='UNSTABLE' and result['point_checks']<4
        witness=cut['witness']
        assert set(witness['first']['sets'][1])!=set(witness['second']['sets'][1])
        for point in witness.values():
            assert point['strict_flags'][1]
            _,order=direct(P,G,seed,point['weights'])
            assert set(point['sets'][1])==set(order[:1])
        clean(d)
    finally:ranker.base_lu.graph.close()


@pytest.mark.parametrize('epsilon',[0.,np.spacing(.5)])
def test_corner_ties_and_nonzero_near_ties_stay_unknown_after_all_precision_tiers(epsilon):
    ranker=PRPackViewRank(sparse.identity(3,format='csc'),alpha=.5)
    c=DomainCompiler(ranker,sparse.csc_matrix((3,3)));d=DomainDirectory(c,{})
    try:
        result=certify_rank1_native(c,d,np.array([.5+epsilon,.5-epsilon,0]),[0],((1,2),(.5,2)),(1,))
        assert result['cuts'][1]['status']=='UNKNOWN'
        assert result['point_checks']==16 and result['corners_visited']==4
        assert all(count==4 for count in result['native_tier_counts'].values())
        assert not ranker.base_lu.graph.previous_sources
        clean(d)
    finally:ranker.base_lu.graph.close()


class FakeCompiler:
    n=4
    generated_full_response_peak_bytes=0
    point_tolerance=7e-11
    def __init__(self,fail_on=None):
        self.calls=[];self.fail_on=fail_on;self.directory=None
    def point(self,seed,sources,weights,ks):
        self.calls.append((self.point_tolerance,tuple(ks)))
        assert self.directory.nbytes<=self.directory.reserved_bytes<=self.directory.budget_bytes
        assert self.directory._live_bytes>=4096 and self.directory._pending_bytes>=4096
        if len(self.calls)==self.fail_on:raise RuntimeError('Injected point failure')
        flags={k:self.point_tolerance<=(1e-8 if k==1 else 1e-10) for k in ks}
        return dict(sets={k:np.arange(k) for k in ks},selected_scores={k:np.arange(k,0,-1,dtype=float) for k in ks},
            strict_flags=flags,score_error_bound=Fraction(1,10**13),
            certified_gap_lower_bounds={k:Fraction(1,10) for k in ks})


def fake(budget=16384,fail_on=None):
    compiler=FakeCompiler(fail_on);directory=DomainDirectory(compiler,{},budget_bytes=budget)
    compiler.directory=directory
    return compiler,directory


def test_adaptive_precision_freezes_each_strict_cut_and_restores_tolerance():
    c,d=fake()
    result=certify_rank1_native(c,d,np.ones(4)/4,[0],((1,2),(.5,2)),(1,2))
    assert all(cut['status']=='STABLE' for cut in result['cuts'].values())
    assert result['point_checks']==12
    assert c.calls==[(1e-6,(1,2)),(1e-8,(1,2)),(1e-10,(2,))]*4
    assert c.point_tolerance==7e-11
    clean(d)


def test_exception_releases_summary_budget_and_restores_tolerance_after_real_eviction():
    c,d=fake(budget=8192,fail_on=2)
    class Cached:
        nbytes=3000
    d.cache[(0,0)]=Cached();d.cache[(0,1)]=Cached();d._cache_bytes=6000;d._track()
    with pytest.raises(RuntimeError,match='Injected'):
        certify_rank1_native(c,d,np.ones(4)/4,[0],((1,2),(.5,2)),(1,))
    assert c.point_tolerance==7e-11
    assert d.evictions==1 and list(d.cache)==[(0,1)] and d._cache_bytes==3000
    clean(d)


def test_insufficient_budget_returns_unknown_without_solves_or_unaccounted_state():
    c,d=fake(budget=2048)
    result=certify_rank1_native(c,d,np.ones(4)/4,[0],((1,2),(.5,2)),(1,2))
    assert result['point_checks']==0 and not c.calls
    assert all(v['status']=='UNKNOWN' for v in result['cuts'].values())
    assert result['native_budget_status']=='insufficient_corner_summary_budget'
    clean(d)


def test_live_family_is_not_overwritten_and_multisource_is_not_corner_certified():
    c,d=fake();d.live.append(object())
    with pytest.raises(RuntimeError,match='idle'):
        certify_rank1_native(c,d,np.ones(4)/4,[0],((1,2),(.5,2)),(1,))
    d.live.clear()
    with pytest.raises(ValueError,match='one valid source'):
        certify_rank1_native(c,d,np.ones(4)/4,[0,1],((1,2),(.5,2)),(1,))
    assert not c.calls
    clean(d)
