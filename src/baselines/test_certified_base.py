import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from src.baselines.certified_base import certified_base
from src.baselines.prpack import PRPackViewRank


def test_actual_residual_lu_bound_earlystop_and_seed_mass():
    rng=np.random.default_rng(29);n=17;W=rng.uniform(.2,1,(n,n))
    P=sparse.csc_matrix(W/W.sum(0));ranker=PRPackViewRank(P,alpha=.5)
    seed=np.arange(n,0,-1,dtype=float)**2;seed/=seed.sum();seed*=3
    original=seed.copy();saved_tol=ranker.base_lu.tol
    y,residual,flags,seconds,levels=certified_base(ranker,seed,(2,5,10))
    expected=spsolve(ranker.A,.5*seed)
    assert np.max(abs(y-expected))<=levels[-1]['score_error_bound']+1e-14
    assert all(flags.values()) and levels[-1]['tolerance']>1e-12
    assert ranker.base_lu.tol==saved_tol and np.array_equal(seed,original)
    assert abs(y.sum()-3)<=residual/.5+1e-13
    assert residual==ranker.base_lu.last_residuals[0] and seconds>0
    ranker.base_lu.graph.close()


def test_ties_reach_last_level_and_restore_tolerance():
    ranker=PRPackViewRank(sparse.eye(12,format='csc'),alpha=.5)
    seed=np.full(12,1/12)
    _,_,flags,_,levels=certified_base(ranker,seed,(2,5,10))
    assert not any(flags.values()) and len(levels)==4
    assert levels[-1]['numerical_boundary'] and ranker.base_lu.tol==1e-12
    ranker.base_lu.graph.close()


def test_all_cuts_required_and_already_measured_residual_used():
    class Backend:
        tol=.12345
        def solve(self,rhs):
            y=2*rhs;y[9]+=self.tol;y[10]-=self.tol
            self.last_residuals=np.array([np.sum(abs(rhs-.5*y))]);return y
    class Ranker:alpha=.5;base_lu=Backend()
    ranker=Ranker()
    # There is deliberately no A attribute: helper must consume already
    # measured residual metadata, not perform a duplicate residual product.
    seed=np.r_[np.linspace(.2,.1,9),.010000005,.01,.005];seed/=seed.sum()
    _,_,flags,_,levels=certified_base(ranker,seed,(2,5,10))
    assert levels[0]['strict_flags'][2] and not levels[0]['strict_flags'][10]
    assert levels[-1]['tolerance']==1e-10 and all(flags.values())
    assert ranker.base_lu.tol==.12345


def test_failed_solve_restores_original_tolerance():
    class Backend:
        tol=.123
        def solve(self,rhs):raise RuntimeError('test interruption')
    class Ranker:alpha=.5;base_lu=Backend()
    ranker=Ranker()
    with pytest.raises(RuntimeError):certified_base(ranker,np.ones(12)/12,(2,5,10))
    assert ranker.base_lu.tol==.123
