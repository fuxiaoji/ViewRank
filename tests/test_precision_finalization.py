import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_compiled_precision import fixed_point_finish,finished


def test_final_refinement_preserves_a_real_machine_boundary():
    ranker=SimpleNamespace(P=sparse.eye(2,format='csc'),alpha=.5,n=2)
    seed=np.array([.5,.5]);zero=sparse.csc_matrix((2,1))
    result=fixed_point_finish(ranker,seed,np.array([0]),[zero],[0.],2.,(1,),np.array([.6,.4]))
    assert not result['completion_certified'][1]
    assert finished(result,(1,))
    assert max(abs(result['scores']-seed))<=result['score_error_bound']
    assert max(abs(result['scores']-seed))<=5e-13


def test_final_refinement_bound_against_independent_dense_systems():
    for seed_id in range(5):
        rng=np.random.default_rng(seed_id);P=rng.random((9,9));P/=P.sum(axis=0)
        group=P[:,[1]].copy();group[4:]=0.;mass=float(group.sum())
        direction=sparse.csc_matrix(group-mass*P[:,[1]])
        strength=4.;scale=(strength-1)/(1+(strength-1)*mass)
        Pv=P.copy();Pv[:,1]+=scale*direction.toarray().ravel()
        seed=rng.random(9);seed/=seed.sum()
        exact=np.linalg.solve(np.eye(9)-.5*Pv,.5*seed)
        ranker=SimpleNamespace(P=sparse.csc_matrix(P),alpha=.5,n=9)
        result=fixed_point_finish(ranker,seed,np.array([1]),[direction],[mass],strength,(1,3),seed)
        assert finished(result,(1,3))
        assert max(abs(result['scores']-exact))<=result['score_error_bound']+1e-15
        if all(result['completion_certified'].values()):
            order=np.argsort(-exact)
            for k in (1,3):assert set(result['sets'][k])==set(order[:k])
