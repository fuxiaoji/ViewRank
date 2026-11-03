import numpy as np
import pytest
from scipy import sparse
from src.viewrank import ViewRank
from src.budgeted_family import interaction_stencil
from src.compiled_policy_cache import CompiledStencilDirectory
from src.packed_policy_cache import PackedStencilDirectory


@pytest.mark.parametrize('exact_core',[True,False])
@pytest.mark.parametrize('stored',[True,False])
def test_packed_layout_preserves_numeric_results_and_exact_byte_difference(exact_core,stored):
    rng=np.random.default_rng(204)
    P=rng.random((29,29));P/=P.sum(axis=0)
    G=P.copy();G[::2]=0.;P=sparse.csc_matrix(P);G=sparse.csc_matrix(G)
    ranker=ViewRank(P,alpha=.5);seed=rng.random(29);seed/=seed.sum()
    y=ranker.base_lu.solve(.5*seed);S=np.array([3,7,2])
    masses=np.array([float(G[:,s].sum()) for s in S])
    directions=[G[:,s]-m*P[:,s] for s,m in zip(S,masses)]
    stencil=interaction_stencil([S,[2,7,10]])
    old=CompiledStencilDirectory(ranker,G,stencil,8,1000000,exact_core=exact_core,retain_deltas=stored)
    new=PackedStencilDirectory(ranker,G,stencil,8,1000000,exact_core=exact_core,retain_deltas=stored)
    with old.prepare(seed,S,directions,masses,base_scores=y,base_residual=0.,ks=(2,5,10)) as a:
        with new.prepare(seed,S,directions,masses,base_scores=y,base_residual=0.,ks=(2,5,10)) as b:
            for strength in [.5,1.,2.,4.]:
                x=a.solve(strength,materialize=True);z=b.solve(strength,materialize=True)
                np.testing.assert_array_equal(x['scores'],z['scores'])
                np.testing.assert_array_equal(x['error_bounds'],z['error_bounds'])
                assert x['initial_certified']==z['initial_certified']
                for k in (2,5,10):np.testing.assert_array_equal(x['sets'][k],z['sets'][k])
            rows=sum(len(atom.rows) for atom in a.atoms)
            plan=sum(len(r) for r in stencil.values()) if exact_core else 0
            assert old.nbytes-new.nbytes==4*(rows+plan)
            assert all(atom.rows.dtype==np.int32 for atom in b.atoms)
            assert new.peak_bytes==new.nbytes
            assert new.peak_live_response_bytes==sum(atom.response_bytes for atom in b.atoms)


def test_packed_eviction_pins_and_peak_survive_cache_hits():
    n=12
    P=np.eye(n)
    for source in range(n):
        P[(source+1)%n,source]=2
        P[(source+3)%n,source]=1
    P=sparse.csc_matrix(P/P.sum(axis=0))
    G=P.copy()
    G.data*=G.indices%2==1
    G.eliminate_zeros()
    ranker=ViewRank(P,alpha=.6)
    seed=np.arange(1.,n+1);seed/=seed.sum()
    y=ranker.base_lu.solve(.4*seed)
    directory=PackedStencilDirectory(ranker,G,interaction_stencil([[1],[2],[3]]),0,200)

    def prepare(source):
        mass=float(G[:,source].sum())
        delta=(G[:,source]-mass*P[:,source]).tocsc();delta.eliminate_zeros()
        return directory.prepare(seed,[source],[delta],[mass],y,0.,full_scan=True)

    with prepare(1) as first:
        atom=first.atoms[0]
        assert directory.peak_live_response_bytes==atom.response_bytes
        # The second live family cannot displace the first family's atom.
        with prepare(2) as bypass:
            assert bypass.route=='native'
            assert directory.cache[first.keys[0]] is atom
    high_water=directory.peak_bytes
    with prepare(1) as hit:
        assert hit.atoms[0] is atom
        assert directory.hits==1
        assert directory.peak_bytes==high_water
    with prepare(2) as second:
        assert second.route=='stencil'
        assert directory.evictions==1
        assert second.atoms[0].rows.dtype==np.int32
        assert directory.nbytes<=directory.budget_bytes
        assert directory.peak_bytes==max(high_water,directory.nbytes)
        assert directory.peak_live_response_bytes==second.atoms[0].response_bytes
    assert not directory._pins and directory._live_core_bytes==0
