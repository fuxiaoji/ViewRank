import numpy as np
import pytest
from scipy import sparse

from src.budgeted_family import interaction_stencil
from src.compiled_policy_cache import (CompiledDenseDirectory, CompiledStencilDirectory,
                                       DESCRIPTOR_BYTES, StoredDirection)
from src.viewrank import ViewRank, _sparse_bytes


class MeasuredLU:
    def __init__(self, solver, A, perturb=0.):
        self.solver, self.A, self.perturb = solver, A, perturb
        self.calls = 0
        self.last_residuals = np.empty(0)

    def solve(self, rhs):
        value = self.solver.solve(rhs)
        value = value + self.perturb * np.linspace(-1, 1, len(value))
        matrix = value.reshape(-1, 1) if value.ndim == 1 else value
        rhs_matrix = np.asarray(rhs).reshape(matrix.shape)
        self.last_residuals = np.sum(abs(rhs_matrix-self.A@matrix), axis=0)
        self.calls += 1
        return value


def problem(perturb=0.):
    n = 12
    W = np.eye(n)
    for source in range(n):
        W[(source+1)%n, source] = 2
        W[(source+3)%n, source] = 1
    P = sparse.csc_matrix(W/W.sum(axis=0))
    G = P.copy()
    for source in range(n):
        lo, hi = G.indptr[source:source+2]
        G.data[lo:hi] *= G.indices[lo:hi] % 2 == 1
    G.eliminate_zeros()
    ranker = ViewRank(P, alpha=.6)
    ranker.base_lu = MeasuredLU(ranker.base_lu, ranker.A, perturb)
    seed = np.arange(1., n+1)
    seed /= seed.sum()
    y = ranker.base_lu.solve(.4*seed)
    ry = float(ranker.base_lu.last_residuals[0])
    sources = [5, 1, 8]
    masses = np.asarray(G.sum(axis=0)).ravel()[sources]
    directions = [(G[:,s]-m*P[:,s]).tocsc() for s,m in zip(sources,masses)]
    for d in directions:
        d.eliminate_zeros()
    return ranker, G, seed, y, ry, sources, directions, masses


def exact(ranker, G, seed, sources, strengths):
    strengths = np.broadcast_to(np.asarray(strengths), (len(sources),))
    P = ranker.P.toarray()
    for source, strength in zip(sources, strengths):
        group = G[:, source].toarray().ravel()
        P[:,source] = (P[:,source]+(strength-1)*group)/(1+(strength-1)*group.sum())
    return np.linalg.solve(np.eye(ranker.n)-ranker.alpha*P, (1-ranker.alpha)*seed)


@pytest.mark.parametrize('kind', ['dense', 'stencil', 'scan'])
def test_implicit_direction_completion_and_full_sources(kind):
    r,G,seed,y,ry,sources,directions,masses = problem()
    directory = (CompiledDenseDirectory(r,G,10000) if kind=='dense' else
                 CompiledStencilDirectory(r,G,interaction_stencil([sources]),1,10000))
    assert np.shares_memory(directory.G.data,G.data)
    assert np.shares_memory(directory.P.data,r.P.data)
    with directory.prepare(seed,sources,directions,masses,base_scores=y,base_residual=ry,
                           ks=(2,5,10),full_scan=kind=='scan') as family:
        assert set(family.original_sources)==set(sources)
        for atom, source in zip(family.atoms, sorted(sources)):
            original = directions[sources.index(source)]
            assert np.max(abs((atom.delta-original).toarray())) < 1e-15
            assert atom.direction.directory is directory
            assert not any(sparse.issparse(value) for value in vars(atom).values())
            assert atom.direction.nbytes==DESCRIPTOR_BYTES==24
            assert atom.residual_l1>=0 and atom.update_l1>0
            assert atom.nbytes==atom.response_bytes+24+16+(8 if kind!='dense' else 0)
        assert directory.memory['retained_sparse_delta_bytes']==0
        for strength in [.5,1.,2.,4.]:
            answer=family.solve(strength,ks=(2,5,10),materialize=True)
            target=exact(r,G,seed,sources,strength)
            assert np.all(abs(answer['scores']-target)<=answer['error_bounds']+2e-13)
            for k in (2,5,10):
                expected=np.lexsort((np.arange(r.n),-target))[:k]
                assert set(answer['sets'][k])==set(expected)
        answer=family.solve(4.,ks=(2,5,10),always_fused=True,materialize=True)
        assert np.max(abs(answer['scores']-exact(r,G,seed,sources,4.)))<1e-12
        assert directory.memory['total_private_retained_basis_bytes']<=10000
        if kind=='dense':
            assert not hasattr(family,'responses')
            assert family.atoms[0].response is directory.cache[family.keys[0]].response
        before=directory.nbytes
        d=np.array([.2,-.1,.3])
        rhs=directory.merge_rhs(family.atoms,d)
        original=sparse.hstack([directions[sources.index(s)] for s in sorted(sources)])
        assert np.max(abs(rhs+r.alpha*(original@d)))<1e-15
        assert directory.nbytes==before
    assert directory._live_core_bytes==0 and not directory._pins


def test_dense_pins_hits_evictions_and_no_full_family_response_copy():
    r,G,seed,y,ry,sources,directions,masses=problem()
    directory=CompiledDenseDirectory(r,G,390)
    # All columns use the same fixed group mass only where this is true.
    def args(ss):
        mm=np.asarray(G.sum(axis=0)).ravel()[ss]
        dd=[(G[:,s]-m*r.P[:,s]).tocsc() for s,m in zip(ss,mm)]
        for d in dd:d.eliminate_zeros()
        return dd,mm
    ds,ms=args([1])
    with directory.prepare(seed,[1],ds,ms,y,ry) as first:
        pinned=first.atoms[0]
        ds,ms=args([1,2])
        calls=r.base_lu.calls
        with directory.prepare(seed,[1,2],ds,ms,y,ry) as second:
            assert second.route=='native'
            assert r.base_lu.calls==calls
            assert directory.cache[first.keys[0]] is pinned
    ds,ms=args([1,2])
    with directory.prepare(seed,[1,2],ds,ms,y,ry) as second:
        assert second.route=='dense'
        assert second.atoms[0] is pinned
        assert directory.nbytes<=390
    ds,ms=args([2,3])
    with directory.prepare(seed,[2,3],ds,ms,y,ry) as third:
        assert third.route=='dense'
        assert directory.evictions>=1
        assert directory.nbytes<=390
    assert directory.peak_bytes<=390


@pytest.mark.parametrize('kind', ['dense','stencil'])
def test_measured_metadata_avoids_prepare_and_target_spmv_and_bounds_coarse_scores(kind):
    r,G,seed,y,ry,sources,directions,masses=problem(perturb=1e-6)
    actual_A=r.A
    class NoSpMV:
        def __matmul__(self, other):
            raise AssertionError('Captured residual metadata must avoid another SpMV.')
    r.A=NoSpMV()
    directory=(CompiledDenseDirectory(r,G,10000) if kind=='dense' else
               CompiledStencilDirectory(r,G,interaction_stencil([sources]),12,10000))
    with directory.prepare(seed,sources,directions,masses,y,ry,ks=(2,5,10)) as family:
        for atom in family.atoms:
            assert atom.residual_l1>0
        answer=family.solve(2.,ks=(2,5,10),materialize=True)
        target=exact(r,G,seed,sources,2.)
        assert np.all(abs(answer['scores']-target)<=answer['error_bounds']+1e-13)
        assert family.base_residual==ry
        if kind=='dense':
            assert np.array_equal(answer['scaled_coefficients'],answer['scales']*answer['coefficients'])
            assert np.array_equal(answer['small_system_residual'],y[family.sources]-answer['K']@answer['coefficients'])
            assert np.max(abs(answer['a']-answer['scales']))==0
    r.A=actual_A


@pytest.mark.parametrize('kind', ['dense','stencil'])
def test_zero_groups_need_no_cached_direction_or_response(kind):
    P=sparse.eye(6,format='csc');G=sparse.csc_matrix((6,6))
    r=ViewRank(P,alpha=.5);seed=np.array([.4,.25,.15,.1,.06,.04]);y=seed.copy()
    sources=[4,1];directions=[sparse.csc_matrix((6,1))]*2;masses=[0.,0.]
    directory=(CompiledDenseDirectory(r,G,1000) if kind=='dense' else
               CompiledStencilDirectory(r,G,interaction_stencil([sources]),0,1000))
    with directory.prepare(seed,sources,directions,masses,y,0.,ks=(1,2)) as family:
        result=family.solve(4.,ks=(1,2),materialize=True)
        assert np.max(abs(result['scores']-seed))<1e-14
        assert set(family.original_sources)==set(sources)
        assert directory.atomic_solve_calls==0
        assert directory.memory['implicit_descriptor_bytes']==0
        assert directory.memory['total_private_retained_basis_bytes']<=1000


def test_dense_vector_strengths_keep_original_source_order():
    r,G,seed,y,ry,sources,directions,masses=problem()
    directory=CompiledDenseDirectory(r,G,10000)
    strengths=np.array([.5,1.2,4.])
    with directory.prepare(seed,sources,directions,masses,y,ry) as family:
        result=family.solve(strengths,ks=(2,5),materialize=True)
        assert np.max(abs(result['scores']-exact(r,G,seed,sources,strengths)))<1e-12


@pytest.mark.parametrize('kind', ['dense','stencil'])
def test_oversized_basis_routes_original_sources_and_vector_strengths_to_point_solver(kind):
    r,G,seed,y,ry,sources,directions,masses=problem()
    class Point:
        def __init__(self):self.current=r.P;self.calls=0;self.updates=[]
        def update_columns(self,ss,D):
            self.updates.append(list(ss))
            E=sparse.csc_matrix((np.ones(len(ss)),(np.arange(len(ss)),ss)),shape=(len(ss),r.n))
            self.current=r.P+D@E
        def solve_linear(self,rhs,alpha,tol):
            self.calls+=1
            return np.linalg.solve(np.eye(r.n)-alpha*self.current.toarray(),rhs),{}
    point=Point()
    stencil=interaction_stencil([sources])
    plan=sum(rows.nbytes for rows in stencil.values())+8*len(stencil)
    directory=(CompiledDenseDirectory(r,G,40,point_graph=point) if kind=='dense' else
               CompiledStencilDirectory(r,G,stencil,1,plan+40,point_graph=point))
    strengths=np.array([.5,1.2,4.])
    with directory.prepare(seed,sources,directions,masses,y,ry) as family:
        assert family.route=='native'
        assert directory.atomic_solve_calls==0
        result=family.solve(strengths,ks=(2,5))
        assert np.max(abs(result['scores']-exact(r,G,seed,sources,strengths)))<1e-12
        assert point.updates==[sources,[]]
        assert point.calls==1
        assert (point.current-r.P).nnz==0
        assert directory.nbytes<=directory.budget_bytes


@pytest.mark.parametrize('kind', ['dense','stencil'])
def test_stored_delta_ablation_keeps_keys_coefficients_and_numerics_identical(kind):
    r,G,seed,y,ry,sources,directions,masses=problem(perturb=1e-6)
    def directory(stored):
        return (CompiledDenseDirectory(r,G,10000,retain_deltas=stored) if kind=='dense' else
                CompiledStencilDirectory(r,G,interaction_stencil([sources]),1,10000,retain_deltas=stored))
    implicit,stored=directory(False),directory(True)
    with implicit.prepare(seed,sources,directions,masses,y,ry) as left:
        with stored.prepare(seed,sources,directions,masses,y,ry) as right:
            assert left.keys==right.keys
            delta_bytes=sum(_sparse_bytes(d) for d in directions)
            assert stored.nbytes-implicit.nbytes==delta_bytes
            assert stored.memory['retained_sparse_delta_bytes']==delta_bytes
            assert stored.memory['direction_descriptor_bytes']==implicit.memory['direction_descriptor_bytes']
            for atom,source in zip(right.atoms,sorted(sources)):
                request=directions[sources.index(source)]
                assert isinstance(atom.direction,StoredDirection)
                assert atom.delta is atom.direction.delta
                assert np.shares_memory(atom.delta.data,request.data)
                assert atom.direction.nbytes==24+_sparse_bytes(request)
            for strength in [.5,1.,4.]:
                a=left.solve(strength,ks=(2,5,10),materialize=True)
                b=right.solve(strength,ks=(2,5,10),materialize=True)
                assert np.array_equal(a['scores'],b['scores'])
                assert np.array_equal(a['error_bounds'],b['error_bounds'])
                assert a['initial_certified']==b['initial_certified']
                if kind=='dense':
                    for key in ['a','c','d','K','small_system_residual']:
                        assert np.array_equal(a[key],b[key])
            a=left.solve(4.,ks=(2,5),always_fused=True,materialize=True)
            b=right.solve(4.,ks=(2,5),always_fused=True,materialize=True)
            assert np.array_equal(a['scores'],b['scores'])
            assert stored.direction_materializations==0
    before=stored.atomic_solve_calls
    with stored.prepare(seed,sources,directions,masses,y,ry) as family:
        assert family.keys==tuple(implicit.cache)
        assert stored.atomic_solve_calls==before
        assert stored.hits==len(family.atoms)


@pytest.mark.parametrize('kind', ['dense','stencil'])
def test_stored_delta_bytes_alone_can_change_budget_admission(kind):
    r,G,seed,y,ry,sources,directions,masses=problem()
    source=[sources[0]];direction=[directions[0]];mass=[masses[0]]
    # Exact one-source retained payload; no output-candidate index for stencil.
    budget=176 if kind=='dense' else 136
    def directory(stored):
        return (CompiledDenseDirectory(r,G,budget,retain_deltas=stored) if kind=='dense' else
                CompiledStencilDirectory(r,G,interaction_stencil([source]),0,budget,retain_deltas=stored))
    implicit,stored=directory(False),directory(True)
    with implicit.prepare(seed,source,direction,mass,y,ry,full_scan=kind=='stencil') as family:
        assert family.route==('dense' if kind=='dense' else 'stencil')
        assert implicit.nbytes==budget
        key=family.keys[0]
    with stored.prepare(seed,source,direction,mass,y,ry,full_scan=kind=='stencil') as family:
        assert family.route=='native'
        assert stored.atomic_solve_calls==0
        assert stored.nbytes<=budget
    assert key==(source[0],float(mass[0]))


@pytest.mark.parametrize('keep', [0,1,12])
def test_no_core_sparse_model_residual_and_every_reported_certificate_are_valid(keep):
    r,G,seed,y,ry,sources,directions,masses=problem(perturb=1e-6)
    actual_A=r.A
    class SparseProducts:
        def __init__(self):self.calls=0
        def __matmul__(self, model):
            assert sparse.issparse(model), 'NoCore charges one sparse response-model product per atom.'
            self.calls+=1
            return actual_A@model
    products=SparseProducts();r.A=products
    directory=CompiledStencilDirectory(r,G,interaction_stencil([sources]),keep,10000,exact_core=False)
    assert directory.stencil=={} and directory.plan_bytes==0
    with directory.prepare(seed,sources,directions,masses,y,ry,ks=(2,5,10)) as family:
        assert family.core_representation=='truncated_numeric_model'
        assert set(family.original_sources)==set(sources)
        assert products.calls==len(family.atoms)
        for atom,source in zip(family.atoms,sorted(sources)):
            assert len(atom.rows)<=keep and atom.tail_inf==0
            model=sparse.csc_matrix((atom.values,(atom.rows,np.zeros(len(atom.rows),dtype=int))),shape=(r.n,1))
            expected=-r.alpha*directions[sources.index(source)]-actual_A@model
            assert atom.residual_l1==float(np.abs(expected.data).sum())
        if keep==0:
            assert np.array_equal(family.core,np.zeros((len(sources),len(sources))))
            _,c,_,K=family.coefficients(4.)
            assert np.array_equal(K,np.eye(len(sources)))
            assert np.array_equal(c,y[family.sources])
        for strength in [.5,1.,2.,4.]:
            answer=family.solve(strength,ks=(2,5,10),materialize=True)
            target=exact(r,G,seed,sources,strength)
            assert np.all(abs(answer['scores']-target)<=answer['error_bounds']+1e-12)
            expected_order=np.lexsort((np.arange(r.n),-target))
            flags=(answer['completion_certified'] if answer['fallback_solve_calls'] else answer['initial_certified'])
            for k,certified in flags.items():
                if certified:
                    assert set(answer['sets'][k])==set(expected_order[:k])
        assert products.calls==len(family.atoms), 'No per-target response-model SpMV is needed.'
    r.A=actual_A


def test_no_core_zero_missing_source_coupling_can_still_legally_certify():
    P=sparse.csc_matrix(np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],
                                [0.,0.,.6,0.],[0.,0.,.4,1.]]))
    G=sparse.csc_matrix(([.4],([3],[2])),shape=(4,4))
    r=ViewRank(P,alpha=.7);seed=np.array([.7,.3,0.,0.]);y=seed.copy()
    direction=(G[:,2]-.4*P[:,2]).tocsc()
    cache=CompiledStencilDirectory(r,G,interaction_stencil([[2]]),0,1000,exact_core=False)
    with cache.prepare(seed,[2],[direction],[.4],y,0.,ks=(1,2)) as family:
        answer=family.solve(4.,ks=(1,2),materialize=True)
        assert family.core[0,0]==0
        assert answer['initial_certified']=={1:True,2:True}
        assert answer['fallback_solve_calls']==0
        assert set(answer['sets'][1])=={0}
        assert set(answer['sets'][2])=={0,1}

