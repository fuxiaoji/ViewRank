from itertools import product

import numpy as np
import pytest
from scipy import sparse

from src.baselines.prpack import PRPackViewRank
from src.budgeted_family import interaction_stencil
from src.multigroup_policy_family import (GROUPS, POLICY_GRID, MultiGroupDirectory,
    NativeGroupFamily, TwoGroupCompiler, _Family, _bytes)
from src.viewrank import ViewRank


def data(n=17, native=False, zero_groups=False):
    rng = np.random.default_rng(118)
    W = rng.random((n, n))+.1
    W[np.diag_indices(n)] += 1.
    P = W/W.sum(axis=0)
    G = P.copy()
    G[(np.indices((n, n)).sum(axis=0) % 3) != 0] = 0.
    G[np.diag_indices(n)] = 0.
    S = np.array([7, 2, 11, 4])
    if zero_groups:
        G[:, 11] = 0.
        G[:, 4] = P[:, 4]
        G[4, 4] = 0.
    ranker = (PRPackViewRank if native else ViewRank)(sparse.csc_matrix(P), alpha=.5)
    seed = rng.random(n)
    seed /= seed.sum()
    return ranker, P, G, S, seed


def truth(P, mention, S, seed, weights, alpha=.5):
    weak = P-mention-np.diag(P.diagonal())
    weights = np.broadcast_to(np.asarray(weights, dtype=float), (len(S), 2))
    Pv = P.copy()
    for i, source in enumerate(S):
        column = (np.eye(len(P))[:, source]*P[source, source]
            +weights[i, 0]*mention[:, source]+weights[i, 1]*weak[:, source])
        Pv[:, source] = column/column.sum()
    scores = np.linalg.solve(np.eye(len(P))-.5*Pv, .5*seed)
    order = np.lexsort((np.arange(len(P)), -scores))
    return scores, order, Pv


def counters_match(directory):
    cache = sum(atom.nbytes for atom in directory.cache.values())
    live = sum(f.metadata_bytes+sum(_bytes(v) for v in f.memo.values())
        +sum(8*len(k)+16 for k in f.seen) for f in directory.live)
    assert directory.nbytes == directory.plan_bytes+cache+live
    assert directory.reserved_bytes == directory.plan_bytes+cache+sum(f.reservation for f in directory.live)
    assert directory._cache_bytes == cache
    assert directory._live_bytes == live
    assert directory.nbytes <= directory.reserved_bytes <= directory.budget_bytes


@pytest.mark.parametrize('mode', ['sparse', 'dense'])
def test_r_source_core_combines_two_groups_and_matches_all_sixteen_views(mode):
    ranker, P, G, S, seed = data(zero_groups=True)
    compiler = TwoGroupCompiler(ranker, sparse.csc_matrix(G))
    plan = compiler.prepare(seed, S)
    assert plan.q == 6 and plan.r == 4
    assert not hasattr(plan, 'directions')
    directory = MultiGroupDirectory(compiler, interaction_stencil([S]), keep=len(P), mode=mode)
    with directory.prepare(plan) as family:
        assert family.core.shape == (4, 6)
        assert all(family.atoms[j] is directory.cache[key] for j, key in enumerate(plan.keys))
        for weights in POLICY_GRID:
            result = family.solve(weights, materialize=True)
            expected, order, Pv = truth(P, G, S, seed, weights)
            a, _ = plan.scales(dict(zip(GROUPS, weights)))
            D = plan.actual_delta(a)
            np.testing.assert_allclose((ranker.P+sparse.csc_matrix(
                (D.data, (D.indices, S[np.repeat(np.arange(len(S)), np.diff(D.indptr))])),
                shape=P.shape)).toarray(), Pv, atol=2e-15)
            np.testing.assert_allclose(result['scores'], expected, atol=2e-13)
            for k in (2, 5, 10):
                assert set(result['sets'][k]) == set(order[:k])
            assert result['acceptance'] == 'strict'
            counters_match(directory)
        assert family.logical_requests == 16 and len(family.seen) == 16
    assert not directory.pins and directory._live_bytes == 0
    counters_match(directory)


@pytest.mark.parametrize('mode,output', [('sparse', 'scan'), ('sparse', 'fused'), ('no_core', 'sparse')])
def test_complete_output_and_generic_residual_controls_preserve_targets(mode, output):
    ranker, P, G, S, seed = data()
    compiler = TwoGroupCompiler(ranker, sparse.csc_matrix(G))
    plan = compiler.prepare(seed, S)
    directory = MultiGroupDirectory(compiler, interaction_stencil([S]), keep=1, mode=mode)
    with directory.prepare(plan, output=output) as family:
        for weights in [(3., .5), (1., 1.), (1., .25), (4., 2.)]:
            result = family.solve(weights, materialize=True)
            expected, order, _ = truth(P, G, S, seed, weights)
            assert np.all(abs(result['scores']-expected) <= result['error_bounds']+2e-14)
            for k in (2, 5, 10):
                assert set(result['sets'][k]) == set(order[:k])
            if output == 'fused' and weights != (1., 1.):
                assert result['fallback_solve_calls'] == 1
            counters_match(directory)


def test_equal_mass_distinct_groups_do_not_collide_in_global_cache():
    P = np.eye(5)
    P[:, 0] = [.5, .25, .25, 0., 0.]
    G = np.zeros((5, 5))
    G[1, 0] = .25
    seed = np.array([1., 0, 0, 0, 0])
    ranker = ViewRank(sparse.csc_matrix(P), alpha=.5)
    compiler = TwoGroupCompiler(ranker, sparse.csc_matrix(G))
    plan = compiler.prepare(seed, [0])
    assert plan.masses[0, 0] == plan.masses[0, 1] == .25
    directory = MultiGroupDirectory(compiler, interaction_stencil([[0]]), keep=5)
    with directory.prepare(plan, ks=(1, 2, 3)) as family:
        assert set(directory.cache) == {(0, 0), (1, 0)}
        assert not np.array_equal(family.atoms[0].values, family.atoms[1].values)
        for weights in [(3., .5), (.5, 3.)]:
            result = family.solve(weights)
            _, order, _ = truth(P, G, [0], seed, weights)
            assert set(result['sets'][2]) == set(order[:2])


def test_zero_groups_keep_all_logical_targets_and_small_summary_reuse():
    n = 5
    ranker = ViewRank(sparse.eye(n, format='csc'), alpha=.5)
    compiler = TwoGroupCompiler(ranker, sparse.csc_matrix((n, n)))
    plan = compiler.prepare(np.array([.5, .25, .15, .07, .03]), [3, 1])
    directory = MultiGroupDirectory(compiler, interaction_stencil([[3, 1]]), keep=1)
    with directory.prepare(plan, ks=(1, 2, 3)) as family:
        for i, weights in enumerate(POLICY_GRID):
            result = family.solve(weights)
            assert result['effective_reused'] == (i > 0)
            np.testing.assert_array_equal(result['semantic_weights'], np.broadcast_to(weights, (2, 2)))
            counters_match(directory)
        assert family.logical_requests == 16
        assert family.memo_hits == 15 and len(family.seen) == 1
        assert family.statistics['logical_targets'] == 16
        assert family.statistics['actual_distinct_operator_views'] is None
        assert not family.atoms and not directory.cache
        for entry in family.memo.values():
            assert 'scores' not in entry and 'error_bounds' not in entry
            assert sum(v.nbytes for v in entry['selected_scores'].values()) == 8*6
    with NativeGroupFamily(plan, ks=(1, 2, 3), mode='osp_gap') as family:
        for weights in POLICY_GRID:
            family.solve(weights)
        assert family.logical_requests == 16 and family.memo_hits == 15


@pytest.mark.parametrize('mode', ['sparse', 'dense'])
def test_empty_original_source_set_keeps_ties_as_numerical_acceptance(mode):
    ranker = ViewRank(sparse.eye(5, format='csc'), alpha=.5)
    compiler = TwoGroupCompiler(ranker, sparse.csc_matrix((5, 5)))
    plan = compiler.prepare(np.full(5, .2), [])
    assert plan.r == plan.q == 0
    directory = MultiGroupDirectory(compiler, {}, mode=mode)
    with directory.prepare(plan, ks=(1, 2, 3)) as family:
        result = family.solve((3., .5), materialize=True)
        assert result['acceptance'] == 'numeric_tolerance'
        assert result['final_score_error_bound'] <= 5e-13
        assert not any(result['initial_certified'].values())
        assert not any(result['completion_certified'].values())
        for k in (1, 2, 3):
            np.testing.assert_array_equal(result['sets'][k], np.arange(k))
        counters_match(directory)


def test_nested_family_pins_and_accounting_survive_memo_overwrite_and_release():
    ranker, P, G, S, seed = data()
    compiler = TwoGroupCompiler(ranker, sparse.csc_matrix(G))
    plan = compiler.prepare(seed, S)
    directory = MultiGroupDirectory(compiler, interaction_stencil([S]), keep=3)
    with directory.prepare(plan) as first:
        first.solve((3., .5))
        first.solve((3., .5), materialize=True)
        counters_match(directory)
        original_atoms = first.atoms
        with directory.prepare(plan) as second:
            assert all(a is b for a, b in zip(original_atoms, second.atoms))
            assert all(directory.pins[k] == 2 for k in plan.keys)
            second.solve((4., .25))
            counters_match(directory)
        assert all(directory.pins[k] == 1 for k in plan.keys)
        counters_match(directory)
    assert directory._live_bytes == directory._live_reserved_bytes == 0
    counters_match(directory)


def test_global_lru_evicts_both_groups_without_splitting_the_budget():
    ranker, P, G, _, seed = data()
    compiler = TwoGroupCompiler(ranker, sparse.csc_matrix(G))
    pairs = [np.array([1, 0]), np.array([3, 2])]
    plans = [compiler.prepare(seed, pair) for pair in pairs]
    assert all(plan.q == 4 for plan in plans)
    stencil = interaction_stencil(pairs)
    plan_bytes = sum(4*len(v)+8 for v in stencil.values())
    reserve = max(_Family.reserve(plan, (2, 5, 10), 'sparse', 'sparse', 4*5) for plan in plans)
    directory = MultiGroupDirectory(compiler, stencil, keep=3, budget_bytes=plan_bytes+reserve+5*(72+12*5))
    for plan in (plans[0], plans[1], plans[0]):
        with directory.prepare(plan) as family:
            assert family.route == 'sparse'
            family.solve((3., .5))
            assert all(key in directory.cache for key in plan.keys)
            counters_match(directory)
        assert not directory.pins
        counters_match(directory)
    assert directory.evictions > 0
    assert set(group for group, source in directory.cache) == {0, 1}


def test_independent_source_group_weights_use_the_original_source_map():
    ranker, P, G, S, seed = data(zero_groups=True)
    compiler = TwoGroupCompiler(ranker, sparse.csc_matrix(G))
    plan = compiler.prepare(seed, S)
    directory = MultiGroupDirectory(compiler, interaction_stencil([S]), keep=len(P))
    weights = np.array([[3., .5], [1., 2.], [4., .25], [2., 1.]])
    with directory.prepare(plan) as family:
        result = family.solve(weights, materialize=True)
        expected, order, _ = truth(P, G, S, seed, weights)
        np.testing.assert_allclose(result['scores'], expected, atol=2e-13)
        for k in (2, 5, 10):
            assert set(result['sets'][k]) == set(order[:k])


def test_dense_scalar_bound_matches_full_coordinate_certificate_without_extra_workspace():
    ranker, P, G, S, seed = data()
    compiler = TwoGroupCompiler(ranker, sparse.csc_matrix(G))
    plan = compiler.prepare(seed, S)
    directory = MultiGroupDirectory(compiler, {}, mode='dense')
    with directory.prepare(plan) as family:
        for weights in POLICY_GRID:
            a, _ = plan.scales(weights)
            compact, c, d = family._initial(a, False)
            assert 'scores' not in compact and 'error_bounds' not in compact
            materialized, _, _ = family._initial(a, True)
            scores = materialized['scores']
            error = compact['score_error_bound']
            order = np.lexsort((np.arange(len(P)), -scores))
            for k in (2, 5, 10):
                upper = scores+error
                upper[order[:k]] = -np.inf
                generic_flag = np.min(scores[order[:k]]-error) > np.max(upper)
                assert compact['initial_certified'][k] == generic_flag
                assert set(compact['sets'][k]) == set(order[:k])
            np.testing.assert_array_equal(materialized['error_bounds'], np.full(len(P), error))


def test_late_refinement_keeps_set_proof_and_uses_current_bounded_scores(monkeypatch):
    ranker, P, G, S, seed = data()
    compiler = TwoGroupCompiler(ranker, sparse.csc_matrix(G))
    plan = compiler.prepare(seed, S)
    directory = MultiGroupDirectory(compiler, interaction_stencil([S]), keep=3)
    current = np.zeros(len(P))
    current[:2] = [.49, .51]
    with directory.prepare(plan, ks=(1, 2)) as family:
        def initial(a, materialize):
            result = dict(sets={1: np.array([0]), 2: np.array([0, 1])},
                selected_scores={1: np.array([.51]), 2: np.array([.51, .49])},
                initial_certified={1: True, 2: False}, score_error_bound=.005)
            return result, np.zeros(plan.r), np.ones(plan.q)
        def completion(*args, **kwargs):
            return dict(scores=current.copy(),
                sets={1: np.array([1]), 2: np.array([1, 0])},
                selected_scores={1: np.array([.51]), 2: np.array([.51, .49])},
                completion_certified={1: False, 2: True}, numerical_allowance=.03,
                fallback_solve_calls=1)
        monkeypatch.setattr(family, '_initial', initial)
        monkeypatch.setattr('src.multigroup_policy_family.complete_family', completion)
        result = family.solve((3., .5), materialize=True)
        assert result['sets'][1].tolist() == [0]  # membership proof remains frozen
        assert result['sets'][2].tolist() == [1, 0]  # current within-set score order
        assert result['selected_scores'][1].tolist() == [.49]
        for k in (1, 2):
            np.testing.assert_array_equal(result['selected_scores'][k], result['scores'][result['sets'][k]])
        assert result['final_score_error_bound'] == .03
        assert result['initial_certified'][1]
        counters_match(directory)


def test_budget_overflow_prepares_no_dense_atoms_and_finishes_actual_complete_view():
    ranker, P, G, S, seed = data(n=96, native=True)
    compiler = TwoGroupCompiler(ranker, sparse.csc_matrix(G))
    plan = compiler.prepare(seed, S)
    native_reserve = plan.nbytes+16*(16*17+48*3+512+8*plan.q)
    budget = native_reserve+1024
    directory = MultiGroupDirectory(compiler, {}, mode='dense', budget_bytes=budget)
    try:
        with directory.prepare(plan) as family:
            assert family.route == 'native'
            assert not directory.cache and not family.atoms
            result = family.solve((3., .5))
            _, order, _ = truth(P, G, S, seed, (3., .5))
            for k in (2, 5, 10):
                assert set(result['sets'][k]) == set(order[:k])
            assert result['acceptance'] in ('strict', 'numeric_tolerance')
            counters_match(directory)
    finally:
        ranker.base_lu.graph.close()


@pytest.mark.parametrize('mode', ['native_gap', 'osp_gap'])
def test_native_control_uses_same_normalized_compiler_and_effective_reuse(mode):
    ranker, P, G, S, seed = data(native=True, zero_groups=True)
    compiler = TwoGroupCompiler(ranker, sparse.csc_matrix(G))
    plan = compiler.prepare(seed, S)
    try:
        with NativeGroupFamily(plan, mode=mode) as family:
            for weights in [(3., .5), (1., 1.), (3., .5)]:
                result = family.solve(weights)
                _, order, _ = truth(P, G, S, seed, weights)
                for k in (2, 5, 10):
                    assert set(result['sets'][k]) == set(order[:k])
            assert family.memo_hits == 1 and len(family.seen) == 2
    finally:
        ranker.base_lu.graph.close()
