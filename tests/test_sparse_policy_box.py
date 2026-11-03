"""Draft domain tests; do not run while a performance worker owns the CPU."""
from itertools import product

import numpy as np
import pytest
from scipy import sparse

from src.budgeted_family import BudgetedStencilDirectory, interaction_stencil
from src.compiled_policy_cache import CompiledDenseDirectory
from src.policy_box import PolicyBox
from src.policy_family_fast import FastPolicyFamily
from src.sparse_policy_box import (DenseAtomsPolicyBox, DenseEnvelopePolicyBox,
    NativePolicyCorners, SparsePolicyBox, box_graph_constants)
from src.viewrank import ViewRank


def fixture(keep=1):
    W = np.array([[2, 1, 0, 0, 0, 0, 1], [1, 2, 1, 0, 0, 0, 0],
        [1, 0, 2, 1, 0, 0, 0], [0, 1, 0, 2, 1, 0, 0],
        [0, 0, 1, 0, 2, 1, 0], [0, 0, 0, 1, 0, 2, 1],
        [1, 0, 0, 0, 1, 1, 2]], dtype=float)
    P = W/W.sum(axis=0)
    G = P.copy()
    G[::2] = 0.
    sources = np.array([4, 1])
    masses = np.array([G[:, s].sum() for s in sources])
    directions = [sparse.csc_matrix((G[:, s]-m*P[:, s]).reshape(-1, 1))
                  for s, m in zip(sources, masses)]
    seed = np.array([.5, 0, 0, .3, 0, .2, 0])
    ranker = ViewRank(sparse.csc_matrix(P), alpha=.72)
    directory = BudgetedStencilDirectory(ranker, interaction_stencil([sources]), keep, 100000)
    family = FastPolicyFamily(ranker, seed, sources, directions, masses,
        directory=directory, ks=(1, 3, 5))
    return family, P, G, sources, directions, masses, seed


def exact(P, G, sources, strengths, seed, alpha):
    Pv = P.copy()
    for s, strength in zip(sources, strengths):
        group = G[:, s]
        Pv[:, s] = (P[:, s]+(strength-1)*group)/(1+(strength-1)*group.sum())
    x = np.linalg.solve(np.eye(len(seed))-alpha*Pv, (1-alpha)*seed)
    return x, np.lexsort((np.arange(len(x)), -x))


@pytest.mark.parametrize('keep', [0, 1, 7])
def test_shared_small_enclosure_matches_existing_prototype(keep):
    family, *_ = fixture(keep)
    constants = box_graph_constants(family.ranker)
    bounds = {1: (1., 2.), 4: (.5, 4.)}
    old = PolicyBox(family).envelope(bounds)
    dense = DenseEnvelopePolicyBox(family, constants).envelope(bounds)
    small = SparsePolicyBox(family, constants).envelope(bounds)
    for result in (dense, small):
        assert result['valid'] == old['valid']
        np.testing.assert_array_equal(result['d0'], old['d0'])
        np.testing.assert_array_equal(result['radius'], old['radius'])
    np.testing.assert_array_equal(dense['scores'], old['scores'])
    np.testing.assert_array_equal(dense['error_bounds'], old['error_bounds'])
    np.testing.assert_array_equal(small['scores'], old['scores'][small['coordinate_nodes']])
    np.testing.assert_array_equal(small['error_bounds'], old['error_bounds'][small['coordinate_nodes']])
    outside = np.setdiff1d(np.arange(family.ranker.n), small['coordinate_nodes'])
    if len(outside):
        assert np.max((old['scores']+old['error_bounds'])[outside]) <= small['outside_upper']


@pytest.mark.parametrize('use_pair', [False, True])
def test_sparse_results_and_actual_witnesses_match_true_views(use_pair):
    family, P, G, _, _, _, seed = fixture(0)
    box = SparsePolicyBox(family)
    bounds = {1: (.5, 4.), 4: (.5, 4.)}
    result = box.query(bounds, ks=(1, 3, 5), use_pair=use_pair)
    for k, answer in result['results'].items():
        if answer['status'] == 'stable':
            for strengths in product((.5, 4.), repeat=2):
                _, order = exact(P, G, family.sources, strengths, seed, .72)
                assert set(order[:k]) == set(answer['indices'])
        elif answer['status'] == 'unstable':
            witness = answer['witness']
            _, order = exact(P, G, witness['sources'], witness['strengths'], seed, .72)
            assert set(order[:k]) == set(witness['indices'])
            assert set(witness['indices']) != set(answer['indices'])
            assert witness['margin'] > 0


def test_sparse_bounds_enclose_complete_scores_at_corners_and_interior():
    family, P, G, _, _, _, seed = fixture(0)
    box = SparsePolicyBox(family)
    envelope = box.envelope({1: (1., 2.), 4: (.5, 4.)})
    assert envelope['valid']
    points = list(product((1., 2.), (.5, 4.)))
    points.extend(np.random.default_rng(41).uniform([1., .5], [2., 4.], size=(30, 2)))
    outside = np.setdiff1d(np.arange(7), box.rows)
    for strengths in points:
        scores, _ = exact(P, G, family.sources, strengths, seed, .72)
        assert np.all(abs(scores[box.rows]-envelope['scores']) <= envelope['error_bounds']+1e-13)
        assert np.all(scores[outside] <= envelope['outside_upper']+1e-13)


def test_unvisited_node_can_enter_at_a_real_corner():
    P = np.eye(5)
    P[:, 2] = [.2, .2, .4, .1, .1]
    G = np.zeros_like(P)
    G[4, 2] = .1
    seed = np.array([.3, .1, .5, .08, .02])
    direction = sparse.csc_matrix((G[:, 2]-.1*P[:, 2]).reshape(-1, 1))
    ranker = ViewRank(sparse.csc_matrix(P), alpha=.8)
    directory = BudgetedStencilDirectory(ranker, interaction_stencil([[2]]), 0, 100000)
    family = FastPolicyFamily(ranker, seed, [2], [direction], [.1], directory=directory, ks=(1,))
    box = SparsePolicyBox(family)
    assert 4 not in box.rows
    answer = box.query({2: (.5, 40.)}, ks=(1,), use_pair=True)['results'][1]
    assert answer['status'] == 'unstable'
    witness = answer['witness']
    _, order = exact(P, G, witness['sources'], witness['strengths'], seed, .8)
    assert set(witness['indices']) == set(order[:1]) == {4}


def test_dense_atoms_receive_same_enclosure_and_corner_early_stop():
    family, P, G, sources, directions, masses, seed = fixture(1)
    directory = CompiledDenseDirectory(family.ranker, sparse.csc_matrix(G), 1000000)
    constants = box_graph_constants(family.ranker)
    with directory.prepare(seed, sources, directions, masses, base_scores=family.y,
                           base_residual=family.base_residual) as dense_family:
        dense = DenseAtomsPolicyBox(dense_family, constants)
        result = dense.query({1: (.5, 4.), 4: (.5, 4.)}, ks=(1, 3, 5), use_pair=True)
        baseline = dense.query({1: (.5, 4.), 4: (.5, 4.)}, ks=(1, 3, 5), use_envelope=False)
        for k in (1, 3, 5):
            answer, reference = result['results'][k], baseline['results'][k]
            if 'numerical_tie' not in (answer['status'], reference['status']):
                assert answer['status'] == reference['status']
                assert set(answer['indices']) == set(reference['indices'])
        envelope = dense.envelope({1: (1., 2.), 4: (.5, 4.)})
        assert envelope['valid']
        assert envelope['coordinate_count'] == len(P)
        assert dense.retained_rows is None  # dense IDs remain implicit
        for strengths in product((1., 2.), (.5, 4.)):
            scores, _ = exact(P, G, dense_family.sources, strengths, seed, .72)
            assert np.all(abs(scores-envelope['scores']) <= envelope['error_bounds']+1e-13)


def test_graph_constants_are_shared_instead_of_recomputed(monkeypatch):
    family, *_ = fixture()
    constants = box_graph_constants(family.ranker)
    def repeated_scan(_):
        raise AssertionError('Graph constants must be charged once in shared setup.')
    monkeypatch.setattr('src.sparse_policy_box.box_graph_constants', repeated_scan)
    sparse_box = SparsePolicyBox(family, constants)
    dense_box = DenseEnvelopePolicyBox(family, constants)
    assert sparse_box.graph_constants is dense_box.graph_constants is constants


def test_zero_direction_bounds_and_numerical_ties_are_retained():
    ranker = ViewRank(sparse.eye(3, format='csc'), alpha=.5)
    family = FastPolicyFamily(ranker, np.array([.5, .5, 0.]), [1],
        [sparse.csc_matrix((3, 1))], [1.], ks=(1, 2, 3))
    result = SparsePolicyBox(family).query({1: (.5, 4.)}, ks=(1, 2, 3))
    assert result['active_sources'].tolist() == []
    assert result['results'][1]['status'] == 'numerical_tie'
    assert result['results'][1]['witness'] is None
    assert result['results'][2]['status'] == result['results'][3]['status'] == 'stable'


def test_corner_control_skips_whole_domain_enclosure(monkeypatch):
    family, *_ = fixture()
    box = SparsePolicyBox(family)
    def unused(_):
        raise AssertionError('Corner baseline must not pay for a full-box envelope.')
    monkeypatch.setattr(box, 'envelope', unused)
    result = box.query({1: (.9, 1.1), 4: (.9, 1.1)}, ks=(1, 3, 5), use_envelope=False)
    assert result['possible_corners'] == 4


@pytest.mark.parametrize('dense_atoms', [False, True])
def test_point_completion_freezes_other_cut_with_its_original_proof(monkeypatch, dense_atoms):
    family, _, G, sources, directions, masses, seed = fixture()
    directory = None
    if dense_atoms:
        directory = CompiledDenseDirectory(family.ranker, sparse.csc_matrix(G), 1000000)
        family = directory.prepare(seed, sources, directions, masses,
            base_scores=family.y, base_residual=family.base_residual)
    box = DenseAtomsPolicyBox(family) if dense_atoms else DenseEnvelopePolicyBox(family)
    scores = np.array([10., 5., 5., 1., 0., 0., 0.])
    monkeypatch.setattr(box, '_envelope_a', lambda low, high:
        dict(scores=scores.copy(), error_bounds=np.zeros(7)))
    calls = []
    def coarse_completion(strengths, ks):
        calls.append(ks)
        # Deliberately different coarse order: its unproved top1 must not
        # inherit the positive top1 margin established by the envelope.
        return np.array([5., 12., 11., 1., 0., 0., 0.]), np.ones(7)*8., 1
    monkeypatch.setattr(box, '_completed_point', coarse_completion)
    point = box._point(np.ones(len(box.sources)), (1, 2))
    assert calls == [(2,)]
    assert point['sets'][1].tolist() == [0]
    assert point['margins'][1] == 5.
    assert set(point['sets'][2]) == {1, 2}
    if directory is not None:
        family.close()


def test_native_corner_margin_uses_matched_frozen_certificate_and_complete_bounds():
    from types import SimpleNamespace
    calls = []
    def solve(strengths, ks):
        calls.append(np.asarray(strengths).copy())
        return dict(sets={k: np.arange(k) for k in ks},
            certified={k: True for k in ks},
            certification_level={k: 0 for k in ks},
            certificate_error_bounds={k: .25 for k in ks},
            certificate_history=[
                dict(level=0, error_bound=.25, certified={k: True for k in ks},
                     gaps={k: 2. for k in ks}),
                dict(level=1, error_bound=.1, certified={k: False for k in ks},
                     gaps={k: 0. for k in ks})], native_solve_calls=1)
    zero = sparse.csc_matrix((10, 1))
    direction = sparse.csc_matrix(([.1, -.1], ([0, 1], [0, 0])), shape=(10, 1))
    family = SimpleNamespace(ranker=SimpleNamespace(n=10, alpha=.5),
        sources=np.array([5, 2, 9]), directions=(zero, direction, direction),
        masses=np.array([1., .5, .2]), solve=solve)
    box = NativePolicyCorners(family)
    point = box._point(np.array([1., 2.]), (1,))
    assert point['margins'][1] == 1.5  # not last level's negative margin
    assert point['sets'][1].tolist() == [0]
    result = box.query({5: (7., 9.), 2: (.5, 2.), 9: (.9, 1.1)}, ks=(1,))
    assert result['active_sources'].tolist() == [2, 9]
    assert result['possible_corners'] == result['corners_visited'] == 4
    assert result['results'][1]['status'] == 'stable'
    assert all(values[0] == 7. for values in calls[1:])
    assert set(tuple(values[1:]) for values in calls[2:]) == set(product((.5, 2.), (.9, 1.1)))
