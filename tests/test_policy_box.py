from itertools import product

import numpy as np
import pytest
from scipy import sparse

from src.budgeted_family import BudgetedStencilDirectory, interaction_stencil
from src.iterative_backend import MatrixFreeViewRank
from src.policy_box import PolicyBox
from src.policy_family_fast import FastPolicyFamily
from src.viewrank import ViewRank


def make_family(keep=2, approximate=False):
    W = np.array([[2, 1, 0, 0, 0, 0, 1], [1, 2, 1, 0, 0, 0, 0],
                  [1, 0, 2, 1, 0, 0, 0], [0, 1, 0, 2, 1, 0, 0],
                  [0, 0, 1, 0, 2, 1, 0], [0, 0, 0, 1, 0, 2, 1],
                  [1, 0, 0, 0, 1, 1, 2]], dtype=float)
    P = W / W.sum(axis=0)
    sources = [4, 1]
    groups, directions, masses = {}, [], []
    for s in sources:
        group = P[:, s].copy()
        group[np.arange(7) % 2 == 0] = 0
        groups[s] = group
        masses.append(group.sum())
        directions.append(sparse.csc_matrix((group - group.sum() * P[:, s]).reshape(-1, 1)))
    seed = np.array([.5, 0, 0, .3, 0, .2, 0])
    ranker = (MatrixFreeViewRank(sparse.csc_matrix(P), alpha=.72, tol=1e-7)
              if approximate else ViewRank(sparse.csc_matrix(P), alpha=.72))
    directory = BudgetedStencilDirectory(ranker, interaction_stencil([sources]), keep, 100000)
    family = FastPolicyFamily(ranker, seed, sources, directions, masses,
                              directory=directory, ks=(1, 3, 5))
    return family, P, seed, groups


def reference(P, seed, alpha, groups, sources, strengths):
    Pv = P.copy()
    for source, strength in zip(sources, strengths):
        group = groups[int(source)]
        Pv[:, source] = (P[:, source] + (strength - 1) * group) / (1 + (strength - 1) * group.sum())
    scores = np.linalg.solve(np.eye(len(seed)) - alpha * Pv, (1 - alpha) * seed)
    order = np.lexsort((np.arange(len(seed)), -scores))
    return scores, order


@pytest.mark.parametrize('keep', [0, 2, 7])
@pytest.mark.parametrize('approximate', [False, True])
def test_posterior_d_and_output_enclose_corners_and_interior(keep, approximate):
    family, P, seed, groups = make_family(keep, approximate)
    box = PolicyBox(family)
    bounds = {1: (1., 2.), 4: (.5, 4.)}
    envelope = box.envelope(bounds)
    assert envelope['valid']
    rng = np.random.default_rng(6)
    points = list(product((1., 2.), (.5, 4.)))
    points.extend(rng.uniform([1., .5], [2., 4.], size=(40, 2)))
    for strengths in points:
        exact, _ = reference(P, seed, family.ranker.alpha, groups, family.sources, strengths)
        a = family.scales(strengths)
        true_d = a * exact[family.sources]
        assert np.all(abs(true_d - envelope['d0']) <= envelope['radius'] + 1e-13)
        assert np.all(abs(exact - envelope['scores']) <= envelope['error_bounds'] + 1e-13)


def simple_family(with_zero=False):
    P = np.array([[0., 0., 0.], [.5, 1., 0.], [.5, 0., 1.]])
    seed = np.array([1., 0., 0.])
    group = np.array([0., .5, 0.])
    direction = sparse.csc_matrix((group - .5 * P[:, 0]).reshape(-1, 1))
    ranker = ViewRank(sparse.csc_matrix(P), alpha=.4)
    sources = [2, 0] if with_zero else [0]
    directions = [sparse.csc_matrix((3, 1)), direction] if with_zero else [direction]
    masses = [1., .5] if with_zero else [.5]
    directory = BudgetedStencilDirectory(ranker, interaction_stencil([sources]), 0, 10000)
    family = FastPolicyFamily(ranker, seed, sources, directions, masses,
                              directory=directory, ks=(1, 2, 3))
    return family, P, seed, {0: group}


def test_box_stable_cut_does_not_follow_another_k_into_completion(monkeypatch):
    family, P, seed, groups = simple_family()
    calls = []
    original = family._complete

    def complete(d, c, ks, started, initial, materialize):
        calls.append(tuple(ks))
        return original(d, c, ks, started, initial, materialize)

    monkeypatch.setattr(family, '_complete', complete)
    result = PolicyBox(family).query({0: (.5, 4.)}, ks=(1, 2))
    assert result['results'][1]['status'] == 'stable'
    assert result['results'][1]['method'] == 'box_envelope'
    assert result['corners_per_k'][1] == 0
    assert calls and all(ks == (2,) for ks in calls)
    answer = result['results'][2]
    assert answer['status'] == 'unstable'
    witness = answer['witness']
    _, order = reference(P, seed, .4, groups, witness['sources'], witness['strengths'])
    assert set(order[:2]) == set(witness['indices'])
    assert set(order[:2]) != set(answer['indices'])


def test_complete_corner_baseline_proves_stability_and_returns_real_witness():
    family, P, seed, groups = simple_family()
    box = PolicyBox(family)
    stable = box.query({0: (1.2, 2.)}, ks=(1, 2), use_envelope=False)
    assert stable['possible_corners'] == stable['corners_visited'] == 2
    for k in (1, 2):
        assert stable['results'][k]['status'] == 'stable'
        assert stable['results'][k]['method'] == 'all_corners'
    unstable = box.query({0: (.5, 4.)}, ks=(2,), use_envelope=False)
    witness = unstable['results'][2]['witness']
    _, order = reference(P, seed, .4, groups, witness['sources'], witness['strengths'])
    assert set(order[:2]) == set(witness['indices'])


def test_zero_directions_are_filtered_without_reassigning_source_bounds():
    family, _, _, _ = simple_family(with_zero=True)
    result = PolicyBox(family).query({2: (7., 9.), 0: (.5, 4.)}, ks=(1, 2))
    assert result['active_sources'].tolist() == [0]
    assert result['possible_corners'] == 2
    assert result['results'][2]['status'] == 'unstable'
    witness = result['results'][2]['witness']
    assert witness['sources'].tolist() == [0]
    assert witness['strengths'][0] in (.5, 4.)


def test_numerical_tie_is_not_reported_as_instability_or_stability():
    ranker = ViewRank(sparse.eye(3, format='csc'), alpha=.5)
    family = FastPolicyFamily(ranker, np.array([.5, .5, 0.]), [1],
                              [sparse.csc_matrix((3, 1))], [1.], ks=(1, 2))
    result = PolicyBox(family).query({1: (.5, 4.)}, ks=(1, 2))
    assert result['results'][1]['status'] == 'numerical_tie'
    assert result['results'][1]['witness'] is None
    assert result['results'][2]['status'] == 'stable'


def test_all_reported_stability_and_witnesses_match_independent_lu():
    family, P, seed, groups = make_family(keep=0)
    bounds = {1: (.5, 4.), 4: (.5, 4.)}
    for use_envelope, use_pair in ((True, False), (False, False), (True, True)):
        result = PolicyBox(family).query(bounds, ks=(1, 3, 5),
                                        use_envelope=use_envelope, use_pair=use_pair)
        for k, answer in result['results'].items():
            assert answer['status'] != 'numerical_tie'
            if answer['status'] == 'stable':
                for strengths in product((.5, 4.), repeat=2):
                    _, order = reference(P, seed, .72, groups, family.sources, strengths)
                    assert set(order[:k]) == set(answer['indices'])
            else:
                witness = answer['witness']
                _, order = reference(P, seed, .72, groups, family.sources, witness['strengths'])
                assert set(order[:k]) == set(witness['indices'])
                assert set(order[:k]) != set(answer['indices'])


def test_pair_bound_covers_corners_and_interior_with_omitted_response_entries():
    family, P, seed, groups = make_family(keep=1)
    box = PolicyBox(family)
    envelope = box.envelope({1: (.5, 2.), 4: (.5, 2.)})
    assert envelope['valid']
    pairs = [(0, 1), (1, 4), (4, 6), (2, 5)]
    lower = {(i, j): box.pair_bound(envelope, i, j) for i, j in pairs}
    points = list(product((.5, 2.), repeat=2))
    points.extend(np.random.default_rng(12).uniform(.5, 2., size=(25, 2)))
    for strengths in points:
        exact, _ = reference(P, seed, .72, groups, family.sources, strengths)
        for i, j in pairs:
            assert lower[i, j] <= exact[i] - exact[j] + 1e-13


def test_shared_response_cancellation_certifies_box_without_corner_solves():
    P = np.array([[1., 0., .2], [0., 1., .2], [0., 0., .6]])
    seed = np.array([.21, .19, .6])
    group = np.array([.2, .2, 0.])
    direction = sparse.csc_matrix((group - .4 * P[:, 2]).reshape(-1, 1))
    ranker = ViewRank(sparse.csc_matrix(P), alpha=.8)
    family = FastPolicyFamily(ranker, seed, [2], [direction], [.4], keep=3, ks=(1, 2))
    box = PolicyBox(family)
    coordinate = box.query({2: (.5, 2.)}, ks=(1,))
    pair = box.query({2: (.5, 2.)}, ks=(1,), use_pair=True)
    assert coordinate['results'][1]['status'] == pair['results'][1]['status'] == 'stable'
    assert coordinate['results'][1]['method'] == 'all_corners'
    assert pair['results'][1]['method'] == 'box_pair'
    assert pair['corners_visited'] == 0
    assert pair['pair_checks'] > 0
    for strength in np.linspace(.5, 2., 15):
        _, order = reference(P, seed, .8, {2: group}, [2], [strength])
        assert set(order[:1]) == set(pair['results'][1]['indices'])


def test_corner_baseline_does_not_pay_for_whole_box_envelope(monkeypatch):
    family, _, _, _ = simple_family()
    box = PolicyBox(family)

    def unused(bounds):
        raise AssertionError('Complete-corner baseline must not compute the box envelope.')

    monkeypatch.setattr(box, 'envelope', unused)
    result = box.query({0: (.5, 4.)}, ks=(1, 2), use_envelope=False)
    assert result['results'][1]['status'] == 'stable'
    assert result['results'][2]['status'] == 'unstable'


def test_pair_certificate_cannot_ignore_an_unstored_outside_competitor():
    P = np.array([[1., 0., .2, 0.], [0., 1., .2, 0.],
                  [0., 0., .6, 0.], [0., 0., 0., 1.]])
    seed = np.array([.17, .15, .4, .28])
    group = np.array([.2, .2, 0., 0.])
    direction = sparse.csc_matrix((group - .4 * P[:, 2]).reshape(-1, 1))
    family = FastPolicyFamily(ViewRank(sparse.csc_matrix(P), alpha=.8), seed,
                              [2], [direction], [.4], keep=3, ks=(1,))
    box = PolicyBox(family)
    assert 3 not in box.retained_rows
    result = box.query({2: (.5, 2.)}, ks=(1,), use_pair=True)
    answer = result['results'][1]
    assert answer['status'] == 'unstable'
    witness = answer['witness']
    _, order = reference(P, seed, .8, {2: group}, [2], witness['strengths'])
    assert set(order[:1]) == set(witness['indices'])

