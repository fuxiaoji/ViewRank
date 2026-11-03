from itertools import product

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import sparse

from src.composable import ComposableBoundaryDirectory
from src.viewrank import ViewRank, direct_solve
from tests.test_viewrank import example, replacement, transition


def test_unseen_combinations_use_exact_boundary_without_dense_responses():
    P = example()
    ranker = ViewRank(P)
    universe = np.array([0, 1, 2, 3])
    directory = ComposableBoundaryDirectory(ranker, universe, keep_per_atom=1)
    seed = np.array([0.1, 0.2, 0.0, 0.6, 0.1])
    y = ranker.base_lu.solve(0.15 * seed)
    seen = set()
    for choices in product([-1, 0, 1], repeat=4):
        sources = universe[np.asarray(choices) >= 0]
        targets = [4 if choices[s] == 0 else (int(s) + 1) % 4 for s in sources]
        D = replacement(P, sources, targets)
        view = directory.compose(sources, D)
        prepared = ranker.prepare(D, sources)
        if prepared.rank == len(sources):
            assert_allclose(view.K, prepared.K, atol=2e-14)
        reference = ranker.solve(seed, prepared)
        result = directory.topk(seed, view, k=2, base_scores=y)
        assert np.all(np.abs(result.scores - reference) <= result.error_bounds + 2e-14)
        assert set(result.indices) == set(np.argsort(-reference)[:2])
        assert result.fallback_solve_calls <= 1
        assert view.new_whole_view == (view.fingerprint not in seen)
        seen.add(view.fingerprint)
    assert directory.misses <= 8
    assert directory.hits > directory.misses
    for atom in directory.cache.values():
        assert atom.boundary.shape == (len(universe),)
        assert atom.rows.size <= 1 and atom.values.size <= 1
        assert not any(isinstance(value, np.ndarray) and value.shape == (ranker.n,)
                       for value in vars(atom).values())


def test_merged_restore_matches_direct_with_at_most_one_solve():
    rng = np.random.default_rng(47)
    P = transition(rng.integers(0, 4, size=(31, 31)))
    for alpha in [0.2, 0.85, 0.99]:
        ranker = ViewRank(P, alpha=alpha)
        sources = np.array([0, 3, 9, 18])
        directory = ComposableBoundaryDirectory(ranker, sources, keep_per_atom=0)
        D = replacement(P, sources, [4, 4, 2, 0])
        view = directory.compose(sources, D)
        Pv = ranker.view_matrix(ranker.prepare(D, sources))
        for _ in range(5):
            seed = rng.random(31)
            seed /= seed.sum()
            expected = direct_solve(Pv, seed, alpha)
            result = directory.topk(seed, view, k=5)
            fused, fused_ids = directory.fused_topk(seed, view, k=5)
            assert result.fallback_one_solve
            assert result.fallback_solve_calls == 1
            assert_allclose(result.scores, expected, atol=1e-13)
            assert set(result.indices) == set(np.argsort(-expected)[:5])
            assert_allclose(fused, expected, atol=1e-13)
            assert set(fused_ids) == set(np.argsort(-expected)[:5])
            assert np.all(np.abs(result.scores - expected) <= result.error_bounds + 1e-13)


def test_empty_update_and_duplicate_source_contract():
    ranker = ViewRank(example())
    directory = ComposableBoundaryDirectory(ranker, [0, 1, 2], keep_per_atom=2)
    empty = directory.compose([], sparse.csc_matrix((ranker.n, 0)))
    seed = np.array([0.2, 0.1, 0.3, 0.1, 0.3])
    result = directory.topk(seed, empty, k=2)
    assert_allclose(result.scores, direct_solve(ranker.P, seed), atol=1e-14)
    assert result.fallback_solve_calls == 0
    unchanged = directory.compose([0, 1], sparse.csc_matrix((ranker.n, 2)))
    assert unchanged.rank == 0 and unchanged.fingerprint == empty.fingerprint
    assert not unchanged.new_whole_view
    assert directory.misses == 0
    with pytest.raises(ValueError, match="repeated-source"):
        directory.compose([0, 0], sparse.csc_matrix((ranker.n, 2)))


def test_atom_cache_covers_delta_and_source_not_only_policy_name():
    ranker = ViewRank(example())
    directory = ComposableBoundaryDirectory(ranker, [0, 1], keep_per_atom=0)
    first = replacement(ranker.P, [0], [3])
    changed = replacement(ranker.P, [0], [4])
    a = directory.atomic(0, first)
    assert directory.atomic(0, first.copy()) is a
    b = directory.atomic(0, changed)
    assert b.fingerprint != a.fingerprint
    assert directory.misses == 2 and directory.hits == 1


def test_normalized_strength_family_uses_one_response_per_direction():
    from src.semantic_basis import SemanticStrengthDirectory, strength_scale
    rng = np.random.default_rng(101)
    P = transition(rng.integers(0, 4, size=(31, 31)))
    ranker = ViewRank(P)
    B = np.array([0, 1, 3, 5])
    directory = SemanticStrengthDirectory(ranker, B, keep_per_atom=2)
    seed = rng.random(31)
    seed /= seed.sum()
    directions, masses = [], []
    for source in B:
        p = P[:, source].toarray().ravel()
        group = p.copy()
        group[::2] = 0
        mass = group.sum()
        directions.append(sparse.csc_matrix((group - mass * p).reshape(-1, 1)))
        masses.append(mass)
    selector = sparse.csc_matrix((np.ones(4), (np.arange(4), B)), shape=(4, 31))
    for strength in [0.2, 0.5, 0.8, 1, 2, 4]:
        view = directory.compose_strength(B, directions, masses, strength)
        D = sparse.hstack([d * strength_scale(strength, m) for d, m in zip(directions, masses)], format="csc")
        expected = direct_solve(P + D @ selector, seed)
        result = directory.topk(seed, view, k=5)
        fused, _ = directory.fused_topk(seed, view, k=5)
        assert_allclose(fused, expected, atol=1e-13)
        assert np.all(np.abs(result.scores - expected) <= result.error_bounds + 1e-12)
    assert directory.atomic_solve_calls == 4
    assert directory.misses == 4
