import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse

from src.envelope import prepare_envelope, envelope_topk
from src.viewrank import ViewRank
from tests.test_viewrank import example, replacement, transition


def test_source_coordinates_equal_exact_boundary_coefficients():
    ranker = ViewRank(example())
    prepared = ranker.prepare(replacement(ranker.P, [0, 1, 3], [3, 4, 0]), [0, 1, 3])
    compressed = prepare_envelope(ranker, prepared, keep_per_column=0)
    seed = np.array([0.5, 0.2, 0.0, 0.2, 0.1])
    exact = ranker.solve(seed, prepared)
    result = envelope_topk(ranker, seed, compressed, k=3, reference_scores=exact)
    assert_allclose(result.scores[prepared.sources], exact[prepared.sources], atol=1e-14)
    assert np.all(np.abs(result.scores-exact) <= result.error_bounds)
    assert set(result.indices) == set(np.argsort(-exact)[:3])


def test_coordinate_bounds_and_exact_sets_across_alpha_and_keep():
    rng = np.random.default_rng(28)
    P = transition(rng.integers(0, 3, size=(41, 41)))
    for alpha in [0.2, 0.85, 0.99]:
        ranker = ViewRank(P, alpha=alpha)
        prepared = ranker.prepare(replacement(P, [0, 1, 7, 15], [5, 5, 3, 0]), [0, 1, 7, 15])
        for keep in [0, 1, 8, 41]:
            compressed = prepare_envelope(ranker, prepared, keep_per_column=keep)
            for _ in range(5):
                seed = rng.random(41)
                seed /= seed.sum()
                exact = ranker.solve(seed, prepared)
                result = envelope_topk(ranker, seed, compressed, k=5, reference_scores=exact)
                assert np.all(np.abs(result.scores-exact) <= result.error_bounds)
                assert set(result.indices) == set(np.argsort(-exact)[:5])
                assert result.numerical_allowance > 0
                assert result.interval_margin > 0 or result.exact_fallback


def test_tie_fallback_and_empty_view():
    ranker = ViewRank(sparse.eye(4, format="csc"))
    prepared = ranker.prepare(np.array([[-1], [1], [0], [0]]), [0])
    compressed = prepare_envelope(ranker, prepared, keep_per_column=0)
    result = envelope_topk(ranker, np.array([0.0, 0.0, 0.5, 0.5]), compressed, k=1)
    assert result.exact_fallback and result.indices.tolist() == [2]
    empty = ranker.prepare(sparse.csc_matrix((4, 0)), [])
    result = envelope_topk(ranker, np.array([1.0, 0, 0, 0]), prepare_envelope(ranker, empty), k=1)
    assert result.initial_certified
    assert result.refined_columns == 0


def test_numerical_base_error_is_included_in_bound():
    ranker = ViewRank(example())
    prepared = ranker.prepare(replacement(ranker.P, [0], [3]), [0])
    compressed = prepare_envelope(ranker, prepared, keep_per_column=ranker.n)
    seed = np.array([0.2, 0.1, 0.0, 0.6, 0.1])
    y = ranker.base_lu.solve((1-ranker.alpha)*seed)
    y[2] += 1e-7
    result = envelope_topk(ranker, seed, compressed, k=2,
                           reference_scores=ranker.solve(seed, prepared), base_scores=y)
    assert np.all(np.abs(result.scores-ranker.solve(seed, prepared)) <= result.error_bounds)
    assert result.numerical_allowance > 1e-7
