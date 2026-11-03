import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.linalg import lu_solve

from src.boundary import boundary_topk, prepare_boundary
from src.viewrank import ViewRank
from tests.test_viewrank import example, replacement, transition


def test_exact_boundary_coupling_survives_zero_response_truncation():
    ranker = ViewRank(example())
    prepared = ranker.prepare(replacement(ranker.P, [0, 1, 3], [3, 4, 0]), [0, 1, 3])
    compressed = prepare_boundary(ranker, prepared, keep_per_column=0)
    assert compressed.Zhat.nnz == 0
    assert_allclose(compressed.K, prepared.K, atol=0)
    assert np.linalg.norm(compressed.K - np.eye(3)) > 0
    seed = np.array([0.5, 0.2, 0.0, 0.2, 0.1])
    exact = ranker.solve(seed, prepared)
    y = ranker.base_lu.solve((1-ranker.alpha)*seed)
    c = lu_solve(compressed.small_lu, y[compressed.sources])
    bound = np.dot(np.abs(c), compressed.tail_inf)
    assert np.max(np.abs(y - exact)) <= bound + 1e-14
    result = boundary_topk(ranker, seed, compressed, k=3, reference_scores=exact)
    assert set(result.indices) == set(np.argsort(-exact)[:3])
    assert result.score_error <= result.bound + 1e-13


def test_direct_tail_bound_and_certified_set_on_random_graphs():
    rng = np.random.default_rng(42)
    P = transition(rng.integers(0, 3, size=(31, 31)))
    for alpha in [0.2, 0.85, 0.99]:
        ranker = ViewRank(P, alpha=alpha)
        prepared = ranker.prepare(replacement(P, [0, 1, 7, 10], [5, 5, 3, 0]), [0, 1, 7, 10])
        for keep in [0, 1, 8, 31]:
            compressed = prepare_boundary(ranker, prepared, keep_per_column=keep)
            for _ in range(4):
                seed = rng.random(31)
                seed /= seed.sum()
                y = ranker.base_lu.solve((1-alpha)*seed)
                c = lu_solve(compressed.small_lu, y[compressed.sources])
                approximate = y-compressed.Zhat@c
                exact = ranker.solve(seed, prepared)
                bound = np.dot(np.abs(c), compressed.tail_inf)
                assert np.max(np.abs(approximate-exact)) <= bound+1e-13
                result = boundary_topk(ranker, seed, compressed, k=5, reference_scores=exact, base_scores=y)
                assert set(result.indices) == set(np.argsort(-exact)[:5])
                assert result.score_error <= result.bound+1e-13
                assert compressed.Zhat.nnz <= keep*prepared.rank


def test_full_keep_initial_certificate_and_query_independence():
    ranker = ViewRank(example())
    prepared = ranker.prepare(replacement(ranker.P, [0], [3]), [0])
    compressed = prepare_boundary(ranker, prepared, keep_per_column=ranker.n)
    seed = np.array([0.2, 0.1, 0.0, 0.6, 0.1])
    result = boundary_topk(ranker, seed, compressed, k=2, reference_scores=ranker.solve(seed, prepared))
    assert result.initial_certified
    assert result.refined_columns == 0
    assert result.bound == 0
    assert result.score_error < 1e-13


def test_tied_boundary_restores_and_uses_stable_order():
    ranker = ViewRank(sparse.eye(4, format="csc"))
    prepared = ranker.prepare(np.array([[-1], [1], [0], [0]]), [0])
    compressed = prepare_boundary(ranker, prepared, keep_per_column=0)
    result = boundary_topk(ranker, np.array([0.0, 0.0, 0.5, 0.5]), compressed, k=1)
    assert result.exact_fallback
    assert result.indices.tolist() == [2]
    assert result.refined_columns == 1


def test_empty_update_requires_no_response():
    ranker = ViewRank(example())
    prepared = ranker.prepare(sparse.csc_matrix((5, 0)), [])
    compressed = prepare_boundary(ranker, prepared, keep_per_column=0)
    seed = np.array([1.0, 0, 0, 0, 0])
    result = boundary_topk(ranker, seed, compressed, k=1)
    assert_allclose(result.scores, ranker.solve(seed, prepared))
    assert result.initial_certified and result.refined_columns == 0
