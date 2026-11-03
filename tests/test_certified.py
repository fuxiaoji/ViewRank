import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse

from src.certified import certified_topk, prepare_compressed
from src.viewrank import ViewRank
from tests.test_viewrank import example, replacement, transition


def exact_topk(scores, k):
    return np.lexsort((np.arange(len(scores)), -scores))[:k]


def test_full_responses_certify_without_refinement():
    P = example()
    ranker = ViewRank(P)
    prepared = ranker.prepare(replacement(P, [0, 1], [3, 4]), [0, 1])
    compressed = prepare_compressed(ranker, prepared, keep_per_column=ranker.n)
    seed = np.array([0.1, 0.3, 0.0, 0.5, 0.1])
    exact = ranker.solve(seed, prepared)
    result = certified_topk(ranker, seed, compressed, k=2, reference_scores=exact)
    assert result.initial_certified
    assert result.refined_columns == 0
    assert result.score_error <= result.bound + 1e-14
    assert set(result.indices) == set(exact_topk(exact, 2))


def test_zero_responses_refine_and_recover_exact_topk():
    P = example()
    ranker = ViewRank(P)
    prepared = ranker.prepare(replacement(P, [0, 1, 3], [3, 4, 0]), [0, 1, 3])
    compressed = prepare_compressed(ranker, prepared, keep_per_column=0)
    seed = np.array([0.5, 0.2, 0.0, 0.2, 0.1])
    exact = ranker.solve(seed, prepared)
    result = certified_topk(ranker, seed, compressed, k=3, reference_scores=exact)
    assert not result.initial_certified
    assert 0 < result.refined_columns <= prepared.rank
    assert result.score_error <= result.bound + 2e-14
    assert set(result.indices) == set(exact_topk(exact, 3))
    assert compressed.Zhat.nnz == 0  # A query does not mutate its shared template.


def test_residual_bound_before_any_refinement():
    rng = np.random.default_rng(12)
    P = transition(rng.integers(0, 4, size=(29, 29)))
    ranker = ViewRank(P, alpha=0.92)
    prepared = ranker.prepare(replacement(P, [0, 4, 12], [4, 4, 1]), [0, 4, 12])
    compressed = prepare_compressed(ranker, prepared, keep_per_column=3)
    for _ in range(8):
        seed = rng.random(29)
        seed /= seed.sum()
        y = ranker.base_lu.solve((1 - ranker.alpha) * seed)
        small = np.eye(prepared.rank) + compressed.Zhat[compressed.sources, :].toarray()
        c = np.linalg.solve(small, y[compressed.sources])
        approximate = y - compressed.Zhat @ c
        exact = ranker.solve(seed, prepared)
        bound = np.dot(np.abs(c), compressed.rho) / (1 - ranker.alpha)
        assert np.linalg.norm(approximate - exact, ord=1) <= bound + 1e-13


def test_topk_matches_exact_for_random_compression_levels():
    rng = np.random.default_rng(21)
    P = transition(rng.integers(0, 3, size=(37, 37)))
    ranker = ViewRank(P)
    prepared = ranker.prepare(replacement(P, [0, 2, 6, 11], [5, 5, 4, 7]), [0, 2, 6, 11])
    for keep in [0, 1, 8, 37]:
        compressed = prepare_compressed(ranker, prepared, keep_per_column=keep)
        for _ in range(4):
            seed = rng.random(37)
            seed /= seed.sum()
            exact = ranker.solve(seed, prepared)
            result = certified_topk(ranker, seed, compressed, k=5, reference_scores=exact)
            assert set(result.indices) == set(exact_topk(exact, 5))
            assert result.score_error <= result.bound + 1e-13


def test_ties_fall_back_to_full_with_stable_node_order():
    P = sparse.eye(4, format="csc")
    ranker = ViewRank(P)
    D = np.zeros((4, 1))
    D[0, 0], D[1, 0] = -1, 1
    prepared = ranker.prepare(D, [0])
    compressed = prepare_compressed(ranker, prepared, keep_per_column=0)
    seed = np.array([0.0, 0.0, 0.5, 0.5])
    result = certified_topk(ranker, seed, compressed, k=1)
    assert result.exact_fallback
    assert result.refined_columns == 1
    assert result.indices.tolist() == [2]


def test_empty_update_and_memory_statistics():
    ranker = ViewRank(example())
    prepared = ranker.prepare(sparse.csc_matrix((5, 0)), [])
    compressed = prepare_compressed(ranker, prepared, keep_per_column=0)
    seed = np.array([1.0, 0, 0, 0, 0])
    result = certified_topk(ranker, seed, compressed, k=2)
    assert_allclose(result.scores, ranker.solve(seed, prepared))
    assert result.initial_certified
    assert result.bound == 0
    assert compressed.prepare_seconds >= 0
    assert result.working_bytes >= compressed.nbytes
    assert result.solve_seconds >= 0
