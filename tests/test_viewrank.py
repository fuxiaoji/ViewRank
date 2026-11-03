import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse

from src.viewrank import ViewRank, direct_solve, power_iteration


def transition(weights):
    weights = np.asarray(weights, dtype=float).copy()
    dangling = np.flatnonzero(weights.sum(axis=0) == 0)
    weights[dangling, dangling] = 1
    return sparse.csc_matrix(weights / weights.sum(axis=0))


def example():
    return transition([[1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [0, 2, 0, 1, 0],
                       [0, 0, 2, 0, 0], [0, 0, 0, 1, 0]])


def replacement(P, sources, destinations):
    D = np.zeros((P.shape[0], len(sources)))
    for j, (source, target) in enumerate(zip(sources, destinations)):
        D[:, j] = -P[:, source].toarray().ravel()
        D[target, j] += 1
    return sparse.csc_matrix(D)


def test_multicolumn_update_matches_direct_and_conserves_probability():
    P = example()
    ranker = ViewRank(P)
    prepared = ranker.prepare(replacement(P, [0, 2, 4], [3, 1, 0]), [0, 2, 4])
    seed = np.array([0.3, 0.1, 0.0, 0.6, 0.0])
    x = ranker.solve(seed, prepared)
    expected = direct_solve(ranker.view_matrix(prepared), seed)
    assert_allclose(x, expected, atol=1e-13)
    assert_allclose(x.sum(), 1.0, atol=1e-13)
    assert x.min() >= -1e-14
    assert ranker.residual(seed, x, prepared) < 1e-13


def test_duplicate_sources_merge_and_shared_target_is_valid():
    P = example()
    D = replacement(P, [0, 3], [2, 2])
    split = sparse.hstack([0.4 * D[:, 0], D[:, 1], 0.6 * D[:, 0]], format="csc")
    ranker = ViewRank(P)
    prepared = ranker.prepare(split, [0, 3, 0])
    assert prepared.sources.tolist() == [0, 3]
    assert prepared.rank == 2
    assert_allclose(prepared.D.toarray(), D.toarray())
    seed = np.array([1.0, 0, 0, 0, 0])
    assert_allclose(ranker.solve(seed, prepared), direct_solve(ranker.view_matrix(prepared), seed), atol=1e-13)


def test_dangling_self_loop_and_no_change_restore_base():
    P = example()
    assert P[4, 4] == 1.0
    ranker = ViewRank(P)
    delta = replacement(P, [1], [4])
    cancelled = ranker.prepare(sparse.hstack([delta, -delta]), [1, 1])
    assert cancelled.rank == 0
    seed = np.array([0, 0, 0, 0, 1.0])
    assert_allclose(ranker.solve(seed, cancelled), seed, atol=1e-14)
    assert_allclose(ranker.view_matrix(cancelled).toarray(), P.toarray())


def test_cache_key_covers_full_update_and_canonical_source_order():
    P = example()
    ranker = ViewRank(P)
    D = replacement(P, [0, 1], [3, 4])
    first = ranker.prepare(D, [0, 1], key="template")
    reordered = ranker.prepare(D[:, [1, 0]], [1, 0], key="template")
    different = ranker.prepare(replacement(P, [0, 1], [4, 3]), [0, 1], key="template")
    assert first is reordered
    assert first.fingerprint != different.fingerprint
    assert ranker.cache_hits == 1
    assert ranker.cache_misses == 2
    assert ranker.cache_bytes == first.nbytes + different.nbytes
    assert first.prepare_seconds >= 0
    assert ranker.base_factor_seconds >= 0


def test_power_iteration_warm_and_cold_match_exact():
    P = example()
    seed = np.array([0.2, 0.3, 0.1, 0.1, 0.3])
    exact = direct_solve(P, seed)
    cold, cold_steps = power_iteration(P, seed, return_iterations=True)
    warm, warm_steps = power_iteration(P, seed, x0=exact, return_iterations=True)
    assert_allclose(cold, exact, atol=1e-10)
    assert_allclose(warm, exact, atol=1e-13)
    assert warm_steps <= cold_steps


def test_random_graph_updates_across_alpha_and_seeds():
    rng = np.random.default_rng(7)
    P = transition(rng.integers(0, 4, size=(17, 17)))
    for alpha in [0.2, 0.85, 0.99]:
        ranker = ViewRank(P, alpha=alpha)
        prepared = ranker.prepare(replacement(P, [2, 4, 10], [5, 5, 0]), [2, 4, 10])
        view = ranker.view_matrix(prepared)
        for _ in range(4):
            seed = rng.random(17)
            seed /= seed.sum()
            x = ranker.solve(seed, prepared)
            assert_allclose(x, direct_solve(view, seed, alpha=alpha), atol=2e-13)
            assert ranker.residual(seed, x, prepared) < 3e-14
