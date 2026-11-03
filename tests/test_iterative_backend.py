import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse

from src.iterative_backend import MatrixFreeViewRank
from src.viewrank import ViewRank
from src.envelope import prepare_envelope, envelope_topk
from tests.test_viewrank import example, replacement, transition


def test_signed_batched_rhs_has_per_column_residual_bound():
    rng = np.random.default_rng(71)
    P = transition(rng.integers(0, 3, size=(33, 33)))
    rhs = rng.normal(size=(33, 4))
    for alpha in [.2, .5, .85, .99]:
        iterative = MatrixFreeViewRank(P, alpha=alpha)
        reference = ViewRank(P, alpha=alpha)
        result = iterative.base_lu.solve(rhs)
        assert np.max(np.sum(np.abs(rhs-iterative.A@result), axis=0)) <= 1e-12
        assert_allclose(result, reference.base_lu.solve(rhs), atol=2e-11)
        assert iterative.base_lu.rhs_columns == 4
        assert iterative.base_lu.calls == 1
        assert iterative.base_payload_bytes > 0


def test_posterior_envelope_covers_true_lu_scores():
    P = example()
    iterative, reference = MatrixFreeViewRank(P, alpha=.5), ViewRank(P, alpha=.5)
    D = replacement(P, [0, 1, 3], [3, 4, 0])
    prepared = iterative.prepare(D, [0, 1, 3])
    exact_prepared = reference.prepare(D, [0, 1, 3])
    seed = np.array([.3, .2, 0, .4, .1])
    exact = reference.solve(seed, exact_prepared)
    for keep in [0, 1, 5]:
        compressed = prepare_envelope(iterative, prepared, keep_per_column=keep)
        result = envelope_topk(iterative, seed, compressed, k=2, reference_scores=exact)
        assert np.all(np.abs(result.scores-exact) <= result.error_bounds)
        assert set(result.indices) == set(np.argsort(-exact)[:2])
        assert result.numerical_allowance > 0


def test_empty_view_vector_rhs_and_cached_preparation():
    ranker = MatrixFreeViewRank(example(), alpha=.5)
    prepared = ranker.prepare(sparse.csc_matrix((5, 0)), [])
    assert ranker.prepare(sparse.csc_matrix((5, 0)), []) is prepared
    seed = np.array([1.0, 0, 0, 0, 0])
    result = ranker.solve(seed, prepared)
    assert ranker.residual(seed, result, prepared) <= 1e-12
    assert abs(result.sum()-1) < 2e-12
