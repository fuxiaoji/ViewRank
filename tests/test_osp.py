import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.linalg import splu

from src.osp import offset_seed, osp_update, propagate_offset


def transitions():
    # Directed cycle plus a dangling-node self-loop; update one complete column.
    P = sparse.csc_matrix(np.array([
        [0, 0, 0.5, 0], [1, 0, 0.5, 0],
        [0, 1, 0, 0], [0, 0, 0, 1],
    ], dtype=float))
    delta = np.array([[0.4], [-0.8], [0.4], [0.0]])
    view = P + sparse.csc_matrix(delta) @ sparse.csc_matrix(([1.0], ([0], [0])), shape=(1, 4))
    return P, sparse.csc_matrix(delta), view.tocsc()


def solve(P, seeds, alpha):
    return splu(sparse.eye(P.shape[0], format="csc") - alpha * P).solve((1 - alpha) * seeds)


def test_signed_offset_includes_zeroth_term_and_matches_direct_solve():
    P, D, view = transitions()
    seed = np.array([1.0, 0, 0, 0])
    y = solve(P, seed, 0.85)
    q = offset_seed(D, [0], y)
    assert q.min() < 0 < q.max()
    result = osp_update(view, D, [0], y, tol=1e-12)
    expected = solve(view, seed, 0.85)
    assert_allclose(result.scores, expected, atol=1e-12)
    assert np.sum(np.abs(result.scores - expected)) <= result.tail_bound_l1[0] + 2e-15
    assert_allclose(result.scores.sum(), 1.0, atol=1e-13)


def test_dense_and_exact_sparse_frontiers_on_multiple_personalizations():
    P, D, view = transitions()
    seeds = np.array([[1, 0.2, 0], [0, 0.3, 0], [0, 0.1, 0], [0, 0.4, 1.0]])
    y = solve(P, seeds, 0.85)
    expected = solve(view, seeds, 0.85)
    for mode in ["dense", "sparse"]:
        result = osp_update(view, D, [0], y, tol=1e-10, mode=mode)
        error = np.sum(np.abs(result.scores - expected), axis=0)
        assert np.all(error <= result.tail_bound_l1 + 3e-15)
        assert np.all(error < 1e-10)
        assert result.first_converged_iteration[-1] == 0
        assert_allclose(result.scores[:, -1], seeds[:, -1], atol=1e-15)


def test_tolerance_trades_iterations_for_bounded_score_error():
    P, D, view = transitions()
    seed = np.array([1.0, 0, 0, 0])
    y = solve(P, seed, 0.85)
    exact = osp_update(view, D, [0], y, tol=1e-12)
    approximate = osp_update(view, D, [0], y, tol=1e-4)
    assert approximate.iterations < exact.iterations
    assert approximate.tail_bound_l1[0] <= 0.85e-4
    assert np.linalg.norm(approximate.scores - solve(view, seed, 0.85), ord=1) <= approximate.tail_bound_l1[0] + 2e-15


def test_zero_change_needs_no_propagation():
    P, _, _ = transitions()
    y = solve(P, np.array([0.2, 0.1, 0.3, 0.4]), 0.85)
    for mode in ["dense", "sparse"]:
        result = propagate_offset(P, y, np.zeros(4), mode=mode)
        assert result.iterations == 0
        assert_allclose(result.scores, y, atol=0)
