import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from src.threshold import SortedResponseIndex


def reference(columns, coeff, k):
    score = np.sum(columns * coeff, axis=1)
    ids = np.lexsort((np.arange(len(score)), -score))[:k]
    return ids, score[ids]


def test_signed_columns_match_dense_across_seeds_and_k():
    for seed in range(8):
        rng = np.random.default_rng(seed)
        columns = rng.normal(size=(151, 7))
        index = SortedResponseIndex(columns)
        for k in [1, 10, 151]:
            coeff = rng.normal(size=7)
            coeff[2] = 0
            result = index.topk(coeff, k)
            ids, scores = reference(columns, coeff, k)
            assert_array_equal(result.indices, ids)
            assert_allclose(result.scores, scores, rtol=0, atol=0)
            assert k <= result.visited_nodes <= 151


def test_all_ties_use_ascending_node_index():
    columns = np.column_stack([np.arange(41), np.arange(41)])
    result = SortedResponseIndex(columns).topk([1, -1], 8)
    assert_array_equal(result.indices, np.arange(8))
    assert_array_equal(result.scores, np.zeros(8))
    assert result.visited_nodes == 41


def test_equal_scores_at_boundary_are_not_certified_prematurely():
    columns = np.array([[3, -1], [2, 0], [1, 1], [0, 2]], dtype=float)
    result = SortedResponseIndex(columns).topk([1, 1], 2)
    assert_array_equal(result.indices, [0, 1])
    assert result.visited_nodes == 4


def test_concentrated_response_stops_before_full_scan():
    columns = np.zeros((1000, 3))
    columns[:4] = [[1, 0.2, -0.1], [0.9, 0.1, -0.2], [0.1, 0, 0], [0.01, 0.01, 0]]
    result = SortedResponseIndex(columns).topk([1, 1, -1], 2)
    assert_array_equal(result.indices, [0, 1])
    assert result.visited_nodes < 20
    assert result.scores[-1] > result.unseen_upper_bound


def test_seed_and_template_indexes_join_without_resorting():
    rng = np.random.default_rng(7)
    seed_columns = rng.random((71, 2))
    template_columns = rng.normal(size=(71, 4))
    seed_index = SortedResponseIndex(seed_columns)
    template_index = SortedResponseIndex(template_columns)
    combined = SortedResponseIndex.concatenate(seed_index, template_index)
    coeff = np.array([0.3, 0.7, -0.1, 0.2, -0.3, 0.5])
    result = combined.topk(coeff, 7)
    ids, scores = reference(np.column_stack([seed_columns, template_columns]), coeff, 7)
    assert_array_equal(result.indices, ids)
    assert_allclose(result.scores, scores, atol=0, rtol=0)


def test_zero_coefficients_and_single_node():
    result = SortedResponseIndex(np.ones((13, 3))).topk(np.zeros(3), 5)
    assert_array_equal(result.indices, np.arange(5))
    assert result.visited_nodes == 0
    single = SortedResponseIndex(np.array([[1.5, 2.0]])).topk([-1, 1], 1)
    assert_array_equal(single.indices, [0])
    assert_array_equal(single.scores, [0.5])


def test_fortran_response_storage_uses_canonical_row_arithmetic():
    rng = np.random.default_rng(19)
    columns = np.asfortranarray(rng.normal(size=(149, 17)))
    coeff = rng.normal(size=17)
    result = SortedResponseIndex(columns).topk(coeff, 10)
    ids, scores = reference(np.ascontiguousarray(columns), coeff, 10)
    assert_array_equal(result.indices, ids)
    assert_allclose(result.scores, scores, atol=0, rtol=0)
