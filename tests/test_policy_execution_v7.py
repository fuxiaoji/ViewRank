import numpy as np
import pytest
from scipy import sparse

import src.compiled_policy_cache as compiler
from src.budgeted_family import interaction_stencil
from src.policy_execution_v7 import (EfficientCompiledFastFamily,
    EfficientCompiledDenseFamily, EfficientDenseDirectory,
    EfficientStencilDirectory, family_padding, top_indices)
from src.viewrank import ViewRank


@pytest.mark.parametrize('size', [1, 2, 11, 129])
@pytest.mark.parametrize('ties', [False, True])
def test_partial_selection_matches_complete_score_and_id_order(size, ties):
    rng = np.random.default_rng(91+size)
    values = rng.normal(size=size)
    if ties:
        values = np.round(values)
    complete = np.lexsort((np.arange(size), -values))
    for k in sorted({0, 1, min(size, 2), min(size, 10), size}):
        np.testing.assert_array_equal(top_indices(values, k), complete[:k])


def test_equal_boundary_chooses_ids_outside_the_partition():
    values = np.array([3., 2., 2., 2., 2., 1., 1.])
    np.testing.assert_array_equal(top_indices(values, 3), [0, 1, 2])


def _input():
    rng = np.random.default_rng(88)
    P = rng.random((37, 37))
    P /= P.sum(axis=0)
    G = P.copy()
    G[::2] = 0.
    P, G = sparse.csc_matrix(P), sparse.csc_matrix(G)
    ranker = ViewRank(P, alpha=.5)
    seed = rng.random(37)
    seed /= seed.sum()
    sources = np.array([4, 9, 2])
    masses = np.array([float(G[:, s].sum()) for s in sources])
    directions = [G[:, s]-m*P[:, s] for s, m in zip(sources, masses)]
    y = ranker.base_lu.solve(.5*seed)
    return ranker, G, seed, sources, directions, masses, y


@pytest.mark.parametrize('fused', [False, True])
def test_cached_pad_and_shared_completion_preserve_outputs(monkeypatch, fused):
    ranker, G, seed, sources, directions, masses, y = _input()
    stencil = interaction_stencil([sources, [2, 4, 11]])
    old = compiler.CompiledStencilDirectory(ranker, G, stencil, 4, 1000000)
    with old.prepare(seed, sources, directions, masses, base_scores=y, base_residual=0.) as a:
        expected = [a.solve(s, materialize=True, always_fused=fused) for s in [.5, 1., 2., 4.]]
        expected_bytes = old.nbytes
    monkeypatch.setattr(compiler, 'CompiledFastFamily', EfficientCompiledFastFamily)
    new = EfficientStencilDirectory(ranker, G, stencil, 4, 1000000)
    with new.prepare(seed, sources, directions, masses, base_scores=y, base_residual=0.) as b:
        assert new.nbytes == expected_bytes+8
        for s, expected_result in zip([.5, 1., 2., 4.], expected):
            result = b.solve(s, materialize=True, always_fused=fused)
            np.testing.assert_array_equal(result['scores'], expected_result['scores'])
            np.testing.assert_array_equal(result['error_bounds'], expected_result['error_bounds'])
            assert result['initial_certified'] == expected_result['initial_certified']
            for k in (2, 5, 10):
                np.testing.assert_array_equal(result['sets'][k], expected_result['sets'][k])
    assert new._live_core_bytes == 0
    assert not new._pins


def test_completion_materializes_each_nonzero_direction_once(monkeypatch):
    ranker, G, seed, sources, directions, masses, y = _input()
    monkeypatch.setattr(compiler, 'CompiledFastFamily', EfficientCompiledFastFamily)
    new = EfficientStencilDirectory(ranker, G, interaction_stencil([sources]), 4, 1000000)
    with new.prepare(seed, sources, directions, masses, base_scores=y, base_residual=0.) as family:
        before = new.direction_materializations
        family.solve(1., always_fused=True)
        assert new.direction_materializations == before
        family.solve(2., always_fused=True)
        assert new.direction_materializations-before == len(family.atoms)


def test_cached_padding_is_in_nested_family_budget_and_release(monkeypatch):
    ranker, G, seed, sources, directions, masses, y = _input()
    monkeypatch.setattr(compiler, 'CompiledFastFamily', EfficientCompiledFastFamily)
    new = EfficientStencilDirectory(ranker, G, interaction_stencil([sources]), 4, 1000000)
    with new.prepare(seed, sources, directions, masses, base_scores=y, base_residual=0.) as a:
        first = new._live_core_bytes
        with new.prepare(seed, sources, directions, masses, base_scores=y, base_residual=0.) as b:
            assert new._live_core_bytes == first+b._metadata_bytes
            assert new._live_core_bytes == a._metadata_bytes+b._metadata_bytes
        assert new._live_core_bytes == first
    assert new._live_core_bytes == 0


def test_dense_control_gets_identical_cached_pad_and_byte_accounting(monkeypatch):
    ranker, G, seed, sources, directions, masses, y = _input()
    old = compiler.CompiledDenseDirectory(ranker, G, 1000000)
    with old.prepare(seed, sources, directions, masses, base_scores=y, base_residual=0.) as a:
        expected = [a.solve(s, materialize=True) for s in [.5, 1., 2., 4.]]
        expected_bytes = old.nbytes
        expected_pad = family_padding(a)
    monkeypatch.setattr(compiler, 'CompiledDenseFamily', EfficientCompiledDenseFamily)
    new = EfficientDenseDirectory(ranker, G, 1000000)
    with new.prepare(seed, sources, directions, masses, base_scores=y, base_residual=0.) as b:
        assert new.nbytes == expected_bytes+8
        assert b._base_pad == family_padding(b) == expected_pad
        for s, reference in zip([.5, 1., 2., 4.], expected):
            result = b.solve(s, materialize=True)
            np.testing.assert_array_equal(result['scores'], reference['scores'])
            np.testing.assert_array_equal(result['error_bounds'], reference['error_bounds'])
            assert result['initial_certified'] == reference['initial_certified']
            for k in (2, 5, 10):
                np.testing.assert_array_equal(result['sets'][k], reference['sets'][k])
    assert new._live_core_bytes == 0
    assert not new._pins


def test_common_selector_binding_covers_each_production_import():
    from importlib import import_module
    from src.policy_execution_v7 import bind_common_selection
    names = ('src.policy_family_fast', 'src.full_scan_policy_family',
        'src.compiled_policy_cache', 'src.policy_box',
        'src.baselines.certified_base', 'src.baselines.certified_native_family',
        'src.baselines.dense_family_budget', 'src.baselines.native_offset_family')
    originals = {name: import_module(name).top_indices for name in names}
    try:
        assert set(bind_common_selection()) == set(names)
        assert all(import_module(name).top_indices is top_indices for name in names)
    finally:
        for name, original in originals.items():
            import_module(name).top_indices = original
