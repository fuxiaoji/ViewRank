from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from scipy import sparse

from src.baselines.prpack import PRPackViewRank
from src.budgeted_family import interaction_stencil
from src.policy_domain_execution import DomainCompiler, DomainDirectory, numeric_payload
from src.policy_domain_polynomial import poly_eval


def random_graph(n=17):
    rng = np.random.default_rng(20261103)
    weights = np.zeros((n, n))
    for column in range(n-1):
        counts = np.bincount(rng.integers(0, n, 31), minlength=n)
        counts[column] += 1
        weights[:, column] = counts/32
    weights[-1, -1] = 1
    mention = weights.copy()
    mention[np.indices(mention.shape)[0] % 2 == 0] = 0
    np.fill_diagonal(mention, 0)
    seed = np.zeros(n); seed[:4] = 1/32; seed[-1] = 7/8
    ranker = PRPackViewRank(sparse.csc_matrix(weights), alpha=.5)
    return ranker, mention, seed


def direct(P, M, sources, seed, weights):
    P, M = P.toarray(), np.asarray(M)
    view = P.copy()
    for source in sources:
        self_loop = np.eye(len(P))[:, source]*P[source, source]
        weak = P[:, source]-M[:, source]-self_loop
        raw = weights[0]*M[:, source]+weights[1]*weak+self_loop
        view[:, source] = raw/raw.sum()
    scores = np.linalg.solve(np.eye(len(P))-.5*view, .5*seed)
    order = np.lexsort((np.arange(len(P)), -scores))
    return scores, order


def ledger(directory):
    assert directory._cache_bytes == sum(atom.nbytes for atom in directory.cache.values())
    assert directory._live_bytes == sum(family.nbytes for family in directory.live)
    assert directory._live_reserved_bytes == sum(family.reservation for family in directory.live)
    assert directory.nbytes == directory.plan_bytes+directory._cache_bytes+directory._live_bytes
    assert directory.reserved_bytes == directory.plan_bytes+directory._cache_bytes+directory._live_reserved_bytes+directory._pending_bytes
    assert directory.nbytes <= directory.reserved_bytes <= directory.budget_bytes


@pytest.mark.parametrize('mode', ['sparse', 'dense', 'dense_pruned', 'scan', 'no_core', 'always_full'])
def test_fit_representation_certifies_real_domain_and_matches_independent_views(mode):
    ranker, M, seed = random_graph()
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        sources = [0, 1, 2, 3]
        directory = DomainDirectory(compiler, interaction_stencil([sources]), keep=3, mode=mode)
        with directory.prepare(seed, sources, ks=(1,)) as family:
            ledger(directory)
            result = family.certify(ks=(1,), depth=1, early_witness=False)
            assert result['cuts'][1]['status'] == 'STABLE'
            assert set(result['cuts'][1]['selected_ids']) == {16}
            ledger(directory)
            for weights in product((.5, 1., 1.5, 2., 4.), (.25, 1., 1.5, 2.)):
                _, order = direct(ranker.P, M, sources, seed, weights)
                assert order[0] == 16
            if mode not in ('dense', 'dense_pruned', 'always_full'):
                assert all(atom.response is None for atom in family.atoms)
            if mode == 'no_core':
                assert all(atom.tail == 0 for atom in family.atoms)
        assert not directory.pins and not directory.live and family.nbytes == 0
        ledger(directory)
    finally:
        ranker.base_lu.graph.close()


def test_global_group_source_reuse_and_pinned_lifetime_are_exact():
    ranker, M, seed = random_graph()
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        stencil = interaction_stencil([[0, 1], [1, 2]])
        directory = DomainDirectory(compiler, stencil, keep=2)
        first = directory.prepare(seed, [0, 1], ks=(1,))
        assert directory.misses == 4 and directory.hits == 0
        assert compiler.exact_source_compilations == 2
        assert [atom.key for atom in first.atoms] == [(0, 0), (0, 1), (1, 0), (1, 1)]
        same = directory.prepare(seed, [0, 1], ks=(1,))
        assert directory.hits == 4 and directory.misses == 4
        assert compiler.exact_source_compilations == 2
        assert all(a is b for a, b in zip(first.atoms, same.atoms))
        assert all(value == 2 for value in directory.pins.values())
        ledger(directory)
        first.close(); ledger(directory)
        assert all(value == 1 for value in directory.pins.values())
        same.close(); ledger(directory)
        with directory.prepare(seed, [1, 2], ks=(1,)):
            assert directory.hits == 6 and directory.misses == 6
            ledger(directory)
    finally:
        ranker.base_lu.graph.close()


def test_budget_unknown_retains_all_sources_and_cuts_without_allocating_a_free_basis():
    ranker, M, seed = random_graph()
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        directory = DomainDirectory(compiler, interaction_stencil([[0, 1, 2, 3]]), budget_bytes=6000)
        with directory.prepare(seed, [0, 1, 2, 3], ks=(1, 2, 5)) as family:
            assert family.route == 'budget_unknown'
            assert list(family.sources) == [0, 1, 2, 3]
            assert not family.atoms and not directory.cache
            result = family.certify(ks=(1, 2, 5), early_witness=False)
            assert set(result['cuts']) == {1, 2, 5}
            assert all(cut['status'] == 'UNKNOWN' for cut in result['cuts'].values())
            ledger(directory)
    finally:
        ranker.base_lu.graph.close()


def test_actual_point_certificate_bounds_independent_direct_solution_and_restores_graph():
    ranker, M, seed = random_graph()
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        for weights in ((1., 1.), (4., .25), (2.5, 1.125)):
            result = compiler.point(seed, [0, 1, 2, 3], weights, ks=(1, 5, 10), materialize=True)
            scores, order = direct(ranker.P, M, [0, 1, 2, 3], seed, weights)
            assert np.max(abs(scores-result['scores'])) <= float(result['score_error_bound'])+2e-16
            for k, strict in result['strict_flags'].items():
                if strict: assert set(result['sets'][k]) == set(order[:k])
            assert not ranker.base_lu.graph.previous_sources
    finally:
        ranker.base_lu.graph.close()


def test_unstable_requires_two_guarded_actual_views_not_a_surrogate_membership_change():
    P = np.zeros((5, 5)); M = np.zeros_like(P)
    P[1, 0] = M[1, 0] = .5; P[2, 0] = .5
    P[2, 1] = M[2, 1] = .5; P[3, 1] = .5
    P[2, 2] = P[3, 3] = P[4, 4] = 1
    seed = np.array([.95, 0, 0, 0, .05])
    ranker = PRPackViewRank(sparse.csc_matrix(P), alpha=.5)
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        directory = DomainDirectory(compiler, interaction_stencil([[0, 1]]), keep=2)
        with directory.prepare(seed, [0, 1], ks=(4,)) as family:
            result = family.certify(box=((.25, 4), (.99, 1.01)), ks=(4,), early_witness=True)
            cut = result['cuts'][4]
            assert cut['status'] == 'UNSTABLE'
            witness = cut['witness']
            assert witness['first']['strict_flags'][4] and witness['second']['strict_flags'][4]
            assert set(witness['first']['sets'][4]) != set(witness['second']['sets'][4])
            for point in witness.values():
                _, order = direct(ranker.P, M, [0, 1], seed, point['weights'])
                assert set(point['sets'][4]) == set(order[:4])
            ledger(directory)
    finally:
        ranker.base_lu.graph.close()


def test_no_source_domain_is_a_constant_actual_operator():
    ranker, M, seed = random_graph()
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        directory = DomainDirectory(compiler, {})
        with directory.prepare(seed, [], ks=(1,)) as family:
            result = family.certify(ks=(1,))
            assert result['cuts'][1]['status'] == 'STABLE'
            assert result['point_checks'] == 0
            ledger(directory)
    finally:
        ranker.base_lu.graph.close()


def test_lru_evicts_only_unpinned_entries_and_ledgers_reconstruct():
    ranker, M, seed = random_graph()
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        directory = DomainDirectory(compiler, interaction_stencil([[s] for s in range(12)]),
            keep=2, budget_bytes=14000)
        for source in range(12):
            with directory.prepare(seed, [source], ks=(1,)) as family:
                assert family.route == 'sparse'
                assert (0, source) in directory.pins and (1, source) in directory.pins
                ledger(directory)
            ledger(directory)
        assert directory.evictions > 0
        assert not directory.pins
        directory.clear()
        assert not directory.cache and directory.nbytes == directory.plan_bytes
    finally:
        ranker.base_lu.graph.close()


def test_outside_candidate_tail_protection_on_a_larger_graph():
    ranker, M, seed = random_graph(n=257)
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        directory = DomainDirectory(compiler, interaction_stencil([[0, 1, 2, 3]]), keep=2)
        with directory.prepare(seed, [0, 1, 2, 3], ks=(1,)) as family:
            assert len(family.candidate_rows) < compiler.n
            assert any(atom.tail > 0 for atom in family.atoms)
            result = family.certify(ks=(1,), early_witness=False)
            assert result['cuts'][1]['status'] == 'STABLE'
            for weights in ((1., .25), (4., 2.), (2.5, .875)):
                _, order = direct(ranker.P, M, [0, 1, 2, 3], seed, weights)
                assert order[0] == 256
            ledger(directory)
    finally:
        ranker.base_lu.graph.close()


@pytest.mark.parametrize('certifier', ['poly', 'interval', 'lifting'])
@pytest.mark.parametrize('mode', ['sparse', 'dense', 'dense_pruned'])
def test_all_strong_certifiers_share_retained_inputs_and_budget(mode, certifier):
    ranker, M, seed = random_graph()
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        directory = DomainDirectory(compiler, interaction_stencil([[0, 1, 2, 3]]), keep=3, mode=mode)
        with directory.prepare(seed, [0, 1, 2, 3], ks=(1,)) as family:
            result = family.certify(ks=(1,), depth=1, certifier=certifier)
            assert result['cuts'][1]['status'] == 'STABLE'
            # Certify-first permits every strong baseline to avoid needless
            # full-output point scans on an already proven stable domain.
            assert result['point_checks'] == 0
            ledger(directory)
    finally:
        ranker.base_lu.graph.close()


def test_dense_pruned_shares_sparse_stencil_candidates_and_tails_but_reads_full_coordinates():
    ranker, M, seed = random_graph(n=257)
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        stencil = interaction_stencil([[0, 1, 2, 3], [0, 40, 41], [1, 90]])
        main = DomainDirectory(compiler, stencil, keep=2, mode='sparse')
        control = DomainDirectory(compiler, stencil, keep=2, mode='dense_pruned')
        with main.prepare(seed, [0, 1, 2, 3], ks=(1,)) as a, control.prepare(seed, [0, 1, 2, 3], ks=(1,)) as b:
            np.testing.assert_array_equal(a.candidate_rows, b.candidate_rows)
            np.testing.assert_array_equal(a.tails, b.tails)
            for aa, bb in zip(a.atoms, b.atoms):
                np.testing.assert_array_equal(aa.rows, bb.rows)
                assert bb.response is not None or bb.zero
            assert b.values is None and b.mask is None
            assert b.columns is not None
            ledger(main); ledger(control)
    finally:
        ranker.base_lu.graph.close()
