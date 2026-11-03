from fractions import Fraction as Q
import hashlib

import numpy as np
import pytest
from scipy import sparse

from src.baselines.prpack import PRPackViewRank
from src.budgeted_family import interaction_stencil
from src.policy_domain_execution import DomainCompiler, DomainDirectory, _top
from src.policy_domain_adaptive import AdaptiveDomainOperator


def graph(n=257):
    rng = np.random.default_rng(31103)
    P = np.zeros((n, n))
    for source in range(n-1):
        counts = np.bincount(rng.integers(0, n, 31), minlength=n)
        counts[source] += 1
        P[:, source] = counts/32
    P[-1, -1] = 1
    M = P.copy(); M[np.indices(M.shape)[0] % 2 == 0] = 0
    np.fill_diagonal(M, 0)
    seed = np.zeros(n); seed[:4] = 1/32; seed[-1] = 7/8
    ranker = PRPackViewRank(sparse.csc_matrix(P), alpha=.5)
    return ranker, M, seed


def ledger(directory):
    assert directory._cache_bytes == sum(atom.nbytes for atom in directory.cache.values())
    assert directory._live_bytes == sum(f.nbytes for f in directory.live)
    assert directory._live_reserved_bytes == sum(f.reservation for f in directory.live)
    assert directory.nbytes <= directory.reserved_bytes <= directory.budget_bytes


def direct(P, M, seed, sources, weights):
    P = P.toarray(); view = P.copy()
    for source in sources:
        self_loop = np.eye(len(P))[:, source]*P[source, source]
        weak = P[:, source]-M[:, source]-self_loop
        column = weights[0]*M[:, source]+weights[1]*weak+self_loop
        view[:, source] = column/column.sum()
    x = np.linalg.solve(np.eye(len(P))-.5*view, .5*seed)
    return np.lexsort((np.arange(len(P)), -x))


def test_preview_is_current_paid_base_only_and_dense_pruned_shares_exact_rows_and_tails():
    ranker, M, seed = graph()
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        sources = [0, 1, 2, 3]
        stencil = interaction_stencil([sources, [0, 100, 120]])
        y, _, _ = compiler.base(seed)
        directories = [DomainDirectory(compiler, stencil, keep=2, preview_keep=16, mode=mode)
            for mode in ('sparse', 'dense_pruned')]
        with directories[0].prepare(seed, sources, base_scores=y, ks=(1,)) as a, directories[1].prepare(seed, sources, base_scores=y, ks=(1,)) as b:
            np.testing.assert_array_equal(a.preview_rows, _top(y, 16))
            np.testing.assert_array_equal(a.candidate_rows, b.candidate_rows)
            np.testing.assert_array_equal(a.tails, b.tails)
            for aa, bb in zip(a.atoms, b.atoms):
                np.testing.assert_array_equal(aa.rows, bb.rows)
                if not aa.zero:
                    assert set(a.preview_rows) <= set(aa.rows)
                    vals, valid = aa.lookup(a.preview_rows)
                    assert np.all(valid)
                    np.testing.assert_array_equal(vals, bb.response[a.preview_rows])
                assert aa.nbytes == aa.rows.nbytes+aa.values.nbytes+80+16+(abs(aa.mass.numerator).bit_length()+7)//8+(aa.mass.denominator.bit_length()+7)//8
            for d in directories: ledger(d)
        old_rows = {key: atom.rows.copy() for key, atom in directories[0].cache.items()}
        old_solves = compiler.atom_solves
        next_seed = seed.copy(); next_seed[-1] -= .5; next_seed[100] += .5
        with directories[0].prepare(next_seed, sources, ks=(1,)) as family:
            assert compiler.atom_solves == old_solves
            for key, rows in old_rows.items():
                np.testing.assert_array_equal(directories[0].cache[key].rows, rows)
            assert directories[0].preview_selections == 2
            ledger(directories[0])
    finally:
        ranker.base_lu.graph.close()


def test_fine_large_cached_atoms_are_reused_and_mixed_actual_row_sizes_are_admitted():
    ranker, M, seed = graph(n=1025)
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M), atom_tolerance=1e-6)
        sources = [0, 1, 2, 3]
        directory = DomainDirectory(compiler, interaction_stencil([sources]), keep=512, preview_keep=16)
        with directory.prepare(seed, sources, ks=(1,)) as family:
            assert len(family.candidate_rows) >= 512
            ledger(directory)
        retained = {key: atom for key, atom in directory.cache.items()}
        directory.keep = 4; compiler.atom_tolerance = 1e-3
        before = compiler.atom_solves
        with directory.prepare(seed, sources, ks=(1,)) as family:
            assert compiler.atom_solves == before
            assert len(family.candidate_rows) >= 512
            assert all(directory.cache[key] is atom for key, atom in retained.items())
            assert all(atom.keep == 512 and atom.generation_tol == 1e-6 for atom in family.atoms)
            ledger(directory)
    finally:
        ranker.base_lu.graph.close()


def test_refinement_invalidations_are_not_evictions_and_pinned_entries_cannot_be_removed():
    ranker, M, seed = graph(n=17)
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        directory = DomainDirectory(compiler, interaction_stencil([[0, 1], [2]]), keep=3)
        with directory.prepare(seed, [0, 1], ks=(1,)):
            with pytest.raises(RuntimeError): directory.invalidate_for_refinement([(0, 0)])
        unrelated = directory.prepare(seed, [2], ks=(1,)); unrelated.close()
        survivor = directory.cache[(0, 2)]
        old_evictions, old_bytes = directory.evictions, directory._cache_bytes
        atom_bytes = directory.cache[(0, 0)].nbytes
        removed = directory.invalidate_for_refinement([(0, 0), (0, 0), (0, 99)])
        assert removed == ((0, 0),)
        assert directory.refinement_invalidations == 1 and directory.evictions == old_evictions
        assert directory._cache_bytes == old_bytes-atom_bytes
        assert directory.cache[(0, 2)] is survivor
        ledger(directory)
    finally:
        ranker.base_lu.graph.close()


def accuracy_graph(zero_weak=False):
    P = np.eye(4); M = np.zeros((4, 4))
    P[:, 0] = [.5, .5 if zero_weak else .25, 0. if zero_weak else .25, 0.]
    M[1, 0] = P[1, 0]
    threshold = Q(5, 9) if zero_weak else Q(4, 7)
    ratio = threshold-Q(1, 100000)
    b = 1/(1+ratio)
    seed = np.array([float(b), 0, 0, float(ratio*b)])
    return PRPackViewRank(sparse.csc_matrix(P), alpha=.5), M, seed


@pytest.mark.parametrize('zero_weak', [False, True])
def test_two_tiers_refine_only_unknown_cuts_and_charge_real_solve_counts(zero_weak):
    ranker, M, seed = accuracy_graph(zero_weak)
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        y, _, _ = compiler.base(seed)
        original_solve = ranker.base_lu.solve
        # Controlled approximate responses test the posterior contract, not a
        # hardware-specific convergence accident. Fine responses are genuine
        # native outputs; every stored coarse perturbation is remeasured.
        def noisy(rhs):
            answer = original_solve(rhs)
            if ranker.base_lu.tol >= 1e-4:
                answer = answer.copy(); answer[1] += .01
            return answer
        ranker.base_lu.solve = noisy
        directory = DomainDirectory(compiler, interaction_stencil([[0]]), preview_keep=0)
        controller = AdaptiveDomainOperator(directory, preview_keep=2)
        entry = (directory.keep, directory.preview_keep, compiler.atom_tolerance)
        answer = controller.execute(seed, [0], base_scores=y, ks=(1, 2), early_witness=False)
        assert len(answer['tiers']) == 2
        assert answer['tiers'][0]['requested_cuts'] == [1, 2]
        assert answer['tiers'][0]['answer']['cuts'][2]['status'] == 'STABLE'
        assert answer['tiers'][1]['requested_cuts'] == [1]
        assert answer['cuts'][1]['status'] == answer['cuts'][2]['status'] == 'STABLE'
        assert answer['cuts'][2] is answer['tiers'][0]['answer']['cuts'][2]
        assert answer['counters']['refinement_invalidations'] == (1 if zero_weak else 2)
        assert answer['counters']['atom_solves'] == (2 if zero_weak else 4)
        assert answer['counters']['backend_calls'] == answer['counters']['atom_solves']
        assert sum(t['counters']['atom_solves'] for t in answer['tiers']) == answer['counters']['atom_solves']
        assert answer['operator_seconds'] == pytest.approx(sum(answer['stage_totals'].values()))
        assert all(answer['stage_totals'][key] > 0 for key in ('prepare_seconds','certify_seconds','release_seconds'))
        assert (directory.keep, directory.preview_keep, compiler.atom_tolerance) == entry
        for weights in ((1., .25), (4., 2.), (2.5, 1.125)):
            order = direct(ranker.P, M, seed, [0], weights)
            for k in (1, 2): assert set(answer['cuts'][k]['selected_ids']) == set(order[:k])
        for atom in directory.cache.values():
            if not atom.zero: assert atom.generation_tol == 1e-6
        ledger(directory)
    finally:
        ranker.base_lu.graph.close()


def test_already_fine_zero_atoms_and_a_true_tie_do_not_repeat_a_tier():
    P = sparse.csc_matrix([[0., 0., 0.], [.5, 1., 0.], [.5, 0., 1.]])
    M = P.copy(); M.data[M.indptr[1]:] = 0.; M.eliminate_zeros()
    seed = np.array([1., 0., 0.])
    ranker = PRPackViewRank(P, alpha=.5)
    try:
        compiler = DomainCompiler(ranker, M, atom_tolerance=1e-6)
        directory = DomainDirectory(compiler, interaction_stencil([[0]]), keep=512)
        with directory.prepare(seed, [0], ks=(2,)):
            assert all(atom.zero for atom in directory.cache.values())
        answer = AdaptiveDomainOperator(directory).execute(seed, [0], ks=(2,), early_witness=False)
        assert len(answer['tiers']) == 1 and answer['cuts'][2]['status'] == 'UNKNOWN'
        assert answer['refinement_skipped_reason']
        assert answer['counters']['atom_solves'] == answer['counters']['refinement_invalidations'] == 0
        assert answer['counters']['hits'] == 2
        ledger(directory)
    finally:
        ranker.base_lu.graph.close()


def test_impossible_work_reservation_does_not_flush_unrelated_cache():
    ranker, M, seed = graph(n=17)
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        directory = DomainDirectory(compiler, interaction_stencil([[0], [1]]), keep=3)
        with directory.prepare(seed, [0], ks=(1,)): pass
        with directory.prepare(seed, [1], ks=(1,)) as family:
            keys = list(directory.cache); evictions = directory.evictions
            assert not family._reserve_work(directory.budget_bytes)
            assert list(directory.cache) == keys and directory.evictions == evictions
            ledger(directory)
    finally:
        ranker.base_lu.graph.close()


def test_chunk_shrinks_into_current_free_space_before_eviction(monkeypatch):
    import src.policy_domain_fast as fast
    ranker, M, seed = graph(n=17)
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        directory = DomainDirectory(compiler, interaction_stencil([[0], [1]]),
            keep=3, budget_bytes=1024**2)
        with directory.prepare(seed, [0], ks=(1,)): pass
        with directory.prepare(seed, [1], ks=(1,)) as family:
            before = list(directory.cache); evictions = directory.evictions
            original = fast.scratch_upper_bound
            def enlarged(domain, n_nodes, chunk_rows=512, n_cuts=3, n_candidates=None):
                return max(original(domain,n_nodes,chunk_rows,n_cuts,n_candidates),chunk_rows*4096)
            monkeypatch.setattr(fast, 'scratch_upper_bound', enlarged)
            answer = family.certify(ks=(1,), early_witness=False)
            assert answer['cuts'][1]['status'] == 'STABLE'
            assert answer['scan_chunk_rows'] == 128
            assert list(directory.cache) == before and directory.evictions == evictions
            ledger(directory)
    finally:
        ranker.base_lu.graph.close()


def test_dot_fallback_backend_is_optional_and_preserves_outward_results():
    ranker, M, seed = graph(n=17)
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        outputs = []
        for backend in ('outward', 'dot_fallback'):
            directory = DomainDirectory(compiler, interaction_stencil([[0,1,2,3]]),
                preview_keep=8, scan_backend=backend)
            with directory.prepare(seed,[0,1,2,3],ks=(1,)) as family:
                answer = family.certify(ks=(1,),early_witness=False)
                outputs.append(answer['cuts'][1])
                assert answer['scan_backend'] == backend
                ledger(directory)
        assert outputs[0]['status'] == outputs[1]['status'] == 'STABLE'
        assert outputs[0]['selected_ids'] == outputs[1]['selected_ids']
    finally:
        ranker.base_lu.graph.close()


def test_audit_sink_writes_each_live_tier_and_scalar_replay_preserves_the_contract(tmp_path):
    from src.policy_domain_polynomial import compile_domain, certify
    ranker, M, seed = accuracy_graph()
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        y, _, _ = compiler.base(seed)
        original = ranker.base_lu.solve
        def noisy(rhs):
            value = original(rhs)
            if ranker.base_lu.tol >= 1e-4:
                value = value.copy(); value[1] += .01
            return value
        ranker.base_lu.solve = noisy
        directory = DomainDirectory(compiler, interaction_stencil([[0]]))
        contracts = []
        def audit(family, tier):
            assert not family.closed and family in directory.live
            data = family.certificate_inputs()
            path = tmp_path / ('tier%d.npz' % tier)
            np.savez(path, y=data['y'], candidate_rows=data['candidate_rows'],
                values=data['retained_values'], mask=data['retained_mask'],
                tails=data['tail_bounds'], H_m=data['H_m'], H_w=data['H_w'],
                sources=data['source_ids'])
            contracts.append({key: value for key, value in data.items() if key in
                ('mu','nu','y_sources','source_ids','box','selected_sets','kappa',
                 'base_residual','atom_residuals','round_pad')})
            # No stored n-by-q basis is returned into the controller.
            return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        answer = AdaptiveDomainOperator(directory).execute(seed, [0], base_scores=y,
            ks=(1,2), early_witness=False, audit_sink=audit)
        assert len(answer['tiers']) == len(contracts) == 2
        assert answer['audit_seconds'] > 0
        assert answer['audit_seconds'] == sum(t['audit_seconds'] for t in answer['tiers'])
        assert answer['operator_seconds']+answer['audit_seconds'] == pytest.approx(answer['raw_operator_wall_seconds'])
        assert sum(answer['stage_totals'].values()) == pytest.approx(answer['operator_seconds'])
        assert answer['actual_route'] == 'adaptive_sparse'
        assert answer['point_checks'] == answer['verified_witnesses'] == 0
        assert not directory.live and not directory.pins
        for record, contract in zip(answer['tiers'], contracts):
            assert record['memory_after_release']['live_family_bytes'] == 0
            assert record['memory_after_release']['pending_bytes'] == 0
            assert record['memory_after_release']['total_private_retained_numeric_bytes'] <= directory.budget_bytes
            path = tmp_path / ('tier%d.npz' % record['tier'])
            assert record['audit']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as data:
                domain = compile_domain(data['H_m'], data['H_w'], contract['mu'],
                    contract['nu'], contract['y_sources'], contract['box'], data['sources'])
                for k, selected in contract['selected_sets'].items():
                    result = certify(domain, data['y'], data['candidate_rows'],
                        data['values'], data['mask'], data['tails'], selected,
                        kappa=contract['kappa'], base_residual=contract['base_residual'],
                        atom_residuals=contract['atom_residuals'], round_pad=0,
                        subdivide_depth=2)
                    assert result['status'] == record['answer']['cuts'][k]['status']
        ledger(directory)
    finally:
        ranker.base_lu.graph.close()


def test_audit_sink_cannot_return_a_retained_basis_and_exception_releases_pins():
    ranker, M, seed = graph(n=17)
    try:
        compiler = DomainCompiler(ranker, sparse.csc_matrix(M))
        directory = DomainDirectory(compiler, interaction_stencil([[0]]))
        entry = (directory.keep, directory.preview_keep, compiler.atom_tolerance)
        with pytest.raises(ValueError, match='file/hash metadata'):
            AdaptiveDomainOperator(directory).execute(seed, [0], ks=(1,),
                early_witness=False, audit_sink=lambda family,tier: {'basis':family.values})
        assert not directory.live and not directory.pins
        assert (directory.keep, directory.preview_keep, compiler.atom_tolerance) == entry
        ledger(directory)
    finally:
        ranker.base_lu.graph.close()
