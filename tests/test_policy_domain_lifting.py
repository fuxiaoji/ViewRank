"""Independent tiny actual-view checks for the source-wise lifted certificate."""
from fractions import Fraction as Q
from itertools import product
import sys

import numpy as np
import pytest

from src.policy_domain_lifting import (compile_lifting, certify_lifting,
    certify_lifting_many, certify_rank1_corners, certify_many, scratch_upper_bound,
    compile_reservation_upper)
import src.policy_domain_lifting as lifting


def solve_q(matrix, rhs):
    # Independent Gauss-Jordan reference, not the producer's elimination.
    a = [list(row)+[v] for row, v in zip(matrix, rhs)]
    n = len(rhs)
    for j in range(n):
        p = next(i for i in range(j, n) if a[i][j])
        a[j], a[p] = a[p], a[j]
        divisor = a[j][j]
        a[j] = [x/divisor for x in a[j]]
        for i in range(n):
            if i != j:
                value = a[i][j]
                a[i] = [x-value*y for x, y in zip(a[i], a[j])]
    return tuple(row[-1] for row in a)


def prepare(P, G, sources, seed, box, *, retain_exact_coefficients=False):
    n, r, alpha = len(seed), len(sources), Q(1, 2)
    A = [[Q(i == j)-alpha*P[i][j] for j in range(n)] for i in range(n)]
    b = [(1-alpha)*v for v in seed]
    exact_y = solve_q(A, b)
    y = np.array(exact_y, dtype=float)
    groups = [G, [[P[i][j]-G[i][j]-(P[i][j] if i == j else 0)
                   for j in range(n)] for i in range(n)]]
    masses = [[sum((group[i][s] for i in range(n)), Q(0)) for s in sources] for group in groups]
    responses, residuals = [], []
    for group, mus in zip(groups, masses):
        for j, s in enumerate(sources):
            rhs = [-alpha*(group[i][s]-mus[j]*P[i][s]) for i in range(n)]
            response = np.array(solve_q(A, rhs), dtype=float)
            responses.append(response)
            residuals.append(sum((abs(rhs[i]-sum((A[i][k]*Q.from_float(response[k])
                for k in range(n)), Q(0))) for i in range(n)), Q(0)))
    H = np.column_stack(responses)
    rho_y = sum((abs(b[i]-sum((A[i][k]*Q.from_float(y[k]) for k in range(n)), Q(0))) for i in range(n)), Q(0))
    domain = compile_lifting(H[sources, :r], H[sources, r:], *masses, y[sources], box, sources,
        retain_exact_coefficients=retain_exact_coefficients)
    contract = dict(kappa=2, base_residual=rho_y, atom_residuals=residuals)
    return domain, y, H, contract


def actual_scores(P, G, sources, seed, weights):
    n = len(seed)
    V = [list(row) for row in P]
    for j, source in enumerate(sources):
        m, w = weights[j]
        col = [P[i][source] if i == source else m*G[i][source]+w*(P[i][source]-G[i][source])
               for i in range(n)]
        total = sum(col, Q(0))
        for i in range(n):
            V[i][source] = col[i]/total
    A = [[Q(i == j)-Q(1, 2)*V[i][j] for j in range(n)] for i in range(n)]
    return solve_q(A, [v/2 for v in seed])


def dense_certificate(domain, y, H, selected, contract, **kwargs):
    return certify_lifting(domain, y, range(len(y)), H, np.ones(H.shape, bool),
        [0]*H.shape[1], selected, **contract, **kwargs)


def stable_rank1():
    # A self=.5, mention U=.25, weak V=.25; U,V,C absorbing.
    P = [[Q(0) for _ in range(4)] for _ in range(4)]
    for i, value in enumerate((Q(1, 2), Q(1, 4), Q(1, 4), Q(0))):
        P[i][0] = value
    for s in (1, 2, 3):
        P[s][s] = Q(1)
    G = [[Q(0) for _ in range(4)] for _ in range(4)]
    G[1][0] = Q(1, 4)
    seed = [Q(500, 797), Q(0), Q(0), Q(297, 797)]
    return P, G, [0], seed, ((1, 2), (1, 2))


def symmetric_two_source():
    # Shared weights cancel exactly, but independent source choices do not.
    P = [[Q(0) for _ in range(5)] for _ in range(5)]
    G = [[Q(0) for _ in range(5)] for _ in range(5)]
    for source in (0, 1):
        P[2][source] = P[3][source] = Q(1, 2)
    G[2][0] = G[3][1] = Q(1, 2)
    for s in (2, 3, 4):
        P[s][s] = Q(1)
    return P, G, [0, 1], [Q(9, 20), Q(9, 20), Q(0), Q(0), Q(1, 10)], ((Q(1, 10), 10), (Q(1, 10), 10))


def test_rank1_relaxation_is_exact_and_all_true_corner_views_agree():
    P, G, sources, seed, box = stable_rank1()
    domain, y, H, contract = prepare(P, G, sources, seed, box)
    result = dense_certificate(domain, y, H, [0], contract, row_chunk=1)
    assert result['status'] == 'STABLE'
    assert result['local_corner_policies'] == 4
    assert result['rank1_complete_corner_family']
    assert result['graph_solves'] == 0
    for m, w in product((Q(1), Q(3, 2), Q(2)), repeat=2):
        scores = actual_scores(P, G, sources, seed, [(m, w)])
        assert scores[0] > max(scores[1:])


def test_local_relaxation_failure_is_unknown_not_a_shared_parameter_witness():
    P, G, sources, seed, box = symmetric_two_source()
    domain, y, H, contract = prepare(P, G, sources, seed, box)
    result = dense_certificate(domain, y, H, [0, 1, 2, 3], contract)
    assert result['status'] == 'UNKNOWN'
    assert 'first_unseparated_local_policy' in result
    assert domain.policy_count == 9  # diagonal corners are the same column
    assert domain.logical_policy_count == 16
    # Every shared interior weight keeps the proposed set strictly above C.
    for m, w in product((Q(1, 10), Q(1), Q(10)), repeat=2):
        scores = actual_scores(P, G, sources, seed, [(m, w)]*2)
        assert min(scores[:4]) > scores[4]


def test_all_policy_coefficients_match_independent_actual_local_views():
    P, G, sources, seed, box = symmetric_two_source()
    domain, y, H, contract = prepare(P, G, sources, seed, box, retain_exact_coefficients=True)
    for policy, coefficients in zip(domain.policy_choices, domain.coefficients):
        actual = actual_scores(P, G, sources, seed, [domain.corners[c] for c in policy])
        reconstructed = [Q.from_float(y[i])-sum((Q.from_float(H[i, j])*c for j, c in enumerate(coefficients)), Q(0))
                         for i in range(len(y))]
        error = contract['kappa']*(contract['base_residual']+sum((abs(c)*rho for c, rho in zip(coefficients, contract['atom_residuals'])), Q(0)))
        assert all(abs(a-b) <= error for a, b in zip(actual, reconstructed))


def test_sparse_tails_cover_all_nodes_and_masked_payload_is_not_read():
    P, G, sources, seed, box = stable_rank1()
    domain, y, H, contract = prepare(P, G, sources, seed, box)
    rows = [0, 1, 2, 3]
    mask = np.ones(H.shape, bool)
    mask[1:3] = False
    tails = [np.max(abs(H[1:3, j])) for j in range(H.shape[1])]
    garbage = H.copy()
    garbage[~mask] = 1e100
    a = certify_lifting(domain, y, rows, H, mask, tails, [0], **contract)
    b = certify_lifting(domain, y, rows, garbage, mask, tails, [0], **contract)
    assert a['status'] == b['status']
    assert a['minimum_relaxed_policy_gap_lower_bound'] == b['minimum_relaxed_policy_gap_lower_bound']
    inflated = certify_lifting(domain, y, [0], H[:1], np.ones((1, 2), bool), [10, 10], [0], **contract)
    assert inflated['status'] == 'UNKNOWN'
    assert inflated['evaluated_candidate_rows'] == 0  # outside bound already fails


def test_multi_cut_stream_reuses_prepared_cores_and_retains_unknown_cuts():
    P, G, sources, seed, box = stable_rank1()
    domain, y, H, contract = prepare(P, G, sources, seed, box)
    result = certify_lifting_many(domain, y, range(4), H, np.ones(H.shape, bool), [0, 0],
        {1: [0], 2: [0, 3]}, **contract, row_chunk=2)
    assert result['results'][1]['status'] == 'STABLE'
    assert result['results'][2]['status'] == 'STABLE'
    assert result['prepared_core_solve_count'] == 4
    assert result['core_solves_during_certification'] == 0
    assert not domain.float_coefficients.flags.writeable
    assert domain.numeric_payload_bytes > domain.float_coefficients.nbytes
    assert domain.python_retained_bytes >= domain.float_coefficients.nbytes


def test_residual_allowances_can_prevent_a_false_stable_result():
    P, G, sources, seed, box = stable_rank1()
    domain, y, H, contract = prepare(P, G, sources, seed, box)
    contract['base_residual'] = Q(1)
    result = dense_certificate(domain, y, H, [0], contract)
    assert result['status'] == 'UNKNOWN'


def test_singular_approximate_core_is_unknown_without_assuming_true_positivity():
    domain = compile_lifting([[-1.5]], [[0]], [Q(1, 2)], [Q(1, 4)], [.5], ((1, 2), (1, 2)), [0])
    assert domain.singular_policies
    result = certify_lifting(domain, [.5, .5], [0, 1], [[-1.5, 0], [0, 0]],
        [[True, True], [True, True]], [0, 0], [0], kappa=2,
        base_residual=0, atom_residuals=[3, 0])
    assert result['status'] == 'UNKNOWN'
    assert result['evaluated_candidate_rows'] == 0


def test_four_global_corners_agree_but_the_interior_is_not_certified_stable():
    # A mention->B/weak->U; B mention->U/weak->V.
    P = [[Q(0) for _ in range(5)] for _ in range(5)]
    G = [[Q(0) for _ in range(5)] for _ in range(5)]
    P[1][0] = P[2][0] = P[2][1] = P[3][1] = Q(1, 2)
    G[1][0] = G[2][1] = Q(1, 2)
    for s in (2, 3, 4):
        P[s][s] = Q(1)
    seed = [Q(19, 20), Q(0), Q(0), Q(0), Q(1, 20)]
    sources = [0, 1]
    box = ((Q(1, 4), 4), (Q(99, 100), Q(101, 100)))
    selected = {0, 1, 2, 4}
    for weights in product(box[0], box[1]):
        scores = actual_scores(P, G, sources, seed, [weights]*2)
        assert set(sorted(range(5), key=lambda i: -scores[i])[:4]) == selected
    interior = actual_scores(P, G, sources, seed, [(Q(1), Q(1))]*2)
    assert set(sorted(range(5), key=lambda i: -interior[i])[:4]) != selected
    domain, y, H, contract = prepare(P, G, sources, seed, box)
    result = dense_certificate(domain, y, H, sorted(selected), contract)
    assert result['status'] == 'UNKNOWN'


def corner_records():
    P, G, sources, seed, box = stable_rank1()
    records = []
    for weights in product((Q(1), Q(2)), repeat=2):
        scores = actual_scores(P, G, sources, seed, [weights])
        order = sorted(range(4), key=lambda i: (-scores[i], i))
        records.append(dict(weights=weights, sets={1: order[:1]}, strict_flags={1: True},
            score_error_bound=Q(0), raw_gaps={1: scores[order[0]]-scores[order[1]]}))
    return records, box


def test_rank1_actual_corner_shortcut_needs_no_response_atoms():
    records, box = corner_records()
    result = certify_rank1_corners(records, box)
    assert result['results'][1]['status'] == 'STABLE'
    assert result['results'][1]['prepared_response_atoms'] == 0
    assert result['complete_rank1_shortcut']


def test_rank1_shortcut_rejects_unproven_flags_and_missing_or_nonsingleton_corners():
    records, box = corner_records()
    records[0]['score_error_bound'] = Q(1)
    result = certify_rank1_corners(records, box)
    assert result['results'][1]['status'] == 'UNKNOWN'
    with pytest.raises(ValueError):
        certify_rank1_corners(records[:3], box)
    with pytest.raises(ValueError):
        certify_rank1_corners(records, box, source_count=2)


def test_source_boundary_payload_must_match_compiled_responses():
    P, G, sources, seed, box = stable_rank1()
    domain, y, H, contract = prepare(P, G, sources, seed, box)
    altered = H.copy()
    altered[0, 0] += .001
    with pytest.raises(ValueError, match='source response'):
        dense_certificate(domain, y, altered, [0], contract)


def test_dense_borrowed_columns_match_matrix_without_retaining_exact_policy_fractions():
    P, G, sources, seed, box = stable_rank1()
    domain, y, H, contract = prepare(P, G, sources, seed, box)
    assert domain.coefficients == ()
    expected = dense_certificate(domain, y, H, [0], contract)
    result = certify_many(domain, y, np.arange(4), None, None, [0, 0], {1: [0]},
        retained_columns=[H[:, 0], H[:, 1]], chunk_rows=1, **contract)
    assert result['cuts'][1]['status'] == expected['status'] == 'STABLE'
    assert result['workspace_peak_bytes'] == scratch_upper_bound(domain, 4, 1, 1, 4)
    assert result['shared_scan']['graph_solves'] == 0
    assert not result['shared_scan']['exact_policy_coefficients_retained']


def test_subdivision_reduces_dependency_relaxation_without_dropping_unknown_boxes():
    P, G, sources, _, _ = symmetric_two_source()
    seed = [Q(17, 40), Q(17, 40), Q(0), Q(0), Q(3, 20)]
    box = ((1, 2), (1, 2))
    domain, y, H, contract = prepare(P, G, sources, seed, box)
    arguments = (domain, y, np.arange(5), H, np.ones(H.shape, bool), [0]*4, {4: [0, 1, 2, 3]})
    parent = certify_many(*arguments, subdivide_depth=0, **contract)
    divided = certify_many(*arguments, subdivide_depth=1, **contract)
    assert parent['cuts'][4]['status'] == 'UNKNOWN'
    assert divided['cuts'][4]['status'] == 'STABLE'
    assert divided['cuts'][4]['leaf_boxes'] == 2
    assert divided['shared_scan']['control_boxes'] == 3
    assert divided['shared_scan']['child_core_solve_count'] == 32
    assert divided['workspace_peak_bytes'] == scratch_upper_bound(domain, 5, n_cuts=1,
        n_candidates=5, subdivide_depth=1)

    # A genuinely broad dependency relaxation still leaves all unknown leaves.
    P, G, sources, seed, box = symmetric_two_source()
    domain, y, H, contract = prepare(P, G, sources, seed, box)
    broad = certify_many(domain, y, range(5), H, np.ones(H.shape, bool), [0]*4,
        {4: [0, 1, 2, 3]}, subdivide_depth=1, **contract)
    assert broad['cuts'][4]['status'] == 'UNKNOWN'
    assert len(broad['cuts'][4]['leaves']) == 2
    assert any(leaf['status'] == 'UNKNOWN' for leaf in broad['cuts'][4]['leaves'])


def test_nonfinite_interval_work_cannot_produce_stable():
    domain = compile_lifting([[0]], [[0]], [Q(1, 4)], [Q(1, 4)], [1e308], ((1, 2), (1, 2)), [0])
    with np.errstate(over='ignore', invalid='ignore'):
        result = certify_many(domain, [1e308, 0.], [0, 1], [[0., 0.], [1e308, 1e308]],
            [[True, True], [True, True]], [0, 0], {1: [0]},
            kappa=2, base_residual=0, atom_residuals=[0, 0])
    assert result['cuts'][1]['status'] == 'UNKNOWN'


def test_raw_background_residual_guard_prevents_wrong_canonical_target_proof():
    P, G, sources, seed, box = stable_rank1()
    domain, y, H, contract = prepare(P, G, sources, seed, box)
    # The basis belongs to column-normalized canonical P. The requested target
    # instead leaves the non-source raw C column unchanged at mass 1+epsilon.
    raw = [list(row) for row in P]
    epsilon = Q(1, 32)
    raw[3][3] += epsilon
    actual = actual_scores(raw, G, sources, seed, [(Q(2), Q(2))])
    assert actual[3] > actual[0]
    # Omitting this target correction would certify the different canonical
    # problem. The actual-target residual bound must make the answer UNKNOWN.
    guarded = dict(contract)
    guarded['base_residual'] += Q(1, 2)*epsilon*abs(Q.from_float(y[3]))
    guarded['atom_residuals'] = [rho+Q(1, 2)*epsilon*abs(Q.from_float(H[3, j]))
                               for j, rho in enumerate(contract['atom_residuals'])]
    guarded['kappa'] = 1/(1-Q(1, 2)*(1+epsilon))
    assert dense_certificate(domain, y, H, [0], guarded)['status'] == 'UNKNOWN'
    assert dense_certificate(domain, y, H, [0], contract)['status'] == 'STABLE'


def four_source_core(*, tiny=False):
    r = 4
    hm = [[Q(1, 32) if i == j else Q(i+j+1, 1024) for j in range(r)] for i in range(r)]
    hw = [[Q(-1, 128) if i == j else Q(i-j, 2048) for j in range(r)] for i in range(r)]
    if tiny:
        # Stored binary64's longest denominator, plus a large numerator.
        hm[0][1] = Q(1, 2**1074)
        hw[2][3] = Q(2**900+1, 2**1000)
    return hm, hw, [Q(3, 11)]*r, [Q(2, 13)]*r, [Q(i+1, 32) for i in range(r)], \
        ((Q(1, 4), Q(4)), (Q(1, 2), Q(3))), list(range(r))


def test_source_corner_columns_are_reused_and_streamed_coefficients_are_unchanged(monkeypatch):
    arguments = four_source_core()
    hm, hw, mu, nu, ys, box, sources = arguments
    observed = [[] for _ in sources]
    expected = []
    original = lifting._solve_exact
    choices = tuple(product(range(4), repeat=4))
    corners = tuple(product(*box))

    def inspect(matrix, rhs):
        policy = choices[len(expected)]
        direct = []
        dm, dw = zip(*((corners[c][0]-1, corners[c][1]-1) for c in policy))
        for i in range(4):
            direct.append([(1+dm[j]*mu[j]+dw[j]*nu[j] if i == j else 0)
                           +dm[j]*hm[i][j]+dw[j]*hw[i][j] for j in range(4)])
        assert matrix == direct
        t = solve_q(direct, rhs)
        expected.append(tuple(dm[j]*t[j] for j in range(4))+tuple(dw[j]*t[j] for j in range(4)))
        for j in range(4):
            observed[j].append(tuple(id(matrix[i][j]) for i in range(4)))
        return original(matrix, rhs)

    monkeypatch.setattr(lifting, '_solve_exact', inspect)
    domain = compile_lifting(*arguments)
    assert domain.policy_choices == choices
    assert domain.policy_count == 256
    assert domain.coefficients == ()
    # At most four exact column objects/source, rather than 256 rebuilt columns.
    for column_ids in observed:
        assert len(set(column_ids)) == 4
        assert all(column_ids.count(value) == 64 for value in set(column_ids))
    expected_exact = expected
    expected = np.array(expected_exact, dtype=float).T
    np.testing.assert_array_equal(domain.float_coefficients, expected)
    # Conversion and absolute enclosures still cover each exact coefficient.
    for policy, values in enumerate(expected_exact):
        for j, value in enumerate(values):
            rounded = Q.from_float(domain.float_coefficients[j, policy])
            assert abs(value-rounded) <= Q.from_float(domain.coefficient_conversion_upper[j, policy])
            assert abs(value) <= Q.from_float(domain.absolute_coefficients_upper[j, policy])


@pytest.mark.parametrize('tiny', [False, True])
def test_compile_reservation_covers_retained_columns_and_live_single_core(tiny):
    arguments = four_source_core(tiny=tiny)
    hm, hw, mu, nu, ys, box, sources = arguments
    r = len(sources)
    largest_exact_locals = 0

    def numeric(value):
        if isinstance(value, Q):
            return 16+(abs(value.numerator).bit_length()+7)//8+(value.denominator.bit_length()+7)//8
        if isinstance(value, (list, tuple)):
            return sum(numeric(v) for v in value)
        return 8 if isinstance(value, int) else 0

    # Observe all live exact-solver locals after every Python statement. Count
    # shared references repeatedly, so this does not hide input/matrix aliasing.
    def trace(frame, event, arg):
        nonlocal largest_exact_locals
        if frame.f_code is lifting._solve_exact.__code__ and event == 'line':
            largest_exact_locals = max(largest_exact_locals,
                sum(numeric(frame.f_locals.get(key)) for key in ('matrix', 'rhs', 'a', 'x', 'scale')))
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        domain = compile_lifting(*arguments)
    finally:
        sys.settrace(previous)
    assert largest_exact_locals > 0
    columns = []
    for j in range(r):
        for m, w in product(*box):
            dm, dw = m-1, w-1
            z = 1+dm*mu[j]+dw*nu[j]
            columns.append((dm, dw, tuple((z if i == j else 0)+dm*hm[i][j]+dw*hw[i][j]
                                         for i in range(r))))
    reconstructed = domain.numeric_payload_bytes+numeric(columns)+largest_exact_locals
    reserved = compile_reservation_upper(*arguments)
    assert reserved >= reconstructed
    # Even the longest binary64 denominator must not create a fictitious
    # multi-MiB batch of 256*8 retained exact policy coefficients.
    assert domain.numeric_payload_bytes < reserved < 1024**2


def test_compile_reservation_deduplicates_actions_without_solving_a_policy(monkeypatch):
    P, G, sources, seed, box = symmetric_two_source()
    domain, _, _, _ = prepare(P, G, sources, seed, box)
    def no_solve(*args, **kwargs):
        raise AssertionError('Reservation must not solve any policy core.')
    monkeypatch.setattr(lifting, '_solve_exact', no_solve)
    reserved = compile_reservation_upper(domain.H_m, domain.H_w, domain.mu,
        domain.nu, domain.y_sources, domain.box, domain.source_ids)
    assert domain.policy_count == 9
    assert reserved >= domain.numeric_payload_bytes
