from fractions import Fraction as Q
from itertools import product

import numpy as np
import pytest
from scipy import sparse

from scripts.natural_domain_adapter import source_columns_exact
from src.policy_domain_source import source_columns_fast, point_columns_fast, sum_dyadics, exact


def graph(n=11, normalized=False):
    rng = np.random.default_rng(310311)
    P = rng.random((n, n))+.03
    if normalized:
        P /= P.sum(axis=0)
    else:
        P *= 19.7
    M = P*rng.choice([0., .25, .5, 1.], size=P.shape)
    np.fill_diagonal(M, 0)
    return sparse.csc_matrix(P), sparse.csc_matrix(M)


def truth_column(partition, weights):
    m, w = map(exact, weights)
    raw = [m*a+w*b+c for a, b, c in zip(partition['mention'], partition['weak'], partition['self_loop'])]
    total = sum(raw, Q(0))
    return [value/total for value in raw]


@pytest.mark.parametrize('normalized', [False, True])
@pytest.mark.parametrize('n', [5, 11])
def test_fast_atom_rhs_mass_and_error_enclosure_match_exact_fraction_oracle(n, normalized):
    P, M = graph(n, normalized)
    for source in (0, n-1):
        old = source_columns_exact(P, M, [source], Q(1, 2))[0]
        fast = source_columns_fast(P, M, [source], Q(1, 2))[0]
        assert old['raw_total'] == fast['raw_total'] and old['masses'] == fast['masses']
        assert fast['raw_partition'] is None
        raw = old['raw_partition']
        for group, name in enumerate(('mention', 'weak')):
            expected = [-Q(1, 2)*(g/old['raw_total']-old['masses'][group]*p/old['raw_total'])
                for g, p in zip(raw[name], raw['original'])]
            error = sum(abs(value-exact(fast['atom_rhs'][group]['rhs'][row]))
                for value, row in zip(expected, raw['rows']))
            assert error <= fast['atom_rhs'][group]['rhs_conversion_error']


@pytest.mark.parametrize('normalized', [False, True])
def test_point_columns_fixed_grid_center_and_cancellation_preserve_mathematical_target(normalized):
    P, M = graph(normalized=normalized)
    sources = [0, 3]
    old = source_columns_exact(P, M, sources, Q(1, 2))
    grid = list(product((1., 2., 3., 4.), (.25, .5, 1., 2.)))
    points = grid+[(2.5, 1.125), (.125, 9.5), (9.5, .125), (1., 1.)]
    for weights in points:
        compiled = point_columns_fast(P, M, sources, weights)
        for before, after in zip(old, compiled):
            truth = truth_column(before['raw_partition'], weights)
            errors = [abs(value-exact(stored)) for value, stored in zip(truth, after['values'])]
            assert sum(errors, Q(0)) <= after['column_conversion_error']
            assert np.all(after['values'] >= 0)
            assert after['raw_total'] == before['raw_total'] and after['masses'] == before['masses']
            assert 'atom_rhs' not in after


def test_high_degree_rhs_and_point_columns_use_enclosures_not_per_entry_claims():
    n = 513
    rows = np.r_[np.arange(n), np.arange(1, n)]
    columns = np.r_[np.zeros(n, dtype=int), np.arange(1, n)]
    values = np.r_[np.full(n, 1/n), np.ones(n-1)]
    P = sparse.csc_matrix((values, (rows, columns)), shape=(n, n))
    mr = np.arange(1, n, 2)
    M = sparse.csc_matrix((np.full(len(mr), 1/n), (mr, np.zeros(len(mr), dtype=int))), shape=(n, n))
    old = source_columns_exact(P, M, [0], Q(1, 2))[0]
    for weights in ((1., .25), (4., 2.), (.125, 9.5), (2.5, 1.125)):
        compiled = point_columns_fast(P, M, [0], weights)[0]
        truth = truth_column(old['raw_partition'], weights)
        error = sum(abs(value-exact(stored)) for value, stored in zip(truth, compiled['values']))
        assert error <= compiled['column_conversion_error']


def test_symbolic_mass_endpoints_are_exact_zero_directions():
    P = sparse.csc_matrix([[0., 0., 0.], [.25, 1., 0.], [.75, 0., 1.]])
    zero = sparse.csc_matrix(P.shape, dtype=np.float64)
    full = P.copy()
    full.data[full.indptr[1]:] = 0.; full.eliminate_zeros()
    for M in (zero, full):
        compiled = source_columns_fast(P, M, [0], .5)[0]
        for atom in compiled['atom_rhs']:
            assert not np.any(atom['rhs']) and atom['rhs_conversion_error'] == 0


def test_dyadic_superaccumulator_matches_exact_fraction_for_spread_signed_exponents():
    values = [(-1 if i % 2 else 1)*float(2**power)*.3125 for i, power in enumerate(range(-1000, 900, 31))]
    assert sum_dyadics(values) == sum(map(exact, values), Q(0))


def test_invalid_source_partitions_and_nonpositive_point_weights_are_rejected():
    P, M = graph()
    bad = M.copy(); bad[0, 0] = .1
    with pytest.raises(ValueError): point_columns_fast(P, bad, [0], (1., 1.))
    bad = M.copy(); bad[1, 0] = 2*P[1, 0]
    with pytest.raises(ValueError): source_columns_fast(P, bad, [0], .5)
    with pytest.raises(ValueError): point_columns_fast(P, M, [0], (0., 1.))
