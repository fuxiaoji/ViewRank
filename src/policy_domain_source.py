"""Exact-mass and enclosed vectorized source compilation.

The mathematical raw inputs and normalized atom RHS are unchanged. Positive
dyadic sums use integer superaccumulation; vector arithmetic only rounds an
RHS once, accompanied by a proved exact-scalar/gamma/underflow L1 enclosure.
Both atom RHS and normalized point columns retain the original mathematical
raw-input target. This is shared infrastructure, not a new core contribution.
"""
from fractions import Fraction as Q
import math

import numpy as np
from scipy import sparse


UNIT = Q(1, 2**53)
NORMAL_TINY = Q(1, 2**1022)


def exact(value):
    return value if isinstance(value, Q) else Q.from_float(float(value))


def sum_dyadics(values):
    """Exact sum of stored floats, without Fraction/gcd work per entry."""
    numerator, exponent = 0, 0
    for value in values:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Nonfinite dyadic input.')
        top, bottom = value.as_integer_ratio()
        power = bottom.bit_length()-1
        if power > exponent:
            numerator <<= power-exponent
            exponent = power
        numerator += top << (exponent-power)
    return Q(numerator, 1 << exponent)


def _normal_array(values, label):
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(label+' must be finite.')
    if np.any((values != 0)&(np.abs(values) < np.finfo(np.float64).tiny)):
        raise ValueError(label+' has nonzero subnormal operands; DAZ can amplify them.')
    return values


def _coefficient(value):
    """Round once; explicitly flush subnormal operands and charge the loss."""
    value = exact(value)
    rounded = float(value)
    if not math.isfinite(rounded):
        raise ValueError('Coefficient conversion overflow.')
    flushed = 0 < abs(rounded) < np.finfo(np.float64).tiny
    if flushed:
        rounded = 0.
    return rounded, abs(value-exact(rounded)), flushed


def _linear_combination(rows, arrays, coefficients, exact_l1_masses, ops, self_position=None):
    rounded = [_coefficient(value) for value in coefficients]
    scalar_conversion = sum((error*mass for (_, error, _), mass in zip(rounded, exact_l1_masses)), Q(0))
    term_norm = sum((abs(exact(value))*mass for (value, _, _), mass in zip(rounded, exact_l1_masses)), Q(0))
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        # Mention: two vector products and one add (gamma_3).
        # Weak: the same two products/add plus one scalar product/add at self
        # (gamma_5 globally is conservative for every other row).
        local = rounded[0][0]*arrays[0]+rounded[1][0]*arrays[1]
        if len(rounded) == 3 and self_position is not None:
            local[self_position] += rounded[2][0]*arrays[2]
    if not np.all(np.isfinite(local)):
        raise ValueError('RHS expression overflowed.')
    path = int(ops)
    gamma = path*UNIT/(1-path*UNIT)
    rounding = gamma*term_norm
    underflow = 4*path*len(rows)*NORMAL_TINY/(1-path*UNIT)
    # The existing residual guard rejects subnormal RHS operands. Flushing
    # them explicitly removes DAZ ambiguity and is covered as a final step.
    subnormal = (local != 0)&(np.abs(local) < np.finfo(np.float64).tiny)
    final_flush = int(np.count_nonzero(subnormal))*NORMAL_TINY
    local[subnormal] = 0.
    error = scalar_conversion+rounding+underflow+final_flush
    return dict(values=local, conversion_error=error,
        enclosure=dict(coefficient_exact_to_float_l1=scalar_conversion,
            floating_expression_rounding_l1=rounding,
            intermediate_underflow_l1=underflow, final_subnormal_flush_l1=final_flush,
            gamma=gamma, expression_ops_per_entry=path,
            coefficient_subnormal_flushes=sum(item[2] for item in rounded),
            interpretation='All enclosure arithmetic is exact Fraction; no observed near-zero residual is substituted for an upper bound.'))


def _rhs(n, rows, arrays, coefficients, exact_l1_masses, ops, self_position=None):
    local = _linear_combination(rows, arrays, coefficients, exact_l1_masses, ops, self_position)
    result = np.zeros(n)
    result[rows] = local['values']
    return dict(rhs=result, rhs_conversion_error=local['conversion_error'], enclosure=local['enclosure'])


def _source_data(P, M, source):
    source = int(source)
    lo, hi = P.indptr[source:source+2]
    ml, mh = M.indptr[source:source+2]
    rows = P.indices[lo:hi]
    p = _normal_array(P.data[lo:hi], 'P source column')
    g_values = _normal_array(M.data[ml:mh], 'mention source column')
    if np.any(p < 0) or np.any(g_values < 0):
        raise ValueError('Raw weights must be nonnegative.')
    positions = np.searchsorted(rows, M.indices[ml:mh])
    if np.any(positions >= len(rows)) or np.any(rows[positions] != M.indices[ml:mh]):
        raise ValueError('Mention contains a non-base coordinate.')
    g = np.zeros(len(rows)); g[positions] = g_values
    if np.any(g > p):
        raise ValueError('Mention exceeds raw P.')
    self_position = int(np.searchsorted(rows, source))
    if self_position >= len(rows) or rows[self_position] != source:
        self_position = None
    if self_position is not None and g[self_position] != 0:
        raise ValueError('Mention must exclude self loops.')
    self_value = 0. if self_position is None else float(p[self_position])
    c, msum, pss = sum_dyadics(p), sum_dyadics(g_values), exact(self_value)
    wsum = c-msum-pss
    if c <= 0 or min(msum, wsum, pss) < 0:
        raise ValueError('Invalid positive source partition.')
    return rows, p, g, self_position, self_value, c, msum, pss, wsum


def _check_matrices(P, M):
    if not sparse.isspmatrix_csc(P) or not sparse.isspmatrix_csc(M) or P.shape != M.shape:
        raise ValueError('P/M must be aligned CSC matrices.')
    if P.dtype != np.float64 or M.dtype != np.float64 or not P.has_canonical_format or not M.has_canonical_format:
        raise ValueError('Require canonical float64 P/M without duplicate coordinates.')


def source_columns_fast(P, M, sources, alpha, include_partition=False):
    """Same masses/RHS meaning as natural_domain_adapter.source_columns_exact.

    Atom order within each source is mention then weak. Default raw_partition
    is None: an atom miss needs neither full Fraction edge lists nor a retained
    partition. Point witnesses can keep using the original exact helper.
    """
    _check_matrices(P, M)
    alpha = exact(alpha)
    _normal_array([float(alpha)], 'alpha')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1).')
    results = []
    n = P.shape[0]
    for source in sources:
        source = int(source)
        rows, p, g, self_position, self_value, c, msum, pss, wsum = _source_data(P, M, source)
        mu, nu = msum/c, wsum/c
        atom_rhs = []
        for group, mass in enumerate((mu, nu)):
            if mass == 0 or mass == 1:
                atom_rhs.append(dict(rhs=np.zeros(n), rhs_conversion_error=Q(0),
                    enclosure=dict(symbolic_zero_group=True)))
            elif group == 0:
                atom_rhs.append(_rhs(n, rows, (g, p),
                    (-alpha/c, alpha*msum/(c*c)), (msum, c), 3))
            else:
                atom_rhs.append(_rhs(n, rows, (p, g, self_value),
                    (alpha*wsum/(c*c)-alpha/c, alpha/c, alpha/c), (c, msum, pss), 5,
                    self_position=self_position))
        partition = None
        if include_partition:
            # This slow optional path is intentionally absent from atom work.
            partition = dict(rows=list(map(int, rows)), original=[exact(x) for x in p],
                mention=[exact(x) for x in g],
                weak=[exact(p[i])-exact(g[i])-(pss if i == self_position else 0) for i in range(len(rows))],
                self_loop=[pss if i == self_position else Q(0) for i in range(len(rows))])
        results.append(dict(source=source, raw_total=c, masses=[mu, nu], normalized_self=pss/c,
            atom_rhs=atom_rhs, raw_partition=partition,
            compiler='dyadic-superaccumulation + exact-scalar/vectorized-enclosed RHS'))
    return results


def point_columns_fast(P, M, sources, weights):
    """Enclose exactly normalized raw source columns, without atom RHS work.

    target = (w/T)P + ((m-w)/T)M + ((1-w)/T)self,
    T = m*sum(M)+w*sum(weak)+self. Coefficients and T are exact Fractions;
    the vectorized 3-term expression has the same gamma_5 enclosure as weak
    atom RHS. Mathematical normalization is never replaced by a float sum.
    """
    _check_matrices(P, M)
    m, w = map(exact, weights)
    if min(m, w) <= 0:
        raise ValueError('Point weights must be positive.')
    results = []
    for source in sources:
        source = int(source)
        rows, p, g, self_position, self_value, c, msum, pss, wsum = _source_data(P, M, source)
        total = m*msum+w*wsum+pss
        local = _linear_combination(rows, (p, g, self_value),
            (w/total, (m-w)/total, (1-w)/total), (c, msum, pss), 5,
            self_position=self_position)
        # Positive in-domain raw weights imply a nonnegative exact column.
        # Rare rounded negatives can be projected to zero; charge the exact
        # stored correction to preserve the same mathematical target.
        negative = local['values'] < 0
        projection = sum_dyadics(-local['values'][negative]) if np.any(negative) else Q(0)
        if projection:
            local['values'][negative] = 0.
        results.append(dict(source=source, rows=rows, values=local['values'],
            column_conversion_error=local['conversion_error']+projection,
            raw_total=c, masses=[msum/c, wsum/c], normalized_self=pss/c,
            weighted_total=total, projection_l1=projection, enclosure=local['enclosure']))
    return results
