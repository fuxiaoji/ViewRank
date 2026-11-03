"""Isolated exact-polynomial certificate for a fixed two-group policy box.

This is a certificate kernel, not a natural-graph ranker or benchmark. For
r<=4 source nodes it compiles stored scalar inputs as exact Fractions:
  M = I + (m-1)(diag(mu)+H_m) + (w-1)(diag(nu)+H_w)
  D = det(M), C_s = det(M with column s replaced by y_S)
  F = ((m-1)C_0,...,(m-1)C_{r-1},(w-1)C_0,...).
Degree-r tensor Bernstein control coefficients retain parameter dependence.

The caller must supply valid residual upper bounds rho_y/rho_atom and a
uniform resolvent/error multiplier kappa for EVERY actual policy in the box.
They must bound the exact vectors represented by the stored y/R values;
tail bounds must cover every omitted R entry. This kernel cannot establish
those contracts from a natural graph. A round_pad is an additional score
error upper bound. Nonnegative coefficients of |F| are coefficientwise abs
of F's Bernstein coefficients, never an assumption that C or F is positive.

All node arithmetic uses nextafter outward float intervals after exact small
polynomial compilation. These interval claims require IEEE binary64 round-
to-nearest ties-to-even, gradual underflow and no denormals-as-zero. The
natural adapter checks this environment; standalone callers must establish it.
STABLE proves the requested membership for the box
under the explicit contracts. UNKNOWN supplies no instability witness.
No solver, graph backend, scipy, or numpy is imported by this module.
"""
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
import math


ZERO = Fraction(0)
ONE = Fraction(1)


def exact(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError('A Boolean is not a stored scalar.')
    if isinstance(value, int):
        return Fraction(value)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('All stored scalars must be finite.')
    return Fraction.from_float(value)


def _clean(poly):
    return {key: value for key, value in poly.items() if value}


def poly_add(left, right):
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, ZERO)+value
    return _clean(result)


def poly_scale(poly, scalar):
    scalar = exact(scalar)
    return _clean({key: value*scalar for key, value in poly.items()})


def poly_mul(left, right):
    result = {}
    for (i, j), a in left.items():
        for (k, l), b in right.items():
            key = (i+k, j+l)
            result[key] = result.get(key, ZERO)+a*b
    return _clean(result)


def poly_eval(poly, m, w):
    m, w = exact(m), exact(w)
    return sum((value*m**i*w**j for (i, j), value in poly.items()), ZERO)


def determinant(matrix):
    n = len(matrix)
    result = {}
    for order in permutations(range(n)):
        sign = -1 if sum(order[i] > order[j] for i in range(n) for j in range(i+1, n)) % 2 else 1
        term = {(0, 0): Fraction(sign)}
        for i, column in enumerate(order):
            term = poly_mul(term, matrix[i][column])
        result = poly_add(result, term)
    return result


def _degree(poly):
    return max((i+j for i, j in poly), default=-1)


def _box(value):
    if len(value) != 2 or any(len(pair) != 2 for pair in value):
        raise ValueError('box=((m_low,m_high),(w_low,w_high)).')
    result = tuple(tuple(exact(x) for x in pair) for pair in value)
    if any(low <= 0 or high <= low for low, high in result):
        raise ValueError('Both positive-weight intervals must have positive width.')
    return result


def bernstein(poly, box, degree):
    """Exact tensor Bernstein controls on the requested rectangle, flattened.

    Index order is i*(degree+1)+j, for m index i then w index j.
    The tensor degrees are (degree,degree), even for smaller total degree.
    """
    (m0, m1), (w0, w1) = box
    dm, dw = m1-m0, w1-w0
    powers = {}
    for (i, j), value in poly.items():
        if i > degree or j > degree:
            raise ValueError('Polynomial exceeds the requested tensor degree.')
        for a in range(i+1):
            for b in range(j+1):
                key = (a, b)
                powers[key] = powers.get(key, ZERO)+value*math.comb(i, a)*m0**(i-a)*dm**a*math.comb(j, b)*w0**(j-b)*dw**b
    controls = []
    for i in range(degree+1):
        for j in range(degree+1):
            controls.append(sum((value*Fraction(math.comb(i, a), math.comb(degree, a))*Fraction(math.comb(j, b), math.comb(degree, b))
                for (a, b), value in powers.items() if a <= i and b <= j), ZERO))
    return tuple(controls)


def bernstein_eval(controls, degree, u, v):
    u, v = exact(u), exact(v)
    return sum((controls[i*(degree+1)+j]*math.comb(degree, i)*u**i*(1-u)**(degree-i)*math.comb(degree, j)*v**j*(1-v)**(degree-j)
        for i in range(degree+1) for j in range(degree+1)), ZERO)


@dataclass(frozen=True)
class ExactDomain:
    r: int
    source_ids: tuple
    mu: tuple
    nu: tuple
    H_m: tuple
    H_w: tuple
    y_sources: tuple
    box: tuple
    D: dict
    C: tuple
    F: tuple

    def controls(self, box=None):
        local = self.box if box is None else box
        return bernstein(self.D, local, self.r), tuple(bernstein(poly, local, self.r) for poly in self.F)


def compile_domain(H_m, H_w, mu, nu, y_sources, box, source_ids):
    """Compile a domain; atom order is all mention sources, then all weak.

    Float inputs are interpreted as their exact stored binary64 values.
    Fractions are also supported for independently exact synthetic checks.
    Source rows and y_S consistency are enforced again by certify().
    """
    source_ids = tuple(int(x) for x in source_ids)
    r = len(source_ids)
    if r < 1 or r > 4 or len(set(source_ids)) != r or min(source_ids) < 0:
        raise ValueError('Require one to four unique nonnegative source IDs.')
    mu, nu, ys = (tuple(exact(x) for x in values) for values in (mu, nu, y_sources))
    hm, hw = (tuple(tuple(exact(x) for x in row) for row in matrix) for matrix in (H_m, H_w))
    if any(len(v) != r for v in (mu, nu, ys, hm, hw)) or any(len(row) != r for matrix in (hm, hw) for row in matrix):
        raise ValueError('Source vectors and H matrices must match source count.')
    if any(x < 0 for x in mu+nu) or any(a+b > 1 for a, b in zip(mu, nu)):
        raise ValueError('Mention/weak masses must be nonnegative and sum to at most one.')
    matrix = []
    for i in range(r):
        row = []
        for j in range(r):
            bm = hm[i][j]+(mu[j] if i == j else 0)
            bw = hw[i][j]+(nu[j] if i == j else 0)
            row.append(_clean({(0, 0): Fraction(i == j)-bm-bw, (1, 0): bm, (0, 1): bw}))
        matrix.append(row)
    D = determinant(matrix)
    C = []
    for source in range(r):
        replaced = [[({(0, 0): ys[i]} if j == source else matrix[i][j]) for j in range(r)] for i in range(r)]
        C.append(determinant(replaced))
    F = tuple(poly_mul({(0, 0): -ONE, (1, 0): ONE}, c) for c in C)
    F += tuple(poly_mul({(0, 0): -ONE, (0, 1): ONE}, c) for c in C)
    if _degree(D) > r or any(_degree(c) > r-1 for c in C) or any(_degree(f) > r for f in F):
        raise ArithmeticError('The claimed polynomial degree invariant failed.')
    return ExactDomain(r, source_ids, mu, nu, hm, hw, ys, _box(box), D, tuple(C), F)


def _down(value):
    return math.nextafter(value, -math.inf)


def _up(value):
    return math.nextafter(value, math.inf)


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    @classmethod
    def of(cls, value):
        value = exact(value)
        nearest = float(value)
        if not math.isfinite(nearest):
            raise OverflowError('Exact coefficient does not fit finite binary64.')
        stored = Fraction.from_float(nearest)
        return cls(_down(nearest) if stored > value else nearest,
                   _up(nearest) if stored < value else nearest)

    def __add__(self, other):
        other = other if isinstance(other, Interval) else Interval.of(other)
        return Interval(_down(self.lo+other.lo), _up(self.hi+other.hi))

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        other = other if isinstance(other, Interval) else Interval.of(other)
        return self+-other

    def __mul__(self, other):
        other = other if isinstance(other, Interval) else Interval.of(other)
        values = (self.lo*other.lo, self.lo*other.hi, self.hi*other.lo, self.hi*other.hi)
        if any(math.isnan(v) for v in values):
            raise ArithmeticError('Nonfinite interval product.')
        return Interval(_down(min(values)), _up(max(values)))


def _nonnegative(values, name):
    result = tuple(exact(value) for value in values)
    if any(value < 0 for value in result):
        raise ValueError(name+' must be nonnegative upper bounds.')
    return result


def certify(domain, y, candidate_rows, retained_values, retained_mask, tail_bounds,
            selected_ids, *, kappa, base_residual, atom_residuals, round_pad=0,
            subdivide_depth=0):
    """Return STABLE/UNKNOWN for a single fixed proposed top-k membership.

    y is the full base vector. retained_values/mask have shape (#candidates,2r).
    A true mask denotes an exactly stored coordinate; every false/outside
    coordinate must be covered by that atom's tail bound. Source rows must
    be candidates, with all their atom coordinates retained and equal to H.
    Dense uses all rows, all masks true and zero tails. Witnesses are omitted.
    """
    y = tuple(exact(value) for value in y)
    candidates = tuple(int(value) for value in candidate_rows)
    selected = tuple(int(value) for value in selected_ids)
    q = 2*domain.r
    values = tuple(tuple(exact(value) for value in row) for row in retained_values)
    masks = tuple(tuple(bool(value) for value in row) for row in retained_mask)
    if not y or len(set(candidates)) != len(candidates) or any(i < 0 or i >= len(y) for i in candidates):
        raise ValueError('Candidate IDs must be distinct nodes in the full base vector.')
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(candidates):
        raise ValueError('Proposed selected IDs must be a nonempty subset of candidates.')
    if len(values) != len(candidates) or len(masks) != len(candidates) or any(len(row) != q for matrix in (values, masks) for row in matrix):
        raise ValueError('Retained matrices must have (#candidate,2r) entries.')
    positions = {node: index for index, node in enumerate(candidates)}
    for i, source in enumerate(domain.source_ids):
        if source not in positions or y[source] != domain.y_sources[i]:
            raise ValueError('Stored source base coordinates differ from compiled y_S.')
        row = positions[source]
        expected = domain.H_m[i]+domain.H_w[i]
        if not all(masks[row]) or values[row] != expected:
            raise ValueError('Source atom coordinates must be retained and equal compiled H.')
    tails = _nonnegative(tail_bounds, 'tail_bounds')
    residuals = _nonnegative(atom_residuals, 'atom_residuals')
    rho, kap, pad = _nonnegative((base_residual, kappa, round_pad), 'error contract')
    if len(tails) != q or len(residuals) != q or kap <= 0 or int(subdivide_depth) != subdivide_depth or subdivide_depth < 0:
        raise ValueError('Require 2r atom bounds, positive kappa and nonnegative integer subdivision depth.')
    outside = [y[i] for i in range(len(y)) if i not in positions]
    outside_base = max(outside) if outside else None
    selected_set = set(selected)
    common_tail = tuple(Interval.of(value) for value in tails)
    residual_intervals = tuple(Interval.of(value) for value in residuals)
    kap_i, rho_i, pad_i = map(Interval.of, (kap, rho, pad))
    y_i = tuple(Interval.of(y[node]) for node in candidates)
    value_i = tuple(tuple(Interval.of(value) for value in row) for row in values)

    def one_box(box):
        D, F = domain.controls(box)
        if any(d <= 0 for d in D):
            return dict(status='UNKNOWN', reason='D Bernstein controls are not all strictly positive.')
        d_min = min(D)
        margin_min = math.inf
        for control, d in enumerate(D):
            di = Interval.of(d)
            if di.lo <= 0:
                return dict(status='UNKNOWN', reason='Positive D control underflowed its finite interval.')
            fi = tuple(Interval.of(f[control]) for f in F)
            ai = tuple(Interval.of(abs(f[control])) for f in F)
            residual = rho_i*di
            for residual_atom, amplitude in zip(residual_intervals, ai):
                residual = residual+residual_atom*amplitude
            error = kap_i*residual+pad_i*di
            if not math.isfinite(error.hi):
                return dict(status='UNKNOWN', reason='Nonfinite outward error bound.')
            selected_lower = math.inf
            competitor_upper = -math.inf
            for row, node in enumerate(candidates):
                numerator = y_i[row]*di
                tail = Interval.of(0)
                for atom in range(q):
                    if masks[row][atom]:
                        numerator = numerator-value_i[row][atom]*fi[atom]
                    else:
                        tail = tail+common_tail[atom]*ai[atom]
                if node in selected_set:
                    selected_lower = min(selected_lower, (numerator-tail).lo)
                else:
                    competitor_upper = max(competitor_upper, (numerator+tail).hi)
            if competitor_upper != -math.inf:
                right = (Interval(competitor_upper, competitor_upper)+Interval.of(2)*error).hi
                margin = _down(selected_lower-right)
                margin_min = min(margin_min, margin)
                if not margin > 0:
                    return dict(status='UNKNOWN', reason='Candidate gap controls do not prove strict separation.', failed_control=control)
            if outside_base is not None:
                upper = Interval.of(outside_base)*di
                for tail_atom, amplitude in zip(common_tail, ai):
                    upper = upper+tail_atom*amplitude
                right = (upper+Interval.of(2)*error).hi
                margin = _down(selected_lower-right)
                margin_min = min(margin_min, margin)
                if not margin > 0:
                    return dict(status='UNKNOWN', reason='Outside-candidate controls do not prove strict separation.', failed_control=control)
        return dict(status='STABLE', reason='Every D control is positive and every required gap control has a strictly positive outward lower bound.',
                    determinant_control_lower_bound=Interval.of(d_min).lo,
                    determinant_control_lower_bound_exact=str(d_min),
                    minimum_gap_control_lower_bound=margin_min if math.isfinite(margin_min) else None)

    leaves = []
    attempts = 0
    def visit(box, depth):
        nonlocal attempts
        attempts += 1
        try:
            result = one_box(box)
        except (OverflowError, ArithmeticError):
            result = dict(status='UNKNOWN', reason='Outward interval arithmetic became nonfinite.')
        if result['status'] == 'STABLE' or depth == 0:
            leaves.append(dict(box=[[str(a), str(b)] for a, b in box], **result))
            return result['status'] == 'STABLE'
        axis = max(range(2), key=lambda i: box[i][1]-box[i][0])
        low, high = box[axis]
        middle = (low+high)/2
        halves = []
        for a, b in ((low, middle), (middle, high)):
            child = list(box)
            child[axis] = (a, b)
            halves.append(tuple(child))
        # Both children are audited; UNKNOWN is not silently dropped.
        outcomes = [visit(child, depth-1) for child in halves]
        return all(outcomes)
    passed = visit(domain.box, int(subdivide_depth))
    return dict(status='STABLE' if passed else 'UNKNOWN', selected_ids=list(selected), k=len(selected),
        source_count=domain.r, atom_count=q, tensor_degree=[domain.r, domain.r],
        attempted_boxes=attempts, leaf_boxes=len(leaves), leaves=leaves,
        contract='Conditional on supplied uniform residual/resolvent/rounding and omitted-coordinate tail upper bounds; no natural-graph proof or actual-view witness is synthesized.')
