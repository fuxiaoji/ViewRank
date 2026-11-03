"""Source-wise corner lifting with shared responses and small exact cores.

Each normalized source column over a positive two-weight rectangle lies in
the convex hull of its four corner columns. Relaxing the sources to choose
corners independently is therefore a sound domain over-approximation. A
discounted linear reward attains its extremum at a deterministic stationary
policy, so strict top-k agreement for all at-most 4**r corner policies proves
agreement throughout the original rectangle. With a slightly nonstochastic
raw background, the same conclusion follows without an MDP interpretation:
each gap is linear-fractional in one source column with positive determinant
denominator, provided the uniform nonnegative-operator contraction holds.
Successively moving each source to a minimizing corner proves the vertex
extremum property. A disagreeing relaxed policy is not a shared-parameter
instability witness.

This module never solves a graph. It shares one prepared response basis across
all policies, prepares their r-by-r cores once across cutoffs, and streams
candidate coordinates in matrix batches. Missing entries and every omitted
node are covered by the caller's response tails. SparseLift and DenseLift use
the same code; DenseLift supplies all rows and zero tails.

The caller supplies canonical exact source masses, and uniform residual and
resolvent bounds for the *actual* target, including any raw-background
correction. The bounds must cover the complete approximate response vectors
before truncation. The arithmetic contract is IEEE binary64 with gradual
underflow, round-to-nearest, no DAZ, and binary64 dot products. No formal graph
contract is established here. Only STABLE or UNKNOWN is returned.
"""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math
import sys

import numpy as np


Q = Fraction
U = Q(1, 2**53)


def _q(value):
    if isinstance(value, Q):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return Q(int(value))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Stored scalars and error bounds must be finite.')
    return Q.from_float(value)


def _upper(value):
    value = _q(value)
    result = float(value)
    if not math.isfinite(result):
        raise OverflowError('A rational bound does not fit binary64.')
    return math.nextafter(result, math.inf) if _q(result) < value else result


def _box(box):
    if len(box) != 2 or any(len(v) != 2 for v in box):
        raise ValueError('box=((mention_low,mention_high),(weak_low,weak_high)).')
    box = tuple(tuple(_q(x) for x in pair) for pair in box)
    if any(low <= 0 or high <= low for low, high in box):
        raise ValueError('Require positive weights and positive interval widths.')
    return box


def _solve_exact(matrix, rhs):
    """Small Gaussian elimination over stored exact constants; no inverse."""
    n = len(rhs)
    a = [list(row)+[rhs[i]] for i, row in enumerate(matrix)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if a[i][j]), None)
        if pivot is None:
            return None
        a[j], a[pivot] = a[pivot], a[j]
        for i in range(j+1, n):
            if a[i][j]:
                scale = a[i][j]/a[j][j]
                for k in range(j, n+1):
                    a[i][k] -= scale*a[j][k]
    x = [Q(0)]*n
    for i in reversed(range(n)):
        x[i] = (a[i][n]-sum((a[i][j]*x[j] for j in range(i+1, n)), Q(0)))/a[i][i]
    return tuple(x)


def _fraction_payload(value):
    if isinstance(value, Q):
        return 16+(abs(value.numerator).bit_length()+7)//8+(value.denominator.bit_length()+7)//8
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (tuple, list)):
        return sum(_fraction_payload(v) for v in value)
    return 8


def _owned_size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size  # owned arrays include their buffer in getsizeof
    if isinstance(value, Q):
        return size+_owned_size(value.numerator, seen)+_owned_size(value.denominator, seen)
    if isinstance(value, (tuple, list)):
        return size+sum(_owned_size(v, seen) for v in value)
    if isinstance(value, dict):
        return size+sum(_owned_size(k, seen)+_owned_size(v, seen) for k, v in value.items())
    if hasattr(value, '__dict__'):
        return size+_owned_size(value.__dict__, seen)
    return size


@dataclass(frozen=True)
class LiftingDomain:
    source_ids: tuple
    mu: tuple
    nu: tuple
    H_m: tuple
    H_w: tuple
    y_sources: tuple
    box: tuple
    corners: tuple
    policy_choices: tuple
    coefficients: tuple
    singular_policies: tuple
    float_coefficients: np.ndarray
    coefficient_conversion_upper: np.ndarray
    absolute_coefficients_upper: np.ndarray

    @property
    def r(self):
        return len(self.source_ids)

    @property
    def policy_count(self):
        return len(self.policy_choices)

    @property
    def logical_policy_count(self):
        return 4**self.r

    @property
    def numeric_payload_bytes(self):
        return sum(_fraction_payload(getattr(self, name)) for name in self.__dataclass_fields__)

    @property
    def python_retained_bytes(self):
        return _owned_size(self)


@dataclass(frozen=True)
class _SourceAction:
    corner: int
    dm: Q
    dw: Q
    column: tuple


def _prepare_lifting(H_m, H_w, mu, nu, y_sources, box, source_ids):
    """Only 4r exact columns, even when there are 4**r policies."""
    ids = tuple(int(v) for v in source_ids)
    r = len(ids)
    if not 1 <= r <= 4 or len(set(ids)) != r or min(ids) < 0:
        raise ValueError('Require one to four distinct nonnegative sources.')
    mu, nu, ys = (tuple(_q(v) for v in x) for x in (mu, nu, y_sources))
    hm, hw = (tuple(tuple(_q(v) for v in row) for row in matrix) for matrix in (H_m, H_w))
    if any(len(v) != r for v in (mu, nu, ys, hm, hw)) or any(len(row) != r for matrix in (hm, hw) for row in matrix):
        raise ValueError('Source dimensions differ.')
    if any(v < 0 for v in mu+nu) or any(a+b > 1 for a, b in zip(mu, nu)):
        raise ValueError('Canonical group masses must be nonnegative and sum to at most one.')
    box = _box(box)
    corners = tuple(product(box[0], box[1]))
    actions = []
    for s in range(r):
        by_column = {}
        for index, (m, w) in enumerate(corners):
            z = 1+(m-1)*mu[s]+(w-1)*nu[s]
            signature = (m*mu[s]/z, w*nu[s]/z, (1-mu[s]-nu[s])/z)
            size = (abs(m-1)+abs(w-1))/z
            if signature not in by_column or size < by_column[signature][0]:
                by_column[signature] = (size, index)
        local = []
        for index in sorted(index for _, index in by_column.values()):
            m, w = corners[index]
            dm, dw = m-1, w-1
            z = 1+dm*mu[s]+dw*nu[s]
            column = tuple((z if i == s else 0)+dm*hm[i][s]+dw*hw[i][s]
                           for i in range(r))
            local.append(_SourceAction(index, dm, dw, column))
        actions.append(tuple(local))
    # Equal normalized group masses give the same exact column; retaining
    # every logical corner label does not require recomputing that operator.
    return ids, mu, nu, hm, hw, ys, box, corners, tuple(actions)


def _compile_reservation_prepared(prepared):
    """Bound live numeric payload, not Python/container or allocator overhead.

    Put each augmented core column over its own common denominator D_j.
    Every elimination entry is a ratio of two minors of the integer-scaled
    columns, divided by D_j. A size-k integer minor is at most k! times the
    product of its column maxima. These bounds also hold after row pivoting.
    Back-substitution products share D_y times a leading minor times the full
    determinant as a common denominator. No 4**r collection of exact solutions
    is present: only one elimination matrix, solution and 2r coefficients live.
    """
    ids, mu, nu, hm, hw, ys, box, corners, actions = prepared
    r, q = len(ids), 2*len(ids)
    p = math.prod(map(len, actions))
    denominators, integer_bits = [], []
    for source_actions in actions:
        values = tuple(v for action in source_actions for v in action.column)
        denominator = math.lcm(*(v.denominator for v in values))
        denominators.append(denominator)
        integer_bits.append(max(1, *(abs(v.numerator*(denominator//v.denominator)).bit_length()
                                     for v in values)))
    denominator = math.lcm(*(v.denominator for v in ys))
    denominators.append(denominator)
    integer_bits.append(max(1, *(abs(v.numerator*(denominator//v.denominator)).bit_length()
                                 for v in ys)))
    # This one bound covers every minor of every source-local policy.
    minor_bits = sum(sorted(integer_bits, reverse=True)[:r])+(math.factorial(r)-1).bit_length()+1
    denominator_bits = max(v.bit_length() for v in denominators)
    delta_bits = max(max(abs(v.numerator).bit_length(), v.denominator.bit_length())
                     for local in actions for action in local for v in (action.dm, action.dw))
    def scalar_bytes(numerator_bits, denominator_bits):
        return 16+(numerator_bits+7)//8+(denominator_bits+7)//8
    matrix_scalar = scalar_bytes(minor_bits, denominator_bits+minor_bits)
    coefficient_bits = denominator_bits+minor_bits+delta_bits
    coefficient_scalar = scalar_bytes(coefficient_bits, coefficient_bits)
    # Products/sums in back substitution share the determinant denominator.
    # Sixteen oversized scalar slots also cover Fraction's transient integer
    # cross-products, gcd/division operands, and outward float conversion.
    # Rounding a C-bit rational needs at most C+54 bits in its stored
    # binary64 Fraction. Rounding its exact conversion error a second time,
    # then comparing the two rationals, uses at most 4C+164 integer bits.
    temporary_bits = max(2*denominator_bits+4*minor_bits+2*delta_bits+r.bit_length()+32,
                         4*coefficient_bits+164)
    exact_work = ((2*r*r+r)*matrix_scalar+(2*r+q)*coefficient_scalar
                  +16*scalar_bytes(temporary_bits, temporary_bits))
    retained = (_fraction_payload((ids, mu, nu, hm, hw, ys, box, corners))
                +8*r*p+8*p+3*q*p*8)
    corner_columns = sum(_fraction_payload((action.dm, action.dw, action.column))
                         for local in actions for action in local)
    input_bits = max(max(abs(v.numerator).bit_length(), v.denominator.bit_length())
        for values in (mu, nu, ys, *(row for matrix in (hm, hw) for row in matrix), *box)
        for v in values)
    # Preparing/deduplicating one source uses at most four three-mass
    # signatures and sizes. Bound their short arithmetic independently of
    # cancellation in the final column values, plus action/index metadata.
    corner_work = 32*scalar_bytes(8*input_bits+16, 8*input_bits+16)
    action_metadata = 8*(12*r+8)
    return int(retained+corner_columns+action_metadata+max(exact_work, corner_work))


def compile_reservation_upper(H_m, H_w, mu, nu, y_sources, box, source_ids):
    """Total owned numeric peak for default streaming ``compile_lifting``.

    Includes its final retained state, 4r deduplicated exact source columns,
    choices and one exact small solve. Caller-owned responses/base remain a
    separate charge. This is not *additional* scratch to add to retained state.
    Python object/container overhead is reported through process RSS, as for
    ``numeric_payload_bytes``. Optional retained exact test coefficients are
    outside this production reservation. No policy core or graph is solved.
    """
    return _compile_reservation_prepared(_prepare_lifting(
        H_m, H_w, mu, nu, y_sources, box, source_ids))


def compile_lifting(H_m, H_w, mu, nu, y_sources, box, source_ids, *,
                    retain_exact_coefficients=False):
    """Prepare source corner columns once, stream exact policy solves.

    Atoms are ordered mention sources then weak sources. Masses may be exact
    Fractions from canonical normalized raw columns. H/y source entries must
    describe the same stored binary64 vectors later supplied to certify.
    Only the optional tiny-test mode retains exact policy coefficients.
    """
    prepared = _prepare_lifting(H_m, H_w, mu, nu, y_sources, box, source_ids)
    ids, mu, nu, hm, hw, ys, box, corners, actions = prepared
    r = len(ids)
    policy_actions = tuple(tuple(action.corner for action in local) for local in actions)
    choices = tuple(product(*policy_actions))
    local_actions = tuple({action.corner: action for action in local} for local in actions)
    coefficients, singular = [], []
    float_coefficients = np.zeros((2*r, len(choices)))
    conversion = np.zeros_like(float_coefficients)
    absolute = np.zeros_like(float_coefficients)
    for index, policy in enumerate(choices):
        columns = tuple(local_actions[s][corner] for s, corner in enumerate(policy))
        matrix = [[columns[j].column[i] for j in range(r)] for i in range(r)]
        t = _solve_exact(matrix, ys)
        if t is None:
            singular.append(index)
            values = None
        else:
            values = tuple(columns[s].dm*t[s] for s in range(r))+tuple(columns[s].dw*t[s] for s in range(r))
        if retain_exact_coefficients:
            coefficients.append(values)
        if values is not None:
            for i, value in enumerate(values):
                rounded = float(value)
                if not math.isfinite(rounded):
                    raise OverflowError('A lifted policy coefficient overflowed.')
                float_coefficients[i, index] = rounded
                conversion[i, index] = _upper(abs(value-_q(rounded)))
                absolute[i, index] = _upper(abs(value))
            del value
        # Do not keep the previous solution alive while computing the next.
        del t, values, matrix, columns
    for array in (float_coefficients, conversion, absolute):
        array.flags.writeable = False
    return LiftingDomain(ids, mu, nu, hm, hw, ys, box, corners, choices,
        tuple(coefficients), tuple(singular), float_coefficients, conversion, absolute)


def _positive_product_upper(left, right, gamma, underflow):
    observed = left@right
    if not np.all(np.isfinite(observed)):
        raise ArithmeticError('Nonfinite lifted coordinate product.')
    denominator = math.nextafter(1-gamma, -math.inf)
    return np.nextafter(np.nextafter(observed+underflow, np.inf)/denominator, np.inf)


def _coordinate_bounds(y, values, missing_tails, domain, posterior, gamma, underflow):
    """Vectorized outward coordinate bounds for all retained local policies."""
    scores = y[:, None]-values@domain.float_coefficients
    if not np.all(np.isfinite(scores)):
        raise ArithmeticError('Nonfinite lifted reconstructed score.')
    absolute = np.abs(values)
    magnitude = _positive_product_upper(absolute, np.abs(domain.float_coefficients), gamma, underflow)
    conversion = (_positive_product_upper(absolute, domain.coefficient_conversion_upper, gamma, underflow)
                  if np.any(domain.coefficient_conversion_upper) else np.zeros(scores.shape))
    magnitude = np.nextafter(np.abs(y[:, None])+magnitude, np.inf)
    arithmetic = np.nextafter(gamma*magnitude, np.inf)
    arithmetic = np.nextafter(np.nextafter(arithmetic+underflow, np.inf)+conversion, np.inf)
    tail = (_positive_product_upper(missing_tails, domain.absolute_coefficients_upper, gamma, underflow)
            if np.any(missing_tails) else np.zeros(scores.shape))
    error = np.nextafter(np.nextafter(arithmetic+tail, np.inf)+posterior[None, :], np.inf)
    return np.nextafter(scores-error, -np.inf), np.nextafter(scores+error, np.inf)


def scratch_upper_bound(domain, n_nodes, chunk_rows=512, n_cuts=3, n_candidates=None,
                        subdivide_depth=0):
    """Conservative numeric workspace reservation; prepared inputs are separate.

    Both candidate and selected-coordinate scans are chunked. Coefficients in
    the compiled domain are retained state, rather than scratch. This includes
    preparation masks/ID arrays and the peak temporary batched products; Python
    container overhead remains visible in process RSS.
    """
    n = int(n_nodes)
    count = n if n_candidates is None else int(n_candidates)
    depth = int(subdivide_depth)
    # Subboxes can lose parent corner coincidences, so descendants may need
    # all 4**r actions even if the root itself deduplicated equivalent columns.
    rows, p, q = min(int(chunk_rows), count), 4**domain.r if depth else domain.policy_count, 2*domain.r
    if n < 1 or count < 1 or rows < 1 or n_cuts < 1 or depth != subdivide_depth or depth < 0:
        raise ValueError('Positive node/candidate/chunk/cut counts required.')
    small = 8*p*(12+4*int(n_cuts))+8*q*p+4096
    preparation = n+count*(48+4*int(n_cuts))
    chunks = 20*rows*p*8+6*rows*q*8+rows*(q+int(n_cuts)+24)
    if depth:
        scalar_bits = max(max(abs(v.numerator).bit_length(), v.denominator.bit_length())
            for v in (*domain.mu, *domain.nu, *domain.y_sources,
                      *(v for row in domain.H_m+domain.H_w for v in row)))
        box_bits = max(max(abs(v.numerator).bit_length(), v.denominator.bit_length())
                       for pair in domain.box for v in pair)+depth+4
        core_scalar_bytes = 128+(2*domain.r*(3*scalar_bits+2*box_bits)+256+7)//8
        # DFS keeps at most `depth` extra small domains. Exact policy
        # coefficients are released after each solve, never retained 4**r times.
        child_payload = (domain.numeric_payload_bytes*p+domain.policy_count-1)//domain.policy_count+4096
        child_work = depth*child_payload+32*domain.r**2*core_scalar_bytes
        small += child_work+(2**depth)*int(n_cuts)*256
    return int(small+max(preparation, chunks))


def certify_lifting_many(domain, y, candidate_rows, retained_values, retained_mask,
                         tail_bounds, selected_sets, *, kappa, base_residual,
                         atom_residuals, round_pad=0, row_chunk=512,
                         retained_columns=None):
    """Prove several requested cuts in one streamed coordinate pass.

    selected_sets maps cutoff to proposed IDs. The residual contract is
    identical for SparseLift/DenseLift and must include raw-background guards.
    A local policy with an unseparated gap produces UNKNOWN, never CHANGED.
    numeric_payload_bytes/python_retained_bytes describe owned small-core state;
    input response arrays belong to, and must be charged by, the caller.
    """
    y = np.asarray(y, dtype=np.float64)
    rows = tuple(int(v) for v in candidate_rows)
    mask = None if retained_mask is None else np.asarray(retained_mask, dtype=bool)
    q = 2*domain.r
    if retained_columns is None:
        values = np.asarray(retained_values, dtype=np.float64)
        columns = None
        if values.shape != (len(rows), q) or not np.all(np.isfinite(values)):
            raise ValueError('Retained values must have finite candidate-by-2r shape.')
    else:
        if retained_values is not None or len(retained_columns) != q:
            raise ValueError('Borrow 2r full response columns and no retained matrix.')
        values = None
        columns = tuple(np.asarray(v, dtype=np.float64) for v in retained_columns)
        if any(v.shape != y.shape or not np.all(np.isfinite(v)) for v in columns):
            raise ValueError('Every borrowed response column must have full-graph shape.')
    selections = {int(k): tuple(int(v) for v in ids) for k, ids in selected_sets.items()}
    if y.ndim != 1 or not len(y) or not np.all(np.isfinite(y)):
        raise ValueError('The stored full base must be a finite binary64 vector.')
    if len(set(rows)) != len(rows) or any(v < 0 or v >= len(y) for v in rows):
        raise ValueError('Candidate rows must be distinct valid node IDs.')
    if mask is None and columns is None or mask is not None and mask.shape != (len(rows), q):
        raise ValueError('Sparse values require a candidate-by-2r retained mask.')
    if not selections or any(len(ids) != k or len(set(ids)) != k or not set(ids) <= set(rows)
                             for k, ids in selections.items()):
        raise ValueError('Every selected set must have its declared positive cutoff and lie in candidates.')
    if any(k < 1 or k > len(y) for k in selections) or int(row_chunk) != row_chunk or row_chunk < 1:
        raise ValueError('Invalid cutoffs or row chunk.')
    position = {node: i for i, node in enumerate(rows)}
    for i, source in enumerate(domain.source_ids):
        if source not in position or _q(y[source]) != domain.y_sources[i]:
            raise ValueError('Source base coordinates differ from the compiled core.')
        source_row = position[source]
        actual = values[source_row] if columns is None else [v[source] for v in columns]
        if (mask is not None and not all(mask[source_row])) or tuple(_q(v) for v in actual) != domain.H_m[i]+domain.H_w[i]:
            raise ValueError('All source response coordinates must be retained and equal the core.')
    tails = tuple(_q(v) for v in tail_bounds)
    residuals = tuple(_q(v) for v in atom_residuals)
    kap, rho, pad = map(_q, (kappa, base_residual, round_pad))
    if len(tails) != q or len(residuals) != q or any(v < 0 for v in tails+residuals+(rho, pad)) or kap <= 0:
        raise ValueError('Require nonnegative tail/residual bounds and a positive uniform kappa.')
    common = dict(local_corner_policies=domain.policy_count,
        logical_corner_policy_combinations=domain.logical_policy_count, source_count=domain.r,
        rank1_complete_corner_family=domain.r == 1,
        numeric_payload_bytes=domain.numeric_payload_bytes,
        python_retained_bytes=domain.python_retained_bytes,
        graph_solves=0, prepared_core_solve_count=domain.policy_count,
        core_solves_during_certification=0,
        contract='Conditional on canonical source masses, coherent stored approximate responses, uniform actual-target residual/resolvent bounds and every omitted-coordinate tail. Local counterpolicies are not actual shared-parameter witnesses.')
    if domain.singular_policies:
        return dict(results={k: dict(status='UNKNOWN', reason='A relaxed approximate source core is singular.',
            selected_ids=list(ids), singular_policy_indices=list(domain.singular_policies)) for k, ids in selections.items()},
            evaluated_candidate_rows=0, early_unknown_stop=True, **common)
    posterior = []
    for policy in range(domain.policy_count):
        # These are outward upper bounds on the exact small-core coefficients;
        # retaining their large Fraction representations is unnecessary.
        posterior.append(_upper(kap*(rho+sum((_q(domain.absolute_coefficients_upper[j, policy])*residuals[j]
            for j in range(q)), Q(0)))+pad))
    posterior = np.asarray(posterior)
    tails_upper = np.array([_upper(v) for v in tails])
    operation_count = 2*q+8
    gamma = _upper(operation_count*U/(1-operation_count*U))
    # Normal-threshold allowance is deliberately more conservative than the
    # subnormal unit; guard assumptions still exclude DAZ on input operands.
    underflow = math.nextafter(operation_count*np.finfo(np.float64).tiny/math.nextafter(1-gamma, -math.inf), math.inf)
    selected_positions = sorted({position[node] for ids in selections.values() for node in ids})
    row_array = np.asarray(rows)
    def gather(indices):
        indices = np.asarray(indices)
        block_mask = mask[indices] if mask is not None else None
        block_values = values[indices] if columns is None else np.column_stack([v[row_array[indices]] for v in columns])
        if block_mask is not None:
            block_values = np.where(block_mask, block_values, 0.)
            missing = np.where(block_mask, 0., tails_upper)
        else:
            missing = np.zeros(block_values.shape)
        return block_values, missing
    selected_lower = {k: np.full(domain.policy_count, np.inf) for k in selections}
    selected_position_sets = {k: {position[node] for node in ids} for k, ids in selections.items()}
    for start in range(0, len(selected_positions), int(row_chunk)):
        indices = selected_positions[start:start+int(row_chunk)]
        selected_values, selected_missing = gather(indices)
        lower, _ = _coordinate_bounds(y[row_array[indices]], selected_values,
            selected_missing, domain, posterior, gamma, underflow)
        for k, selected in selected_position_sets.items():
            local = [i for i, index in enumerate(indices) if index in selected]
            if local:
                selected_lower[k] = np.minimum(selected_lower[k], np.min(lower[local], axis=0))
    competitor_upper = {k: np.full(domain.policy_count, -np.inf) for k in selections}
    margins = {k: math.inf for k in selections}
    unresolved = set(selections)
    failed_policy = {}
    outside_mask = np.ones(len(y), dtype=bool)
    outside_mask[list(rows)] = False
    outside_base = float(np.max(y[outside_mask])) if np.any(outside_mask) else None
    if outside_base is not None:
        tail = _positive_product_upper(tails_upper[None, :], domain.absolute_coefficients_upper, gamma, underflow)[0]
        upper = np.nextafter(np.nextafter(outside_base+tail, np.inf)+posterior, np.inf)
        for k in tuple(unresolved):
            competitor_upper[k] = upper.copy()
            gap = np.nextafter(selected_lower[k]-upper, -np.inf)
            margins[k] = float(np.min(gap))
            if not np.all(gap > 0):
                failed_policy[k] = int(np.flatnonzero(gap <= 0)[0])
                unresolved.remove(k)
    evaluated = 0
    try:
        for start in range(0, len(rows), int(row_chunk)):
            if not unresolved:
                break
            stop = min(start+int(row_chunk), len(rows))
            block_values, missing = gather(range(start, stop))
            _, upper = _coordinate_bounds(y[np.asarray(rows[start:stop])], block_values,
                missing, domain, posterior, gamma, underflow)
            evaluated += stop-start
            for k in tuple(unresolved):
                selected = set(selections[k])
                competitors = [i for i, node in enumerate(rows[start:stop]) if node not in selected]
                if competitors:
                    competitor_upper[k] = np.maximum(competitor_upper[k], np.max(upper[competitors], axis=0))
                gap = np.nextafter(selected_lower[k]-competitor_upper[k], -np.inf)
                margins[k] = float(np.min(gap))
                if not np.all(gap > 0):
                    failed_policy[k] = int(np.flatnonzero(gap <= 0)[0])
                    unresolved.remove(k)
    except (ArithmeticError, OverflowError):
        return dict(results={k: dict(status='UNKNOWN', reason='Nonfinite outward coordinate evaluation.', selected_ids=list(ids))
            for k, ids in selections.items()}, evaluated_candidate_rows=evaluated, early_unknown_stop=True, **common)
    results = {}
    for k, ids in selections.items():
        passed = k in unresolved
        results[k] = dict(status='STABLE' if passed else 'UNKNOWN', selected_ids=list(ids),
            minimum_relaxed_policy_gap_lower_bound=margins[k] if math.isfinite(margins[k]) else None,
            reason='Every source-wise corner policy strictly separates the requested set, including all omitted nodes.' if passed else
                'A relaxed local policy gap or tail enclosure is unseparated; this is not an actual policy-change witness.')
        if k in failed_policy:
            index = failed_policy[k]
            results[k]['first_unseparated_local_policy'] = list(domain.policy_choices[index])
    return dict(results=results, evaluated_candidate_rows=evaluated,
        early_unknown_stop=evaluated < len(rows), maximum_coordinate_batch_shape=[min(int(row_chunk), len(rows)), domain.policy_count],
        **common)


def certify_lifting(domain, y, candidate_rows, retained_values, retained_mask,
                     tail_bounds, selected_ids, **kwargs):
    """Single-cut wrapper; compile_lifting state is reusable across cutoffs."""
    selected_ids = tuple(selected_ids)
    result = certify_lifting_many(domain, y, candidate_rows, retained_values,
        retained_mask, tail_bounds, {len(selected_ids): selected_ids}, **kwargs)
    return dict(result['results'][len(selected_ids)],
        **{k: v for k, v in result.items() if k != 'results'})


def certify_many(domain, y, candidate_rows, retained_values, retained_mask,
                 tail_bounds, selected_sets, *, kappa, base_residual,
                 atom_residuals, round_pad=0, subdivide_depth=0,
                 retained_columns=None, chunk_rows=512, node_chunk_rows=None):
    """Engine API: common longest-axis subdivision with small-core reuse.

    Every child borrows the same response arrays. Only source-local corner
    coefficients are recompiled; no graph solve or graph-size response copy
    is introduced. Both children are kept in the proof accounting, including
    UNKNOWN outcomes. All cutoffs share each coordinate scan.
    """
    if node_chunk_rows is not None:
        chunk_rows = node_chunk_rows
    if int(subdivide_depth) != subdivide_depth or subdivide_depth < 0:
        raise ValueError('Subdivision depth must be a nonnegative integer.')
    depth = int(subdivide_depth)
    selections = {int(k): tuple(int(v) for v in ids) for k, ids in selected_sets.items()}
    rows = tuple(candidate_rows)
    state = {k: dict(attempts=0, leaves=[]) for k in selections}
    stats = dict(control_boxes=0, candidate_node_policies=0, node_chunks=0,
        cuts=len(selections), chunk_rows=int(chunk_rows), graph_solves=0,
        prepared_root_core_solve_count=domain.policy_count,
        logical_root_corner_policy_combinations=domain.logical_policy_count,
        child_core_solve_count=0, exact_policy_coefficients_retained=bool(domain.coefficients))
    workspace = scratch_upper_bound(domain, len(y), chunk_rows, len(selections), len(rows), depth)

    def visit(current, remaining_depth, active):
        stats['control_boxes'] += 1
        for key in active:
            state[key]['attempts'] += 1
        try:
            result = certify_lifting_many(current, y, rows, retained_values, retained_mask,
                tail_bounds, active, kappa=kappa, base_residual=base_residual,
                atom_residuals=atom_residuals, round_pad=round_pad,
                row_chunk=chunk_rows, retained_columns=retained_columns)
            outcomes = result['results']
            evaluated = result['evaluated_candidate_rows']
            stats['candidate_node_policies'] += evaluated*current.policy_count
            stats['node_chunks'] += (evaluated+int(chunk_rows)-1)//int(chunk_rows)
        except (ArithmeticError, OverflowError):
            outcomes = {k: dict(status='UNKNOWN', selected_ids=list(ids),
                reason='Lifted coefficient or outward coordinate arithmetic became nonfinite.') for k, ids in active.items()}
        passed, pending = {}, {}
        for key, result in outcomes.items():
            if result['status'] == 'STABLE' or remaining_depth == 0:
                state[key]['leaves'].append(dict(box=[[str(a), str(b)] for a, b in current.box], **result))
                passed[key] = result['status'] == 'STABLE'
            else:
                pending[key] = active[key]
        if pending:
            axis = max(range(2), key=lambda i: current.box[i][1]-current.box[i][0])
            low, high = current.box[axis]
            middle = (low+high)/2
            child_results = []
            for a, b in ((low, middle), (middle, high)):
                box = list(current.box)
                box[axis] = (a, b)
                try:
                    child = compile_lifting(current.H_m, current.H_w, current.mu, current.nu,
                        current.y_sources, box, current.source_ids)
                    stats['child_core_solve_count'] += child.policy_count
                    child_results.append(visit(child, remaining_depth-1, pending))
                    del child
                except (ArithmeticError, OverflowError):
                    for key, ids in pending.items():
                        state[key]['attempts'] += 1
                        state[key]['leaves'].append(dict(box=[[str(x), str(z)] for x, z in box],
                            status='UNKNOWN', selected_ids=list(ids), reason='Child local-policy core arithmetic became nonfinite.'))
                    child_results.append({key: False for key in pending})
            passed.update({key: all(result[key] for result in child_results) for key in pending})
        return passed

    outcomes = visit(domain, depth, selections)
    cuts = {key: dict(status='STABLE' if outcomes[key] else 'UNKNOWN',
        selected_ids=list(ids), k=key, source_count=domain.r, atom_count=2*domain.r,
        attempted_boxes=state[key]['attempts'], leaf_boxes=len(state[key]['leaves']),
        leaves=state[key]['leaves'], local_corner_policies_per_box=domain.policy_count,
        contract='Source-wise corner relaxation with exact small cores, guarded approximate-response reconstruction, all omitted outputs bounded. A local policy is not an actual shared-parameter change witness.')
        for key, ids in selections.items()}
    return dict(cuts=cuts, workspace_peak_bytes=workspace, peak_numeric_scratch_bytes=workspace,
        workspace_peak_is_conservative_reservation=True, shared_scan=stats)


def certify_rank1_corners(point_results, box, requested_ks=None, *, source_count=1):
    """Complete rank-one shortcut using four *actual* corner certificates.

    No atoms are required. Each record supplies weights, sets, strict_flags,
    score_error_bound and raw_gaps (or certified_gap_lower_bounds). The supplied
    error/gap bounds must have been verified for the exact actual normalized
    corner views by the caller, including raw-background/residual allowances.
    Mere flags without gap evidence are not accepted. This function does not
    turn a different corner membership into an instability report.
    """
    if source_count != 1:
        raise ValueError('The four-corner completeness theorem requires one source.')
    box = _box(box)
    corners = set(product(box[0], box[1]))
    records = {}
    for record in point_results:
        weights = tuple(_q(v) for v in record['weights'])
        if weights not in corners or weights in records:
            raise ValueError('Require each true domain corner exactly once.')
        records[weights] = record
    if set(records) != corners:
        raise ValueError('All four actual corners are required.')
    first = next(iter(records.values()))
    ks = tuple(int(k) for k in (requested_ks if requested_ks is not None else first['sets']))
    output = {}
    for k in ks:
        memberships, gaps = [], []
        valid = k > 0
        for record in records.values():
            def field(name):
                values = record.get(name, {})
                return values.get(k, values.get(str(k)))
            ids = field('sets')
            flag = field('strict_flags')
            if ids is None or len(ids) != k or len(set(ids)) != k or flag is not True:
                valid = False
                continue
            certified = field('certified_gap_lower_bounds')
            if certified is not None:
                gap = _q(certified)
            else:
                raw_gap = field('raw_gaps')
                error = record.get('score_error_bound')
                if raw_gap is None or error is None or _q(error) < 0:
                    valid = False
                    continue
                gap = _q(raw_gap)-2*_q(error)
            valid &= gap > 0
            memberships.append(frozenset(ids))
            gaps.append(gap)
        passed = bool(valid and len(memberships) == 4 and len(set(memberships)) == 1)
        output[k] = dict(status='STABLE' if passed else 'UNKNOWN',
            selected_ids=sorted(memberships[0]) if memberships else [],
            minimum_certified_corner_gap=str(min(gaps)) if gaps else None,
            actual_corner_views=4, prepared_response_atoms=0,
            reason='Rank-one true score-difference numerators are affine with a positive denominator; all four actual corners strictly agree.' if passed else
                'Actual corner membership or strict gap evidence does not establish rank-one domain stability.')
    return dict(results=output, source_count=1, complete_rank1_shortcut=True,
        contract='Conditional on supplied actual-corner score/gap residual certificates and one-source normalized two-group semantics; no relaxed local policy is a real witness.')
