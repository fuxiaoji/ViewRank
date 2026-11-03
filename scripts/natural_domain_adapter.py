"""Minimal fixed-first-question natural adapter for the polynomial kernel.

No performance comparison or cache-ranker claim. The mathematical target
keeps non-source raw P columns unchanged and exactly normalizes source raw
mention/weak/self weights. A canonical all-column-normalized P' is only an
algebraic reference; its deviation from the old raw background is explicitly
charged in the residual bounds and old-target resolvent multiplier.
"""
from fractions import Fraction as Q
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
from scipy import sparse


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.policy_domain_polynomial import compile_domain, certify

U = Q(1, 2**53)
TINY = Q(1, 2**1022)  # Normal threshold, deliberately conservative for FTZ.
GRID = tuple((m, w) for m in (1., 2., 3., 4.) for w in (.25, .5, 1., 2.))
INTERIORS = tuple((m, w) for m in (1.5, 2.5, 3.5) for w in (.5, 1.125, 1.75))
KS = (2, 5, 10)
DATASETS = ('hotpot_pool7405', 'musique_pool2417')


def exact(value):
    if isinstance(value, Q):
        return value
    if isinstance(value, (int, np.integer)):
        return Q(int(value))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite stored scalar.')
    return Q.from_float(value)


def gamma(count):
    count = int(count)
    if count < 0 or count*U >= 1:
        raise ValueError('Invalid operation count for gamma_n.')
    return count*U/(1-count*U)


def checked_float_array(values, label, allow_subnormal=False):
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(label+' contains nonfinite values.')
    if not allow_subnormal and np.any((values != 0)&(np.abs(values) < np.finfo(np.float64).tiny)):
        raise ValueError(label+' contains a nonzero subnormal operand (DAZ guard).')
    return values


def kernel_runtime_guard():
    """The nextafter interval kernel needs gradual underflow and no DAZ.

    The residual guard's conservative FTZ allowance does not transfer to
    interval multiplication. Runtime variables prevent constant folding.
    """
    one, half = float(1), float(.5)
    tiny = float(np.finfo(np.float64).tiny)
    eta = math.nextafter(float(0), one)
    sub = tiny*half
    ulp_half = float(2**-53)
    second_successor = math.nextafter(math.nextafter(one, math.inf), math.inf)
    tests = dict(normal_product_to_subnormal_nonzero=sub > 0,
        subnormal_input_roundtrip=sub*float(2) == tiny,
        smallest_subnormal_survives_multiply=eta*one == eta,
        smallest_subnormal_sum=eta+eta == math.nextafter(eta, math.inf),
        positive_ties_to_even=one+ulp_half == one and one+float(3)*ulp_half == second_successor,
        negative_ties_to_even=-one-ulp_half == -one)
    if not all(tests.values()):
        raise ValueError('Polynomial interval kernel requires IEEE nearest/gradual-underflow/no-DAZ: '+str(tests))
    return tests


def sort_only_csc_copy(matrix, label):
    """Explicit coordinate-preserving sorting, with no duplicate summation.

    Some frozen graphs are unsorted unique CSC and cache canonical=False.
    Construct fresh arrays by integer-index permutation only; reject every
    duplicate coordinate instead of silently changing the numerical data.
    """
    if not sparse.isspmatrix_csc(matrix) or matrix.dtype != np.float64:
        raise ValueError(label+' must already be float64 CSC.')
    before = {name: hashlib.sha256(getattr(matrix, name).tobytes()).hexdigest()
        for name in ('data', 'indices', 'indptr')}
    values, indices = matrix.data.copy(), matrix.indices.copy()
    changed_columns = 0
    for column in range(matrix.shape[1]):
        lo, hi = matrix.indptr[column:column+2]
        order = np.argsort(matrix.indices[lo:hi], kind='stable')
        sorted_indices = matrix.indices[lo:hi][order]
        if np.any(np.diff(sorted_indices) == 0):
            raise ValueError(label+' has duplicate coordinates; sort-only preparation rejects them.')
        changed_columns += int(not np.array_equal(order, np.arange(hi-lo)))
        indices[lo:hi] = sorted_indices
        values[lo:hi] = matrix.data[lo:hi][order]
    result = sparse.csc_matrix((values, indices, matrix.indptr.copy()), shape=matrix.shape)
    if not result.has_canonical_format or result.nnz != matrix.nnz:
        raise ValueError('Coordinate-preserving CSC preparation failed.')
    after = {name: hashlib.sha256(getattr(result, name).tobytes()).hexdigest()
        for name in ('data', 'indices', 'indptr')}
    return result, dict(label=label, operation='Explicit sort-only copy of paired index/value entries; no sum_duplicates, normalization, duplicate coalescing or floating data arithmetic.',
        original_has_canonical_format=bool(matrix.has_canonical_format),
        original_has_sorted_indices=bool(matrix.has_sorted_indices), changed_columns=changed_columns,
        sorted_has_canonical_format=True, duplicate_coordinates=0, input_nnz=matrix.nnz,
        output_nnz=result.nnz, input_array_sha256=before, sorted_array_sha256=after,
        input_file_unchanged=True, mathematical_coordinate_multiset_preserved=True)


def norm1_upper(values):
    """Exact Fraction upper bound on the L1 norm of stored float64 values.

    A gamma_n inverse enclosure covers positive-sum rounding. Extra normal-
    threshold allowance covers subnormal terms lost to DAZ/FTZ in a residual
    norm reduction. Overflow is rejected, never represented by a small bound.
    """
    values = checked_float_array(values, 'norm values', allow_subnormal=True)
    observed = float(np.sum(np.abs(values), dtype=np.float64))
    if not math.isfinite(observed):
        raise ValueError('L1 norm reduction overflowed.')
    g = gamma(values.size+1)
    return (exact(observed)+2*values.size*TINY)/(1-g)


def raw_column_enclosure(P):
    """Enclose all exact positive raw-column sums without Fraction per edge."""
    P = sparse.csc_matrix(P, copy=False)
    if P.dtype != np.float64 or not P.has_canonical_format:
        raise ValueError('Raw P must be canonical float64 CSC, with no duplicates.')
    checked_float_array(P.data, 'raw P')
    if P.shape[0] != P.shape[1] or np.any(P.data < 0):
        raise ValueError('Raw P must be square and nonnegative.')
    degrees = np.diff(P.indptr)
    if np.any(degrees == 0):
        raise ValueError('Every raw column needs positive mass.')
    sums = np.asarray(P.sum(axis=0), dtype=np.float64).ravel()
    checked_float_array(sums, 'raw column sums')
    if np.any(sums <= 0):
        raise ValueError('Column summation lost positive mass.')
    degree = int(degrees.max())
    g = gamma(degree+1)
    # Inputs are nonnegative normals or zero. Their sums cannot underflow;
    # the same normal threshold allowance is retained conservatively.
    low = max(Q(0), (exact(float(sums.min()))-2*degree*TINY)/(1+g))
    high = (exact(float(sums.max()))+2*degree*TINY)/(1-g)
    if low <= 0:
        raise ValueError('Cannot certify a positive lower raw column sum.')
    delta = max(Q(0), 1-low, high-1)
    return dict(c_lo=low, c_hi=high, delta=delta, max_column_nnz=degree,
        observed_min_column_sum=float(sums.min()), observed_max_column_sum=float(sums.max()),
        gamma_column=g, stochasticity_note='For nonnegative raw P, ||P/c-P||_1=max_j|c_j-1| exactly; delta encloses it.')


class SparseResidualGuard:
    """Posterior residual bounds for an arbitrary stored approximate vector.

    Normal inputs and alpha<1 prevent DAZ amplification. We allow FTZ or
    subnormal intermediate results via four normal-threshold units per
    elementary operation, propagated through the maximum operation path.
    Relative-error terms use gamma_(2*maxRowNNZ+3). All bound arithmetic is
    exact Fraction arithmetic; observed sparse operations remain float64.
    Assumptions: IEEE binary64 round-to-nearest, standard correctly rounded
    basic add/subtract/multiply, and sparse matvec without unbounded fastmath.
    """
    def __init__(self, P, alpha):
        self.P = sparse.csc_matrix(P, copy=False)
        self.column = raw_column_enclosure(self.P)
        self.alpha = exact(alpha)
        checked_float_array([float(self.alpha)], 'alpha')
        if not 0 < self.alpha < 1:
            raise ValueError('alpha must be strictly between zero and one.')
        self.alpha_float = float(self.alpha)
        if exact(self.alpha_float) != self.alpha:
            raise ValueError('alpha must be exactly representable in stored binary64.')
        self.csr = self.P.tocsr()
        self.max_row_nnz = int(np.diff(self.csr.indptr).max())
        self.path_operations = 2*self.max_row_nnz+3
        self.gamma_path = gamma(self.path_operations)
        self.total_operations = 2*self.P.nnz+3*self.P.shape[0]
        self.underflow_allowance = 4*self.total_operations*TINY/(1-self.path_operations*U)
        self.kappa_canonical = 1/(1-self.alpha)
        old_norm = max(Q(1), self.column['c_hi'])
        if self.alpha*old_norm >= 1:
            raise ValueError('Old raw-background resolvent cannot be bounded by this norm contract.')
        self.kappa_old = 1/(1-self.alpha*old_norm)

    def residual_upper(self, v, rhs_float, rhs_conversion_error=Q(0), old_background=False):
        v = checked_float_array(v, 'stored solution')
        rhs = checked_float_array(rhs_float, 'stored RHS')
        if v.shape != (self.P.shape[0],) or rhs.shape != v.shape:
            raise ValueError('Vector and raw graph dimensions differ.')
        conversion = exact(rhs_conversion_error)
        if conversion < 0:
            raise ValueError('RHS conversion upper error must be nonnegative.')
        with np.errstate(over='raise', invalid='raise', under='ignore'):
            product = self.csr@v
            residual_float = (rhs-v)+self.alpha_float*product
        checked_float_array(product, 'observed sparse matvec', allow_subnormal=True)
        checked_float_array(residual_float, 'observed residual', allow_subnormal=True)
        nv, nb = norm1_upper(v), norm1_upper(rhs)
        observed_upper = norm1_upper(residual_float)
        arithmetic = self.gamma_path*(nb+nv+self.alpha*self.column['c_hi']*nv)
        raw_upper = observed_upper+arithmetic+self.underflow_allowance+conversion
        normalization = self.alpha*self.column['delta']*nv
        old_perturbation = normalization if old_background else Q(0)
        upper = raw_upper+normalization+old_perturbation
        return upper, dict(upper=upper, observed_residual_l1=float(np.sum(abs(residual_float))),
            observed_residual_norm_upper=observed_upper, vector_norm_upper=nv,
            rhs_norm_upper=nb, sparse_expression_rounding_upper=arithmetic,
            underflow_FTZ_DAZ_intermediate_upper=self.underflow_allowance,
            rhs_exact_to_float_error=conversion, raw_expression_residual_upper=raw_upper,
            canonical_column_normalization_correction=normalization,
            old_raw_background_perturbation=old_perturbation,
            scope='Old target uniform coefficient residual contract' if old_background else 'Canonical P/c base-system residual',
            path_operations=self.path_operations, gamma_path=self.gamma_path,
            max_row_nnz=self.max_row_nnz)


def residual_upper(P, v, rhs_float, rhs_conversion_error, alpha, column_contract=None, old_background=False):
    """Convenience API for independent tiny-graph guard testing."""
    guard = SparseResidualGuard(P, alpha)
    if column_contract is not None and column_contract != guard.column:
        raise ValueError('Supplied column enclosure does not describe this graph.')
    return guard.residual_upper(v, rhs_float, rhs_conversion_error, old_background)


def exact_sparse_rhs(n, coordinates):
    """Round an exact sparse RHS once and retain its exact L1 conversion error."""
    result = np.zeros(n)
    error = Q(0)
    for index, value in coordinates.items():
        value = exact(value)
        rounded = float(value)
        if not math.isfinite(rounded):
            raise ValueError('RHS conversion overflowed.')
        result[index] = rounded
        error += abs(value-exact(rounded))
    checked_float_array(result, 'rounded exact RHS')
    return result, error


def source_columns_exact(P, M, sources, alpha):
    """Exact dyadic raw partitions, normalized masses, and canonical atom RHS."""
    if not sparse.isspmatrix_csc(P) or not sparse.isspmatrix_csc(M) or P.shape != M.shape:
        raise ValueError('Source P/M must be aligned CSC matrices.')
    if P.dtype != np.float64 or M.dtype != np.float64 or not P.has_canonical_format or not M.has_canonical_format:
        raise ValueError('Source P/M must be canonical float64, with no duplicate coordinates.')
    output = []
    n = P.shape[0]
    alpha = exact(alpha)
    for source in sources:
        lo, hi = P.indptr[source:source+2]
        ml, mh = M.indptr[source:source+2]
        rows = [int(x) for x in P.indices[lo:hi]]
        p = {row: exact(value) for row, value in zip(rows, P.data[lo:hi])}
        mention = {int(row): exact(value) for row, value in zip(M.indices[ml:mh], M.data[ml:mh])}
        if any(row not in p for row in mention) or mention.get(source, Q(0)) != 0:
            raise ValueError('Mention must be a self-free subdistribution of raw P.')
        total = sum(p.values(), Q(0))
        if total <= 0:
            raise ValueError('Empty source column.')
        m = {row: mention.get(row, Q(0)) for row in rows}
        w = {row: p[row]-m[row]-(p[row] if row == source else 0) for row in rows}
        if any(value < 0 for value in m.values()) or any(value < 0 for value in w.values()):
            raise ValueError('Exact raw partition is negative.')
        masses = [sum(group.values(), Q(0))/total for group in (m, w)]
        atom_rhs = []
        for group, mass in zip((m, w), masses):
            exact_rhs = {row: -alpha*(group[row]/total-mass*p[row]/total) for row in rows}
            rhs, conversion = exact_sparse_rhs(n, exact_rhs)
            atom_rhs.append(dict(rhs=rhs, rhs_conversion_error=conversion))
        output.append(dict(source=int(source), raw_total=total, masses=masses,
            normalized_self=p.get(source, Q(0))/total, atom_rhs=atom_rhs,
            raw_partition=dict(rows=rows, original=[p[row] for row in rows],
                mention=[m[row] for row in rows], weak=[w[row] for row in rows],
                self_loop=[p[row] if row == source else Q(0) for row in rows])))
    return output


def serial(value):
    if isinstance(value, Q):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('w', encoding='utf-8') as handle:
        json.dump(serial(value), handle, indent=2, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def natural_first_question(folder, output, oracle, tolerance):
    started = perf_counter()
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    with (folder/'queries.jsonl').open(encoding='utf-8') as handle:
        raw = json.loads(next(handle))
    query = {key: raw[key] for key in ('qid', 'sources', 'seed_indices', 'seed_values')}
    query['original_question_index'] = raw.get('original_question_index', 0)
    if query['original_question_index'] != 0 or len(set(query['sources'])) != len(query['sources']):
        raise ValueError('Require the unchanged first original query and unique source IDs.')
    if len(set(query['seed_indices'])) != len(query['seed_indices']) or len(query['seed_indices']) != len(query['seed_values']):
        raise ValueError('Seed coordinates must be distinct and aligned.')
    P = sparse.load_npz(folder/'graph.npz')
    M = sparse.load_npz(folder/'mention_subdistribution.npz')
    if not sparse.isspmatrix_csc(P) or not sparse.isspmatrix_csc(M):
        raise ValueError('Frozen packages must already be CSC; no implicit canonicalization.')
    if P.shape != M.shape or P.shape != (manifest['nodes'], manifest['nodes']):
        raise ValueError('Frozen P/M graph dimensions disagree.')
    P, P_sort = sort_only_csc_copy(P, 'raw P')
    M, M_sort = sort_only_csc_copy(M, 'raw M')
    checked_float_array(M.data, 'raw mention')
    if np.any(M.data < 0):
        raise ValueError('Raw mention weights must be nonnegative.')
    alpha = exact(manifest['alpha'])
    guard = SparseResidualGuard(P, alpha)
    n = P.shape[0]
    source_ids = list(map(int, query['sources']))
    if len(source_ids) < 1 or len(source_ids) > 4 or any(s < 0 or s >= n for s in source_ids):
        raise ValueError('Prototype requires the complete original one-to-four sources.')
    seed = np.zeros(n)
    seed[query['seed_indices']] = query['seed_values']
    checked_float_array(seed, 'stored seed')
    if np.any(seed < 0):
        raise ValueError('Seed must be nonnegative.')
    rhs_base, conversion_base = exact_sparse_rhs(n, {int(i): (1-alpha)*exact(value)
        for i, value in zip(query['seed_indices'], query['seed_values'])})
    source_columns = source_columns_exact(P, M, source_ids, alpha)
    # A numerical backend supplies approximate stored vectors only. Every
    # residual contract is established independently afterwards.
    sys.path.insert(0, str(ROOT))
    from src.baselines.prpack import PRPackViewRank
    ranker = PRPackViewRank(P, alpha=float(alpha), tol=tolerance)
    t = perf_counter()
    try:
        y = ranker.base_lu.solve(rhs_base)
        R_columns, rhs_errors = [], []
        for group in range(2):
            for source in source_columns:
                spec = source['atom_rhs'][group]
                R_columns.append(ranker.base_lu.solve(spec['rhs']))
                rhs_errors.append(spec['rhs_conversion_error'])
        R = np.column_stack(R_columns)
    finally:
        ranker.base_lu.graph.close()
    solver_seconds = perf_counter()-t
    runtime = kernel_runtime_guard()
    rho_y, base_guard = guard.residual_upper(y, rhs_base, conversion_base, old_background=True)
    rho_atoms, atom_guards = [], []
    for j, column in enumerate(R_columns):
        group, source_index = divmod(j, len(source_ids))
        spec = source_columns[source_index]['atom_rhs'][group]
        rho, audit = guard.residual_upper(column, spec['rhs'], spec['rhs_conversion_error'], old_background=True)
        rho_atoms.append(rho); atom_guards.append(audit)
    mu = [column['masses'][0] for column in source_columns]
    nu = [column['masses'][1] for column in source_columns]
    domain = compile_domain(R[source_ids, :len(source_ids)], R[source_ids, len(source_ids):],
        mu, nu, y[source_ids], ((1, 4), (.25, 2)), source_ids)
    ids = np.arange(n)
    base_order = np.lexsort((ids, -y))
    keep_sets = []
    candidates = set(source_ids)|set(map(int, base_order[:128]))
    for column in R_columns:
        rows = set(map(int, np.lexsort((ids, -np.abs(column)))[:128]))|set(source_ids)
        candidates.update(rows)
        keep_sets.append(rows)
    candidate_rows = sorted(candidates)
    mask = [[node in keep for keep in keep_sets] for node in candidate_rows]
    tails = []
    for j, keep in enumerate(keep_sets):
        outside = np.ones(n, dtype=bool)
        outside[list(keep)] = False
        maximum = float(np.max(np.abs(R[outside, j]))) if np.any(outside) else 0.
        tails.append(math.nextafter(maximum, math.inf))
    certificate_results = []
    t = perf_counter()
    for depth in (0, 2):
        for k in KS:
            selected = list(map(int, base_order[:k]))
            result = certify(domain, y, candidate_rows, R[candidate_rows, :], mask, tails,
                selected, kappa=guard.kappa_old, base_residual=rho_y,
                atom_residuals=rho_atoms, round_pad=0, subdivide_depth=depth)
            certificate_results.append(dict(requested_subdivision_depth=depth, **result))
    certificate_seconds = perf_counter()-t
    output.mkdir(parents=True, exist_ok=False)
    vectors_path = output/'stored_vectors.npz'
    np.savez_compressed(vectors_path, y=y, R=R, seed=seed,
        candidate_rows=np.array(candidate_rows, dtype=np.int64),
        retained_mask=np.array(mask, dtype=bool), tail_bounds=np.array(tails),
        source_ids=np.array(source_ids, dtype=np.int64))
    sanity = []
    t = perf_counter()
    for category, points in (('original_16_grid', GRID), ('frozen_9_interiors', INTERIORS)):
        for weights in points:
            view = oracle.actual_view(P, M, source_ids, weights)
            scores, iterations, measured_residual = oracle.solve_reference(view, seed, float(alpha), 5e-15)
            order = np.lexsort((ids, -scores))[:max(KS)]
            memberships = {k: sorted(map(int, order[:k])) for k in KS}
            conflicts = [dict(k=result['k'], depth=result['requested_subdivision_depth'])
                for result in certificate_results if result['status'] == 'STABLE'
                and sorted(result['selected_ids']) != memberships[result['k']]]
            sanity.append(dict(category=category, weights=weights, memberships=memberships,
                iterations=iterations, measured_float_actual_view_residual_l1=measured_residual,
                stable_membership_conflicts=conflicts))
    sanity_seconds = perf_counter()-t
    record = dict(dataset=folder.name, query_index=0, query=query, nodes=n,
        input_canonical_checks=dict(P_CSC=True, M_CSC=True, shape_equal=True,
            float64=True, no_duplicate_entries=True, no_duplicate_summation=True,
            explicit_sort_only_copies=[P_sort, M_sort]),
        alpha=alpha, column_contract=guard.column, kappa_canonical=guard.kappa_canonical,
        kappa_old_raw_background=guard.kappa_old,
        uniform_old_transition_l1_norm_upper=max(Q(1), guard.column['c_hi']),
        resolvent_denominator_lower=1-alpha*max(Q(1), guard.column['c_hi']),
        arithmetic_guard_assumptions=[
            'IEEE binary64 round-to-nearest basic add/subtract/multiply; sparse matvec does not use lower precision or unbounded fastmath.',
            'P, stored y/R, RHS and alpha are finite; nonzero subnormal operands are rejected to avoid DAZ amplification.',
            'Intermediate underflow/FTZ is charged by four normal-threshold units per operation with 1/(1-path*u) propagation.',
            'Positive sum and norm reductions are bounded by exact Fraction gamma_n inverse enclosures; all bound arithmetic is exact Fraction arithmetic.',
            'All observed sparse products/residual/norm reductions must remain finite; overflow rejects execution.'
        ],
        polynomial_interval_runtime_guard=runtime,
        base_guard=base_guard, atom_guards=atom_guards,
        source_metadata=[dict(source=c['source'], exact_raw_total=c['raw_total'],
            canonical_masses=c['masses'], canonical_self=c['normalized_self']) for c in source_columns],
        certificate_results=certificate_results, candidate_count=len(candidate_rows),
        retained_top_rule='128 base top(y) + topabs128 per atom + every original source; all source atom coordinates retained.',
        all_sanity= sanity, stable_sanity_conflicts=sum(len(s['stable_membership_conflicts']) for s in sanity),
        stored_vectors_sha256=sha(vectors_path),
        diagnostic_seconds=dict(approximate_vector_backend=solver_seconds,
            certificate_kernel=certificate_seconds, cold_PI_sanity=sanity_seconds,
            total_to_record=perf_counter()-started),
        math_target='non-source raw P entries unchanged; selected source raw dyadic mention/weak/self partition multiplied by (m,w,1) and normalized in exact arithmetic.',
        canonical_reference='Pprime=P/c on every column; source canonical c cancels in mathematical target. Raw non-source background perturbation is explicitly charged to every residual coefficient, and old kappa covers max(1,c_hi).',
        sanity_scope='Float raw-column materialization by existing independent oracle, including its rounded weak subtraction and column normalization. These 25 point solves are numerical consistency checks, not the proof and not bitwise identical to the exact source-partition mathematical target.',
        round_pad_scope='zero: exact stored-coefficient polynomial arithmetic plus outward intervals and independently bounded vector residuals. No unproved generic epsilon pad is used. Literal floating source-matrix materialization is not covered as an exact-byte target.',
        no_performance_or_memory_claim=True, full_response_scratch_not_budgeted_ranker=True,
        status='needs_review' if any(s['stable_membership_conflicts'] for s in sanity) else 'Minimal fixed-first-query conditional natural certificate demonstration; preserve UNKNOWN outcomes.')
    write_json(output/'result.json', record)
    return record


def run_natural(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Use a new empty output directory; partial/completed runs are never overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    protocol = Path(args.scope).resolve()
    if not protocol.exists():
        raise RuntimeError('The pre-run fixed-two-query scope is missing.')
    paths = [Path(__file__), ROOT/'src/policy_domain_polynomial.py',
        HERE/'build_multigroup_oracle.py', ROOT/'src/baselines/prpack.py',
        ROOT/'src/viewrank.py', ROOT/'baselines/build/Release/prpack_native.dll', protocol]
    hashes = {str(path): sha(path) for path in paths}
    snapshot = output/'source_snapshot'; snapshot.mkdir()
    for index, path in enumerate(paths):
        shutil.copy2(path, snapshot/f'{index:02d}_{path.name}')
    inputs = {dataset: {name: sha(Path(args.input_root)/dataset/name) for name in
        ('manifest.json', 'queries.jsonl', 'graph.npz', 'mention_subdistribution.npz')} for dataset in DATASETS}
    write_json(output/'job_manifest.json', dict(frozen_scope=protocol.read_text(encoding='utf-8'),
        source_sha256=hashes, input_sha256=inputs, arguments=vars(args),
        datasets=DATASETS, queries_per_dataset=1, original_query_index=0, ks=KS,
        box=((1, 4), (.25, 2)), depths=(0, 2), L=128, grid=GRID, interiors=INTERIORS,
        execution_scope='No timing comparison. Full-vector responses are scratch for a conditional mathematical certificate demonstration.'))
    from threadpoolctl import threadpool_limits
    sys.path.insert(0, str(ROOT/'scripts'))
    from pin_timing_worker import pin_timing_worker
    pin_timing_worker(31)
    oracle = load_module(HERE/'build_multigroup_oracle.py', 'independent_natural_sanity_oracle')
    results = []
    with threadpool_limits(limits=1):
        for dataset in DATASETS:
            result = natural_first_question(Path(args.input_root)/dataset, output/dataset, oracle, args.tolerance)
            results.append(result)
            print(json.dumps(dict(dataset=dataset, query_index=0,
                certificates=[dict(k=c['k'], depth=c['requested_subdivision_depth'], status=c['status']) for c in result['certificate_results']],
                stable_sanity_conflicts=result['stable_sanity_conflicts'])), flush=True)
            if result['stable_sanity_conflicts']:
                break
    drift = [path for path, old in hashes.items() if sha(path) != old]
    conflicts = sum(r['stable_sanity_conflicts'] for r in results)
    summary = dict(complete_datasets=len(results), expected_datasets=2,
        original_questions=len(results), source_drift=drift,
        stable_certificates=sum(c['status']=='STABLE' for r in results for c in r['certificate_results']),
        unknown_certificates=sum(c['status']=='UNKNOWN' for r in results for c in r['certificate_results']),
        sanity_views=sum(len(r['all_sanity']) for r in results),
        stable_sanity_conflicts=conflicts,
        status='needs_review' if conflicts else 'complete_conditional_math_target_demonstration',
        no_performance_or_memory_claim=True, mathematical_target_and_float_sanity_scope_distinct=True,
        interpretation='Fixed original first query per corpus only. Contracts, exact target, retained full-vector evidence and every STABLE/UNKNOWN outcome are recorded. No full workload, acceptance, speed, RSS or cache-budget conclusion.')
    write_json(output/'summary.json', summary)
    if drift:
        raise RuntimeError('Source changed during this demonstration.')
    print(json.dumps(dict(output=str(output), **summary)), flush=True)
    if conflicts:
        raise SystemExit(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', default=str(ROOT/'results/natural_policy_workload'))
    parser.add_argument('--scope', default=str(ROOT/'notes/natural_two_query_scope.json'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--tolerance', type=float, default=1e-12)
    args = parser.parse_args()
    if args.tolerance != 1e-12:
        raise ValueError('This fixed diagnostic scope uses backend tolerance 1e-12.')
    run_natural(args)

