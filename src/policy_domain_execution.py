"""Budgeted retained representations for continuous two-group policy domains.

Shared graph/backend/guard inputs, current seed/base vectors, one generated
full-response/RHS scratch, actual-view point-solve scratch and returned output
are separate from the private retained numeric budget. Cache atoms, exact
mass/error metadata, registered stencil, source core, candidate matrices and
polynomial/scan reservations belong to that budget. Their sizes are exposed.
No whole n*q response family is relabeled as free scratch.
"""
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
import importlib
import math
from time import perf_counter

import numpy as np
from scipy import sparse

from scripts.natural_domain_adapter import (SparseResidualGuard, exact,
    exact_sparse_rhs, kernel_runtime_guard, norm1_upper, sort_only_csc_copy,
    source_columns_exact)
from .policy_domain_polynomial import compile_domain, certify as certify_reference
from .policy_domain_source import source_columns_fast, point_columns_fast


DEFAULT_BOX = ((1., 4.), (.25, 2.))
GRID = tuple((m, w) for m in (1., 2., 3., 4.) for w in (.25, .5, 1., 2.))


def fraction_bytes(value):
    value = exact(value)
    return 16+(abs(value.numerator).bit_length()+7)//8+(value.denominator.bit_length()+7)//8


def numeric_payload(value, seen=None):
    """Logical numeric payload; object/container overhead belongs to OS RSS."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, Fraction):
        return fraction_bytes(value)
    if isinstance(value, (float, int, np.number)):
        return 8
    if isinstance(value, dict):
        return sum(numeric_payload(k, seen)+numeric_payload(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sum(numeric_payload(v, seen) for v in value)
    if hasattr(value, '__dict__'):
        return numeric_payload(vars(value), seen)
    return 0


def _top(values, k):
    """Global score/ID deterministic selection, including boundary ties."""
    values = np.asarray(values)
    k = min(int(k), len(values))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k == len(values):
        return np.lexsort((np.arange(len(values)), -values))
    threshold = np.partition(values, len(values)-k)[len(values)-k]
    above = np.flatnonzero(values > threshold)
    equal = np.flatnonzero(values == threshold)[:k-len(above)]
    chosen = np.concatenate((above, equal))
    return chosen[np.lexsort((chosen, -values[chosen]))]


def _float_upper(value):
    value = exact(value)
    nearest = float(value)
    return math.nextafter(nearest, math.inf) if exact(nearest) < value else nearest


@dataclass
class DomainAtom:
    key: tuple
    mass: Fraction
    rows: np.ndarray
    values: np.ndarray
    rho: float
    norm_upper: float
    tail: float
    response: object = None
    zero: bool = False
    generation_tol: float = math.inf
    keep: int = 0

    @property
    def nbytes(self):
        arrays = (0 if self.response is None else self.response.nbytes)+self.rows.nbytes+self.values.nbytes
        return arrays+80+fraction_bytes(self.mass)

    def lookup(self, nodes):
        nodes = np.asarray(nodes, dtype=np.int64)
        if self.zero:
            return np.zeros(len(nodes)), np.ones(len(nodes), dtype=bool)
        if self.response is not None:
            return self.response[nodes], np.ones(len(nodes), dtype=bool)
        positions = np.searchsorted(self.rows, nodes)
        valid = positions < len(self.rows)
        valid[valid] &= self.rows[positions[valid]] == nodes[valid]
        values = np.zeros(len(nodes))
        values[valid] = self.values[positions[valid]]
        return values, valid


class DomainCompiler:
    """Shared read-only mathematical raw graph and validated residual guard."""
    def __init__(self, ranker, mention, atom_tolerance=1e-12, point_tolerance=1e-12,
                 source_compiler='fast'):
        if source_compiler not in ('fast', 'exact'):
            raise ValueError('Source compiler is fast or exact.')
        self.source_compiler = source_compiler
        self.ranker = ranker
        self.P, self.P_sort = sort_only_csc_copy(ranker.P, 'domain raw P')
        self.M, self.M_sort = sort_only_csc_copy(sparse.csc_matrix(mention), 'domain raw M')
        if self.P.shape != self.M.shape:
            raise ValueError('P/M dimensions differ.')
        self.n, self.alpha = ranker.n, exact(ranker.alpha)
        self.guard = SparseResidualGuard(self.P, self.alpha)
        self.atom_tolerance, self.point_tolerance = float(atom_tolerance), float(point_tolerance)
        self.runtime = kernel_runtime_guard()
        self.atom_solves = self.point_solves = self.base_solves = 0
        self.generated_full_response_peak_bytes = 0
        self.source_rhs_peak_bytes = 0
        self.exact_source_compilations = 0
        self.point_source_compilations = 0

    @property
    def shared_input_bytes(self):
        return sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes for a in (self.P, self.M, self.guard.csr))+numeric_payload(self.guard.column)

    def source_spec(self, source):
        self.exact_source_compilations += 1
        function = source_columns_fast if self.source_compiler == 'fast' else source_columns_exact
        spec = function(self.P, self.M, [source], self.alpha)[0]
        self.source_rhs_peak_bytes = max(self.source_rhs_peak_bytes,
            sum(item['rhs'].nbytes for item in spec['atom_rhs'])+numeric_payload(spec['raw_partition']))
        return spec

    def base(self, seed):
        seed = np.asarray(seed, dtype=np.float64)
        coordinates = {int(i): (1-self.alpha)*exact(seed[i]) for i in np.flatnonzero(seed)}
        rhs, error = exact_sparse_rhs(self.n, coordinates)
        previous = self.ranker.base_lu.tol
        self.ranker.base_lu.tol = self.atom_tolerance
        try:
            y = self.ranker.base_lu.solve(rhs)
            self.base_solves += 1
        finally:
            self.ranker.base_lu.tol = previous
        rho, audit = self.guard.residual_upper(y, rhs, error, old_background=True)
        return y, rho, audit

    def point(self, seed, sources, weights, ks=(2, 5, 10), materialize=False):
        """Actual-view posterior point certificate, not a sample-domain proof.

        Exact source columns are rounded once for a solver input. Independent
        raw residual and exact source conversion error cover the mathematical
        source-normalized/raw-background target, including update rounding.
        """
        sources = np.asarray(sources, dtype=np.int64)
        m, w = map(exact, weights)
        if min(m, w) <= 0:
            raise ValueError('Point weights must be positive.')
        view = self.P.copy()
        conversion_norm = Fraction(0)
        if self.source_compiler == 'fast':
            columns = point_columns_fast(self.P, self.M, sources, (m, w))
            self.point_source_compilations += len(columns)
            for column in columns:
                lo, hi = view.indptr[column['source']:column['source']+2]
                view.data[lo:hi] = column['values']
                conversion_norm = max(conversion_norm, column['column_conversion_error'])
        else:
            for source in sources:
                spec = source_columns_exact(self.P, self.M, [int(source)], self.alpha)[0]
                self.point_source_compilations += 1
                raw = spec['raw_partition']
                numerators = [m*a+w*b+c for a, b, c in zip(raw['mention'], raw['weak'], raw['self_loop'])]
                total = sum(numerators, Fraction(0))
                normalized = [value/total for value in numerators]
                lo, hi = view.indptr[source:source+2]
                rounded = np.array([float(value) for value in normalized])
                view.data[lo:hi] = rounded
                conversion_norm = max(conversion_norm, sum(abs(value-exact(rounded[i])) for i, value in enumerate(normalized)))
        rhs, rhs_error = exact_sparse_rhs(self.n, {int(i): (1-self.alpha)*exact(seed[i]) for i in np.flatnonzero(seed)})
        D = (view[:, sources]-self.P[:, sources]).tocsc()
        graph = self.ranker.base_lu.graph
        graph.update_columns(sources, D)
        try:
            scores, native_stats = graph.solve_linear(rhs, float(self.alpha), self.point_tolerance)
            self.point_solves += 1
        finally:
            graph.update_columns([], sparse.csc_matrix((self.n, 0)))
        point_guard = SparseResidualGuard(view, self.alpha)
        _, audit = point_guard.residual_upper(scores, rhs, rhs_error)
        rho = audit['raw_expression_residual_upper']+self.alpha*conversion_norm*audit['vector_norm_upper']
        bound = self.guard.kappa_old*rho
        order = _top(scores, min(max(ks)+1, self.n))
        gaps = {k: None if k == self.n else exact(scores[order[k-1]])-exact(scores[order[k]]) for k in ks}
        flags = {k: k == self.n or gaps[k] > 2*bound for k in ks}
        result = dict(weights=tuple(map(float, weights)), sets={k: order[:k].copy() for k in ks},
            selected_scores={k: scores[order[:k]].copy() for k in ks}, strict_flags=flags,
            score_error_bound=bound, residual_upper=rho, source_column_rounding_norm_upper=conversion_norm,
            raw_gaps=gaps, certified_gap_lower_bounds={k: None if gaps[k] is None else gaps[k]-2*bound for k in ks},
            native_stats=native_stats)
        if materialize:
            result['scores'] = scores
        return result


class DomainDirectory:
    """One global (group,source) LRU and pin set with honest live reservations."""
    MODES = ('sparse', 'dense', 'dense_pruned', 'no_core', 'scan', 'always_full')
    def __init__(self, compiler, stencil, budget_bytes=16*1024**2, keep=128, mode='sparse',
                 preview_keep=0, scan_backend='outward'):
        if mode not in self.MODES or min(keep, preview_keep) < 0 or budget_bytes <= 0:
            raise ValueError('Invalid mode, L or budget.')
        if scan_backend not in ('outward', 'dot_fallback'):
            raise ValueError('Scan backend is outward or dot_fallback.')
        self.compiler, self.mode = compiler, mode
        self.budget_bytes, self.keep = int(budget_bytes), min(int(keep), compiler.n)
        self.preview_keep = min(int(preview_keep), compiler.n)
        self.scan_backend = scan_backend
        use_stencil = mode in ('sparse', 'dense_pruned', 'scan')
        self.stencil = {int(source): np.unique(np.asarray(rows, dtype=np.int32)) for source, rows in stencil.items()} if use_stencil else {}
        self.plan_bytes = sum(rows.nbytes+8 for rows in self.stencil.values())
        if self.plan_bytes > self.budget_bytes:
            raise MemoryError('Registered stencil exceeds this private budget.')
        self.cache, self.pins, self.live = OrderedDict(), {}, []
        self._cache_bytes = self._live_bytes = self._live_reserved_bytes = self._pending_bytes = 0
        self.hits = self.misses = self.evictions = 0
        self.refinement_invalidations = self.preview_selections = 0
        self.peak_bytes = self.peak_reserved_bytes = self.plan_bytes

    @property
    def nbytes(self):
        return self.plan_bytes+self._cache_bytes+self._live_bytes

    @property
    def reserved_bytes(self):
        return self.plan_bytes+self._cache_bytes+self._live_reserved_bytes+self._pending_bytes

    @property
    def memory(self):
        return dict(total_private_retained_numeric_bytes=self.nbytes,
            peak_private_retained_numeric_bytes=self.peak_bytes,
            admission_reserved_numeric_bytes=self.reserved_bytes,
            peak_admission_reserved_numeric_bytes=self.peak_reserved_bytes,
            cache_atom_bytes=self._cache_bytes, live_family_bytes=self._live_bytes,
            stencil_bytes=self.plan_bytes, pending_bytes=self._pending_bytes,
            preview_keep=self.preview_keep, preview_selections=self.preview_selections,
            refinement_invalidations=self.refinement_invalidations,
            generated_full_response_peak_bytes=self.compiler.generated_full_response_peak_bytes,
            generation_scratch_scope='One full atom response/RHS and exact source compilation scratch; no n*q free basis.')

    def clear(self):
        if self.live or self.pins:
            raise RuntimeError('Release live families before clearing their pinned cache.')
        self.cache.clear(); self._cache_bytes = 0
        self._track()

    def invalidate_for_refinement(self, keys):
        """Explicit accuracy upgrade, separate from capacity/LRU eviction."""
        if self.live or self.pins or self._pending_bytes:
            raise RuntimeError('Release all live/pinned families before invalidating atoms.')
        removed = []
        for key in dict.fromkeys(tuple(map(int, key)) for key in keys):
            atom = self.cache.pop(key, None)
            if atom is not None:
                self._cache_bytes -= atom.nbytes
                self.refinement_invalidations += 1
                removed.append(key)
        self._track()
        return tuple(removed)

    def compatible(self, atom, keep, tolerance):
        # A stored zero vector/RHS error enclosure cannot improve through a
        # tighter backend stopping tolerance or additional zero coordinates.
        return atom.zero or atom.generation_tol <= float(tolerance) and atom.keep >= min(int(keep), self.compiler.n)

    def _track(self):
        self.peak_bytes = max(self.peak_bytes, self.nbytes)
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, self.reserved_bytes)
        assert 0 <= self.nbytes <= self.reserved_bytes <= self.budget_bytes

    def _evict(self, extra=0, protected=()):
        protected = set(protected)
        while self.reserved_bytes+extra > self.budget_bytes:
            key = next((key for key in self.cache if key not in self.pins and key not in protected), None)
            if key is None:
                return False
            self._cache_bytes -= self.cache.pop(key).nbytes
            self.evictions += 1
        return True

    def _atom_estimate(self, source, preview_count=0):
        if self.mode in ('dense', 'dense_pruned', 'always_full'):
            rows = min(self.compiler.n, self.keep+len(self.stencil.get(source, ()))+preview_count+1)
            return self.compiler.n*8+(4*rows if self.mode == 'dense_pruned' else 0)+1024
        rows = min(self.compiler.n, self.keep+len(self.stencil.get(source, ()))+preview_count+1)
        return rows*12+1024

    def _build_atom(self, key, spec, preview_rows=()):
        group, source = key
        compiler = self.compiler
        rhs_spec = spec['atom_rhs'][group]
        rhs = rhs_spec['rhs']
        mass = spec['masses'][group]
        tags = dict(generation_tol=float(compiler.atom_tolerance),
            keep=compiler.n if self.mode in ('dense', 'always_full') else self.keep)
        if not np.any(rhs):
            return DomainAtom(key, mass, np.empty(0, dtype=np.int32), np.empty(0),
                _float_upper(rhs_spec['rhs_conversion_error']), 0., 0., zero=True, **tags)
        previous = compiler.ranker.base_lu.tol
        compiler.ranker.base_lu.tol = compiler.atom_tolerance
        try:
            response = compiler.ranker.base_lu.solve(rhs)
            compiler.atom_solves += 1
        finally:
            compiler.ranker.base_lu.tol = previous
        compiler.generated_full_response_peak_bytes = max(compiler.generated_full_response_peak_bytes, response.nbytes)
        if self.mode in ('dense', 'dense_pruned', 'always_full'):
            rho, audit = compiler.guard.residual_upper(response, rhs, rhs_spec['rhs_conversion_error'], old_background=True)
            rows = np.empty(0, dtype=np.int32)
            tail = 0.
            if self.mode == 'dense_pruned':
                rows = np.union1d(np.union1d(_top(abs(response), self.keep),
                    self.stencil.get(source, np.empty(0, dtype=np.int32))), [source]).astype(np.int32)
                rows = np.union1d(rows, preview_rows).astype(np.int32)
                omitted = np.ones(compiler.n, dtype=bool); omitted[rows] = False
                maximum = float(np.max(abs(response[omitted]))) if np.any(omitted) else 0.
                tail = math.nextafter(maximum, math.inf) if maximum else 0.
                rows.setflags(write=False)
            response.setflags(write=False)
            return DomainAtom(key, mass, rows, np.empty(0), _float_upper(rho),
                _float_upper(audit['vector_norm_upper']), tail, response=response, **tags)
        rows = np.union1d(_top(abs(response), self.keep), self.stencil.get(source, np.empty(0, dtype=np.int32)))
        rows = np.union1d(rows, preview_rows)
        if self.mode != 'no_core':
            rows = np.union1d(rows, [source])
        rows = rows.astype(np.int32)
        values = response[rows].copy()
        if self.mode == 'no_core':
            model = np.zeros(compiler.n)
            model[rows] = values
            rho, audit = compiler.guard.residual_upper(model, rhs, rhs_spec['rhs_conversion_error'], old_background=True)
            tail = 0.
        else:
            rho, audit = compiler.guard.residual_upper(response, rhs, rhs_spec['rhs_conversion_error'], old_background=True)
            omitted = np.ones(compiler.n, dtype=bool); omitted[rows] = False
            maximum = float(np.max(abs(response[omitted]))) if np.any(omitted) else 0.
            tail = math.nextafter(maximum, math.inf) if maximum else 0.
        rows.setflags(write=False); values.setflags(write=False)
        return DomainAtom(key, mass, rows, values, _float_upper(rho), _float_upper(audit['vector_norm_upper']), tail, **tags)

    def prepare(self, seed, sources, base_scores=None, base_rho=None, ks=(2, 5, 10)):
        sources = np.asarray(sources, dtype=np.int64)
        if len(set(map(int, sources))) != len(sources) or len(sources) > 4:
            raise ValueError('Keep all original unique sources, at most four.')
        if np.any(sources < 0) or np.any(sources >= self.compiler.n):
            raise ValueError('Source outside graph.')
        seed = np.asarray(seed, dtype=np.float64)
        if seed.shape != (self.compiler.n,):
            raise ValueError('Seed dimension differs.')
        if base_scores is None:
            y, rho, base_audit = self.compiler.base(seed)
        else:
            y = np.asarray(base_scores, dtype=np.float64)
            rhs, conversion = exact_sparse_rhs(self.compiler.n, {int(i): (1-self.compiler.alpha)*exact(seed[i]) for i in np.flatnonzero(seed)})
            guarded, base_audit = self.compiler.guard.residual_upper(y, rhs, conversion, old_background=True)
            # An externally supplied rho may conservatively enlarge, never
            # replace, this independently computed natural residual guard.
            rho = max(guarded, exact(base_rho)) if base_rho is not None else guarded
        keys = tuple((group, int(source)) for group in range(2) for source in sources)
        incompatible = [key for key in keys if key in self.cache and
            not self.compatible(self.cache[key], self.keep, self.compiler.atom_tolerance)]
        if incompatible:
            raise ValueError('Requested atom refinement requires explicit unpinned invalidation: '+str(incompatible))
        preview = _top(y, self.preview_keep).astype(np.int32) if self.preview_keep else np.empty(0, dtype=np.int32)
        self.preview_selections += bool(self.preview_keep)
        registered = self.mode not in ('sparse', 'scan') or all(
            set(map(int, sources)) <= set(map(int, self.stencil.get(int(source), ()))) for source in sources)
        n = self.compiler.n
        q = len(keys)
        # Existing atoms can carry a fine/larger sketch than this request.
        # Admission must account for their actual mixed retained row counts.
        row_upper = sum(len(self.cache[key].rows) if key in self.cache else
            min(n, self.keep+len(self.stencil.get(key[1], ()))+len(preview)+1) for key in keys)
        max_candidate = (n if self.mode in ('dense', 'always_full', 'scan') else
            min(n, min(128, n)+row_upper+len(sources)+len(preview)))
        matrix_bytes = 0 if self.mode in ('dense', 'dense_pruned', 'always_full') else max_candidate*q*9
        # Sparse/Scan assembly retains only final matrices plus one lookup
        # column/index scratch, never a second nCandidate*q lookup family.
        assembly = (64*max_candidate+32*row_upper if matrix_bytes else 0)
        family_upper = 8192+4*max_candidate+matrix_bytes+assembly+8*len(sources)*max(q, 1)+preview.nbytes
        missing_upper = sum(self._atom_estimate(source, len(preview)) for group, source in keys if (group, source) not in self.cache)
        possible = registered and self._evict(family_upper+missing_upper, protected=keys)
        if not possible:
            # All cuts/sources remain in an explicit insufficient-budget
            # family. It may still obtain a verified actual-view witness.
            family = DomainFamily(self, seed, sources, y, rho, base_audit, (), ks, route='budget_unknown', preview_rows=preview)
            if not self._evict(family.nbytes):
                raise MemoryError('Even minimal query metadata cannot fit alongside existing pinned families.')
            self.live.append(family); self._live_bytes += family.nbytes; self._live_reserved_bytes += family.nbytes
            family.reservation = family.nbytes
            self._track()
            return family
        self._pending_bytes += family_upper+missing_upper
        self._track()
        pinned = []
        try:
            # One source's exact partition/two RHS may serve its two misses
            # in this prepare call; no source/RHS progress survives the call.
            # LRU visits are source-major; certificate atom order remains
            # group-major, matching the mathematical H/F column convention.
            for source in map(int, sources):
                source_keys = ((0, source), (1, source))
                spec = self.compiler.source_spec(source) if any(key not in self.cache for key in source_keys) else None
                for key in source_keys:
                    if key in self.cache:
                        self.hits += 1; self.cache.move_to_end(key)
                    else:
                        self.misses += 1
                        atom = self._build_atom(key, spec, preview)
                        estimate = self._atom_estimate(source, len(preview))
                        if atom.nbytes > estimate:
                            raise MemoryError('Actual atom metadata exceeds the conservative admission reservation.')
                        self.cache[key] = atom; self._cache_bytes += atom.nbytes
                        self._pending_bytes -= estimate
                    self.pins[key] = self.pins.get(key, 0)+1
                    pinned.append(key)
                    self._track()
                del spec
            atoms = tuple(self.cache[key] for key in keys)
            family = DomainFamily(self, seed, sources, y, rho, base_audit, atoms, ks,
                route='dense' if self.mode in ('dense', 'always_full') else self.mode, preview_rows=preview)
            if family.nbytes > family_upper:
                raise MemoryError('Actual live state exceeds admission reservation.')
            self._pending_bytes -= family_upper
            self.live.append(family); self._live_bytes += family.nbytes; self._live_reserved_bytes += family.nbytes
            family.reservation = family.nbytes
            self._track()
            return family
        except Exception:
            # Preparation failed; release only this transaction's pins.
            for key in pinned:
                self.pins[key] -= 1
                if not self.pins[key]: del self.pins[key]
            self._pending_bytes = 0
            raise


class DomainFamily:
    def __init__(self, directory, seed, sources, y, rho, base_audit, atoms, ks, route,
                 preview_rows=()):
        self.directory, self.compiler = directory, directory.compiler
        self.seed, self.y = seed, y
        self.sources = sources.astype(np.int32)
        self.rho, self.base_audit = rho, base_audit
        self.atoms, self.route = atoms, route
        self.ks, self.closed = tuple(map(int, ks)), False
        self.reservation = 0
        self.point_checks = self.verified_witnesses = self.certificate_calls = 0
        self._zero_columns = []
        self._transient_bytes = 0
        self.preview_rows = np.asarray(preview_rows, dtype=np.int32)
        self.candidate_rows = np.empty(0, dtype=np.int32)
        self.values = self.mask = self.columns = None
        self.Hm = self.Hw = np.empty((len(sources), 0))
        self.mu = self.nu = ()
        self.tails = np.empty(0)
        self.atom_rhos = ()
        self._metadata_bytes = 256+self.sources.nbytes+numeric_payload((rho, base_audit))
        if atoms:
            self.mu = tuple(atom.mass for atom in atoms[:len(sources)])
            self.nu = tuple(atom.mass for atom in atoms[len(sources):])
            H = np.column_stack([atom.lookup(sources)[0] for atom in atoms])
            self.Hm, self.Hw = H[:, :len(sources)].copy(), H[:, len(sources):].copy()
            if route in ('dense', 'dense_pruned'):
                base_rows = self.preview_rows[:min(128, self.compiler.n)] if len(self.preview_rows) >= min(128, self.compiler.n) else _top(y, min(128, self.compiler.n))
                self.candidate_rows = (np.arange(self.compiler.n, dtype=np.int32) if route == 'dense' else
                    np.unique(np.concatenate([base_rows, sources, self.preview_rows, *[atom.rows for atom in atoms]])).astype(np.int32))
                self.columns = []
                for atom in atoms:
                    if atom.zero:
                        zero = np.array([0.]); self._zero_columns.append(zero)
                        self.columns.append(np.broadcast_to(zero, (self.compiler.n,)))
                    else:
                        self.columns.append(atom.response)
            else:
                base_rows = self.preview_rows[:min(128, self.compiler.n)] if len(self.preview_rows) >= min(128, self.compiler.n) else _top(y, min(128, self.compiler.n))
                rows = (np.arange(self.compiler.n) if route == 'scan' else
                    np.unique(np.concatenate([base_rows, sources, self.preview_rows, *[atom.rows for atom in atoms]])))
                self.candidate_rows = rows.astype(np.int32)
                self.values = np.empty((len(rows), len(atoms)))
                self.mask = np.empty((len(rows), len(atoms)), dtype=bool)
                for column, atom in enumerate(atoms):
                    values, valid = atom.lookup(rows)
                    self.values[:, column] = values
                    self.mask[:, column] = valid
                    del values, valid
                if route == 'no_core':
                    # Missing model entries are exactly zero, including H.
                    self.mask[np.searchsorted(rows, sources), :] = True
            self.tails = np.array([atom.tail for atom in atoms])
            self.atom_rhos = tuple(atom.rho for atom in atoms)

    @property
    def nbytes(self):
        return (self._metadata_bytes+self._transient_bytes+sum(array.nbytes for array in
            (self.candidate_rows, self.preview_rows, self.Hm, self.Hw, self.tails, *self._zero_columns))+
            (0 if self.values is None else self.values.nbytes+self.mask.nbytes)+32*len(self.atoms))

    def _reserve_work(self, amount):
        amount = int(amount)
        directory = self.directory
        mandatory = (directory.plan_bytes+directory._live_reserved_bytes+directory._pending_bytes+
            sum(directory.cache[key].nbytes for key in directory.pins))
        if amount+mandatory > directory.budget_bytes:
            return False
        if not self.directory._evict(amount):
            return False
        self.directory._live_reserved_bytes += amount
        self.reservation += amount
        self.directory._track()
        return True

    @property
    def accounting(self):
        return dict(requested_mode=self.directory.mode, actual_route=self.route,
            source_count=len(self.sources), atom_count=len(self.atoms),
            candidate_count=len(self.candidate_rows), retained_numeric_bytes=self.nbytes,
            admission_reserved_numeric_bytes=self.reservation,
            directory_numeric_bytes=self.directory.nbytes,
            directory_reserved_bytes=self.directory.reserved_bytes,
            numeric_budget_bytes=self.directory.budget_bytes,
            atom_hits=self.directory.hits, atom_misses=self.directory.misses,
            atom_evictions=self.directory.evictions,
            point_checks=self.point_checks, verified_witnesses=self.verified_witnesses,
            preview_keep=self.directory.preview_keep, current_preview_rows=len(self.preview_rows),
            refinement_invalidations=self.directory.refinement_invalidations,
            base_residual_upper=self.rho,
            source_core_model='Recertified truncated response model' if self.route == 'no_core' else 'Retained full-response source coordinates')

    @property
    def statistics(self):
        return self.accounting

    def _release_work(self, amount):
        self.directory._live_reserved_bytes -= amount
        self.reservation -= amount
        self.directory._track()

    def _set_transient(self, amount):
        self.directory._live_bytes += amount-self._transient_bytes
        self._transient_bytes = amount
        self.directory._track()

    def _polynomial_reservation(self, box, certifier):
        r = len(self.sources)
        scalars = [*self.mu, *self.nu, *self.y[self.sources], *self.Hm.ravel(), *self.Hw.ravel()]
        bits = max((max(abs(exact(x).numerator).bit_length(), exact(x).denominator.bit_length()) for x in scalars), default=1)
        box_bits = max(max(abs(exact(x).numerator).bit_length(), exact(x).denominator.bit_length()) for pair in box for x in pair)
        # Each source column has common dyadic-H / exact-mass denominators.
        # This deliberately conservative reservation covers retained D/C/F,
        # Bernstein controls and small arithmetic intermediates before their
        # exact numeric payload can be measured. It is not an RSS estimate.
        coefficient_bytes = 64+(2*max(r, 1)*(3*bits+2*box_bits)+128+7)//8
        coefficient_count = (r+1)**2*(1+3*r)
        if certifier == 'lifting':
            lifting = importlib.import_module('src.policy_domain_lifting')
            if hasattr(lifting, 'compile_reservation_upper'):
                return int(lifting.compile_reservation_upper(self.Hm, self.Hw, self.mu, self.nu,
                    self.y[self.sources], box, self.sources))
            coefficient_count = max(coefficient_count, 4**r*(2*r+4))
        return 4096+3*coefficient_count*coefficient_bytes

    def certificate_inputs(self, box=DEFAULT_BOX, ks=None):
        """Expose the same retained representation to every domain certifier."""
        if self.closed:
            raise RuntimeError('Family is closed.')
        ks = self.ks if ks is None else tuple(ks)
        order = _top(self.y, min(max(ks), self.compiler.n))
        return dict(H_m=self.Hm, H_w=self.Hw, mu=self.mu, nu=self.nu,
            y_sources=self.y[self.sources], source_ids=self.sources, box=box,
            y=self.y, candidate_rows=self.candidate_rows, retained_values=self.values,
            retained_mask=self.mask, tail_bounds=self.tails,
            retained_columns=self.columns,
            selected_sets={int(k): order[:int(k)].copy() for k in ks},
            kappa=self.compiler.guard.kappa_old, base_residual=self.rho,
            atom_residuals=self.atom_rhos, round_pad=0)

    def _estimate_sets(self, weights, ks):
        if not self.atoms:
            return None
        count = len(self.candidate_rows)
        workspace = 3*count*8+1024
        if not self._reserve_work(workspace):
            return None
        self._set_transient(workspace)
        try:
            m, w = map(float, weights)
            mu, nu = np.array(list(map(float, self.mu))), np.array(list(map(float, self.nu)))
            Z = 1+(m-1)*mu+(w-1)*nu
            matrix = np.diag(Z)+(m-1)*self.Hm+(w-1)*self.Hw
            t = np.linalg.solve(matrix, self.y[self.sources])
            coefficients = np.concatenate(((m-1)*t, (w-1)*t))
            if self.columns is not None:
                scores = self.y[self.candidate_rows].copy()
                for value, column in zip(coefficients, self.columns):
                    if value: scores -= value*column[self.candidate_rows]
                rows = self.candidate_rows
            else:
                scores = self.y[self.candidate_rows]-self.values@coefficients
                rows = self.candidate_rows
            order = _top(scores, min(max(ks), len(scores)))
            return {k: set(map(int, rows[order[:k]])) for k in ks}
        except np.linalg.LinAlgError:
            return None
        finally:
            self._set_transient(0); self._release_work(workspace)

    def _witnesses(self, box, ks, proposed, enabled):
        if not enabled or not len(self.sources):
            return {}
        (ml, mh), (wl, wh) = tuple(tuple(map(float, pair)) for pair in box)
        center = ((ml+mh)/2, (wl+wh)/2)
        points = [center, (ml, wl), (ml, wh), (mh, wl), (mh, wh)]
        points += [p for p in GRID if ml <= p[0] <= mh and wl <= p[1] <= wh]
        points = list(dict.fromkeys(points))
        reference = {}
        if ml <= 1 <= mh and wl <= 1 <= wh:
            order = _top(self.y, min(max(ks)+1, self.compiler.n))
            bound = self.compiler.guard.kappa_old*self.rho
            for k in ks:
                if k == self.compiler.n or exact(self.y[order[k-1]])-exact(self.y[order[k]]) > 2*bound:
                    reference[k] = dict(weights=(1., 1.), sets={k: order[:k].copy()},
                        strict_flags={k: True}, score_error_bound=bound,
                        origin='Stored base with independently guarded identity-target residual')
        found = {}
        for weights in points:
            unresolved = [k for k in ks if k not in found]
            if not unresolved: break
            estimates = self._estimate_sets(weights, unresolved)
            need = estimates is None or any(k not in reference or estimates[k] != set(map(int, reference[k]['sets'][k])) for k in unresolved)
            if not need: continue
            result = self.compiler.point(self.seed, self.sources, weights, ks=tuple(unresolved))
            self.point_checks += 1
            for k in unresolved:
                if not result['strict_flags'][k]: continue
                if k not in reference:
                    reference[k] = result
                elif set(map(int, result['sets'][k])) != set(map(int, reference[k]['sets'][k])):
                    found[k] = dict(status='UNSTABLE', witness=dict(first=reference[k], second=result),
                        reason='Two distinct in-domain actual mathematical views have independently strict, different top-k memberships.')
                    self.verified_witnesses += 1
        return found

    def certify(self, box=DEFAULT_BOX, ks=None, depth=2, certifier='poly', early_witness=True,
                chunk_rows=512, witness_order='after'):
        if self.closed:
            raise RuntimeError('Family is closed.')
        ks = self.ks if ks is None else tuple(map(int, ks))
        if not ks or min(ks) < 1 or max(ks) > self.compiler.n:
            raise ValueError('Requested cuts must exist in the graph.')
        if certifier not in ('poly', 'interval', 'lifting'):
            raise ValueError('Unknown domain certifier.')
        if witness_order not in ('before', 'after'):
            raise ValueError('Witness order is before or after the domain certificate.')
        (ml, mh), (wl, wh) = box
        if not (0 < ml < mh and 0 < wl < wh):
            raise ValueError('Positive nonempty rectangle required.')
        started = perf_counter()
        self.certificate_calls += 1
        inputs = self.certificate_inputs(box, ks)
        selected = inputs['selected_sets']
        cuts = self._witnesses(box, ks, selected, early_witness and witness_order == 'before')
        remaining = {k: ids for k, ids in selected.items() if k not in cuts}
        reason = None
        scan_stats = {}
        compiled_payload = workspace_reserved = compilation_reserved = 0
        chosen_chunk = 0
        if not remaining:
            pass
        elif not len(self.sources):
            order = _top(self.y, min(max(ks)+1, self.compiler.n))
            bound = self.compiler.guard.kappa_old*self.rho
            for k, chosen in remaining.items():
                stable = k == self.compiler.n or exact(self.y[order[k-1]])-exact(self.y[order[k]]) > 2*bound
                cuts[k] = dict(status='STABLE' if stable else 'UNKNOWN', selected_ids=chosen.tolist(), k=k,
                    reason='No sources: constant raw-background operator, independently guarded base membership.')
        elif self.route == 'budget_unknown':
            reason = 'budget_or_unregistered_stencil'
        else:
            compile_reserve = self._polynomial_reservation(box, certifier)
            if not self._reserve_work(compile_reserve):
                reason = 'polynomial_workspace_budget'
            else:
                compilation_reserved = compile_reserve
                try:
                    if certifier == 'lifting':
                        module = importlib.import_module('src.policy_domain_lifting')
                        domain = module.compile_lifting(self.Hm, self.Hw, self.mu, self.nu,
                            self.y[self.sources], box, self.sources)
                    else:
                        module_name = ('src.policy_domain_scan' if self.directory.scan_backend == 'dot_fallback'
                            else 'src.policy_domain_fast') if certifier == 'poly' else 'src.policy_domain_interval'
                        module = importlib.import_module(module_name)
                        domain = compile_domain(self.Hm, self.Hw, self.mu, self.nu, self.y[self.sources], box, self.sources)
                    compiled_payload = numeric_payload(domain)
                    if compiled_payload > compile_reserve:
                        raise MemoryError('Compiled polynomial payload exceeds its admission reservation.')
                    self._set_transient(compiled_payload)
                    # Keep the small compilation work reservation throughout
                    # the scan; this also covers exact Bernstein conversion.
                    chosen_chunk = max(1, int(chunk_rows))
                    minimum_chunk = min(64, chosen_chunk)
                    while True:
                        if hasattr(module, 'scratch_upper_bound'):
                            extra = dict(subdivide_depth=depth) if certifier == 'lifting' else {}
                            estimate = module.scratch_upper_bound(domain, self.compiler.n,
                                chunk_rows=chosen_chunk, n_cuts=len(remaining), n_candidates=len(self.candidate_rows), **extra)
                        else:
                            controls = max((len(self.sources)+1)**2, 4**len(self.sources) if certifier == 'lifting' else 1)
                            estimate = 16384+8*self.compiler.n+len(remaining)*len(self.candidate_rows)+chosen_chunk*(8*len(self.atoms)+len(self.atoms)+12*8)*max(1, controls)
                        # First fit the scan into currently free space. Only
                        # after reaching the fixed minimum chunk may it evict
                        # enough unpinned atoms for that smallest workspace.
                        if self.directory.reserved_bytes+estimate <= self.directory.budget_bytes:
                            if self._reserve_work(estimate):
                                workspace_reserved = int(estimate); break
                        if chosen_chunk == minimum_chunk:
                            if self._reserve_work(estimate):
                                workspace_reserved = int(estimate)
                            break
                        chosen_chunk = max(minimum_chunk, chosen_chunk//2)
                    if not workspace_reserved:
                        reason = 'scan_workspace_budget'
                    else:
                        function = getattr(module, 'certify_many', None) or module.certify_lifting_many
                        result = function(domain, self.y, self.candidate_rows, self.values, self.mask,
                            self.tails, remaining, kappa=self.compiler.guard.kappa_old,
                            base_residual=self.rho, atom_residuals=self.atom_rhos, round_pad=0,
                            subdivide_depth=depth, retained_columns=self.columns, chunk_rows=chosen_chunk)
                        scan_stats = {key: value for key, value in result.items() if key != 'cuts'}
                        for k, value in result['cuts'].items(): cuts[int(k)] = value
                        actual_workspace = int(result.get('workspace_peak_bytes', result.get('peak_workspace_bytes', 0)))
                        if actual_workspace > workspace_reserved:
                            raise MemoryError('Certifier exceeded its admitted workspace upper bound.')
                finally:
                    self._set_transient(0)
                    if workspace_reserved: self._release_work(workspace_reserved)
                    self._release_work(compile_reserve)
        for k, chosen in remaining.items():
            if k not in cuts:
                cuts[k] = dict(status='UNKNOWN', selected_ids=chosen.tolist(), k=k,
                    reason=reason or 'Certificate did not prove the complete domain.')
        if early_witness and witness_order == 'after':
            unknown = [k for k in ks if cuts[k]['status'] == 'UNKNOWN']
            if unknown:
                cuts.update(self._witnesses(box, unknown, {k: selected[k] for k in unknown}, True))
        self.directory._track()
        return dict(cuts=cuts, requested_mode=self.directory.mode, actual_route=self.route,
            certifier=certifier, point_checks=self.point_checks, verified_witnesses=self.verified_witnesses,
            witness_order=witness_order,
            scan_backend=self.directory.scan_backend if certifier == 'poly' else certifier,
            scan_chunk_rows=chosen_chunk,
            retained_numeric_bytes=self.directory.nbytes, private_budget_bytes=self.directory.budget_bytes,
            compiled_domain_numeric_payload_bytes=compiled_payload,
            polynomial_work_reservation_bytes=compilation_reserved,
            scan_workspace_reservation_bytes=workspace_reserved,
            polynomial_and_scan_workspace_reservation_bytes=compilation_reserved+workspace_reserved,
            scan=scan_stats, operator_seconds=perf_counter()-started,
            memory_scope='Private retained cache/live/candidate/exact-metadata and admitted polynomial/scan work; shared graph/backend, current seed/base and one-atom/point-solve generation scratch separate.')

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def close(self):
        if self.closed: return
        directory = self.directory
        old = self.nbytes
        for atom in self.atoms:
            directory.pins[atom.key] -= 1
            if not directory.pins[atom.key]: del directory.pins[atom.key]
        directory._live_bytes -= old; directory._live_reserved_bytes -= self.reservation
        directory.live.remove(self)
        self.atoms = (); self.columns = self.values = self.mask = None
        self.Hm = self.Hw = np.empty((0, 0)); self.tails = np.empty(0)
        self.candidate_rows = np.empty(0, dtype=np.int32); self._zero_columns = []
        self.preview_rows = np.empty(0, dtype=np.int32)
        self.y = self.seed = None; self.closed = True
        self.base_audit = None; self.rho = Fraction(0); self.mu = self.nu = ()
        self.sources = np.empty(0, dtype=np.int32); self._metadata_bytes = 0
        self.reservation = 0
        directory._track()


Compiler = DomainCompiler
Directory = DomainDirectory
