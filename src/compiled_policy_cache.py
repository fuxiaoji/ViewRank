"""Pinned response caches with implicit directions for a fixed shared P/G pair.

Input directions remain request input. Cached atoms contain only a 24-byte
logical (source, mass, directory reference) descriptor, response state and
measured residual metadata. P and G are immutable, shared directory-version
inputs. retain_deltas=True is the single-factor stored-delta control: retain
the request CSC arrays once without another copy. No n-by-r family copy is
retained in either mode.
"""
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from weakref import ref

import numpy as np
from scipy import sparse

from .baselines.dense_family_budget import DenseBudgetFamily
from .full_scan_policy_family import FullScanPolicyFamily
from .policy_family_fast import FastPolicyFamily, top_indices
from .semantic_basis import strength_scale
from .viewrank import _sparse_bytes


DESCRIPTOR_BYTES = 24


@dataclass(frozen=True)
class ImplicitDirection:
    source: int
    mass: float
    _directory_ref: object

    @property
    def directory(self):
        return self._directory_ref()

    @property
    def nbytes(self):
        return DESCRIPTOR_BYTES

    def materialize(self):
        directory = self.directory
        started = perf_counter()
        delta = (directory.G[:, self.source] - self.mass * directory.P[:, self.source]).tocsc()
        delta.eliminate_zeros()
        directory.direction_materializations += 1
        directory.direction_materialization_seconds += perf_counter() - started
        return delta


@dataclass(frozen=True)
class StoredDirection(ImplicitDirection):
    delta: sparse.csc_matrix

    @property
    def nbytes(self):
        return DESCRIPTOR_BYTES + _sparse_bytes(self.delta)

    def materialize(self):
        return self.delta


@dataclass
class CompiledAtom:
    direction: ImplicitDirection
    residual_l1: float
    update_l1: float
    response: np.ndarray | None = None
    rows: np.ndarray | None = None
    values: np.ndarray | None = None
    tail_inf: float = 0.

    @property
    def delta(self):
        # Compatibility access: the returned sparse matrix is transient.
        return self.direction.materialize()

    @property
    def response_bytes(self):
        return (self.response.nbytes if self.response is not None
                else self.rows.nbytes + self.values.nbytes)

    @property
    def nbytes(self):
        return self.direction.nbytes + 16 + self.response_bytes + (8 if self.response is None else 0)


class _PinnedFamily:
    @property
    def nbytes(self):
        return self.directory.nbytes

    @property
    def memory(self):
        return self.directory.memory

    def close(self):
        if self.closed:
            return
        for key in self.keys:
            self.directory._pins[key] -= 1
            if not self.directory._pins[key]:
                del self.directory._pins[key]
        self.directory._live_core_bytes -= self._metadata_bytes
        self.atoms = ()
        self.keys = ()
        self._native_offset = None
        for name in ('core', 'boundary_matrix', 'sources', 'masses', 'positions',
                     'rows', 'boundary_positions', 'source_positions', 'active'):
            if hasattr(self, name):
                setattr(self, name, None)
        if hasattr(self, 'atom_positions'):
            self.atom_positions = ()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CompiledFastFamily(_PinnedFamily, FastPolicyFamily):
    pass


class CompiledFullScanFamily(_PinnedFamily, FullScanPolicyFamily):
    pass


class CompiledDenseFamily(_PinnedFamily, DenseBudgetFamily):
    """Pinned vectors, the same captured residual contract, no target SpMV."""

    def coefficients(self, strengths):
        strengths = np.broadcast_to(np.asarray(strengths, dtype=float), self.original_masses.shape)
        all_a = np.array([strength_scale(l, m) for l, m in zip(strengths, self.original_masses)])
        a = all_a[self.active]
        K = np.eye(len(a)) + self.core * a[None, :]
        c = np.linalg.solve(K, self.y[self.sources]) if len(a) else np.empty(0)
        return a, c, a * c, K

    def solve(self, strengths, ks=(2, 5, 10), always_fused=False, materialize=False):
        if self.closed:
            raise RuntimeError('family pins have been released')
        if self.route == 'native':
            if self._native_offset is None:
                from .baselines.native_offset_family import NativeOffsetFamily
                self._native_offset = NativeOffsetFamily(self.ranker, self.seed, self.original_sources,
                    self.original_directions, self.original_masses, base_scores=self.y,
                    base_residual=self.base_residual, point_graph=self.directory.point_graph)
            return DenseBudgetFamily.solve(self, strengths, ks=ks)
        started = perf_counter()
        a, c, d, K = self.coefficients(strengths)
        h = self.y[self.sources] - K @ c
        if always_fused:
            result = FastPolicyFamily._complete(self, d, c, tuple(ks), started, {}, materialize)
        else:
            scores = self.y.copy()
            for value, atom in zip(d, self.atoms):
                if value:
                    scores -= value * atom.response
            h_inf = float(np.max(abs(h))) if len(h) else 0.
            beta = self.base_residual + sum(abs(v) * atom.residual_l1 for v, atom in zip(d, self.atoms))
            beta += sum(abs(v * scale) * atom.update_l1 for v, scale, atom in zip(h, a, self.atoms))
            pad = 64 * np.finfo(float).eps * (1 + np.linalg.norm(self.y, 1)) / (1 - self.ranker.alpha)
            numerical = beta / (1 - self.ranker.alpha) + h_inf + pad
            order = top_indices(scores, min(max(ks) + 1, self.ranker.n))
            certified = {int(k): bool(k == self.ranker.n or scores[order[k-1]]-scores[order[k]] > 2*numerical) for k in ks}
            result = dict(scores=scores, sets={int(k): order[:k] for k in ks},
                selected_scores={int(k): scores[order[:k]] for k in ks},
                initial_certified=certified, completion_certified=certified,
                numerical_allowance=numerical, beta=beta, fallback_solve_calls=0,
                seconds=perf_counter()-started, payload_bytes=self.nbytes,
                route='dense', route_reason=None, identity_reused=not np.any(d),
                basis_memory=self.memory)
            if materialize:
                result['error_bounds'] = np.full(self.ranker.n, numerical)
        result.update(scales=a, coefficients=c, scaled_coefficients=d, K=K,
                      a=a, c=c, d=d, small_system_residual=h,
                      base_residual=self.base_residual)
        result['seconds'] = perf_counter()-started
        return result


class _CompiledDirectory:
    def __init__(self, ranker, G, budget_bytes, stencil=None, keep=0, point_graph=None,
                 retain_deltas=False, exact_core=True):
        self.ranker = ranker
        self.P = ranker.P
        self.G = sparse.csc_matrix(G, copy=False)
        self.version = (id(self.P), id(G))
        self._descriptor_scope = ref(self)
        self.stencil = stencil
        self.keep = min(int(keep), ranker.n)
        self.budget_bytes = self.full_budget = int(budget_bytes)
        self.point_graph = point_graph
        self.retain_deltas = bool(retain_deltas)
        self.exact_core = bool(exact_core)
        self.plan_bytes = (sum(rows.nbytes for rows in stencil.values()) + 8*len(stencil)
                           if stencil is not None else 0)
        if self.plan_bytes > self.budget_bytes:
            raise ValueError('The interaction plan alone exceeds the retained budget.')
        self.cache = OrderedDict()
        self._pins = {}
        self._payload_bytes = self.plan_bytes
        self._live_core_bytes = 0
        self._preparing_masses = {}
        self.hits = self.misses = self.evictions = self.atomic_solve_calls = 0
        self.atomic_prepare_seconds = 0.
        self.fallback_families = self.native_direct_calls = self.native_offset_calls = 0
        self.direction_materializations = 0
        self.direction_materialization_seconds = 0.
        self.peak_bytes = self.plan_bytes
        self.peak_live_response_bytes = 0

    @property
    def nbytes(self):
        return self._payload_bytes + self._live_core_bytes

    @property
    def response_bytes(self):
        return sum(atom.response_bytes for atom in self.cache.values())

    @property
    def memory(self):
        live = [self.cache[key] for key in self._pins]
        return dict(cache_atom_basis_bytes=self._payload_bytes,
            cache_response_bytes=self.response_bytes, plan_bytes=self.plan_bytes,
            live_atom_basis_bytes=sum(a.nbytes for a in live),
            live_response_bytes=sum(a.response_bytes for a in live),
            live_core_metadata_bytes=self._live_core_bytes,
            total_private_retained_basis_bytes=self.nbytes,
            peak_private_retained_basis_bytes=self.peak_bytes,
            peak_live_response_bytes=self.peak_live_response_bytes,
            direction_descriptor_bytes=DESCRIPTOR_BYTES*len(self.cache),
            implicit_descriptor_bytes=DESCRIPTOR_BYTES*sum(not isinstance(a.direction,StoredDirection) for a in self.cache.values()),
            retained_sparse_delta_bytes=sum(_sparse_bytes(a.direction.delta) for a in self.cache.values()
                                            if isinstance(a.direction,StoredDirection)))

    def _track(self):
        self.peak_bytes = max(self.peak_bytes, self.nbytes)
        self.peak_live_response_bytes = max(self.peak_live_response_bytes,
            sum(self.cache[key].response_bytes for key in self._pins))

    def atomic(self, source, delta):
        mass = self._preparing_masses[int(source)]
        key = (int(source), float(mass))
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        started = perf_counter()
        rhs = (-self.ranker.alpha * delta).toarray().ravel()
        response = self.ranker.base_lu.solve(rhs)
        measured = getattr(self.ranker.base_lu, 'last_residuals', None)
        no_core = self.stencil is not None and not self.exact_core
        residual = (0. if no_core else float(measured[0]) if measured is not None else
                    float(np.linalg.norm(rhs - self.ranker.A @ response, 1)))
        descriptor = (StoredDirection(int(source), float(mass), self._descriptor_scope, delta)
                      if self.retain_deltas else
                      ImplicitDirection(int(source), float(mass), self._descriptor_scope))
        atom = CompiledAtom(descriptor, residual, float(np.linalg.norm(rhs, 1)))
        if self.stencil is None:
            atom.response = response
        else:
            top = (np.argpartition(abs(response), -self.keep)[-self.keep:]
                   if self.keep else np.empty(0, dtype=np.int64))
            atom.rows = (np.union1d(top, self.stencil[int(source)]) if self.exact_core else np.sort(top))
            atom.values = response[atom.rows].copy()
            omitted = np.ones(self.ranker.n, dtype=bool)
            omitted[atom.rows] = False
            atom.tail_inf = float(np.max(abs(response[omitted]))) if omitted.any() else 0.
            if no_core:
                model = sparse.csc_matrix((atom.values, (atom.rows, np.zeros(len(atom.rows), dtype=np.int64))),
                                          shape=(self.ranker.n, 1))
                residual_column = -self.ranker.alpha * delta - self.ranker.A @ model
                atom.residual_l1 = float(np.abs(residual_column.data).sum())
                atom.tail_inf = 0.
        self.cache[key] = atom
        self._pins[key] = 1
        self._payload_bytes += atom.nbytes
        self.atomic_solve_calls += 1
        self.atomic_prepare_seconds += perf_counter() - started
        self._track()
        return atom

    def boundary_block(self, atoms, sources):
        sources = np.asarray(sources, dtype=np.int64)
        columns = []
        for atom in atoms:
            positions = np.searchsorted(atom.rows, sources)
            if self.exact_core:
                if np.any(positions >= len(atom.rows)) or np.any(atom.rows[positions] != sources):
                    raise ValueError('Requested complete sources need unregistered interaction entries.')
                columns.append(atom.values[positions])
            else:
                valid = positions < len(atom.rows)
                valid[valid] &= atom.rows[positions[valid]] == sources[valid]
                column = np.zeros(len(sources))
                column[valid] = atom.values[positions[valid]]
                columns.append(column)
        return sources, np.column_stack(columns) if columns else np.empty((len(sources), 0))

    def merge_rhs(self, atoms, coefficients):
        """Materialize each needed direction once; caller charges this work."""
        rhs = np.zeros(self.ranker.n)
        for coefficient, atom in zip(coefficients, atoms):
            if coefficient:
                delta = atom.delta
                rhs[delta.indices] -= self.ranker.alpha * coefficient * delta.data
        return rhs

    def prepare(self, seed, sources, directions, masses, base_scores=None, base_residual=None,
                ks=(2, 5, 10), always_fused=False, full_scan=False):
        started = perf_counter()
        sources = np.asarray(sources, dtype=np.int64)
        masses = np.asarray(masses, dtype=float)
        directions = tuple(directions)
        specs = [(int(i), (int(sources[i]), float(masses[i])))
                 for i in np.argsort(sources, kind='stable') if directions[i].nnz]
        r = len(specs)
        if base_scores is None:
            base_scores = self.ranker.base_lu.solve((1-self.ranker.alpha)*np.asarray(seed))
            measured = getattr(self.ranker.base_lu, 'last_residuals', None)
            if base_residual is None and measured is not None:
                base_residual = float(measured[0])
        if base_residual is None:
            base_residual = float(np.linalg.norm((1-self.ranker.alpha)*np.asarray(seed)-self.ranker.A@base_scores, 1))
        if self.stencil is None:
            sizes = {key: self.ranker.n*8 + DESCRIPTOR_BYTES + 16 for _, key in specs}
            metadata_upper = 8*(r*r+3*r)+8
        else:
            rows = {key: min(self.ranker.n, self.keep+len(self.stencil.get(key[0], ()))) for _,key in specs}
            sizes = {key: 16*count+DESCRIPTOR_BYTES+24 for key,count in rows.items()}
            retained = sum(rows.values())
            metadata_upper = 8*(r*r+len(sources)*r+3*r+len(sources))+8
            if not (always_fused or full_scan):
                candidates = min(self.ranker.n, retained+len(sources)+max(ks))
                metadata_upper += 8*(candidates+len(sources)+r+retained)
        if self.retain_deltas:
            for index,key in specs:
                sizes[key] += _sparse_bytes(directions[index])
        pinned = sum(self.cache[key].nbytes for key in self._pins)
        extra = sum(size for key,size in sizes.items() if key not in self._pins)
        if self.plan_bytes+pinned+extra+self._live_core_bytes+metadata_upper > self.budget_bytes:
            self.fallback_families += 1
            family = CompiledDenseFamily(self, seed, sources, directions, masses, base_scores,
                                         route='native', reason='compiled_family_exceeds_retained_budget')
            family.base_residual = float(base_residual)
            family.prepare_seconds = perf_counter()-started
            return family
        for _, key in specs:
            if key in self.cache:
                self.cache.move_to_end(key)
                self._pins[key] = self._pins.get(key, 0)+1
        missing = sum(size for key,size in sizes.items() if key not in self.cache)
        while self.nbytes+missing+metadata_upper > self.budget_bytes:
            key = next(key for key in self.cache if key not in self._pins)
            self._payload_bytes -= self.cache.pop(key).nbytes
            self.evictions += 1
        self._preparing_masses = {key[0]:key[1] for _,key in specs}
        try:
            if self.stencil is None:
                atoms = tuple(self.atomic(sources[i], directions[i]) for i,_ in specs)
                active = np.array([i for i,_ in specs], dtype=np.int64)
                family = CompiledDenseFamily(self, seed, sources, directions, masses, base_scores,
                    route='dense', atoms=atoms, keys=tuple(key for _,key in specs), active=active)
                arrays = [family.core, family.sources, family.masses, family.active]
            else:
                cls = CompiledFullScanFamily if full_scan else CompiledFastFamily
                options = {} if full_scan else dict(candidate_outputs=not always_fused)
                family = cls(self.ranker, seed, sources, directions, masses, directory=self,
                    base_scores=base_scores, base_residual=base_residual, ks=ks, **options)
                family.keys = tuple(key for _,key in specs)
                arrays = [family.original_sources, family.sources, family.masses, family.positions,
                          family.core, family.boundary_matrix]
                if not (always_fused or full_scan):
                    arrays += [family.rows, family.boundary_positions, family.source_positions, *family.atom_positions]
                family.route = 'stencil'
                family.route_reason = None
        finally:
            self._preparing_masses = {}
        family.base_residual = float(base_residual)
        family.core_representation = 'untruncated' if self.exact_core else 'truncated_numeric_model'
        family._metadata_bytes = sum(array.nbytes for array in {id(a):a for a in arrays}.values())+8
        family.closed = False
        family._native_offset = None
        self._live_core_bytes += family._metadata_bytes
        self._track()
        assert self.nbytes <= self.budget_bytes
        family.prepare_seconds = perf_counter()-started
        return family


class CompiledStencilDirectory(_CompiledDirectory):
    """exact_core=False uses only top-L responses and a generic residual bound.

    Missing source rows are zero in Khat; their reconstructed scores are not
    exact target scores. The global beta covers truncation and coefficient
    error. No interaction plan or true-tail certificate is retained then.
    """
    def __init__(self, ranker, G, stencil, keep, budget_bytes, point_graph=None, retain_deltas=False,
                 exact_core=True):
        super().__init__(ranker, G, budget_bytes, stencil=stencil if exact_core else {}, keep=keep,
                         point_graph=point_graph, retain_deltas=retain_deltas, exact_core=exact_core)


class CompiledDenseDirectory(_CompiledDirectory):
    def __init__(self, ranker, G, budget_bytes, point_graph=None, retain_deltas=False):
        super().__init__(ranker, G, budget_bytes, point_graph=point_graph, retain_deltas=retain_deltas)
