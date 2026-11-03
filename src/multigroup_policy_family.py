"""Two-group normalized policies with q response atoms and an r-source core.

Mention and weak-link weights vary independently; self-loop weight stays one.
Atoms are globally cached by (group, source). Full responses are never copied
into a live family. Actual update matrices are assembled only for native work.
"""
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from weakref import ref

import numpy as np
from scipy import sparse

from .baselines.certified_native_family import CertifiedNativeFamily
from .policy_execution_v7 import complete_family, top_indices
from .viewrank import _sparse_bytes

GROUPS = ('mention', 'weak')
POLICY_GRID = tuple((m, w) for m in (1., 2., 3., 4.) for w in (.25, .5, 1., 2.))


def _flags(result, ks):
    return {k: bool(result.get('initial_certified', {}).get(k, False)
        or result.get('completion_certified', {}).get(k, False)) for k in ks}


def _bound(result):
    return float(result.get('score_error_bound', result.get('numerical_allowance', np.inf)))


def _bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_bytes(k)+_bytes(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sum(map(_bytes, value))
    if isinstance(value, str):
        return len(value.encode())
    return 8 if isinstance(value, (int, float, bool, np.number)) else 0


def _copy(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    return value


class TwoGroupCompiler:
    """Immutable P/mention/weak inputs, shared by atom and native executors."""
    def __init__(self, ranker, mention):
        started = perf_counter()
        self.ranker, self.P = ranker, ranker.P
        mention = sparse.csc_matrix(mention, copy=True)
        mention.sum_duplicates()
        mention.sort_indices()
        weak = (self.P-mention-sparse.diags(self.P.diagonal(), format='csc')).tocsc()
        weak.eliminate_zeros()
        weak.sort_indices()
        if np.any(mention.data < 0) or np.any(weak.data < 0):
            raise ValueError('Mention, weak and self must form nonnegative source subdistributions.')
        self.groups = (mention, weak)
        self.direction_materializations = 0
        self.setup_seconds = perf_counter()-started

    @property
    def shared_input_bytes(self):
        return _sparse_bytes(self.P)+sum(_sparse_bytes(g) for g in self.groups)

    def delta(self, source, group, mass):
        self.direction_materializations += 1
        delta = (self.groups[group][:, source]-mass*self.P[:, source]).tocsc()
        delta.eliminate_zeros()
        return delta

    def prepare(self, seed, sources, base_scores=None, base_residual=None):
        return GroupPlan(self, seed, sources, base_scores, base_residual)


class GroupPlan:
    """Current query metadata; no sparse direction/response matrices retained."""
    def __init__(self, compiler, seed, sources, base_scores, base_residual):
        started = perf_counter()
        self.compiler, self.ranker = compiler, compiler.ranker
        self.seed = np.asarray(seed, dtype=float)
        self.sources = np.array(sources, dtype=np.int64, copy=True)
        if len(set(map(int, self.sources))) != len(self.sources):
            raise ValueError('Each original source occurs once in the source core.')
        self.r = len(self.sources)
        self.y = (self.ranker.base_lu.solve((1-self.ranker.alpha)*self.seed)
            if base_scores is None else np.asarray(base_scores, dtype=float))
        self.base_residual = (float(np.linalg.norm((1-self.ranker.alpha)*self.seed-self.ranker.A@self.y, 1))
            if base_residual is None else float(base_residual))
        self.base_pad = 64*np.finfo(float).eps*(1+np.linalg.norm(self.y, 1))/(1-self.ranker.alpha)
        self.masses = np.array([[float(g.data[g.indptr[s]:g.indptr[s+1]].sum())
            for g in compiler.groups] for s in self.sources], dtype=float).reshape(self.r, 2)
        specs = []
        for i, source in enumerate(self.sources):
            for group in range(2):
                if compiler.delta(int(source), group, self.masses[i, group]).nnz:
                    specs.append((i, group))
        self.source_map = np.array([i for i, _ in specs], dtype=np.int64)
        self.group_map = np.array([g for _, g in specs], dtype=np.int64)
        self.keys = tuple((g, int(self.sources[i])) for i, g in specs)
        self.q = len(specs)
        self.prepare_seconds = perf_counter()-started

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.sources, self.masses, self.source_map, self.group_map))+16*self.q+16

    def scales(self, weights):
        if hasattr(weights, 'keys'):
            weights = np.column_stack([np.broadcast_to(np.asarray(weights[g], dtype=float), (self.r,)) for g in GROUPS])
        else:
            weights = np.broadcast_to(np.asarray(weights, dtype=float), (self.r, 2))
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError('Both group weights must be finite and positive.')
        denominator = 1+((weights-1)*self.masses).sum(axis=1)
        if np.any(denominator <= 0):
            raise ValueError('The computed normalized source denominator must be positive.')
        all_a = (weights-1)/denominator[:, None]
        return all_a[self.source_map, self.group_map], weights

    def actual_delta(self, a):
        """D=U T, preserving every original source, including zero columns."""
        columns = [sparse.csc_matrix((self.ranker.n, 1)) for _ in self.sources]
        for coefficient, i, group in zip(a, self.source_map, self.group_map):
            if coefficient:
                delta = self.compiler.delta(int(self.sources[i]), int(group), self.masses[i, group])
                columns[i] = columns[i]+coefficient*delta
        result = (sparse.hstack(columns, format='csc') if columns else
            sparse.csc_matrix((self.ranker.n, 0)))
        result.eliminate_zeros()
        return result


@dataclass(frozen=True)
class _Direction:
    source: int
    group: int
    mass: float
    owner: object

    def materialize(self):
        return self.owner().delta(self.source, self.group, self.mass)


@dataclass
class _Atom:
    direction: _Direction
    residual_l1: float
    update_l1: float
    response: object = None
    rows: object = None
    values: object = None
    tail_inf: float = 0.

    @property
    def delta(self):
        return self.direction.materialize()

    @property
    def nbytes(self):
        # Descriptor, logical cache key, measured residual/update/tail scalars.
        return 72+(self.response.nbytes if self.response is not None else self.rows.nbytes+self.values.nbytes)


class MultiGroupDirectory:
    """One global LRU/pin set for both groups, including live family state."""
    def __init__(self, compiler, stencil, keep=128, budget_bytes=4*1024**2, mode='sparse'):
        if mode not in ('sparse', 'dense', 'no_core'):
            raise ValueError('Mode is sparse, dense or no_core.')
        self.compiler, self.ranker, self.mode = compiler, compiler.ranker, mode
        self.keep, self.budget_bytes = min(int(keep), self.ranker.n), int(budget_bytes)
        self.stencil = ({int(s): np.asarray(rows, dtype=np.int32) for s, rows in stencil.items()}
            if mode == 'sparse' else {})
        self.plan_bytes = sum(v.nbytes for v in self.stencil.values())+8*len(self.stencil)
        self.cache, self.pins, self.live = OrderedDict(), {}, []
        self._cache_bytes = 0
        self._live_bytes = self._live_reserved_bytes = 0
        self.hits = self.misses = self.evictions = 0
        self.peak_bytes = self.peak_reserved_bytes = self.plan_bytes

    @property
    def nbytes(self):
        return self.plan_bytes+self._cache_bytes+self._live_bytes

    @property
    def reserved_bytes(self):
        return self.plan_bytes+self._cache_bytes+self._live_reserved_bytes

    @property
    def memory(self):
        return dict(total_private_retained_basis_bytes=self.nbytes,
            peak_private_retained_basis_bytes=self.peak_bytes,
            admission_reserved_bytes=self.reserved_bytes,
            peak_admission_reserved_bytes=self.peak_reserved_bytes,
            cache_atom_basis_bytes=self._cache_bytes,
            live_family_state_bytes=self._live_bytes, stencil_bytes=self.plan_bytes)

    def _track(self):
        self.peak_bytes = max(self.peak_bytes, self.nbytes)
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, self.reserved_bytes)
        assert self.nbytes <= self.reserved_bytes <= self.budget_bytes

    def _evict(self, extra):
        while self.reserved_bytes+extra > self.budget_bytes:
            key = next((k for k in self.cache if k not in self.pins), None)
            if key is None:
                raise MemoryError('Other live families leave insufficient retained metadata space.')
            self._cache_bytes -= self.cache.pop(key).nbytes
            self.evictions += 1

    def merge_rhs(self, atoms, coefficients):
        rhs = np.zeros(self.ranker.n)
        for value, atom in zip(coefficients, atoms):
            if value:
                delta = atom.delta
                rhs[delta.indices] -= self.ranker.alpha*value*delta.data
        return rhs

    def _atomic(self, plan, j):
        key = plan.keys[j]
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        i, group = int(plan.source_map[j]), int(plan.group_map[j])
        source, mass = int(plan.sources[i]), float(plan.masses[i, group])
        direction = _Direction(source, group, mass, ref(self.compiler))
        delta = direction.materialize()
        rhs = (-self.ranker.alpha*delta).toarray().ravel()
        response = self.ranker.base_lu.solve(rhs)
        measured = getattr(self.ranker.base_lu, 'last_residuals', None)
        residual = float(measured[0]) if measured is not None else float(np.linalg.norm(rhs-self.ranker.A@response, 1))
        atom = _Atom(direction, residual, float(np.linalg.norm(rhs, 1)))
        if self.mode == 'dense':
            atom.response = response
        else:
            top = np.argpartition(abs(response), -self.keep)[-self.keep:] if self.keep else np.empty(0, dtype=np.int64)
            rows = np.union1d(top, self.stencil[source]) if self.mode == 'sparse' else np.sort(top)
            atom.rows, atom.values = rows.astype(np.int32), response[rows].copy()
            outside = np.ones(self.ranker.n, dtype=bool)
            outside[rows] = False
            atom.tail_inf = float(np.max(abs(response[outside]))) if np.any(outside) else 0.
            if self.mode == 'no_core':
                model = sparse.csc_matrix((atom.values, (atom.rows, np.zeros(len(rows), dtype=int))), shape=(self.ranker.n, 1))
                atom.residual_l1 = float(abs(-self.ranker.alpha*delta-self.ranker.A@model).sum())
                atom.tail_inf = 0.
        self.cache[key], self.pins[key] = atom, 1
        self._cache_bytes += atom.nbytes
        return atom

    def prepare(self, plan, ks=(2, 5, 10), output='sparse'):
        if output not in ('sparse', 'scan', 'fused'):
            raise ValueError('Output execution is sparse, scan or fused.')
        sizes = {}
        row_upper = 0
        registered = True
        for key in plan.keys:
            group, source = key
            if self.mode == 'sparse':
                registered &= source in self.stencil and set(map(int, plan.sources)) <= set(map(int, self.stencil.get(source, ())))
            count = min(self.ranker.n, self.keep+len(self.stencil.get(source, ())))
            row_upper += count
            sizes[key] = 72+(8*self.ranker.n if self.mode == 'dense' else 12*count)
        reserve = _Family.reserve(plan, ks, self.mode, output, row_upper)
        pinned = sum(self.cache[k].nbytes for k in self.pins)
        incoming = sum((self.cache[k].nbytes if k in self.cache else size) for k, size in sizes.items() if k not in self.pins)
        required = self.plan_bytes+pinned+self._live_reserved_bytes+incoming+reserve
        if not registered or required > self.budget_bytes:
            family = _Family(plan, ks, directory=self, route='native', native_mode='osp_gap')
            self._evict(family.reservation)
        else:
            for key in plan.keys:
                if key in self.cache:
                    self.pins[key] = self.pins.get(key, 0)+1
            missing = sum(size for key, size in sizes.items() if key not in self.cache)
            self._evict(reserve+missing)
            atoms = tuple(self._atomic(plan, j) for j in range(plan.q))
            family = _Family(plan, ks, directory=self, route=self.mode, output=output, atoms=atoms)
        self.live.append(family)
        self._live_bytes += family.nbytes
        self._live_reserved_bytes += family.reservation
        self._track()
        return family


class _ActualViewNative(CertifiedNativeFamily):
    """Existing native solver, supplied an already compiled D and shared base.

    Dummy masses zero/amplitude two encode unit scaling of actual D; these
    values are adapter internals, never the requested semantic group weights.
    """
    def __init__(self, owner, D, mode):
        started = perf_counter()
        self.ranker, self.mode = owner.ranker, mode
        self.seed, self.sources, self.y = owner.seed, owner.sources, owner.y
        self.directions = tuple(D[:, i] for i in range(owner.plan.r))
        self.masses = np.zeros(owner.plan.r)
        self.graph = getattr(self.ranker.base_lu, 'graph', None)
        self.base_seconds = self.base_residual_seconds = 0.
        self.base_residual, self.base_pad = owner.base_residual, owner._base_pad
        self.native_solve_calls, self._base_outputs = 0, {}
        self.prepare_seconds = perf_counter()-started


class _Family:
    @staticmethod
    def memo_upper(plan, ks):
        return 16*(16*sum(ks)+48*len(ks)+512+8*plan.q)

    @classmethod
    def reserve(cls, plan, ks, mode, output, retained):
        indexes = 8*plan.r*plan.q
        if mode != 'dense' and output == 'sparse':
            candidates = min(plan.ranker.n, retained+plan.r+max(ks))
            indexes += 8*(candidates+retained+plan.r)
        return plan.nbytes+indexes+cls.memo_upper(plan, ks)

    def __init__(self, plan, ks, directory=None, route='native', native_mode='native_gap',
                 output='sparse', atoms=(), budget_bytes=4*1024**2):
        self.plan, self.ranker, self.directory = plan, plan.ranker, directory
        self.seed, self.y, self.sources = plan.seed, plan.y, plan.sources
        self.base_residual, self._base_pad = plan.base_residual, plan.base_pad
        self.ks = tuple(sorted(set(map(int, ks))))
        if not self.ks or self.ks[0] < 1 or self.ks[-1] > self.ranker.n:
            raise ValueError('Every cutoff lies in [1,n].')
        self.route, self.native_mode, self.output = route, native_mode, output
        self.atoms, self.keys = atoms, plan.keys if atoms else ()
        self.memo, self.seen = {}, set()
        self._memo_bytes = self._seen_bytes = 0
        self.logical_requests = self.memo_hits = self.symbolic_identity_requests = 0
        self.closed = False
        arrays = []
        if route != 'native':
            if route == 'dense':
                self.core = np.column_stack([a.response[self.sources] for a in atoms]) if atoms else np.empty((plan.r, 0))
            else:
                self.core = np.zeros((plan.r, plan.q))
                for j, atom in enumerate(atoms):
                    positions = np.searchsorted(atom.rows, self.sources)
                    valid = positions < len(atom.rows)
                    valid[valid] &= atom.rows[positions[valid]] == self.sources[valid]
                    self.core[valid, j] = atom.values[positions[valid]]
            arrays.append(self.core)
            if route != 'dense' and output == 'sparse':
                stored = np.union1d(np.concatenate([a.rows for a in atoms]) if atoms else np.empty(0, dtype=np.int64), self.sources)
                mask = np.ones(self.ranker.n, dtype=bool)
                mask[stored] = False
                unaffected = np.flatnonzero(mask)
                best = unaffected[top_indices(self.y[unaffected], min(max(self.ks)+1, len(unaffected)))] if len(unaffected) else unaffected
                self.rows = np.union1d(stored, best[:max(self.ks)])
                self.outside_base = float(self.y[best[max(self.ks)]]) if len(best) > max(self.ks) else -np.inf
                self.atom_positions = tuple(np.searchsorted(self.rows, a.rows) for a in atoms)
                self.source_positions = np.searchsorted(self.rows, self.sources)
                arrays += [self.rows, self.source_positions, *self.atom_positions]
        self.metadata_bytes = plan.nbytes+sum(a.nbytes for a in arrays)
        self.reservation = (plan.nbytes+self.memo_upper(plan, self.ks) if route == 'native' else
            self.reserve(plan, self.ks, route, output, sum(len(a.rows) for a in atoms) if route != 'dense' else 0))
        self.budget_bytes = directory.budget_bytes if directory is not None else int(budget_bytes)
        if self.reservation > self.budget_bytes:
            raise MemoryError('The fixed sixteen-target metadata does not fit the declared budget.')

    @property
    def nbytes(self):
        return self.metadata_bytes+self._memo_bytes+self._seen_bytes

    @property
    def statistics(self):
        return dict(logical_targets=self.logical_requests,
            symbolic_effective_targets=len(self.seen), summary_reuse_hits=self.memo_hits,
            zero_group_directions=2*self.plan.r-self.plan.q,
            symbolic_identity_targets=self.symbolic_identity_requests,
            actual_distinct_operator_views=None,
            effective_key_scope='Nonzero-group coefficient tuples; degenerate linear dependence may miss reuse equally for all methods. Actual distinct operators need an untimed audit.')

    def _native(self, a, mode):
        D = self.plan.actual_delta(a)
        family = _ActualViewNative(self, D, mode)
        # Dummy amplitude 2 with masses zero means scale exactly one. The
        # actual semantic weights remain the caller's separate recorded input.
        return family.solve(2., ks=self.ks), D

    def _initial(self, a, materialize):
        r = self.ranker
        K = np.eye(self.plan.r)
        for j, i in enumerate(self.plan.source_map):
            K[:, i] += a[j]*self.core[:, j]
        c = np.linalg.solve(K, self.y[self.sources])
        d = a*c[self.plan.source_map]
        if self.output == 'fused':
            return complete_family(self, d, c, self.ks, perf_counter(), {}, True), c, d
        h = self.y[self.sources]-K@c
        beta = self.base_residual+sum(abs(v)*atom.residual_l1 for v, atom in zip(d, self.atoms))
        beta += sum(abs(scale*h[i])*atom.update_l1 for scale, i, atom in zip(a, self.plan.source_map, self.atoms))
        numerical = beta/(1-r.alpha)+(float(np.max(abs(h))) if len(h) else 0.)+self._base_pad
        if self.route == 'dense':
            scores = self.y.copy()
            for value, atom in zip(d, self.atoms):
                if value:
                    scores -= value*atom.response
            scores[self.sources] = c
            order = top_indices(scores, min(max(self.ks)+1, r.n))
            margins = {k: np.inf if k == r.n else
                float(scores[order[k-1]]-scores[order[k]]-2*numerical) for k in self.ks}
            result = dict(sets={k: order[:k] for k in self.ks},
                selected_scores={k: scores[order[:k]] for k in self.ks},
                initial_certified={k: v > 0 for k, v in margins.items()}, initial_margins=margins,
                completion_certified={}, score_error_bound=numerical,
                candidate_nodes=r.n, fallback_solve_calls=0)
            if materialize:
                result.update(scores=scores, error_bounds=np.full(r.n, numerical))
            return result, c, d
        sparse_output = self.output == 'sparse'
        rows = self.rows if sparse_output else np.arange(r.n)
        scores = self.y[rows].copy()
        contributions = np.array([abs(v)*atom.tail_inf for v, atom in zip(d, self.atoms)])
        tail = float(contributions.sum())
        errors = np.full(len(rows), tail)
        for j, (value, atom) in enumerate(zip(d, self.atoms)):
            positions = self.atom_positions[j] if sparse_output else atom.rows
            scores[positions] -= value*atom.values
            errors[positions] -= contributions[j]
        positions = self.source_positions if sparse_output else self.sources
        scores[positions], errors[positions] = c, 0.
        np.maximum(errors, 0., out=errors)
        errors += numerical
        order = top_indices(scores, max(self.ks))
        margins = {}
        for k in self.ks:
            upper = scores+errors
            upper[order[:k]] = -np.inf
            outside = max(float(np.max(upper)), self.outside_base+tail+numerical) if sparse_output else float(np.max(upper))
            margins[k] = float(np.min(scores[order[:k]]-errors[order[:k]])-outside)
        result = dict(sets={k: rows[order[:k]] for k in self.ks},
            selected_scores={k: scores[order[:k]] for k in self.ks},
            initial_certified={k: v > 0 for k, v in margins.items()}, initial_margins=margins,
            completion_certified={}, score_error_bound=tail+numerical, candidate_nodes=len(rows), fallback_solve_calls=0)
        if materialize:
            full, bounds = self.y.copy(), np.full(r.n, tail+numerical)
            full[rows], bounds[rows] = scores, errors
            result.update(scores=full, error_bounds=bounds)
        return result, c, d

    def _warm_finish(self, D, scores):
        r = self.ranker
        for step in range(100):
            updated = r.alpha*(r.P@scores+D@scores[self.sources])+(1-r.alpha)*self.seed
            difference = float(np.linalg.norm(updated-scores, 1))
            scores = updated
            bound = r.alpha*difference/(1-r.alpha)+64*np.finfo(float).eps*(1+np.linalg.norm(scores, 1))/(1-r.alpha)
            order = top_indices(scores, min(max(self.ks)+1, r.n))
            flags = {k: k == r.n or scores[order[k-1]]-scores[order[k]] > 2*bound for k in self.ks}
            if all(flags.values()) or bound <= 5e-13:
                return dict(scores=scores, sets={k: order[:k] for k in self.ks},
                    selected_scores={k: scores[order[:k]] for k in self.ks},
                    completion_certified=flags, score_error_bound=bound, fixed_point_refinement_steps=step+1)
        raise RuntimeError('Actual two-group view did not reach the declared numerical contract.')

    def solve(self, weights, materialize=False):
        if self.closed:
            raise RuntimeError('Family pins have been released.')
        started = perf_counter()
        a, semantic_weights = self.plan.scales(weights)
        key = tuple(map(float, a))
        self.logical_requests += 1
        self.symbolic_identity_requests += not np.any(a)
        if key not in self.seen and len(self.seen) == 16:
            raise ValueError('A family retains the declared sixteen distinct target summaries.')
        if key not in self.seen:
            self.seen.add(key)
            added = 8*len(key)+16
            self._seen_bytes += added
            if self.directory is not None:
                self.directory._live_bytes += added
        if not materialize and key in self.memo:
            self.memo_hits += 1
            result = _copy(self.memo[key])
            result.update(effective_reused=True, semantic_weights=np.array(semantic_weights, copy=True), seconds=perf_counter()-started)
            return result
        frozen = {}
        initial, calls, osp_repairs, native_refinements = {}, 0, 0, 0
        D = None
        def remember(result):
            for k, strict in _flags(result, self.ks).items():
                if strict and k not in frozen:
                    frozen[k] = (result['sets'][k].copy(), result['selected_scores'][k].copy())
        def done(result):
            return len(frozen) == len(self.ks) or _bound(result) <= 5e-13
        if self.route == 'native':
            result, D = self._native(a, self.native_mode)
            initial = result.get('initial_certified', {}).copy()
        else:
            try:
                result, c, d = self._initial(a, materialize)
                initial = result.get('initial_certified', {}).copy()
            except np.linalg.LinAlgError:
                if self.route != 'no_core':
                    raise
                result, D = self._native(a, 'osp_gap')
                osp_repairs += 1
        remember(result)
        calls += result.get('fallback_solve_calls', 0)
        if not done(result) and self.route != 'native' and self.output != 'fused' and D is None:
            result = complete_family(self, d, c, self.ks, started, {}, True)
            calls += result.get('fallback_solve_calls', 0)
            remember(result)
        if not done(result) and not (self.route == 'native' and self.native_mode == 'osp_gap') and not osp_repairs:
            result, D = self._native(a, 'osp_gap')
            osp_repairs += 1
            remember(result)
        if not done(result) and not (self.route == 'native' and self.native_mode == 'native_gap'):
            result, D = self._native(a, 'native_gap')
            native_refinements += 1
            remember(result)
        if not done(result):
            if D is None:
                D = self.plan.actual_delta(a)
            result = self._warm_finish(D, result['scores'])
            remember(result)
        sets, selected = {}, {}
        current_scores = result.get('scores')
        for k in self.ks:
            ids = frozen[k][0] if k in frozen else result['sets'][k].copy()
            if current_scores is not None:
                ids = ids[np.lexsort((ids, -current_scores[ids]))]
                selected[k] = current_scores[ids].copy()
            else:
                selected[k] = frozen[k][1] if k in frozen else result['selected_scores'][k].copy()
            sets[k] = ids
        final = dict(sets=sets, selected_scores=selected,
            initial_certified=initial, completion_certified={k: k in frozen for k in self.ks},
            final_score_error_bound=_bound(result), acceptance='strict' if len(frozen) == len(self.ks) else 'numeric_tolerance',
            candidate_nodes=result.get('candidate_nodes', self.ranker.n), fallback_solve_calls=calls,
            osp_repairs=osp_repairs, native_refinements=native_refinements,
            fixed_point_refinement_steps=result.get('fixed_point_refinement_steps', 0), route=self.route)
        # Only bounded small arrays enter the retained memo; never scores/error
        # vectors from the current output or completion workspace.
        old_bytes = _bytes(self.memo[key]) if key in self.memo else 0
        self.memo[key] = _copy(final)
        added = _bytes(self.memo[key])-old_bytes
        self._memo_bytes += added
        if self.directory is not None:
            self.directory._live_bytes += added
        assert self.nbytes <= self.reservation
        if self.directory is not None:
            self.directory._track()
        if materialize:
            if 'scores' not in result:
                raise RuntimeError('A materialized request needs its current bounded score workspace.')
            final['scores'] = result['scores']
            final['error_bounds'] = result.get('error_bounds', np.full(self.ranker.n, _bound(result)))
        final.update(effective_reused=False, semantic_weights=np.array(semantic_weights, copy=True), seconds=perf_counter()-started)
        return final

    def close(self):
        if self.closed:
            return
        if self.directory is not None:
            for key in self.keys:
                self.directory.pins[key] -= 1
                if not self.directory.pins[key]:
                    del self.directory.pins[key]
            self.directory.live.remove(self)
            self.directory._live_bytes -= self.nbytes
            self.directory._live_reserved_bytes -= self.reservation
        self.memo.clear()
        self.seen.clear()
        self._memo_bytes = self._seen_bytes = 0
        self.atoms = ()
        self.keys = ()
        self.plan = self.sources = self.core = self.rows = self.source_positions = None
        self.atom_positions = ()
        self.metadata_bytes = self.reservation = 0
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class NativeGroupFamily(_Family):
    """Matched no-atom point control using the identical lazy D compiler."""
    def __init__(self, plan, ks=(2, 5, 10), mode='native_gap', budget_bytes=4*1024**2):
        if mode not in ('native_gap', 'osp_gap', 'offset_gap'):
            raise ValueError('Native mode is native_gap, osp_gap or offset_gap.')
        super().__init__(plan, ks, native_mode=mode, budget_bytes=budget_bytes)
