"""Continuous-domain execution on retained output rows and one outside bound.

The sparse and dense-atom executors share the identical posterior coefficient
enclosure and complete-corner/witness schedule. A failed enclosure is never
an instability result. Existing PolicyBox and V7 execution files are unchanged.
"""
from itertools import product
from time import perf_counter

import numpy as np

from .policy_box import PolicyBox, _EPS, _dot, _gamma, _margin, _upper
from .policy_execution_v7 import complete_family, top_indices
from .semantic_basis import strength_scale


def box_graph_constants(ranker):
    """Compute immutable graph norms once, charged in shared graph setup.

    Pass the same returned object to sparse, dense and corner executors.
    Its setup_seconds is a measured operator component, not free metadata.
    """
    started = perf_counter()
    n, alpha = ranker.n, ranker.alpha
    anorm = float(np.max(np.asarray(abs(ranker.A).sum(axis=0))))
    pnorm = float(np.max(np.asarray(abs(ranker.P).sum(axis=0))))
    pnorm = float(_upper(pnorm, pnorm, n+2))
    return dict(anorm=anorm, pnorm=pnorm, kappa=1./(1.-alpha*pnorm),
        rounding=_gamma(2*n+8), setup_seconds=perf_counter()-started,
        numeric_payload_bytes=32)


def coefficient_enclosure(low, high, y_source, H, masses, eta_y, eta):
    """The existing PolicyBox posterior formula, with no graph-size arrays.

    Returns d0 and a verified componentwise radius for all parameters in the
    box. Every intermediate matrix is r by r. No subdivision is introduced.
    """
    r = len(masses)
    a0 = (low+high)/2.
    width = _upper((high-low)/2., abs(low)+abs(high))
    center = 1.+a0/(1.-masses*a0)
    F = np.eye(r)+a0[:, None]*H
    Q = np.linalg.solve(F, np.eye(r)) if r else F.copy()
    d0 = Q@(a0*y_source)
    v = y_source-H@d0
    result = dict(valid=False, center_strengths=center, d0=d0, v=v,
                  radius=None, contraction=np.inf)
    if r:
        EH = np.broadcast_to(eta, (r, r))
        aq, ah = abs(Q), abs(H)
        ferr = _gamma(4)*(np.eye(r)+abs(a0[:, None]*H))
        inverse_residual = _upper(abs(np.eye(r)-Q@F),
            np.eye(r)+aq@abs(F), 2*r+6)
        inverse_residual += _dot(aq, ferr)
        r0 = _upper(abs(a0*y_source-F@d0),
            abs(a0*y_source)+abs(F)@abs(d0), 2*r+6)
        r0 += _dot(ferr, abs(d0))
        vmax = _upper(abs(v), abs(y_source)+ah@abs(d0), 2*r+6)
        abar = _upper(abs(a0)+width, abs(a0)+width)
        B = inverse_residual+_dot(aq*width[None, :], ah)
        B += _dot(aq*abar[None, :], EH)
        B = _upper(B, B, 8)
        f = _dot(aq, r0+width*vmax+abar*(eta_y+_dot(EH, abs(d0))))
        f = _upper(f, f, 8)
        try:
            solutions = np.linalg.solve(np.eye(r)-B,
                np.column_stack([np.ones(r), f]))
        except np.linalg.LinAlgError:
            return result
        w = solutions[:, 0]
        if not np.all(np.isfinite(solutions)) or np.any(w <= 0):
            return result
        contraction = float(np.max(np.nextafter(_dot(B, w)/w, np.inf)))
        result['contraction'] = contraction
        if contraction >= 1:
            return result
        radius = np.maximum(solutions[:, 1], 0.)
        deficit = _upper(f+_dot(B, radius)-radius,
            f+_dot(B, radius)+radius, 6)
        correction = (max(float(np.max(deficit/w)), 0.)/(1.-contraction)
            +32*_EPS*(1.+float(np.max(radius))))
        radius = np.nextafter(radius+correction*w, np.inf)
        if np.any(radius < _upper(f+_dot(B, radius), f+_dot(B, radius), 4)):
            return result
    else:
        radius = np.empty(0)
        result['contraction'] = 0.
    result.update(valid=True, radius=radius,
        magnitude=_upper(abs(d0)+radius, abs(d0)+radius),
        coefficient_working_bytes=int(6*r*r*8+8*r*8))
    return result


class _DenseFamilyView:
    """Expose a dense atomic family in active-source coefficient coordinates."""
    def __init__(self, family):
        self.inner = family
        for name in ('ranker', 'seed', 'sources', 'masses', 'core', 'atoms',
                     'y', 'base_residual', 'directory'):
            setattr(self, name, getattr(family, name))
        if hasattr(family, '_base_pad'):
            self._base_pad = family._base_pad

    @property
    def nbytes(self):
        return self.inner.nbytes

    def scales(self, strengths):
        strengths = np.broadcast_to(np.asarray(strengths, dtype=float), self.masses.shape)
        return np.array([strength_scale(s, m) for s, m in zip(strengths, self.masses)])

    def coefficients(self, strengths):
        a = self.scales(strengths)
        K = np.eye(len(a))+self.core*a[None, :]
        c = np.linalg.solve(K, self.y[self.sources]) if len(a) else np.empty(0)
        return a, c, a*c, K


class DenseEnvelopePolicyBox(PolicyBox):
    """Full-output reference using the same small enclosure as sparse output.

    A normal sparse-response FastPolicyFamily gives the full-coordinate
    ablation. A compiled dense-atom family gives DenseAtoms-BoxEnvelope with
    zero response truncation. Both receive the identical domain algorithm.
    """
    def __init__(self, family, graph_constants=None):
        started = perf_counter()
        self.dense_atoms = bool(family.atoms and
            all(getattr(atom, 'response', None) is not None for atom in family.atoms))
        # An empty dense family has a dense route but no active directions.
        self.dense_atoms |= getattr(family, 'route', None) == 'dense'
        self.family = _DenseFamilyView(family) if self.dense_atoms else family
        self.sources = np.asarray(family.sources, dtype=np.int64)
        self.r = len(self.sources)
        self.n = family.ranker.n
        self.alpha = family.ranker.alpha
        self.y, self.H = family.y, family.core
        self.atoms, self.masses = family.atoms, family.masses
        self.graph_constants = (box_graph_constants(family.ranker)
            if graph_constants is None else graph_constants)
        anorm = self.graph_constants['anorm']
        self.kappa = self.graph_constants['kappa']
        rounding = self.graph_constants['rounding']
        rhs = (1.-self.alpha)*family.seed
        ry = max(family.base_residual,
            float(np.linalg.norm(rhs-family.ranker.A@self.y, 1)))
        self.eta_y = self.kappa*(ry+rounding*(
            np.linalg.norm(rhs, 1)+anorm*np.linalg.norm(self.y, 1)))
        eta = []
        for atom in self.atoms:
            response_norm = (np.linalg.norm(atom.response, 1) if self.dense_atoms else
                abs(atom.values).sum()+(self.n-len(atom.rows))*atom.tail_inf)
            eta.append(self.kappa*(atom.residual_l1+rounding*(atom.update_l1+anorm*response_norm)))
        self.eta = np.asarray(eta)
        self.EH = np.broadcast_to(self.eta, (self.r, self.r))
        # Dense coordinates are implicit. There is no n-element ID index.
        retained = [atom.rows for atom in self.atoms]+[family.boundary_sources] if not self.dense_atoms else []
        self.retained_rows = np.unique(np.concatenate(retained)) if retained else None
        self.prepare_seconds = perf_counter()-started

    @property
    def metadata_bytes(self):
        return self.eta.nbytes+8+(0 if self.retained_rows is None else self.retained_rows.nbytes)

    def _domain(self, low, high):
        return coefficient_enclosure(low, high, self.y[self.sources], self.H,
            self.masses, self.eta_y, self.eta)

    def _envelope_a(self, low, high):
        result = self._domain(low, high)
        d0 = result['d0']
        scores = self.y.copy()
        if self.dense_atoms:
            for value, atom in zip(d0, self.atoms):
                scores -= value*atom.response
        else:
            for value, atom in zip(d0, self.atoms):
                scores[atom.rows] -= value*atom.values
        scores[self.sources] = result['v']
        result.update(scores=scores, error_bounds=np.full(self.n, np.inf),
                      coordinate_count=self.n)
        if not result['valid']:
            return result
        radius, magnitude = result['radius'], result['magnitude']
        tails = (np.zeros(self.r) if self.dense_atoms else
                 np.array([atom.tail_inf for atom in self.atoms]))
        common = self.eta_y+float(_dot(tails+self.eta, magnitude))
        errors = np.full(self.n, common)
        arithmetic = abs(self.y).copy()
        for j, atom in enumerate(self.atoms):
            if self.dense_atoms:
                errors += abs(atom.response)*radius[j]
                arithmetic += abs(atom.response*d0[j])
            else:
                errors[atom.rows] += abs(atom.values)*radius[j]-atom.tail_inf*magnitude[j]
                arithmetic[atom.rows] += abs(atom.values*d0[j])
        if self.r:
            errors[self.sources] = self.eta_y+_dot(abs(self.H), radius)+_dot(self.EH, magnitude)
            arithmetic[self.sources] = abs(self.y[self.sources])+abs(self.H)@abs(d0)
        errors = _upper(np.maximum(errors, 0.), abs(errors)+arithmetic+common, 8*self.r+16)
        result.update(error_bounds=errors, working_bytes=int(scores.nbytes+
            errors.nbytes+arithmetic.nbytes+result['coefficient_working_bytes']))
        return result

    def _response_rows(self, nodes):
        if not self.dense_atoms:
            return super()._response_rows(nodes)
        rows = (np.column_stack([atom.response[nodes] for atom in self.atoms])
                if self.atoms else np.empty((len(nodes), 0)))
        return rows, np.zeros(rows.shape, dtype=bool)

    def _pair_lower(self, envelope, i, j, row_i, row_j, missing_i, missing_j):
        if not self.dense_atoms:
            return super()._pair_lower(envelope, i, j, row_i, row_j, missing_i, missing_j)
        d0, radius, magnitude = envelope['d0'], envelope['radius'], envelope['magnitude']
        difference = row_i-row_j
        center = self.y[i]-self.y[j]-difference@d0
        variation = float(_dot(abs(difference), radius))
        numerical = 2*self.eta_y+2*float(_dot(self.eta, magnitude))
        arithmetic = _gamma(8*self.r+20)*(abs(self.y[i])+abs(self.y[j])+
            (abs(row_i)+abs(row_j))@magnitude+variation+numerical)
        return float(np.nextafter(center-variation-numerical-arithmetic, -np.inf))

    def _pair_margin(self, envelope, selected):
        if not self.dense_atoms:
            return super()._pair_margin(envelope, selected)
        scores, errors = envelope['scores'], envelope['error_bounds']
        mask = np.ones(self.n, dtype=bool)
        mask[selected] = False
        lower = float(np.min(scores[selected]-errors[selected]))
        disputed = np.flatnonzero(mask & (scores+errors >= lower))
        clear = mask & (scores+errors < lower)
        margin = (float(lower-np.max((scores+errors)[clear])) if np.any(clear) else np.inf)
        nodes = np.union1d(selected, disputed)
        rows, missing = self._response_rows(nodes)
        checks = 0
        for i in selected:
            pi = int(np.searchsorted(nodes, i))
            for j in disputed:
                pj = int(np.searchsorted(nodes, j))
                bound = self._pair_lower(envelope, i, j,
                    rows[pi], rows[pj], missing[pi], missing[pj])
                checks += 1
                margin = min(margin, bound)
                if margin <= 0:
                    return margin, checks
        return margin, checks

    def _completed_point(self, strengths, ks):
        # Both representations receive exactly the same merged-point fallback.
        started = perf_counter()
        a, c, d, _ = self.family.coefficients(strengths)
        completed = complete_family(self.family, d, c, ks, started, {}, True)
        scores = completed['scores']
        ranker = self.family.ranker
        rhs = (1.-self.alpha)*self.family.seed
        residual = rhs-ranker.A@scores
        scale_norm = 0.
        for aj, source, atom in zip(a, self.sources, self.atoms):
            value = self.alpha*aj*scores[source]
            delta = atom.delta
            residual[delta.indices] += value*delta.data
            scale_norm += abs(value)*abs(delta.data).sum()
        roundoff = _gamma(2*self.n+4*self.r+16)*(
            np.linalg.norm(rhs, 1)+(1+self.alpha)*np.linalg.norm(scores, 1)+scale_norm)
        error = self.kappa*(np.linalg.norm(residual, 1)+roundoff)
        return scores, np.full(self.n, np.nextafter(error, np.inf)), completed['fallback_solve_calls']

    def _point(self, strengths, ks):
        a = self.family.scales(strengths)
        envelope = self._envelope_a(a, a)
        scores, errors = envelope['scores'], envelope['error_bounds']
        order = top_indices(scores, max(ks))
        margins = {k: _margin(scores, errors, order[:k]) for k in ks}
        sets = {k: order[:k].copy() for k in ks}
        pending = [k for k in ks if margins[k] <= 0]
        calls = 0
        if pending:
            scores, errors, calls = self._completed_point(strengths, tuple(pending))
            order = top_indices(scores, max(ks))
            for k in pending:
                margins[k] = _margin(scores, errors, order[:k])
                sets[k] = order[:k].copy()
        return dict(sets=sets, margins=margins,
                    fallback_solve_calls=calls)


class DenseAtomsPolicyBox(DenseEnvelopePolicyBox):
    """Dense-atomic response comparator with the full shared box envelope."""
    def __init__(self, family, graph_constants=None):
        if getattr(family, 'route', None) != 'dense':
            raise ValueError('A prepared dense atomic family is required.')
        super().__init__(family, graph_constants=graph_constants)


class SparsePolicyBox(DenseEnvelopePolicyBox):
    """Candidate-only domain bounds; every unvisited graph node is covered.

    The prepared family's candidate rows already contain every stored atom
    row, the complete source boundary, and its kmax best unaffected base
    nodes. Query k must not exceed that prepared kmax. Sparse arrays borrow
    existing family indexes rather than making an n-dimensional side index.
    """
    def __init__(self, family, graph_constants=None):
        started = perf_counter()
        if not hasattr(family, 'rows'):
            raise ValueError('Sparse box execution requires prepared candidate output rows.')
        super().__init__(family, graph_constants=graph_constants)
        if self.dense_atoms:
            raise ValueError('Use DenseAtomsPolicyBox for full response vectors.')
        self.rows = family.rows
        self.candidate_k = family.kmax
        self.atom_positions = family.atom_positions
        self.source_positions = family.source_positions
        self.outside_base = family.outside_base
        # One per-question scan, charged as preparation, and no retained mask.
        # A global |y| bound also covers signed roundoff in an approximate base.
        self.outside_abs_base = float(np.max(abs(self.y)))
        self.prepare_seconds = perf_counter()-started

    @property
    def metadata_bytes(self):
        return super().metadata_bytes+16  # two retained outside-bound scalars

    def _envelope_a(self, low, high):
        result = self._domain(low, high)
        d0 = result['d0']
        scores = self.y[self.rows].copy()
        for value, atom, pos in zip(d0, self.atoms, self.atom_positions):
            scores[pos] -= value*atom.values
        scores[self.source_positions] = result['v']
        result.update(scores=scores, error_bounds=np.full(len(self.rows), np.inf),
            coordinate_nodes=self.rows, coordinate_count=len(self.rows), outside_upper=np.inf)
        if not result['valid']:
            return result
        radius, magnitude = result['radius'], result['magnitude']
        tails = np.array([atom.tail_inf for atom in self.atoms])
        common = self.eta_y+float(_dot(tails+self.eta, magnitude))
        errors = np.full(len(self.rows), common)
        arithmetic = abs(self.y[self.rows]).copy()
        for j, (atom, pos) in enumerate(zip(self.atoms, self.atom_positions)):
            errors[pos] += abs(atom.values)*radius[j]-atom.tail_inf*magnitude[j]
            arithmetic[pos] += abs(atom.values*d0[j])
        if self.r:
            errors[self.source_positions] = self.eta_y+_dot(abs(self.H), radius)+_dot(self.EH, magnitude)
            arithmetic[self.source_positions] = abs(self.y[self.sources])+abs(self.H)@abs(d0)
        errors = _upper(np.maximum(errors, 0.), abs(errors)+arithmetic+common, 8*self.r+16)
        outside_error = float(_upper(common, 2*abs(common)+self.outside_abs_base, 8*self.r+16))
        outside_upper = (float(np.nextafter(self.outside_base+outside_error, np.inf))
                         if np.isfinite(self.outside_base) else -np.inf)
        result.update(error_bounds=errors, outside_error=outside_error,
            outside_upper=outside_upper, working_bytes=int(scores.nbytes+
                errors.nbytes+arithmetic.nbytes+result['coefficient_working_bytes']))
        return result

    def _candidate_margin(self, envelope, chosen):
        if len(chosen) == self.n:
            return np.inf
        scores, errors = envelope['scores'], envelope['error_bounds']
        upper = scores+errors
        upper[chosen] = -np.inf
        outside = max(float(np.max(upper)), envelope['outside_upper'])
        return float(np.min(scores[chosen]-errors[chosen])-outside)

    def _pair_margin(self, envelope, selected):
        positions = np.searchsorted(self.rows, selected)
        scores, errors = envelope['scores'], envelope['error_bounds']
        lower = float(np.min(scores[positions]-errors[positions]))
        margin = lower-envelope['outside_upper']
        if margin <= 0:
            return margin, 0
        mask = np.ones(len(self.rows), dtype=bool)
        mask[positions] = False
        disputed_pos = np.flatnonzero(mask & (scores+errors >= lower))
        clear = mask & (scores+errors < lower)
        if np.any(clear):
            margin = min(margin, float(lower-np.max((scores+errors)[clear])))
        disputed = self.rows[disputed_pos]
        nodes = np.union1d(selected, disputed)
        response_rows, missing = self._response_rows(nodes)
        checks = 0
        for i in selected:
            pi = int(np.searchsorted(nodes, i))
            si = int(np.searchsorted(self.rows, i))
            for j, sj in zip(disputed, disputed_pos):
                if scores[si]-errors[si] > scores[sj]+errors[sj]:
                    bound = float(scores[si]-errors[si]-scores[sj]-errors[sj])
                else:
                    pj = int(np.searchsorted(nodes, j))
                    bound = self._pair_lower(envelope, i, j,
                        response_rows[pi], response_rows[pj], missing[pi], missing[pj])
                    checks += 1
                margin = min(margin, bound)
                if margin <= 0:
                    return margin, checks
        return margin, checks

    def _point(self, strengths, ks):
        a = self.family.scales(strengths)
        envelope = self._envelope_a(a, a)
        order = top_indices(envelope['scores'], max(ks))
        margins = {k: self._candidate_margin(envelope, order[:k]) for k in ks}
        sets = {k: self.rows[order[:k]].copy() for k in ks}
        pending = [k for k in ks if margins[k] <= 0]
        calls = 0
        if pending:
            scores, errors, calls = self._completed_point(strengths, tuple(pending))
            full_order = top_indices(scores, max(ks))
            for k in pending:
                margins[k] = _margin(scores, errors, full_order[:k])
                sets[k] = full_order[:k].copy()
        return dict(sets=sets, margins=margins, fallback_solve_calls=calls)

    def query(self, bounds, ks=(2, 5, 10), use_envelope=True, use_pair=False):
        started = perf_counter()
        bounds = self._bounds(bounds)
        ks = tuple(sorted(set(map(int, ks))))
        if not ks or ks[0] < 1 or ks[-1] > self.n or ks[-1] > self.candidate_k:
            raise ValueError('Requested k must lie within the prepared candidate cutoff.')
        if use_envelope:
            envelope = self.envelope(bounds)
        else:
            a0 = (self.family.scales(bounds[:, 0])+self.family.scales(bounds[:, 1]))/2
            envelope = dict(valid=False, center_strengths=1+a0/(1-self.masses*a0))
        answers, counts = {}, {k: 0 for k in ks}
        calls = pair_checks = 0
        if use_envelope and envelope['valid']:
            order = top_indices(envelope['scores'], max(ks))
            for k in ks:
                margin = self._candidate_margin(envelope, order[:k])
                selected = self.rows[order[:k]]
                if margin > 0:
                    answers[k] = dict(status='stable', indices=selected.copy(),
                        margin=margin, method='box_envelope', witness=None)
                elif use_pair:
                    margin, checks = self._pair_margin(envelope, selected)
                    pair_checks += checks
                    if margin > 0:
                        answers[k] = dict(status='stable', indices=selected.copy(),
                            margin=margin, method='box_pair', witness=None)
        pending = [k for k in ks if k not in answers]
        reference, minimum = {}, {}
        if pending:
            point = self._point(envelope['center_strengths'], pending)
            calls += point['fallback_solve_calls']
            for k in pending:
                reference[k] = point['sets'][k]
                minimum[k] = np.inf
                if point['margins'][k] <= 0:
                    answers[k] = dict(status='numerical_tie', indices=reference[k],
                        margin=point['margins'][k], method='center', witness=None)
        varying = np.flatnonzero(bounds[:, 0] != bounds[:, 1])
        visited = 0
        for choices in product((0, 1), repeat=len(varying)):
            pending = [k for k in ks if k not in answers]
            if not pending:
                break
            strengths = bounds[:, 0].copy()
            for axis, side in zip(varying, choices):
                strengths[axis] = bounds[axis, side]
            point = self._point(strengths, pending)
            visited += 1
            calls += point['fallback_solve_calls']
            for k in pending:
                counts[k] += 1
                if point['margins'][k] <= 0:
                    answers[k] = dict(status='numerical_tie', indices=reference[k],
                        margin=point['margins'][k], method='corner', witness=None)
                elif set(point['sets'][k]) != set(reference[k]):
                    entering = sorted(set(point['sets'][k])-set(reference[k]))
                    leaving = sorted(set(reference[k])-set(point['sets'][k]))
                    witness = dict(sources=self.sources.copy(), strengths=strengths.copy(),
                        indices=point['sets'][k].copy(), margin=point['margins'][k],
                        entering_node=int(entering[0]), leaving_node=int(leaving[0]),
                        pair_gap_lower_bound=point['margins'][k])
                    answers[k] = dict(status='unstable', indices=reference[k],
                        margin=point['margins'][k], method='corner', witness=witness)
                else:
                    minimum[k] = min(minimum[k], point['margins'][k])
        for k in ks:
            if k not in answers:
                answers[k] = dict(status='stable', indices=reference[k],
                    margin=minimum[k], method='all_corners', witness=None)
        return dict(results=answers, active_sources=self.sources.copy(),
            center_strengths=envelope['center_strengths'], envelope_valid=envelope['valid'],
            corners_visited=visited, possible_corners=2**len(varying), corners_per_k=counts,
            pair_checks=pair_checks, fallback_solve_calls=calls,
            candidate_nodes=len(self.rows), named_envelope_working_bytes=envelope.get('working_bytes'),
            seconds=perf_counter()-started, prepare_seconds=self.prepare_seconds)


class _NativeCoordinates:
    def __init__(self, masses):
        self.masses = masses

    def scales(self, strengths):
        strengths = np.broadcast_to(np.asarray(strengths, dtype=float), self.masses.shape)
        return np.array([strength_scale(s, m) for s, m in zip(strengths, self.masses)])


class NativePolicyCorners(SparsePolicyBox):
    """Direct native corner control, with no response-atom preparation.

    Each strict set keeps the gap/bound from the exact history entry that
    certified that frozen set, even if later native levels change scores.
    """
    def __init__(self, native_family):
        started = perf_counter()
        self.native_family = native_family
        self.n = native_family.ranker.n
        self.alpha = native_family.ranker.alpha
        self.original_sources = native_family.sources
        active = sorted((int(s), i) for i, (s, direction) in enumerate(
            zip(native_family.sources, native_family.directions)) if direction.nnz)
        self.sources = np.array([s for s, _ in active], dtype=np.int64)
        self.r = len(self.sources)
        self.active_positions = np.array([i for _, i in active], dtype=np.int64)
        self.masses = native_family.masses[self.active_positions]
        self.family = _NativeCoordinates(self.masses)
        self.candidate_k = self.n
        self.rows = np.empty(0, dtype=np.int64)
        self._base_strengths = np.ones(len(self.original_sources))
        self.prepare_seconds = perf_counter()-started

    @property
    def metadata_bytes(self):
        return self.sources.nbytes+self.active_positions.nbytes+self.masses.nbytes+self._base_strengths.nbytes

    def _point(self, strengths, ks):
        full = self._base_strengths.copy()
        full[self.active_positions] = strengths
        result = self.native_family.solve(full, ks=ks)
        margins = {}
        for k in ks:
            if not result['certified'].get(k, False):
                margins[k] = 0.
                continue
            level = result['certification_level'][k]
            bound = result['certificate_error_bounds'][k]
            matched = next((entry for entry in result['certificate_history']
                if entry['level'] == level and entry['error_bound'] == bound
                and entry['certified'].get(k, False)), None)
            if matched is None:
                raise RuntimeError('The frozen native cut has no matching certificate history.')
            margins[k] = float(matched['gaps'][k]-2*bound)
        return dict(sets={k: result['sets'][k].copy() for k in ks}, margins=margins,
            fallback_solve_calls=result['native_solve_calls'])

    def query(self, bounds, ks=(2, 5, 10), use_envelope=False, use_pair=False):
        if hasattr(bounds, 'keys'):
            # Zero directions are fixed at an actual legal original lower
            # bound instead of being assigned another source's parameter.
            self._base_strengths[:] = [bounds[int(s)][0] for s in self.original_sources]
        return super().query(bounds, ks=ks, use_envelope=False, use_pair=False)
