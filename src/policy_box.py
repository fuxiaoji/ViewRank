"""Continuous independent-source policy boxes: one envelope, then exact corners.

Bounds may be a {source: (low, high)} mapping (zero directions are ignored),
or an (active_sources, 2) array in ``family.sources`` order.  All output cuts
are checked independently.  No parameter subdivision or numerical kernel is
introduced; uncertain point outputs use FastPolicyFamily's merged solve.
"""
from itertools import product
from time import perf_counter

import numpy as np

from .policy_family_fast import top_indices


_EPS = np.finfo(float).eps


def _gamma(length):
    value = max(1, int(length)) * _EPS
    return value / (1.0 - value)


def _upper(value, magnitude, length=4):
    return np.nextafter(value + _gamma(length) * magnitude, np.inf)


def _dot(a, b):
    """Upper bound for a nonnegative product, including dot rounding."""
    value = a @ b
    return _upper(value, value, a.shape[-1] + 2)


def _margin(scores, errors, chosen):
    if len(chosen) == len(scores):
        return np.inf
    upper = scores + errors
    upper[chosen] = -np.inf
    return float(np.min(scores[chosen] - errors[chosen]) - np.max(upper))


class PolicyBox:
    """Wrap an already prepared FastPolicyFamily; its cache remains unchanged."""

    def __init__(self, family):
        started = perf_counter()
        self.family = family
        self.sources = np.asarray(family.sources, dtype=np.int64)
        self.r = len(self.sources)
        ranker = family.ranker
        self.n = ranker.n
        self.alpha = ranker.alpha
        self.y = family.y
        self.H = family.core
        self.atoms = family.atoms
        self.masses = family.masses
        # One-norm residual allowance for the existing base and atom solves.
        anorm = float(np.max(np.asarray(abs(ranker.A).sum(axis=0))))
        pnorm = float(np.max(np.asarray(abs(ranker.P).sum(axis=0))))
        pnorm = float(_upper(pnorm, pnorm, self.n + 2))
        self.kappa = 1.0 / (1.0 - self.alpha * pnorm)
        rounding = _gamma(2 * self.n + 8)
        rhs = (1.0 - self.alpha) * family.seed
        ry = max(family.base_residual,
                 float(np.linalg.norm(rhs - ranker.A @ self.y, 1)))
        self.eta_y = self.kappa * (ry + rounding * (
            np.linalg.norm(rhs, 1) + anorm * np.linalg.norm(self.y, 1)))
        eta = []
        for atom in self.atoms:
            response_norm = (np.abs(atom.values).sum()
                             + (self.n - len(atom.rows)) * atom.tail_inf)
            eta.append(self.kappa * (atom.residual_l1 + rounding * (
                atom.update_l1 + anorm * response_norm)))
        self.eta = np.asarray(eta)
        self.EH = np.broadcast_to(self.eta, (self.r, self.r))
        retained = [atom.rows for atom in self.atoms] + [family.boundary_sources]
        self.retained_rows = np.unique(np.concatenate(retained))
        self.prepare_seconds = perf_counter() - started

    def _bounds(self, bounds):
        if hasattr(bounds, 'keys'):
            result = np.array([bounds[int(s)] for s in self.sources], dtype=float).reshape(self.r, 2)
        else:
            result = np.asarray(bounds, dtype=float).reshape(self.r, 2)
        if not np.all(np.isfinite(result)) or np.any(result <= 0) or np.any(result[:, 0] > result[:, 1]):
            raise ValueError('Each active source needs finite 0 < low <= high.')
        return result

    def envelope(self, bounds):
        """Posterior d enclosure and full-coordinate bounds over the entire box.

        ``valid=False`` is a failed sufficient bound, never an instability
        result.  Returned work arrays are transient and their bytes reported.
        """
        bounds = self._bounds(bounds)
        return self._envelope_a(self.family.scales(bounds[:, 0]),
                                self.family.scales(bounds[:, 1]))

    def _envelope_a(self, low, high):
        a0 = (low + high) / 2.0
        width = _upper((high - low) / 2.0, np.abs(low) + np.abs(high))
        center = 1.0 + a0 / (1.0 - self.masses * a0)
        F = np.eye(self.r) + a0[:, None] * self.H
        Q = np.linalg.solve(F, np.eye(self.r)) if self.r else F.copy()
        d0 = Q @ (a0 * self.y[self.sources])
        scores = self.y.copy()
        tails = np.array([atom.tail_inf for atom in self.atoms])
        for value, atom in zip(d0, self.atoms):
            scores[atom.rows] -= value * atom.values
        v = self.y[self.sources] - self.H @ d0
        scores[self.sources] = v
        result = dict(valid=False, center_strengths=center, d0=d0,
                      scores=scores, error_bounds=np.full(self.n, np.inf),
                      radius=None, contraction=np.inf)
        if self.r:
            aq, ah = abs(Q), abs(self.H)
            # Include formation errors of F, its inverse residual, d0 and v.
            ferr = _gamma(4) * (np.eye(self.r) + abs(a0[:, None] * self.H))
            inverse_residual = _upper(abs(np.eye(self.r) - Q @ F),
                                      np.eye(self.r) + aq @ abs(F), 2 * self.r + 6)
            inverse_residual += _dot(aq, ferr)
            r0 = _upper(abs(a0 * self.y[self.sources] - F @ d0),
                        abs(a0 * self.y[self.sources]) + abs(F) @ abs(d0), 2 * self.r + 6)
            r0 += _dot(ferr, abs(d0))
            vmax = _upper(abs(v), abs(self.y[self.sources]) + ah @ abs(d0), 2 * self.r + 6)
            abar = _upper(abs(a0) + width, abs(a0) + width)
            B = inverse_residual + _dot(aq * width[None, :], ah)
            B += _dot(aq * abar[None, :], self.EH)
            B = _upper(B, B, 8)
            f = _dot(aq, r0 + width * vmax + abar * (
                self.eta_y + _dot(self.EH, abs(d0))))
            f = _upper(f, f, 8)
            try:
                solutions = np.linalg.solve(np.eye(self.r) - B,
                                            np.column_stack([np.ones(self.r), f]))
            except np.linalg.LinAlgError:
                return result
            w = solutions[:, 0]
            if not np.all(np.isfinite(solutions)) or np.any(w <= 0):
                return result
            contraction = float(np.max(np.nextafter(_dot(B, w) / w, np.inf)))
            result['contraction'] = contraction
            if contraction >= 1:
                return result
            radius = np.maximum(solutions[:, 1], 0.0)
            deficit = _upper(f + _dot(B, radius) - radius,
                             f + _dot(B, radius) + radius, 6)
            correction = (max(float(np.max(deficit / w)), 0.0) / (1.0 - contraction)
                          + 32 * _EPS * (1.0 + float(np.max(radius))))
            radius = np.nextafter(radius + correction * w, np.inf)
            if np.any(radius < _upper(f + _dot(B, radius), f + _dot(B, radius), 4)):
                return result
        else:
            radius = np.empty(0)
            result['contraction'] = 0.0
        magnitude = _upper(abs(d0) + radius, abs(d0) + radius)
        common = self.eta_y + float(_dot(tails + self.eta, magnitude))
        errors = np.full(self.n, common)
        arithmetic = abs(self.y).copy()
        for j, atom in enumerate(self.atoms):
            errors[atom.rows] += abs(atom.values) * radius[j] - atom.tail_inf * magnitude[j]
            arithmetic[atom.rows] += abs(atom.values * d0[j])
        if self.r:
            errors[self.sources] = self.eta_y + _dot(abs(self.H), radius) + _dot(self.EH, magnitude)
            arithmetic[self.sources] = abs(self.y[self.sources]) + abs(self.H) @ abs(d0)
        # Subtraction of stored-tail contributions also gets an arithmetic bound.
        errors = _upper(np.maximum(errors, 0.0), abs(errors) + arithmetic + common,
                        8 * self.r + 16)
        result.update(valid=True, error_bounds=errors, radius=radius,
                      working_bytes=int(scores.nbytes + errors.nbytes + arithmetic.nbytes
                                        + 6 * self.r * self.r * 8 + 8 * self.r * 8))
        return result

    def _response_rows(self, nodes):
        rows = np.zeros((len(nodes), self.r))
        omitted = np.ones(rows.shape, dtype=bool)
        for j, atom in enumerate(self.atoms):
            positions = np.searchsorted(atom.rows, nodes)
            valid = positions < len(atom.rows)
            valid[valid] &= atom.rows[positions[valid]] == nodes[valid]
            rows[valid, j] = atom.values[positions[valid]]
            omitted[valid, j] = False
        positions = np.searchsorted(self.family.boundary_sources, nodes)
        valid = positions < len(self.family.boundary_sources)
        valid[valid] &= self.family.boundary_sources[positions[valid]] == nodes[valid]
        rows[valid] = self.family.boundary_matrix[positions[valid]]
        omitted[valid] = False
        return rows, omitted

    def _pair_lower(self, envelope, i, j, row_i, row_j, missing_i, missing_j):
        d0, radius = envelope['d0'], envelope['radius']
        magnitude = _upper(abs(d0) + radius, abs(d0) + radius)
        difference = row_i - row_j
        center = self.y[i] - self.y[j] - difference @ d0
        variation = float(_dot(abs(difference), radius))
        tails = np.array([atom.tail_inf for atom in self.atoms])
        truncation = float(_dot((missing_i.astype(float) + missing_j) * tails, magnitude))
        numerical = 2 * self.eta_y + 2 * float(_dot(self.eta, magnitude))
        arithmetic = _gamma(8 * self.r + 20) * (
            abs(self.y[i]) + abs(self.y[j]) + (abs(row_i) + abs(row_j)) @ magnitude
            + variation + truncation + numerical)
        return float(np.nextafter(center - variation - truncation - numerical - arithmetic, -np.inf))

    def pair_bound(self, envelope, i, j):
        """Lower bound on x_i(lambda)-x_j(lambda) throughout a valid envelope."""
        if not envelope['valid']:
            return -np.inf
        rows, missing = self._response_rows(np.array([i, j], dtype=np.int64))
        return self._pair_lower(envelope, i, j, rows[0], rows[1], missing[0], missing[1])

    def _pair_margin(self, envelope, selected):
        # Every stored row is included; outside these rows the coordinate upper
        # bound already covers all omitted output, including unseen candidates.
        outside = np.ones(self.n, dtype=bool)
        outside[self.retained_rows] = False
        outside[selected] = False
        scores, errors = envelope['scores'], envelope['error_bounds']
        lower = scores[selected] - errors[selected]
        margin = (float(np.min(lower) - np.max((scores + errors)[outside]))
                  if np.any(outside) else np.inf)
        if margin <= 0:
            return margin, 0
        candidates = np.setdiff1d(self.retained_rows, selected)
        disputed = candidates[(scores + errors)[candidates] >= np.min(lower)]
        clear = np.setdiff1d(candidates, disputed)
        if len(clear):
            margin = min(margin, float(np.min(lower) - np.max((scores + errors)[clear])))
        nodes = np.union1d(selected, disputed)
        rows, missing = self._response_rows(nodes)
        checks = 0
        for i in selected:
            pi = int(np.searchsorted(nodes, i))
            for j in disputed:
                if scores[i] - errors[i] > scores[j] + errors[j]:
                    bound = float(scores[i] - errors[i] - scores[j] - errors[j])
                else:
                    pj = int(np.searchsorted(nodes, j))
                    bound = self._pair_lower(envelope, i, j, rows[pi], rows[pj], missing[pi], missing[pj])
                    checks += 1
                margin = min(margin, bound)
                if margin <= 0:
                    return margin, checks
        return margin, checks

    def _point(self, strengths, ks):
        a = self.family.scales(strengths)
        bound = self._envelope_a(a, a)
        scores, errors = bound['scores'], bound['error_bounds']
        order = top_indices(scores, max(ks))
        margins = {k: _margin(scores, errors, order[:k]) for k in ks}
        pending = [k for k in ks if margins[k] <= 0]
        calls = 0
        if pending:
            completed = self.family.solve(strengths, ks=tuple(pending),
                                          always_fused=True, materialize=True)
            calls = completed['fallback_solve_calls']
            scores = completed['scores']
            # Independent residual of the complete requested view.
            ranker = self.family.ranker
            rhs = (1.0 - self.alpha) * self.family.seed
            residual = rhs - ranker.A @ scores
            scale_norm = 0.0
            for aj, source, atom in zip(a, self.sources, self.atoms):
                value = self.alpha * aj * scores[source]
                residual[atom.delta.indices] += value * atom.delta.data
                scale_norm += abs(value) * np.abs(atom.delta.data).sum()
            roundoff = _gamma(2 * self.n + 4 * self.r + 16) * (
                np.linalg.norm(rhs, 1) + (1 + self.alpha) * np.linalg.norm(scores, 1) + scale_norm)
            error = self.kappa * (np.linalg.norm(residual, 1) + roundoff)
            errors = np.full(self.n, np.nextafter(error, np.inf))
            order = top_indices(scores, max(ks))
            # A completed solve may certify other cuts; a certified cut is never rerun.
            for k in pending:
                margins[k] = _margin(scores, errors, order[:k])
        return dict(sets={k: order[:k].copy() for k in ks}, margins=margins,
                    fallback_solve_calls=calls)

    def query(self, bounds, ks=(2, 5, 10), use_envelope=True, use_pair=False):
        """Per-k stable / unstable / numerical_tie results and actual witnesses.

        Set use_envelope=False for the complete-corner baseline with the same
        point oracle.  use_pair=True adds shared-coefficient pair bounds only
        for failed box cuts. Preparation of the family is charged separately.
        """
        started = perf_counter()
        bounds = self._bounds(bounds)
        ks = tuple(sorted(set(map(int, ks))))
        if not ks or ks[0] < 1 or ks[-1] > self.n:
            raise ValueError('Every k must lie in [1, number of nodes].')
        if use_envelope:
            envelope = self.envelope(bounds)
        else:
            a0 = (self.family.scales(bounds[:, 0]) + self.family.scales(bounds[:, 1])) / 2
            envelope = dict(valid=False, center_strengths=1 + a0 / (1 - self.masses * a0))
        answers = {}
        counts = {k: 0 for k in ks}
        calls = 0
        pair_checks = 0
        if use_envelope and envelope['valid']:
            order = top_indices(envelope['scores'], max(ks))
            for k in ks:
                margin = _margin(envelope['scores'], envelope['error_bounds'], order[:k])
                if margin > 0:
                    answers[k] = dict(status='stable', indices=order[:k].copy(),
                                      margin=margin, method='box_envelope', witness=None)
                elif use_pair:
                    margin, checks = self._pair_margin(envelope, order[:k])
                    pair_checks += checks
                    if margin > 0:
                        answers[k] = dict(status='stable', indices=order[:k].copy(),
                                          margin=margin, method='box_pair', witness=None)
        pending = [k for k in ks if k not in answers]
        reference = {}
        minimum = {}
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
                    entering = sorted(set(point['sets'][k]) - set(reference[k]))
                    leaving = sorted(set(reference[k]) - set(point['sets'][k]))
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
                    center_strengths=envelope['center_strengths'],
                    envelope_valid=envelope['valid'], corners_visited=visited,
                    possible_corners=2 ** len(varying), corners_per_k=counts,
                    pair_checks=pair_checks,
                    fallback_solve_calls=calls, seconds=perf_counter() - started,
                    prepare_seconds=self.prepare_seconds)

