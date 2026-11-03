"""Dependency-loss control for the continuous policy-domain certificate.

The same exact small system and stored response contract are used as in the
polynomial method. Within each cell, every F_j/D is bounded independently;
the output scan therefore discards their shared-parameter dependence. This
is a sound interval control, with identical optional spatial subdivision.
"""
from fractions import Fraction
import math

import numpy as np

from .policy_domain_polynomial import Interval, exact
from .policy_domain_controls import controls_fast


def scratch_upper_bound(domain, n_nodes, chunk_rows=512, n_cuts=3,
                        n_candidates=None):
    count = int(n_nodes if n_candidates is None else n_candidates)
    rows = min(int(chunk_rows), count)
    q, controls = 2*domain.r, (domain.r+1)**2
    small = 8*controls*(24+8*q+6*int(n_cuts))+1024
    prepare = int(n_nodes)+count*(40+q)+16*int(n_cuts)*count
    chunks = rows*(8*q+q+12*8)
    return int(small+max(prepare,chunks))


def _enclose(value):
    interval = Interval.of(value)
    return interval.lo, interval.hi


def certify_many(domain, y, candidate_rows, retained_values, retained_mask,
                 tail_bounds, selected_sets, *, kappa, base_residual,
                 atom_residuals, round_pad=0, subdivide_depth=0,
                 retained_columns=None, chunk_rows=2048):
    """Certify all requested cutoffs using independent coefficient intervals.

    Full response columns may be borrowed without a candidate-by-q copy.
    Missing coordinates use the same atom tails as the polynomial executor.
    STABLE is conditional on the caller's uniform residual and tail contracts.
    UNKNOWN is not an instability witness or a completed stability decision.
    """
    y = np.asarray(y, dtype=np.float64)
    rows = np.asarray(candidate_rows, dtype=np.int64)
    selections = {int(k): tuple(map(int, ids)) for k, ids in selected_sets.items()}
    q = 2*domain.r
    values = None if retained_values is None else np.asarray(retained_values, dtype=np.float64)
    masks = None if retained_mask is None else np.asarray(retained_mask, dtype=bool)
    if y.ndim != 1 or not np.all(np.isfinite(y)) or rows.ndim != 1:
        raise ValueError('Finite full base and one-dimensional candidate rows required.')
    if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(y)):
        raise ValueError('Candidate IDs must be unique graph nodes.')
    if values is None:
        if retained_columns is None or len(retained_columns) != q:
            raise ValueError('Provide retained values or all borrowed response columns.')
    elif values.shape != (len(rows), q) or not np.all(np.isfinite(values)):
        raise ValueError('Retained values must have candidate-by-2r shape.')
    if masks is not None and masks.shape != (len(rows), q):
        raise ValueError('Retained mask shape differs.')
    row_set = set(map(int, rows))
    if not selections or any(k < 1 or len(ids) != k or len(set(ids)) != k
                             or not set(ids) <= row_set for k, ids in selections.items()):
        raise ValueError('Selected sets must match their cutoffs and lie in candidates.')
    tails = tuple(exact(v) for v in tail_bounds)
    residuals = tuple(exact(v) for v in atom_residuals)
    kap, rho, pad = map(exact, (kappa, base_residual, round_pad))
    if len(tails) != q or len(residuals) != q or any(v < 0 for v in tails+residuals) or kap <= 0 or rho < 0 or pad < 0:
        raise ValueError('Invalid response error contract.')
    if int(subdivide_depth) != subdivide_depth or subdivide_depth < 0 or chunk_rows < 1:
        raise ValueError('Nonnegative depth and positive chunk size required.')
    positions = {int(node): i for i, node in enumerate(rows)}
    for i, source in enumerate(domain.source_ids):
        if source not in positions or exact(y[source]) != domain.y_sources[i]:
            raise ValueError('Base source coordinates differ from the compiled core.')
        position = positions[source]
        if masks is not None and not np.all(masks[position]):
            raise ValueError('Every exact source interaction must be retained.')
        actual = (values[position] if values is not None else
                  [column[source] for column in retained_columns])
        if tuple(map(exact, actual)) != domain.H_m[i]+domain.H_w[i]:
            raise ValueError('Stored source interactions differ from the compiled core.')
    selected_masks = {k: np.isin(rows, ids) for k, ids in selections.items()}
    outside_mask = np.ones(len(y), dtype=bool)
    outside_mask[rows] = False
    outside = float(np.max(y[outside_mask])) if np.any(outside_mask) else None
    peak = 0
    attempts = 0

    def cell(box):
        nonlocal peak
        dc, fc = controls_fast(domain, box)
        dlo, dhi = min(dc), max(dc)
        if dlo <= 0:
            return {k: dict(status='UNKNOWN', reason='Cannot establish positive denominator on the cell.') for k in selections}
        coeff_lo, coeff_hi, amplitudes = [], [], []
        for controls in fc:
            endpoints = [min(controls)/dlo, min(controls)/dhi,
                         max(controls)/dlo, max(controls)/dhi]
            lo, hi = min(endpoints), max(endpoints)
            coeff_lo.append(_enclose(lo)[0]); coeff_hi.append(_enclose(hi)[1])
            amplitudes.append(max(abs(lo), abs(hi)))
        error = kap*(rho+sum((a*r for a, r in zip(amplitudes, residuals)), Fraction(0)))+pad
        error_hi = _enclose(error)[1]
        node_tail = np.array([_enclose(a*t)[1] for a, t in zip(amplitudes, tails)])
        if not np.all(np.isfinite(coeff_lo+coeff_hi)) or not math.isfinite(error_hi):
            return {k: dict(status='UNKNOWN', reason='Coefficient/error interval overflow.') for k in selections}
        minimum = {k: math.inf for k in selections}
        maximum = {k: -math.inf for k in selections}
        for start in range(0, len(rows), int(chunk_rows)):
            end = min(start+int(chunk_rows), len(rows))
            nodes = rows[start:end]
            block = (values[start:end] if values is not None else
                     np.column_stack([column[nodes] for column in retained_columns]))
            if not np.all(np.isfinite(block)):
                raise ValueError('Response coordinates must be finite stored scalars.')
            present = None if masks is None else masks[start:end]
            low = y[nodes].copy(); high = y[nodes].copy()
            for j in range(q):
                value = block[:, j] if present is None else np.where(present[:, j], block[:, j], 0.)
                a = value*coeff_lo[j]; b = value*coeff_hi[j]
                product_low = np.nextafter(np.minimum(a, b), -np.inf)
                product_high = np.nextafter(np.maximum(a, b), np.inf)
                low = np.nextafter(low-product_high, -np.inf)
                high = np.nextafter(high-product_low, np.inf)
                if present is not None:
                    tail = np.where(present[:, j], 0., node_tail[j])
                    low = np.nextafter(low-tail, -np.inf)
                    high = np.nextafter(high+tail, np.inf)
            low = np.nextafter(low-error_hi, -np.inf)
            high = np.nextafter(high+error_hi, np.inf)
            peak = max(peak, block.nbytes+6*low.nbytes+(0 if present is None else present.nbytes))
            for k, selected in selected_masks.items():
                selected = selected[start:end]
                if np.any(selected): minimum[k] = min(minimum[k], float(np.min(low[selected])))
                if np.any(~selected): maximum[k] = max(maximum[k], float(np.max(high[~selected])))
        outside_upper = None
        if outside is not None:
            outside_exact = exact(outside)+sum((a*t for a, t in zip(amplitudes, tails)), Fraction(0))+error
            outside_upper = _enclose(outside_exact)[1]
        result = {}
        for k in selections:
            upper = maximum[k] if outside_upper is None else max(maximum[k], outside_upper)
            gap = math.nextafter(minimum[k]-upper, -math.inf)
            stable = gap > 0
            result[k] = dict(status='STABLE' if stable else 'UNKNOWN',
                reason='Independent coefficient intervals separate every competitor.' if stable else 'Independent coefficient intervals overlap.',
                minimum_gap_lower_bound=gap if math.isfinite(gap) else None)
        return result

    leaves = []
    def visit(box, depth):
        nonlocal attempts
        attempts += 1
        results = cell(box)
        if all(r['status'] == 'STABLE' for r in results.values()) or depth == 0:
            leaves.append(dict(box=[[str(a), str(b)] for a,b in box], cuts=results))
            return {k: result['status'] == 'STABLE' for k,result in results.items()}
        axis = max(range(2), key=lambda i: box[i][1]-box[i][0])
        low, high = box[axis]; midpoint = (low+high)/2
        outcomes = []
        for a,b in ((low,midpoint),(midpoint,high)):
            child=list(box);child[axis]=(a,b)
            outcomes.append(visit(tuple(child), depth-1))
        return {k: all(child[k] for child in outcomes) for k in selections}
    passed = visit(domain.box, int(subdivide_depth))
    cuts = {k: dict(status='STABLE' if ok else 'UNKNOWN', selected_ids=list(selections[k]),
        k=k, attempted_boxes=attempts, leaf_boxes=len(leaves),
        contract='Independent F/D intervals under the same response residual/tail contracts.') for k,ok in passed.items()}
    reserved=scratch_upper_bound(domain,len(y),chunk_rows,len(selections),len(rows))
    return dict(cuts=cuts, attempted_boxes=attempts, leaf_boxes=len(leaves), leaves=leaves,
        peak_workspace_bytes=peak, workspace_peak_bytes=max(peak,reserved), source_count=domain.r, atom_count=q,
        method='independent_coefficient_intervals')


def certify_interval(domain, y, candidate_rows, retained_values, retained_mask,
                     tail_bounds, selected_ids, **kwargs):
    k=len(selected_ids)
    result=certify_many(domain,y,candidate_rows,retained_values,retained_mask,
        tail_bounds,{k:selected_ids},**kwargs)
    return result['cuts'][k]
