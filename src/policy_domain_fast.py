"""Chunked NumPy execution of the exact-polynomial domain certificate.

Small D/F polynomials still compile as exact Fractions in the original
kernel. Every node multiply/add/subtract is rounded outwards with nextafter;
no dot product or unbounded fused expression replaces that arithmetic.
Callers must establish the same nearest/gradual-underflow/no-DAZ environment
and residual/resolvent/tail contracts as policy_domain_polynomial.certify.
"""
from __future__ import annotations

import math
import numpy as np

from .policy_domain_polynomial import Interval, exact
from .policy_domain_controls import controls_fast

_NEG, _POS = -np.inf, np.inf
_CONTRACT = ('Conditional on supplied uniform residual/resolvent/rounding and '
             'omitted-coordinate tail upper bounds; no natural-graph proof or '
             'actual-view witness is synthesized.')


def scratch_upper_bound(domain, n_nodes, chunk_rows=512, n_cuts=3,
                        n_candidates=None):
    """Conservative peak numeric scratch reservation for float64 inputs.

    Input base/response/candidate arrays are owned and charged by the caller.
    This covers preparation temporaries plus chunk interval/product/reduction
    arrays; no full n-by-control or n-by-atom response copy is constructed.
    Python object overhead is outside this numeric estimate and remains part
    of process RSS. Fraction-valued node inputs additionally need converted
    interval arrays, reported by the actual call.
    """
    n, count = int(n_nodes), int(n_nodes if n_candidates is None else n_candidates)
    rows = min(int(chunk_rows), count)
    q, c = 2*domain.r, (domain.r+1)**2
    small = 8*c*(24+8*q+6*int(n_cuts))+1024
    prepare = n+count*(40+q)+16*int(n_cuts)*count
    chunks = 18*rows*c*8+6*rows*q*8+rows*(q+int(n_cuts)+32)
    return int(small+max(prepare, chunks))


def _array_intervals(values):
    if isinstance(values, np.ndarray) and values.dtype.kind == 'f':
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError('All stored scalars must be finite.')
        allocated = 0 if np.shares_memory(array, values) else array.nbytes
        return (array, array), allocated
    array = np.asarray(values, dtype=object)
    low = np.empty(array.shape, dtype=np.float64)
    high = np.empty_like(low)
    for index in np.ndindex(array.shape):
        interval = Interval.of(array[index])
        low[index], high[index] = interval.lo, interval.hi
    return (low, high), low.nbytes+high.nbytes


def _iv(value):
    interval = Interval.of(value)
    return interval.lo, interval.hi


def _mul(left, right):
    a, b = left; c, d = right
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        if a is b:
            p, q = np.multiply(a, c), np.multiply(a, d)
            low, high = np.minimum(p, q), np.maximum(p, q)
        elif c is d:
            p, q = np.multiply(a, c), np.multiply(b, c)
            low, high = np.minimum(p, q), np.maximum(p, q)
        else:
            low = np.minimum(np.minimum(np.multiply(a, c), np.multiply(a, d)),
                             np.minimum(np.multiply(b, c), np.multiply(b, d)))
            high = np.maximum(np.maximum(np.multiply(a, c), np.multiply(a, d)),
                              np.maximum(np.multiply(b, c), np.multiply(b, d)))
        if np.any(np.isnan(low)) or np.any(np.isnan(high)):
            raise ArithmeticError('Nonfinite interval product.')
        return np.nextafter(low, _NEG), np.nextafter(high, _POS)


def _add(left, right):
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        return (np.nextafter(np.add(left[0], right[0]), _NEG),
                np.nextafter(np.add(left[1], right[1]), _POS))


def _sub(left, right):
    return _add(left, (-right[1], -right[0]))


def _take(pair, indices):
    low = pair[0][indices]
    return low, low if pair[0] is pair[1] else pair[1][indices]


def _column(pair):
    low = pair[0][:, None]
    return low, low if pair[0] is pair[1] else pair[1][:, None]


def _prepare(domain, y, candidates, values, mask, tails, selected_sets,
             kappa, base_residual, atom_residuals, pad, retained_columns):
    if isinstance(candidates, np.ndarray) and candidates.dtype.kind in 'iu':
        rows = candidates
    else:
        rows = np.asarray(tuple(int(x) for x in candidates), dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError('Candidate IDs must be a vector.')
    y_pair, extra = _array_intervals(y)
    if y_pair[0].ndim != 1 or not len(y_pair[0]):
        raise ValueError('The full base vector must be nonempty.')
    n, q = len(y_pair[0]), 2*domain.r
    if np.any(rows < 0) or np.any(rows >= n) or len(np.unique(rows)) != len(rows):
        raise ValueError('Candidate IDs must be distinct nodes in the full base vector.')
    positions = {source: int(np.flatnonzero(rows == source)[0])
                 for source in domain.source_ids if np.any(rows == source)}
    if retained_columns is not None:
        if values is not None or len(retained_columns) != q:
            raise ValueError('Supply 2r global response columns and no retained matrix.')
        columns = []
        for value in retained_columns:
            pair, allocated = _array_intervals(value); extra += allocated
            if pair[0].shape != (n,):
                raise ValueError('Response columns must be indexed by full graph node ID.')
            columns.append(pair)
        value_pair = None
    else:
        value_pair, allocated = _array_intervals(values); extra += allocated
        if value_pair[0].shape != (len(rows), q):
            raise ValueError('Retained matrix must have (#candidate,2r) entries.')
        columns = None
    masks = None if mask is None else np.asarray(mask, dtype=bool)
    if masks is None and retained_columns is None:
        raise ValueError('Matrix input requires its retained-coordinate mask.')
    if masks is not None and masks.shape != (len(rows), q):
        raise ValueError('Retained mask must have (#candidate,2r) entries.')
    for i, source in enumerate(domain.source_ids):
        if source not in positions or exact(y[source]) != domain.y_sources[i]:
            raise ValueError('Stored source base coordinates differ from compiled y_S.')
        row = positions[source]
        expected = domain.H_m[i]+domain.H_w[i]
        actual = ([retained_columns[j][source] for j in range(q)] if columns is not None
                  else values[row] if isinstance(values, np.ndarray) else values[row])
        if ((masks is not None and not np.all(masks[row])) or
                tuple(exact(x) for x in actual) != expected):
            raise ValueError('Source atom coordinates must be retained and equal compiled H.')
    tails = tuple(exact(v) for v in tails)
    residuals = tuple(exact(v) for v in atom_residuals)
    rho, kap, pad = (exact(v) for v in (base_residual, kappa, pad))
    if len(tails) != q or len(residuals) != q or any(v < 0 for v in tails+residuals+(rho,pad)) or kap <= 0:
        raise ValueError('Require nonnegative 2r tails/residuals and positive kappa.')
    selections = {}
    for key, selected in selected_sets.items():
        selected = np.asarray(tuple(int(x) for x in selected), dtype=np.int64)
        if (not len(selected) or len(np.unique(selected)) != len(selected) or
                not np.all(np.isin(selected, rows)) or int(key) != len(selected)):
            raise ValueError('Each cutoff needs distinct selected candidate IDs of its stated size.')
        selections[key] = selected
    if not selections:
        raise ValueError('At least one requested cutoff is required.')
    if len(rows) == n:
        outside = None
    else:
        keep = np.ones(n, dtype=bool); keep[rows] = False
        outside = (float(np.max(y_pair[0], where=keep, initial=-np.inf)),
                   float(np.max(y_pair[1], where=keep, initial=-np.inf)))
    return dict(rows=rows, n=n, q=q, y=y_pair, values=value_pair, columns=columns,
        masks=masks, outside=outside, selections=selections, tails=tuple(_iv(v) for v in tails),
        residuals=tuple(_iv(v) for v in residuals), rho=_iv(rho), kap=_iv(kap), pad=_iv(pad),
        converted_numeric_bytes=extra)


def certify_many(domain, y, candidate_rows, retained_values, retained_mask,
                 tail_bounds, selected_sets, *, kappa, base_residual,
                 atom_residuals, round_pad=0, subdivide_depth=0,
                 retained_columns=None, chunk_rows=512, node_chunk_rows=None):
    """Certify several cutoffs, sharing controls, node scans and subdivision.

    Returns {cuts: {k: original-style result}, workspace_peak_bytes,
    shared_scan}. Dense can pass retained_values=retained_mask=None and
    retained_columns=[full_graph_response_j,...]; these vectors are borrowed,
    never copied into an n-by-q matrix. Sparse matrices align with candidate
    rows. Per-cut proofs and leaves remain independent.
    """
    if node_chunk_rows is not None:
        chunk_rows = node_chunk_rows
    if int(chunk_rows) != chunk_rows or chunk_rows < 1 or int(subdivide_depth) != subdivide_depth or subdivide_depth < 0:
        raise ValueError('Positive integer chunk rows and nonnegative subdivision depth required.')
    chunk_rows = int(chunk_rows)
    prepared = _prepare(domain, y, candidate_rows, retained_values, retained_mask,
        tail_bounds, selected_sets, kappa, base_residual, atom_residuals, round_pad, retained_columns)
    keys = tuple(prepared['selections'])
    state = {key: dict(attempts=0, leaves=[]) for key in keys}
    stats = dict(control_boxes=0, candidate_node_controls=0, node_chunks=0,
        cuts=len(keys), chunk_rows=chunk_rows)
    workspace = scratch_upper_bound(domain, prepared['n'], chunk_rows, len(keys), len(prepared['rows']))+prepared['converted_numeric_bytes']

    def one_box(box, active):
        stats['control_boxes'] += 1
        D, F = controls_fast(domain, box)
        if any(d <= 0 for d in D):
            return {key: dict(status='UNKNOWN', reason='D Bernstein controls are not all strictly positive.') for key in active}
        ds = np.array([_iv(d) for d in D], dtype=np.float64).T
        di = ds[0], ds[1]
        if np.any(di[0] <= 0):
            return {key: dict(status='UNKNOWN', reason='Positive D control underflowed its finite interval.') for key in active}
        coefficients = [tuple(np.array([_iv(x) for x in f], dtype=np.float64).T) for f in F]
        amplitudes = [tuple(np.array([_iv(abs(x)) for x in f], dtype=np.float64).T) for f in F]
        residual = _mul(prepared['rho'], di)
        for rho, amplitude in zip(prepared['residuals'], amplitudes):
            residual = _add(residual, _mul(rho, amplitude))
        error = _add(_mul(prepared['kap'], residual), _mul(prepared['pad'], di))
        if not np.all(np.isfinite(error[1])):
            return {key: dict(status='UNKNOWN', reason='Nonfinite outward error bound.') for key in active}
        twice_error = _mul((2.,2.), error)[1]
        count = len(D)
        selected_low = {key: np.full(count, np.inf) for key in active}
        competitor_high = {key: np.full(count, -np.inf) for key in active}
        for start in range(0, len(prepared['rows']), chunk_rows):
            stop = min(start+chunk_rows, len(prepared['rows']))
            nodes = prepared['rows'][start:stop]
            numerator = _mul(_column(_take(prepared['y'], nodes)), di)
            zero = np.zeros((len(nodes),count), dtype=np.float64)
            tail = zero, zero
            masks = None if prepared['masks'] is None else prepared['masks'][start:stop]
            for atom in range(prepared['q']):
                values = (_take(prepared['columns'][atom], nodes) if prepared['columns'] is not None
                          else _take(prepared['values'], (slice(start,stop),atom)))
                term = _mul(_column(values), coefficients[atom])
                updated = _sub(numerator, term)
                if masks is None:
                    numerator = updated
                else:
                    retained = masks[:,atom,None]
                    numerator = (np.where(retained, updated[0], numerator[0]),
                                 np.where(retained, updated[1], numerator[1]))
                    addition = _add(tail, _mul(prepared['tails'][atom], amplitudes[atom]))
                    tail = (np.where(retained,tail[0],addition[0]),
                            np.where(retained,tail[1],addition[1]))
            low = _sub(numerator,tail)[0]
            high = _add(numerator,tail)[1]
            for key in active:
                selected = np.isin(nodes,prepared['selections'][key],assume_unique=True)
                selected_low[key] = np.minimum(selected_low[key],np.min(low,axis=0,where=selected[:,None],initial=np.inf))
                competitor_high[key] = np.maximum(competitor_high[key],np.max(high,axis=0,where=~selected[:,None],initial=-np.inf))
            stats['candidate_node_controls'] += len(nodes)*count
            stats['node_chunks'] += 1
        outside = None
        if prepared['outside'] is not None:
            upper = _mul(prepared['outside'],di)
            for tail,amplitude in zip(prepared['tails'],amplitudes):
                upper = _add(upper,_mul(tail,amplitude))
            outside = upper[1]
        results = {}
        with np.errstate(over='ignore',invalid='ignore',under='ignore'):
            for key in active:
                candidate_right = np.nextafter(competitor_high[key]+twice_error,_POS)
                candidate_margin = np.nextafter(selected_low[key]-candidate_right,_NEG)
                outside_margin = (None if outside is None else
                    np.nextafter(selected_low[key]-np.nextafter(outside+twice_error,_POS),_NEG))
                minimum = np.inf
                for control in range(count):
                    if competitor_high[key][control] != -np.inf:
                        margin = candidate_margin[control];minimum=min(minimum,margin)
                        if not margin > 0:
                            results[key]=dict(status='UNKNOWN',reason='Candidate gap controls do not prove strict separation.',failed_control=control);break
                    if outside_margin is not None:
                        margin = outside_margin[control];minimum=min(minimum,margin)
                        if not margin > 0:
                            results[key]=dict(status='UNKNOWN',reason='Outside-candidate controls do not prove strict separation.',failed_control=control);break
                else:
                    dmin=min(D)
                    results[key]=dict(status='STABLE',reason='Every D control is positive and every required gap control has a strictly positive outward lower bound.',
                        determinant_control_lower_bound=Interval.of(dmin).lo,
                        determinant_control_lower_bound_exact=str(dmin),
                        minimum_gap_control_lower_bound=float(minimum) if math.isfinite(minimum) else None)
        return results

    def visit(box,depth,active):
        for key in active:state[key]['attempts']+=1
        try:
            results=one_box(box,active)
        except (OverflowError,ArithmeticError):
            results={key:dict(status='UNKNOWN',reason='Outward interval arithmetic became nonfinite.') for key in active}
        outcomes={};remaining=[]
        for key,result in results.items():
            if result['status']=='STABLE' or depth==0:
                state[key]['leaves'].append(dict(box=[[str(a),str(b)] for a,b in box],**result))
                outcomes[key]=result['status']=='STABLE'
            else:remaining.append(key)
        if remaining:
            axis=max(range(2),key=lambda i:box[i][1]-box[i][0]);lo,hi=box[axis];mid=(lo+hi)/2
            children=[]
            for a,b in ((lo,mid),(mid,hi)):
                child=list(box);child[axis]=(a,b);children.append(tuple(child))
            left=visit(children[0],depth-1,remaining);right=visit(children[1],depth-1,remaining)
            outcomes.update({key:left[key] and right[key] for key in remaining})
        return outcomes

    outcomes=visit(domain.box,int(subdivide_depth),keys)
    cuts={key:dict(status='STABLE' if outcomes[key] else 'UNKNOWN',
        selected_ids=prepared['selections'][key].tolist(),k=len(prepared['selections'][key]),
        source_count=domain.r,atom_count=prepared['q'],tensor_degree=[domain.r,domain.r],
        attempted_boxes=state[key]['attempts'],leaf_boxes=len(state[key]['leaves']),
        leaves=state[key]['leaves'],contract=_CONTRACT) for key in keys}
    return dict(cuts=cuts,workspace_peak_bytes=workspace,
        peak_numeric_scratch_bytes=workspace,shared_scan=stats)


def certify_fast(domain,y,candidate_rows,retained_values,retained_mask,tail_bounds,
                 selected_ids,**kwargs):
    """Original single-cut API, using the same shared scan implementation."""
    selected_ids=tuple(selected_ids);key=len(selected_ids)
    result=certify_many(domain,y,candidate_rows,retained_values,retained_mask,
        tail_bounds,{key:selected_ids},**kwargs)
    return dict(result['cuts'][key],workspace_peak_bytes=result['workspace_peak_bytes'],
        peak_numeric_scratch_bytes=result['peak_numeric_scratch_bytes'],shared_scan=result['shared_scan'])
