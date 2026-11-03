"""Batched BLAS dot execution of the global-theta Bernstein proof.

Stored binary64 node arrays use enclosed batched dot products. Exact small
D/F controls, denominator tests, scalar posterior and outside envelopes
are unchanged. Non-array/Fraction node inputs delegate to the current
nextafter-per-atom implementation. No graph solve or repository mutation.

As for the existing lifting path, the arithmetic contract requires binary64
dot products/basic operations, nearest rounding, gradual underflow and no
DAZ. Lower-precision or approximate BLAS kernels are outside that contract.
"""
from __future__ import annotations
from fractions import Fraction as Q
import math
import numpy as np

from . import policy_domain_fast as old
from .policy_domain_controls import controls_fast
from .policy_domain_polynomial import Interval

U=Q(1,2**53);TINY=Q(1,2**1022)


def _upper(value):
    value=value if isinstance(value,Q) else Q.from_float(float(value))
    rounded=float(value)
    if not math.isfinite(rounded):raise OverflowError('Bound does not fit binary64.')
    return math.nextafter(rounded,math.inf) if Q.from_float(rounded)<value else rounded


def _positive_product_upper(left,right,gamma,underflow):
    observed=left@right
    if not np.all(np.isfinite(observed)):raise ArithmeticError('Nonfinite positive dot.')
    denominator=math.nextafter(1-gamma,-math.inf)
    return np.nextafter(np.nextafter(observed+underflow,np.inf)/denominator,np.inf)


def _rounded_controls(D,F):
    d=np.asarray([float(value) for value in D])
    f=np.asarray([[float(value) for value in poly] for poly in F])
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(f)):
        raise ArithmeticError('Nonfinite rounded control.')
    dr=np.asarray([_upper(abs(value-Q.from_float(float(mid)))) for value,mid in zip(D,d)])
    fr=np.asarray([[_upper(abs(value-Q.from_float(float(mid)))) for value,mid in zip(poly,row)] for poly,row in zip(F,f)])
    absolute=np.asarray([[_upper(abs(value)) for value in poly] for poly in F])
    return d,f,dr,fr,absolute


def node_bounds(y,values,missing,rounded,gamma,underflow):
    """Enclose each exact retained numerator +/- omitted-coordinate tails.

    Node input values are exact stored binary64. D/F coefficient rounding
    uses exact Fraction conversion radii. This routine excludes the separate
    posterior reconstruction residual, applied uniformly by the caller.
    """
    d,f,dr,fr,absolute=rounded
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        scores=y[:,None]*d[None,:]-values@f
        if not np.all(np.isfinite(scores)):raise ArithmeticError('Nonfinite numerator dot.')
        abs_values=np.abs(values);abs_y=np.abs(y[:,None])
        magnitude=_positive_product_upper(abs_values,np.abs(f),gamma,underflow)
        magnitude=np.nextafter(np.nextafter(abs_y*np.abs(d),np.inf)+magnitude,np.inf)
        arithmetic=np.nextafter(np.nextafter(gamma*magnitude,np.inf)+underflow,np.inf)
        if np.any(fr) or np.any(dr):
            conversion=(_positive_product_upper(abs_values,fr,gamma,underflow) if np.any(fr)
                        else np.zeros(scores.shape))
            conversion=np.nextafter(conversion+np.nextafter(abs_y*dr,np.inf),np.inf)
            arithmetic=np.nextafter(arithmetic+conversion,np.inf)
        if missing is not None and np.any(missing):
            tail=_positive_product_upper(missing,absolute,gamma,underflow)
            arithmetic=np.nextafter(arithmetic+tail,np.inf)
        if not np.all(np.isfinite(arithmetic)):raise ArithmeticError('Nonfinite node error bound.')
        return np.nextafter(scores-arithmetic,-np.inf),np.nextafter(scores+arithmetic,np.inf)


def scratch_upper_bound(domain,n_nodes,chunk_rows=512,n_cuts=3,n_candidates=None):
    n=int(n_nodes);count=n if n_candidates is None else int(n_candidates)
    rows=min(int(chunk_rows),count);q=2*domain.r;c=(domain.r+1)**2
    small=8*c*(32+12*q+6*int(n_cuts))+2048
    preparation=n+count*(40+q)+16*int(n_cuts)*count
    chunks=20*rows*c*8+8*rows*q*8+rows*(q+int(n_cuts)+32)
    return int(small+max(preparation,chunks))


def _stored_float_inputs(y,values,columns):
    def okay(array):return isinstance(array,np.ndarray) and array.dtype==np.float64
    return okay(y) and ((columns is None and okay(values)) or
        columns is not None and all(okay(v) for v in columns))


def certify_many(domain,y,candidate_rows,retained_values,retained_mask,
                 tail_bounds,selected_sets,*,kappa,base_residual,atom_residuals,
                 round_pad=0,subdivide_depth=0,retained_columns=None,
                 chunk_rows=512,node_chunk_rows=None):
    kwargs=dict(kappa=kappa,base_residual=base_residual,atom_residuals=atom_residuals,
        round_pad=round_pad,subdivide_depth=subdivide_depth,retained_columns=retained_columns,
        chunk_rows=chunk_rows,node_chunk_rows=node_chunk_rows)
    if not _stored_float_inputs(y,retained_values,retained_columns):
        result=old.certify_many(domain,y,candidate_rows,retained_values,retained_mask,
            tail_bounds,selected_sets,**kwargs)
        result['implementation']='reference_delegate_nonfloat_node_inputs'
        return result
    if node_chunk_rows is not None:chunk_rows=node_chunk_rows
    if int(chunk_rows)!=chunk_rows or chunk_rows<1 or int(subdivide_depth)!=subdivide_depth or subdivide_depth<0:
        raise ValueError('Positive chunk rows and nonnegative integer depth required.')
    chunk_rows=int(chunk_rows)
    p=old._prepare(domain,y,candidate_rows,retained_values,retained_mask,tail_bounds,
        selected_sets,kappa,base_residual,atom_residuals,round_pad,retained_columns)
    keys=tuple(p['selections']);state={key:dict(attempts=0,leaves=[]) for key in keys}
    q=p['q'];operations=2*q+3;g=operations*U/(1-operations*U)
    gamma=_upper(g);underflow=_upper(operations*TINY/(1-g))
    stats=dict(control_boxes=0,candidate_node_controls=0,node_chunks=0,cuts=len(keys),
        chunk_rows=chunk_rows,gamma_operations=operations,gamma_upper=gamma,
        underflow_per_dot_upper=underflow)
    workspace=scratch_upper_bound(domain,p['n'],chunk_rows,len(keys),len(p['rows']))

    def one_box(box,active):
        stats['control_boxes']+=1
        D,F=controls_fast(domain,box)
        if any(d<=0 for d in D):
            return {key:dict(status='UNKNOWN',reason='D Bernstein controls are not all strictly positive.') for key in active}
        ds=np.asarray([old._iv(d) for d in D]).T;di=ds[0],ds[1]
        if np.any(di[0]<=0):
            return {key:dict(status='UNKNOWN',reason='Positive D control underflowed its finite interval.') for key in active}
        amplitudes=[tuple(np.asarray([old._iv(abs(x)) for x in f]).T) for f in F]
        residual=old._mul(p['rho'],di)
        for rho,amplitude in zip(p['residuals'],amplitudes):
            residual=old._add(residual,old._mul(rho,amplitude))
        error=old._add(old._mul(p['kap'],residual),old._mul(p['pad'],di))
        if not np.all(np.isfinite(error[1])):
            return {key:dict(status='UNKNOWN',reason='Nonfinite outward posterior error bound.') for key in active}
        twice_error=old._mul((2.,2.),error)[1]
        rounded=_rounded_controls(D,F);count=len(D)
        tails=np.asarray([item[1] for item in p['tails']])
        selected_low={key:np.full(count,np.inf) for key in active}
        competitor_high={key:np.full(count,-np.inf) for key in active}
        for start in range(0,len(p['rows']),chunk_rows):
            stop=min(start+chunk_rows,len(p['rows']));nodes=p['rows'][start:stop]
            values=(np.column_stack([pair[0][nodes] for pair in p['columns']]) if p['columns'] is not None
                    else p['values'][0][start:stop])
            masks=None if p['masks'] is None else p['masks'][start:stop]
            if masks is None:missing=None
            else:
                values=np.where(masks,values,0.)
                missing=np.where(masks,0.,tails)
            low,high=node_bounds(p['y'][0][nodes],values,missing,rounded,gamma,underflow)
            for key in active:
                selected=np.isin(nodes,p['selections'][key],assume_unique=True)
                selected_low[key]=np.minimum(selected_low[key],np.min(low,axis=0,where=selected[:,None],initial=np.inf))
                competitor_high[key]=np.maximum(competitor_high[key],np.max(high,axis=0,where=~selected[:,None],initial=-np.inf))
            stats['candidate_node_controls']+=len(nodes)*count;stats['node_chunks']+=1
        outside=None
        if p['outside'] is not None:
            upper=old._mul(p['outside'],di)
            for tail,amplitude in zip(p['tails'],amplitudes):upper=old._add(upper,old._mul(tail,amplitude))
            outside=upper[1]
        results={}
        with np.errstate(over='ignore',invalid='ignore',under='ignore'):
            for key in active:
                candidate=np.nextafter(selected_low[key]-np.nextafter(competitor_high[key]+twice_error,np.inf),-np.inf)
                other=(None if outside is None else
                    np.nextafter(selected_low[key]-np.nextafter(outside+twice_error,np.inf),-np.inf))
                minimum=np.inf
                for control in range(count):
                    if competitor_high[key][control]!=-np.inf:
                        margin=candidate[control];minimum=min(minimum,margin)
                        if not margin>0:
                            results[key]=dict(status='UNKNOWN',reason='Candidate dot-enclosed controls do not prove strict separation.',failed_control=control);break
                    if other is not None:
                        margin=other[control];minimum=min(minimum,margin)
                        if not margin>0:
                            results[key]=dict(status='UNKNOWN',reason='Outside-candidate controls do not prove strict separation.',failed_control=control);break
                else:
                    dmin=min(D)
                    results[key]=dict(status='STABLE',reason='Every D control and every dot-enclosed global-theta gap control is strictly positive.',
                        determinant_control_lower_bound=Interval.of(dmin).lo,
                        determinant_control_lower_bound_exact=str(dmin),
                        minimum_gap_control_lower_bound=float(minimum) if math.isfinite(minimum) else None)
        return results

    def visit(box,depth,active):
        for key in active:state[key]['attempts']+=1
        try:results=one_box(box,active)
        except (OverflowError,ArithmeticError):
            results={key:dict(status='UNKNOWN',reason='Dot enclosure arithmetic became nonfinite.') for key in active}
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
    cuts={key:dict(status='STABLE' if outcomes[key] else 'UNKNOWN',selected_ids=p['selections'][key].tolist(),
        k=len(p['selections'][key]),source_count=domain.r,atom_count=q,tensor_degree=[domain.r,domain.r],
        attempted_boxes=state[key]['attempts'],leaf_boxes=len(state[key]['leaves']),leaves=state[key]['leaves'],
        contract=old._CONTRACT) for key in keys}
    return dict(cuts=cuts,workspace_peak_bytes=workspace,peak_numeric_scratch_bytes=workspace,
        shared_scan=stats,implementation='dot_gamma_global_theta_bernstein')


def certify_fast(domain,y,candidate_rows,retained_values,retained_mask,tail_bounds,selected_ids,**kwargs):
    ids=tuple(selected_ids);result=certify_many(domain,y,candidate_rows,retained_values,retained_mask,
        tail_bounds,{len(ids):ids},**kwargs)
    return dict(result['cuts'][len(ids)],workspace_peak_bytes=result['workspace_peak_bytes'],
        shared_scan=result['shared_scan'],implementation=result['implementation'])
