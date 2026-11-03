"""Dot-first polynomial scans with sound outward fallback for unresolved cuts.

Every physical representation may use this backend. It accepts each dot
certificate independently and sends only UNKNOWN cutoffs to the original
per-atom outward implementation. Both attempts remain in real call cost;
reported numeric workspace is the maximum because their chunk frames do
not overlap. The original fast implementation is never monkeypatched.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np

from . import policy_domain_dot as dot
from . import policy_domain_fast as outward


def scratch_upper_bound(domain,n_nodes,chunk_rows=512,n_cuts=3,n_candidates=None):
    return max(dot.scratch_upper_bound(domain,n_nodes,chunk_rows,n_cuts,n_candidates),
               outward.scratch_upper_bound(domain,n_nodes,chunk_rows,n_cuts,n_candidates))


def _stored_float_inputs(y,values,columns):
    def okay(array):return isinstance(array,np.ndarray) and array.dtype==np.float64
    return okay(y) and ((columns is None and okay(values)) or
        columns is not None and all(okay(column) for column in columns))


def certify_many(domain,y,candidate_rows,retained_values,retained_mask,
                 tail_bounds,selected_sets,*,kappa,base_residual,atom_residuals,
                 round_pad=0,subdivide_depth=0,retained_columns=None,
                 chunk_rows=512,node_chunk_rows=None):
    """Same global-theta proof inputs; preserve the old outward coverage.

    Returns cuts, workspace_peak_bytes and shared_scan with separately timed
    dot/fallback stages. Fraction and non-array node inputs go directly to
    the original implementation, avoiding a delegation cycle.
    """
    selected_sets={key:tuple(ids) for key,ids in selected_sets.items()}
    if not isinstance(candidate_rows,np.ndarray):candidate_rows=tuple(candidate_rows)
    tail_bounds=tuple(tail_bounds);atom_residuals=tuple(atom_residuals)
    kwargs=dict(kappa=kappa,base_residual=base_residual,atom_residuals=atom_residuals,
        round_pad=round_pad,subdivide_depth=subdivide_depth,retained_columns=retained_columns,
        chunk_rows=chunk_rows,node_chunk_rows=node_chunk_rows)
    if not _stored_float_inputs(y,retained_values,retained_columns):
        started=perf_counter()
        result=outward.certify_many(domain,y,candidate_rows,retained_values,retained_mask,
            tail_bounds,selected_sets,**kwargs)
        seconds=perf_counter()-started
        for cut in result['cuts'].values():cut['scan_backend']='outward_nonfloat'
        return dict(cuts=result['cuts'],workspace_peak_bytes=result['workspace_peak_bytes'],
            peak_numeric_scratch_bytes=result['workspace_peak_bytes'],implementation='dot_then_outward',
            shared_scan=dict(dot=None,outward_fallback=result['shared_scan'],
                dot_seconds=0.,outward_seconds=seconds,fallback_cutoffs=list(selected_sets),
                dot_cut_statuses={},nonfloat_direct_outward=True,
                candidate_node_controls=result['shared_scan'].get('candidate_node_controls',0),
                control_boxes=result['shared_scan'].get('control_boxes',0)))
    started=perf_counter()
    first=dot.certify_many(domain,y,candidate_rows,retained_values,retained_mask,
        tail_bounds,selected_sets,**kwargs)
    dot_seconds=perf_counter()-started
    dot_workspace=first['workspace_peak_bytes'];dot_stats=first['shared_scan']
    statuses={key:cut['status'] for key,cut in first['cuts'].items()}
    fallback_sets={key:selected_sets[key] for key,status in statuses.items() if status=='UNKNOWN'}
    cuts={key:cut for key,cut in first['cuts'].items() if cut['status']=='STABLE'}
    for cut in cuts.values():cut['scan_backend']='dot'
    # Dot returns only compact proof metadata, not arrays. Drop all failed
    # proof frames before invoking outward; no double chunk/basis is held.
    del first
    fallback_stats=None;outward_seconds=0.;outward_workspace=0
    if fallback_sets:
        started=perf_counter()
        second=outward.certify_many(domain,y,candidate_rows,retained_values,retained_mask,
            tail_bounds,fallback_sets,**kwargs)
        outward_seconds=perf_counter()-started
        outward_workspace=second['workspace_peak_bytes'];fallback_stats=second['shared_scan']
        for key,cut in second['cuts'].items():
            cut['scan_backend']='outward_fallback';cut['dot_status']='UNKNOWN';cuts[key]=cut
        del second
    workspace=max(dot_workspace,outward_workspace)
    return dict(cuts={key:cuts[key] for key in selected_sets},workspace_peak_bytes=workspace,
        peak_numeric_scratch_bytes=workspace,implementation='dot_then_outward',
        shared_scan=dict(dot=dot_stats,outward_fallback=fallback_stats,
            dot_seconds=dot_seconds,outward_seconds=outward_seconds,
            fallback_cutoffs=list(fallback_sets),dot_cut_statuses=statuses,
            nonfloat_direct_outward=False,
            candidate_node_controls=sum(stats.get('candidate_node_controls',0)
                for stats in (dot_stats,fallback_stats) if stats is not None),
            control_boxes=sum(stats.get('control_boxes',0)
                for stats in (dot_stats,fallback_stats) if stats is not None)))


def certify_fast(domain,y,candidate_rows,retained_values,retained_mask,
                 tail_bounds,selected_ids,**kwargs):
    ids=tuple(selected_ids)
    result=certify_many(domain,y,candidate_rows,retained_values,retained_mask,
        tail_bounds,{len(ids):ids},**kwargs)
    return dict(result['cuts'][len(ids)],workspace_peak_bytes=result['workspace_peak_bytes'],
        peak_numeric_scratch_bytes=result['peak_numeric_scratch_bytes'],
        shared_scan=result['shared_scan'],implementation=result['implementation'])
