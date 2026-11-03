"""Adaptive actual-corner certificates for one changed source column.

With one source and a fixed raw background, every exact score gap multiplied
by the positive source denominator/determinant is affine in (m,w). Strict
agreement at all four corners therefore proves whole-rectangle membership.
Two strictly certified different actual memberships prove instability. This
baseline builds no response atoms and requires no caller base vector.
"""
from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from scripts.natural_domain_adapter import exact


def _upper(value):
    value=exact(value);rounded=float(value)
    return math.nextafter(rounded,math.inf) if exact(rounded)<value else rounded


def _lower(value):
    value=exact(value);rounded=float(value)
    return math.nextafter(rounded,-math.inf) if exact(rounded)>value else rounded


def _compact(result,k,weights):
    """Bounded corner summary; never retain an n-vector or full solver state."""
    point=dict(weights=tuple(map(float,weights)),weights_exact=tuple(str(v) for v in weights),
        sets={k:np.asarray(result['sets'][k],dtype=np.int64).copy()},
        selected_scores={k:np.asarray(result['selected_scores'][k],dtype=np.float64).copy()},
        strict_flags={k:True},score_error_bound=_upper(result['score_error_bound']))
    if 'certified_gap_lower_bounds' in result:
        gap=result['certified_gap_lower_bounds'][k]
        point['certified_gap_lower_bounds']={k:None if gap is None else _lower(gap)}
    return point


def certify_rank1_native(compiler,directory,seed,sources,box,ks,
                         ladder=(1e-6,1e-8,1e-10,1e-12)):
    """Four actual corners with per-corner adaptive numerical precision.

    The directory must have no live family/pins. Corner summaries are admitted
    against its private budget, possibly evicting ordinary unpinned LRU atoms.
    Those real evictions remain in the subsequent method cache trajectory.
    The temporary live/pending charge and compiler point tolerance are always
    restored, including on exceptions. Point solving retains its usual
    current-view/solver scratch scope, with no free retained response basis.
    """
    started=perf_counter()
    sources=np.asarray(sources,dtype=np.int64)
    ks=tuple(dict.fromkeys(int(k) for k in ks))
    if sources.shape!=(1,) or sources[0]<0 or sources[0]>=compiler.n:
        raise ValueError('The native affine-corner baseline requires one valid source.')
    if not ks or min(ks)<1 or max(ks)>compiler.n:
        raise ValueError('Requested cuts must exist in the graph.')
    if directory.compiler is not compiler:
        raise ValueError('Use the same compiler for point solves and the directory.')
    if (directory.live or directory.pins or directory._pending_bytes or
            directory._live_bytes or directory._live_reserved_bytes):
        raise RuntimeError('Native corner stage requires an idle directory.')
    box=tuple(tuple(exact(v) for v in pair) for pair in box)
    if len(box)!=2 or any(len(pair)!=2 or not 0<pair[0]<pair[1] for pair in box):
        raise ValueError('Positive nonempty two-weight rectangle required.')
    ladder=tuple(float(value) for value in ladder)
    if not ladder or any(not math.isfinite(v) or v<=0 for v in ladder):
        raise ValueError('A nonempty positive finite tolerance ladder is required.')
    if any(a<=b for a,b in zip(ladder,ladder[1:])):
        raise ValueError('Tolerance ladder must become strictly finer.')
    (ml,mh),(wl,wh)=box
    corners=((ml,wl),(ml,wh),(mh,wl),(mh,wh))
    # Covers four compact set/score summaries, flags/bounds and tier counters.
    # For the requested (2,5,10) and four tiers this is exactly 4096 bytes.
    reservation=max(4096,1024+4*(16*sum(ks)+80*len(ks)+128)+4*len(ladder)*32)
    before_memory=dict(directory.memory)
    before_evictions=directory.evictions
    cuts={};proofs={k:[] for k in ks};tiers=[]
    point_checks=0;point_seconds=0.;completed_corners=0
    tier_counts={value:0 for value in ladder}
    budget_status='admitted'
    charged=False
    original_tolerance=compiler.point_tolerance
    try:
        if directory.plan_bytes+reservation>directory.budget_bytes or not directory._evict(reservation):
            budget_status='insufficient_corner_summary_budget'
            cuts={k:dict(status='UNKNOWN',k=k,reason='Native corner summaries do not fit the declared private budget.') for k in ks}
        else:
            directory._pending_bytes+=reservation
            directory._live_bytes+=reservation
            charged=True;directory._track()
            for corner_index,weights in enumerate(corners):
                active=tuple(k for k in ks if k not in cuts)
                if not active:break
                corner_proofs={}
                for tolerance in ladder:
                    unresolved=tuple(k for k in active if k not in corner_proofs)
                    if not unresolved:break
                    compiler.point_tolerance=tolerance
                    tick=perf_counter()
                    result=compiler.point(seed,sources,weights,ks=unresolved)
                    elapsed=perf_counter()-tick
                    point_checks+=1;point_seconds+=elapsed;tier_counts[tolerance]+=1
                    strict=[]
                    for k in unresolved:
                        if not result['strict_flags'][k]:continue
                        proof=_compact(result,k,weights)
                        corner_proofs[k]=proof;strict.append(k)
                        previous=proofs[k]
                        if previous and set(proof['sets'][k])!=set(previous[0]['sets'][k]):
                            cuts[k]=dict(status='UNSTABLE',k=k,
                                witness=dict(first=previous[0],second=proof),
                                reason='Two strictly certified actual corner views have different top-k memberships.')
                    tiers.append(dict(corner_index=corner_index,tolerance=tolerance,
                        requested_ks=unresolved,strict_ks=tuple(strict),seconds=elapsed))
                    del result
                completed_corners+=1
                for k,proof in corner_proofs.items():proofs[k].append(proof)
            for k in ks:
                if k in cuts:continue
                if len(proofs[k])==4:
                    cuts[k]=dict(status='STABLE',k=k,selected_ids=proofs[k][0]['sets'][k].tolist(),
                        corner_proofs=proofs[k],
                        reason='Four strictly certified equal actual corner memberships; the positive-denominator rank-one cleared score gaps are affine throughout the box.')
                else:
                    cuts[k]=dict(status='UNKNOWN',k=k,strict_corners=len(proofs[k]),
                        reason='At least one corner remained numerically unresolved; no domain claim is inferred.')
    finally:
        compiler.point_tolerance=original_tolerance
        if charged:
            directory._live_bytes-=reservation
            directory._pending_bytes-=reservation
            directory._track()
    return dict(cuts=cuts,actual_route='native_rank1',certifier='actual_corner_affine_rank1',
        point_checks=point_checks,verified_witnesses=sum(c['status']=='UNSTABLE' for c in cuts.values()),
        corners_visited=completed_corners,native_tiers=tiers,native_tier_counts=tier_counts,
        native_stages=dict(actual_point_seconds=point_seconds,summary_and_admission_seconds=perf_counter()-started-point_seconds),
        native_budget_status=budget_status,corner_summary_reservation_bytes=reservation,
        native_cache_evictions=directory.evictions-before_evictions,
        private_memory_before=before_memory,private_memory_after=dict(directory.memory),
        operator_seconds=perf_counter()-started,
        proof_scope='One original changed source, exact mathematical source normalization and fixed raw background under the compiler uniform contraction/residual contract. Four strictly equal corner sets prove the box; samples are not used for multi-source certification.',
        memory_scope='Admitted compact four-corner summaries only. No atom/base preparation; current actual-view solver/output scratch is separate, with no retained full response columns.')
