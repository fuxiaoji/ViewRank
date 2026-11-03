"""A fixed, charged two-tier accuracy policy for continuous-domain queries.

The tiers are declared before execution, never selected by a measured time.
Only UNKNOWN cutoffs reach the next tier. Previously proven memberships and
actual instability witnesses are frozen. Cache refinement is explicit and
separate from capacity eviction. All preparation, certificates, witnesses,
invalidations and release work remain inside this operation's measured wall.
"""
from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np

from .policy_domain_execution import DEFAULT_BOX


@dataclass(frozen=True)
class AccuracyTier:
    keep: int
    atom_tolerance: float


DEFAULT_TIERS = (AccuracyTier(128, 1e-3), AccuracyTier(512, 1e-6))


def _snapshot(directory):
    compiler = directory.compiler
    result = {name: getattr(directory, name) for name in
        ('hits', 'misses', 'evictions', 'refinement_invalidations', 'preview_selections')}
    result.update({name: getattr(compiler, name) for name in
        ('base_solves', 'atom_solves', 'point_solves', 'exact_source_compilations', 'point_source_compilations')})
    solver = compiler.ranker.base_lu
    result['backend_calls'] = int(getattr(solver, 'calls', 0))
    result['backend_native_calls'] = int(getattr(solver, 'native_calls', 0))
    return result


def _delta(after, before):
    return {key: after[key]-before[key] for key in before}


class AdaptiveDomainOperator:
    """Fixed coarse-then-fine controller over one independent directory.

    A supplied base is reused across tiers. Otherwise one base is prepared
    at the compiler's entry tolerance before changing atom tiers, and that
    stored base is reused. Each actual prepare still independently guards it.
    Default witnesses occur only in the final required tier, after the domain
    test. An already-fine initial cache makes the first tier final without
    repeating the same proof under a different requested label.
    """
    def __init__(self, directory, tiers=DEFAULT_TIERS, preview_keep=128,
                 witness_policy='final'):
        self.directory = directory
        self.tiers = tuple(t if isinstance(t, AccuracyTier) else AccuracyTier(int(t[0]), float(t[1])) for t in tiers)
        if not 1 <= len(self.tiers) <= 2 or any(t.keep < 0 or int(t.keep) != t.keep or
                not math.isfinite(t.atom_tolerance) or not t.atom_tolerance > 0 for t in self.tiers):
            raise ValueError('Use one or two declared positive-tolerance tiers.')
        if any(b.keep < a.keep or b.atom_tolerance > a.atom_tolerance for a, b in zip(self.tiers, self.tiers[1:])):
            raise ValueError('A later tier must retain at least as much and request no coarser tolerance.')
        if preview_keep < 0 or witness_policy not in ('final', 'none'):
            raise ValueError('Preview is nonnegative; witness policy is final or none.')
        self.preview_keep, self.witness_policy = int(preview_keep), witness_policy

    @property
    def protocol(self):
        return dict(tiers=[dict(keep=t.keep, atom_tolerance=t.atom_tolerance) for t in self.tiers],
            preview_keep=self.preview_keep, witness_policy=self.witness_policy,
            refine_only='UNKNOWN', witness_order='after',
            resolved_cut_policy='Freeze STABLE/verified UNSTABLE; never recheck or choose faster outcomes.',
            cache_policy='Reuse sufficiently fine/larger atoms, explicitly invalidate only inadequate needed keys after close.')

    def _ready(self, keys, tier):
        directory = self.directory
        return all(key in directory.cache and directory.compatible(directory.cache[key], tier.keep, tier.atom_tolerance) for key in keys)

    def execute(self, seed, sources, base_scores=None, base_rho=None, ks=(2, 5, 10),
                box=DEFAULT_BOX, depth=2, certifier='poly', chunk_rows=512,
                early_witness=True, audit_sink=None):
        directory, compiler = self.directory, self.directory.compiler
        if directory.live or directory.pins or directory._pending_bytes:
            raise RuntimeError('Adaptive execution owns an unpinned, inactive directory.')
        ks = tuple(dict.fromkeys(map(int, ks)))
        if not ks or min(ks) < 1 or max(ks) > compiler.n:
            raise ValueError('Requested cuts must exist in the full graph.')
        sources = np.asarray(sources, dtype=np.int64)
        keys = tuple((group, int(source)) for group in range(2) for source in sources)
        old = (directory.keep, directory.preview_keep, compiler.atom_tolerance)
        before = _snapshot(directory)
        started = perf_counter()
        prepare_seconds = release_seconds = audit_seconds = 0.
        tier_records, cuts = [], {}
        remaining = list(ks)
        shared_base = base_scores
        skip_reason = None
        try:
            if shared_base is None:
                t = perf_counter()
                shared_base, computed_rho, _ = compiler.base(np.asarray(seed, dtype=np.float64))
                prepare_seconds += perf_counter()-t
                if base_rho is None: base_rho = computed_rho
            pre_tier_counters = _delta(_snapshot(directory), before)
            directory.preview_keep = min(self.preview_keep, compiler.n)
            for tier_index, tier in enumerate(self.tiers):
                if not remaining: break
                tier_started = perf_counter()
                tier_before = _snapshot(directory)
                invalidated = ()
                inadequate = [key for key in keys if key in directory.cache and
                    not directory.compatible(directory.cache[key], tier.keep, tier.atom_tolerance)]
                if inadequate:
                    # No live family exists here. Unrelated atoms and their
                    # LRU order are retained; these are not capacity evictions.
                    invalidated = directory.invalidate_for_refinement(inadequate)
                directory.keep = min(tier.keep, compiler.n)
                compiler.atom_tolerance = tier.atom_tolerance
                final_tier = tier_index == len(self.tiers)-1
                ready_for_final = self._ready(keys, self.tiers[-1])
                final_attempt = final_tier or ready_for_final
                t = perf_counter()
                family = directory.prepare(seed, sources, base_scores=shared_base,
                    base_rho=base_rho, ks=tuple(remaining))
                prepared_seconds = perf_counter()-t
                prepare_seconds += prepared_seconds
                try:
                    # Newly generated zero descriptors may complete a fine
                    # cache without a solve; no duplicate tier can improve it.
                    ready_for_final = self._ready(keys, self.tiers[-1])
                    final_attempt = final_tier or ready_for_final
                    before_cuts = list(remaining)
                    t = perf_counter()
                    answer = family.certify(box=box, ks=tuple(remaining), depth=depth,
                        certifier=certifier, chunk_rows=chunk_rows,
                        early_witness=bool(early_witness and self.witness_policy == 'final' and final_attempt),
                        witness_order='after')
                    certified_seconds = perf_counter()-t
                    accounting = family.accounting
                    atom_tags = [dict(key=atom.key, generation_tol=atom.generation_tol,
                        keep=atom.keep, rows=len(atom.rows), zero=atom.zero, nbytes=atom.nbytes) for atom in family.atoms]
                    contract_metadata = dict(original_sources=family.sources.tolist(),
                        mention_masses=family.mu, weak_masses=family.nu,
                        base_residual=family.rho, atom_residuals=family.atom_rhos,
                        atom_tails=family.tails.tolist(), kappa=compiler.guard.kappa_old)
                    for k in before_cuts:
                        value = answer['cuts'][k]
                        if value['status'] not in ('STABLE', 'UNSTABLE', 'UNKNOWN'):
                            raise RuntimeError('Invalid domain outcome.')
                        cuts[k] = value
                    remaining = [k for k in before_cuts if cuts[k]['status'] == 'UNKNOWN']
                    audit_metadata = None
                    tier_audit_seconds = 0.
                    if audit_sink is not None:
                        t = perf_counter()
                        audit_metadata = audit_sink(family, tier_index)
                        tier_audit_seconds = perf_counter()-t
                        audit_seconds += tier_audit_seconds
                        # Audit files may preserve full tier evidence, but
                        # returned metadata must not retain the old basis.
                        def check_metadata(value):
                            if isinstance(value, np.ndarray):
                                raise ValueError('audit_sink must return file/hash metadata, not retained numeric arrays.')
                            if isinstance(value, dict):
                                for item in value.values(): check_metadata(item)
                            elif isinstance(value, (list, tuple)):
                                for item in value: check_metadata(item)
                        check_metadata(audit_metadata)
                finally:
                    t = perf_counter(); family.close(); released_seconds = perf_counter()-t
                    release_seconds += released_seconds
                tier_records.append(dict(tier=tier_index, requested_keep=tier.keep,
                    effective_keep=min(tier.keep, compiler.n), atom_tolerance=tier.atom_tolerance,
                    requested_cuts=before_cuts, remaining_unknown=list(remaining),
                    frozen_cuts=[k for k in ks if k in cuts and cuts[k]['status'] != 'UNKNOWN'],
                    invalidated_keys=invalidated, ready_for_final=ready_for_final,
                    prepare_seconds=prepared_seconds, certify_seconds=certified_seconds,
                    release_seconds=released_seconds, tier_wall_seconds=perf_counter()-tier_started-tier_audit_seconds,
                    audit_seconds=tier_audit_seconds, audit=audit_metadata,
                    counters=_delta(_snapshot(directory), tier_before),
                    memory_after_release=directory.memory,
                    accounting=accounting, atom_tags=atom_tags, contract=contract_metadata, answer=answer))
                if remaining and ready_for_final and not final_tier:
                    skip_reason = 'Needed atoms already satisfy the final tier; repeating the same retained proof cannot improve it.'
                    break
        finally:
            directory.keep, directory.preview_keep, compiler.atom_tolerance = old
        raw_elapsed = perf_counter()-started
        elapsed = raw_elapsed-audit_seconds
        stage_totals = dict(prepare_seconds=prepare_seconds,
            release_seconds=release_seconds,
            certify_seconds=max(0., elapsed-prepare_seconds-release_seconds))
        return dict(cuts={k: cuts[k] for k in ks}, tiers=tier_records,
            stage_totals=stage_totals, operator_seconds=elapsed,
            raw_operator_wall_seconds=raw_elapsed, audit_seconds=audit_seconds,
            counters=_delta(_snapshot(directory), before),
            pre_tier_counters=pre_tier_counters,
            point_checks=sum(t['answer'].get('point_checks',0) for t in tier_records),
            verified_witnesses=sum(t['answer'].get('verified_witnesses',0) for t in tier_records),
            requested_mode=directory.mode, actual_route='adaptive_'+directory.mode,
            refinement_skipped_reason=skip_reason, protocol=self.protocol,
            memory=directory.memory, original_sources=sources.tolist(),
            requested_cuts=list(ks),
            accounting_scope='Whole execute wall includes every tier, decision, invalidation, actual solve, proof, point check and release. Producer outer wall is authoritative; do not choose the faster tier post hoc.')


AdaptiveController = AdaptiveDomainOperator
