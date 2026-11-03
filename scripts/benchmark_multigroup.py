"""Paired query-level measurements for the fixed two-group policy family.

The default plan path reads only public package metadata and uses no solver.
Run freezes its exact source/input/job manifest before numerical work. Paired
TOTAL is active query work plus identically charged shared startup and own
setup; it is neither process wall time nor service batch latency. Process RSS
in paired execution describes the combined cohort only.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KS = (2, 5, 10)
GRID = tuple((m, w) for m in (1., 2., 3., 4.) for w in (.25, .5, 1., 2.))
STAGES = ('seed_input_seconds', 'base_seconds', 'group_plan_seconds',
          'family_prepare_seconds', 'target_seconds', 'release_seconds')


@dataclass(frozen=True)
class Spec:
    name: str
    mode: str
    atom_tol: float = 1e-3
    output: str = 'sparse'
    keep: int = 128

    @property
    def exact_stencil(self):
        return self.mode == 'sparse'


SPECS = (
    Spec('sparse_a1e3', 'sparse'), Spec('sparse_a1e4', 'sparse', 1e-4),
    Spec('dense_a1e3', 'dense'), Spec('dense_a1e4', 'dense', 1e-4),
    Spec('osp_gap', 'osp_gap'), Spec('native_gap', 'native_gap'),
    Spec('offset_gap', 'offset_gap'), Spec('no_core_a1e3', 'no_core'),
    Spec('scan_a1e3', 'sparse', output='scan'),
    Spec('fused_a1e3', 'sparse', output='fused'),
    Spec('core_fused_a1e3', 'sparse', output='fused', keep=0))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def native_json(value):
    if hasattr(value, 'tolist'):
        return native_json(value.tolist())
    if isinstance(value, dict):
        return {str(k): native_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native_json(v) for v in value]
    if isinstance(value, float) and not __import__('math').isfinite(value):
        return 'Infinity' if value > 0 else '-Infinity' if value < 0 else 'NaN'
    return value


def write_json(path, value):
    path.write_text(json.dumps(native_json(value), indent=2, allow_nan=False), encoding='utf-8')


def emit(handle, value):
    handle.write(json.dumps(native_json(value), allow_nan=False)+'\n')


def selected_specs(args):
    if args.execution == 'single':
        if not args.single_specname:
            raise ValueError('--execution single requires --single-specname.')
        return tuple(s for s in SPECS if s.name == args.single_specname)
    if args.single_specname:
        raise ValueError('--single-specname is only valid with --execution single.')
    return SPECS


def method_order(specs, qi, block):
    """Deterministic cyclic order, reversed on alternating qid/block parity."""
    order = list(specs)
    if (qi+block) % 2:
        order.reverse()
    shift = (qi+block) % len(order)
    return order[shift:]+order[:shift]


def public_prefix(args):
    folder = Path(args.input_root).resolve()/args.dataset
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    queries = []
    with (folder/'queries.jsonl').open(encoding='utf-8') as handle:
        for _, line in zip(range(args.queries), handle):
            q = json.loads(line)
            # Retain only execution inputs; no question text/annotation is needed.
            queries.append({k: q[k] for k in ('qid', 'sources', 'seed_indices', 'seed_values')}
                | {'original_question_index': q.get('original_question_index', len(queries))})
    if len(queries) != args.queries or len({q['qid'] for q in queries}) != len(queries):
        raise ValueError('The complete original prefix must exist with unique qids.')
    for q in queries:
        if len(set(q['sources'])) != len(q['sources']):
            raise ValueError('Original source arrays must not contain duplicates.')
        if len(q['seed_indices']) != len(q['seed_values']):
            raise ValueError('Seed coordinates and values must match.')
    return folder, manifest, queries


def source_paths():
    paths = [Path(__file__), ROOT/'scripts/process_memory.py',
             ROOT/'scripts/pin_timing_worker.py',
             ROOT/'baselines/build/Release/prpack_native.dll',
             ROOT/'baselines/CMakeLists.txt',
             ROOT/'baselines/prpack/prpack_solver_sccgs.cpp']
    paths += sorted((ROOT/'src').glob('*.py'))
    paths += sorted((ROOT/'src/baselines').glob('*.py'))
    return sorted(set(p for p in paths if p.is_file()))


def protocol(args, manifest, queries, specs):
    return dict(protocol='two-group-query-paired-v1', dataset=args.dataset,
        execution=args.execution, blocks=args.blocks, questions=len(queries),
        original_qids=[q['qid'] for q in queries],
        original_question_indices=[q['original_question_index'] for q in queries],
        original_sources=[q['sources'] for q in queries],
        seed_source_sha256=[hashlib.sha256(json.dumps(q, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest() for q in queries],
        alpha=float(manifest['alpha']), ks=KS, policy_grid=GRID,
        specs=[asdict(s) for s in specs], budget_bytes=int(args.budget_mb*1024**2),
        base_mode=args.base_mode, base_tolerances=[1e-6, 1e-8, 1e-10, 1e-12],
        fine_base_tolerance=1e-12, completion_tolerance=args.completion_tol,
        warmup=dict(rhs='(1-alpha)*e0', tolerance=1e-12,
            charge='Entire graph.solve_linear wall charged identically to every logical method.'),
        method_order='Reverse on odd(qi+block), then rotate left by (qi+block)%method_count.',
        cpu=args.cpu, process_priority=args.priority, threadpool_limit=1,
        cost='Each method independently creates its seed, base y, GroupPlan, live family and all sixteen point answers. TOTAL=sum(active query stage intervals)+charged shared startup+own planning/directory setup. Streamed I/O and record serialization are separate.',
        memory='Numeric atom/cache/stencil/core/candidate/16-summary reservations are per method. Shared input/backend and current seed/base/output scratch are separate. Paired OS memory is COMBINED cohort only; single mode is a fresh-process method observation.',
        scope='Original first requested prefix. Sixteen real policy parameter requests; zero directions retain all logical requests. No production-workload or complete-corpus performance claim.',
        no_gold_access=True)


def plan(args):
    _, manifest, queries = public_prefix(args)
    result = protocol(args, manifest, queries, selected_specs(args))
    result.update(phase='plan', native_experiment_started=False,
        nodes=manifest.get('nodes'), transition_nnz=manifest.get('transition_nnz'))
    print(json.dumps(result, indent=2))
    return result


def freeze(args, folder, manifest, queries, specs, output):
    """Immutable exact pilot scope is saved before importing/creating a solver."""
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Use an empty output directory; frozen or partial runs are never overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in source_paths()}
    input_hashes = {name: digest(folder/name) for name in (
        'manifest.json', 'queries.jsonl', 'graph.npz', 'mention_subdistribution.npz')}
    snapshot = output/'source_snapshot'
    for name in source_hashes:
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
        if digest(target) != source_hashes[name]:
            raise RuntimeError('A source changed while taking the pre-run snapshot.')
    job = protocol(args, manifest, queries, specs)
    job.update(arguments=vars(args), source_sha256=source_hashes,
        input_root=str(folder), input_sha256=input_hashes,
        jobs=[dict(block=b, spec=s.name) for b in range(args.blocks) for s in specs],
        snapshot_complete_before_numerical_execution=True)
    write_json(output/'job_manifest.json', job)
    return job


class GraphMeter:
    """One shared transparent graph proxy; no scores or RHS values are cached."""
    def __init__(self, graph):
        self.graph, self.counts = graph, Counter()

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def _solve(self, kind, function, *args):
        started = perf_counter()
        scores, stats = function(*args)
        self.counts[kind+'_calls'] += 1
        self.counts[kind+'_seconds'] += perf_counter()-started
        for key in ('preprocess_seconds', 'edge_map_seconds', 'compute_seconds', 'edges_touched'):
            self.counts[kind+'_'+key] += stats.get(key, 0)
        return scores, stats

    def solve(self, *args):
        return self._solve('positive', self.graph.solve, *args)

    def solve_linear(self, *args):
        return self._solve('linear', self.graph.solve_linear, *args)

    def update_columns(self, sources, deltas):
        started = perf_counter()
        result = self.graph.update_columns(sources, deltas)
        kind = 'weight_update' if len(sources) else 'weight_restore'
        self.counts[kind+'_calls'] += 1
        self.counts[kind+'_seconds'] += perf_counter()-started
        return result

    def snapshot(self):
        return self.counts.copy()


def difference(after, before):
    return {key: after.get(key, 0)-before.get(key, 0) for key in after.keys() | before.keys()}


def set_priority(label):
    import ctypes as ct
    from ctypes import wintypes
    code = {'normal': 0x20, 'above-normal': 0x8000}[label]
    api = ct.WinDLL('kernel32').SetPriorityClass
    api.argtypes, api.restype = [wintypes.HANDLE, wintypes.DWORD], wintypes.BOOL
    if not api(wintypes.HANDLE(-1), code):
        raise ct.WinError()
    return code


def run(args):
    producer_started = perf_counter()
    folder, manifest, queries = public_prefix(args)
    specs, output = selected_specs(args), Path(args.output).resolve()
    frozen = freeze(args, folder, manifest, queries, specs, output)
    # Numerical imports and native constructors occur only after the freeze.
    import numpy as np
    from scipy import sparse
    from threadpoolctl import threadpool_limits
    from process_memory import process_memory
    from pin_timing_worker import pin_timing_worker
    from src.baselines.prpack import PRPackViewRank
    from src.baselines.certified_base import certified_base
    from src.budgeted_family import interaction_stencil
    from src.multigroup_policy_family import (MultiGroupDirectory, NativeGroupFamily,
        TwoGroupCompiler, POLICY_GRID)
    from src.policy_execution_v7 import bind_common_selection, top_indices
    from src.viewrank import _sparse_bytes
    if GRID != POLICY_GRID:
        raise RuntimeError('The frozen logical grid differs from the executor grid.')
    pin_timing_worker(args.cpu)
    priority = set_priority(args.priority)
    bind_common_selection()
    budget, alpha = frozen['budget_bytes'], float(manifest['alpha'])
    block_summaries = []
    with threadpool_limits(limits=1), ExitStack() as stack:
        streams = {}
        for spec in specs:
            directory = output/spec.name
            directory.mkdir()
            streams[spec.name] = dict(
                prediction=stack.enter_context((directory/'predictions.jsonl').open('w', encoding='utf-8')),
                levels=stack.enter_context((directory/'base_levels.jsonl').open('w', encoding='utf-8')),
                measurement=stack.enter_context((directory/'measurements.csv').open('w', newline='', encoding='utf-8')),
                writer=None)
        for block in range(args.blocks):
            P = sparse.load_npz(folder/'graph.npz').tocsc()
            mention = sparse.load_npz(folder/'mention_subdistribution.npz').tocsc()
            started = perf_counter()
            ranker = PRPackViewRank(P, alpha=alpha, tol=1e-12)
            constructor_seconds = perf_counter()-started
            del P
            started = perf_counter()
            compiler = TwoGroupCompiler(ranker, mention)
            compiler_seconds = perf_counter()-started
            del mention
            started = perf_counter()
            warm_rhs = np.zeros(ranker.n)
            warm_rhs[0] = 1-alpha
            warm_scores, warm_stats = ranker.base_lu.graph.solve_linear(warm_rhs, alpha, 1e-12)
            warmup_seconds = perf_counter()-started
            del warm_rhs, warm_scores
            started = perf_counter()
            meter = GraphMeter(ranker.base_lu.graph)
            ranker.base_lu.graph = meter
            meter_setup_seconds = perf_counter()-started
            common_startup = constructor_seconds+compiler_seconds+warmup_seconds+meter_setup_seconds
            backend_memory = meter.memory
            shared_inputs = compiler.shared_input_bytes
            backend_owned = _sparse_bytes(ranker.A)+sum(v for k, v in backend_memory.items() if k.endswith('_bytes'))
            states = {}
            for spec in specs:
                started = perf_counter()
                stencil = interaction_stencil([q['sources'] for q in queries]) if spec.exact_stencil else {}
                planning = perf_counter()-started
                started = perf_counter()
                directory = (MultiGroupDirectory(compiler, stencil, keep=spec.keep,
                    budget_bytes=budget, mode=spec.mode) if spec.mode in ('sparse', 'dense', 'no_core') else None)
                directory_setup = perf_counter()-started
                del stencil
                states[spec.name] = dict(directory=directory, planning_seconds=planning,
                    directory_setup_seconds=directory_setup, totals=Counter(), counts=Counter(),
                    active_seconds=[], acceptance=Counter(), routes=Counter(),
                    peak_numeric_bytes=directory.nbytes if directory is not None else 0,
                    peak_reserved_bytes=directory.reserved_bytes if directory is not None else 0,
                    compiled_r=Counter(), compiled_q=Counter(), order_positions=Counter())
            block_started = perf_counter()
            try:
                for qi, query in enumerate(queries):
                    for position, spec in enumerate(method_order(specs, qi, block)):
                        state, stream = states[spec.name], streams[spec.name]
                        directory = state['directory']
                        if meter.previous_sources:
                            raise RuntimeError('A preceding method did not restore the shared base weights.')
                        before_graph = meter.snapshot()
                        before_base = (ranker.base_lu.calls, ranker.base_lu.native_calls)
                        before_direction = compiler.direction_materializations
                        before_cache = (directory.hits, directory.misses, directory.evictions) if directory is not None else (0, 0, 0)
                        stages = {}
                        started = perf_counter()
                        seed = np.zeros(ranker.n)
                        seed[query['seed_indices']] = query['seed_values']
                        sources = np.array(query['sources'], dtype=np.int64)
                        stages['seed_input_seconds'] = perf_counter()-started
                        started = perf_counter()
                        if args.base_mode == 'certified':
                            y, residual, base_flags, _, levels = certified_base(ranker, seed, KS)
                        else:
                            ranker.base_lu.tol = 1e-12
                            y = ranker.base_lu.solve((1-alpha)*seed)
                            residual = float(ranker.base_lu.last_residuals[0])
                            pad = 64*np.finfo(float).eps*(1+np.linalg.norm(y, 1))/(1-alpha)
                            bound = residual/(1-alpha)+pad
                            order = top_indices(y, min(max(KS)+1, ranker.n))
                            gaps = {k: float(y[order[k-1]]-y[order[k]]) for k in KS}
                            base_flags = {k: gaps[k] > 2*bound for k in KS}
                            levels = [dict(tolerance=1e-12, actual_residual_l1=residual,
                                score_error_bound=bound, gaps=gaps, strict_flags=base_flags)]
                        stages['base_seconds'] = perf_counter()-started
                        base_calls = ranker.base_lu.calls-before_base[0]
                        started = perf_counter()
                        group_plan = compiler.prepare(seed, sources, base_scores=y, base_residual=residual)
                        stages['group_plan_seconds'] = perf_counter()-started
                        before_prepare_calls = ranker.base_lu.calls
                        started = perf_counter()
                        ranker.base_lu.tol = spec.atom_tol
                        family = (directory.prepare(group_plan, ks=KS, output=spec.output) if directory is not None else
                            NativeGroupFamily(group_plan, ks=KS, mode=spec.mode, budget_bytes=budget))
                        ranker.base_lu.tol = args.completion_tol
                        stages['family_prepare_seconds'] = perf_counter()-started
                        prep_calls = ranker.base_lu.calls-before_prepare_calls
                        numeric_peak = directory.nbytes if directory is not None else family.nbytes
                        reserved_peak = directory.reserved_bytes if directory is not None else family.reservation
                        route, r, q = family.route, group_plan.r, group_plan.q
                        # Small metadata, copied outside charged algorithm intervals.
                        source_metadata = dict(original_sources=sources.tolist(),
                            group_masses=group_plan.masses.tolist(),
                            atom_keys=[list(key) for key in group_plan.keys],
                            atom_source_map=group_plan.source_map.tolist(),
                            atom_group_map=group_plan.group_map.tolist(),
                            compiled_source_count=r, compiled_atom_count=q,
                            seed_source_sha256=frozen['seed_source_sha256'][qi])
                        target_seconds, target_counts = 0., Counter()
                        point_acceptance = Counter()
                        for target_index, weights in enumerate(GRID):
                            before_target_graph = meter.snapshot()
                            before_merge_calls = ranker.base_lu.calls
                            started = perf_counter()
                            result = family.solve(weights)
                            if meter.previous_sources:
                                raise RuntimeError('A target did not restore the shared base weights.')
                            elapsed = perf_counter()-started
                            target_seconds += elapsed
                            actual_merges = ranker.base_lu.calls-before_merge_calls
                            graph_delta = difference(meter.snapshot(), before_target_graph)
                            current_numeric = directory.nbytes if directory is not None else family.nbytes
                            current_reserved = directory.reserved_bytes if directory is not None else family.reservation
                            numeric_peak, reserved_peak = max(numeric_peak, current_numeric), max(reserved_peak, current_reserved)
                            initial = sum(bool(result['initial_certified'].get(k, False)) for k in KS)
                            final = sum(bool(result['completion_certified'].get(k, False)) for k in KS)
                            target_counts['initial_certified_cuts'] += initial
                            target_counts['final_certified_cuts'] += final
                            target_counts['actual_merged_solve_calls'] += actual_merges
                            target_counts['effective_reused_targets'] += bool(result['effective_reused'])
                            # Historical summary counters must not count as new work on memo hits.
                            if not result['effective_reused']:
                                for name in ('osp_repairs', 'native_refinements', 'fixed_point_refinement_steps'):
                                    target_counts[name] += result[name]
                            point_acceptance[result['acceptance']] += 1
                            emit(stream['prediction'], dict(dataset=args.dataset, execution=args.execution,
                                spec=spec.name, block=block, question_index=qi,
                                original_question_index=query['original_question_index'], qid=query['qid'],
                                method_order_position=position, target_index=target_index,
                                requested_group_weights=weights, requested_ks=KS, **source_metadata,
                                answer=result, measured_target_seconds=elapsed,
                                actual_merged_solve_calls=actual_merges, graph_call_delta=graph_delta))
                            del result
                        stages['target_seconds'] = target_seconds
                        statistics = family.statistics
                        query_graph = difference(meter.snapshot(), before_graph)
                        before_close = (directory.nbytes if directory is not None else family.nbytes)
                        started = perf_counter()
                        family.close()
                        del family, group_plan, y, seed, sources
                        stages['release_seconds'] = perf_counter()-started
                        active = sum(stages.values())
                        cache_after = (directory.hits, directory.misses, directory.evictions) if directory is not None else (0, 0, 0)
                        cache_delta = tuple(after-before for after, before in zip(cache_after, before_cache))
                        row = dict(dataset=args.dataset, execution=args.execution, spec=spec.name,
                            block=block, question_index=qi, qid=query['qid'],
                            original_question_index=query['original_question_index'], method_order_position=position,
                            original_source_count=r, compiled_atom_count=q, route=route,
                            **stages, active_query_seconds=active,
                            base_levels=len(levels), base_residual_l1=residual,
                            base_backend_solve_calls=base_calls, atom_backend_solve_calls=prep_calls,
                            all_backend_solve_calls=ranker.base_lu.calls-before_base[0],
                            all_backend_native_calls=ranker.base_lu.native_calls-before_base[1],
                            graph_positive_solve_calls=query_graph.get('positive_calls', 0),
                            graph_linear_solve_calls=query_graph.get('linear_calls', 0),
                            graph_weight_update_calls=query_graph.get('weight_update_calls', 0),
                            graph_weight_restore_calls=query_graph.get('weight_restore_calls', 0),
                            direction_materializations=compiler.direction_materializations-before_direction,
                            atom_hits=cache_delta[0], atom_misses=cache_delta[1], atom_evictions=cache_delta[2],
                            peak_private_retained_numeric_bytes=numeric_peak,
                            peak_admission_reserved_numeric_bytes=reserved_peak,
                            retained_numeric_bytes_before_close=before_close,
                            retained_numeric_bytes_after_close=directory.nbytes if directory is not None else 0,
                            **statistics, **target_counts,
                            strict_targets=point_acceptance['strict'],
                            numeric_tolerance_targets=point_acceptance['numeric_tolerance'])
                        # Text-valued scope is in the manifest/predictions, not CSV accounting.
                        row.pop('effective_key_scope')
                        if stream['writer'] is None:
                            stream['writer'] = csv.DictWriter(stream['measurement'], fieldnames=list(row))
                            stream['writer'].writeheader()
                        stream['writer'].writerow(row)
                        emit(stream['levels'], dict(dataset=args.dataset, spec=spec.name, block=block,
                            question_index=qi, qid=query['qid'], strict_flags=base_flags, levels=levels,
                            measured_base_seconds=stages['base_seconds'], graph_call_stages=query_graph))
                        state['totals'].update(stages)
                        state['counts'].update(target_counts)
                        for name in ('logical_targets', 'symbolic_effective_targets', 'summary_reuse_hits',
                                'zero_group_directions', 'symbolic_identity_targets'):
                            state['counts'][name] += statistics[name]
                        state['counts'].update({name: value for name, value in query_graph.items() if name.endswith('_calls')})
                        state['counts']['direction_materializations'] += row['direction_materializations']
                        state['active_seconds'].append(active)  # one scalar/query; no result collection
                        state['acceptance'].update(point_acceptance)
                        state['routes'][route] += 1
                        state['compiled_r'][r] += 1
                        state['compiled_q'][q] += 1
                        state['order_positions'][position] += 1
                        state['peak_numeric_bytes'] = max(state['peak_numeric_bytes'], numeric_peak)
                        state['peak_reserved_bytes'] = max(state['peak_reserved_bytes'], reserved_peak)
                    if (qi+1) % 10 == 0 or qi+1 == len(queries):
                        for stream in streams.values():
                            stream['prediction'].flush()
                            stream['levels'].flush()
                            stream['measurement'].flush()
                        print(json.dumps(dict(dataset=args.dataset, block=block, completed_qids=qi+1,
                            execution=args.execution)), flush=True)
                # All methods have released each live family before this observation.
                combined_memory = process_memory()
                combined_numeric = sum(s['directory'].nbytes for s in states.values() if s['directory'] is not None)
                summaries = {}
                for spec in specs:
                    state, directory = states[spec.name], states[spec.name]['directory']
                    own_setup = state['planning_seconds']+state['directory_setup_seconds']
                    total_active = sum(state['active_seconds'])
                    summary = dict(spec=asdict(spec), block=block, questions=len(queries),
                        charged_shared_startup_seconds=common_startup,
                        planning_seconds=state['planning_seconds'], directory_setup_seconds=state['directory_setup_seconds'],
                        own_setup_seconds=own_setup, **state['totals'], active_query_seconds=total_active,
                        total_seconds=common_startup+own_setup+total_active,
                        p95_active_query_seconds=float(np.percentile(state['active_seconds'], 95)),
                        counts=state['counts'], acceptance=state['acceptance'], routes=state['routes'],
                        compiled_source_count_histogram=state['compiled_r'], compiled_atom_count_histogram=state['compiled_q'],
                        method_order_position_histogram=state['order_positions'],
                        peak_private_retained_numeric_bytes=state['peak_numeric_bytes'],
                        peak_admission_reserved_numeric_bytes=state['peak_reserved_bytes'],
                        numeric_budget_bytes=budget, actual_distinct_operator_views=None,
                        cache_memory=directory.memory if directory is not None else None,
                        cache_hits=directory.hits if directory is not None else 0,
                        cache_misses=directory.misses if directory is not None else 0,
                        cache_evictions=directory.evictions if directory is not None else 0,
                        process_memory_scope='COMBINED cohort; see block summary' if args.execution == 'paired' else 'Single method fresh process; includes shared backend/input/runtime.')
                    summaries[spec.name] = summary
                    write_json(output/spec.name/f'block_{block}_summary.json', summary)
                block_summary = dict(block=block, execution=args.execution,
                    nodes=ranker.n, transition_nnz=ranker.P.nnz,
                    group_nnz={name: group.nnz for name, group in zip(('mention', 'weak'), compiler.groups)},
                    common_startup_seconds=common_startup, constructor_seconds=constructor_seconds,
                    compiler_seconds=compiler_seconds, neutral_graph_warmup_seconds=warmup_seconds,
                    neutral_graph_warmup_stats=warm_stats, graph_meter_setup_seconds=meter_setup_seconds,
                    shared_input_P_Gm_Gw_numeric_bytes=shared_inputs,
                    backend_owned_numeric_bytes=backend_owned, backend_memory=backend_memory,
                    combined_retained_numeric_bytes_after_queries=combined_numeric,
                    combined_process_memory=combined_memory,
                    process_memory_scope='COMBINED cohort' if args.execution == 'paired' else 'SINGLE method process',
                    cohort_loop_wall_seconds=perf_counter()-block_started, specs=summaries)
                block_summaries.append(block_summary)
                write_json(output/f'block_{block}_summary.json', block_summary)
            finally:
                # On failure preserve partial streamed records and the pre-run manifest.
                ranker.base_lu.graph.close()
                del directory, state, states, compiler, ranker, meter
    changed_sources = [name for name, old in frozen['source_sha256'].items() if digest(ROOT/name) != old]
    result = dict(protocol=frozen['protocol'], arguments=vars(args), blocks=block_summaries,
        producer_wall_seconds=perf_counter()-producer_started, priority_class=priority,
        complete_blocks=len(block_summaries), expected_blocks=args.blocks,
        source_changes_during_run=changed_sources,
        inner_native_certificate_histories_recorded=False,
        job_manifest_sha256=digest(output/'job_manifest.json'),
        interpretation='Paired TOTAL sums active query execution plus charged shared startup and own setup. It is not service latency. Paired RSS is combined; use single execution for a method process-memory observation. No result correctness/performance verdict is inferred.')
    write_json(output/'summary.json', result)
    if changed_sources:
        raise RuntimeError('Sources changed during execution; outputs are preserved but not a frozen run.')
    print(json.dumps(dict(output=str(output), complete_blocks=len(block_summaries),
        execution=args.execution)), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('plan', 'run'), default='plan')
    p.add_argument('--dataset', default='hotpot_pool7405')
    p.add_argument('--input-root', default=str(ROOT/'results/natural_policy_workload'))
    p.add_argument('--queries', type=int, default=100)
    p.add_argument('--blocks', type=int, default=1)
    p.add_argument('--execution', choices=('paired', 'single'), default='paired')
    p.add_argument('--single-specname', choices=tuple(s.name for s in SPECS))
    p.add_argument('--base-mode', choices=('certified', 'fine'), default='certified')
    p.add_argument('--completion-tol', type=float, default=1e-6)
    p.add_argument('--budget-mb', type=float, default=4)
    p.add_argument('--cpu', type=int, default=31)
    p.add_argument('--priority', choices=('normal', 'above-normal'), default='above-normal')
    p.add_argument('--output', default=str(ROOT/'results/multigroup/calibration_plan'))
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.queries < 1 or args.blocks < 1 or args.budget_mb <= 0 or args.completion_tol <= 0:
        raise ValueError('Questions, blocks, budget and tolerance must be positive.')
    (plan if args.phase == 'plan' else run)(args)
