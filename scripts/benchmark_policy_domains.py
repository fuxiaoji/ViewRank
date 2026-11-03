"""Frozen paired or single-process continuous-domain operator measurements.

All methods receive the same original prefix, sources, seeds, box and cuts.
Active TOTAL and resolved coverage are reported together. The default plan
does not import a numerical backend. Fresh single-process runs additionally
provide method-level OS peaks and complete producer wall time.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark_multigroup import (public_prefix, digest, native_json,
    method_order, set_priority)

KS=(2,5,10)
BOX=((1.,4.),(.25,2.))


def serial(value):
    if isinstance(value,Fraction):return str(value)
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    if hasattr(value,'tolist'):return serial(value.tolist())
    return native_json(value)


def write_json(path,value):
    path.write_text(json.dumps(serial(value),indent=2,allow_nan=False),encoding='utf-8')


def emit(handle,value):
    handle.write(json.dumps(serial(value),allow_nan=False)+'\n')


@dataclass(frozen=True)
class Spec:
    name:str
    mode:str='sparse'
    certifier:str='poly'
    atom_tol:float=1e-4
    keep:int=512
    rank1_native:bool=False
    adaptive:bool=False
    preview_keep:int=0
    depth_override:int|None=None


SPECS=(
    Spec('poly_sparse_l512_a1e4'),
    Spec('poly_sparse_l512_a1e6',atom_tol=1e-6),
    Spec('poly_sparse_l128_a1e3',atom_tol=1e-3,keep=128),
    Spec('poly_sparse_l128_a1e4',keep=128),
    Spec('poly_sparse_l128_a1e6',atom_tol=1e-6,keep=128),
    Spec('poly_sparse_l2048_a1e4',keep=2048),
    Spec('poly_dense_a1e3',mode='dense',atom_tol=1e-3),
    Spec('poly_dense_a1e4',mode='dense'),
    Spec('poly_dense_pruned_a1e4',mode='dense_pruned'),
    Spec('poly_dense_pruned_a1e3',mode='dense_pruned',atom_tol=1e-3),
    Spec('poly_dense_pruned_a1e6',mode='dense_pruned',atom_tol=1e-6),
    Spec('poly_dense_a1e6',mode='dense',atom_tol=1e-6),
    Spec('lift_sparse_l512_a1e4',certifier='lifting'),
    Spec('lift_dense_a1e4',mode='dense',certifier='lifting'),
    Spec('lift_dense_pruned_a1e4',mode='dense_pruned',certifier='lifting'),
    Spec('native_rank1_then_lift_pruned',mode='dense_pruned',certifier='lifting',rank1_native=True),
    Spec('interval_sparse_l512_a1e4',certifier='interval'),
    Spec('interval_dense_pruned_a1e4',mode='dense_pruned',certifier='interval'),
    Spec('no_core_l512_a1e4',mode='no_core'),
    Spec('scan_l512_a1e4',mode='scan'),
    Spec('adaptive_preview',adaptive=True,preview_keep=128),
    Spec('adaptive_no_preview',adaptive=True),
    Spec('fine_preview',atom_tol=1e-6,preview_keep=128),
    Spec('coarse_preview',atom_tol=1e-3,keep=128,preview_keep=128),
    Spec('coarse_no_preview_l256',atom_tol=1e-3,keep=256),
    Spec('dp_preview_a1e4',mode='dense_pruned',preview_keep=128),
    Spec('dp_preview_fine',mode='dense_pruned',atom_tol=1e-6,preview_keep=128),
    Spec('lift_dp_preview',mode='dense_pruned',certifier='lifting',preview_keep=128,depth_override=4),
    Spec('lift_dp_preview_fine',mode='dense_pruned',certifier='lifting',atom_tol=1e-6,preview_keep=128,depth_override=4),
    Spec('native_then_lift_preview_fine',mode='dense_pruned',certifier='lifting',atom_tol=1e-6,rank1_native=True,preview_keep=128,depth_override=4),
    Spec('interval_preview_fine',certifier='interval',atom_tol=1e-6,preview_keep=128),
    Spec('no_core_preview_fine',mode='no_core',atom_tol=1e-6,preview_keep=128),
    Spec('scan_preview_fine',mode='scan',atom_tol=1e-6,preview_keep=128))


def specs_for(args):
    names=set(args.specs.split(',')) if args.specs else {s.name for s in SPECS}
    if names-{s.name for s in SPECS}:raise ValueError('Unknown spec name.')
    selected=tuple(s for s in SPECS if s.name in names)
    if args.execution=='single' and len(selected)!=1:
        raise ValueError('Single execution needs exactly one named --specs method.')
    return selected


def source_paths():
    paths=[Path(__file__),ROOT/'scripts/benchmark_multigroup.py',
        ROOT/'scripts/natural_domain_adapter.py',ROOT/'scripts/process_memory.py',
        ROOT/'scripts/pin_timing_worker.py',ROOT/'baselines/build/Release/prpack_native.dll']
    paths+=list((ROOT/'src').glob('*.py'))+list((ROOT/'src/baselines').glob('*.py'))
    return sorted(set(p for p in paths if p.is_file()))


def protocol(args,folder,manifest,queries,specs):
    return dict(protocol='continuous-domain-paired-v3',dataset=args.dataset,
        input_root=str(folder),arguments=vars(args),questions=len(queries),
        original_qids=[q['qid'] for q in queries],
        original_sources=[q['sources'] for q in queries],
        seed_source_sha256=[hashlib.sha256(json.dumps(q,sort_keys=True,separators=(',',':')).encode()).hexdigest() for q in queries],
        box=BOX,ks=KS,specs=[asdict(s) for s in specs],blocks=args.blocks,
        budget_bytes=int(args.budget_mb*1024**2),base_tolerance=args.base_tol,
        maximum_subdivision_depth=args.depth,verified_point_witnesses=args.witness,
        scan_backend=args.scan_backend,
        adaptive_policy=dict(tiers=[dict(keep=128,atom_tolerance=1e-3),dict(keep=512,atom_tolerance=1e-6)],
            refine_only='UNKNOWN',witness_policy='final',reuse='Already fine/larger atoms reused; all costs charged'),
        cpu=args.cpu,priority=args.priority,execution=args.execution,
        query_order='Unchanged original prefix. Reverse/rotate method order by qid and block.',
        primary_metrics=['complete active operator TOTAL','p95 active query time','STABLE/verified UNSTABLE/UNKNOWN per-cut and all-cut coverage','single-process complete producer wall and OS working-set peaks'],
        paired_cost='Identical charged graph/compiler/neutral warmup startup + own stencil/directory setup + seed/base/prepare/certify/release intervals. Excludes stream serialization, imports, input file load and cohort waiting.',
        memory_scope='Private retained cache+live state and declared admission reservation remain separate from input/backend/current atom scratch. Paired process memory is COMBINED; only single mode supports one-method OS peaks.',
        coverage_rule='UNKNOWN is unresolved, never counted as completed or made to appear faster by conditional filtering. All original queries stay in totals.',
        phase=('exploratory calibration; not a full-workload acceptance or speed conclusion'
            if args.study_phase=='calibration' else 'fixed complete-workload evaluation'),
        no_gold_access=True,nodes=manifest['nodes'])


def freeze(args,folder,manifest,queries,specs):
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise RuntimeError('Use a new empty output directory.')
    output.mkdir(parents=True,exist_ok=True)
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in source_paths()}
    for name,old in hashes.items():
        target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
        if digest(target)!=old:raise RuntimeError('Source changed during freeze.')
    job=protocol(args,folder,manifest,queries,specs)
    job.update(source_sha256=hashes,input_sha256={name:digest(folder/name) for name in
        ('manifest.json','queries.jsonl','graph.npz','mention_subdistribution.npz')})
    write_json(output/'job_manifest.json',job)
    return output,job


def run(args,folder,manifest,queries,specs):
    producer_started=perf_counter()
    output,job=freeze(args,folder,manifest,queries,specs)
    import numpy as np
    from scipy import sparse
    from threadpoolctl import threadpool_limits
    from process_memory import process_memory
    from pin_timing_worker import pin_timing_worker
    from natural_domain_adapter import sort_only_csc_copy
    from src.baselines.prpack import PRPackViewRank
    from src.budgeted_family import interaction_stencil
    from src.policy_domain_execution import DomainCompiler,DomainDirectory
    from src.policy_domain_native import certify_rank1_native
    from src.policy_domain_adaptive import AdaptiveDomainOperator
    from src.policy_domain_controls import prepare_geometry,clear_geometry_cache
    pin_timing_worker(args.cpu);priority=set_priority(args.priority)
    all_blocks=[]
    with threadpool_limits(1),ExitStack() as stack:
        streams={}
        for spec in specs:
            path=output/spec.name;path.mkdir()
            streams[spec.name]=stack.enter_context((path/'questions.jsonl').open('w',encoding='utf-8'))
        for block in range(args.blocks):
            rawP=sparse.load_npz(folder/'graph.npz');rawM=sparse.load_npz(folder/'mention_subdistribution.npz')
            started=perf_counter()
            P,p_sort=sort_only_csc_copy(rawP,'raw P');M,m_sort=sort_only_csc_copy(rawM,'raw M')
            del rawP,rawM
            ranker=PRPackViewRank(P,alpha=float(manifest['alpha']),tol=args.base_tol)
            compiler=DomainCompiler(ranker,M)
            del P,M
            warm=np.zeros(ranker.n);warm[0]=1-ranker.alpha
            ranker.base_lu.graph.solve_linear(warm,ranker.alpha,1e-12)
            del warm
            startup=perf_counter()-started
            geometry_seconds=0.;geometry_integer_payload=0
            if any(s.certifier in ('poly','interval') for s in specs):
                geometry_started=perf_counter();clear_geometry_cache()
                boxes=[]
                def geometry_boxes(box,depth):
                    boxes.append(box)
                    if depth:
                        axis=max(range(2),key=lambda i:box[i][1]-box[i][0])
                        lo,hi=box[axis];mid=(lo+hi)/2
                        for a,b in ((lo,mid),(mid,hi)):
                            child=list(box);child[axis]=(a,b)
                            geometry_boxes(tuple(child),depth-1)
                geometry_boxes(BOX,args.depth)
                for r in sorted({len(q['sources']) for q in queries if q['sources']}):
                    for box in boxes:
                        geometry_integer_payload+=prepare_geometry(box,r).integer_payload_bytes
                geometry_seconds=perf_counter()-geometry_started
            states={}
            for spec in specs:
                started=perf_counter()
                stencil=(interaction_stencil([q['sources'] for q in queries
                    if not (spec.rank1_native and len(q['sources'])==1)])
                    if spec.mode in ('sparse','scan','dense_pruned') else {})
                directory=DomainDirectory(compiler,stencil,budget_bytes=job['budget_bytes'],keep=spec.keep,
                    mode=spec.mode,preview_keep=spec.preview_keep,scan_backend=args.scan_backend)
                states[spec.name]=dict(directory=directory,setup_seconds=perf_counter()-started+
                    (geometry_seconds if spec.certifier in ('poly','interval') else 0),
                    stages=Counter(),statuses=Counter(),all_cut=Counter(),active=[],routes=Counter())
                del stencil
            cohort_started=perf_counter()
            try:
                for qi,query in enumerate(queries):
                    for position,spec in enumerate(method_order(specs,qi,block)):
                        state=states[spec.name];directory=state['directory'];stages={}
                        before_calls=ranker.base_lu.calls
                        cache_names=('hits','misses','evictions','refinement_invalidations')
                        before_cache=tuple(getattr(directory,name) for name in cache_names)
                        started=perf_counter()
                        seed=np.zeros(ranker.n);seed[query['seed_indices']]=query['seed_values']
                        stages['seed_seconds']=perf_counter()-started
                        family=None;y=None;audit_metadata=None
                        audit_enabled=bool(args.audit_specs and spec.name in args.audit_specs.split(','))
                        def save_audit(current,tier_index=None):
                            if current.values is None:return None
                            audit_dir=output/spec.name/'audit';audit_dir.mkdir(exist_ok=True)
                            suffix='' if tier_index is None else f'_t{tier_index}'
                            audit_path=audit_dir/f'b{block}_q{qi:05d}{suffix}.npz'
                            np.savez_compressed(audit_path,y=y,candidate_rows=current.candidate_rows,
                                retained_values=current.values,retained_mask=current.mask,
                                H_m=current.Hm,H_w=current.Hw,source_ids=current.sources,
                                tail_bounds=current.tails)
                            return dict(path=str(audit_path.relative_to(output)),sha256=digest(audit_path),
                                mu=current.mu,nu=current.nu,base_residual=current.rho,
                                atom_residuals=current.atom_rhos,kappa=compiler.guard.kappa_old)
                        if spec.rank1_native and len(query['sources'])==1:
                            stages['base_seconds']=stages['prepare_seconds']=0.
                            started=perf_counter()
                            answer=certify_rank1_native(compiler,directory,seed,query['sources'],BOX,KS)
                            stages['certify_seconds']=perf_counter()-started
                        else:
                            started=perf_counter()
                            ranker.base_lu.tol=args.base_tol
                            y=ranker.base_lu.solve((1-ranker.alpha)*seed)
                            stages['base_seconds']=perf_counter()-started
                            depth=args.depth if spec.depth_override is None else spec.depth_override
                            if spec.adaptive:
                                started=perf_counter()
                                answer=AdaptiveDomainOperator(directory,preview_keep=spec.preview_keep).execute(
                                    seed,query['sources'],base_scores=y,ks=KS,box=BOX,depth=depth,
                                    certifier=spec.certifier,early_witness=args.witness,
                                    audit_sink=save_audit if audit_enabled else None)
                                outer=perf_counter()-started-answer['audit_seconds']
                                stages['prepare_seconds']=answer['stage_totals']['prepare_seconds']
                                stages['release_seconds']=answer['stage_totals']['release_seconds']
                                stages['certify_seconds']=outer-stages['prepare_seconds']-stages['release_seconds']
                            else:
                                started=perf_counter()
                                ranker.base_lu.tol=spec.atom_tol
                                compiler.atom_tolerance=spec.atom_tol
                                family=directory.prepare(seed,query['sources'],base_scores=y,ks=KS)
                                stages['prepare_seconds']=perf_counter()-started
                                started=perf_counter()
                                answer=family.certify(box=BOX,ks=KS,depth=depth,
                                    early_witness=args.witness,certifier=spec.certifier)
                                stages['certify_seconds']=perf_counter()-started
                        if family is not None and audit_enabled:
                            # Verification evidence is streamed outside active
                            # operator intervals. Performance-only runs omit it.
                            audit_metadata=save_audit(family)
                        if not spec.adaptive:
                            started=perf_counter()
                            if family is not None:family.close()
                            stages['release_seconds']=perf_counter()-started
                        active=sum(stages.values())
                        statuses=Counter(v['status'] for v in answer['cuts'].values())
                        if set(answer['cuts'])!={2,5,10}:raise RuntimeError('Missing cut outputs.')
                        if any(v not in ('STABLE','UNSTABLE','UNKNOWN') for v in statuses):raise RuntimeError('Unknown result class.')
                        resolved=not statuses['UNKNOWN']
                        state['stages'].update(stages);state['statuses'].update(statuses)
                        state['all_cut']['resolved' if resolved else 'unresolved']+=1
                        state['active'].append(active)
                        state['routes'][answer.get('actual_route',spec.mode)]+=1
                        emit(streams[spec.name],dict(dataset=args.dataset,spec=spec.name,block=block,
                            question_index=qi,qid=query['qid'],original_sources=query['sources'],
                            seed_source_sha256=job['seed_source_sha256'][qi],method_order_position=position,
                            stages=stages,active_query_seconds=active,
                            backend_solve_calls=ranker.base_lu.calls-before_calls,
                            cache_delta=dict(zip(cache_names,
                                [getattr(directory,name)-old for name,old in zip(cache_names,before_cache)])),
                            private_memory=directory.memory,audit=audit_metadata,answer=answer))
                        del family,seed,y,answer
                    if (qi+1)%10==0 or qi+1==len(queries):
                        for handle in streams.values():handle.flush()
                        print(json.dumps(dict(dataset=args.dataset,block=block,completed_questions=qi+1)),flush=True)
                summaries={}
                for spec in specs:
                    state=states[spec.name];active=sum(state['active'])
                    summary=dict(spec=asdict(spec),questions=len(queries),block=block,
                        charged_shared_startup_seconds=startup,own_setup_seconds=state['setup_seconds'],
                        stages=state['stages'],active_query_seconds=active,
                        total_seconds=startup+state['setup_seconds']+active,
                        p95_active_query_seconds=float(np.percentile(state['active'],95)),
                        statuses=state['statuses'],all_cut_queries=state['all_cut'],routes=state['routes'],
                        cache={name:getattr(state['directory'],name) for name in cache_names},
                        private_memory=state['directory'].memory)
                    summaries[spec.name]=summary
                    write_json(output/spec.name/f'block_{block}_summary.json',summary)
                block_summary=dict(block=block,execution=args.execution,shared_startup_seconds=startup,
                    geometry_preparation_seconds=geometry_seconds,
                    shared_geometry_integer_payload_bytes=geometry_integer_payload,
                    geometry_scope='Immutable box/degree transform only. Preparation charged to every logical poly/interval method; includes no query/source/seed progress. Not private cached response state; actual process RSS includes all object overhead.',
                    sort_only_audits=dict(P=p_sort,M=m_sort),specs=summaries,
                    cohort_loop_wall_seconds=perf_counter()-cohort_started,process_memory=process_memory(),
                    process_memory_scope='COMBINED cohort' if args.execution=='paired' else 'Single method fresh process')
                write_json(output/f'block_{block}_summary.json',block_summary);all_blocks.append(block_summary)
            finally:
                ranker.base_lu.graph.close()
                del states,directory,state,compiler,ranker
    drift=[name for name,old in job['source_sha256'].items() if digest(ROOT/name)!=old]
    result=dict(protocol=job['protocol'],complete_blocks=len(all_blocks),expected_blocks=args.blocks,
        blocks=all_blocks,producer_wall_seconds=perf_counter()-producer_started,
        actual_priority_class=priority,source_drift=drift,job_manifest_sha256=digest(output/'job_manifest.json'))
    write_json(output/'summary.json',result)
    if drift:raise RuntimeError('Source changed during numerical execution.')
    print(json.dumps(dict(output=str(output),complete_blocks=len(all_blocks))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('plan','run'),default='plan')
    p.add_argument('--study-phase',choices=('calibration','evaluation'),default='calibration')
    p.add_argument('--dataset',default='hotpot_pool7405')
    p.add_argument('--input-root',default=str(ROOT/'results/natural_policy_workload'))
    p.add_argument('--queries',type=int,default=100)
    p.add_argument('--blocks',type=int,default=1)
    p.add_argument('--budget-mb',type=float,default=16)
    p.add_argument('--depth',type=int,default=2)
    p.add_argument('--base-tol',type=float,default=1e-10)
    p.add_argument('--scan-backend',choices=('outward','dot_fallback'),default='outward')
    p.add_argument('--execution',choices=('paired','single'),default='paired')
    p.add_argument('--specs')
    p.add_argument('--audit-specs',help='Optional comma-separated sparse specs for untimed compressed proof evidence; omit in performance-only single runs.')
    p.add_argument('--witness',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--cpu',type=int,default=31)
    p.add_argument('--priority',choices=('normal','above-normal'),default='above-normal')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.queries<1 or args.blocks<1 or args.budget_mb<=0 or args.depth<0 or args.base_tol<=0:raise ValueError('Invalid positive experiment dimensions.')
    folder,manifest,queries=public_prefix(args);specs=specs_for(args)
    if args.phase=='plan':print(json.dumps(protocol(args,folder,manifest,queries,specs),indent=2))
    else:run(args,folder,manifest,queries,specs)


if __name__=='__main__':main()
