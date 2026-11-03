"""Residual-budgeted policy families, with identical base/finalization controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.baselines.prpack import PRPackViewRank
from src.baselines.certified_base import certified_base
from src.baselines.certified_native_family import CertifiedNativeFamily
from src.baselines.dense_family_budget import BasisResidentDenseDirectory
from src.budgeted_family import interaction_stencil
from src.retained_policy_family import RetainedStencilDirectory
from src.policy_family_fast import FastPolicyFamily,top_indices
from src.semantic_basis import strength_scale
from src.viewrank import _sparse_bytes
from process_memory import process_memory
from pin_timing_worker import pin_timing_worker


class ResidualRecorder:
    """Attach already measured solve metadata to explicit dense atoms once."""
    def __init__(self,solver):self.solver=solver;self.record=False;self.residuals={}
    def __getattr__(self,name):return getattr(self.solver,name)
    @property
    def tol(self):return self.solver.tol
    @tol.setter
    def tol(self,value):self.solver.tol=value
    def solve(self,rhs):
        result=self.solver.solve(rhs)
        if self.record:self.residuals[id(result)]=float(self.solver.last_residuals[0])
        return result


def dense_query(family,strength,ks):
    """Full responses, same analytical numerical certificate, partial selection."""
    started=perf_counter();r=family.ranker
    a=np.array([strength_scale(strength,m) for m in family.masses])
    K=np.eye(len(a))+family.core*a[None,:]
    c=np.linalg.solve(K,family.y[family.sources]) if len(a) else np.empty(0)
    d=a*c;scores=family.y.copy()
    for value,atom in zip(d,family.atoms):
        if value:scores-=value*atom.response
    h=family.y[family.sources]-K@c
    residual=family.base_residual+sum(abs(v)*atom.residual_l1 for v,atom in zip(d,family.atoms))
    residual+=sum(abs(v*scale)*atom.update_l1 for v,scale,atom in zip(h,a,family.atoms))
    pad=64*np.finfo(float).eps*(1+np.linalg.norm(family.y,1))/(1-r.alpha)
    error=residual/(1-r.alpha)+pad
    order=top_indices(scores,min(max(ks)+1,r.n))
    flags={k:(k==r.n or scores[order[k-1]]-scores[order[k]]>2*error) for k in ks}
    if not all(flags.values()):
        return FastPolicyFamily._complete(family,d,c,ks,started,flags,False)
    return dict(sets={k:order[:k] for k in ks},selected_scores={k:scores[order[:k]] for k in ks},
                initial_certified=flags,completion_certified={},fallback_solve_calls=0,
                seconds=perf_counter()-started,candidate_nodes=r.n,score_error_bound=error)


def flags(result,ks):
    return {k:bool(result.get('initial_certified',{}).get(k,False) or
                   result.get('completion_certified',{}).get(k,False)) for k in ks}


def finished(result,ks):
    if result is None:return False
    if all(flags(result,ks).values()):return True
    # A numerical boundary is retained, never relabelled a strict certificate.
    # Sup-norm score error <= half the fixed reference tolerance suffices for
    # that separate numerical contract, including machine-boundary ties.
    bound=result.get('score_error_bound',result.get('numerical_allowance',np.inf))
    return bound<=5e-13


def fixed_point_finish(ranker,seed,sources,directions,masses,strength,ks,scores):
    """Remove a residual floor left by an inexact shared base, if necessary."""
    a=[strength_scale(strength,m) for m in masses]
    D=sparse.hstack([v*d for v,d in zip(a,directions)],format='csc')
    for step in range(60):
        updated=ranker.alpha*(ranker.P@scores+D@scores[sources])+(1-ranker.alpha)*seed
        difference=float(np.linalg.norm(updated-scores,1));scores=updated
        pad=64*np.finfo(float).eps*(1+np.linalg.norm(scores,1))/(1-ranker.alpha)
        bound=ranker.alpha*difference/(1-ranker.alpha)+pad
        order=top_indices(scores,min(max(ks)+1,ranker.n))
        strict={k:bool(k==ranker.n or scores[order[k-1]]-scores[order[k]]>2*bound) for k in ks}
        if all(strict.values()) or bound<=5e-13:
            return dict(scores=scores,sets={k:order[:k] for k in ks},
                selected_scores={k:scores[order[:k]] for k in ks},completion_certified=strict,
                score_error_bound=bound,fixed_point_refinement_steps=step+1)
    raise RuntimeError('Final view refinement did not meet the declared numeric contract.')


def run(args):
    folder=ROOT/'results/natural_policy_workload'/args.dataset
    input_manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    alpha=float(input_manifest['alpha'])
    P=sparse.load_npz(folder/'graph.npz').tocsc();G=sparse.load_npz(folder/'mention_subdistribution.npz').tocsc()
    queries=[json.loads(line) for line in (folder/'queries.jsonl').read_text(encoding='utf-8').splitlines()][:args.queries or None]
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    ks=(2,5,10);budget=int(args.budget_mb*1024**2)
    controls=args.method in ['native_gap','offset_gap','osp_gap'];rows=[];predictions=[];runs=[];base_levels=[]
    for repeat in range(args.repeats):
        ranker=PRPackViewRank(P,alpha=alpha,tol=1e-12)
        if args.method=='dense' and args.storage=='explicit':ranker.base_lu=ResidualRecorder(ranker.base_lu)
        setup=ranker.base_factor_seconds;directory=None;planning=0.
        if not controls:
            if args.method=='dense':
                if args.storage in ['compiled','stored']:
                    from src.compiled_policy_cache import CompiledDenseDirectory
                    directory=CompiledDenseDirectory(ranker,G,budget,retain_deltas=args.storage=='stored')
                else:directory=BasisResidentDenseDirectory(ranker,budget)
            else:
                t=perf_counter();stencil=interaction_stencil([q['sources'] for q in queries]);planning=perf_counter()-t
                if args.storage in ['compiled','stored']:
                    from src.compiled_policy_cache import CompiledStencilDirectory
                    directory=CompiledStencilDirectory(ranker,G,stencil,args.keep,budget,
                        retain_deltas=args.storage=='stored',exact_core=args.method!='no_core')
                else:directory=RetainedStencilDirectory(ranker,stencil,args.keep,budget)
                del stencil
        setup+=planning;totals=dict(common_input_seconds=0.,base_seconds=0.,prepare_seconds=0.,online_seconds=0.)
        peak_private=0;loop=perf_counter()
        for qi,q in enumerate(queries):
            t=perf_counter();seed=np.zeros(P.shape[0]);seed[q['seed_indices']]=q['seed_values']
            strengths=tuple(q['requested_strengths'])
            sources=np.asarray(q['sources'],dtype=np.int64);masses=np.asarray(q['source_group_masses'])
            directions=[(G[:,int(s)]-m*P[:,int(s)]).tocsc() for s,m in zip(sources,masses)]
            for direction in directions:direction.eliminate_zeros()
            common=perf_counter()-t
            ranker.base_lu.tol=1e-12
            if args.base_mode=='certified':y,ry,base_flags,base_time,levels=certified_base(ranker,seed,ks)
            else:
                t=perf_counter();y=ranker.base_lu.solve((1-alpha)*seed);ry=float(ranker.base_lu.last_residuals[0])
                base_time=perf_counter()-t;base_flags={};levels=[]
            base_levels.append(dict(repeat=repeat,qid=q['qid'],levels=levels))
            before_hits,before_misses=(directory.hits,directory.misses) if directory is not None else (0,0)
            t=perf_counter();ranker.base_lu.tol=args.atom_tol
            if controls:family=CertifiedNativeFamily(ranker,seed,sources,directions,masses,base_scores=y,base_residual=ry,mode=args.method)
            elif args.method=='dense':
                options=dict(base_scores=y)
                if args.storage in ['compiled','stored']:options.update(base_residual=ry,ks=ks)
                else:ranker.base_lu.record=True
                family=directory.prepare(seed,sources,directions,masses,**options)
                if args.storage=='explicit':
                    ranker.base_lu.record=False
                    for atom in family.atoms:
                        if not hasattr(atom,'residual_l1'):
                            atom.residual_l1=ranker.base_lu.residuals.pop(id(atom.response))
                            atom.update_l1=float(ranker.alpha*abs(atom.delta.data).sum())
                family.base_residual=ry
            else:family=directory.prepare(seed,sources,directions,masses,base_scores=y,base_residual=ry,ks=ks,
                always_fused=args.method=='fused',full_scan=args.method=='scan')
            prepare=perf_counter()-t;ranker.base_lu.tol=args.completion_tol
            finalizer=None;osp_finisher=None;online=0.;refinements=0;osp_repairs=0;initial_count=0;completion_count=0;merged_calls=0;singular_cores=0
            for strength in strengths:
                point_started=perf_counter()
                if controls:result=family.solve(strength,ks)
                elif getattr(family,'route',None)=='native':result=None
                elif args.method=='dense':result=dense_query(family,strength,ks)
                else:
                    try:result=family.solve(strength,ks=ks,always_fused=args.method=='fused')
                    except np.linalg.LinAlgError:
                        if args.method!='no_core':raise
                        result=None;singular_cores+=1
                original_initial={} if result is None else result.get('initial_certified',{}).copy()
                refined=False
                if result is not None:
                    initial_count+=sum(result.get('initial_certified',{}).values())
                    completion_count+=sum(result.get('completion_certified',{}).values())
                    merged_calls+=result.get('fallback_solve_calls',0)
                repaired=False
                if not finished(result,ks) and args.method not in ['osp_gap','native_gap']:
                    if osp_finisher is None:osp_finisher=CertifiedNativeFamily(ranker,seed,sources,directions,masses,
                        base_scores=y,base_residual=ry,mode='osp_gap')
                    result=osp_finisher.solve(strength,ks);osp_repairs+=1;repaired=True
                if not finished(result,ks) and args.method!='native_gap':
                    if finalizer is None:finalizer=CertifiedNativeFamily(ranker,seed,sources,directions,masses,
                        base_scores=y,base_residual=ry,mode='native_gap')
                    result=finalizer.solve(strength,ks);refinements+=1;refined=True
                if not finished(result,ks):
                    result=fixed_point_finish(ranker,seed,sources,directions,masses,strength,ks,result['scores'])
                online+=perf_counter()-point_started
                assert finished(result,ks)
                final_bound=result.get('score_error_bound',result.get('numerical_allowance'))
                predictions.append(dict(repeat=repeat,qid=q['qid'],strength=strength,
                    sets={str(k):result['sets'][k].tolist() for k in ks},
                    selected_scores={str(k):result['selected_scores'][k].tolist() for k in ks},
                    boundary_scores={str(k):float(result['selected_scores'][k][-1]) for k in ks},
                    initial_certified={str(k):bool(v) for k,v in original_initial.items()},
                    completion_certified={str(k):bool(v) for k,v in (flags(result,ks) if refined or repaired else result.get('completion_certified',{})).items()},
                    final_native_refinement=refined,
                    acceptance='strict' if all(flags(result,ks).values()) else 'numeric_tolerance',
                    final_score_error_bound=None if final_bound is None else float(final_bound),
                    fixed_point_refinement_steps=result.get('fixed_point_refinement_steps',0),
                    initial_margins={str(k):float(v) for k,v in result.get('initial_margins',{}).items()},
                    candidate_nodes=result.get('candidate_nodes',P.shape[0])))
            private=(directory.memory['total_private_retained_basis_bytes']
                if directory is not None and args.storage=='explicit' and args.method!='dense'
                else directory.nbytes if directory is not None else 0)
            if directory is not None:assert private<=budget
            peak_private=max(peak_private,private)
            rows.append(dict(repeat=repeat,question_index=qi,qid=q['qid'],sources=sources.tolist(),
                common_input_seconds=common,base_seconds=base_time,prepare_seconds=prepare,online_seconds=online,
                total_seconds=common+base_time+prepare+online,private_retained_bytes=private,
                peak_private_retained_bytes=peak_private,initial_certified_k=initial_count,
                completion_certified_k=completion_count,final_native_refinements=refinements,osp_repairs=osp_repairs,merged_solves=merged_calls,
                singular_core_fallbacks=singular_cores,
                atom_hits=directory.hits-before_hits if directory is not None else 0,
                atom_misses=directory.misses-before_misses if directory is not None else 0,
                base_levels=len(levels),base_residual=ry,route=getattr(family,'route',args.method)))
            for key,value in [('common_input_seconds',common),('base_seconds',base_time),('prepare_seconds',prepare),('online_seconds',online)]:totals[key]+=value
            if directory is not None:family.close()
            del family,finalizer,osp_finisher
            if (qi+1)%250==0:print(json.dumps(dict(method=args.method,repeat=repeat,completed=qi+1)),flush=True)
        backend_bytes=sum(v for key,v in ranker.base_lu.graph.memory.items() if key.endswith('_bytes'))+_sparse_bytes(ranker.A)
        runs.append(dict(repeat=repeat,setup_seconds=setup,planning_seconds=planning,**totals,
            charged_seconds=setup+sum(totals.values()),loop_wall_seconds=perf_counter()-loop,
            peak_private_retained_bytes=peak_private,hits=directory.hits if directory is not None else 0,
            misses=directory.misses if directory is not None else 0,
            backend_owned_numeric_bytes=backend_bytes,input_P_G_bytes=_sparse_bytes(P)+_sparse_bytes(G),
            final_cache_bytes=directory.nbytes if directory is not None else 0,**process_memory()))
        ranker.base_lu.graph.close();del ranker,directory
    with (output/'measurements.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'predictions.jsonl').write_text('\n'.join(json.dumps(p) for p in predictions)+'\n',encoding='utf-8')
    (output/'base_levels.jsonl').write_text('\n'.join(json.dumps(p,default=lambda x:x.tolist()) for p in base_levels)+'\n',encoding='utf-8')
    files=[Path(__file__),*[ROOT/path for path in ['src/baselines/certified_base.py',
        'src/baselines/certified_native_family.py','src/baselines/prpack.py',
        'src/baselines/dense_family_budget.py','src/policy_family_fast.py','src/policy_family.py',
        'src/full_scan_policy_family.py','src/budgeted_family.py','src/retained_policy_family.py',
        'src/semantic_basis.py','src/viewrank.py','src/osp.py','scripts/process_memory.py',
        'scripts/pin_timing_worker.py','baselines/build/Release/prpack_native.dll']]]
    if args.storage in ['compiled','stored'] and not controls:files.append(ROOT/'src/compiled_policy_cache.py')
    summary=dict(arguments=vars(args),questions=len(queries),corpus_nodes=P.shape[0],corpus_transition_nnz=P.nnz,
        requested_ks=list(ks),runs=runs,budget_is_active=not controls,alpha=alpha,
        target_views_per_repeat=sum(len(q['requested_strengths']) for q in queries),
        requested_sets_per_repeat=len(ks)*sum(len(q['requested_strengths']) for q in queries),
        median_charged_seconds=float(np.median([r['charged_seconds'] for r in runs])),
        workload_manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        contract='Same complete targets and certified base for every method. All algorithm residual/gap/refinement work charged. Coarse atom errors enter posterior bounds; unresolved views use positive native refinement. Plan/cache/live family basis budget is separate from process working set.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(dict(arguments=vars(args),seconds=summary['median_charged_seconds'],runs=runs)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='hotpot_pool7405')
    p.add_argument('--method',choices=['stencil','dense','scan','fused','no_core','native_gap','offset_gap','osp_gap'],default='stencil')
    p.add_argument('--storage',choices=['compiled','stored','explicit'],default='compiled')
    p.add_argument('--base-mode',choices=['certified','fine'],default='certified')
    p.add_argument('--atom-tol',type=float,default=1e-4);p.add_argument('--completion-tol',type=float,default=1e-6)
    p.add_argument('--queries',type=int,default=0);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--budget-mb',type=float,default=4);p.add_argument('--keep',type=int,default=128)
    p.add_argument('--output',default='results/policy_families/v6_smoke')
    p.add_argument('--variant',default='main')
    p.add_argument('--cpu',type=int,default=31)
    args=p.parse_args();pin_timing_worker(args.cpu)
    with threadpool_limits(limits=1):run(args)
