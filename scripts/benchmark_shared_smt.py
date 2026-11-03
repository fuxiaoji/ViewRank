"""Shared-parameter QF_NRA comparator on identical frozen numerical states.

This is a solution-function SMT baseline inspired by Junges et al. (2024),
not a reproduction of PROPhESY. Both tests borrow the same local source core,
residual allowances and full-output tails. SAT is only failure of this sufficient
certificate, never an actual instability witness. Prefix and solver timeout are
fixed before measuring; all requested cuts remain in denominators.
"""
from pathlib import Path
from fractions import Fraction as Q
from time import perf_counter
from hashlib import sha256
import argparse,json,sys
import math
import numpy as np
import z3
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.policy_domain_polynomial import compile_domain
from src.policy_domain_fast import certify_many

def rat(x):
    q=x if isinstance(x,Q) else Q(x) if isinstance(x,str) else Q(float(x))
    return z3.RealVal(f'{q.numerator}/{q.denominator}')

def smt(domain,a,selected,guards,timeout):
    start=perf_counter();m,w=z3.Reals('m w');D=z3.Real('D')
    F=[z3.Real(f'f{i}') for i in range(2*domain.r)]
    def poly(p):return z3.Sum([rat(c)*m**i*w**j for (i,j),c in p.items()])
    s=z3.SolverFor('QF_NRA');s.set(timeout=timeout,random_seed=0)
    (ml,mh),(wl,wh)=domain.box;s.add(m>=rat(ml),m<=rat(mh),w>=rat(wl),w<=rat(wh),D==poly(domain.D))
    for f,p in zip(F,domain.F):s.add(f==poly(p))
    amp=[z3.If(f>=0,f,-f) for f in F]
    error=rat(guards['kappa'])*(rat(guards['base_residual'])*D+z3.Sum([rat(r)*f for r,f in zip(guards['atom_residuals'],amp)]))
    candidate=list(map(int,a['candidate_rows']));ys=a['y'];sel=set(selected);lower=[];upper=[]
    for pos,node in enumerate(candidate):
        num=rat(ys[node])*D-z3.Sum([rat(val)*f for val,f,keep in zip(a['retained_values'][pos],F,a['retained_mask'][pos]) if keep])
        tail=z3.Sum([rat(t)*f for t,f,keep in zip(a['tail_bounds'],amp,a['retained_mask'][pos]) if not keep])
        (lower if node in sel else upper).append(num-tail-2*error if node in sel else num+tail)
    outside=np.ones(len(ys),bool);outside[candidate]=False
    bad=[D<=0]
    if np.any(outside):
        outside_bound=rat(float(np.max(ys[outside])))*D+z3.Sum([rat(t)*f for t,f in zip(a['tail_bounds'],amp)])
        bad.extend(lo<=outside_bound for lo in lower)
    if upper:
        # An existential threshold separates the two disjunctions without k*C clauses.
        q=z3.Real('boundary');bad.append(z3.And(z3.Or([lo<=q for lo in lower]),z3.Or([hi>=q for hi in upper])))
    s.add(z3.Or(bad));build=perf_counter()-start;t=perf_counter();answer=s.check();solve=perf_counter()-t
    return dict(status='STABLE' if answer==z3.unsat else 'UNKNOWN',solver_result=str(answer),encoding_seconds=build,solve_seconds=solve,total_seconds=perf_counter()-start,unknown_reason=s.reason_unknown() if answer==z3.unknown else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--frozen-root',type=Path,default=Path(r'E:/ICDE2027-research-data/expanded-contexts-v1/round_domain_v3'));p.add_argument('--queries',type=int,default=20);p.add_argument('--timeout-ms',type=int,default=200);p.add_argument('--output',type=Path,default=ROOT/'results/professor_revision_v1/shared_smt_prefix20');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(status='FROZEN_BEFORE_RUN',questions_per_graph=args.queries,timeout_ms_per_cut=args.timeout_ms,ks=[2,5,10],blocks=1,scope='Verifier only on identical frozen states; no end-to-end operator or memory superiority claim.',guard_policy='Each exact residual/resolvent bound rounded UP to a binary64 dyadic, identically supplied to both verifiers; independently checked >= original. Avoids huge raw summation-denominator artifacts.',z3_version=z3.get_version_string(),source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    with (args.output/'rows.jsonl').open('w',encoding='utf8') as out:
        for short in ['hot','mu']:
            folder=args.frozen_root/f'full_{short}_main_v3'
            trace=folder/'fine_preview/questions.jsonl'
            for row in map(json.loads,trace.open(encoding='utf8')):
                if row['block']!=0:continue
                if row['question_index']>=args.queries:break
                meta=row['audit'];path=folder/Path(meta['path']);raw=path.read_bytes();assert sha256(raw).hexdigest()==meta['sha256']
                with np.load(path) as archive:a={k:archive[k] for k in archive.files}
                domain=compile_domain(a['H_m'],a['H_w'],list(map(Q,meta['mu'])),list(map(Q,meta['nu'])),a['y'][a['source_ids']],((1,4),(.25,2)),a['source_ids'])
                def up(value):
                    raw=Q(value);upper=Q(math.nextafter(float(raw),math.inf));assert upper>=raw;return upper
                guards=dict(kappa=up(meta['kappa']),base_residual=up(meta['base_residual']),atom_residuals=list(map(up,meta['atom_residuals'])))
                selected={k:row['answer']['cuts'][str(k)].get('selected_ids',np.argsort(-a['y'])[:k].tolist()) for k in [2,5,10]}
                t=perf_counter();bern=certify_many(domain,a['y'],a['candidate_rows'],a['retained_values'],a['retained_mask'],a['tail_bounds'],selected,subdivide_depth=2,**guards);bt=perf_counter()-t
                results={str(k):smt(domain,a,ids,guards,args.timeout_ms) for k,ids in selected.items()}
                out.write(json.dumps(dict(dataset=short,question_index=row['question_index'],qid=row['qid'],r=domain.r,bernstein_seconds=bt,bernstein_cuts={str(k):x['status'] for k,x in bern['cuts'].items()},smt=results))+'\n');out.flush()
                print(short,row['question_index'],flush=True)
    (args.output/'complete.json').write_text(json.dumps(dict(status='COMPLETE',protocol=protocol)),encoding='utf8')
if __name__=='__main__':main()
