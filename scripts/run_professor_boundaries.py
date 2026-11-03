"""Frozen propagation/graph-size extension; serial complete cohorts, no filtering."""
from pathlib import Path
from hashlib import sha256
import json,sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
OUT=ROOT/'results/professor_revision_v1';INPUT=OUT/'boundary_inputs'

def producer():
    import benchmark_policy_domains as b
    b.SPECS+= (b.Spec('lift_sparse_fine_preview',certifier='lifting',atom_tol=1e-6,preview_keep=128,depth_override=4),)
    b.main()

def main():
    plan=[]
    for dataset in ['hotpot_pool7405','musique_pool2417']:
        for alpha in [.5,.85,.95]:plan.append((dataset,alpha,100))
    for dataset in ['hotpot_pool1000','musique_pool1000']:plan.append((dataset,.5,100))
    jobs=[]
    for dataset,alpha,n in plan:
        source=ROOT/'results/natural_policy_workload'/dataset
        variant=f'{dataset}_a{str(alpha).replace(".","p")}';dest=INPUT/variant;dest.mkdir(parents=True,exist_ok=True)
        for filename in ['graph.npz','mention_subdistribution.npz','queries.jsonl']:
            p=dest/filename
            if not p.exists():p.hardlink_to(source/filename)
        manifest=json.loads((source/'manifest.json').read_text(encoding='utf8'));manifest.update(alpha=alpha,controlled_extension='Same graph/query arrays; propagation strength override only',parent_dataset=dataset)
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
        jobs.append(dict(dataset=variant,parent=dataset,alpha=alpha,questions=n,output=str(OUT/'boundaries'/variant)))
    protocol=dict(status='FROZEN_BEFORE_RUN',jobs=jobs,blocks=1,methods=['fine_preview','dp_preview_fine','lift_sparse_fine_preview'],all_original_prefix_queries_included=True,order='Listed order, serial; paired method rotations from existing producer',scope='Propagation: fixed arrays. Graph-size: natural context pools, with their declared graph-dependent seeds/sources; not an isolated size-only effect. Amazon: complete pre-existing 16-query graph workload.',process_memory='Paired process peak includes all method directories and is not assigned to any individual method.',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'boundary_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    for job in jobs:
        output=Path(job['output']);output.parent.mkdir(exist_ok=True)
        command=[sys.executable,str(Path(__file__)),'--produce','--phase','run','--study-phase','evaluation','--input-root',str(INPUT),'--dataset',job['dataset'],'--queries',str(job['questions']),'--blocks','1','--budget-mb','16','--depth','2','--scan-backend','dot_fallback','--specs',','.join(protocol['methods']),'--output',str(output)]
        print('START',job['dataset'],flush=True)
        with (OUT/(job['dataset']+'.log')).open('w',encoding='utf8') as log:
            r=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f"Boundary job failed: {job['dataset']} (see log)")
        print('COMPLETE',job['dataset'],flush=True)
    (OUT/'boundaries_complete.json').write_text(json.dumps(dict(status='COMPLETE',jobs=jobs)),encoding='utf8')

if __name__=='__main__':
    if '--produce' in sys.argv:sys.argv.remove('--produce');producer()
    else:main()
