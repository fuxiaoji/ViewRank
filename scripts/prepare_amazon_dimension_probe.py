"""Fixed-source-size structural probe on the independent Amazon topology.

All eight r=4 seed/source requests are retained unchanged. The eight r=16
requests are outside the current compiler's admission contract and receive no
performance claim. Probe variants differ only in explicitly added self mass.
"""
from pathlib import Path
from hashlib import sha256
import json
from scipy import sparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/natural_policy_workload/amazon_scale96';OUT=ROOT/'results/professor_revision_v1/amazon_probe_inputs';OUT.mkdir(exist_ok=True)
queries=[json.loads(s) for s in (BASE/'queries.jsonl').read_text(encoding='utf8').splitlines()];admitted=[q for q in queries if len(q['sources'])==4];assert len(admitted)==8
P=sparse.load_npz(BASE/'graph.npz').tocsc();M=sparse.load_npz(BASE/'mention_subdistribution.npz').tocsc();assert np.all(P.diagonal()==0)
jobs=[]
for name,self_raw in [('amazon_r4_original',0),('amazon_r4_self05',.5)]:
    dest=OUT/name;dest.mkdir(exist_ok=True)
    if self_raw:
        deg=np.diff(P.indptr).astype(float)
        p=(P@sparse.diags(deg/(deg+self_raw))+sparse.diags(self_raw/(deg+self_raw))).tocsc()
        m=(M@sparse.diags(deg/(deg+self_raw))).tocsc()
    else:p,m=P,M
    sparse.save_npz(dest/'graph.npz',p);sparse.save_npz(dest/'mention_subdistribution.npz',m)
    (dest/'queries.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in admitted),encoding='utf8')
    manifest=dict(dataset=name,alpha=.85,nodes=p.shape[0],transition_nnz=p.nnz,question_count=8,controlled_probe=True,self_raw_weight=self_raw,source_seed_arrays_unchanged=True,parent='amazon_scale96',group='Original fixed SHA256 half-neighbor group; not semantic mentions',selection_rule='All and only pre-existing r=4 queries, chosen structurally before this probe is run; no outcome-based filtering.')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');jobs.append(manifest)
(OUT/'protocol.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_PROBE',jobs=jobs,excluded_unsupported_source_count_16=8,ks=[2,5,10],box=[[1,4],[.25,2]],blocks=1),indent=2),encoding='utf8')
print('Prepared both complete r4 strata; no original source array truncated.')
