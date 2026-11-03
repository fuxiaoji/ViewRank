"""Exact structural dimension audit over every original formal query/source.

For disjoint mention/weak/self groups, three positive groups give local rank 2.
Exactly two give one of three independent direction types: mention+weak (ratio),
mention+self (m only), weak+self (w only). Distinct types across columns give
global rank 2. Group positivity uses exact sums of stored dyadic numbers.
"""
from pathlib import Path
from fractions import Fraction as Q
from collections import Counter
from hashlib import sha256
import json,sys
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/professor_revision_v1';OUT.mkdir(exist_ok=True)
reports=[]
for dataset,n in [('hotpot_pool7405',1000),('musique_pool2417',2417)]:
    folder=ROOT/'results/natural_policy_workload'/dataset
    P=sparse.load_npz(folder/'graph.npz').tocsc();M=sparse.load_npz(folder/'mention_subdistribution.npz').tocsc()
    P.sort_indices();M.sort_indices()
    queries=[json.loads(s) for s in (folder/'queries.jsonl').read_text(encoding='utf8').splitlines()][:n]
    sources=sorted({s for q in queries for s in q['sources']});stats={};rows=[]
    for s in sources:
        lo,hi=P.indptr[s:s+2];ml,mh=M.indptr[s:s+2]
        vals=P.data[lo:hi];ids=P.indices[lo:hi]
        positions=np.searchsorted(ids,M.indices[ml:mh])
        assert np.all(ids[positions]==M.indices[ml:mh])
        assert np.array_equal(vals[positions],M.data[ml:mh]), 'Group supports must be disjoint.'
        assert not np.any(M.indices[ml:mh]==s), 'Mention must exclude the self loop.'
        total=sum(map(Q.from_float,map(float,vals)),Q(0));mention=sum(map(Q.from_float,map(float,M.data[ml:mh])),Q(0))
        selfq=sum((Q.from_float(float(v)) for i,v in zip(ids,vals) if i==s),Q(0));weak=total-mention-selfq
        assert total>0 and min(mention,weak,selfq)>=0
        groups=tuple(g for g,mass in [('m',mention),('w',weak),('s',selfq)] if mass>0)
        rank=2 if len(groups)==3 else 1 if len(groups)==2 else 0
        stats[s]=(rank,groups,float(selfq/total))
    for q in queries:
        ranks=[stats[s][0] for s in q['sources']];types={stats[s][1] for s in q['sources'] if stats[s][0]}
        rank=2 if 2 in ranks or len(types)>1 else 1 if types else 0
        rows.append(dict(qid=q['qid'],question_index=q['original_question_index'],sources=q['sources'],effective_dimension=rank))
    report=dict(dataset=dataset,questions=n,nodes=P.shape[0],unique_modified_sources=len(sources),
        zero_self_sources=sum(stats[s][2]==0 for s in sources),
        self_mass_quantiles=dict(zip(['min','q25','median','q75','max'],map(float,np.quantile([stats[s][2] for s in sources],[0,.25,.5,.75,1])))),
        query_dimensions=dict(Counter(row['effective_dimension'] for row in rows)),
        source_group_types={','.join(k):v for k,v in Counter(stats[s][1] for s in sources).items()},
        input_sha256={f:sha256((folder/f).read_bytes()).hexdigest() for f in ['graph.npz','mention_subdistribution.npz','queries.jsonl']},rows=rows)
    reports.append(report)
result=dict(status='PASS',method=__doc__,datasets=reports)
(OUT/'effective_dimension.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps([{k:v for k,v in r.items() if k not in ['rows','input_sha256']} for r in reports]))
