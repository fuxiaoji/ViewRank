"""Rebuild the two Amazon graphs/groups from the hash-pinned official edge file.

The eight full, original seed/source requests are copied from their frozen file.
No new selection, truncation or performance measurement occurs here.
"""
from pathlib import Path
from hashlib import sha256
import argparse, gzip, json
import numpy as np
from scipy import sparse
ROOT = Path(__file__).resolve().parents[1]
HASH = 'bb4b9fcf549c9174049e885e9a0a4057c9724846e7297c4a1ea299b8e92eec34'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=ROOT/'data/scale_public/com-amazon.ungraph.txt.gz')
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    assert sha256(args.archive.read_bytes()).hexdigest()==HASH
    with gzip.open(args.archive,'rt') as f: edges=np.loadtxt(f,dtype=np.int64,comments='#')
    ids, inv=np.unique(edges,return_inverse=True); e=inv.reshape(-1,2)
    rows=np.concatenate([e[:,1],e[:,0]]);cols=np.concatenate([e[:,0],e[:,1]])
    W=sparse.csc_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(ids),len(ids)))
    W.sum_duplicates();degree=np.asarray(W.sum(axis=0)).ravel()
    P=(W@sparse.diags(1/degree)).tocsc()
    assert P.shape==(334863,334863) and P.nnz==1851744 and len(edges)==925872
    eligible=np.flatnonzero(np.diff(P.indptr)>=2)
    B=eligible[np.lexsort((eligible,-np.diff(P.indptr)[eligible]))[:128]]
    gi=[];gj=[];gv=[]
    for source in B:
        dest=P.indices[P.indptr[source]:P.indptr[source+1]]
        hashes=[int.from_bytes(sha256(f'strength-group:{source}:{v}'.encode()).digest()[:8],'little') for v in dest]
        chosen=dest[np.argsort(hashes,kind='stable')[:max(1,len(dest)//2)]]
        gi.extend(chosen);gj.extend([source]*len(chosen));gv.extend(np.asarray(P[chosen,source].toarray()).ravel())
    M=sparse.csc_matrix((gv,(gi,gj)),shape=P.shape)
    base=ROOT/'results/professor_revision_v1/amazon_probe_inputs'
    queries=(base/'amazon_r4_original/queries.jsonl').read_text(encoding='utf8')
    qs=[json.loads(q) for q in queries.splitlines()]
    assert len(qs)==8 and all(len(q['sources'])==4 for q in qs)
    assert [q['seed_indices'][0] for q in qs]==B[:8].tolist()
    evidence=[]
    for name,self_raw in [('amazon_r4_original',0),('amazon_r4_self05',.5)]:
        pp=P if not self_raw else (P@sparse.diags(degree/(degree+self_raw))+sparse.diags(self_raw/(degree+self_raw))).tocsc()
        mm=M if not self_raw else (M@sparse.diags(degree/(degree+self_raw))).tocsc()
        dest=args.output/name;dest.mkdir(parents=True,exist_ok=True)
        for filename,matrix in [('graph.npz',pp),('mention_subdistribution.npz',mm)]:
            reference=sparse.load_npz(base/name/filename).tocsc(); difference=(reference-matrix).tocoo()
            assert difference.nnz==0, (name,filename,'stored graph differs')
            sparse.save_npz(dest/filename,matrix)
        (dest/'queries.jsonl').write_text(queries,encoding='utf8')
        (dest/'manifest.json').write_bytes((base/name/'manifest.json').read_bytes())
        evidence.append(dict(dataset=name,graph_and_group_exactly_equal=True,all_full_sources_unchanged=True))
    (args.output/'rebuild_check.json').write_text(json.dumps(dict(status='PASS',raw_sha256=HASH,variants=evidence),indent=2),encoding='utf8')
    print('PASS: both Amazon graph/group matrices exactly match frozen inputs; eight complete requests unchanged')

if __name__=='__main__':main()
