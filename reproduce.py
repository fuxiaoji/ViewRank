"""ViewRank reviewer entrypoint. Run --download --quick after cloning the repo."""
from pathlib import Path
from hashlib import sha256
from fractions import Fraction
from urllib.request import Request, urlopen
import argparse, csv, json, re, shutil, subprocess, sys, zipfile
ROOT=Path(__file__).resolve().parent

def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def run(script,*args):
    subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,check=True)

def download():
    assets=json.loads((ROOT/'ARTIFACTS.json').read_text())['assets']
    for asset in assets:
        target=ROOT/'downloads'/asset['name'];target.parent.mkdir(exist_ok=True)
        if not target.exists():
            print('Downloading',asset['name'],flush=True)
            with urlopen(Request(asset['url'],headers={'User-Agent':'ViewRank-reproduction'}),timeout=60) as response,target.open('wb') as f:
                shutil.copyfileobj(response,f,8*1024*1024)
        assert digest(target)==asset['sha256'],f"Archive hash mismatch: {target}"
        with zipfile.ZipFile(target) as z:
            for name in z.namelist():
                assert (ROOT/name).resolve().is_relative_to(ROOT.resolve())
            z.extractall(ROOT)

def verify():
    counts={}
    for name in ['FILE_MANIFEST_data.json','FILE_MANIFEST_proofs.json','FILE_MANIFEST_code.json']:
        manifest=json.loads((ROOT/name).read_text())
        for entry in manifest['files']:
            path=ROOT/entry['member']
            assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256'],str(path)
        counts[name]=len(manifest['files'])
    print('PASS: all archived/code payload hashes',counts,flush=True)
    return counts

def table_blocks(path):
    s=path.read_text(encoding='utf8');out={}
    for block in re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',s,re.S):
        label=re.search(r'\\label\{([^}]+)\}',block)
        content=re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}',block,re.S)
        if label and content:out[label.group(1)]=re.sub(r'\s+','',content.group())
    return out

def tables(out):
    out.mkdir(parents=True,exist_ok=True)
    main=ROOT/'results/domain_main_v3';supp=ROOT/'results/domain_supplement_v3'
    report=supp/'evidence_summary_v3/report.json'
    run('scripts/make_domain_results.py','--root',main,'--matched-root',supp,'--output',out/'domain_results.tex')
    run('scripts/make_domain_supplement_results.py','--report',report,'--capacity-match',supp/'independent_verification_v3/capacity_query_set_match_v2.json','--output',out/'domain_supplement_results.tex')
    run('scripts/make_domain_source_count.py','--report',report,'--output',out/'domain_source_count.tex')
    run('scripts/report_professor_revision.py','--tables-only','--output',out/'domain_professor_revision.tex')
    run('scripts/prepare_submission_details.py','--tables-only','--output',out/'details')
    comparisons=[]
    for name in ['domain_results.tex','domain_supplement_results.tex','domain_professor_revision.tex']:
        original=table_blocks(ROOT/'paper/generated'/name);rebuilt=table_blocks(out/name)
        for label,value in original.items():
            assert rebuilt[label]==value,(label,'table values differ')
            comparisons.append(label)
    assert table_blocks(ROOT/'results/submission_details_v4/tristates.tex')==table_blocks(out/'details/tristates.tex')
    comparisons.append('tab:tristates')
    # Table I declares method roles, rather than an experimental aggregate.
    methods=table_blocks(ROOT/'paper/main.tex')['tab:methods']
    (out/'table_I_method_contract.txt').write_text(methods,encoding='utf8')
    # Table V's displayed negative upper bounds must round outward from exact evidence.
    cases=json.loads((ROOT/'evidence/independent_natural_pair_check.json').read_text())['cases']
    source=(ROOT/'paper/generated/domain_natural_examples.tex').read_text()
    displayed=re.findall(r'\$(-[\d.]+)\\!\\times\\!10\^\{(-?\d+)\}\$',source)
    assert len(displayed)==len(cases)==2
    natural=[]
    for case,(coefficient,exponent) in zip(cases,displayed):
        bound=Fraction(coefficient)*Fraction(10)**int(exponent)
        assert Fraction(case['exact_pair_upper'])<=bound<0 and case['passed']
        natural.append(dict(dataset=case['dataset'],question_index=case['question_index'],k=case['k'],exact_upper=case['exact_pair_upper'],displayed_upper=str(bound)))
    (out/'table_V_exact_bounds.json').write_text(json.dumps(natural,indent=2),encoding='utf8')
    result=dict(status='PASS',regenerated_tables=comparisons,table_I='Declared method contract extracted',table_V='Exact saved bounds checked against outward-rounded displayed values',
        scope='Reaggregation of recorded experiments; not a fresh timing experiment. Every formal cohort and all outcomes retained.')
    (out/'TABLE_REPRODUCTION.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('PASS: Tables I--IX and source-count data checked',flush=True)
    return result

def proof(out):
    out.mkdir(parents=True,exist_ok=True)
    for short,qi,k in [('hot',0,2),('mu',1,10)]:
        run('results/domain_supplement_v3/independent_verification_v3/verify_saved_scalar_portable_v1.py',
            '--phase','replay','--run',ROOT/f'results/domain_main_v3/full_{short}_main_v3',
            '--spec','fine_preview','--question-index',qi,'--block',0,'--k',k,'--output',out/short)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['download','quick','verify','tables','proof','paper']:
        p.add_argument('--'+flag,action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'reproduced')
    args=p.parse_args();args.output=args.output.resolve()
    if args.download:download()
    if args.quick or args.verify:verify()
    if args.quick or args.tables:tables(args.output/'tables')
    if args.quick or args.proof:proof(args.output/'proof')
    if args.paper:
        target=args.output/'paper';target.mkdir(parents=True,exist_ok=True)
        subprocess.run(['tectonic',str(ROOT/'paper/main.tex'),'--outdir',str(target),'--keep-logs'],cwd=ROOT,check=True)
    if not any([args.download,args.quick,args.verify,args.tables,args.proof,args.paper]):p.print_help()

if __name__=='__main__':main()
