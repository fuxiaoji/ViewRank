"""Build final manuscript tables only from complete frozen domain workloads."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[1]
LABELS={
    'fine_preview':'ViewRank--Sparse',
    'dp_preview_a1e4':'ViewRank--DensePruned ($10^{-4}$)',
    'dp_preview_fine':'ViewRank--DensePruned ($10^{-6}$)',
    'lift_dp_preview_fine':'Source lifting, full responses',
    'native_then_lift_preview_fine':'Native rank-one + source lifting',
}
MATCHED_LABELS={
    'fine_preview':'ViewRank--Sparse (paired A)',
    'lift_sparse_fine_preview':'SourceLift, compressed (paired A)',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(path):
    result=json.loads((path/'summary.json').read_text(encoding='utf-8'))
    job=json.loads((path/'job_manifest.json').read_text(encoding='utf-8'))
    if result.get('source_drift') != [] or result.get('complete_blocks') != result.get('expected_blocks') or result['complete_blocks'] != job['blocks']:
        raise ValueError('Manuscript input must be a complete frozen run')
    if result['job_manifest_sha256'] != sha(path/'job_manifest.json'):
        raise ValueError('Manuscript manifest SHA differs from the completed summary')
    if len(result['blocks']) != job['blocks'] or {b['block'] for b in result['blocks']} != set(range(job['blocks'])):
        raise ValueError('Manuscript input is missing complete block payloads')
    result['blocks']=sorted(result['blocks'],key=lambda b:b['block'])
    return job,result


def extract_method(job,result,name):
    rows=[b['specs'][name] for b in result['blocks']]
    count=job['questions']
    for block,row in enumerate(rows):
        if row['questions'] != count or row['block'] != block or sum(row['statuses'].values()) != count*len(job['ks']):
            raise ValueError('Method block has a different original-query/cut denominator')
        if sum(row['all_cut_queries'].values()) != count:
            raise ValueError('Method block has an incomplete all-cut coverage denominator')
    return dict(questions=count,blocks=len(rows),
        total_seconds=[r['total_seconds'] for r in rows],
        p95_ms=[r['p95_active_query_seconds']*1000 for r in rows],
        all_cut_resolved=[r['all_cut_queries'].get('resolved',0) for r in rows],
        statuses=[r['statuses'] for r in rows],
        private_peak_bytes=[r['private_memory']['peak_private_retained_numeric_bytes'] for r in rows])


def provenance(path):
    return dict(path=str(path),summary_sha256=sha(path/'summary.json'),
        manifest_sha256=sha(path/'job_manifest.json'))


def matched_checks(main,matched):
    main_primary=next(s for s in main['specs'] if s['name']=='fine_preview')
    specs={s['name']:s for s in matched['specs']}
    shared_sources={k:v for k,v in matched['source_sha256'].items()
        if k != 'scripts/benchmark_policy_domains_supplement.py'}
    checks=dict(dataset_equal=main['dataset']==matched['dataset'],
        original_query_count_equal=main['questions']==matched['questions'],
        original_qids_equal=main['original_qids']==matched['original_qids'],
        original_sources_equal=main['original_sources']==matched['original_sources'],
        original_seed_source_hashes_equal=main['seed_source_sha256']==matched['seed_source_sha256'],
        input_file_hashes_equal=main['input_sha256']==matched['input_sha256'],
        common_implementation_hashes_equal=main['source_sha256']==shared_sources,
        original_ks_equal=main['ks']==matched['ks'],original_box_equal=main['box']==matched['box'],
        budget_equal_16MiB=main['budget_bytes']==matched['budget_bytes']==16*1024**2,
        base_tolerance_equal=main['base_tolerance']==matched['base_tolerance'],
        three_complete_blocks=main['blocks']==matched['blocks']==3,
        paired_execution=main['execution']==matched['execution']=='paired',
        two_method_alternating=matched.get('method_order_policy')=='two_alternating_else_original',
        matched_method_set=set(specs)==set(MATCHED_LABELS),
        primary_configuration_equal=specs.get('fine_preview')==main_primary,
        both_compressed_fine=all(s['mode']=='sparse' and s['keep']==512 and
            s['preview_keep']==128 and s['atom_tol']==1e-6 for s in specs.values()),
        lifting_depth_four=specs.get('lift_sparse_fine_preview',{}).get('depth_override')==4)
    if not all(checks.values()):
        raise ValueError('Matched-cohort validation failed: '+', '.join(k for k,v in checks.items() if not v))
    return checks


def table_values(facts,datasets,name,matched=False):
    values=[]
    for dataset in datasets:
        row=(facts[dataset]['same_compressed_cohort']['methods'][name]
            if matched else facts[dataset][name])
        values.extend([f"{median(row['total_seconds']):.2f}",
            f"{100*median(row['all_cut_resolved'])/row['questions']:.2f}",f"{median(row['p95_ms']):.1f}"])
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(r'E:/ICDE2027-research-data/expanded-contexts-v1/round_domain_v3'))
    p.add_argument('--matched-root',type=Path,default=ROOT/'results/domain_supplement_v3')
    p.add_argument('--output',type=Path,default=ROOT/'paper/generated/domain_results.tex')
    args=p.parse_args()
    paths=[args.root/f'full_{name}_main_v3' for name in ('hot','mu')]
    runs=[read_run(path) for path in paths]
    matched_paths=[args.matched_root/f'A_core_matched_{name}' for name in ('hot','mu')]
    matched_runs=[read_run(path) for path in matched_paths]
    facts={}
    for path,(job,result),matched_path,(matched_job,matched_result) in zip(paths,runs,matched_paths,matched_runs):
        dataset=job['dataset'];facts[dataset]={}
        for spec in job['specs']:
            name=spec['name']
            facts[dataset][name]=extract_method(job,result,name)
        checks=matched_checks(job,matched_job)
        methods={name:extract_method(matched_job,matched_result,name) for name in MATCHED_LABELS}
        main,lift=(methods[name] for name in MATCHED_LABELS)
        facts[dataset]['original_main_cohort']=provenance(path)
        facts[dataset]['same_compressed_cohort']=dict(**provenance(matched_path),checks=checks,
            methods=methods,ks=matched_job['ks'],box=matched_job['box'],budget_bytes=matched_job['budget_bytes'],
            method_configurations=matched_job['specs'],
            paired_primary_cost_reduction_percent=[100*(1-a/b) for a,b in zip(main['total_seconds'],lift['total_seconds'])],
            paired_primary_p95_reduction_percent=[100*(1-a/b) for a,b in zip(main['p95_ms'],lift['p95_ms'])],
            paired_primary_resolved_question_difference=[a-b for a,b in zip(main['all_cut_resolved'],lift['all_cut_resolved'])],
            comparison_scope='Same-cohort, same-block pairs only. Original-main and matched A absolute times are not pooled. Repeats reuse the same original queries.')
    lines=[r'\subsection{Complete-workload operator comparison}',r'\label{sec:domain-results}',
        r'Table~\ref{tab:domain-main} reports the fixed workloads, including startup, method-owned setup, all query stages, and unresolved requests. Times and p95 values are medians of three complete blocks; coverage requires every requested cutoff resolved. All block values are retained in the artifact.',
        r'\begin{table*}[t]\centering\small',
        r'\caption{Complete policy-domain workloads at 16 MiB retained-state budget. Total seconds and p95 milliseconds are complete-block medians; resolved coverage requires all three cutoffs. DensePruned shares ViewRank\textquotesingle s polynomial certificate. The separated bottom panel is an independent paired cohort with the same compressed representation for both certificates. Compare costs within a panel; absolute times are not pooled across cohorts.}',
        r'\label{tab:domain-main}',
        r'\begin{tabular}{lrrrrrr}\toprule',
        r'& \multicolumn{3}{c}{HotpotQA, 1,000 questions}&\multicolumn{3}{c}{MuSiQue, 2,417 questions}\\',
        r'Method & Total (s)&Resolved (\%)&p95 (ms)&Total (s)&Resolved (\%)&p95 (ms)\\\midrule']
    datasets=[job['dataset'] for job,_ in runs]
    for name,label in LABELS.items():
        lines.append(label+' & '+' & '.join(table_values(facts,datasets,name))+r'\\')
    lines += [r'\midrule',r'\multicolumn{7}{l}{\emph{Separate paired cohort A: same compressed representation}}\\']
    for name,label in MATCHED_LABELS.items():
        lines.append(label+' & '+' & '.join(table_values(facts,datasets,name,matched=True))+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table*}']
    for dataset in datasets:
        main=facts[dataset]['fine_preview']
        lift=facts[dataset]['lift_dp_preview_fine']
        reductions=[100*(1-a/b) for a,b in zip(main['total_seconds'],lift['total_seconds'])]
        facts[dataset]['paired_main_vs_full_lifting_reductions_percent']=reductions
    hot,mu=datasets
    gains=[facts[d]['paired_main_vs_full_lifting_reductions_percent'] for d in datasets]
    matched=[facts[d]['same_compressed_cohort'] for d in datasets]
    reductions=[row['paired_primary_cost_reduction_percent'] for row in matched]
    lines.append('The strongest layout-matched comparison is cohort A. '
        'Both methods use sparse $L=512$ retention, 128 current-base preview coordinates, and atom tolerance $10^{-6}$; SourceLift receives subdivision depth four. '
        'ViewRank--Sparse lowers complete-workload cost in every paired block: '
        f'{min(reductions[0]):.1f}--{max(reductions[0]):.1f}\\% on HotpotQA and '
        f'{min(reductions[1]):.1f}--{max(reductions[1]):.1f}\\% on MuSiQue. '
        'All original requests and unresolved outcomes remain included.')
    coverage=[];tail=[]
    for row in matched:
        primary=row['methods']['fine_preview'];lift=row['methods']['lift_sparse_fine_preview']
        coverage.append(f"{median(primary['all_cut_resolved']):.0f} versus {median(lift['all_cut_resolved']):.0f}")
        tail.append(f"{median(primary['p95_ms']):.1f} versus {median(lift['p95_ms']):.1f}")
    lines.append(f'Complete-audit counts are {coverage[0]} on HotpotQA and {coverage[1]} on MuSiQue, identical across repetitions. '
        f'Median all-question p95 values are {tail[0]} ms and {tail[1]} ms, respectively. '
        'The shared-domain certificate therefore improves complete cost, resolution and tail latency after matching compressed retention.')
    lines.append('The original main cohort also compares full-response source lifting: '
        f'the primary reduces complete cost by {min(gains[0]):.1f}--{max(gains[0]):.1f}\\% on HotpotQA and '
        f'{min(gains[1]):.1f}--{max(gains[1]):.1f}\\% on MuSiQue. '
        'The atom-free rank-one hybrid remains visible alongside it.')
    sparse=facts[hot]['fine_preview'];dense=facts[hot]['dp_preview_fine']
    hot_reduction=100*(1-median(sparse['total_seconds'])/median(dense['total_seconds']))
    lines.append('The same certificate supports complementary physical layouts. '
        f'Sparse retention reduces HotpotQA total cost by {hot_reduction:.1f}\\% relative to same-tolerance DensePruned, '
        'while DensePruned gives lower total cost on MuSiQue. '
        'The fine full-response layout resolves one additional question on each dataset.')
    lines += [r'\input{generated/domain_performance.tex}',
        r'\IfFileExists{generated/domain_source_count.tex}{\input{generated/domain_source_count.tex}}{}',
        r'% Remaining component controls, budgets and fresh-process memory come from their own complete cohorts.',
        r'\IfFileExists{generated/domain_supplement_results.tex}{\input{generated/domain_supplement_results.tex}}{}']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (args.output.parent/'domain_result_facts.json').write_text(json.dumps(facts,indent=2),encoding='utf-8')
    print('COMPLETE_MAIN_TABLE_BUILT',str(args.output))


if __name__=='__main__':
    main()
