"""Write manuscript findings from the complete, validated supplement report."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'fine_preview': 'ViewRank--Sparse',
    'no_core_preview_fine': 'NoCore',
    'scan_preview_fine': 'All-row Scan',
    'interval_preview_fine': 'Independent intervals',
    'adaptive_preview': 'Two-tier controller',
}
DATASETS = ('hot', 'mu')
MIB = 1024**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'results/domain_supplement_v3/evidence_summary_v3/report.json')
    parser.add_argument('--output', type=Path, default=ROOT/'paper/generated/domain_supplement_results.tex')
    parser.add_argument('--capacity-match', type=Path, default=ROOT/'results/domain_supplement_v3/independent_verification_v3/capacity_query_set_match_v2.json')
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    if report['status'] != 'COMPLETE' or report['partial_metrics_admitted']:
        raise ValueError('Manuscript findings require the complete, nonpartial supplement report.')
    rows = {(r['cohort_id'], r['method'], r['block']): r
            for r in report['tables']['method_blocks']}
    if len(rows) != len(report['tables']['method_blocks']):
        raise ValueError('Duplicate method/cohort/block rows in the report.')
    if any(r['block'] != 0 for r in rows.values() if r['group'] in ('B', 'C')):
        raise ValueError('The frozen B and C cohorts each contain exactly one paired block.')

    def row(cohort, method):
        result = rows[cohort, method, 0]
        if result['execution'] != 'paired':
            raise ValueError('Mechanism and budget rows must come from paired cohorts.')
        return result

    def check_cohort(values, budget):
        anchor = values[0]
        fields = ('cohort_id', 'dataset', 'questions', 'population_sha256', 'ks', 'box', 'budget_bytes')
        if any(tuple(r[k] for k in fields) != tuple(anchor[k] for k in fields) for r in values):
            raise ValueError('A manuscript comparison mixes populations or cohorts.')
        if anchor['budget_bytes'] != budget*MIB:
            raise ValueError('Manuscript cohort has the wrong admission budget.')

    def cov(r):
        return 100*r['resolved_all_cut_queries']/r['questions']

    lines = [r'\subsection{Matched mechanism controls}',
        r'Table~\ref{tab:domain-ablation} gives a separately scheduled paired block of each complete workload. All methods use the same original questions, policy domain, cutoffs, and 16\,MiB budget. Fixed-precision controls use the primary\textquotesingle s $10^{-6}$ response tolerance. Total costs include every unresolved request; the controller includes both tiers and all retries.',
        r'\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}',
        r'\caption{Matched complete-workload controls, one block per graph. Resolved coverage requires all three cutoffs of a question.}',
        r'\label{tab:domain-ablation}',
        r'\begin{tabular}{lrrrr}\toprule',
        r'&\multicolumn{2}{c}{HotpotQA}&\multicolumn{2}{c}{MuSiQue}\\',
        r'Method&Time (s)&Done (\%)&Time (s)&Done (\%)\\\midrule']
    for name, label in LABELS.items():
        values = []
        for short in DATASETS:
            r = row('B_own_matched_'+short, name)
            values.extend([f"{r['total_seconds']:.2f}", f'{cov(r):.2f}'])
        lines.append(label+' & '+' & '.join(values)+r'\\')
    for short in DATASETS:
        controls = [row('B_own_matched_'+short, name) for name in LABELS]
        check_cohort(controls, 16)
        if any(r['method_configuration']['atom_tol'] != 1e-6 for r in controls
                if r['method'] != 'adaptive_preview'):
            raise ValueError('A fixed-precision B control is not at the declared fine tolerance.')
    lines.append(r'\bottomrule\end{tabular}\end{table}')
    primary = [row('B_own_matched_'+s, 'fine_preview') for s in DATASETS]
    scan = [row('B_own_matched_'+s, 'scan_preview_fine') for s in DATASETS]
    core = [row('B_own_matched_'+s, 'no_core_preview_fine') for s in DATASETS]
    interval = [row('B_own_matched_'+s, 'interval_preview_fine') for s in DATASETS]
    adaptive = [row('B_own_matched_'+s, 'adaptive_preview') for s in DATASETS]
    lines.append('All-row Scan costs '+
        ' and '.join(f"{a['total_seconds']/p['total_seconds']:.2f}$\\times$" for a,p in zip(scan,primary))+
        ' the primary total on HotpotQA and MuSiQue, respectively. It resolves '+
        ' and '.join(str(a['resolved_all_cut_queries']) for a in scan)+
        ' complete audits, compared with '+
        ' and '.join(str(p['resolved_all_cut_queries']) for p in primary)+
        ' for the primary. This one-block control reports the whole-operator effect of all-row reconstruction together with its audit coverage.')
    lines.append('The recertified truncated-response NoCore control changes total cost by '+
        ' and '.join(f"{100*(a['total_seconds']/p['total_seconds']-1):+.1f}\\%" for a,p in zip(core,primary))+
        ', with complete-audit counts of '+
        ' and '.join(str(a['resolved_all_cut_queries']) for a in core)+
        '. NoCore jointly changes retained responses and their recertified posterior bounds. These are one-block whole-operator comparisons.')
    lines.append('Independent coefficient intervals change total cost by '+
        ' and '.join(f"{100*(a['total_seconds']/p['total_seconds']-1):+.1f}\\%" for a,p in zip(interval,primary))+
        ' on HotpotQA and MuSiQue, respectively, and resolve '+
        ' and '.join(str(a['resolved_all_cut_queries']) for a in interval)+
        ' complete audits, compared with '+
        ' and '.join(str(p['resolved_all_cut_queries']) for p in primary)+
        ' for the primary. Cost and coverage must be read together: this control tests the loss of common polynomial dependence, rather than a cost reduction at matched coverage.')
    changes = [100*(a['total_seconds']/p['total_seconds']-1) for a,p in zip(adaptive,primary)]
    lines.append('The optional fixed two-tier controller has total-cost changes of '+
        ' and '.join(f'{x:+.1f}\\%' for x in changes)+
        ', with '+ ' and '.join(str(a['resolved_all_cut_queries']) for a in adaptive)+
        ' complete audits. Its all-question p95 values are '+
        ' and '.join(f"{a['p95_active_query_seconds']*1000:.1f}" for a in adaptive)+
        ' ms, compared with '+
        ' and '.join(f"{p['p95_active_query_seconds']*1000:.1f}" for p in primary)+
        ' ms for the primary. Fixed fine precision remains the predeclared primary configuration.')

    if not report.get('certificate_capacity_policy') or report['certificate_capacity_policy']['verified_point_witnesses'] is not False:
        raise ValueError('Capacity findings require the separately frozen witness-off study.')
    lines += [r'\subsection{Certificate capacity and process memory}',
        r'Figure~\ref{fig:domain-budget} reports separate witness-free certificate-capacity runs: one paired block per graph at each budget. Success requires all three cutoffs STABLE for an original question. Tables~\ref{tab:domain-main} and~\ref{tab:domain-ablation} instead report complete three-valued audit resolution.',
        r'\begin{figure*}[t]\centering',
        r'\includegraphics[width=\textwidth]{../figures/domain_budget.pdf}',
        r'\caption{Certificate capacity under 1, 4, and 16\,MiB admission budgets, including retained state and admitted family-private numerical work. Every original question remains in each denominator; success requires all three cutoffs STABLE. All actual-point witness search is disabled. One paired block per graph and budget; admission accounting is distinct from operating-system memory.}',
        r'\label{fig:domain-budget}\end{figure*}']
    for budget in (1, 4, 16):
        counts = []
        for short in DATASETS:
            p = row(f'C_certificate_budget{budget}_{short}', 'fine_preview')
            d = row(f'C_certificate_budget{budget}_{short}', 'dp_preview_fine')
            l = row(f'C_certificate_budget{budget}_{short}', 'lift_sparse_fine_preview')
            check_cohort([p,d,l], budget)
            if any(r['verified_point_witnesses'] is not False or r['verified_unstable_cuts'] != 0 or
                    r['every_cut_stable_queries'] != r['resolved_all_cut_queries'] for r in (p,d,l)):
                raise ValueError('Certificate-only findings contain witness-enabled resolution.')
            counts.append(f"{p['every_cut_stable_queries']}/{p['questions']} versus {d['every_cut_stable_queries']} for DensePruned and {l['every_cut_stable_queries']} for sparse lifting")
        if budget == 1:
            lines.append(f'At {budget}\\,MiB, Sparse certifies all cutoffs for '+
                counts[0]+' on HotpotQA; the corresponding MuSiQue counts are '+counts[1]+'.')
    matched = json.loads(args.capacity_match.read_text(encoding='utf-8'))
    if matched['status'] != 'PASS' or len(matched['results']) != 2 or any(
            not r['exactly_same_original_qid_set'] or not r['both_stable_memberships_agree']
            for r in matched['results']):
        raise ValueError('The cross-budget question-set claim requires its independent match report.')
    sizes = [r['intersection'] for r in matched['results']]
    lines.append('At 4\\,MiB, compressed retention certifies exactly the same '+
        str(sizes[0])+' HotpotQA and '+str(sizes[1])+
        ' MuSiQue original questions as fine DensePruned at 16\\,MiB. Independent comparison confirms identical question sets and memberships at every cutoff. '+
        'These are the measured configurations in Fig.~\\ref{fig:domain-budget}. '+
        'The Sparse HotpotQA counts are 905 at 4\\,MiB and 904 at 16\\,MiB; preview rows depend on cache-miss history and need not be nested across budgets. '+
        'All attempted work remains charged.')

    fresh = {(r['cohort_id'],r['method']):r for r in report['tables']['D_fresh_single']}
    if len(fresh) != 6 or any(r['execution'] != 'single' or r['block'] != 0 for r in fresh.values()):
        raise ValueError('D must contain six distinct fresh single-method one-block runs.')
    for r in fresh.values():
        manifest = json.loads(Path(r['manifest_path']).read_text(encoding='utf-8'))
        if manifest['arguments'].get('audit_specs') is not None:
            raise ValueError('Fresh-process manuscript measurements must omit proof-evidence serialization.')
    lines += [r'\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}',
        r'\caption{Fresh single-method processes, one complete run per graph at 16\,MiB with fine responses and no audit serialization. Wall is recorded producer time (seconds); peak is the cumulative process peak working set (MiB) read at block end, including the graph and backend.}',
        r'\label{tab:domain-memory}',
        r'\begin{tabular}{lrrrr}\toprule',
        r'&\multicolumn{2}{c}{HotpotQA}&\multicolumn{2}{c}{MuSiQue}\\',
        r'Method&Wall&Peak&Wall&Peak\\\midrule']
    for name,label in [('fine_preview','ViewRank--Sparse'),('dp_preview_fine','DensePruned'),('lift_sparse_fine_preview','SourceLift, compressed')]:
        values = []
        for short in DATASETS:
            r = fresh[f'D_single_{name}_{short}',name]
            values.extend([f"{r['producer_wall_seconds']:.2f}", f"{r['recorded_process_peak_working_set_bytes']/MIB:.1f}"])
        lines.append(label+' & '+' & '.join(values)+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}',
        r'Table~\ref{tab:domain-memory} uses fresh single-method processes without audit serialization. Producer wall spans source freezing through execution and preceding serialization, excluding command-line/input-metadata parsing and final-summary serialization. The cumulative operating-system peak is sampled at block end. Admission covers retained atoms, live families, and proof work; common graph/backend and response-generation scratch also contribute to process memory.']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('COMPLETE_SUPPLEMENT_TEXT_BUILT',str(args.output))


if __name__ == '__main__':
    main()
