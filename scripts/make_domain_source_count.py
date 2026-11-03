"""Build a descriptive source-count table from the final precomputed report.

Reads only report.json. It never reads question streams, graph/response files,
or invokes the summarizer or a numerical backend. Run after --r-strata reporting.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (('hot', 'hotpot_pool7405', 'HotpotQA'), ('mu', 'musique_pool2417', 'MuSiQue'))
METHODS = ('fine_preview', 'lift_sparse_fine_preview')
BLOCKS = (0, 1, 2)
SOURCE_COUNTS = (1, 2, 3, 4)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_index(values, fields):
    result = {}
    for row in values:
        key = tuple(row[field] for field in fields)
        need(key not in result, 'Duplicate source-count evidence row: '+str(key))
        result[key] = row
    return result


def trend_text(label, values):
    observed = [(r, values[r]['ratio_median']) for r in SOURCE_COUNTS if values[r]['questions']]
    if len(observed) < 2:
        return f'{label} has {len(observed)} nonempty source-count stratum in this workload.'
    numbers = [value for _, value in observed]
    if all(a == b for a, b in zip(numbers, numbers[1:])):
        pattern = 'is constant across the observed source-count strata'
    elif all(a <= b for a, b in zip(numbers, numbers[1:])):
        pattern = 'is nondecreasing across the observed source-count strata'
    elif all(a >= b for a, b in zip(numbers, numbers[1:])):
        pattern = 'is nonincreasing across the observed source-count strata'
    else:
        pattern = 'varies nonmonotonically across the observed source-count strata'
    first, last = observed[0], observed[-1]
    return (f'On {label}, the observed median SourceLift/ViewRank active-cost ratio {pattern}; '
        f'it is {first[1]:.2f} at $r={first[0]}$ and {last[1]:.2f} at $r={last[0]}$.')


def build(report):
    need(report.get('status') == 'COMPLETE' and report.get('partial_metrics_admitted') is False,
        'Source-count text requires the complete nonpartial supplement report.')
    strata = report['tables'].get('r_strata_active_only')
    need(strata, 'No precomputed source-count strata. Generate the final report with --r-strata first.')
    cohort_names = {'A_core_matched_'+short for short, _, _ in DATASETS}
    a_strata = [row for row in strata if row['cohort_id'] in cohort_names]
    need(a_strata and all(row['method'] in METHODS and row['r'] in SOURCE_COUNTS and
        row['block'] in BLOCKS for row in a_strata), 'Unexpected A method, source-count or block; no stratum may be silently omitted.')
    indexed = unique_index(a_strata, ('cohort_id', 'method', 'block', 'r'))
    methods = unique_index([row for row in report['tables']['method_blocks'] if row['cohort_id'] in cohort_names],
        ('cohort_id', 'method', 'block'))
    need(len(methods) == 12, 'Require both A methods and all three blocks for both graphs.')
    facts = {}
    for short, dataset, label in DATASETS:
        cohort = 'A_core_matched_'+short
        anchor = methods[cohort, METHODS[0], 0]
        for method in METHODS:
            for block in BLOCKS:
                row = methods[cohort, method, block]
                need(row['dataset'] == dataset and row['execution'] == 'paired', 'Wrong A dataset/execution.')
                fields = ('questions', 'population_sha256', 'ks', 'box', 'budget_bytes')
                need(all(row[k] == anchor[k] for k in fields), 'A methods/blocks do not share the original population and policy domain.')
                entries = [indexed[cohort, method, block, r] for r in SOURCE_COUNTS
                    if (cohort, method, block, r) in indexed]
                need(all(value['dataset'] == dataset for value in entries), 'Stratum dataset differs from cohort.')
                need(sum(value['questions'] for value in entries) == row['questions'], 'Source-count strata omit original questions.')
                need(math.isclose(sum(value['active_query_seconds'] for value in entries),
                    row['active_query_seconds'], rel_tol=1e-9, abs_tol=1e-7), 'Stratum active sums differ from the all-query block.')
        grouped = {}
        for r in SOURCE_COUNTS:
            entries = [indexed.get((cohort, method, block, r)) for method in METHODS for block in BLOCKS]
            if not any(entries):
                grouped[r] = dict(questions=0, primary_active_seconds=[None]*3,
                    source_lift_active_seconds=[None]*3, raw_paired_ratios=[None]*3,
                    ratio_median=None, ratio_min=None, ratio_max=None,
                    empty_stratum=True)
                continue
            need(all(entries), 'A nonempty stratum is missing a method/block.')
            counts = [row['questions'] for row in entries]
            need(len(set(counts)) == 1 and counts[0] > 0, 'Original stratum counts differ across paired methods or repeats.')
            primary = [indexed[cohort, METHODS[0], b, r] for b in BLOCKS]
            lifted = [indexed[cohort, METHODS[1], b, r] for b in BLOCKS]
            costs = [row['active_query_seconds'] for row in primary+lifted]
            need(all(isinstance(x, (float, int)) and math.isfinite(x) and x > 0 for x in costs),
                'Nonempty strata need positive finite measured active costs.')
            ratios = [other['active_query_seconds']/base['active_query_seconds'] for base, other in zip(primary, lifted)]
            grouped[r] = dict(questions=counts[0], empty_stratum=False,
                primary_active_seconds=[row['active_query_seconds'] for row in primary],
                source_lift_active_seconds=[row['active_query_seconds'] for row in lifted],
                raw_paired_ratios=ratios, ratio_median=median(ratios), ratio_min=min(ratios), ratio_max=max(ratios),
                primary_statuses_by_block=[{key:row[key] for key in
                    ('stable_cuts','verified_unstable_cuts','unknown_cuts','resolved_all_cut_queries')} for row in primary],
                source_lift_statuses_by_block=[{key:row[key] for key in
                    ('stable_cuts','verified_unstable_cuts','unknown_cuts','resolved_all_cut_queries')} for row in lifted])
        facts[dataset] = dict(cohort_id=cohort, original_questions=anchor['questions'],
            population_sha256=anchor['population_sha256'],ks=anchor['ks'],box=anchor['box'],budget_bytes=anchor['budget_bytes'],
            method_configurations={name:methods[cohort,name,0]['method_configuration'] for name in METHODS},
            summary_path=anchor['summary_path'],summary_sha256=anchor['summary_sha256'],
            manifest_path=anchor['manifest_path'],manifest_sha256=anchor['manifest_sha256'],
            strata=grouped,description=trend_text(label, grouped))
    lines = [r'\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}',
        r'\caption{Source-count strata in cohort A. $n$ counts original questions. Lift/VR is the median of three paired summed active-cost ratios, including UNKNOWN. Startup/setup are excluded. Repeated blocks are descriptive.}',
        r'\label{tab:domain-source-count}',r'\begin{tabular}{crrrr}\toprule',
        r'&\multicolumn{2}{c}{HotpotQA}&\multicolumn{2}{c}{MuSiQue}\\',
        r'$r$&$n$&Lift/VR&$n$&Lift/VR\\\midrule']
    for r in SOURCE_COUNTS:
        values = []
        for _, dataset, _ in DATASETS:
            value = facts[dataset]['strata'][r]
            values.extend((str(value['questions']), f"{value['ratio_median']:.2f}" if value['questions'] else '---'))
        lines.append(str(r)+' & '+' & '.join(values)+r'\\')
    endpoints = [[facts[dataset]['strata'][r]['ratio_median'] for _,dataset,_ in DATASETS] for r in (1,4)]
    lines += [r'\bottomrule\end{tabular}\end{table}',
        r'Table~\ref{tab:domain-source-count} retains all source-count strata and original requests. '+
        f'The SourceLift/ViewRank active-cost ratios are {endpoints[1][0]:.2f}/{endpoints[1][1]:.2f} at $r=4$, '+
        f'compared with {endpoints[0][0]:.2f}/{endpoints[0][1]:.2f} for single-source queries (HotpotQA/MuSiQue). '+
        'This descriptive breakdown places the largest gains in the multi-source work addressed by the compiler.']
    return facts, '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'results/domain_supplement_v3/evidence_summary_v3/report.json')
    parser.add_argument('--output', type=Path, default=ROOT/'paper/generated/domain_source_count.tex')
    parser.add_argument('--facts', type=Path, help='Default: domain_source_count_facts.json beside the TeX output.')
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    facts, text = build(report)
    record = dict(version='domain-source-count-v1',report_path=str(args.report),report_sha256=sha(args.report),
        script_sha256=sha(Path(__file__)),blocks=list(BLOCKS),source_counts=list(SOURCE_COUNTS),
        methods=list(METHODS),datasets=facts,
        ratio_scope='For each r and block, divide SourceLift summed active cost by primary summed active cost over the same original queries; then take the median of the three ratios.',
        startup_scope='No startup/setup allocation. Every original query, including UNKNOWN, remains.',
        inference_scope='Three repeated timing blocks are descriptive, not new independent questions. No monotonic trend is assumed.',
        input_scope='Only final report.json; no trace, graph, NPZ or numerical backend access.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding='utf-8')
    facts_path = args.facts or args.output.with_name(args.output.stem+'_facts.json')
    facts_path.parent.mkdir(parents=True, exist_ok=True)
    facts_path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    print('COMPLETE_SOURCE_COUNT_TABLE_BUILT', str(args.output))


if __name__ == '__main__':
    main()
