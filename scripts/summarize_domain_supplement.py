"""Stdlib-only summary of completed frozen domain supplement cohorts.

Default phase is metadata-only PLAN. SUMMARIZE requires complete, drift-free
jobs; partial results need explicit --allow-partial and remain visibly pending.
No graph, numerical backend, NumPy, or response evidence is imported or loaded.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


DEFAULT_PROTOCOL = Path('D:/desktop/ICDE 2027/results/domain_supplement_v3/certificate_capacity_v4/protocol_v4.json')
DEFAULT_OUTPUT = Path('D:/desktop/ICDE 2027/results/domain_supplement_v3/evidence_summary_v3')
STATUSES = ('STABLE', 'UNSTABLE', 'UNKNOWN')
STAGES = ('seed_seconds', 'base_seconds', 'prepare_seconds', 'certify_seconds', 'release_seconds')
CSV_FILES = ('cohorts.csv', 'method_blocks.csv', 'A_core_matched.csv',
    'A_replicate_summary.csv', 'B_own_matched.csv', 'C_budget.csv',
    'C_budget_comparisons.csv', 'C_certificate_capacity.csv', 'C_budget16_coverage_references.csv',
    'D_fresh_single.csv', 'r_strata_active_only.csv', 'per_cut_coverage.csv')


class EvidenceError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise EvidenceError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024**2), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def number(value, label):
    need(isinstance(value, (int, float)) and not isinstance(value, bool) and
        math.isfinite(value) and value >= 0, label+' must be a finite nonnegative number.')
    return value


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7)


def population(job):
    return hashlib.sha256(json.dumps({key: job[key] for key in
        ('original_qids', 'original_sources', 'seed_source_sha256')},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def csv_table(path, rows, required=()):
    fields = list(required)
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(',', ':'))
                if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def expected_plan(protocol_path, cohort):
    path = Path(protocol_path).parent/cohort['plan_file']
    need(sha(path) == cohort['plan_sha256'], 'Frozen plan hash mismatch: '+cohort['id'])
    return read(path)


def method_row(cohort, job, summary, block, method, value, provenance):
    n, ks = job['questions'], job['ks']
    need(value['questions'] == n and value['block'] == block, 'Wrong method block/population.')
    spec = next(s for s in job['specs'] if s['name'] == method)
    need(value['spec'] == spec, 'Method configuration differs from manifest.')
    count = {status: value['statuses'].get(status, 0) for status in STATUSES}
    need(set(value['statuses']) <= set(STATUSES) and
        all(isinstance(x, int) and x >= 0 for x in count.values()) and
        sum(count.values()) == n*len(ks), 'Incomplete all-cut status denominator.')
    done = value['all_cut_queries']
    need(sum(done.values()) == n and set(done) <= {'resolved', 'unresolved'} and
        all(isinstance(x,int) and x >= 0 for x in done.values()), 'Incomplete all-query denominator.')
    capacity_only = cohort['group'] == 'C' and cohort.get('verified_point_witnesses') is False
    if capacity_only:
        need(job['verified_point_witnesses'] is False and job['arguments']['witness'] is False,
            'Pure certificate capacity cannot enable actual-point witnesses.')
        need(count['UNSTABLE'] == 0, 'Witness-off capacity contains a verified UNSTABLE result.')
        need(done.get('resolved', 0)*len(ks) <= count['STABLE'] and
            done.get('unresolved', 0) <= count['UNKNOWN'],
            'Witness-off all-STABLE question counts are inconsistent with per-cut counts.')
    for key in STAGES:
        number(value['stages'][key], key)
    active = number(value['active_query_seconds'], 'active_query_seconds')
    total = number(value['total_seconds'], 'total_seconds')
    need(close(sum(value['stages'][key] for key in STAGES), active), 'Exclusive stage sum mismatch.')
    need(close(value['charged_shared_startup_seconds']+value['own_setup_seconds']+active, total), 'TOTAL sum mismatch.')
    memory = value['private_memory']
    for key in ('peak_private_retained_numeric_bytes', 'peak_admission_reserved_numeric_bytes'):
        number(memory[key], key)
    need(memory['peak_private_retained_numeric_bytes'] <= memory['peak_admission_reserved_numeric_bytes'] <= job['budget_bytes'],
        'Private peak budget identity failed.')
    return dict(cohort_id=cohort['id'], group=cohort['group'], dataset=job['dataset'],
        block=block, method=method, execution=job['execution'], questions=n,
        ks=ks, box=job['box'], budget_bytes=job['budget_bytes'],
        method_configuration=spec, population_sha256=population(job),
        total_seconds=total, active_query_seconds=active,
        p95_active_query_seconds=number(value['p95_active_query_seconds'], 'p95'),
        charged_shared_startup_seconds=value['charged_shared_startup_seconds'],
        own_setup_seconds=value['own_setup_seconds'],
        **{key: value['stages'][key] for key in STAGES},
        stable_cuts=count['STABLE'], verified_unstable_cuts=count['UNSTABLE'],
        unknown_cuts=count['UNKNOWN'], total_cuts=n*len(ks),
        resolved_all_cut_queries=done.get('resolved', 0),
        unresolved_all_cut_queries=done.get('unresolved', 0),
        resolved_all_cut_fraction=done.get('resolved', 0)/n,
        verified_point_witnesses=job['verified_point_witnesses'],
        every_cut_stable_queries=done.get('resolved', 0) if capacity_only else None,
        every_cut_stable_fraction=done.get('resolved', 0)/n if capacity_only else None,
        outcome_scope=('Certificate capacity only: all cutoffs STABLE, no point witnesses.' if capacity_only else
            'Full operator: all cutoffs STABLE or verified UNSTABLE; UNKNOWN is unresolved.'),
        cache=value['cache'], routes=value['routes'],
        peak_private_retained_numeric_bytes=memory['peak_private_retained_numeric_bytes'],
        peak_admission_reserved_numeric_bytes=memory['peak_admission_reserved_numeric_bytes'],
        terminal_private_retained_numeric_bytes=memory['total_private_retained_numeric_bytes'],
        terminal_admission_reserved_numeric_bytes=memory['admission_reserved_numeric_bytes'],
        summary_path=provenance['summary_path'], summary_sha256=provenance['summary_sha256'],
        manifest_path=provenance['manifest_path'], manifest_sha256=provenance['manifest_sha256'],
        cost_population='Every original query, including UNKNOWN. No completion-subset filtering.',
        memory_scope='Private retained/admission numeric accounting. No paired OS peak is attributed to a method.')


def load_cohort(cohort, plan, expected_sources, expected_inputs):
    """Return all expected jobs visibly, but admit metrics only after completion."""
    path = Path(cohort['output'])
    state = dict(cohort_id=cohort['id'], group=cohort['group'], dataset=cohort['dataset'],
        output=str(path), expected_blocks=cohort['blocks'], expected_questions=cohort['questions'],
        expected_methods=cohort['specs'], status='PENDING', reason='summary.json is absent')
    summary_path, manifest_path = path/'summary.json', path/'job_manifest.json'
    if not summary_path.is_file():
        return dict(state=state, methods=[], job=None)
    try:
        result = read(summary_path)
    except (OSError, json.JSONDecodeError) as error:
        state['reason'] = 'Final summary unreadable: '+str(error)
        return dict(state=state, methods=[], job=None)
    try:
        if result.get('complete_blocks', 0) < cohort['blocks']:
            state['reason'] = 'Expected blocks have not completed.'
            return dict(state=state, methods=[], job=None)
        need(result.get('complete_blocks') == result.get('expected_blocks') == cohort['blocks'], 'Unexpected/incomplete block count.')
        need(result.get('source_drift') == [], 'source_drift is not an explicit empty list.')
        need(manifest_path.is_file(), 'Completed summary has no manifest.')
        job = read(manifest_path)
        need(result['job_manifest_sha256'] == sha(manifest_path), 'Manifest hash mismatch.')
        keys = ('dataset', 'questions', 'blocks', 'box', 'ks', 'budget_bytes',
            'base_tolerance', 'maximum_subdivision_depth', 'verified_point_witnesses',
            'scan_backend', 'cpu', 'priority', 'execution', 'specs',
            'original_qids', 'original_sources', 'seed_source_sha256', 'method_order_policy')
        for key in keys:
            need(job.get(key) == plan.get(key), 'Frozen plan mismatch: '+key)
        if cohort['group'] == 'C' and cohort.get('verified_point_witnesses') is False:
            need(cohort['id'].startswith('C_certificate_budget') and cohort['blocks'] == 1 and
                job['budget_bytes'] in (1024**2, 4*1024**2, 16*1024**2) and
                job['verified_point_witnesses'] is False and '--no-witness' in cohort['args'],
                'Capacity cohort identity, one-block budget, or witness-off flag differs.')
        need(job['source_sha256'] == expected_sources, 'Source hash set differs from frozen cohort.')
        need(job['input_sha256'] == expected_inputs, 'Input hash set differs from frozen cohort.')
        need(len(job['original_qids']) == len(set(job['original_qids'])) == cohort['questions'], 'Original query count/uniqueness mismatch.')
        need(len(job['original_sources']) == len(job['seed_source_sha256']) == cohort['questions'], 'Original source/seed count mismatch.')
        need(len(result['blocks']) == cohort['blocks'], 'Missing block payloads.')
        need({b['block'] for b in result['blocks']} == set(range(cohort['blocks'])), 'Block IDs differ.')
        provenance = dict(summary_path=str(summary_path), summary_sha256=sha(summary_path),
            manifest_path=str(manifest_path), manifest_sha256=sha(manifest_path),
            input_sha256=job['input_sha256'], source_drift=result['source_drift'],
            population_sha256=population(job), block_files=[], method_files=[])
        rows = []
        names = {spec['name'] for spec in job['specs']}
        for block in sorted(result['blocks'], key=lambda x: x['block']):
            b = block['block']; block_path = path/f'block_{b}_summary.json'
            need(block_path.is_file() and read(block_path) == block, 'Block summary file/payload differs.')
            scope = 'Single method fresh process' if job['execution'] == 'single' else 'COMBINED cohort'
            need(block['execution'] == job['execution'] and block['process_memory_scope'] == scope, 'OS memory scope differs.')
            need(set(block['specs']) == names, 'Missing method block summaries.')
            provenance['block_files'].append(dict(path=str(block_path), sha256=sha(block_path)))
            for method, value in block['specs'].items():
                method_path = path/method/f'block_{b}_summary.json'
                need(method_path.is_file() and read(method_path) == value, 'Method summary file/payload differs.')
                provenance['method_files'].append(dict(path=str(method_path), sha256=sha(method_path)))
                rows.append(method_row(cohort, job, result, b, method, value, provenance))
        state.update(status='COMPLETE', reason='', complete_blocks=result['complete_blocks'],
            summary_sha256=provenance['summary_sha256'], manifest_sha256=provenance['manifest_sha256'])
        return dict(state=state, methods=rows, job=job, result=result, provenance=provenance)
    except (EvidenceError, KeyError, TypeError, ValueError, OSError) as error:
        state.update(status='INVALID', reason=str(error))
        return dict(state=state, methods=[], job=None)


def paired(rows, group):
    out = []
    by_block = defaultdict(dict)
    for row in rows:
        if row['group'] == group:
            by_block[(row['cohort_id'], row['block'])][row['method']] = row
    for (cohort, block), methods in sorted(by_block.items()):
        primary = methods['fine_preview']
        for name, other in sorted(methods.items()):
            if name == 'fine_preview':
                continue
            need(primary['population_sha256'] == other['population_sha256'], 'Paired populations differ.')
            value = dict(cohort_id=cohort, dataset=primary['dataset'], block=block,
                budget_bytes=primary['budget_bytes'], questions=primary['questions'],
                primary='fine_preview', comparison=name, ks=primary['ks'], box=primary['box'],
                population_sha256=primary['population_sha256'],
                comparison_over_primary_total_ratio=other['total_seconds']/primary['total_seconds'],
                comparison_over_primary_p95_ratio=other['p95_active_query_seconds']/primary['p95_active_query_seconds'],
                aggregate_coverage_counts_equal=(primary['stable_cuts'], primary['verified_unstable_cuts'], primary['unknown_cuts'], primary['resolved_all_cut_queries']) ==
                    (other['stable_cuts'], other['verified_unstable_cuts'], other['unknown_cuts'], other['resolved_all_cut_queries']),
                coverage_match_scope='Aggregate counts only; per-question outcome identity is not assumed.',
                ratio_scope='Raw all-query costs, not a conditional speed claim; compare coverage alongside the ratios.')
            for prefix, row in (('primary', primary), ('comparison', other)):
                for field in ('total_seconds', 'active_query_seconds', 'p95_active_query_seconds',
                    'stable_cuts', 'verified_unstable_cuts', 'unknown_cuts',
                    'resolved_all_cut_queries', 'unresolved_all_cut_queries'):
                    value[prefix+'_'+field] = row[field]
                value[prefix+'_configuration'] = row['method_configuration']
                if group == 'C':
                    value[prefix+'_every_cut_stable_queries'] = row['every_cut_stable_queries']
                    value[prefix+'_every_cut_stable_fraction'] = row['every_cut_stable_fraction']
            if group == 'C':
                value['outcome_scope'] = primary['outcome_scope']
            value.update(summary_path=primary['summary_path'], summary_sha256=primary['summary_sha256'],
                manifest_path=primary['manifest_path'], manifest_sha256=primary['manifest_sha256'])
            out.append(value)
    return out


def a_replicates(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row['group'] == 'A':
            grouped[(row['dataset'], row['method'])].append(row)
    out = []
    for (dataset, method), reps in sorted(grouped.items()):
        values = dict(dataset=dataset, method=method, blocks=len(reps),
            distinct_original_queries=reps[0]['questions'],
            budget_bytes=reps[0]['budget_bytes'],ks=reps[0]['ks'],box=reps[0]['box'],
            method_configuration=reps[0]['method_configuration'],
            repeated_query_executions=sum(r['questions'] for r in reps),
            coverage_by_block=[{k:r[k] for k in ('block','stable_cuts','verified_unstable_cuts','unknown_cuts','resolved_all_cut_queries')} for r in reps],
            inference_scope='Repeated timing blocks reuse the original questions; no extra independent samples or significance claim.',
            p95_scope='Summaries of block-level p95 values, never a pooled p95.')
        for key in ('total_seconds', 'p95_active_query_seconds'):
            vals = [r[key] for r in reps]
            values.update({key+'_mean':statistics.mean(vals), key+'_min':min(vals),
                key+'_max':max(vals), key+'_stdev':statistics.stdev(vals) if len(vals)>1 else None})
        out.append(values)
    return out


def fresh_single(loaded):
    out = []
    for item in loaded:
        if item['state']['group'] != 'D' or item['state']['status'] != 'COMPLETE':
            continue
        need(item['job']['execution'] == 'single' and len(item['methods']) == 1, 'Fresh single must contain one method and one block.')
        row = dict(item['methods'][0]); block = item['result']['blocks'][0]
        row.update(producer_wall_seconds=number(item['result']['producer_wall_seconds'], 'producer_wall_seconds'),
            recorded_process_peak_working_set_bytes=number(block['process_memory']['process_peak_working_set_bytes'], 'peak working set'),
            recorded_process_working_set_bytes=block['process_memory']['process_working_set_bytes'],
            recorded_process_private_commit_bytes=block['process_memory']['process_private_commit_bytes'],
            os_memory_scope='One fresh single-method process including shared graph/backend; NOT paired COMBINED RSS.',
            os_peak_observation='Cumulative process peak working set sampled at block end, as recorded by the producer.')
        out.append(row)
    return out


def reference16(rows):
    fields = ('cohort_id','dataset','block','method','execution','questions','ks','box',
        'budget_bytes','method_configuration','population_sha256','stable_cuts',
        'verified_unstable_cuts','unknown_cuts','total_cuts','resolved_all_cut_queries',
        'unresolved_all_cut_queries','resolved_all_cut_fraction','summary_path','summary_sha256',
        'manifest_path','manifest_sha256')
    out = []
    for row in rows:
        if row['group'] == 'A' or row['group'] == 'B16_REFERENCE' and row['method'] == 'dp_preview_fine':
            value = {key:row[key] for key in fields}
            value['comparison_scope'] = 'Historical full-operator 16MiB coverage artifact only. A primary/SparseLift and original-main DP are different cohorts; no absolute times are pooled. Witness-enabled resolution is never a reference point in the new witness-off certificate-capacity study.'
            out.append(value)
    return out


def percentile95(values):
    data = sorted(values); point = .95*(len(data)-1); lower = math.floor(point); upper = math.ceil(point)
    return data[lower]+(point-lower)*(data[upper]-data[lower])


def row_strata(loaded):
    """Optional post-timing streaming pass; excludes every startup/setup share."""
    out, per_cut, files = [], [], []
    for item in loaded:
        if item['state']['status'] != 'COMPLETE':
            continue
        job = item['job']; n = job['questions']; folder = Path(item['state']['output'])
        lookup = {(r['method'],r['block']):r for r in item['methods']}
        for spec in job['specs']:
            name = spec['name']; path = folder/name/'questions.jsonl'; h = hashlib.sha256()
            groups = defaultdict(lambda:dict(active=[],status=Counter(),resolved=0))
            cuts = defaultdict(Counter); block_active = Counter(); block_statuses = defaultdict(Counter)
            block_resolved = Counter(); seen = 0
            with path.open('rb') as handle:
                for raw in handle:
                    h.update(raw); row = json.loads(raw); b, qi = divmod(seen,n)
                    need((row['block'],row['question_index'],row['qid']) == (b,qi,job['original_qids'][qi]), 'Optional row scan lost original order.')
                    need(row['original_sources'] == job['original_sources'][qi], 'Optional row source identity differs.')
                    need(row['seed_source_sha256'] == job['seed_source_sha256'][qi], 'Optional row seed/source fingerprint differs.')
                    need(set(row['answer']['cuts']) == {str(k) for k in job['ks']}, 'Optional row lost cut outputs.')
                    active = number(row['active_query_seconds'], 'row active time')
                    r = len(row['original_sources']); bucket = groups[(b,r)]
                    statuses = [row['answer']['cuts'][str(k)]['status'] for k in job['ks']]
                    need(set(statuses) <= set(STATUSES), 'Optional row has invalid status.')
                    bucket['active'].append(active); bucket['status'].update(statuses)
                    bucket['resolved'] += 'UNKNOWN' not in statuses
                    block_statuses[b].update(statuses); block_resolved[b] += 'UNKNOWN' not in statuses
                    for k in job['ks']:cuts[(b,k)][row['answer']['cuts'][str(k)]['status']] += 1
                    block_active[b] += active; seen += 1
            need(seen == n*job['blocks'], 'Optional row count is incomplete.')
            for b in range(job['blocks']):
                expected=lookup[(name,b)]
                need(close(block_active[b],expected['active_query_seconds']), 'Optional all-row active sum differs from summary.')
                need(tuple(block_statuses[b][s] for s in STATUSES) ==
                    (expected['stable_cuts'],expected['verified_unstable_cuts'],expected['unknown_cuts']),
                    'Optional all-row status counts differ from summary.')
                need(block_resolved[b] == expected['resolved_all_cut_queries'],
                    'Optional all-row resolved-query count differs from summary.')
            files.append(dict(path=str(path),sha256=h.hexdigest(),rows=seen))
            for (b,r), data in sorted(groups.items()):
                out.append(dict(cohort_id=item['state']['cohort_id'],dataset=job['dataset'],method=name,
                    block=b,r=r,questions=len(data['active']),active_query_seconds=sum(data['active']),
                    mean_active_query_seconds=statistics.mean(data['active']),p95_active_query_seconds=percentile95(data['active']),
                    stable_cuts=data['status']['STABLE'],verified_unstable_cuts=data['status']['UNSTABLE'],
                    unknown_cuts=data['status']['UNKNOWN'],resolved_all_cut_queries=data['resolved'],
                    cost_scope='All original rows in this stratum. Active query cost only; no startup/setup allocation or completed-only filtering.'))
            for (b,k), count in sorted(cuts.items()):
                per_cut.append(dict(cohort_id=item['state']['cohort_id'],dataset=job['dataset'],method=name,
                    block=b,k=k,questions=n,stable=count['STABLE'],verified_unstable=count['UNSTABLE'],unknown=count['UNKNOWN']))
    return out, per_cut, files


def plan_metadata(protocol_path, protocol, output, rows_requested):
    return dict(mode='PLAN_ONLY',reporter=str(Path(__file__).resolve()),reporter_sha256=sha(Path(__file__)),
        protocol=str(protocol_path),protocol_sha256=sha(protocol_path),output=str(output),
        expected_cohorts=[{key:c[key] for key in ('id','group','dataset','questions','blocks','execution','output')} for c in protocol['cohorts']],
        tables=list(CSV_FILES),default_reads='Only completed summary/manifest/block-summary JSON files; no questions, graph, NPZ or PPR.',
        expected_method_block_rows=sum(c['blocks']*len(c['specs']) for c in protocol['cohorts']),
        expected_same_block_comparisons={g:sum(c['blocks']*(len(c['specs'])-1) for c in protocol['cohorts'] if c['group']==g) for g in ('A','B','C')},
        optional_rows=rows_requested,completion_gate=f"All {len(protocol['cohorts'])} cohorts completed, expected blocks, explicit empty source_drift and frozen source/input/population/configuration. Otherwise explicit PENDING/INVALID ledger.",
        capacity_policy=protocol.get('capacity_policy'),
        numerical_or_result_aggregation_executed=False)


def summarize(args, protocol):
    loaded = []
    for cohort in protocol['cohorts']:
        loaded.append(load_cohort(cohort,expected_plan(args.protocol,cohort),protocol['source_sha256'],cohort['input_sha256']))
    # Original B16 DP reference has its own cohort and source-hash set.
    main_protocol = read(protocol['required_main_completion']['protocol'])
    expected_main_sources = {k:v for k,v in protocol['source_sha256'].items()
        if k != protocol['wrapper_tracking']['relative_source_path']}
    refs = []
    for original in main_protocol['cohorts']:
        if original['phase'] != 'main':
            continue
        plan = original['plan']; dataset = plan['dataset']
        cohort = dict(id='B16_reference_original_main_'+dataset,group='B16_REFERENCE',
            output=plan['arguments']['output'],dataset=dataset,questions=plan['questions'],
            blocks=plan['blocks'],specs=[s['name'] for s in plan['specs']])
        input_hashes = next(c['input_sha256'] for c in protocol['cohorts'] if c['dataset']==dataset)
        item=load_cohort(cohort,plan,expected_main_sources,input_hashes)
        if item['state']['status']=='COMPLETE':
            frozen_a=expected_plan(args.protocol,next(c for c in protocol['cohorts'] if c['group']=='A' and c['dataset']==dataset))
            aligned=(population(item['job'])==population(frozen_a) and
                item['job']['ks']==frozen_a['ks'] and item['job']['box']==frozen_a['box'] and
                item['job']['budget_bytes']==frozen_a['budget_bytes']==16*1024**2)
            if not aligned:
                item['state'].update(status='INVALID',reason='B16 reference population/ks/box/budget differs from frozen A cohort.')
                item['methods']=[]
        refs.append(item)
    states = [item['state'] for item in loaded]
    reference_states = [item['state'] for item in refs]
    unfinished = [s for s in states+reference_states if s['status'] != 'COMPLETE']
    invalid = [s for s in unfinished if s['status'] == 'INVALID']
    admitted = not unfinished or args.allow_partial
    rows = [r for item in loaded for r in item['methods']] if admitted else []
    ref_rows = [r for item in refs for r in item['methods']] if admitted else []
    a = paired(rows,'A'); b = paired(rows,'B'); c = paired(rows,'C')
    strata, per_cut, row_files = row_strata(loaded) if args.r_strata and admitted else ([],[],[])
    # Reference cohorts are exported as coverage only. Their cost fields never
    # enter A/B/C comparisons, averages or fresh-process memory tables.
    capacity = [r for r in rows if r['group']=='C']
    tables = dict(method_blocks=rows,A_core_matched=a,A_replicate_summary=a_replicates(rows),
        B_own_matched=b,C_budget=capacity,C_budget_comparisons=c,C_certificate_capacity=capacity,
        C_budget16_coverage_references=reference16(rows+ref_rows),
        D_fresh_single=fresh_single(loaded) if admitted else [],r_strata_active_only=strata,
        per_cut_coverage=per_cut)
    status = 'COMPLETE' if not unfinished else ('INVALID' if invalid else 'PENDING')
    report = dict(version='domain-supplement-evidence-v4',status=status,
        generated_utc=datetime.now(timezone.utc).isoformat(),reporter_sha256=sha(Path(__file__)),
        protocol_path=str(args.protocol),protocol_sha256=sha(args.protocol),
        cohorts=states,reference_cohorts=reference_states,partial_metrics_admitted=bool(unfinished and admitted),
        unavailable_metrics_reason=None if admitted else 'No numerical comparison admitted until every required cohort is complete and valid.',
        all_original_query_policy='Costs and coverage include every original question and cut, including UNKNOWN. No completed-subset filter.',
        repeat_policy='A has three timing replicas of the same original questions, not three independent populations. Block-level p95 is never pooled.',
        paired_memory_policy='No COMBINED paired RSS is attributed to a method. Fresh-single OS peaks are in D only.',
        budget16_reference_policy='Historical full-operator coverage artifact only, never used for certificate capacity. Cohorts differ; absolute times are never pooled.',
        certificate_capacity_policy=protocol.get('capacity_policy'),
        optional_r_strata='Active query work only, no shared startup/setup allocation; disabled by default.',
        provenance=[item['provenance'] for item in loaded+refs if item['state']['status']=='COMPLETE'],
        optional_row_file_sha256=row_files,tables=tables)
    args.output.mkdir(parents=True,exist_ok=True)
    csv_table(args.output/'cohorts.csv',states+reference_states,('cohort_id','group','dataset','status','reason'))
    for name, values in tables.items():
        csv_table(args.output/(name+'.csv'),values,('cohort_id','dataset','method','block'))
    write(args.output/'report.json',report)
    print(json.dumps(dict(status=status,completed=sum(s['status']=='COMPLETE' for s in states),
        expected=len(states),pending_or_invalid=[s['cohort_id'] for s in unfinished],output=str(args.output))))
    return 0 if status=='COMPLETE' or args.allow_partial and not invalid else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('plan','summarize'),default='plan')
    p.add_argument('--protocol',type=Path,default=DEFAULT_PROTOCOL)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--plan-output',type=Path,help='Optional small metadata plan file; does not read running results.')
    p.add_argument('--allow-partial',action='store_true',help='Admit completed cohorts with all PENDING/INVALID entries still explicit.')
    p.add_argument('--r-strata',action='store_true',help='After timing only: stream all original question rows for active-only r strata and per-k coverage.')
    args = p.parse_args(); protocol = read(args.protocol)
    if args.phase == 'plan':
        result = plan_metadata(args.protocol,protocol,args.output,args.r_strata)
        if args.plan_output:
            args.plan_output.parent.mkdir(parents=True,exist_ok=True);write(args.plan_output,result)
        print(json.dumps(result,indent=2));return
    raise SystemExit(summarize(args,protocol))


if __name__ == '__main__':
    main()
