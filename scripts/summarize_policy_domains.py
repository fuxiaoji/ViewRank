"""Summarize complete fixed policy-domain runs without selecting questions."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(run):
    job, result = read(run/'job_manifest.json'), read(run/'summary.json')
    if result['source_drift'] or result['complete_blocks'] != result['expected_blocks']:
        raise ValueError(f'Incomplete or source-drifted run: {run}')
    n, blocks = job['questions'], result['complete_blocks']
    output = []
    for spec in job['specs']:
        name = spec['name']
        rows = [json.loads(line) for line in (run/name/'questions.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(rows) != n*blocks:
            raise ValueError(f'Wrong complete trace length: {run}/{name}')
        for block in range(blocks):
            current = [row for row in rows if row['block'] == block]
            if [row['qid'] for row in current] != job['original_qids']:
                raise ValueError('Question order changed')
            s = result['blocks'][block]['specs'][name]
            by_r = defaultdict(lambda: dict(questions=0, active_seconds=0., resolved_questions=0, statuses=Counter()))
            tiers = Counter()
            for row in current:
                group = by_r[len(row['original_sources'])]
                statuses = Counter(v['status'] for v in row['answer']['cuts'].values())
                group['questions'] += 1
                group['active_seconds'] += row['active_query_seconds']
                group['resolved_questions'] += not statuses['UNKNOWN']
                group['statuses'].update(statuses)
                tiers[len(row['answer'].get('tiers', []))] += 1
            output.append(dict(run=str(run), dataset=job['dataset'], budget_bytes=job['budget_bytes'],
                questions=n, block=block, method=name, total_seconds=s['total_seconds'],
                p95_ms=s['p95_active_query_seconds']*1000,
                resolved_questions=s['all_cut_queries'].get('resolved',0),
                stable_cuts=s['statuses'].get('STABLE',0), unstable_cuts=s['statuses'].get('UNSTABLE',0),
                unknown_cuts=s['statuses'].get('UNKNOWN',0), stages=s['stages'],cache=s['cache'],
                private_peak_bytes=s['private_memory']['peak_private_retained_numeric_bytes'],
                admission_peak_bytes=s['private_memory']['peak_admission_reserved_numeric_bytes'],
                process_memory=result['blocks'][block]['process_memory'],
                execution=job['execution'], by_source_count=dict(by_r), tiers=dict(tiers)))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    records=[row for run in args.runs for row in summarize(run)]
    groups=defaultdict(list)
    for row in records:
        groups[(row['run'],row['method'])].append(row)
    aggregate=[]
    for (_, method), rows in groups.items():
        first=rows[0]
        aggregate.append({key:first[key] for key in ('run','dataset','budget_bytes','questions','execution')} |
            dict(method=method,blocks=len(rows),median_total_seconds=median(r['total_seconds'] for r in rows),
                median_p95_ms=median(r['p95_ms'] for r in rows),
                resolved_questions=[r['resolved_questions'] for r in rows],
                stable_cuts=[r['stable_cuts'] for r in rows],unstable_cuts=[r['unstable_cuts'] for r in rows],
                unknown_cuts=[r['unknown_cuts'] for r in rows],
                max_private_peak_bytes=max(r['private_peak_bytes'] for r in rows),
                max_admission_peak_bytes=max(r['admission_peak_bytes'] for r in rows)))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(dict(records=records,aggregate=aggregate),indent=2),encoding='utf-8')
    keys=list(aggregate[0])
    with (args.output/'aggregate.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(aggregate)
    for row in aggregate:
        print(row['dataset'],row['method'],round(row['median_total_seconds'],3),
            round(row['median_p95_ms'],2),row['resolved_questions'],row['unknown_cuts'])


if __name__=='__main__':
    main()
