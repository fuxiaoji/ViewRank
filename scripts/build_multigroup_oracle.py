"""Independent actual-column cold-PI references for the sixteen two-group views.

No ranking engine, source core, response cache, or gold labels are imported.
Raw mention/weak/self subdistributions are weighted and each complete source
column is normalized by its measured raw sum. Cold PI uses this actual view.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import reduce
from math import gcd
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

GRID = tuple((m, w) for m in (1., 2., 3., 4.) for w in (.25, .5, 1., 2.))
KS = (2, 5, 10)
INPUT_FILES = ('manifest.json', 'queries.jsonl', 'graph.npz', 'mention_subdistribution.npz')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def public_query(q, index):
    return ({k: q[k] for k in ('qid', 'sources', 'seed_indices', 'seed_values')} |
            {'original_question_index': q.get('original_question_index', index)})


def query_sha(q):
    return hashlib.sha256(json.dumps(q, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_package(folder):
    manifest = read(folder / 'manifest.json')
    hashes = {name: sha(folder / name) for name in INPUT_FILES}
    for name in INPUT_FILES[1:]:
        spec = manifest['files'][name]
        if spec['sha256'] != hashes[name] or spec['bytes'] != (folder / name).stat().st_size:
            raise ValueError('Package payload differs from manifest: ' + name)
    raw_queries = lines(folder / 'queries.jsonl')
    queries = [public_query(q, i) for i, q in enumerate(raw_queries)]
    P = sparse.load_npz(folder / 'graph.npz').tocsc()
    M = sparse.load_npz(folder / 'mention_subdistribution.npz').tocsc()
    P.sum_duplicates(); M.sum_duplicates(); P.sort_indices(); M.sort_indices()
    if P.shape != M.shape or P.shape != (manifest['nodes'], manifest['nodes']):
        raise ValueError('Graph dimensions differ')
    if np.any(P.data < 0) or np.any(M.data < 0):
        raise ValueError('Negative graph/mention weights')
    if np.max(abs(np.asarray(P.sum(axis=0)).ravel() - 1)) > 1e-12:
        raise ValueError('Base graph is not column stochastic')
    if np.any(M.diagonal() != 0):
        raise ValueError('Mention must exclude self loops')
    return manifest, hashes, queries, P, M


def source_parts(P, M, sources):
    """Align raw mention weights to base column entries, deriving weak by row."""
    parts = []
    for source in sources:
        lo, hi = P.indptr[source:source+2]
        ml, mh = M.indptr[source:source+2]
        rows, original = P.indices[lo:hi], P.data[lo:hi]
        mention = np.zeros(len(rows))
        pos = np.searchsorted(rows, M.indices[ml:mh])
        if np.any(pos >= len(rows)) or np.any(rows[pos] != M.indices[ml:mh]):
            raise ValueError('Mention contains a non-base edge')
        mention[pos] = M.data[ml:mh]
        self_loop = np.where(rows == source, original, 0.)
        weak = original - mention - self_loop
        if np.any(weak < 0):
            raise ValueError('Raw subdistribution has negative weak edge')
        parts.append(dict(source=int(source), start=int(lo), end=int(hi), rows=rows,
                          original=original, mention=mention, weak=weak, self_loop=self_loop))
    return parts


def source_metadata(P, M, query):
    parts = source_parts(P, M, query['sources'])
    masses, atom_keys, source_map, group_map = [], [], [], []
    for i, p in enumerate(parts):
        group_masses = [float(p[name].sum()) for name in ('mention', 'weak')]
        masses.append(group_masses)
        for g, name in enumerate(('mention', 'weak')):
            if np.any(p[name] - group_masses[g] * p['original'] != 0):
                atom_keys.append([g, p['source']]); source_map.append(i); group_map.append(g)
    return dict(group_masses=masses, atom_keys=atom_keys,
                atom_source_map=source_map, atom_group_map=group_map,
                compiled_source_count=len(parts), compiled_atom_count=len(atom_keys))


def actual_view(P, M, sources, weights):
    view = P.copy()
    m, w = weights
    for p in source_parts(P, M, sources):
        # Deliberately avoid the engine's low-rank coefficient formula.
        raw = m*p['mention'] + w*p['weak'] + p['self_loop']
        total = float(raw.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError('Actual raw column has nonpositive/invalid sum')
        view.data[p['start']:p['end']] = raw / total
    return view


def exact_column_parts(P, M, sources):
    """Dyadic integer input columns; weak=P-mention-self in exact arithmetic."""
    result=[]
    for p in source_parts(P,M,sources):
        ratios=[[float(x).as_integer_ratio() for x in p[name]] for name in ('original','mention')]
        exponent=max((den.bit_length()-1 for group in ratios for _,den in group), default=0)
        integers=[[num << (exponent-(den.bit_length()-1)) for num,den in group] for group in ratios]
        original,mention=integers
        self_loop=[value if row == p['source'] else 0 for row,value in zip(p['rows'],original)]
        weak=[v-m-s for v,m,s in zip(original,mention,self_loop)]
        if any(v<0 for v in weak):
            raise ValueError('Exact dyadic input weak subdistribution is negative')
        result.append((p['source'],p['rows'].tolist(),mention,weak,self_loop))
    return result


def exact_operator_signature(parts, weights):
    """Projective primitive integers identify exact normalized raw columns.

    All grid weights have denominator dividing four. The signature is for
    mathematical normalized views of the original float input rationals,
    avoiding any equality decision based on rounded floating output bytes.
    """
    m4,w4=(round(4*v) for v in weights)
    h=hashlib.sha256()
    for source,rows,mention,weak,self_loop in parts:
        raw=[m4*a+w4*b+4*c for a,b,c in zip(mention,weak,self_loop)]
        divisor=reduce(gcd,raw,0)
        if divisor<=0:
            raise ValueError('Exact raw column has no positive mass')
        primitive=[v//divisor for v in raw]
        h.update(json.dumps([source,rows,primitive],separators=(',',':')).encode())
    return h.hexdigest()


def float_column_signature(view, sources):
    h=hashlib.sha256()
    for source in sources:
        lo,hi=view.indptr[source:source+2]
        h.update(view.indices[lo:hi].astype('<i8').tobytes())
        h.update(view.data[lo:hi].astype('<f8').tobytes())
    return h.hexdigest()


def solve_reference(view, seed, alpha, tolerance):
    scores = seed.copy()
    for iteration in range(10000):
        new = alpha*(view@scores) + (1-alpha)*seed
        diff = float(np.linalg.norm(new-scores, 1))
        scores = new
        if diff <= tolerance:
            break
    else:
        raise RuntimeError('Independent actual-view PI did not converge')
    residual = float(np.linalg.norm(scores-alpha*(view@scores)-(1-alpha)*seed, 1))
    if residual > tolerance:
        raise RuntimeError('Actual-view residual exceeds tolerance')
    return scores, iteration+1, residual


def run_dataset(args, dataset):
    folder = Path(args.input_root).resolve()/dataset
    output = Path(args.output).resolve()/dataset
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Use a new empty oracle output directory: ' + str(output))
    manifest, inputs, all_queries, P, M = load_package(folder)
    queries = all_queries[:args.queries or None]
    if args.queries and len(queries) != args.queries:
        raise ValueError('Requested original prefix is unavailable')
    output.mkdir(parents=True, exist_ok=True)
    alpha = float(manifest['alpha'])
    write(output/'frozen_input.json', dict(input_root=str(folder), input_sha256=inputs,
        public_queries=queries, policy_grid=GRID, oracle_script_sha256=sha(__file__),
        frozen_before_reference_solves=True))
    max_residual = 0.; stages = dict(assembly_seconds=0., solve_seconds=0., selection_seconds=0.)
    iterations = []; started = perf_counter()
    exact_counts=[]; byte_counts=[]; changed={str(k):0 for k in KS}; identity_roundoff=0.
    with (output/'references.jsonl').open('w', encoding='utf-8') as handle, (output/'operators.jsonl').open('w', encoding='utf-8') as operators:
        for qi, q in enumerate(queries):
            seed = np.zeros(P.shape[0]); seed[q['seed_indices']] = q['seed_values']
            if len(set(q['sources'])) != len(q['sources']) or abs(seed.sum()-1) > 1e-12:
                raise ValueError('Invalid seed/source inputs')
            metadata = source_metadata(P, M, q)
            exact_parts=exact_column_parts(P,M,q['sources'])
            query_records=[]
            for ti, weights in enumerate(GRID):
                start = perf_counter(); view = actual_view(P, M, q['sources'], weights)
                stages['assembly_seconds'] += perf_counter()-start
                start = perf_counter(); scores, steps, residual = solve_reference(view, seed, alpha, args.residual_tol)
                stages['solve_seconds'] += perf_counter()-start
                start = perf_counter(); top = np.lexsort((np.arange(P.shape[0]), -scores))[:11]
                stages['selection_seconds'] += perf_counter()-start
                record = dict(qid=q['qid'], question_index=qi, original_question_index=q['original_question_index'],
                    target_index=ti, requested_group_weights=weights, original_sources=q['sources'],
                    seed_source_sha256=query_sha(q), **metadata,
                    exact_normalized_operator_sha256=exact_operator_signature(exact_parts,weights),
                    floating_source_columns_sha256=float_column_signature(view,q['sources']),
                    reference_top11=top.tolist(), reference_top11_scores=scores[top].tolist(),
                    actual_l1_residual=residual, posterior_l1_error_bound=residual/(1-alpha), iterations=steps)
                handle.write(json.dumps(record, allow_nan=False)+'\n')
                query_records.append(record)
                if weights==(1.,1.):
                    for source in q['sources']:
                        lo,hi=P.indptr[source:source+2]
                        identity_roundoff=max(identity_roundoff,float(np.max(abs(view.data[lo:hi]-P.data[lo:hi]),initial=0)))
                iterations.append(steps); max_residual=max(max_residual,residual)
            identity=query_records[2]
            changed_targets={str(k):[r['target_index'] for r in query_records if set(r['reference_top11'][:k])!=set(identity['reference_top11'][:k])] for k in KS}
            exact_count=len({r['exact_normalized_operator_sha256'] for r in query_records})
            byte_count=len({r['floating_source_columns_sha256'] for r in query_records})
            exact_counts.append(exact_count); byte_counts.append(byte_count)
            for k in KS:
                changed[str(k)]+=bool(changed_targets[str(k)])
            operators.write(json.dumps(dict(qid=q['qid'],question_index=qi,logical_targets=16,
                exact_distinct_normalized_operators=exact_count,floating_byte_distinct_source_columns=byte_count,
                identity_target_index=2,identity_exact_operator_sha256=identity['exact_normalized_operator_sha256'],
                target_exact_operator_sha256=[r['exact_normalized_operator_sha256'] for r in query_records],
                strict_literal_membership_changed_target_indices=changed_targets,
                distinction_scope='Exact rational primitive columns use original P/M float values as dyadic rationals and weak=P-M-self. SHA256 collision resistance assumed. Floating byte counts are separate and may reflect numerical roundoff.'),allow_nan=False)+'\n')
            if (qi+1)%10 == 0 or qi+1 == len(queries):
                handle.flush()
                print(json.dumps(dict(dataset=dataset, completed_questions=qi+1, elapsed_seconds=perf_counter()-started)), flush=True)
    summary = dict(protocol='two-group-independent-raw-normalized-cold-pi-v1', dataset=dataset,
        questions=len(queries), targets=len(queries)*len(GRID), policy_grid=GRID, ks=KS, alpha=alpha,
        input_root=str(folder), input_sha256=inputs, references_sha256=sha(output/'references.jsonl'),
        operators_sha256=sha(output/'operators.jsonl'),
        frozen_input_sha256=sha(output/'frozen_input.json'), script_sha256=sha(__file__),
        residual_tolerance=args.residual_tol, max_actual_l1_residual=max_residual,
        max_iterations=max(iterations), wall_seconds=perf_counter()-started, **stages,
        actual_operator_audit=dict(exact_distinct_operator_histogram={str(n):exact_counts.count(n) for n in sorted(set(exact_counts))},
            floating_byte_distinct_operator_histogram={str(n):byte_counts.count(n) for n in sorted(set(byte_counts))},
            questions_with_any_literal_topk_membership_change_vs_identity=changed,
            maximum_identity_normalization_coordinate_roundoff=identity_roundoff,
            sample_scope='Questions are the sampling units; sixteen views are correlated parameter requests, never sixteen independent questions.'),
        oracle_scope='Independent raw mention/weak/self weighting; measured complete-column normalization; cold PI on actual complete view; actual fixed-point residual. No engine imports, response/core formula, timed output, or evaluation labels.')
    write(output/'summary.json', summary)
    print(json.dumps(dict(dataset=dataset, targets=summary['targets'], max_actual_l1_residual=max_residual, output=str(output))), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--datasets', nargs='+', default=['hotpot_pool7405','musique_pool2417'])
    parser.add_argument('--queries', type=int, default=2)
    parser.add_argument('--residual-tol', type=float, default=5e-15)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    if args.queries < 0 or not 0 < args.residual_tol <= 5e-15:
        raise ValueError('Invalid prefix/tolerance')
    with threadpool_limits(limits=1):
        for dataset in args.datasets:
            run_dataset(args,dataset)
