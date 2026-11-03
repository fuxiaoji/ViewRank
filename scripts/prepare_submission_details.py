"""Extract closing evidence from immutable ledgers; export the executed SMT formula.

Formula export calls the published encoder with a capture solver, without solving.
Recorded encoding/solve timings always come from the original execution ledger.
"""
from pathlib import Path
from fractions import Fraction as Q
from hashlib import sha256
import argparse, collections, csv, importlib.util, json, math, sys
import numpy as np
import z3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.policy_domain_polynomial import compile_domain

METHODS = ['fine_preview', 'dp_preview_fine', 'lift_sparse_fine_preview']
NAMES = [f'{d}_a{a}' for d in ['hotpot_pool7405', 'musique_pool2417']
         for a in ['0p5', '0p85', '0p95']] + [
             'hotpot_pool1000_a0p5', 'musique_pool1000_a0p5',
             'amazon_r4_original', 'amazon_r4_self05']

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf8')

def csvout(path, rows):
    with path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-root', type=Path, default=ROOT/'supplement/frozen_states')
    parser.add_argument('--output', type=Path, default=ROOT/'results/submission_details_v4')
    parser.add_argument('--tables-only', action='store_true')
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    base = ROOT/'results/professor_revision_v1'
    counts = []; unknown = []
    for name in NAMES:
        summary = json.loads((base/'boundaries'/name/'summary.json').read_text())
        assert summary['complete_blocks'] == 1 and not summary['source_drift']
        for method in METHODS:
            path = base/'boundaries'/name/method/'questions.jsonl'
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            statuses = collections.Counter()
            resolved = 0
            for row in rows:
                cuts = row['answer']['cuts']; resolved += all(c['status'] != 'UNKNOWN' for c in cuts.values())
                for k, cut in cuts.items():
                    statuses[cut['status']] += 1
                    if cut['status'] == 'UNKNOWN':
                        reasons = sorted({leaf.get('reason', 'not_recorded') for leaf in cut.get('leaves', []) if leaf['status'] == 'UNKNOWN'})
                        reason = cut.get('reason') or ' | '.join(reasons) or 'not_recorded'
                        unknown.append(dict(cohort=name, method=method, question_index=row['question_index'],
                            qid=row['qid'], k=int(k), reason=reason,
                            actual_route=row['answer'].get('actual_route'), point_checks=row['answer'].get('point_checks'),
                            original_sources=json.dumps(row['original_sources']),
                            provenance=str(path.relative_to(ROOT)), ledger_sha256=sha256(path.read_bytes()).hexdigest()))
            expected = summary['blocks'][0]['specs'][method]
            assert dict(statuses) == expected['statuses'] and resolved == expected['all_cut_queries'].get('resolved', 0)
            assert sum(statuses.values()) == 3*len(rows)
            counts.append(dict(cohort=name, method=method, questions=len(rows), stable=statuses['STABLE'],
                verified_unstable=statuses['UNSTABLE'], unknown=statuses['UNKNOWN'], all_cut_resolved_questions=resolved,
                complete_seconds=expected['total_seconds'], ledger_sha256=sha256(path.read_bytes()).hexdigest()))
    csvout(out/'tristates.csv', counts); csvout(out/'unknown_cuts.csv', unknown)
    dump(out/'unknown_reasons.json', dict(collections.Counter(u['reason'] for u in unknown)))
    tex = [r'\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}',
           r'\caption{Cutoff outcomes for the complete additional cohorts: STABLE / verified UNSTABLE / UNKNOWN. Each lexical row has 300 cuts; each Amazon row has 24. Resolving every query can include instability. VR, DP and Lift follow Tables~\ref{tab:alpha-boundary}--\ref{tab:graph-boundary}.}',
           r'\label{tab:tristates}', r'\begin{tabular}{lrrr}\toprule Setting & VR & DP & Lift\\\midrule']
    labels = [f'{g}, {a}' for g in ['Hot', 'Mu'] for a in ['.50', '.85', '.95']] + ['Hot small', 'Mu small', 'Amazon', 'Amazon +self']
    for name, label in zip(NAMES, labels):
        group = [c for c in counts if c['cohort'] == name]
        tex.append(label+' & '+' & '.join(f"{c['stable']}/{c['verified_unstable']}/{c['unknown']}" for c in group)+r'\\')
    tex += [r'\bottomrule\end{tabular}\end{table}']
    (out/'tristates.tex').write_text('\n'.join(tex)+'\n', encoding='utf8')
    if args.tables_only:
        print('PASS: complete three-state tables rebuilt from all 30 method/cohort ledgers')
        return

    source = base/'executed_source/benchmark_shared_smt.py'
    spec = importlib.util.spec_from_file_location('published_smt', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original_factory = z3.SolverFor
    captured = []
    class Capture:
        def __init__(self, logic):
            self.solver = original_factory(logic); captured.append(self.solver)
        def set(self, **kwargs): self.solver.set(**kwargs)
        def add(self, *args): self.solver.add(*args)
        def check(self): return z3.unknown
        def reason_unknown(self): return 'EXPORT_ONLY_NO_SOLVE'
    module.z3.SolverFor = Capture
    diagnostics = []; timings = []
    recorded = [json.loads(s) for s in (base/'shared_smt_dyadic_prefix20_5s/rows.jsonl').read_text().splitlines()]
    try:
        for ds in ['hot', 'mu']:
            folder = args.frozen_root/f'full_{ds}_main_v3'
            trace = [json.loads(s) for s in (folder/'fine_preview/questions.jsonl').read_text().splitlines()]
            trace = {r['question_index']: r for r in trace if r['block'] == 0 and r['question_index'] < 20}
            for rec in [r for r in recorded if r['dataset'] == ds]:
                row = trace[rec['question_index']]; meta = row['audit']; path = folder/meta['path']
                assert sha256(path.read_bytes()).hexdigest() == meta['sha256']
                with np.load(path) as a0: a = {k:a0[k] for k in a0.files}
                domain = compile_domain(a['H_m'], a['H_w'], list(map(Q, meta['mu'])), list(map(Q, meta['nu'])),
                    a['y'][a['source_ids']], ((1,4),(.25,2)), a['source_ids'])
                def up(x):
                    x = Q(x); result = Q(math.nextafter(float(x), math.inf)); assert result >= x; return result
                guards = dict(kappa=up(meta['kappa']), base_residual=up(meta['base_residual']), atom_residuals=list(map(up,meta['atom_residuals'])))
                for k in [2,5,10]:
                    chosen = row['answer']['cuts'][str(k)].get('selected_ids', np.argsort(-a['y'])[:k].tolist())
                    module.smt(domain, a, chosen, guards, 5000)
                    solver = captured.pop(); assertions = solver.assertions(); seen = set(); stack = list(assertions)
                    while stack:
                        expr = stack.pop()
                        if expr.get_id() in seen: continue
                        seen.add(expr.get_id()); stack.extend(expr.children())
                    C = len(a['candidate_rows']); q = 2*domain.r
                    diagnostics.append(dict(dataset=ds, question_index=rec['question_index'], qid=rec['qid'], k=k,
                        r=domain.r, candidates=C, selected=len(chosen), F_variables=q,
                        top_level_assertions=len(assertions), polynomial_defining_equalities=1+q,
                        box_inequalities=4, candidate_threshold_inequalities=C,
                        outside_gap_inequalities=len(chosen) if C < len(a['y']) else 0,
                        sign_tests=q, unique_ast_nodes=len(seen),
                        polynomial_monomials=len(domain.D)+sum(map(len, domain.F)),
                        max_total_degree=max(i+j for p in [domain.D,*domain.F] for i,j in p), frozen_state_sha256=meta['sha256']))
                    timings.append(dict(dataset=ds, question_index=rec['question_index'], qid=rec['qid'], k=k, **rec['smt'][str(k)]))
                    if ds == 'hot' and rec['question_index'] == 0 and k == 2:
                        (out/'example_hot_q0_k2.smt2').write_text('(set-option :timeout 5000)\n(set-option :random-seed 0)\n'+solver.to_smt2(), encoding='utf8')
    finally:
        z3.SolverFor = original_factory
    assert len(diagnostics) == len(timings) == 120
    csvout(out/'smt_encoding_counts.csv', diagnostics); csvout(out/'smt_all_120_cuts.csv', timings)
    aggregates = {}
    for ds in ['hot','mu']:
        group = [t for t in timings if t['dataset'] == ds]
        aggregates[ds] = dict(encoding_seconds=sum(t['encoding_seconds'] for t in group),
            solve_seconds=sum(t['solve_seconds'] for t in group), total_seconds=sum(t['total_seconds'] for t in group),
            solver_outcomes=dict(collections.Counter(t['solver_result'] for t in group)))
    dump(out/'SMT_SETTINGS.json', dict(runtime_z3=z3.get_version_string(), package='z3-solver==5.3.3.0',
        logic='QF_NRA', constructor='SolverFor', timeout_ms_per_cut=5000, random_seed=0,
        tactic='Logic-specific default; no explicit tactic pipeline', source_sha256=sha256(source.read_bytes()).hexdigest(),
        reuse='Common source polynomial compiler once per query, excluded both timings. D/F named and reused within cutoff; fresh solver and complete encoding for each cutoff, no cross-cutoff reuse.',
        export='Reinvoked the exact published encoder without solving; measured times are only from immutable original logs.', aggregates=aggregates,
        count_ranges={key:[min(d[key] for d in diagnostics),max(d[key] for d in diagnostics)] for key in ['r','candidates','unique_ast_nodes','polynomial_monomials','max_total_degree']}))
    dump(out/'summary.json', dict(status='PASS', cohorts=len(NAMES), method_cohorts=len(counts), unknown_cuts=len(unknown), smt_cuts=len(timings), smt=aggregates))
    print(json.dumps(dict(status='PASS', smt=aggregates, unknown_reasons=dict(collections.Counter(u['reason'] for u in unknown)))))

if __name__ == '__main__': main()
