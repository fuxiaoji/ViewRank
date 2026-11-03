"""Meaningful isolated certificate checks, never a performance benchmark.

All small synthetic graphs use column-normalized transitions. Direct actual
views are assembled independently of the certificate. Exact rational solves
check the dyadic five-node example; a 17-node randomized graph also uses
independent dense solves and exact residual upper bounds for stored floats.
"""
from fractions import Fraction as Q
import json
import math
import os
from pathlib import Path
import random
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.policy_domain_polynomial import (compile_domain, certify, poly_eval,
    bernstein, bernstein_eval, exact, Interval)


OUT = Path(__file__).resolve().parents[1]/'results'/'policy_domain_selfcheck'


def solve_exact(A, rhs):
    n = len(A)
    augmented = [list(row)+[rhs[i]] for i, row in enumerate(A)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if augmented[i][j])
        augmented[j], augmented[pivot] = augmented[pivot], augmented[j]
        value = augmented[j][j]
        augmented[j] = [x/value for x in augmented[j]]
        for i in range(n):
            if i != j:
                value = augmented[i][j]
                augmented[i] = [x-value*y for x, y in zip(augmented[i], augmented[j])]
    return [row[-1] for row in augmented]


def matrix_A(P, alpha=Q(1, 2)):
    n = len(P)
    return [[Q(i == j)-alpha*P[i][j] for j in range(n)] for i in range(n)]


def base_and_atoms(P, M, W, seed, sources, float_storage=False):
    n = len(P)
    alpha = Q(1, 2)
    A = matrix_A(P)
    b = [(1-alpha)*value for value in seed]
    y = solve_exact(A, b)
    masses = [[sum(G[i][s] for i in range(n)) for s in sources] for G in (M, W)]
    rhs_atoms = [[-alpha*(G[i][s]-mass*P[i][s]) for i in range(n)]
        for G, group_mass in zip((M, W), masses) for s, mass in zip(sources, group_mass)]
    atom_columns = [solve_exact(A, rhs) for rhs in rhs_atoms]
    if float_storage:
        y = [exact(float(value)) for value in y]
        atom_columns = [[exact(float(value)) for value in column] for column in atom_columns]
    R = [list(values) for values in zip(*atom_columns)]
    residual = lambda x, rhs: sum(abs(rhs[i]-sum(A[i][j]*x[j] for j in range(n))) for i in range(n))
    rho_y = residual(y, b)
    rho_atom = [residual(column, rhs) for column, rhs in zip(atom_columns, rhs_atoms)]
    return dict(y=y, R=R, mu=masses[0], nu=masses[1], rho_y=rho_y,
        rho_atom=rho_atom, Hm=[R[s][:len(sources)] for s in sources],
        Hw=[R[s][len(sources):] for s in sources], sources=sources)


def actual_view(P, M, W, sources, m, w):
    m, w = exact(m), exact(w)
    n = len(P)
    view = [list(row) for row in P]
    for source in sources:
        self_value = P[source][source]
        total = self_value+m*sum(row[source] for row in M)+w*sum(row[source] for row in W)
        for i in range(n):
            view[i][source] = ((self_value if i == source else 0)+m*M[i][source]+w*W[i][source])/total
    return view


def cold_exact(P, M, W, seed, sources, m, w):
    return solve_exact(matrix_A(actual_view(P, M, W, sources, m, w)), [Q(1, 2)*v for v in seed])


def top(scores, k):
    return tuple(sorted(sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]))


def compile_for(data, box):
    return compile_domain(data['Hm'], data['Hw'], data['mu'], data['nu'],
        [data['y'][s] for s in data['sources']], box, data['sources'])


def certify_for(domain, data, selected, candidates=None, mask=None, tails=None, depth=0):
    n, q = len(data['y']), len(data['R'][0])
    candidates = list(range(n)) if candidates is None else list(candidates)
    mask = [[True]*q for _ in candidates] if mask is None else mask
    if tails is None:
        kept = {node: i for i, node in enumerate(candidates)}
        tails = [max((abs(data['R'][node][j]) for node in range(n)
            if node not in kept or not mask[kept[node]][j]), default=Q(0)) for j in range(q)]
    return certify(domain, data['y'], candidates, [data['R'][i] for i in candidates], mask,
        tails, selected, kappa=2, base_residual=data['rho_y'], atom_residuals=data['rho_atom'],
        subdivide_depth=depth)


def empty_graph(n):
    return [[Q(0) for _ in range(n)] for _ in range(n)]


def chain_graph():
    P, M, W = (empty_graph(5) for _ in range(3))
    M[1][0] = W[2][0] = M[2][1] = W[3][1] = Q(1, 2)
    for i in range(5):
        for j in range(5):
            P[i][j] = M[i][j]+W[i][j]
    for i in (2, 3, 4):
        P[i][i] = Q(1)
    a = exact(.95)
    return P, M, W, [a, Q(0), Q(0), Q(0), 1-a], [0, 1]


def dependency_graph():
    P, M, W = (empty_graph(4) for _ in range(3))
    P[0][0] = Q(1, 2)
    M[1][0] = W[2][0] = Q(1, 4)
    P[1][0] = P[2][0] = Q(1, 4)
    for i in (1, 2, 3):
        P[i][i] = 1
    return P, M, W, [Q(500, 797), Q(0), Q(0), Q(297, 797)], [0]


def omission_graph():
    P, M, W = (empty_graph(4) for _ in range(3))
    M[1][0] = P[1][0] = Q(1, 8)
    W[2][0] = P[2][0] = Q(7, 8)
    for i in (1, 2, 3):
        P[i][i] = 1
    a = exact(.9)
    return P, M, W, [a, Q(0), Q(0), 1-a], [0]


def main():
    checks = []
    def record(name, condition, **details):
        assert condition, name
        checks.append(dict(name=name, passed=True, **details))

    # Independent polynomial identities and coefficientwise |F| envelope.
    rng = random.Random(20261103)
    evaluations = 0
    for r in range(1, 5):
        hm = [[Q(rng.randint(-3, 3), 16) for _ in range(r)] for _ in range(r)]
        hw = [[Q(rng.randint(-3, 3), 16) for _ in range(r)] for _ in range(r)]
        ys = [Q(rng.randint(1, 5), 32) for _ in range(r)]
        mu, nu = [Q(1, 4)]*r, [Q(1, 4)]*r
        domain = compile_domain(hm, hw, mu, nu, ys, ((Q(1, 2), 2), (Q(1, 4), 3)), list(range(r)))
        D, F = domain.controls()
        for _ in range(12):
            u, v = Q(rng.randint(0, 32), 32), Q(rng.randint(0, 32), 32)
            m = Q(1, 2)+Q(3, 2)*u
            w = Q(1, 4)+Q(11, 4)*v
            assert bernstein_eval(D, r, u, v) == poly_eval(domain.D, m, w)
            c = [poly_eval(poly, m, w) for poly in domain.C]
            det = poly_eval(domain.D, m, w)
            for i in range(r):
                lhs = sum((Q(i == j)+(m-1)*(hm[i][j]+(mu[j] if i == j else 0))+(w-1)*(hw[i][j]+(nu[j] if i == j else 0)))*c[j] for j in range(r))
                assert lhs == ys[i]*det
            for poly, controls in zip(domain.F, F):
                exact_value = poly_eval(poly, m, w)
                assert exact_value == bernstein_eval(controls, r, u, v)
                assert abs(exact_value) <= bernstein_eval(tuple(map(abs, controls)), r, u, v)
            evaluations += 1
    record('Exact D/C/F identities and Bernstein |F| envelope for r=1..4', True, sampled_parameter_pairs=evaluations)

    # Every interval arithmetic operation encloses its exact scalar result.
    for _ in range(100):
        a, b = Q(rng.randint(-1000, 1000), rng.randint(1, 333)), Q(rng.randint(-1000, 1000), rng.randint(1, 333))
        ia, ib = Interval.of(a), Interval.of(b)
        for output, value in ((ia+ib, a+b), (ia-ib, a-b), (ia*ib, a*b)):
            assert exact(output.lo) <= value <= exact(output.hi)
    record('Outward coefficient/add/subtract/multiply enclosures', True, exact_operations=300)

    P, M, W, seed, sources = chain_graph()
    data = base_and_atoms(P, M, W, seed, sources)
    record('Five-node stored vectors have exactly zero residual', data['rho_y'] == 0 and not any(data['rho_atom']))
    domain = compile_for(data, ((.25, 4), (.99, 1.01)))
    corners = [top(cold_exact(P, M, W, seed, sources, m, w), 4) for m in (.25, 4) for w in (.99, 1.01)]
    interior = top(cold_exact(P, M, W, seed, sources, 1, 1), 4)
    dense = certify_for(domain, data, corners[0], depth=3)
    candidates = list(corners[0])
    sparse = certify_for(domain, data, corners[0], candidates=candidates, depth=3)
    record('Four equal corner memberships do not imply whole-box stability',
        len(set(corners)) == 1 and corners[0] != interior and dense['status'] == sparse['status'] == 'UNKNOWN',
        corner_membership=corners[0], actual_interior_membership=interior,
        dense_certificate=dense['status'], sparse_certificate=sparse['status'])
    exact_compares = 0
    for m in (Q(1, 4), Q(1), Q(2), Q(4)):
        for w in (Q(99, 100), Q(1), Q(101, 100)):
            det = poly_eval(domain.D, m, w)
            f = [poly_eval(poly, m, w) for poly in domain.F]
            proposal = [data['y'][i]-sum(data['R'][i][j]*f[j] for j in range(4))/det for i in range(5)]
            assert proposal == cold_exact(P, M, W, seed, sources, m, w)
            exact_compares += 1
    record('Cleared-denominator response equals independent actual-view rational solve', True, exact_views=exact_compares)

    P, M, W, seed, sources = dependency_graph()
    data = base_and_atoms(P, M, W, seed, sources, float_storage=True)
    domain = compile_for(data, ((1, 2), (1, 2)))
    result = certify_for(domain, data, [0])
    naive_box_source_score = Q(10, 17)*seed[0]
    record('Dependency-preserving domain passes where independent a-box cannot separate',
        result['status'] == 'STABLE' and naive_box_source_score < seed[3],
        status=result['status'], exact_stored_float_residual_upper=str(data['rho_y']),
        independent_box_source_score=str(naive_box_source_score), fixed_competitor=str(seed[3]))
    domain_cross = compile_for(data, ((.5, 1.5), (.5, 1.5)))
    _, controls = domain_cross.controls()
    result = certify_for(domain_cross, data, [0])
    record('|F| envelope handles parameter crossings through weight=1',
        result['status'] == 'STABLE' and any(min(f) < 0 < max(f) for f in controls), status=result['status'])
    direct_checks = 0
    for m in (Q(1, 2), Q(3, 4), Q(1), Q(3, 2), Q(2)):
        for w in (Q(1, 2), Q(1), Q(3, 2), Q(2)):
            assert top(cold_exact(P, M, W, seed, sources, m, w), 1) == (0,)
            direct_checks += 1
    record('Stable dependency example agrees with exact actual-view solves', True, exact_views=direct_checks)

    P, M, W, seed, sources = omission_graph()
    data = base_and_atoms(P, M, W, seed, sources)
    domain = compile_for(data, ((8, 16), (.99, 1.01)))
    candidates = [0, 2, 3]
    honest = certify_for(domain, data, [0, 2], candidates=candidates, depth=2)
    dishonest = certify_for(domain, data, [0, 2], candidates=candidates, tails=[0, 0])
    actual = top(cold_exact(P, M, W, seed, sources, 12, 1), 2)
    record('Outside-candidate tail prevents a false subset certificate', honest['status'] == 'UNKNOWN' and actual != (0, 2),
        actual_membership=actual, honest_status=honest['status'])
    record('False tail contract is demonstrably unsafe and cannot be auto-validated',
        dishonest['status'] == 'STABLE' and actual != (0, 2),
        invalid_contract_status=dishonest['status'], actual_membership=actual,
        warning='This deliberately invalid tail input is NOT a valid certificate. Caller must independently establish omitted-coordinate envelopes.')
    wrong = certify_for(domain, data, [2], depth=2)
    record('Incorrect selected subset returns UNKNOWN', wrong['status'] == 'UNKNOWN')

    # This is an algebra/coverage selfcheck, not a claim of Markov realization.
    anti = compile_domain([[0, 2], [-2, 0]], [[0, 0], [0, 0]], [0, 0], [0, 0], [1, 1], ((.1, 1.9), (.9, 1.1)), [0, 1])
    kwargs = dict(y=[1, 1], candidate_rows=[0, 1], retained_values=[[0, 2, 0, 0], [-2, 0, 0, 0]],
        retained_mask=[[True]*4]*2, tail_bounds=[0]*4, selected_ids=[0, 1],
        kappa=1, base_residual=0, atom_residuals=[0]*4)
    before, after = certify(anti, **kwargs), certify(anti, **kwargs, subdivide_depth=1)
    record('Exact subdivision proves positive D controls after parent-box failure',
        before['status'] == 'UNKNOWN' and after['status'] == 'STABLE' and after['leaf_boxes'] == 2,
        parent_status=before['status'], child_status=after['status'], leaves=after['leaf_boxes'],
        scope='Pure polynomial D=1+4(m-1)^2 positivity and domain coverage check; selected set is all nodes, no physical graph/residual claim.')
    neg = compile_domain([[-2]], [[0]], [0], [0], [1], ((2, 3), (.9, 1.1)), [0])
    negative = certify(neg, [1], [0], [[-2, 0]], [[True, True]], [0, 0], [0],
        kappa=1, base_residual=0, atom_residuals=[0, 0], subdivide_depth=2)
    record('Nonpositive determinant is never accepted', negative['status'] == 'UNKNOWN')

    malformed = 0
    try:
        compile_domain([[0]], [[0]], [Q(3, 4)], [Q(1, 2)], [1], ((1, 2), (1, 2)), [0])
    except ValueError:
        malformed += 1
    try:
        certify(domain, data['y'], [0, 2, 3], [[1, 0], data['R'][2], data['R'][3]],
            [[True, True]]*3, [1, 1], [0], kappa=2, base_residual=0, atom_residuals=[0, 0])
    except ValueError:
        malformed += 1
    try:
        bad_y = list(data['y']); bad_y[0] += 1
        certify(domain, bad_y, [0, 2, 3], [data['R'][i] for i in (0, 2, 3)],
            [[True, True]]*3, [1, 1], [0], kappa=2, base_residual=0, atom_residuals=[0, 0])
    except ValueError:
        malformed += 1
    record('Illegal masses, source H mismatch and y_S mismatch are rejected', malformed == 3)

    # Randomized 17-node physical graph: raw integer edge counts sum to 32,
    # so P, group masses, seed and alpha are exact dyadics. Only y/R are
    # rounded for storage, and their residuals are recomputed EXACTLY.
    import numpy as np
    n, sources = 17, [0, 1, 2, 3]
    P, M, W = (empty_graph(n) for _ in range(3))
    for source in range(n-1):
        weights = [0]*n
        weights[source] = 1
        for _ in range(31):
            weights[rng.randrange(n)] += 1
        for i, count in enumerate(weights):
            P[i][source] = Q(count, 32)
            if i != source:
                (M if i % 2 else W)[i][source] = Q(count, 32)
    P[-1][-1] = 1
    seed = [Q(1, 32) if i in sources else Q(0) for i in range(n)]
    seed[-1] = Q(7, 8)
    data = base_and_atoms(P, M, W, seed, sources, float_storage=True)
    domain = compile_for(data, ((.5, 2), (.5, 2)))
    dense = certify_for(domain, data, [16], depth=2)
    # Retain all source coordinates plus two largest coordinates/atom, then
    # derive rigorous omitted tails from the full stored vectors.
    keep = set(sources)|{16}
    by_atom = []
    for atom in range(8):
        chosen = set(sorted(range(n), key=lambda i: abs(data['R'][i][atom]), reverse=True)[:2])|set(sources)
        by_atom.append(chosen)
        keep.update(chosen)
    candidates = sorted(keep)
    masks = [[node in by_atom[atom] for atom in range(8)] for node in candidates]
    sparse = certify_for(domain, data, [16], candidates=candidates, mask=masks, depth=2)
    record('17-node randomized actual graph has Dense and omitted-tail STABLE certificates',
        dense['status'] == sparse['status'] == 'STABLE', dense_status=dense['status'],
        sparse_status=sparse['status'], candidates=len(candidates), nodes=n,
        stored_float_base_residual_upper=str(data['rho_y']),
        maximum_stored_float_atom_residual_upper=str(max(data['rho_atom'])))
    views = 0
    worst_actual_residual = 0.
    for _ in range(60):
        m, w = .5+1.5*rng.random(), .5+1.5*rng.random()
        actual_P = actual_view(P, M, W, sources, m, w)
        A = np.array(matrix_A(actual_P), dtype=float)
        b = .5*np.array(seed, dtype=float)
        x = np.linalg.solve(A, b)
        assert top(x, 1) == (16,)
        worst_actual_residual = max(worst_actual_residual, float(np.linalg.norm(b-A@x, 1)))
        views += 1
    record('17-node certificates agree with independent direct actual-view solves', True,
        actual_views=views, maximum_measured_direct_solve_residual_l1=worst_actual_residual,
        note='These sampled solves supplement the analytic conditional certificate; they do not replace its residual/tail contracts.')

    result = dict(all_pass=True, checks=checks, check_count=len(checks),
        status='Isolated synthetic certificate prototype validated; no natural-graph completeness or performance claim.',
        files=dict(kernel=str(Path(__file__).resolve().parents[1]/'src'/'policy_domain_polynomial.py'), selfcheck=str(Path(__file__).resolve())))
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'policy_domain_selfcheck.json').exists():
        raise FileExistsError('Use a new output directory; existing checks are preserved.')
    with (OUT/'policy_domain_selfcheck.json').open('w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(dict(all_pass=True, checks=len(checks), output=str(OUT/'policy_domain_selfcheck.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    OUT = parser.parse_args().output.resolve()
    main()
