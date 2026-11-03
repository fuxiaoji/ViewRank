"""Public HotpotQA retrieval and query-local view execution without an LLM.

The pooled setting is a derived fixed-candidate workload, not HotpotQA fullwiki.
Only public context/question fields construct indexes, graphs and local views.
Gold supporting_facts are accessed solely by the evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import lu_factor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.viewrank import ViewRank, PreparedView, power_iteration
from src.envelope import prepare_envelope, envelope_topk

STOP = set("a an the and or of to in on for at by from as is are was were be been being it its their his her he she they them this that these those who what which when where why how did does do have has had with into between both than also not but if while after before during would could should can may will you your we our i other such some all most more many each about over under out up through only very much".split())


def raw_tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def tokens(text):
    return [t for t in raw_tokens(text) if len(t) > 1 and t not in STOP]


class Index:
    def __init__(self, docs):
        self.docs = docs
        self.n = len(docs)
        self.terms = [Counter(tokens(d[0]+" "+d[1])) for d in docs]
        self.df = Counter(t for d in self.terms for t in d)
        self.lengths = np.array([sum(d.values()) for d in self.terms])
        self.avgdl = max(float(self.lengths.mean()), 1)
        self.posting = defaultdict(list)
        for i, d in enumerate(self.terms):
            for t, tf in d.items():
                idf = np.log(1+(self.n-self.df[t]+.5)/(self.df[t]+.5))
                score = idf*tf*2.5/(tf+1.5*(.25+.75*self.lengths[i]/self.avgdl))
                self.posting[t].append((i, score))
        self.posting = {t: (np.array([p[0] for p in v]), np.array([p[1] for p in v])) for t, v in self.posting.items()}
        self.title_starts = defaultdict(list)
        for i, d in enumerate(docs):
            name = raw_tokens(re.sub(r"\([^)]*\)", "", d[0]))
            if name:
                self.title_starts[name[0]].append((tuple(name), i))

    def bm25(self, query):
        scores = np.zeros(self.n)
        for term in set(tokens(query)):
            if term in self.posting:
                rows, weights = self.posting[term]
                scores[rows] += weights
        return scores

    def mentions(self, text):
        words = raw_tokens(text)
        found = set()
        for pos, term in enumerate(words):
            for name, node in self.title_starts.get(term, []):
                if tuple(words[pos:pos+len(name)]) == name:
                    found.add(node)
        return found


def graph(index, lexical_degree=12):
    """Title-mention links plus sparse shared-rare-term associations."""
    n = index.n
    entity = [set() for _ in range(n)]
    lexical = [defaultdict(float) for _ in range(n)]
    for i, (_, body) in enumerate(index.docs):
        for j in index.mentions(body):
            if i != j:
                entity[i].add(j)
                entity[j].add(i)
    selected = defaultdict(list)
    for i, terms in enumerate(index.terms):
        eligible = [t for t in terms if 2 <= index.df[t] <= 32]
        eligible.sort(key=lambda t: (index.df[t], t))
        for t in eligible[:8]:
            selected[t].append(i)
    for ids in selected.values():
        weight = 1/max(len(ids)-1, 1)
        for i in ids:
            for j in ids:
                if i != j:
                    lexical[i][j] += weight
    rows, cols, data = [], [], []
    for i in range(n):
        weights = {j: min(w, 2.0) for j, w in sorted(lexical[i].items(), key=lambda p: (-p[1], p[0]))[:lexical_degree]}
        for j in entity[i]:
            weights[j] = weights.get(j, 0)+3.0
        weights[i] = weights.get(i, 0)+.5
        for j, weight in weights.items():
            rows.append(j); cols.append(i); data.append(weight)
    W = sparse.csc_matrix((data, (rows, cols)), shape=(n, n))
    sums = np.asarray(W.sum(axis=0)).ravel()
    P = (W@sparse.diags(1/sums)).tocsc()
    return P, entity


def seed_vector(scores, keep=32):
    seed = np.zeros(len(scores))
    order = np.argsort(-scores, kind="stable")[:min(keep, len(scores))]
    seed[order] = scores[order]
    if seed.sum() == 0:
        seed[:] = 1/len(seed)
    else:
        seed /= seed.sum()
    return seed


def local_view(P, entity, index, query, bm25, rank=4):
    anchors = sorted(index.mentions(query), key=lambda i: (-len(index.docs[i][0]), i))
    sources = anchors[:rank] if anchors else np.argsort(-bm25, kind="stable")[:min(2, rank)].tolist()
    deltas, valid = [], []
    for source in sources:
        original = P[:, source].toarray().ravel()
        new = original.copy()
        # Same public relation policy for every query. Question determines
        # which source columns are edited, never their gold support status.
        destinations = np.flatnonzero(original)
        for dst in destinations:
            if dst != source:
                new[dst] *= 3.0 if dst in entity[source] else .5
        new /= new.sum()
        delta = new-original
        if np.any(delta):
            valid.append(source); deltas.append(delta)
    return np.asarray(valid, dtype=int), (np.column_stack(deltas) if deltas else np.empty((P.shape[0], 0)))


def retrieve_metrics(order, titles, gold):
    result = {}
    for k in [2, 5, 10]:
        predicted = {titles[i] for i in order[:k]}
        result[f"recall_at_{k}"] = len(predicted & gold)/len(gold)
        result[f"all_support_at_{k}"] = int(gold <= predicted)
    return result


def summarize(rows):
    summary = []
    for setting, method in sorted(set((r["setting"], r["method"]) for r in rows)):
        subset = [r for r in rows if r["setting"] == setting and r["method"] == method]
        entry = dict(setting=setting, method=method, questions=len(subset))
        for key in ["recall_at_2", "all_support_at_2", "recall_at_5", "all_support_at_5", "recall_at_10", "all_support_at_10", "gold_pool_coverage"]:
            entry[key] = float(np.mean([r[key] for r in subset]))
        summary.append(entry)
    return summary


def write_outputs(out, rows, execution, meta):
    out.mkdir(parents=True, exist_ok=True)
    for name, values in [("retrieval", rows), ("execution", execution), ("summary", summarize(rows))]:
        (out/f"{name}.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
        if values:
            with (out/f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    (out/"run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def distractor(data, args):
    rows = []
    t = perf_counter()
    for qi, item in enumerate(data[:args.queries or None]):
        docs = [(title, " ".join(sentences)) for title, sentences in item["context"]]
        index = Index(docs)
        P, entity = graph(index)
        bm25 = index.bm25(item["question"])
        seed = seed_vector(bm25)
        vr = ViewRank(P, alpha=args.alpha)
        y = vr.base_lu.solve((1-args.alpha)*seed)
        sources, D = local_view(P, entity, index, item["question"], bm25)
        prepared = vr.prepare(D, sources)
        view_scores = vr.solve(seed, prepared)
        # No evaluator annotation is read above this line.
        gold = {title for title, _ in item["supporting_facts"]}
        titles = [d[0] for d in docs]
        for method, scores in [("BM25", bm25), ("AssociationPPR", y), ("EntityPolicyPPR", view_scores)]:
            order = np.argsort(-scores, kind="stable")
            rows.append(dict(setting="official_dev_distractor_candidates", method=method, qid=item["_id"],
                             question_index=qi, gold_pool_coverage=len(set(titles)&gold)/len(gold),
                             **retrieve_metrics(order, titles, gold)))
        if (qi+1) % 1000 == 0:
            print(json.dumps(dict(stage="distractor", questions=qi+1, seconds=perf_counter()-t)), flush=True)
    return rows


def pooled(data, args):
    # Freeze corpus from public contexts before reading any support annotations.
    by_title = {}
    conflicts = 0
    for item in data[:args.pool_questions]:
        for title, sentences in item["context"]:
            text = " ".join(sentences)
            if title in by_title and by_title[title] != text:
                conflicts += 1
            by_title.setdefault(title, text)
    docs = list(by_title.items())
    index = Index(docs)
    t = perf_counter(); P, entity = graph(index); graph_seconds = perf_counter()-t
    vr = ViewRank(P, alpha=args.alpha)
    n = len(docs)
    print(json.dumps(dict(stage="pool_ready", nodes=n, nnz=P.nnz, graph_seconds=graph_seconds,
                          base_factor_seconds=vr.base_factor_seconds)), flush=True)
    sparse.save_npz(ROOT/"data/hotpot_pool_graph.npz", P)
    (ROOT/"data/hotpot_pool_titles.json").write_text(json.dumps([d[0] for d in docs]), encoding="utf-8")
    zcache = {}; atomic_hits = 0; templates = set()
    rows, execution = [], []
    start = perf_counter()
    for qi, item in enumerate(data[:args.queries or None]):
        t = perf_counter(); bm25 = index.bm25(item["question"]); lexical_ms = (perf_counter()-t)*1000
        seed = seed_vector(bm25)
        t = perf_counter(); y = vr.base_lu.solve((1-args.alpha)*seed); base_ms = (perf_counter()-t)*1000
        sources, D = local_view(P, entity, index, item["question"], bm25)
        templates.add(tuple(sources))
        t = perf_counter(); missing = 0; columns = []
        for j, source in enumerate(sources):
            if int(source) not in zcache:
                zcache[int(source)] = vr.base_lu.solve(-args.alpha*D[:, j]); missing += 1
            else:
                atomic_hits += 1
            columns.append(zcache[int(source)])
        Z = np.column_stack(columns) if columns else np.empty((n, 0))
        K = np.eye(len(sources))+Z[sources, :]
        prepared = PreparedView(sparse.csc_matrix(D), sources, Z, K,
                                lu_factor(K) if len(sources) else None, str(tuple(sources)), 0)
        response_prepare_ms = (perf_counter()-t)*1000
        t = perf_counter()
        c = np.linalg.solve(K, y[sources]) if len(sources) else np.empty(0)
        exact = y-Z@c
        exact_correction_ms = (perf_counter()-t)*1000
        compressed = prepare_envelope(vr, prepared, keep_per_column=args.keep)
        certified = {k: envelope_topk(vr, seed, compressed, k=k, reference_scores=exact, base_scores=y)
                     for k in [2, 5, 10]}
        result = certified[10]
        # Warm PI is an independent implementation of the same public policy.
        t = perf_counter()
        pi = y.copy(); iterations = 0
        if len(sources):
            # Matrix-free view iteration avoids per-query graph/LU rebuilding.
            pi = y.copy(); teleport = (1-args.alpha)*seed
            for iterations in range(1, 10000):
                new = args.alpha*(P@pi+D@pi[sources])+teleport
                if np.linalg.norm(new-pi, ord=1) <= 1e-11*(1-args.alpha):
                    pi = new; break
                pi = new
        pi_ms = (perf_counter()-t)*1000
        assert np.max(np.abs(pi-exact)) < 1e-9
        cold_ms = 0.0; cold_iterations = 0; cold_error = 0.0
        if getattr(args, "cold_reference", False):
            t = perf_counter()
            cold = seed.copy(); teleport = (1-args.alpha)*seed
            for cold_iterations in range(1, 10000):
                new = args.alpha*(P@cold+D@cold[sources])+teleport
                if np.linalg.norm(new-cold, ord=1) <= 1e-13*(1-args.alpha):
                    cold = new; break
                cold = new
            cold_ms = (perf_counter()-t)*1000
            cold_error = float(np.max(np.abs(cold-exact)))
            assert cold_error < 1e-9
        reference_order = np.argsort(-exact, kind="stable")
        agree = all(set(certified[k].indices) == set(reference_order[:k]) for k in certified)
        assert agree
        for certified_result in certified.values():
            assert certified_result.score_error <= certified_result.bound+1e-12
            assert np.all(np.abs(certified_result.scores-exact) <= certified_result.error_bounds+1e-12)
        # Gold is evaluation only, after every index/seed/view computation.
        gold = {title for title, _ in item["supporting_facts"]}
        titles = [d[0] for d in docs]
        for method, scores in [("BM25", bm25), ("AssociationPPR", y), ("EntityPolicyPPR", exact)]:
            rows.append(dict(setting="derived_fixed_1000context_pool", method=method, qid=item["_id"],
                             question_index=qi, gold_pool_coverage=len(set(titles)&gold)/len(gold),
                             **retrieve_metrics(np.argsort(-scores, kind="stable"), titles, gold)))
        cert_metrics = {}
        for k, certified_result in certified.items():
            predicted = {titles[i] for i in certified_result.indices}
            cert_metrics[f"recall_at_{k}"] = len(predicted & gold)/len(gold)
            cert_metrics[f"all_support_at_{k}"] = int(gold <= predicted)
        rows.append(dict(setting="derived_fixed_1000context_pool", method="CertifiedEntityPolicyPPR", qid=item["_id"],
                         question_index=qi, gold_pool_coverage=len(set(titles)&gold)/len(gold), **cert_metrics))
        execution.append(dict(qid=item["_id"], question_index=qi, n=n, rank=len(sources),
                              sources=sources.tolist(), unseen_atomic_responses=missing,
                              lexical_ms=lexical_ms, base_solve_ms=base_ms,
                              response_prepare_ms=response_prepare_ms, exact_correction_ms=exact_correction_ms,
                              compression_prepare_ms=compressed.prepare_seconds*1000,
                              certified_correction_ms=result.solve_seconds*1000,
                              certified_all_k_correction_ms=sum(r.solve_seconds for r in certified.values())*1000,
                              warm_pi_ms=pi_ms, pi_iterations=iterations,
                              cold_view_pi_ms=cold_ms, cold_view_pi_iterations=cold_iterations,
                              cold_view_pi_error=cold_error,
                              initial_certified=result.initial_certified, refined_columns=result.refined_columns,
                              exact_fallback=result.exact_fallback, top10_set_agreement=agree,
                              score_error=result.score_error, score_bound=result.bound,
                              dense_response_bytes=Z.nbytes, compressed_response_bytes=compressed.response_bytes,
                              compressed_object_bytes=compressed.nbytes,
                              exact_residual=vr.residual(seed, exact, prepared)))
        if (qi+1) % 100 == 0:
            print(json.dumps(dict(stage="pool", questions=qi+1, seconds=perf_counter()-start,
                                  atomic_responses=len(zcache), initial_cert_rate=np.mean([r["initial_certified"] for r in execution]))), flush=True)
    pool_meta = dict(pool_context_questions=args.pool_questions, pool_nodes=n, pool_nnz=P.nnz,
                     duplicate_title_text_conflicts=conflicts, graph_seconds=graph_seconds,
                     base_factor_seconds=vr.base_factor_seconds, atomic_cached_sources=len(zcache),
                     atomic_response_cache_hits=atomic_hits, distinct_source_templates=len(templates),
                     dense_evaluator_atomic_cache_bytes=sum(z.nbytes for z in zcache.values()),
                     compressed_note="Per-view compressed payload; dense atomic response cache retained by evaluator, so no total-process compression claim.")
    if hasattr(vr.base_lu, "max_returned_residual"):
        pool_meta.update(backend="matrix_free_fixed_point", backend_tolerance=vr.base_lu.tol,
                         base_matrix_payload_bytes=vr.base_payload_bytes,
                         backend_calls=vr.base_lu.calls, backend_rhs_columns=vr.base_lu.rhs_columns,
                         backend_iterations=vr.base_lu.iterations,
                         backend_max_returned_residual=vr.base_lu.max_returned_residual)
    return rows, execution, pool_meta


def main(args):
    path = ROOT/args.input
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = dict(source=json.loads((ROOT/"data/hotpot_source.json").read_text()),
                official_dev_questions=len(data), query_count=args.queries or len(data),
                alpha=args.alpha, keep_per_response=args.keep, pool_questions=args.pool_questions,
                protocol="public-hotpot-no-llm-retrieval-v1", ordering="Official file order; no seeds selected; no gold fields used for graph, seed or view.",
                metrics="Supporting-title passage Recall@k and all-support@k, not official sentence-support F1 or answer accuracy.",
                scope="Distractor candidate ranking plus explicitly derived fixed-context-pool retrieval, not official fullwiki.")
    rows, execution = [], []
    if args.mode in ["distractor", "both"]:
        rows += distractor(data, args)
    if args.mode in ["pool", "both"]:
        r, execution, extra = pooled(data, args); rows += r; meta.update(extra)
    write_outputs(ROOT/args.output, rows, execution, meta)
    print(json.dumps(summarize(rows)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/hotpot_dev_distractor_v1.json")
    p.add_argument("--output", default="results/public_qa_pilot")
    p.add_argument("--mode", choices=["distractor", "pool", "both"], default="both")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--pool-questions", type=int, default=1000)
    p.add_argument("--keep", type=int, default=64)
    p.add_argument("--alpha", type=float, default=.5)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        main(args)
