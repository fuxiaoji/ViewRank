"""Natural-anchor planning and declared policy-sensitivity input packages.

Every question occurs once in the package and retains its complete natural
source set. The four strengths are requested outputs of a declared what-if
task, not observed production traffic or copied question requests.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from public_qa_benchmark import Index, graph, local_view, seed_vector

OUT = ROOT / "results/natural_policy_workload"
PROTOCOL = {
    "protocol": "natural-anchor-policy-sensitivity-v1",
    "questions_per_dataset": 1000,
    "ordering": "first 1000 records in the public official-file order; no selected seeds or questions",
    "datasets": ["HotpotQA distractor dev", "MuSiQue-Ans dev"],
    "alpha": 0.5, "requested_k": [2, 5, 10], "strengths": [0.5, 1.0, 2.0, 4.0],
    "group": "explicit title-mention destinations; complement includes every other destination and the self-loop",
    "target": "At each original source, multiply the group's base probabilities by lambda and renormalize the complete outgoing distribution once.",
    "different_from_previous_policy": "The new family is group/background lambda scaling; it is not the earlier EntityPolicyPPR mention*3/weak*0.5/self*1 policy, whose existing quality numbers must not be reused.",
    "question_identity": "one question record requests all four target-policy results; this is a declared sensitivity task, not an observed query stream",
    "source_policy": "same question/title matching and BM25 fallback as the existing public benchmark; rank cap 4 is part of that policy, never a boundary-coverage truncation",
    "hotpot_source_contract": "all 1000 sources arrays must equal the established original-query trace exactly",
    "musique_identity": "global key=(title, paragraph_text, same-question identical-content occurrence ordinal); explicit (qid, paragraph idx)->doc ID mapping; no title-only merging",
    "gold_boundary": "docs/graph/groups/seeds/sources use only public text and question fields; supporting annotations are written separately to evaluation.jsonl",
    "planning": {"train_questions": 200, "test_questions": 800, "budgets": [128, 512, 2048],
                 "retained_output_coordinates": 64,
                 "planners": ["degree", "training_frequency", "training_request_bundle_greedy", "training_CV_degree_frequency_mix"],
                 "CV": "four contiguous 50-query folds within the first 200 only; maximize complete-request coverage, then source-event coverage, then degree fraction",
                 "degree_fractions": [0.0, 0.25, 0.5, 0.75, 1.0],
                 "cache_traces": "cold test and training-warmed test; outside-B views retain all original sources and route as complete fallbacks"},
    "scope": "fixed-public-input exploratory systems workload; no trained reader, production-traffic, or blind-confirmatory claim",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def project_input(name):
    if name == "hotpot_pool1000":
        source = ROOT / "data/hotpot_dev_distractor_v1.json"
        raw = json.loads(source.read_text("utf-8"))[:1000]
        catalog, records, mapping = {}, [], []
        for item in raw:
            for title, sentences in item["context"]:
                text = " ".join(sentences)
                if title in catalog:
                    assert records[catalog[title]][1] == text
                else:
                    catalog[title] = len(records); records.append((title, text))
        questions = [{"qid": item["_id"], "question": item["question"], "original_question_index": qi}
                     for qi, item in enumerate(raw)]
        evaluation = [{"qid": item["_id"], "support_doc_ids": sorted({catalog[title] for title, _ in item["supporting_facts"]})}
                      for item in raw]
        identity = "one node per exact title/text pair; all first1000 Hotpot title texts are consistent"
    else:
        source = ROOT / "data/musique_ans_v1.0_dev.jsonl"
        raw = [json.loads(line) for line in source.read_text("utf-8").splitlines()[:1000]]
        catalog, records, mapping, evaluation = {}, [], [], []
        for item in raw:
            occurrence = Counter(); local = {}; gold = []
            for paragraph in item["paragraphs"]:
                pair = (paragraph["title"], paragraph["paragraph_text"])
                ordinal = occurrence[pair]; occurrence[pair] += 1
                key = (*pair, ordinal)
                if key not in catalog:
                    catalog[key] = len(records); records.append(pair)
                node = catalog[key]
                assert paragraph["idx"] not in local
                local[paragraph["idx"]] = node
                mapping.append({"qid": item["id"], "paragraph_idx": paragraph["idx"], "doc_id": node})
                if paragraph["is_supporting"]:
                    gold.append(node)
            assert len(set(local.values())) == len(local)
            evaluation.append({"qid": item["id"], "support_doc_ids": sorted(gold),
                               "original_candidate_indices": sorted(local)})
        questions = [{"qid": item["id"], "question": item["question"], "original_question_index": qi}
                     for qi, item in enumerate(raw)]
        identity = PROTOCOL["musique_identity"]
    return source, records, questions, evaluation, mapping, identity


def build_package(name):
    started = perf_counter()
    source, docs, questions, evaluation, mapping, identity = project_input(name)
    folder = OUT / name; folder.mkdir(parents=True, exist_ok=True)
    t = perf_counter(); index = Index(docs); index_seconds = perf_counter() - t
    t = perf_counter(); P, entity = graph(index); graph_seconds = perf_counter() - t
    assert np.max(np.abs(np.asarray(P.sum(axis=0)).ravel() - 1)) < 1e-12
    if name == "hotpot_pool1000":
        old = sparse.load_npz(ROOT / "data/hotpot_pool_graph.npz").tocsc()
        difference = P - old
        assert difference.nnz == 0 or np.max(np.abs(difference.data)) < 1e-15
        existing = {row["qid"]: row["sources"] for row in json.loads((ROOT / "results/public_qa_iterative/execution.json").read_text())}
    else:
        existing = None
    group_rows, group_cols, group_data = [], [], []
    for source_node in range(P.shape[0]):
        col = P[:, source_node]
        for target, value in zip(col.indices, col.data):
            if int(target) in entity[source_node] and target != source_node:
                group_rows.append(int(target)); group_cols.append(source_node); group_data.append(float(value))
    G = sparse.csc_matrix((group_data, (group_rows, group_cols)), shape=P.shape)
    masses = np.asarray(G.sum(axis=0)).ravel()
    t = perf_counter(); prepared_queries = []
    for record in questions:
        scores = index.bm25(record["question"]); seed = seed_vector(scores)
        sources, _ = local_view(P, entity, index, record["question"], scores)
        if existing is not None:
            assert sources.tolist() == existing[record["qid"]]
        support = np.flatnonzero(seed)
        prepared_queries.append({**record, "seed_indices": support.tolist(), "seed_values": seed[support].tolist(),
                                 "sources": sources.tolist(), "source_group_masses": masses[sources].tolist(),
                                 "requested_strengths": PROTOCOL["strengths"], "requested_k": PROTOCOL["requested_k"]})
    query_seconds = perf_counter() - t
    write_jsonl(folder / "docs.jsonl", [{"id": i, "title": title, "text": text} for i, (title, text) in enumerate(docs)])
    write_jsonl(folder / "queries.jsonl", prepared_queries)
    write_jsonl(folder / "evaluation.jsonl", evaluation)
    if mapping:
        write_jsonl(folder / "paragraph_mapping.jsonl", mapping)
    sparse.save_npz(folder / "graph.npz", P)
    sparse.save_npz(folder / "mention_subdistribution.npz", G)
    manifest = {
        "protocol": PROTOCOL["protocol"], "dataset": name, "status": "prepared, not numerical execution results",
        "input_path": str(source), "input_sha256": digest(source), "question_count": len(prepared_queries),
        "nodes": P.shape[0], "transition_nnz": P.nnz, "mention_group_nnz": G.nnz,
        "doc_identity": identity, "alpha": PROTOCOL["alpha"], "strengths": PROTOCOL["strengths"],
        "requested_k": PROTOCOL["requested_k"], "source_arrays_original_and_untruncated": True,
        "source_events": sum(len(q["sources"]) for q in prepared_queries),
        "zero_group_source_events": sum(m <= 1e-14 or m >= 1 - 1e-14 for q in prepared_queries for m in q["source_group_masses"]),
        "cost_seconds": {"index": index_seconds, "graph": graph_seconds, "queries": query_seconds, "package": perf_counter() - started},
        "files": {filename: {"bytes": (folder / filename).stat().st_size, "sha256": digest(folder / filename)}
                  for filename in ["docs.jsonl", "queries.jsonl", "evaluation.jsonl", "graph.npz", "mention_subdistribution.npz"]
                  + (["paragraph_mapping.jsonl"] if mapping else [])},
        "consumer": {
            "P0": "graph.npz, column-stochastic",
            "p_G": "mention_subdistribution.npz, each column is a subdistribution of P0",
            "seed": "queries.jsonl sparse seed_indices/seed_values; missing coordinates zero",
            "S": "queries.jsonl sources, preserve complete array; do not project onto any global directory",
            "view": "For each s in S, p(lambda)=(P0[:,s]+(lambda-1)*p_G[:,s])/(1+(lambda-1)*m_s)",
            "gold": "evaluation.jsonl is scorer-only; do not supply it to index, seed, source, or execution planning",
        },
    }
    write_json(folder / "manifest.json", manifest)
    print(json.dumps({"stage": "package_ready", "dataset": name, "folder": str(folder),
                      "nodes": P.shape[0], "queries": len(prepared_queries), "seconds": perf_counter() - started}), flush=True)
    return P, prepared_queries, docs


def natural_statistics(queries):
    frequency = Counter(s for q in queries for s in q["sources"])
    seen, views = set(), set(); hits = complete = whole_hits = new_combination_hits = 0
    for q in queries:
        S = set(q["sources"]); signature = tuple(sorted(S))
        hit = sum(s in seen for s in q["sources"])
        all_hit = bool(S) and S <= seen
        hits += hit; complete += all_hit
        whole_hits += signature in views
        new_combination_hits += all_hit and signature not in views
        seen |= S; views.add(signature)
    events = sum(frequency.values())
    return {"questions": len(queries), "source_events": events, "unique_sources": len(frequency),
            "cross_question_source_hits": hits, "cross_question_source_hit_rate_all_events": hits / events if events else 0,
            "all_sources_previously_seen_questions": complete, "whole_source_set_repeats": whole_hits,
            "new_combination_with_all_sources_preseen": new_combination_hits,
            "frequency_histogram": dict(sorted(Counter(frequency.values()).items())),
            "top_sources": frequency.most_common(25)}


def degree_order(P):
    degree = np.diff(P.indptr)
    return sorted(np.flatnonzero(degree >= 2).tolist(), key=lambda node: (-int(degree[node]), node)), degree


def mixed_boundary(training, order, degree, budget, fraction):
    counts = Counter(s for q in training for s in q["sources"])
    initial = order[:int(budget * fraction)]; chosen = set(initial)
    ranking = sorted(order, key=lambda s: (-counts[s], -int(degree[s]), s))
    return initial + [s for s in ranking if s not in chosen][:budget - len(initial)]


def greedy_boundary(training, order, degree, budget):
    requests = [set(q["sources"]) for q in training]
    candidates = sorted(set(tuple(sorted(S)) for S in requests))
    chosen = set(); frequency = Counter(s for S in requests for s in S)
    while len(chosen) < budget:
        best = None
        for candidate in candidates:
            missing = set(candidate) - chosen
            if not missing or len(missing) > budget - len(chosen):
                continue
            added = chosen | missing
            gain = sum(S <= added and not S <= chosen for S in requests)
            score = (gain / len(missing), sum(frequency[s] for s in missing), sum(int(degree[s]) for s in missing), tuple(-s for s in candidate))
            if best is None or score > best[0]:
                best = (score, missing)
        if best is None:
            break
        chosen |= best[1]
    return sorted(chosen) + [s for s in order if s not in chosen][:budget - len(chosen)]


def coverage(queries, boundary):
    B = set(boundary)
    return sum(set(q["sources"]) <= B for q in queries), sum(s in B for q in queries for s in q["sources"])


def cache_trace(queries, B, initial=None, initial_views=None):
    B = set(B); cache = set(initial or []); seen_views = set(initial_views or []); hits = misses = covered = new_combo = 0
    for q in queries:
        S = set(q["sources"])
        if not S <= B:
            continue
        covered += 1; signature = tuple(sorted(S))
        hit = sum(s in cache for s in S); miss = len(S) - hit
        hits += hit; misses += miss
        new_combo += bool(S) and miss == 0 and signature not in seen_views
        cache |= S; seen_views.add(signature)
    events = sum(len(q["sources"]) for q in queries)
    return cache, {"test_queries": len(queries), "complete_covered_queries": covered,
                   "complete_coverage": covered / len(queries), "complete_fallback_queries": len(queries) - covered,
                   "natural_directory_atomic_hits": hits, "atomic_misses": misses,
                   "hit_rate_all_test_source_events": hits / events if events else 0,
                   "hit_rate_only_served_events": hits / (hits + misses) if hits + misses else 0,
                   "new_whole_views_with_all_atoms_available": new_combo,
                   "cached_atom_count": len(cache)}


def plan_analysis(name, P, queries, docs):
    training, test = queries[:200], queries[200:]
    order, degree = degree_order(P)
    whole_stats = natural_statistics(queries)
    whole_stats["top_source_titles"] = [{"node": s, "title": docs[s][0], "occurrences": count} for s, count in whole_stats["top_sources"]]
    cold_stats = natural_statistics(test)
    seen_train = {s for q in training for s in q["sources"]}
    plans, results = [], []
    for budget in PROTOCOL["planning"]["budgets"]:
        cv = []
        for fraction in PROTOCOL["planning"]["degree_fractions"]:
            full = source_events = 0
            for fold in range(4):
                start, stop = fold * 50, (fold + 1) * 50
                fit = training[:start] + training[stop:]; validate = training[start:stop]
                B = mixed_boundary(fit, order, degree, budget, fraction)
                a, b = coverage(validate, B); full += a; source_events += b
            cv.append({"degree_fraction": fraction, "complete_queries": full, "source_events": source_events})
        selected = max(cv, key=lambda x: (x["complete_queries"], x["source_events"], x["degree_fraction"]))["degree_fraction"]
        boundaries = {
            "degree": order[:budget],
            "training_frequency": mixed_boundary(training, order, degree, budget, 0),
            "training_request_bundle_greedy": greedy_boundary(training, order, degree, budget),
            "training_CV_degree_frequency_mix": mixed_boundary(training, order, degree, budget, selected),
        }
        for method, B in boundaries.items():
            assert len(set(B)) == len(B) == budget
            plans.append({"budget": budget, "method": method, "B": B,
                          "selected_degree_fraction": selected if method.startswith("training_CV") else None,
                          "training_CV": cv if method.startswith("training_CV") else None})
            train_cache, train_trace = cache_trace(training, B)
            for warmth, initial in [("cold_test", set()), ("training_warmed_test", train_cache)]:
                trained_views = ({tuple(sorted(q["sources"])) for q in training if set(q["sources"]) <= set(B)}
                                 if warmth == "training_warmed_test" else set())
                atoms, counts = cache_trace(test, B, initial, trained_views)
                membership = coverage(test, B)[1]
                # Exact layout-size accounting for the declared L64 object;
                # top-L nonzeros have not been solved here, so keep64 is an
                # explicit conservative capacity, not a measured compression.
                delta_nnz = sum(int(degree[s]) for s in atoms)
                bytes_capacity = budget * 8 + len(atoms) * (budget * 8 + 64 * 16 + 40) + delta_nnz * 12
                results.append({"dataset": name, "budget": budget, "method": method, "cache_start": warmth,
                                "train_complete_covered_queries": train_trace["complete_covered_queries"],
                                "training_cached_atoms": len(initial), "test_source_membership_events": membership,
                                "all_test_source_events": sum(len(q["sources"]) for q in test),
                                "directory_payload_capacity_bytes": bytes_capacity,
                                "payload_status": "L64 array-capacity accounting; no response computation/latency claim",
                                **counts})
    folder = OUT / name
    write_json(folder / "B_plans.json", plans)
    with (folder / "planning_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    stats = {"all_1000": whole_stats, "test_800_cold_unrestricted_reuse": cold_stats,
             "training_unique_sources": len(seen_train),
             "test_source_events_seen_in_training": sum(s in seen_train for q in test for s in q["sources"]),
             "test_complete_sources_seen_in_training": sum(set(q["sources"]) <= seen_train for q in test),
             "training_queries": 200, "test_queries": 800,
             "planners_never_change_sources": True, "test_labels_used": False}
    write_json(folder / "natural_reuse.json", stats)
    print(json.dumps({"stage": "natural_planning_ready", "dataset": name, "natural": whole_stats,
                      "test_cold": [r for r in results if r["cache_start"] == "cold_test"]}), flush=True)


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protocol.json", PROTOCOL)
    for name in args.datasets:
        if args.plan_only:
            folder = OUT / name
            P = sparse.load_npz(folder / "graph.npz").tocsc()
            queries = [json.loads(line) for line in (folder / "queries.jsonl").read_text("utf-8").splitlines()]
            docs = [(row["title"], row["text"]) for row in
                    (json.loads(line) for line in (folder / "docs.jsonl").read_text("utf-8").splitlines())]
        else:
            P, queries, docs = build_package(name)
        plan_analysis(name, P, queries, docs)
    write_json(OUT / "script_provenance.json", {"script": str(Path(__file__).resolve()), "sha256": digest(Path(__file__)),
                                               "initial_package_generator_snapshot": "source_snapshot/natural_policy_workload_initial.py",
                                               "initial_package_generator_sha256": digest(OUT / "source_snapshot/natural_policy_workload_initial.py")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["hotpot_pool1000", "musique_pool1000"],
                        default=["hotpot_pool1000", "musique_pool1000"])
    parser.add_argument("--plan-only", action="store_true", help="Recompute planning metadata without touching frozen engine packages")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args)
