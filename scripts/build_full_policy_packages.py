"""Build all-record public policy inputs without touching the frozen prefixes.

Numerical execution and gold scoring are separate.  The existing index,
graph, seed and complete-source helpers are reused without changing them.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from natural_policy_workload import digest, natural_statistics, write_json, write_jsonl
from public_qa_benchmark import Index, graph, local_view, seed_vector

OUT = ROOT / "results/natural_policy_workload"
DATASETS = {"hotpot_pool7405": ("hotpot_dev_distractor_v1.json", 7405),
            "musique_pool2417": ("musique_ans_v1.0_dev.jsonl", 2417)}
STRENGTHS, KS = [0.5, 1., 2., 4.], [2, 5, 10]
IDENTITY = "global key=(title, exact paragraph text, same-question identical-content occurrence ordinal); explicit original context/paragraph index mapping; no title-only merging"


def public_records(name, raw):
    """Catalog text and question fields only; do not access support annotations."""
    catalog, docs, questions, mapping, local_maps = {}, [], [], [], {}
    hotpot = name.startswith("hotpot")
    for qi, item in enumerate(raw):
        qid = item["_id"] if hotpot else item["id"]
        questions.append(dict(qid=qid, question=item["question"], original_question_index=qi))
        paragraphs = [(i, title, " ".join(sentences)) for i, (title, sentences) in enumerate(item["context"])] if hotpot else [
            (paragraph["idx"], paragraph["title"], paragraph["paragraph_text"]) for paragraph in item["paragraphs"]]
        occurrence, local, by_title = Counter(), {}, defaultdict(list)
        for original_idx, title, text in paragraphs:
            pair = title, text
            key = (*pair, occurrence[pair]); occurrence[pair] += 1
            if key not in catalog:
                catalog[key] = len(docs)
                docs.append(pair)
            node = catalog[key]
            assert original_idx not in local
            local[original_idx] = node
            by_title[title].append(node)
            mapping.append(dict(qid=qid, **{("context_index" if hotpot else "paragraph_idx"): original_idx}, doc_id=node))
        assert len(local) == len(set(local.values()))
        local_maps[qid] = by_title if hotpot else local
    assert len({q["qid"] for q in questions}) == len(questions)
    return docs, questions, mapping, local_maps


def scorer_annotations(name, raw, local_maps):
    """Only the separate evaluator output receives these annotation fields."""
    result = []
    hotpot = name.startswith("hotpot")
    for item in raw:
        qid = item["_id"] if hotpot else item["id"]
        local = local_maps[qid]
        if hotpot:
            titles = {title for title, _ in item["supporting_facts"]}
            assert all(len(local[title]) == 1 for title in titles)
            result.append(dict(qid=qid, support_doc_ids=sorted(local[title][0] for title in titles)))
        else:
            gold = [local[p["idx"]] for p in item["paragraphs"] if p["is_supporting"]]
            result.append(dict(qid=qid, support_doc_ids=sorted(gold), original_candidate_indices=sorted(local)))
    return result


def input_plan(queries):
    stencils = defaultdict(set)
    for q in queries:
        for source in q["sources"]:
            stencils[source].update(q["sources"])
    sizes = [len(v) for v in stencils.values()]
    return dict(scope="Public input source co-occurrence only; no gold, no solver responses, no cache timing.",
                all_questions=len(queries), distinct_sources=len(stencils),
                source_interaction_coordinates=sum(sizes),
                full_union_coordinates=len(stencils)**2,
                exact_stencil_value_bytes_if_all_atoms_resident=8 * sum(sizes),
                exact_full_union_value_bytes_if_all_atoms_resident=8 * len(stencils)**2,
                stencil_size_quantiles={str(q): float(np.quantile(sizes, q)) for q in [0, .5, .95, 1]},
                cooccurrence_rows={str(s): sorted(rows) for s, rows in sorted(stencils.items())})


def build(name):
    started = perf_counter()
    source = ROOT / "data" / DATASETS[name][0]
    raw = (json.loads(source.read_text(encoding="utf-8")) if name.startswith("hotpot") else
           [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()])
    assert len(raw) == DATASETS[name][1]
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    docs, questions, mapping, local_maps = public_records(name, raw)
    print(json.dumps(dict(stage="catalog", dataset=name, questions=len(questions), nodes=len(docs))), flush=True)
    t = perf_counter(); index = Index(docs); index_seconds = perf_counter() - t
    print(json.dumps(dict(stage="index", dataset=name, seconds=index_seconds)), flush=True)
    t = perf_counter(); P, entity = graph(index); graph_seconds = perf_counter() - t
    assert np.max(np.abs(np.asarray(P.sum(axis=0)).ravel() - 1)) < 1e-12
    print(json.dumps(dict(stage="graph", dataset=name, nodes=P.shape[0], nnz=P.nnz, seconds=graph_seconds)), flush=True)
    group_rows, group_cols, group_values = [], [], []
    for s in range(P.shape[0]):
        column = P[:, s]
        for target, value in zip(column.indices, column.data):
            if int(target) in entity[s] and target != s:
                group_rows.append(int(target)); group_cols.append(s); group_values.append(float(value))
    G = sparse.csc_matrix((group_values, (group_rows, group_cols)), shape=P.shape)
    masses = np.asarray(G.sum(axis=0)).ravel()
    prepared = []
    t = perf_counter()
    for qi, q in enumerate(questions):
        bm25 = index.bm25(q["question"])
        seed = seed_vector(bm25)
        sources, _ = local_view(P, entity, index, q["question"], bm25)
        support = np.flatnonzero(seed)
        prepared.append(dict(**q, seed_indices=support.tolist(), seed_values=seed[support].tolist(),
                             sources=sources.tolist(), source_group_masses=masses[sources].tolist(),
                             requested_strengths=STRENGTHS, requested_k=KS))
        if (qi + 1) % 1000 == 0:
            print(json.dumps(dict(stage="queries", dataset=name, completed=qi + 1, seconds=perf_counter() - t)), flush=True)
    query_seconds = perf_counter() - t
    write_jsonl(folder / "docs.jsonl", [dict(id=i, title=title, text=text) for i, (title, text) in enumerate(docs)])
    write_jsonl(folder / "queries.jsonl", prepared)
    sparse.save_npz(folder / "graph.npz", P)
    sparse.save_npz(folder / "mention_subdistribution.npz", G)
    mapping_name = "context_mapping.jsonl" if name.startswith("hotpot") else "paragraph_mapping.jsonl"
    write_jsonl(folder / mapping_name, mapping)
    # The public engine files are complete before support annotations are read.
    evaluation = scorer_annotations(name, raw, local_maps)
    write_jsonl(folder / "evaluation.jsonl", evaluation)
    reuse = natural_statistics(prepared)
    write_json(folder / "natural_reuse.json", reuse)
    plan = input_plan(prepared)
    write_json(folder / "source_interaction_plan.json", plan)
    identity_stats = Counter(title for title, _ in docs)
    nondegenerate = sum(any(1e-14 < m < 1 - 1e-14 for m in q["source_group_masses"]) for q in prepared)
    manifest = dict(protocol="natural-anchor-policy-sensitivity-v1", extension="all-official-development-records-v1",
                    dataset=name, status="prepared, not numerical execution results", input_path=str(source),
                    input_sha256=digest(source), question_count=len(prepared), nodes=P.shape[0], transition_nnz=P.nnz,
                    mention_group_nnz=G.nnz, doc_identity=IDENTITY, alpha=.5, strengths=STRENGTHS,
                    requested_k=KS, source_arrays_original_and_untruncated=True,
                    source_events=sum(len(q["sources"]) for q in prepared),
                    zero_group_source_events=sum(m <= 1e-14 or m >= 1 - 1e-14 for q in prepared for m in q["source_group_masses"]),
                    nondegenerate_questions=nondegenerate, nonidentity_nondegenerate_targets=3 * nondegenerate,
                    unique_titles=len(identity_stats), titles_with_multiple_paragraph_identities=sum(v > 1 for v in identity_stats.values()),
                    mapping_records=len(mapping), targets=len(prepared) * 4, requested_sets_per_repeat=len(prepared) * 12,
                    ordering="all records in original public-file order, no result-based record selection",
                    old_prefix_packages="Unchanged. Expanded-corpus BM25/graph/anchors are recomputed using the same helpers, not copied from a different graph.",
                    cost_seconds=dict(index=index_seconds, graph=graph_seconds, queries=query_seconds, package=perf_counter() - started),
                    consumer=dict(P0="graph.npz, column-stochastic", p_G="mention_subdistribution.npz, subdistribution of P0 with no self loops",
                                  seed="queries.jsonl sparse seed_indices/seed_values", S="queries.jsonl complete sources; no boundary projection",
                                  view="For each s in S, p(lambda)=(P0[:,s]+(lambda-1)*p_G[:,s])/(1+(lambda-1)*m_s)",
                                  gold="evaluation.jsonl is scorer-only; no annotation enters engine files"),
                    files={filename: dict(bytes=(folder / filename).stat().st_size, sha256=digest(folder / filename))
                           for filename in ["docs.jsonl", "queries.jsonl", "evaluation.jsonl", "graph.npz", "mention_subdistribution.npz", mapping_name,
                                            "natural_reuse.json", "source_interaction_plan.json"]},
                    generator_sha256={str(path.relative_to(ROOT)): digest(path) for path in [Path(__file__), ROOT / "scripts/public_qa_benchmark.py", ROOT / "scripts/natural_policy_workload.py"]})
    if name.startswith("musique"):
        manifest["hop_mix"] = dict(sorted(Counter(q["qid"].split("__", 1)[0] for q in questions).items()))
    write_json(folder / "manifest.json", manifest)
    snapshot = folder / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in [Path(__file__), ROOT / "scripts/public_qa_benchmark.py", ROOT / "scripts/natural_policy_workload.py"]:
        shutil.copyfile(path, snapshot / path.name)
    print(json.dumps(dict(stage="ready", dataset=name, folder=str(folder), nodes=P.shape[0], nnz=P.nnz,
                          questions=len(prepared), seconds=perf_counter() - started,
                          stencil_coordinates=plan["source_interaction_coordinates"], unique_sources=plan["distinct_sources"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        for name in args.datasets:
            build(name)
