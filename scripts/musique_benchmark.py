"""Frozen HotpotQA retrieval policy on all MuSiQue-Ans dev paragraphs.

Uses paragraph idx for evaluation; repeated titles are distinct candidates.
No MuSiQue-dependent parameter selection, LLM, or gold-derived graph input.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from public_qa_benchmark import Index, graph, local_view, seed_vector, STOP
from src.viewrank import ViewRank


def metrics(order, paragraph_ids, gold):
    result = {}
    for k in [2, 5, 10]:
        chosen = [int(paragraph_ids[i]) for i in order[:k]]
        predicted = set(chosen)
        result[f"recall_at_{k}"] = len(predicted & gold)/len(gold)
        result[f"all_support_at_{k}"] = int(gold <= predicted)
        result[f"predicted_idxs_at_{k}"] = chosen
    return result


def summarize(rows):
    result = []
    hops = sorted(set(r["hopcount"] for r in rows))
    for hop in ["all"]+hops:
        for method in ["BM25", "AssociationPPR", "EntityPolicyPPR"]:
            selected = [r for r in rows if r["method"] == method and (hop == "all" or r["hopcount"] == hop)]
            entry = dict(hopcount=hop, method=method, questions=len(selected))
            for k in [2, 5, 10]:
                for name in [f"recall_at_{k}", f"all_support_at_{k}"]:
                    entry[name] = float(np.mean([r[name] for r in selected]))
            result.append(entry)
    return result


def write(out, filename, values):
    (out/f"{filename}.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
    with (out/f"{filename}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)


def run(args):
    path = ROOT/args.input
    data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = data[:args.queries or None]
    rows = []
    started = perf_counter()
    paragraph_counts, hop_counts = Counter(), Counter()
    duplicate_title_cases = 0
    for qi, item in enumerate(selected):
        # The inference construction consumes only the public question,
        # paragraph title and paragraph text. Candidate order is preserved.
        paragraphs = item["paragraphs"]
        docs = [(p["title"], p["paragraph_text"]) for p in paragraphs]
        index = Index(docs)
        P, entity = graph(index)
        bm25 = index.bm25(item["question"])
        seed = seed_vector(bm25)
        ranker = ViewRank(P, alpha=.5)
        y = ranker.base_lu.solve(.5*seed)
        sources, D = local_view(P, entity, index, item["question"], bm25)
        prepared = ranker.prepare(D, sources)
        viewed = ranker.solve(seed, prepared)
        # Evaluator-only fields are read after all three scores are computed.
        paragraph_ids = [int(p["idx"]) for p in paragraphs]
        gold = {int(p["idx"]) for p in paragraphs if p["is_supporting"]}
        match = re.match(r"(\d+)hop", item["id"])
        hopcount = int(match.group(1)) if match else len(item["question_decomposition"])
        hop_counts[hopcount] += 1
        paragraph_counts[len(paragraphs)] += 1
        duplicate_title_cases += len({p["title"] for p in paragraphs}) < len(paragraphs)
        for method, scores in [("BM25", bm25), ("AssociationPPR", y), ("EntityPolicyPPR", viewed)]:
            order = np.argsort(-scores, kind="stable")
            rows.append(dict(qid=item["id"], question_index=qi, method=method,
                             hopcount=hopcount, paragraph_count=len(paragraphs),
                             supporting_paragraphs=len(gold), **metrics(order, paragraph_ids, gold)))
        if (qi+1) % 500 == 0:
            print(json.dumps(dict(questions=qi+1, seconds=perf_counter()-started)), flush=True)
    out = ROOT/args.output; out.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    write(out, "retrieval", rows); write(out, "summary", summary)
    meta = dict(dataset="MuSiQue-Ans v1.0 dev", source=json.loads((ROOT/"data/musique_source.json").read_text()),
                official_dev_questions=len(data), evaluated_questions=len(selected),
                alpha=.5, parameters="Frozen HotpotQA policy, same Index/graph/local_view imports; no MuSiQue tuning.",
                paragraph_counts=dict(paragraph_counts), hop_counts=dict(hop_counts),
                duplicate_title_cases=int(duplicate_title_cases),
                gold_policy="is_supporting and paragraph idx used only after scoring; no title deduplication; hopcount only stratifies evaluation.",
                metrics="Paragraph-idx Recall@2/5/10 and all-support@2/5/10, not answer F1 or official support F1.",
                input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                reused_policy_sha256=hashlib.sha256((ROOT/"scripts/public_qa_benchmark.py").read_bytes()).hexdigest(),
                stopword_sha256=hashlib.sha256("\n".join(sorted(STOP)).encode()).hexdigest(),
                elapsed_seconds=perf_counter()-started)
    (out/"run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps([r for r in summary if r["hopcount"] == "all"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/musique_ans_v1.0_dev.jsonl")
    p.add_argument("--output", default="results/public_qa_musique")
    p.add_argument("--queries", type=int, default=0)
    with threadpool_limits(limits=1):
        run(p.parse_args())
