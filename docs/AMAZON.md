# Exact Amazon input recipe

Official source: [SNAP Amazon product co-purchasing network and ground-truth communities](https://snap.stanford.edu/data/com-Amazon.html), distributed with Yang and Leskovec, *Defining and Evaluating Network Communities based on Ground-truth*, ICDM 2012.

File: `com-amazon.ungraph.txt.gz` from `https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz`. SHA256: `bb4b9fcf549c9174049e885e9a0a4057c9724846e7297c4a1ea299b8e92eec34`. The hash identifies the exact distributed snapshot; the file header gives no separate version/date. The archive and node mapping accompany the data release.

1. Ignore `#` header lines and read 925,872 unweighted undirected edges.
2. Map the sorted unique official node identifiers to 0-based matrix coordinates (334,863 nodes).
3. Emit both directions of every edge, sum duplicates, and column-normalize. This yields 1,851,744 transition entries and zero diagonal. Columns are source distributions.
4. Select a graph-only directory B of the 128 highest outdegrees; break degree ties by ascending matrix ID. The first eight directory nodes are fixed one-hot seeds.
5. The historical source selector ranks eligible B nodes by decreasing base PPR at alpha .85, then decreasing outdegree, then increasing ID, taking four. The original recovery stored the complete source sets in ascending matrix-coordinate order. The eight authoritative seed/source arrays, including original IDs, are in `results/professor_revision_v1/amazon_probe_inputs/amazon_r4_original/queries.jsonl`. They are copied unchanged for both probes. All eight r=4 requests are retained; the older r=16 requests are outside the implementation and excluded structurally before the probe.
6. At each source s in B, enumerate nonzero destination coordinates in ascending order. Compute SHA256 of UTF-8 `strength-group:{s}:{destination}`; interpret the first eight digest bytes as an unsigned little-endian integer. Stable-sort by this integer and retain the first `max(1, degree(s)//2)` destinations as the first group, with their original transition weights. All remaining nonself destinations form the second group. This is a controlled hash partition, not semantic relation annotation.
7. The original variant uses P and this group M. For the added-self variant with raw self weight c=.5, degree d gives `Pnew = P diag(d/(d+c)) + diag(c/(d+c))` and `Mnew = M diag(d/(d+c))`. Seeds and all source arrays stay fixed.

Both probes use alpha .85, weights `[1,4] × [.25,2]`, k=2/5/10, 16 MiB admission, one complete paired block, and the same declared three methods. Original zero self mass gives a ratio-only policy; added self mass gives two policy directions.

`python scripts/rebuild_amazon_inputs.py --output NEW` rebuilds both graph/group matrices from the included hash-pinned archive and requires exact sparse equality with the prepared inputs. It checks the seed-directory ordering and copies all eight full requests unchanged. This input check was executed successfully for the release.
