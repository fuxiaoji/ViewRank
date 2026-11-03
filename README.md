# ViewRank

**Certifying Top-k Stability over Local Graph Policy Domains**  
Wenji Fu, Southwestern University of Finance and Economics  
ORCID: [0009-0003-7757-140X](https://orcid.org/0009-0003-7757-140X)

[Manuscript](paper/main.pdf) · [Reproduction assets](https://github.com/fuxiaoji/ViewRank/releases/tag/v4-submission) · [Table map](docs/TABLE_MAP.md) · [Amazon recipe](docs/AMAZON.md) · [Numerical reruns](docs/EXPERIMENTS.md)

ViewRank asks whether the same top-k node set persists throughout a continuous rectangle of local relation weights. It compiles the modified sources into a small affine system and connects determinant-cleared scores to numerical residual and output-tail bounds. Each cutoff returns **STABLE**, verified **UNSTABLE**, or **UNKNOWN**.

On the complete 3,417-question workloads, three paired blocks save 34.7–35.3% (HotpotQA) and 65.9–66.0% (MuSiQue) against source lifting with matched compressed retention, while resolving more complete audits. The paper also compares competitive pruned full-response controls, budget capacity, process memory, shared-core SMT and additional propagation/topology settings. All formal requests and all outcomes remain in the records.

## Quick reproduction

The code is in Git; larger input graphs, execution ledgers and saved numerical states are release assets. Python 3.10.6 is the recorded environment.

```console
git clone https://github.com/fuxiaoji/ViewRank.git
cd ViewRank
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt -r requirements-professor-revision.txt
.venv/Scripts/python.exe reproduce.py --download --quick
```

On Unix, use `.venv/bin/python` in place of `.venv/Scripts/python.exe` for the saved-data checks. The native timing backend's recorded build recipe targets Windows x64/MSVC.

The command downloads and checks two SHA256-pinned archives, verifies every supplied payload, rebuilds Tables II–IV and VI–IX from complete records, checks Table I's declared method contract and Table V's exact bound rounding, regenerates source-count data, and independently replays two saved scalar certificates. Outputs are written to `reproduced/`. Saved certificate replay checks the supplied algebra and allowances; full-response premise audits are separately included in the evidence.

For individual steps:

```console
python reproduce.py --verify
python reproduce.py --tables --output reproduced_tables
python reproduce.py --proof --output reproduced_proofs
python reproduce.py --paper --output reproduced_paper
```

The last command requires Tectonic and IEEEtran. The manuscript has 12 pages of body; the remaining page contains AI disclosure and references. No appendix is used.

## Further checks

Rebuild both Amazon variants from the included official edge archive and compare their graph/group matrices exactly:

```console
python scripts/rebuild_amazon_inputs.py --output reproduced_amazon
```

Export encoding counts and an SMT-LIB instance by invoking the exact measured encoder without solving:

```console
python scripts/prepare_submission_details.py --frozen-root supplement/frozen_states --output reproduced_encoding
```

Rerun the shared-state verification comparison (40 queries, 120 cutoffs, 5 s solver timeout per cutoff):

```console
python scripts/benchmark_shared_smt.py --frozen-root supplement/frozen_states --queries 20 --timeout-ms 5000 --output reproduced_smt
```

Measured encoding/solve times are separately recorded in `results/submission_details_v4/smt_all_120_cuts.csv`. SMT SAT means a sufficient certificate failed, and is not a verified instability witness.

## Contents

`src/`, `tests/`, `scripts/`: owned implementation and reporting/verification entrypoints.  
`paper/`, `figures/`: current manuscript, vector figures and figure source data.  
`results/domain_main_v3/`: original complete main workloads.  
`results/domain_supplement_v3/`: matched controls, capacity and independent verification.  
`results/professor_revision_v1/`: ten complete extension cohorts and shared-core verifier study.  
`results/submission_details_v4/`: full three-state counts, UNKNOWN reasons, all SMT cutoff times/counts.  
`results/natural_policy_workload/`: prepared lexical input graphs and original seeds/source sets.  
`supplement/frozen_states/`: 40 shared-verifier numerical states.  
`evidence/`: premise audits and full-workload proof/witness records.  
`engineering_history/`: explicitly excluded interrupted/earlier engineering runs.

The primary implementation admits original source sets of size at most four. Lexical main workloads use all original 1,000/2,417 questions and three blocks. Each additional alpha/context setting uses one complete paired block; the Amazon probes use all eight pre-existing r=4 requests. Repeated blocks and cutoffs are not additional independent questions.

The pinned upstream PRPACK source and compiled binaries are retrieved/built separately. Dataset origins and upstream credit are retained in their manifests and `baselines/`. The paper discloses AI assistance. This repository is a research artifact accompanying a manuscript prepared for submission; no conference acceptance is asserted.
