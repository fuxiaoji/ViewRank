# Fresh numerical execution

The quick entrypoint reaggregates original measurements and replays saved proof states. Fresh timing runs require the native backend and a new output directory. They will record their own hardware/compiler/environment and should be compared within each newly paired run.

## Backend

The recorded platform is Windows 11 x64, Python 3.10.6, single-threaded OpenBLAS; hardware and detailed environment are in `evidence/environment_v3.json`. Build with CMake and MSVC x64:

```console
git clone https://github.com/dgleich/prpack baselines/prpack
git -C baselines/prpack checkout 8462aa4161c8c1390455e48046f2ff52c9493cde
git -C baselines/prpack apply ../prpack_upstream.patch
cmake -S baselines -B baselines/build
cmake --build baselines/build --config Release
python -m pytest -q
```

Own CMake/adapter code is supplied. Upstream source/binaries are obtained separately with its license and provenance. The adapter exposes signed, unnormalized response solves; new DLL bytes need not equal historical compiler output.

## Main and matched experiments

Original prepared inputs are under `results/natural_policy_workload/`. Run from the repository root:

```console
python scripts/benchmark_policy_domains.py --phase run --study-phase evaluation --dataset hotpot_pool7405 --input-root results/natural_policy_workload --queries 1000 --blocks 3 --budget-mb 16 --depth 2 --base-tol 1e-10 --execution paired --scan-backend dot_fallback --specs fine_preview,dp_preview_a1e4,dp_preview_fine,lift_dp_preview_fine,native_then_lift_preview_fine --audit-specs fine_preview --witness --output NEW_HOT_MAIN
```

MuSiQue uses `--dataset musique_pool2417 --queries 2417`. The `--help` output describes planning and execution controls. Authoritative manifests for the original main workloads are `results/domain_main_v3/full_{hot,mu}_main_v3/job_manifest.json`. Preserve the original order, domain, cutoffs, atom/base tolerances, depth and complete source arrays.

The supplement wrapper `scripts/benchmark_policy_domains_supplement.py` registers the matched compressed lifting and mechanism-control configurations. Its `--help` exposes the same producer options. Cohort A uses `fine_preview,lift_sparse_fine_preview`, three blocks, witnesses on, and the `two_alternating_else_original` method schedule. The exact matched manifests are `results/domain_supplement_v3/A_core_matched_{hot,mu}/job_manifest.json`.

The original 16-job supplement consists of A (2 matched-cache workloads), B (2 full-workload mechanism-control runs), D (6 fresh single-method processes), and C (6 witness-free capacity runs at 1/4/16 MiB). Manifests under `protocols/certificate_capacity_v4/` and complete output folders retain all original options. C success requires every cutoff STABLE; A/B/D resolution permits STABLE or verified UNSTABLE. Timing includes every attempted request, including UNKNOWN.

## New alpha and topology settings

Use `scripts/run_professor_boundaries.py --produce` with a single new producer command rather than its historical whole-study coordinator. `--produce` registers compressed lifting; producer options then follow. For example:

```console
python scripts/run_professor_boundaries.py --produce --phase run --study-phase evaluation --input-root results/professor_revision_v1/boundary_inputs --dataset hotpot_pool7405_a0p95 --queries 100 --blocks 1 --budget-mb 16 --depth 2 --scan-backend dot_fallback --specs fine_preview,dp_preview_fine,lift_sparse_fine_preview --output NEW_HOT_ALPHA95
```

The six alpha variants and two small lexical variants have prepared input folders in `boundary_inputs/`. For Amazon use `--input-root results/professor_revision_v1/amazon_probe_inputs --dataset amazon_r4_original --queries 8` (or `amazon_r4_self05`). Preserve the published options in each complete `boundaries/<setting>/job_manifest.json`. The old interrupted r=16 engineering attempt is excluded from every new formal comparison.

## Independent evidence

`evidence/` includes complete unique-cut universal-proof records, actual-view witness verification, exact graph-column mass audits and 52 full-response premise audits. The saved scalar checker replays the local polynomial proof conditionally on supplied uniform residual/tail allowances. It does not claim to redo a graph solve or the full-response premise audit. Both evidence layers are preserved and their scopes are distinguished in the manuscript.

## Figures

Final PDF/SVG figures, source CSV/NPZ/JSON data and audit records are supplied. `scripts/make_domain_column_figures.py` and `scripts/report_professor_revision.py` contain the original plotting code. Their optional host-side panel-alignment checker is documented in the code; table reproduction with `--tables-only` has no dependency on that checker. Figures use all formal aggregate data. Selected natural illustrations are labeled as selected cases; main performance claims use the complete workloads.
