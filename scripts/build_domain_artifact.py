"""Plan the ViewRank release locally; --build streams archives only after readiness.

Default --plan performs static Python dependency inspection, file-stat inventory
and small manifest/summary checks. It never opens a graph, question trace, proof
NPZ, numerical backend or network connection. No large file is hashed/copied.
Build requires both main runs, all16 formal supplementary runs, and root's final
readiness flag with the exact final paper PDF hash. No publication/upload occurs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ROUND = Path("E:/ICDE2027-research-data/expanded-contexts-v1/round_domain_v3")
SUPPLEMENT = ROOT/"results/domain_supplement_v3"
COHORTS = (["A_core_matched_"+s for s in ("hot", "mu")] +
           ["B_own_matched_"+s for s in ("hot", "mu")] +
           [f"C_certificate_budget{b}_{s}" for b in (1, 4, 16) for s in ("hot", "mu")] +
           [f"D_single_{m}_{s}" for m in
            ("fine_preview", "dp_preview_fine", "lift_sparse_fine_preview")
            for s in ("hot", "mu")])
ENTRYPOINTS = (
    "benchmark_policy_domains.py", "benchmark_policy_domains_supplement.py",
    "benchmark_multigroup.py", "natural_domain_adapter.py",
    "pin_timing_worker.py", "process_memory.py",
    "build_full_policy_packages.py", "public_qa_benchmark.py",
    "natural_policy_workload.py", "musique_benchmark.py",
    "summarize_policy_domains.py", "summarize_domain_supplement.py",
    "make_domain_results.py", "make_domain_supplement_results.py",
    "make_domain_performance.py", "make_domain_mechanism.py", "make_domain_budget.py",
    "make_domain_source_count.py",
    "polynomial_core_independent_check.py", "natural_contract_independent_check.py",
    "selfcheck_policy_domain_polynomial.py", "build_domain_artifact.py")
INPUT_FILES = (
    "graph.npz", "mention_subdistribution.npz", "queries.jsonl", "manifest.json",
    "docs.jsonl", "evaluation.jsonl",
    "source_interaction_plan.json", "natural_reuse.json")
EVIDENCE_DIRS = (
    "independent_formal_hot_main_v3_v1", "independent_formal_mu_main_v3_v1",
    "independent_formal_all_witness_v1", "independent_formal_truth_v3_v1",
    "full_response_contract_audit_v1", "exact_column_mass_audit_v1")
EVIDENCE_FILES = (
    "verify_policy_domains_v3_full_v3.py", "environment_v3.json",
    "complete_workload_protocol_v3.json", "independent_natural_pair_check.py",
    "independent_natural_pair_check.json", "natural_dependence_examples_v1.py",
    "natural_dependence_examples_v1.json", "natural_dependence_examples_v1.txt",
    "natural_dependence_examples_case_text_v1.txt",
    "domain_performance_qa_v1.json", "domain_performance_qa_v1.txt")
BINARY_SUFFIXES = {".dll", ".exe", ".lib", ".exp", ".obj", ".pdb", ".so", ".dylib"}
PRIVATE_NAMES = ("author_metadata", "author_retrieval", "finance", "dllm", "chat", "thread")
SKIP_PARTS = {"__pycache__", ".git", ".pytest_cache", "history", "drafts"}


@dataclass(frozen=True)
class Entry:
    source: Path
    member: str
    pack: str
    category: str
    size: int
    mtime_ns: int


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def small_run_gate(path):
    """Only final small metadata; no trace parsing or historical rewrite."""
    try:
        raw = (path/"job_manifest.json").read_bytes()
        job = json.loads(raw)
        result = read(path/"summary.json")
        ok = (result.get("source_drift") == [] and
              result["complete_blocks"] == result["expected_blocks"] == job["blocks"] and
              result["job_manifest_sha256"] == hashlib.sha256(raw).hexdigest())
        return dict(path=str(path), status="COMPLETE" if ok else "NOT_COMPLETE",
                    expected_blocks=job["blocks"], complete_blocks=result["complete_blocks"],
                    questions=job["questions"], source_drift=result.get("source_drift"))
    except (OSError, ValueError, KeyError) as error:
        return dict(path=str(path), status="PENDING", reason=str(error))


def supplement_cohorts(args):
    """The final protocol, not directory-name inference, defines formal outputs."""
    protocol = read(args.capacity_protocol)
    cohorts = protocol["cohorts"]
    if (protocol["version"] != "continuous-domain-supplement-certificate-capacity-v4" or
            len(cohorts) != 16 or {c["id"] for c in cohorts} != set(COHORTS)):
        raise ValueError("Final capacity protocol must declare exactly16 formal cohorts")
    return cohorts


def excluded(path):
    lower = path.as_posix().lower()
    return (bool(SKIP_PARTS.intersection(path.parts)) or
            path.suffix.lower() in BINARY_SUFFIXES or
            any(x in lower for x in PRIVATE_NAMES) or
            "/baselines/prpack/" in lower or
            "/baselines/build/" in lower or
            "_pre_review" in path.name or path.is_symlink())


def inventory(args):
    entries, missing, omitted = {}, [], []
    proof_index = read(args.proof_index) if args.proof_index.is_file() else {}
    required_proofs = {Path(p).resolve() for p in proof_index.get("required_npz_paths", [])}

    def add(path, member, category, pack="core", required=False):
        path = Path(path)
        if not path.is_file():
            if required:
                missing.append(str(path))
            return
        if excluded(path):
            omitted.append(dict(path=str(path), reason="excluded third-party binary/private/history"))
            return
        if pack == "proofs" and not args.all_proof_copies:
            main0 = (category == "main_complete_records" and
                     "/fine_preview/audit/b0_q" in member)
            independent = category in ("independent_checks", "additional_final_review")
            if not (main0 or independent or path.resolve() in required_proofs):
                omitted.append(dict(path=str(path), bytes=path.stat().st_size,
                    reason="OPTIONAL_REDUNDANT_PROOF_COPY; use --all-proof-copies"))
                return
        stat = path.stat()
        key = (pack, member)
        value = Entry(path, member, pack, category, stat.st_size, stat.st_mtime_ns)
        if key in entries and entries[key].source.resolve() != path.resolve():
            raise ValueError("Conflicting package member: "+member)
        entries[key] = value

    def tree(folder, member_root, category, separate_npz=False):
        folder = Path(folder)
        if not folder.is_dir():
            return
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                pack = "proofs" if separate_npz and path.suffix.lower() == ".npz" else "core"
                add(path, (Path(member_root)/path.relative_to(folder)).as_posix(),
                    category, pack)

    # Preserve original project layout; only project-owned sources/tests.
    tree(ROOT/"src", "src", "own_source")
    tree(ROOT/"tests", "tests", "tests")
    script_names = set(ENTRYPOINTS) | {p.name for p in (ROOT/"scripts").glob("make_domain*.py")}
    for folder in (ROOT/"src", ROOT/"tests"):
        for path in folder.rglob("*.py"):
            if not excluded(path):
                for module in local_imports(path):
                    helper = module.split(".")[0]+".py"
                    if (ROOT/"scripts"/helper).is_file():
                        script_names.add(helper)
    queue = list(script_names)
    while queue:
        name = queue.pop()
        path = ROOT/"scripts"/name
        if not path.exists():
            continue
        for module in local_imports(path):
            helper = module.split(".")[0]+".py"
            if (ROOT/"scripts"/helper).is_file() and helper not in script_names:
                script_names.add(helper)
                queue.append(helper)
    for name in sorted(script_names):
        add(ROOT/"scripts"/name, "scripts/"+name, "scripts", required=True)
    for name in ("CMakeLists.txt", "prpack_provenance.json", "prpack_upstream.patch"):
        add(ROOT/"baselines"/name, "baselines/"+name, "external_build_recipe", required=True)
    for name in ("requirements-lock.txt", "requirements-expanded-corpus.txt", "pyproject.toml"):
        add(ROOT/name, name, "environment", required=True)
    for name in ("domain_artifact_plan_v3.txt", "domain_artifact_reproduction_v3.txt",
                 "domain_prpack_credit.txt", "domain_independent_theory_review_v3.txt"):
        add(ROOT/"notes"/name, "notes/"+name, "documentation", required=True)
    add(args.readiness, "notes/domain_final_readiness_v3.json", "final_readiness", required=True)
    for name in ("hotpot_pool7405", "musique_pool2417"):
        folder = ROOT/"results/natural_policy_workload"/name
        for filename in INPUT_FILES:
            add(folder/filename, f"results/natural_policy_workload/{name}/{filename}",
                "complete_inputs", required=True)
        mapping = "context_mapping.jsonl" if name.startswith("hotpot") else "paragraph_mapping.jsonl"
        add(folder/mapping, f"results/natural_policy_workload/{name}/{mapping}",
            "complete_inputs", required=True)
    for name in ("hotpot_dev_distractor_v1.json", "musique_ans_v1.0_dev.jsonl",
                 "hotpot_source.json", "musique_source.json"):
        add(ROOT/"data"/name, "data/"+name, "public_rebuild_inputs", required=True)
    # Main paper input closure. Never include histories, author retrieval or old figures.
    seen = set()

    def paper(path):
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        add(path, path.relative_to(ROOT).as_posix(), "final_paper", required=True)
        if path.suffix != ".tex" or not path.exists():
            return
        text = path.read_text(encoding="utf-8-sig")
        for target in re.findall(r"\\(?:input|include)\{([^}]+)\}", text):
            p = ROOT/"paper"/target
            if not p.suffix:
                p = p.with_suffix(".tex")
            paper(p)
        for target in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            add(ROOT/"paper"/target, (ROOT/"paper"/target).resolve().relative_to(ROOT).as_posix(),
                "final_figures", required=True)
    paper(ROOT/"paper/main.tex")
    add(ROOT/"paper/main.pdf", "paper/main.pdf", "final_paper", required=True)
    for path in sorted((ROOT/"figures").glob("domain_*")):
        add(path, "figures/"+path.name, "final_figures")
    for path in sorted((ROOT/"paper/generated").glob("domain_*")):
        add(path, "paper/generated/"+path.name, "final_paper")
    add(ROOT/"paper/generated/domain_source_count_facts.json",
        "paper/generated/domain_source_count_facts.json", "final_paper", required=True)

    # Historical bytes stay unchanged. Selected unique NPZ evidence goes to a
    # companion archive with the original relative extraction paths.
    for short in ("hot", "mu"):
        name = f"full_{short}_main_v3"
        tree(args.round_root/name, "results/domain_main_v3/"+name,
             "main_complete_records", separate_npz=True)
    for cohort in supplement_cohorts(args):
        output = Path(cohort["output"])
        member = output.relative_to(args.supplement_root).as_posix()
        tree(output, "results/domain_supplement_v3/"+member,
             "supplement_complete_records", separate_npz=True)
    tree(args.supplement_root/"evidence_summary_v3",
         "results/domain_supplement_v3/evidence_summary_v3", "supplement_report")
    add(args.supplement_root/"evidence_summary_v3/report.json",
        "results/domain_supplement_v3/evidence_summary_v3/report.json",
        "supplement_report", required=True)
    tree(args.supplement_root/"independent_verification_v3",
         "results/domain_supplement_v3/independent_verification_v3", "independent_checks", True)
    for folder in EVIDENCE_DIRS:
        tree(args.round_root/folder, "evidence/"+folder, "independent_checks", True)
    tree(args.round_root/"complete_main_summary_v3",
         "evidence/complete_main_summary_v3", "main_report")
    for name in EVIDENCE_FILES:
        add(args.round_root/name, "evidence/"+name, "independent_checks", required=True)
    for path in args.round_root.glob("natural_instability_*v*.*"):
        if path.suffix in (".json", ".txt"):
            add(path, "evidence/"+path.name, "natural_case_evidence")
    # Current formal protocol includes preserved A/B/D and six witness-off C jobs.
    protocol = args.capacity_protocol
    for name in ("protocol_v4.json", "run_certificate_capacity_v4.py",
                 "prepare_certificate_capacity_v4.py", "README_v4.txt",
                 "reporting_manifest_v4.json", "reporter_plan_v4.json",
                 "source_manifest_v4.json", "output_paths_v4.json"):
        add(protocol.parent/name, "protocols/certificate_capacity_v4/"+name,
            "frozen_protocol", required=True)
    for cohort in supplement_cohorts(args):
        rel = Path(cohort["plan_file"])
        add(protocol.parent/rel, "protocols/certificate_capacity_v4/"+rel.as_posix(),
            "frozen_protocol", required=True)
    # Prior witness-on C plans and interrupted records are history, never metrics
    # or formal completion gates. Retain original bytes in an isolated subtree.
    prior = args.round_root/"supplement_v3/protocol_v3.json"
    history = "engineering_history/capacity_v3/"
    add(prior, history+"protocol_v3.json", "engineering_history", required=True)
    add(args.round_root/"supplement_v3/run_supplement_v3_v3.py",
        history+"run_supplement_v3_v3.py", "engineering_history", required=True)
    if prior.exists():
        for cohort in read(prior)["cohorts"]:
            rel = Path(cohort["plan_file"])
            add(prior.parent/rel, history+rel.as_posix(),
                "engineering_history", required=True)
    tree(args.supplement_root/"C_budget1_hot", history+"C_budget1_hot",
         "engineering_history")
    for path in args.extra_evidence:
        path = path.resolve()
        if path.is_dir():
            tree(path, "evidence/additional/"+path.name, "additional_final_review", True)
        else:
            add(path, "evidence/additional/"+path.name, "additional_final_review", required=True)
    if args.proof_index.is_file():
        add(args.proof_index, "evidence/artifact_proof_index.json",
            "independent_checks", required=True)
    for path in required_proofs:
        if path.is_relative_to(args.supplement_root.resolve()):
            member = "results/domain_supplement_v3/"+path.relative_to(
                args.supplement_root.resolve()).as_posix()
        elif path.is_relative_to(args.round_root.resolve()):
            rel = path.relative_to(args.round_root.resolve())
            member = ("results/domain_main_v3/" if rel.parts[0].startswith("full_")
                      else "evidence/")+rel.as_posix()
        else:
            member = "evidence/additional_required_npz/"+path.name
        add(path, member, "independent_checks", "proofs", required=True)
    # Root may name final independent review files in its readiness record.
    if args.readiness.exists():
        ready = read(args.readiness)
        def review_strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, list):
                for item in value:
                    yield from review_strings(item)
            elif isinstance(value, dict):
                for item in value.values():
                    yield from review_strings(item)
        for value in (s for key, value in ready.items() if "review" in key.lower()
                      for s in review_strings(value)):
            path = Path(value)
            if not path.is_absolute():
                path = ROOT/path
            if path.is_file():
                add(path, "evidence/final_reviews/"+path.name, "final_review")
    return sorted(entries.values(), key=lambda e: (e.pack, e.member)), missing, omitted


def local_imports(path):
    result = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def dependency_audit(entries):
    own_modules, imports = set(), set()
    for e in entries:
        if e.source.suffix == ".py" and e.category in ("own_source", "tests", "scripts"):
            own_modules.add(e.source.stem)
            imports.update(local_imports(e.source))
    external = sorted({x.split(".")[0] for x in imports}-
                      set(sys.stdlib_module_names)-own_modules-{"src", "scripts", "tests", "baselines"})
    return dict(external_top_level_imports=external,
        required_python="3.10.6; exact package versions in requirements-lock.txt",
        native="Windows x64 MSVC/CMake; rebuild pinned PRPACK, do not ship upstream/DLL.",
        figure_qa="audit_panel_alignment is a host QA-only dependency, not distributed; final plots/CSV and QA reports are included. Measurement and numerical tests do not use it.",
        portability="Historical absolute paths/hashes are retained. New experiments use explicit new input-root/output and create a fresh freeze; no cross-compiler DLL-byte identity is promised.")


def plan(args):
    entries, missing, omitted = inventory(args)
    gates = [small_run_gate(args.round_root/f"full_{s}_main_v3") for s in ("hot", "mu")]
    gates += [dict(id=c["id"], group=c["group"],
                   outcome_scope=("CERTIFICATE_CAPACITY_WITNESS_OFF" if c["group"] == "C"
                                  else "FULL_THREE_STATE_OPERATOR"),
                   **small_run_gate(Path(c["output"]))) for c in supplement_cohorts(args)]
    by_category = defaultdict(lambda: dict(files=0, bytes=0))
    for e in entries:
        v = by_category[e.pack+"/"+e.category]
        v["files"] += 1
        v["bytes"] += e.size
    proof_counts = Counter("block0" if re.search(r"(?:^|/)b0_q", e.member) else
                           "later_blocks_or_other" for e in entries if e.pack == "proofs")
    by_cohort = defaultdict(Counter)
    for e in entries:
        match = re.search(r"/audit/b(\d+)_q\d+\.npz$", e.member)
        if e.pack == "proofs" and match:
            by_cohort[e.member.rsplit("/audit/", 1)[0]]["block"+match[1]] += 1
    truth_path = args.round_root/"independent_formal_truth_v3_v1/independent_counts.json"
    truth = read(truth_path) if truth_path.exists() else {}
    proof_truth = {key: truth.get(key) for key in (
        "status", "real_original_questions", "unique_primary_question_cuts",
        "primary_unique_statuses", "primary_scalar_proof_instances_covered",
        "primary_scalar_proofs_executed", "primary_identical_numeric_proof_reuses",
        "all_primary_block_statuses_and_stable_memberships_identical")}
    readiness = dict(status="PENDING", path=str(args.readiness))
    if args.readiness.exists():
        r = read(args.readiness)
        readiness.update(ready_for_submission=r.get("ready_for_submission") is True,
                         declared_paper_pdf_sha256=r.get("paper_pdf_sha256"))
    data = dict(mode="PLAN_ONLY_NO_LARGE_COPY", created_utc=datetime.now(timezone.utc).isoformat(),
        destination=str(args.output), cohorts=gates, readiness=readiness,
        formal_supplement_protocol=str(args.capacity_protocol),
        formal_supplement_count=16,
        engineering_history="Old C witness-on plans and interrupted C_budget1_hot prefix are stored separately under engineering_history/capacity_v3 and are excluded from formal cohort gates and metrics.",
        missing_required_files=missing, category_totals=dict(by_category),
        total_uncompressed_bytes={p: sum(e.size for e in entries if e.pack == p)
                                  for p in ("core", "proofs")},
        proof_file_counts=dict(proof_counts), proof_files_by_cohort_spec=dict(by_cohort),
        main_independent_proof_truth=proof_truth,
        proof_policy=("All NPZ copies" if args.all_proof_copies else
            "Default: original main fine_preview block0, independent evidence NPZ, and explicit required proof index. Repeated cohort/block copies optional; all original files remain untouched."),
        proof_index=str(args.proof_index), all_proof_copies=args.all_proof_copies,
        exclusions=omitted, dependencies=dependency_audit(entries),
        files=[dict(source=str(e.source), member=e.member, pack=e.pack,
                    category=e.category, bytes=e.size) for e in entries])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"artifact_plan_v3.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf8")
    instructions = ROOT/"notes/domain_artifact_reproduction_v3.txt"
    if instructions.is_file():
        (args.output/"README_REPRODUCTION.txt").write_bytes(instructions.read_bytes())
    return entries, data


def build(args, entries, data):
    if data["missing_required_files"]:
        raise ValueError("Required files missing; inspect artifact_plan_v3.json")
    if len(data["cohorts"]) != 18 or any(x["status"] != "COMPLETE" for x in data["cohorts"]):
        raise ValueError("Both main and all16 formal supplement runs must be complete and drift-free")
    ready = read(args.readiness)
    if ready.get("ready_for_submission") is not True:
        raise ValueError("Final readiness is not true")
    pdf = ROOT/"paper/main.pdf"
    if ready.get("paper_pdf_sha256") != digest(pdf):
        raise ValueError("Final PDF does not match root's readiness hash")
    required = sum(e.size for e in entries)+128*1024**2
    if shutil.disk_usage(args.output).free < required:
        raise ValueError("Insufficient D: space for conservative uncompressed archive upper bound")
    # No destructive cleanup, recursive move, or overwrite.
    for pack in ("core", "proofs"):
        target = args.output/f"ICDE2027_ViewRank_{pack}.zip"
        temporary = target.with_suffix(".partial.zip")
        if target.exists() or temporary.exists():
            raise FileExistsError("Use a new output directory; archive already exists: "+str(target))
    manifest = dict(final_pdf_sha256=ready["paper_pdf_sha256"], readiness=ready,
        exclusions=data["exclusions"], cohort_gates=data["cohorts"], files=[])
    for pack in ("core", "proofs"):
        target = args.output/f"ICDE2027_ViewRank_{pack}.zip"
        temporary = target.with_suffix(".partial.zip")
        with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            print("ARCHIVE_START", pack, flush=True)
            for index, e in enumerate((e for e in entries if e.pack == pack), 1):
                stat = e.source.stat()
                if (stat.st_size, stat.st_mtime_ns) != (e.size, e.mtime_ns):
                    raise ValueError("Source changed after planning: "+str(e.source))
                h = hashlib.sha256()
                # NPZ is already compressed; retain its bytes without a redundant compression pass.
                info = zipfile.ZipInfo(e.member)
                info.compress_type = zipfile.ZIP_STORED if e.source.suffix == ".npz" else zipfile.ZIP_DEFLATED
                with e.source.open("rb") as source, archive.open(info, "w", force_zip64=True) as dest:
                    for chunk in iter(lambda: source.read(1024**2), b""):
                        h.update(chunk); dest.write(chunk)
                if e.source.stat().st_mtime_ns != e.mtime_ns:
                    raise ValueError("Source changed while archiving: "+str(e.source))
                manifest["files"].append(dict(pack=pack, member=e.member,
                    bytes=e.size, sha256=h.hexdigest(), category=e.category))
                if index % 500 == 0:
                    print("ARCHIVE_FILES", pack, index, flush=True)
            archive.writestr(f"ARCHIVE_SCOPE_{pack}.txt",
                f"{pack} companion archive. Extract both archives into the same directory.\n"+
                "Historical manifests are unchanged. Third-party PRPACK source/DLL excluded.\n"+
                "engineering_history/capacity_v3 is excluded from all formal metrics.\n")
            archive.writestr(f"FILE_MANIFEST_{pack}.json", json.dumps(
                dict(pack=pack, final_pdf_sha256=ready["paper_pdf_sha256"],
                     files=[f for f in manifest["files"] if f["pack"] == pack]), indent=2))
            if pack == "core":
                archive.writestr("README_REPRODUCTION.txt",
                    (ROOT/"notes/domain_artifact_reproduction_v3.txt").read_bytes())
                archive.writestr("PATH_MAP.json", json.dumps({
                    str(ROOT): ".",
                    str(args.round_root/"full_hot_main_v3"): "results/domain_main_v3/full_hot_main_v3",
                    str(args.round_root/"full_mu_main_v3"): "results/domain_main_v3/full_mu_main_v3",
                    str(args.supplement_root): "results/domain_supplement_v3",
                    "note": "Historical provenance only; do not rewrite frozen manifests in place."}, indent=2))
                archive.writestr("PROOF_SCOPE.json", json.dumps({
                    key: data[key] for key in ("proof_policy", "proof_index",
                        "all_proof_copies", "proof_files_by_cohort_spec",
                        "main_independent_proof_truth")}, indent=2))
        temporary.rename(target)
    manifest["archives"] = [dict(name=p.name, bytes=p.stat().st_size, sha256=digest(p))
                            for p in args.output.glob("ICDE2027_ViewRank_*.zip")]
    (args.output/"artifact_manifest_v3.json").write_text(
        json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    print("LOCAL_ARCHIVES_BUILT_NO_UPLOAD", args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Default: stat-only inventory and small metadata gates")
    mode.add_argument("--build", action="store_true", help="After timing/final readiness: stream core and proof archives")
    p.add_argument("--round-root", type=Path, default=ROUND)
    p.add_argument("--supplement-root", type=Path, default=SUPPLEMENT)
    p.add_argument("--capacity-protocol", type=Path,
        help="Final16-cohort protocol; defaults to supplement-root/certificate_capacity_v4/protocol_v4.json")
    p.add_argument("--readiness", type=Path, default=ROOT/"notes/domain_final_readiness_v3.json")
    p.add_argument("--output", type=Path, default=ROOT/"submission/ICDE2027_ViewRank")
    p.add_argument("--extra-evidence", type=Path, nargs="*", default=[])
    p.add_argument("--proof-index", type=Path, default=SUPPLEMENT/
        "independent_verification_v3/proof_payload_index_v3.json",
        help="Small independent-verifier index with required_npz_paths for additional unique stable evidence.")
    p.add_argument("--all-proof-copies", action="store_true",
        help="Also package all later-block and supplementary duplicate NPZ copies.")
    args = p.parse_args()
    args.capacity_protocol = args.capacity_protocol or args.supplement_root/"certificate_capacity_v4/protocol_v4.json"
    entries, data = plan(args)
    print(json.dumps(dict(mode="BUILD" if args.build else "PLAN",
        files=len(entries), uncompressed_bytes=data["total_uncompressed_bytes"],
        complete_cohorts=sum(x["status"] == "COMPLETE" for x in data["cohorts"]),
        expected_cohorts=18, formal_supplement_cohorts=16,
        missing_required=len(data["missing_required_files"]),
        destination=str(args.output)), indent=2))
    if args.build:
        build(args, entries, data)


if __name__ == "__main__":
    main()
