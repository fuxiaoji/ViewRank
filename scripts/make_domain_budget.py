"""Prepare the certificate-capacity budget frontier; render only with --render.

Figure contract (before data are read):
  Question: how does retained numerical-state capacity affect whole-domain
  STABLE certification for the polynomial layouts and source-wise lifting?
  Evidence: equal HotpotQA/MuSiQue panels preserve the larger/smaller-graph
  strata, every original question and cutoff, and one witness-off paired block.
  Archetype/backend: quantitative grid; Python/Matplotlib only.
  Export: 183 x 65 mm, editable PDF/SVG, >=5 pt glyphs, source CSV, contract,
  mandatory rendered panel alignment, PDF font and collision audits.

Only the frozen capacity protocol, job_manifest.json and summary.json are opened. No question traces, graph
arrays, solver, or unfinished trace are read. All inputs must be complete and
source-frozen before any output is written. Default execution exports the raw
CSV/contract without opening a plotting backend. --render is an explicit final
step after the frozen supplementary cohorts finish.

The x axis is a numerical-state admission budget that includes family work;
it is not operating-system RSS. All six cohorts disable point witnesses. The
fraction with every cutoff STABLE is distinct from full three-state audit
resolution. Each point has one block; no error bar or confidence interval is drawn.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPACITY = ROOT / "results/domain_supplement_v3/certificate_capacity_v4"
DEFAULT_SKILL = Path.home() / ".codex/skills/nature-figure/scripts"
METHODS = (
    ("fine_preview", "VR-Sparse", "#176c74", "o", "-"),
    ("lift_sparse_fine_preview", "Sparse source lifting", "#737f8c", "^", "--"),
    ("dp_preview_fine", "VR-DensePruned (fine)", "#84b8b7", "s", "-."),
)
MIB = 1024**2
DATASETS = (("hot", "hotpot_pool7405", "HotpotQA", 1000),
            ("mu", "musique_pool2417", "MuSiQue", 2417))
EXPECTED = {
    "fine_preview": dict(mode="sparse", certifier="poly", atom_tol=1e-6,
                         keep=512, preview_keep=128, adaptive=False,
                         rank1_native=False, depth_override=None),
    "lift_sparse_fine_preview": dict(
        mode="sparse", certifier="lifting", atom_tol=1e-6, keep=512,
        preview_keep=128, adaptive=False, rank1_native=False, depth_override=4),
    "dp_preview_fine": dict(mode="dense_pruned", certifier="poly", atom_tol=1e-6,
                           keep=512, preview_keep=128, adaptive=False,
                           rank1_native=False, depth_override=None),
}


@dataclass(frozen=True)
class Source:
    dataset_short: str
    dataset: str
    display: str
    questions: int
    budget_mib: int
    cohort: str
    path: Path
    methods: tuple[str, ...]
    required_blocks: int
    planned_inputs: dict[str, str]
    planned_sources: dict[str, str]
    protocol_sha256: str


def sources(capacity_root: Path, protocol_path: Path | None = None) -> list[Source]:
    """Resolve all six C outputs from the authoritative frozen protocol."""
    protocol_path = protocol_path or capacity_root/"protocol_v4.json"
    protocol, protocol_hash = read_json(protocol_path)
    require(protocol["version"] == "continuous-domain-supplement-certificate-capacity-v4",
            "Wrong certificate-capacity protocol")
    require(protocol["capacity_policy"]["verified_point_witnesses"] is False and
            protocol["capacity_policy"]["budgets_mib"] == [1, 4, 16],
            "Capacity protocol must disable point witnesses at all three budgets")
    cohorts = {c["id"]: c for c in protocol["cohorts"] if c["group"] == "C"}
    expected = {f"C_certificate_budget{b}_{s}" for b in (1, 4, 16)
                for s, *_ in DATASETS}
    require(len(cohorts) == 6 and set(cohorts) == expected,
            "Capacity protocol must contain exactly the six witness-off C cohorts")
    result = []
    all_methods = tuple(m[0] for m in METHODS)
    for short, dataset, display, count in DATASETS:
        for budget in (1, 4, 16):
            cohort = f"C_certificate_budget{budget}_{short}"
            planned = cohorts[cohort]
            require(planned["dataset"] == dataset and planned["questions"] == count and
                    planned["budget_mib"] == budget and planned["blocks"] == 1 and
                    planned["verified_point_witnesses"] is False and
                    set(planned["specs"]) == set(all_methods) and
                    "--no-witness" in planned["args"],
                    cohort+": wrong frozen population/configuration/witness policy")
            result.append(Source(short, dataset, display, count, budget, cohort,
                                 Path(planned["output"]), all_methods, 1,
                                 planned["input_sha256"], protocol["source_sha256"],
                                 protocol_hash))
    return result


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_complete(source: Source) -> tuple[dict, dict, dict]:
    """Validate the frozen manifest/summary contract without opening traces."""
    job, job_hash = read_json(source.path/"job_manifest.json")
    summary, summary_hash = read_json(source.path/"summary.json")
    prefix = f"{source.cohort}: "
    require(job["protocol"] == summary["protocol"] == "continuous-domain-paired-v3",
            prefix+"wrong complete-workload protocol")
    require(job["arguments"]["study_phase"] == "evaluation",
            prefix+"calibration cannot replace final budget evidence")
    blocks = job["blocks"]
    require(isinstance(blocks, int) and blocks > 0, prefix+"invalid repeats")
    require(summary.get("source_drift") == [], prefix+"source_drift must be []")
    require(summary["complete_blocks"] == summary["expected_blocks"] == blocks,
            prefix+"not all manifest repeats are complete")
    require(summary["job_manifest_sha256"] == job_hash,
            prefix+"summary/manifest hash mismatch")
    require(blocks == source.required_blocks == 1,
            prefix+"expected one complete witness-off paired block")
    require(job["dataset"] == source.dataset, prefix+"wrong dataset")
    require(job["questions"] == source.questions, prefix+"wrong question count")
    require(len(job["original_qids"]) == source.questions,
            prefix+"missing original question identifiers")
    require(len(set(job["original_qids"])) == source.questions,
            prefix+"duplicate original question identifiers")
    require(len(job["original_sources"]) == source.questions,
            prefix+"missing complete original source sets")
    require(job["budget_bytes"] == source.budget_mib*MIB,
            prefix+"wrong numerical-state admission budget")
    require(job["ks"] == [2, 5, 10] and
            job["box"] == [[1., 4.], [.25, 2.]], prefix+"wrong domain/cutoffs")
    require(job["base_tolerance"] == 1e-10 and
            job["maximum_subdivision_depth"] == 2,
            prefix+"wrong base tolerance/default polynomial depth")
    require(job["scan_backend"] == "dot_fallback",
            prefix+"wrong common polynomial scan backend")
    require(job["verified_point_witnesses"] is False and
            job["arguments"]["witness"] is False,
            prefix+"capacity evidence must disable actual-point witness search")
    require(job["execution"] == "paired", prefix+"capacity run must be paired")
    require(job["input_sha256"] == source.planned_inputs and
            job["source_sha256"] == source.planned_sources,
            prefix+"inputs/numerical sources differ from frozen capacity protocol")
    require(job.get("no_gold_access") is True, prefix+"gold-access contract differs")
    specs = {s["name"]: s for s in job["specs"]}
    require(set(source.methods) == specs.keys(), prefix+"capacity method set differs")
    for name in source.methods:
        require(all(specs[name].get(key) == value
                    for key, value in EXPECTED[name].items()),
                prefix+name+" is not the declared fine matched configuration")
    block_ids = [b["block"] for b in summary["blocks"]]
    require(len(block_ids) == blocks and sorted(block_ids) == list(range(blocks)),
            prefix+"missing or duplicate completed block")
    for block in summary["blocks"]:
        for name in source.methods:
            row = block["specs"][name]
            require(row["spec"] == specs[name], prefix+"row/spec mismatch")
            require(row["questions"] == source.questions and
                    row["block"] == block["block"], prefix+"row population mismatch")
            resolved = row["all_cut_queries"].get("resolved", 0)
            unresolved = row["all_cut_queries"].get("unresolved", 0)
            status = row["statuses"]
            require(set(status) <= {"STABLE", "UNSTABLE", "UNKNOWN"},
                    prefix+"unexpected proof status")
            require(all(isinstance(v, int) and v >= 0 for v in status.values()),
                    prefix+"invalid status count")
            require(sum(status.values()) == 3*source.questions,
                    prefix+"not all requested cutoffs counted")
            require(status.get("UNSTABLE", 0) == 0,
                    prefix+"witness-off capacity study cannot contain UNSTABLE")
            require(isinstance(resolved, int) and isinstance(unresolved, int) and
                    min(resolved, unresolved) >= 0 and
                    resolved+unresolved == source.questions,
                    prefix+"all-cut denominator mismatch")
            unknown = status.get("UNKNOWN", 0)
            require(unresolved <= unknown <= 3*unresolved,
                    prefix+"UNKNOWN inconsistent with unresolved questions")
            memory = row["private_memory"]
            require(0 <= memory["peak_private_retained_numeric_bytes"] <=
                    job["budget_bytes"] and
                    0 <= memory["peak_admission_reserved_numeric_bytes"] <=
                    job["budget_bytes"], prefix+"recorded numerical budget exceeded")
    provenance = dict(cohort=source.cohort, path=str(source.path.resolve()),
        budget_mib=source.budget_mib, methods=source.methods, blocks=blocks,
        capacity_protocol_sha256=source.protocol_sha256,
        verified_point_witnesses=False, metric="every_cutoff_STABLE",
        job_manifest_sha256=job_hash, summary_sha256=summary_hash,
        source_drift=[], source_sha256=job["source_sha256"],
        input_sha256=job["input_sha256"], memory_scope=job["memory_scope"])
    return job, summary, provenance


def collect(source_list: list[Source]) -> tuple[list[dict], list[dict], list[dict]]:
    rows, provenance, populations = [], [], {}
    for source in source_list:
        job, summary, proof = read_complete(source)
        identity = {key: job[key] for key in
                    ("original_qids", "original_sources", "seed_source_sha256",
                     "input_sha256")}
        if source.dataset in populations:
            require(identity == populations[source.dataset],
                    source.cohort+": original questions/seeds/sources/input differ")
        else:
            populations[source.dataset] = identity
        provenance.append(proof)
        for block in sorted(summary["blocks"], key=lambda b: b["block"]):
            for name in source.methods:
                row = block["specs"][name]
                resolved = row["all_cut_queries"].get("resolved", 0)
                memory = row["private_memory"]
                rows.append(dict(dataset=source.dataset, display=source.display,
                    method=name, budget_mib=source.budget_mib,
                    cohort=source.cohort, block=block["block"],
                    manifest_repeats=job["blocks"], original_questions=job["questions"],
                    all_cut_stable_questions=resolved,
                    questions_with_any_unknown=job["questions"]-resolved,
                    all_cut_stable_percent=100*resolved/job["questions"],
                    verified_point_witnesses=False,
                    stable_cutoffs=row["statuses"].get("STABLE", 0),
                    unstable_cutoffs=row["statuses"].get("UNSTABLE", 0),
                    unknown_cutoffs=row["statuses"].get("UNKNOWN", 0),
                    peak_retained_numeric_bytes=memory["peak_private_retained_numeric_bytes"],
                    peak_admission_numeric_bytes=memory["peak_admission_reserved_numeric_bytes"],
                    manifest_sha256=proof["job_manifest_sha256"],
                    summary_sha256=proof["summary_sha256"]))
    groups = []
    for _, dataset, display, count in DATASETS:
        for name, label, *_ in METHODS:
            for budget in (1, 4, 16):
                selected = [r for r in rows if r["dataset"] == dataset and
                            r["method"] == name and r["budget_mib"] == budget]
                require(len(selected) == 1, f"Expected one {dataset}/{name}/{budget} MiB block")
                percentage = selected[0]["all_cut_stable_percent"]
                groups.append(dict(dataset=dataset, display=display, method=name,
                    method_label=label, budget_mib=budget, questions=count,
                    repeats=1, center_percent=percentage,
                    cohorts=sorted({r["cohort"] for r in selected})))
    return rows, groups, provenance


def qa_commands(output: Path, skill: Path) -> list[list[str]]:
    py = sys.executable
    return [
        [py, str(skill/"validate_figure.py"), str(Path(__file__).resolve()), "--json"],
        [py, str(skill/"audit_pdf_text.py"), str(output)+".pdf", "--min-pt", "5", "--json"],
        [py, str(skill/"audit_figure_collisions.py"), str(output)+".pdf",
         "--json-out", str(output)+".collision-audit.json",
         "--overlay-pdf", str(output)+".collision-audit.pdf"],
    ]


def contract(args, groups: list[dict], provenance: list[dict]) -> dict:
    return dict(
        status="DATA_VALIDATED_NOT_RENDERED", backend="Python/Matplotlib",
        question="How does retained numerical-state capacity affect whole-domain STABLE certification?",
        conclusion="Measured coverage frontier; no method advantage is assumed before reading complete evidence.",
        archetype="quantitative grid", physical_size_mm=[183, 65], glyph_floor_pt=5,
        panels=dict(a="HotpotQA: 1,000 original questions; larger-graph budget stratum.",
                    b="MuSiQue: all 2,417 original questions; independent smaller-graph stratum."),
        x_axis="Numerical-state admission budget: 1, 4, 16 MiB; log2 positions, lines are visual guides.",
        y_axis="Percent of all original questions for which every k=2,5,10 is STABLE. Point witnesses are disabled; UNSTABLE must be zero and any UNKNOWN excludes a question from the numerator.",
        budget_scope="Includes retained atoms, metadata and admitted family/proof work; excludes separately reported common graph/backend and documented generation scratch. Not operating-system RSS.",
        center="The single complete paired block at each graph/budget point.",
        spread="No error bars: all six cohorts have one block. No confidence interval or p-value.",
        cohort_scope="Only six C_certificate_budget{1,4,16}_{hot,mu} cohorts from protocol_v4 output paths. All budgets disable point witnesses. A/B/D and original main three-state full-operator resolution are separate and are not pooled.",
        methods="All atom tolerances 1e-6, keep512, preview128, base1e-10. Polynomial depth2; lifting depth4. Fine DensePruned is the same certificate in a full-response layout.",
        input_acceptance="One manifest-complete paired block; source_drift exactly []; manifest hash matches summary; frozen protocol input/source hashes match; witnesses disabled and UNSTABLE zero; all original question/seed/source/input identities match per dataset.",
        capacity_protocol=str((args.protocol or args.capacity_root/"protocol_v4.json").resolve()),
        source_data=str(Path(str(args.output)+".source.csv").resolve()),
        aggregated_points=groups, provenance=provenance,
        exclusions="None among the requested method/budget/dataset/repeat observations. No question/cutoff filtering.",
        automatic_qa_commands=qa_commands(args.output, args.skill_scripts),
        alignment="require_matplotlib_panel_alignment after final draw, strict1.5pt, before every export.",
        export_scope="Editable vector PDF/SVG are authoritative; PNG is a600dpi QA preview. TIFF is unnecessary for this vector line figure.",
        visual_review="Pending until final PDF is inspected at its physical size.",
        proposed_caption=("Certificate capacity versus retained numerical-state admission budget. "
            "All original HotpotQA (n=1,000) and MuSiQue (n=2,417) questions and all three cutoffs are included. "
            "The numerator contains only questions whose every cutoff is STABLE; any UNKNOWN remains in the denominator. "
            "All three budgets use newly measured witness-off paired runs, one block per point, with no error bars. "
            "Curves retain the observed values without monotonic adjustment. "
            "Budget includes family work and is not process RSS. This capacity study is separate from three-state audit resolution."))


def render(groups: list[dict], output: Path, skill: Path) -> dict:
    # Importing the plotting backend is reserved for the explicit final step.
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    sys.path.insert(0, str(skill))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "text.usetex": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 65/25.4),
                             sharex=True, sharey=True)
    fig.subplots_adjust(left=.080, right=.985, bottom=.28, top=.78, wspace=.20)
    try:
        legend_handles = {}
        for index, (ax, (_, dataset, display, count)) in enumerate(zip(axes, DATASETS)):
            # Sparse is drawn last; marker shapes and line patterns also distinguish overlap.
            for name, label, color, marker, line_style in reversed(METHODS):
                points = sorted((g for g in groups if g["dataset"] == dataset and
                                 g["method"] == name), key=lambda g: g["budget_mib"])
                x = [p["budget_mib"] for p in points]
                y = [p["center_percent"] for p in points]
                require(all(value > 0 for value in x),
                        "The logarithmic budget axis requires strictly positive MiB values.")
                (handle,) = ax.plot(x, y, color=color, linestyle=line_style,
                    marker=marker, markersize=3.8, markeredgewidth=.85,
                    markerfacecolor="white" if name != "fine_preview" else color,
                    linewidth=1.1, label=label, zorder=3 if name == "fine_preview" else 2)
                legend_handles[name] = handle
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 4, 16], ["1", "4", "16"])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set(xlim=(.8, 20), ylim=(-3, 103), yticks=[0, 25, 50, 75, 100])
            ax.tick_params(width=.7, length=2.5, pad=3)
            ax.set_title(f"{display}: {count:,} questions", pad=9)
            ax.annotate("ab"[index], xy=(0, 1), xycoords="axes fraction",
                        xytext=(-20, 9), textcoords="offset points",
                        weight="bold", fontsize=8.5, annotation_clip=False)
            ax.grid(axis="y", linewidth=.45, color="#dddddd", zorder=0)
        axes[0].set_ylabel("All cutoffs STABLE (%)", labelpad=6)
        fig.supxlabel("Retained numerical-state budget (MiB)", fontsize=7.5, y=.115)
        fig.legend([legend_handles[m[0]] for m in METHODS], [m[1] for m in METHODS],
                   loc="upper center", bbox_to_anchor=(.54, .995), ncol=3,
                   frameon=False, handlelength=2.2, columnspacing=2.0)
        fig.text(.54, .026, "Budget includes family work; process RSS is reported separately.",
                 ha="center", va="bottom", fontsize=6.5)
        fig.canvas.draw()
        require_matplotlib_panel_alignment(
            fig, json_out=str(output)+".alignment.json",
            overlay_svg=str(output)+".alignment.svg", tolerance_pt=1.5,
            gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
        # Keep the prescribed physical size; no tight-bbox rescaling.
        fig.savefig(str(output)+".pdf")
        fig.savefig(str(output)+".svg")
        fig.savefig(str(output)+".png", dpi=600)
    finally:
        plt.close(fig)
    qa = []
    for number, command in enumerate(qa_commands(output, skill)):
        result = subprocess.run(command, text=True, capture_output=True)
        log_path = Path(str(output)+f".qa{number+1}.log")
        log_path.write_text(result.stdout+result.stderr, encoding="utf8")
        qa.append(dict(command=command, returncode=result.returncode,
                       log=str(log_path.resolve())))
        if number == 1:
            Path(str(output)+".font-audit.json").write_text(
                result.stdout, encoding="utf8")
        require(result.returncode == 0,
                "Rendered QA requires repair/review: "+str(log_path))
    return dict(automatic_checks=qa,
                visual_review="Final physical-size panel inspection still required.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-root", type=Path, default=DEFAULT_CAPACITY,
                        help="Directory containing the frozen certificate-capacity protocol.")
    parser.add_argument("--protocol", type=Path,
                        help="Optional protocol path; cohort.output remains authoritative.")
    parser.add_argument("--output", type=Path, default=ROOT/"figures/domain_budget")
    parser.add_argument("--skill-scripts", type=Path, default=DEFAULT_SKILL)
    parser.add_argument("--render", action="store_true",
                        help="After complete data validation, render and run final PDF QA.")
    parser.add_argument("--print-plan", action="store_true",
                        help="Print the input plan only; do not read runs or create output.")
    parser.add_argument("--print-qa-commands", action="store_true",
                        help="Print source/font/collision QA commands without reading runs.")
    args = parser.parse_args(argv)
    if args.print_qa_commands:
        for command in qa_commands(args.output, args.skill_scripts):
            print(subprocess.list2cmdline(command))
        return 0
    source_list = sources(args.capacity_root, args.protocol)
    if args.print_plan:
        print(json.dumps([dict(cohort=s.cohort, path=str(s.path),
            dataset=s.dataset, questions=s.questions, budget_mib=s.budget_mib,
            methods=s.methods, required_blocks=s.required_blocks)
            for s in source_list], indent=2))
        return 0
    # No output before every requested cohort is accepted; unfinished runs fail.
    rows, groups, provenance = collect(source_list)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(args.output)+".source.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    evidence = contract(args, groups, provenance)
    Path(str(args.output)+".caption.txt").write_text(
        evidence["proposed_caption"]+"\n", encoding="utf8")
    contract_path = Path(str(args.output)+".contract.json")
    contract_path.write_text(json.dumps(evidence, indent=2)+"\n", encoding="utf8")
    if args.render:
        evidence["status"] = "RENDER_REQUESTED_QA_PENDING"
        contract_path.write_text(json.dumps(evidence, indent=2)+"\n", encoding="utf8")
        try:
            evidence["rendered_qa"] = render(groups, args.output, args.skill_scripts)
        except Exception as error:
            evidence["status"] = "RENDER_QA_REQUIRES_REVIEW"
            evidence["render_error"] = str(error)
            contract_path.write_text(json.dumps(evidence, indent=2)+"\n", encoding="utf8")
            raise
        evidence["status"] = "RENDERED_AUTOMATIC_QA_COMPLETE_VISUAL_REVIEW_PENDING"
        contract_path.write_text(json.dumps(evidence, indent=2)+"\n", encoding="utf8")
        print("COMPLETE_BUDGET_FIGURE_RENDERED", args.output)
    else:
        print("COMPLETE_BUDGET_DATA_EXPORTED_NOT_RENDERED", args.output)
        print("Use --render after reviewing the source CSV and contract.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit("Budget figure input/QA gate failed: "+str(error))
