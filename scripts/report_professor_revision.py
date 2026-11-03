"""Join only complete declared revision cohorts; produce tables and column figure."""
from pathlib import Path
from hashlib import sha256
import json,sys,collections
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/professor_revision_v1';sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
if '--tables-only' not in sys.argv:
    from audit_panel_alignment import require_matplotlib_panel_alignment
names=[f'{d}_a{a}' for d in ['hotpot_pool7405','musique_pool2417'] for a in ['0p5','0p85','0p95']]+['hotpot_pool1000_a0p5','musique_pool1000_a0p5','amazon_r4_original','amazon_r4_self05']
data={};provenance=[]
for name in names:
    folder=OUT/'boundaries'/name;p=folder/'summary.json';r=json.loads(p.read_text());job=json.loads((folder/'job_manifest.json').read_text())
    assert r['complete_blocks']==r['expected_blocks']==1 and not r['source_drift']
    assert sha256((folder/'job_manifest.json').read_bytes()).hexdigest()==r['job_manifest_sha256']
    data[name]=r['blocks'][0];provenance.append(dict(name=name,summary_sha256=sha256(p.read_bytes()).hexdigest(),questions=job['questions']))
smt_folder=OUT/'shared_smt_dyadic_prefix20_5s';assert (smt_folder/'complete.json').exists()
smt=[json.loads(s) for s in (smt_folder/'rows.jsonl').read_text().splitlines()];assert len(smt)==40
sr={}
for ds in ['hot','mu']:
    rows=[r for r in smt if r['dataset']==ds]
    sr[ds]=dict(questions=len(rows),cuts=60,bernstein_stable=sum(v=='STABLE' for r in rows for v in r['bernstein_cuts'].values()),bernstein_seconds=sum(r['bernstein_seconds'] for r in rows),smt_stable=sum(v['status']=='STABLE' for r in rows for v in r['smt'].values()),smt_seconds=sum(v['total_seconds'] for r in rows for v in r['smt'].values()),solver_outcomes=dict(collections.Counter(v['solver_result'] for r in rows for v in r['smt'].values())))
report=dict(status='COMPLETE_DECLARED_EXTENSION',cohorts=data,smt=sr,provenance=provenance,excluded='amazon_scale96_a0p85 is an interrupted engineering run containing unsupported r=16 queries; no metrics admitted. Amazon r4 probes have their own frozen structural selection protocol.')
if '--output' not in sys.argv:
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
if '--tables-only' not in sys.argv:
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':7.2,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'legend.frameon':False})
    fig,axes=plt.subplots(1,2,figsize=(89/25.4,58/25.4));fig.subplots_adjust(left=.14,right=.98,bottom=.28,top=.78,wspace=.30)
    colors=['#73818f','#426a9a','#176c74','#b76832'];labels=['Setup','Base','Responses','Audit']
    for ax,ds,title in zip(axes,['hotpot_pool7405','musique_pool2417'],['(a) HotpotQA','(b) MuSiQue']):
        specs=[data[f'{ds}_a{a}']['specs']['fine_preview'] for a in ['0p5','0p85','0p95']];bottom=np.zeros(3)
        totals=np.array([r['total_seconds'] for r in specs]);base=np.array([r['stages']['seed_seconds']+r['stages']['base_seconds'] for r in specs]);responses=np.array([r['stages']['prepare_seconds'] for r in specs]);audit=np.array([r['stages']['certify_seconds']+r['stages']['release_seconds'] for r in specs]);setup=totals-base-responses-audit
        for vals,label,color in zip([setup,base,responses,audit],labels,colors):ax.bar(np.arange(3),vals,bottom=bottom,width=.56,color=color,label=label);bottom+=vals
        for x,r in enumerate(specs):ax.text(x,totals[x]+totals.max()*.025,f'{totals[x]:.1f}',ha='center',fontsize=6.5)
        ax.set(xticks=[0,1,2],xticklabels=['.50','.85','.95'],xlabel=r'Propagation $\alpha$',ylim=(0,totals.max()*1.18))
        ax.text(0,1.12,title,transform=ax.transAxes,fontsize=7.4,fontweight='bold')
    axes[0].set_ylabel('Complete cost (s)')
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.54,.005),ncol=4,fontsize=6.5,handlelength=1,handletextpad=.3,columnspacing=.65)
    stem=ROOT/'figures/domain_boundary_v3';fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',strict=True)
    fig.savefig(str(stem)+'.pdf');fig.savefig(str(stem)+'.svg');fig.savefig(str(stem)+'.png',dpi=600);plt.close(fig)

def cell(spec):return f"{spec['total_seconds']:.1f}/{spec['all_cut_queries'].get('resolved',0)}/{spec['statuses'].get('UNKNOWN',0)}"
def table_rows(selected):
    return '\n'.join(label+' & '+' & '.join(cell(data[name]['specs'][method]) for method in ['fine_preview','dp_preview_fine','lift_sparse_fine_preview'])+r'\\' for label,name in selected)
lines=[r'\subsection{Shared-parameter verifier and operating boundaries}',
r'The new studies address verifier choice and operating conditions separately from the complete three-block primary workloads. All prefixes, source arrays and methods are fixed before each study; every requested cutoff remains in its denominator.',
r'\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}',
r'\caption{Shared-core verifier comparison, first 20 questions per graph, all 60 cutoffs. Entries are seconds / STABLE cuts. Both consume identical retained rows, tails and upward-rounded residual/resolvent bounds. SMT has a 5\,s solver limit per cutoff; all encoding time is included. One block; graph preparation and the common compiler are excluded.}',r'\label{tab:shared-smt}',r'\begin{tabular}{lrr}\toprule Graph & Bernstein & Shared SMT\\\midrule']
for ds,title in [('hot','HotpotQA'),('mu','MuSiQue')]:
    r=sr[ds];lines.append(f"{title} & {r['bernstein_seconds']:.3f}/{r['bernstein_stable']} & {r['smt_seconds']:.2f}/{r['smt_stable']}"+r'\\')
lines += [r'\bottomrule\end{tabular}\end{table}',
r'The solution-function comparator follows the shared-parameter SMT route of Junges et al.~\cite{junges2024}, implemented here with Z3 QF\_NRA rather than the PROPhESY tool. It asks whether any in-domain policy has nonpositive approximate determinant or violates a guarded selected--competitor gap, including all outside nodes through the same tails. UNSAT certifies stability; SAT only refutes this sufficient test and is not an actual instability witness. The three MuSiQue SAT cases remain UNKNOWN; the other unproved SMT cuts time out. Table~\ref{tab:shared-smt} isolates verification on the same source core; it is not an end-to-end or solver-workspace-budget-matched comparison. The SMT baseline passes an independent exact three-node positive/negative membership check.',
r'\input{generated/domain_smt_encoding}',
r'\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2pt}',r'\caption{Propagation boundary on fixed first-100 prefixes, one complete paired block at each setting. Cells: complete seconds / all-cut resolved questions / UNKNOWN cutoffs. VR: Sparse; DP: fine DensePruned; Lift: compressed SourceLift. Every original prefix request remains charged.}',r'\label{tab:alpha-boundary}',r'\begin{tabular}{lrrr}\toprule Graph, $\alpha$ & VR & DP & Lift\\\midrule',
table_rows([(f'{short}, {a:.2f}',f'{ds}_a{str(a).replace(".","p")}') for ds,short in [('hotpot_pool7405','Hot'),('musique_pool2417','Mu')] for a in [.5,.85,.95]]),r'\bottomrule\end{tabular}\end{table}',
r'At $\alpha=.95$, both primary prefixes resolve all 100 requests. HotpotQA is near parity with compressed lifting (3.1\% lower cost in this block); MuSiQue retains 19.8\% lower cost. Figure~\ref{fig:boundary-stages} separates the increased graph-response work from audit work. These are controlled operating points, not new independent questions or three-block confidence intervals.',
r'\begin{figure}[t]\centering\includegraphics[width=\columnwidth]{../figures/domain_boundary_v3.pdf}',r'\caption{Primary complete-cost stages for Table~\ref{tab:alpha-boundary}. Base includes seed construction; Audit includes certification, actual witnesses and release; Setup includes charged startup and method setup. One complete block per setting.}',r'\label{fig:boundary-stages}\end{figure}',
r'\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2pt}',r'\caption{Additional graph contexts and controlled Amazon probes; same cell format as Table~\ref{tab:alpha-boundary}. Small lexical pools use 100 queries and $\alpha=.5$. Amazon has 334,863 nodes, $\alpha=.85$, and all eight pre-existing $r=4$ seed/source requests. $+$self adds raw self weight .5 before normalization. One complete paired block; 16\,MiB admission budget.}',r'\label{tab:graph-boundary}',r'\begin{tabular}{lrrr}\toprule Graph (nodes) & VR & DP & Lift\\\midrule',
table_rows([('Hot (9,769)','hotpot_pool1000_a0p5'),('Mu (12,493)','musique_pool1000_a0p5'),('Amazon','amazon_r4_original'),('Amazon +self','amazon_r4_self05')]),r'\bottomrule\end{tabular}\end{table}',
r'The smaller natural context pools retain their declared graph-dependent seeds/sources, so the comparison with the full pools changes graph context as well as size. Amazon uses a fixed hash-based neighbor grouping, not semantic mention links. Its zero-self original is a ratio-only control; the added-self variant exercises two policy directions. Both preserve the complete selected source arrays. The eight original $r=16$ requests are outside the current implementation\textquotesingle s $r\le4$ admission contract and receive no performance claim. Sparse resolves 8/8 in both probes, versus 3/8 for DP, and reduces complete cost against compressed lifting by 37.7\%/40.0\%. These limited probes extend execution evidence beyond the lexical topology without establishing application accuracy.']
# Keep observed numerical-state/process peaks and stage details available, with their distinct scopes.
private=[b['specs']['fine_preview']['private_memory']['peak_admission_reserved_numeric_bytes']/1024**2 for b in data.values()]
process=[b['process_memory']['process_peak_working_set_bytes']/1024**2 for b in data.values()]
lines.append(f"Across these ten complete cohorts, the primary's admitted numerical-state peaks span {min(private):.2f}--{max(private):.2f} MiB. Fresh cohort process peaks span {min(process):.1f}--{max(process):.1f} MiB and include all three paired method directories plus graph/backend state; they are not individual-method RSS values. Exact stage and memory records accompany the tables.")
destination = Path(sys.argv[sys.argv.index('--output')+1]) if '--output' in sys.argv else ROOT/'paper/generated/domain_professor_revision.tex'
destination.write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(complete_cohorts=len(data),smt=sr,primary_numeric_peak_range_mib=[min(private),max(private)],paired_process_peak_range_mib=[min(process),max(process)])))
