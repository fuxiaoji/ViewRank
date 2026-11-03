"""Final complete-workload figure; all bars use within-block paired ratios.

Contract: compare the whole-domain operator with strong regional methods on
both complete workloads. Two equal panels stratify by dataset. New certificate
layouts form one color family, regional controls a neutral family. Medians and
min/max of all three blocks are descriptive, not confidence intervals. Coverage
labels use every original question. No incomplete cohort or selected seed is
accepted. The same-cache lifting comparison has its own paired primary, so
absolute times are never pooled across differently scheduled cohorts.
"""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,r'C:/Users/fwj/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment

METHODS=(
    ('fine_preview','VR-Sparse','#176c74','main'),
    ('dp_preview_a1e4','VR-DP\ncoarse','#91babc','main'),
    ('dp_preview_fine','VR-DP\nfine','#bfd3d2','main'),
    ('lift_sparse_fine_preview','Lift\nSparse','#adb4bd','matched'),
    ('lift_dp_preview_fine','Lift\nDP','#818b96','main'),
    ('native_then_lift_preview_fine','Native\nhybrid','#5c6775','main'),
)
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})


def read_run(path):
    job_raw=(path/'job_manifest.json').read_bytes()
    summary_raw=(path/'summary.json').read_bytes()
    job=json.loads(job_raw)
    data=json.loads(summary_raw)
    if (data['source_drift'] != [] or
            data['complete_blocks'] != data['expected_blocks'] or
            data['complete_blocks'] != job['blocks'] or job['blocks'] != 3):
        raise ValueError('Figure requires all three complete source-frozen blocks')
    if data['job_manifest_sha256'] != sha256(job_raw).hexdigest():
        raise ValueError('Summary does not match its frozen manifest')
    data['figure_input_provenance']=dict(cohort=path.name,path=str(path.resolve()),
        job_manifest_sha256=sha256(job_raw).hexdigest(),
        summary_sha256=sha256(summary_raw).hexdigest(),source_drift=[])
    return job,data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hot-main',type=Path,required=True)
    p.add_argument('--mu-main',type=Path,required=True)
    p.add_argument('--hot-matched',type=Path,required=True)
    p.add_argument('--mu-matched',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'figures/domain_performance')
    args=p.parse_args()
    rows=[]
    provenance=[]
    datasets=[]
    for label,main_path,matched_path in [('HotpotQA',args.hot_main,args.hot_matched),
            ('MuSiQue',args.mu_main,args.mu_matched)]:
        runs={'main':read_run(main_path),'matched':read_run(matched_path)}
        if runs['main'][0]['original_qids'] != runs['matched'][0]['original_qids']:
            raise ValueError('Matched cohort has different original questions')
        for field in ('original_sources','seed_source_sha256','input_sha256'):
            if runs['main'][0][field] != runs['matched'][0][field]:
                raise ValueError('Matched cohort differs in '+field)
        provenance.extend(data['figure_input_provenance'] for _,data in runs.values())
        datasets.append((label,runs['main'][0]['questions']))
        for name,display,color,group in METHODS:
            job,data=runs[group]
            for block in data['blocks']:
                current=block['specs'][name]
                reference=block['specs']['fine_preview']
                rows.append(dict(dataset=label,questions=job['questions'],method=name,
                    display=display,group=group,block=block['block'],
                    source_cohort=data['figure_input_provenance']['cohort'],
                    total_seconds=current['total_seconds'],
                    paired_primary_seconds=reference['total_seconds'],
                    paired_cost_ratio=current['total_seconds']/reference['total_seconds'],
                    resolved_questions=current['all_cut_queries'].get('resolved',0),
                    resolved_percent=100*current['all_cut_queries'].get('resolved',0)/job['questions']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with Path(str(args.output)+'.source.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axs=plt.subplots(1,2,figsize=(183/25.4,67/25.4),sharey=True)
    fig.subplots_adjust(left=.075,right=.985,bottom=.28,top=.84,wspace=.16)
    upper=max(r['paired_cost_ratio'] for r in rows)
    for index,(ax,(label,n)) in enumerate(zip(axs,datasets)):
        for position,(name,display,color,_) in enumerate(METHODS):
            selected=[r for r in rows if r['dataset']==label and r['method']==name]
            ratios=np.array([r['paired_cost_ratio'] for r in selected])
            center=float(np.median(ratios))
            ax.bar(position,center,width=.70,color=color,zorder=2)
            ax.errorbar(position,center,yerr=[[center-ratios.min()],[ratios.max()-center]],
                color='#333333',capsize=2,lw=.7,zorder=3)
            ax.plot(position+np.array([-.08,0,.08]),ratios,'o',markersize=1.6,color='#333333',zorder=4)
            coverage=float(np.median([r['resolved_percent'] for r in selected]))
            ax.text(position,float(ratios.max())+.045*upper,f'{coverage:.2f}%',
                ha='center',va='bottom',fontsize=6.2)
        ax.axhline(1,color='#333333',ls='--',lw=.65,zorder=1)
        ax.set_xticks(range(len(METHODS)),[m[1] for m in METHODS],fontsize=6.5)
        ax.tick_params(axis='x',length=0,pad=5)
        ax.set(xlim=(-.6,len(METHODS)-.4),ylim=(0,upper*1.19))
        ax.set_title(f'{label}: {n:,} questions',fontsize=8,pad=12)
        ax.text(-.10,1.10,'ab'[index],transform=ax.transAxes,fontweight='bold',fontsize=8.5)
        # The paired-reference line is sufficient; no grid crosses value labels.
        ax.grid(False)
    axs[0].set_ylabel('Total cost / paired VR-Sparse cost')
    fig.text(.53,.025,'Labels: questions with all three cutoffs resolved',ha='center',fontsize=7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(args.output)+'.alignment.json',
        overlay_svg=str(args.output)+'.alignment.svg',tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    fig.savefig(str(args.output)+'.pdf')
    fig.savefig(str(args.output)+'.svg')
    fig.savefig(str(args.output)+'.png',dpi=600)
    Path(str(args.output)+'.contract.json').write_text(json.dumps(dict(
        conclusion='Compare complete-cost efficiency of shared-domain certificates and strong regional controls, with coverage visible.',
        panels='Equal dataset strata, all methods and three complete blocks retained.',
        uncertainty='Median and full min/max across three scheduled blocks; no confidence interval or p-value.',
        source='CSV contains every plotted block and its own paired denominator; no cross-cohort absolute-time pooling.',
        scope='VR-DP means VR-DensePruned: the same new polynomial certificate in a pruned full-response physical layout. Lift DP uses the corresponding full-response pruned layout.',
        source_provenance=provenance,
        raster_scope='PNG is the 600dpi QA preview; editable vector PDF/SVG are authoritative and no TIFF is required.',
        glyph_floor_pt=5,
        physical_size_mm=[183,67]),indent=2),encoding='utf-8')
    print('COMPLETE_PAIRED_FIGURE_BUILT',str(args.output))


if __name__=='__main__':
    main()
