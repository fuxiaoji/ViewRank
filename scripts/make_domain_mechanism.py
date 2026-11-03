"""Analytic constructed example: shared weights exclude a relaxed counterpolicy.

Figure contract: show why retaining the common two weights can avoid sourcewise
subdivision. One quantitative panel, five-node hand construction, no empirical
observations or uncertainty estimates. The exact formulas and full grid are
exported with the figure. Final width is 89 mm, vector text and 600 dpi preview.
"""
from pathlib import Path
import json
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SKILL=Path(r'C:/Users/fwj/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures/domain_mechanism'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})


def main():
    lo,hi=1/3,16/17
    t=np.linspace(lo,hi,401)
    ta,tb=np.meshgrid(t,t)
    u=7/16-7*ta/16+7*ta*tb/32
    v=7*ta*(1-tb)/32
    # Full comparison, not a two-node shortcut: A,B,U,V,C.
    scores=np.stack([np.full_like(ta,7/16),7*ta/32,u,v,np.full_like(ta,1/8)])
    excluded=np.argmin(scores,axis=0)
    assert set(np.unique(excluded)) <= {2,3}
    assert np.all(np.argmin(scores[:,np.arange(len(t)),np.arange(len(t))],axis=0)==3)
    fig,ax=plt.subplots(figsize=(89/25.4,77/25.4))
    fig.subplots_adjust(left=.19,bottom=.20,right=.98,top=.98)
    ax.contourf(ta,tb,(excluded==2).astype(int),levels=[-.5,.5,1.5],colors=['#e7edf0','#e3b78a'])
    ax.contour(ta,tb,u-v,levels=[0],colors=['#9f673c'],linewidths=.65)
    ax.plot(t,t,color='#176c74',lw=2.1,zorder=4)
    ax.plot(hi,lo,marker='o',markersize=4.8,color='#9f673c',markeredgecolor='white',markeredgewidth=.7,zorder=5)
    ax.text(.42,.79,'Top-4: A, B, U, C',ha='left',va='center')
    ax.annotate('Relaxed top-4:\nA, B, V, C',xy=(.925,.35),xytext=(.70,.49),
        arrowprops={'arrowstyle':'-','color':'#9f673c','lw':.65},
        ha='left',va='center',fontsize=6.8)
    ax.annotate('Shared weights',xy=(.63,.63),xytext=(.37,.56),
        arrowprops={'arrowstyle':'-','color':'#176c74','lw':.65},color='#176c74',fontsize=7)
    ax.annotate('Relaxed counterpolicy',xy=(hi,lo),xytext=(.38,.29),
        arrowprops={'arrowstyle':'-','color':'#9f673c','lw':.65},color='#85522e',fontsize=6.5)
    ax.set(xlim=(.30,.97),ylim=(.25,.97),xlabel=r'Source A: mention fraction $t_A$',ylabel=r'Source B: mention fraction $t_B$')
    ax.set_xticks([1/3,.5,.75,16/17],['1/3','0.5','0.75','16/17'])
    ax.set_yticks([1/3,.5,.75,16/17],['1/3','0.5','0.75','16/17'])
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(OUT)+'.alignment.json',strict=True)
    fig.savefig(str(OUT)+'.pdf')
    fig.savefig(str(OUT)+'.svg')
    fig.savefig(str(OUT)+'.png',dpi=600)
    np.savez_compressed(str(OUT)+'.source.npz',t_A=ta,t_B=tb,scores=scores,excluded=excluded)
    Path(str(OUT)+'.contract.json').write_text(json.dumps({
        'claim':'Common weights lie on the diagonal and retain top-4; independently relaxed source weights include a different membership.',
        'type':'exact constructed illustration, not natural-query performance',
        'grid_points':int(ta.size),'selection':'Full declared local rectangle, 401x401 uniform points; no observed queries selected',
        'proof':'For common t, V<=7/128, B>=7/96, U>=7/32, C=1/8; all four selected scores strictly exceed V.',
        'independent_relaxed_point':{'t_A':'16/17','t_B':'1/3','V':'7/51','U':'77/816'},
        'statistics':'Analytic functions; no stochastic replicates, uncertainty bars or statistical tests.',
        'alignment':'Single panel: not applicable',
    },indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
