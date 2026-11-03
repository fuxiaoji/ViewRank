"""Final IEEE column figures: draw at 89 mm, rather than shrink wide panels.

The user's preferred reference style uses compact figures embedded in the two
text columns. Each figure is composed natively at column width, with short labels,
live vector text and the same frozen evidence used by the preceding revision.
"""
from pathlib import Path
from hashlib import sha256
import csv
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

import make_domain_visual_revision as v
ROOT,OUT=v.ROOT,v.OUT
TEAL,PALE,INK,GRAY,LIGHT,ORANGE,SAND,BLUE,PURPLE=(v.TEAL,v.PALE,v.INK,v.GRAY,v.LIGHT,v.ORANGE,v.SAND,v.BLUE,v.PURPLE)
text,rect,arrow,route,canvas=v.text,v.rect,v.arrow,v.route,v.canvas
plt.rcParams.update({'font.size':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
                     'axes.labelsize':7.3,'axes.titlesize':7.5,'axes.linewidth':.6,
                     'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'pdf.fonttype':42,'svg.fonttype':'none'})


def save(fig,name,claim,source,kind='measurement',**kw):
    stem=OUT/name
    ids=[chr(97+i) for i in range(len(fig.axes))]
    layout={'panel_ids':ids,'row_groups':[ids]} if len(ids)>1 else {}
    fig.canvas.draw()
    v.require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',
        overlay_svg=str(stem)+'.alignment.svg',strict=True,**layout)
    fig.savefig(str(stem)+'.pdf')
    fig.savefig(str(stem)+'.svg')
    fig.savefig(str(stem)+'.png',dpi=600)
    record=dict(version='column-visual-revision-v2',claim=claim,type=kind,source=source,
        backend='Python/Matplotlib',final_width_mm=89,
        selection='All original questions and all recorded blocks in each declared cohort.',
        statistics='Three-block points are descriptive, not confidence intervals. Single-block panels have no error bars.',
        input_report_sha256=sha256(v.REPORT_PATH.read_bytes()).hexdigest(),
        producer_script='scripts/make_domain_column_figures.py',
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_graphics_helper_sha256=sha256(Path(v.__file__).read_bytes()).hexdigest())
    record.update(kw)
    Path(str(stem)+'.contract.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    plt.close(fig)


def example():
    fig,ax=canvas(89,56)
    text(ax,2,6,'Which fanzine interviewed the Sex Pistols bassist\nwhile he was in The Flowers of Romance?',7.2)
    x0,y0,w,h=9,21,24,24
    rect(ax,x0,y0,w,h,PALE,TEAL,lw=.65)
    for m,weak,tag,dy in [(1,1,'P1',3),(2.5,1.125,'P2',-3)]:
        x=x0+(m-1)/3*w;y=y0+h-(weak-.25)/1.75*h
        ax.plot(x,y,'o',ms=4,color=TEAL,mec='white',mew=.5)
        text(ax,x+1.8,y+dy,tag,6.8,color=TEAL,weight='bold')
    text(ax,21,16,'Policy domain',7.2,ha='center')
    text(ax,21,51,r'Mention $m$',7.2,ha='center')
    text(ax,2.8,33,r'Weak $w$',7.2,ha='center',rotation=90,rotation_mode='anchor')
    for x,s in [(9,'1'),(33,'4')]:text(ax,x,47,s,6.5,ha='center')
    for y,s in [(21,'2'),(45,'0.25')]:text(ax,7.2,y,s,6.5,ha='right')
    for y,title,name,col,fc in [(17,'P1: (1, 1)','Ex Pistols',GRAY,LIGHT),
                               (36,'P2: (2.5, 1.125)','Sid Vicious*',TEAL,PALE)]:
        rect(ax,42,y,45,15,fc,col,lw=.65)
        text(ax,44,y+3,title,7,color=col,weight='bold')
        text(ax,44,y+7.8,'1  Flowers of Romance*',6.9)
        text(ax,44,y+12,'2  '+name,7,color=col,weight='bold')
    save(fig,'domain_example','Two actual common-weight policies change a real top-2 membership.',
         'figures/domain_example.source.json; qi264/k2',kind='selected actual-view witness',
         selection='Same explicitly posthoc positive illustration as the frozen manuscript; not an accuracy experiment.')


def overview():
    fig,ax=canvas(89,83)
    rect(ax,2,2,85,10,LIGHT,GRAY,lw=.65)
    text(ax,44.5,7,r'Fixed $P_0$; query $h,S,\Theta,K$',7.5,ha='center')
    rect(ax,2,20,30,30,PALE,TEAL,lw=.7)
    text(ax,17,24,'Response directory',7.1,ha='center',weight='bold',color=TEAL)
    text(ax,17,31,r'$\hat R_j,\ \hat H$',7.5,ha='center')
    text(ax,17,37.5,r'Rows $I_j$; tails $\tau_j$',7.5,ha='center')
    text(ax,17,43.5,r'Residuals $\rho_j$',7.5,ha='center')
    text(ax,17,48,'LRU; active pins',6.5,ha='center')
    rect(ax,42,20,45,13,'white',TEAL,lw=.7)
    text(ax,64.5,23.8,'1  Affine r×r core (III)',7.3,ha='center',weight='bold',color=TEAL)
    text(ax,64.5,29.2,r'$\hat y_S,\hat H\ \longrightarrow\ M,\hat D,\hat F$',7.5,ha='center')
    rect(ax,42,44,45,17,'white',TEAL,lw=.7)
    text(ax,64.5,48,'2  Whole-output test (IV)',7.3,ha='center',weight='bold',color=TEAL)
    text(ax,64.5,53.5,'Candidates + outside bounds',6.8,ha='center')
    text(ax,64.5,58,'Bernstein cells over the domain',6.8,ha='center')
    rect(ax,2,65,30,13,LIGHT,GRAY,lw=.65)
    text(ax,17,68.6,'Numerical solver',7.1,ha='center',weight='bold')
    text(ax,17,74.2,'Base, responses, witnesses',6.5,ha='center')
    rect(ax,42,72,45,7,PALE,TEAL,lw=.7)
    text(ax,64.5,75.5,'STABLE / UNSTABLE / UNKNOWN',6.6,ha='center')
    arrow(ax,(64.5,12),(64.5,20))
    text(ax,66,16,r'base $\hat y$',7.5)
    arrow(ax,(32,28),(42,28),TEAL);text(ax,37,24,r'$\hat H$',7.5,ha='center')
    route(ax,[(32,45),(37,45),(37,52),(42,52)],TEAL)
    arrow(ax,(64.5,33),(64.5,44),TEAL);text(ax,66,38.5,r'$\hat D,\hat F$',7.5)
    arrow(ax,(64.5,61),(64.5,72),TEAL)
    text(ax,66,66.4,'per cutoff',6.5)
    arrow(ax,(12,50),(12,65),GRAY,ls='--');text(ax,10.5,57.5,'miss',6.5,ha='right')
    arrow(ax,(23,65),(23,50),TEAL);text(ax,24.5,57.5,'fill',6.5)
    route(ax,[(42,58),(35,58),(35,68),(32,68)],GRAY,ls='--')
    text(ax,38,65,'actual-policy\nwitnesses',6.5)
    arrow(ax,(32,75.5),(42,75.5),GRAY)
    save(fig,'domain_overview','The source compiler and whole-output test reuse budgeted numerical responses.',
         ['paper/main.tex Sections II-V','src/retained_policy_family.py','src/policy_domain_execution.py'],
         kind='compact execution schematic',selection='Grounded modules; each cutoff has its own three-valued result.',
         budget='Directory and admitted family-private numerical work share the same byte budget. Graph/backend are separate.')


def mechanism():
    fig=plt.figure(figsize=(89/25.4,57/25.4))
    a=fig.add_axes([.025,.23,.395,.59]);b=fig.add_axes([.58,.23,.395,.59])
    for ax,title in [(a,'(a) Shared weights'),(b,'(b) Relaxed domain')]:
        ax.text(0,1.18,title,transform=ax.transAxes,fontsize=7.5,fontweight='bold',va='bottom')
    a.set(xlim=(0,10),ylim=(8,0));a.axis('off')
    pts={'A':(2,2),'B':(8,2),'U':(2,6),'V':(8,6),'C':(5,6)}
    for src,dst,g in [('A','B','m'),('A','U','w'),('B','U','m'),('B','V','w')]:
        p,q=np.array(pts[src]),np.array(pts[dst]);d=(q-p)/np.linalg.norm(q-p)
        arrow(a,p+d*.56,q-d*.56,TEAL if g=='m' else ORANGE,lw=.8)
        xy=(4.2,3.05) if src=='B' and dst=='U' else tuple((p+q)/2+np.array([.35,-.35]))
        text(a,*xy,g,7,color=TEAL if g=='m' else ORANGE)
    for node,(x,y) in pts.items():
        a.add_patch(Circle((x,y),.55,facecolor=PALE if node in('A','B') else 'white',edgecolor=TEAL if node in('A','B') else GRAY,lw=.7))
        text(a,x,y,node,7.2,ha='center')
    text(a,2,.35,r'$h_A=7/8$',7.5,ha='center');text(a,5.7,5.0,r'$h_C=1/8$',7.5,ha='center')
    for node in ('U','V','C'):
        x,y=pts[node];arrow(a,(x-.32,y+.47),(x+.32,y+.47),GRAY,rad=.95,lw=.65)
    lo,hi=1/3,16/17;t=np.linspace(lo,hi,401);ta,tb=np.meshgrid(t,t)
    u=7/16-7*ta/16+7*ta*tb/32;vv=7*ta*(1-tb)/32
    b.contourf(ta,tb,(u<vv).astype(int),levels=[-.5,.5,1.5],colors=[LIGHT,SAND])
    b.contour(ta,tb,u-vv,levels=[0],colors=[ORANGE],linewidths=.6)
    b.plot(t,t,color=TEAL,lw=1.25);b.plot(hi,lo,'o',ms=3.2,color=ORANGE,mec='white',mew=.4)
    b.text(.37,.84,'A,B,U,C',fontsize=6.6)
    b.text(.57,.43,'A,B,V,C',fontsize=6.6,color=ORANGE)
    b.text(.36,.64,'Shared',fontsize=6.5,color=TEAL,rotation=43,rotation_mode='anchor')
    b.set(xlim=(lo-.02,hi+.02),ylim=(lo-.02,hi+.02),xlabel=r'$t_A$',ylabel=r'$t_B$')
    b.set_xticks([lo,hi],['1/3','16/17']);b.set_yticks([lo,hi],['1/3','16/17'])
    b.yaxis.set_label_coords(-.19,.5)
    fig.text(.025,.055,r'Common policy: $t_A=t_B=m/(m+w)$',fontsize=7.5,color=TEAL)
    save(fig,'domain_mechanism','Independent source fractions admit a membership excluded by shared weights.',
         'Exact five-node construction; domain_mechanism.source.npz retains full analytic grid and boundary curves.',
         kind='analytic mechanism',selection='Full declared constructed interval, not empirical observations.',
         change='The redundant score-boundary panel from the wide revision is expressed by the exact text derivation.')


def output_certificate():
    fig,ax=canvas(89,69)
    text(ax,2,4,'(a) Retain the core',7.4,weight='bold')
    text(ax,49,4,'(b) Bound every node',7.4,weight='bold')
    for j in range(6):
        for i in range(8):
            fc=TEAL if i<2 else PALE if (i+j)%3==0 or i==3 else LIGHT
            rect(ax,2+j*3.15,12+i*3.0,3.15,3,fc,'white',lw=.3)
    text(ax,23,15,r'Source $\hat H$',7.5)
    text(ax,23,24,r'Kept $I_j$',7.5)
    text(ax,23,33,r'Tail $\tau_j$',7.5)
    for y,label,fc,col in [(11,r'$T$: lower $B_T$',PALE,TEAL),
                            (21,r'$\mathcal{C}\backslash T$: upper $B_U$',LIGHT,GRAY),
                            (31,r'$V\backslash\mathcal{C}$: upper $B_O$',SAND,ORANGE)]:
        rect(ax,49,y,38,7.2,fc,col,lw=.65);text(ax,68,y+3.6,label,7.5,ha='center')
    arrow(ax,(42,24),(48,24),GRAY,lw=.7)
    text(ax,2,42,'(c) Test every policy cell',7.4,weight='bold')
    rect(ax,4,47,23,17,PALE,TEAL,lw=.65)
    ax.plot([15.5,15.5],[47,64],color='white',lw=.8);ax.plot([4,27],[55.5,55.5],color='white',lw=.8)
    ax.plot([21.25,21.25],[55.5,64],color='white',lw=.8)
    text(ax,1.5,55.5,r'$w$',7,ha='center');text(ax,15.5,66.8,r'$m$',7,ha='center')
    text(ax,32,49,r'$\hat D>0$',7.5,color=TEAL)
    text(ax,32,56,r'$B_T>\max(B_U,B_O)+2E$',7.5)
    text(ax,32,63,'All cells pass: STABLE',7.1,color=TEAL,weight='bold')
    save(fig,'domain_output_certificate','Coherent source interactions and output tails support every-node, every-policy tests.',
         'paper/main.tex Eqs. 8-12; src/policy_domain_fast.py',kind='compact certificate schematic',
         selection='Illustrative matrix and partition; no measured cell count or retained fraction is implied.')


def performance():
    fig,axes=plt.subplots(1,2,figsize=(89/25.4,54/25.4))
    fig.subplots_adjust(left=.14,right=.98,bottom=.29,top=.82,wspace=.30)
    rows=[]
    for ax,(ds,display,n),letter in zip(axes,v.DATASETS,'ab'):
        a=[r for r in v.TABLES['A_core_matched'] if r['dataset']==ds]
        ratios=np.array([r['primary_total_seconds']/r['comparison_total_seconds'] for r in a])
        qvr=a[0]['primary_resolved_all_cut_queries'];ql=a[0]['comparison_resolved_all_cut_queries'];p=qvr/n*100;l=ql/n*100
        ax.plot([np.median(ratios),1],[p,l],color=GRAY,lw=.7)
        ax.plot([1],[l],'s',ms=3.6,color=GRAY)
        ax.plot(ratios,[p]*3,'o',ms=3,color=TEAL,mec='white',mew=.35)
        ax.annotate(f'{qvr:,}',xy=(np.median(ratios),p),xytext=(0,6),textcoords='offset points',ha='left',fontsize=6.8,color=TEAL,fontweight='bold')
        ax.annotate(f'{ql:,}',xy=(1,l),xytext=(2,-5),textcoords='offset points',ha='left',va='top',fontsize=6.8,color=GRAY)
        gains=100*(1-ratios)
        ax.text(.04,.12,f'{gains.min():.1f}–{gains.max():.1f}%\nless cost',transform=ax.transAxes,fontsize=6.8,color=TEAL,fontweight='bold')
        ax.set(xlim=(.22,1.34),ylim=(97.5,100.2),xticks=[.25,.5,1],yticks=[98,99,100],xlabel='Cost / Lift')
        ax.text(0,1.13,f'({letter}) {display}',transform=ax.transAxes,fontsize=7.4,fontweight='bold')
        for b,ratio in enumerate(ratios):rows.append(dict(dataset=ds,block=b,ratio=ratio,questions=n,primary_resolved=qvr,lift_resolved=ql))
    axes[0].set_ylabel('Resolved (%)');axes[1].set_yticklabels([])
    handles=[Line2D([],[],marker='o',ls='',color=TEAL,ms=3.5,label='ViewRank'),Line2D([],[],marker='s',ls='',color=GRAY,ms=3.5,label='SourceLift')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.56,.005),borderaxespad=.1,ncol=2,fontsize=6.7,handletextpad=.3,columnspacing=1.2)
    with (OUT/'domain_performance.source.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(fig,'domain_performance','Matched compressed certification lowers complete cost while resolving more audits.',
         str(v.REPORT_PATH)+' tables.A_core_matched',blocks=[0,1,2],
         normalization='Within A block: ViewRank complete cost divided only by its paired SourceLift; all original questions including UNKNOWN.',
         visual_axis='Labeled 97.5-100.2% resolution range; counts label the exact complete populations.')


def coupling():
    fig,ax=plt.subplots(figsize=(89/25.4,43/25.4))
    fig.subplots_adjust(left=.16,right=.95,bottom=.27,top=.94)
    rows=[]
    for ds,display,n in v.DATASETS:
        f=v.FACTS['datasets'][ds]['strata'];rs=np.arange(1,5);vals=np.array([f[str(r)]['raw_paired_ratios'] for r in rs]);med=np.median(vals,axis=1)
        color,marker=(TEAL,'o') if ds.startswith('hot') else (BLUE,'s')
        ax.plot(rs,med,color=color,marker=marker,ms=3.3,lw=1,label=display)
        for j,r in enumerate(rs):
            ax.plot([r]*3,vals[j],'.',ms=2.5,color=color)
            rows.extend(dict(dataset=ds,r=int(r),block=b,questions=f[str(r)]['questions'],ratio=float(val)) for b,val in enumerate(vals[j]))
        ax.annotate(f'{med[-1]:.2f}×',xy=(4,med[-1]),xytext=(-1,5),ha='right',textcoords='offset points',fontsize=7,color=color,fontweight='bold')
    ax.axhline(1,lw=.6,ls='--',color=GRAY)
    ax.set(xlim=(.8,4.35),ylim=(.7,5.6),xticks=[1,2,3,4],yticks=[1,2,3,4,5],xlabel=r'Modified sources $r$',ylabel='Active cost: Lift / VR')
    ax.legend(loc='upper left',fontsize=6.8,handlelength=1.4)
    with (OUT/'domain_coupling.source.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(fig,'domain_coupling','The largest matched active-cost gains occur in multi-source queries.',
         str(v.FACTS_PATH),blocks=[0,1,2],
         denominator='Every original question in each r stratum, including UNKNOWN; startup/setup excluded. Ratios of summed active costs.')


def controls():
    fig,axes=plt.subplots(1,2,figsize=(89/25.4,51/25.4))
    fig.subplots_adjust(left=.14,right=.98,bottom=.24,top=.84,wspace=.30)
    frozen=[r for r in v.TABLES['method_blocks'] if r['group']=='B']
    methods=[('fine_preview','VR',TEAL,'o'),('no_core_preview_fine','Core−',ORANGE,'X'),
             ('scan_preview_fine','Scan',BLUE,'s'),('interval_preview_fine','Int.',GRAY,'D'),
             ('adaptive_preview','2-tier',PURPLE,'^')]
    for ax,(ds,display,n),letter in zip(axes,v.DATASETS,'ab'):
        d=[r for r in frozen if r['dataset']==ds];base=next(r for r in d if r['method']=='fine_preview')
        for method,label,color,marker in methods:
            r=next(r for r in d if r['method']==method);x=r['total_seconds']/base['total_seconds'];y=r['resolved_all_cut_fraction']*100
            ax.plot(x,y,marker=marker,ms=3.5,color=color,mec='white',mew=.35,ls='')
            if label=='Scan':label+=f'\n{x:.2f}×';offset=(-2,-16);ha='right'
            elif label=='VR':offset=(3,-11);ha='left'
            elif label=='2-tier':offset=(2,6);ha='left'
            elif label=='Int.':offset=(4,-2);ha='left'
            else:offset=(3,5);ha='left'
            ax.annotate(label,xy=(x,y),xytext=offset,textcoords='offset points',fontsize=6.8,color=color,ha=ha)
        ax.set(xlim=(.65,5.9),ylim=(0,115),xticks=[1,3,5],yticks=[0,50,100],xlabel='Cost / VR')
        ax.text(0,1.12,f'({letter}) {display}',transform=ax.transAxes,fontsize=7.4,fontweight='bold')
    axes[0].set_ylabel('Resolved (%)');axes[1].set_yticklabels([])
    (OUT/'domain_controls.source.json').write_text(json.dumps(frozen,indent=2),encoding='utf8')
    save(fig,'domain_controls','Core coherence and all-output pruning improve the cost-resolution operating point.',
         str(v.REPORT_PATH)+' tables.method_blocks group B',blocks=[0],
         labels={'VR':'ViewRank','Core−':'NoCore','Scan':'All-row Scan','Int.':'Independent intervals','2-tier':'Optional two-tier controller'})


def budget():
    fig,axes=plt.subplots(1,2,figsize=(89/25.4,53/25.4))
    fig.subplots_adjust(left=.14,right=.98,bottom=.24,top=.78,wspace=.30)
    data=v.TABLES['C_certificate_capacity'];rows=[]
    for ax,(ds,display,n),letter in zip(axes,v.DATASETS,'ab'):
        for method,label,color,marker,ls in [('fine_preview','Sparse',TEAL,'o','-'),
              ('dp_preview_fine','DensePruned',BLUE,'s','--'),('lift_sparse_fine_preview','Lift',GRAY,'^',':')]:
            a=sorted([r for r in data if r['dataset']==ds and r['method']==method],key=lambda r:r['budget_bytes'])
            x=[r['budget_bytes']/1024**2 for r in a];counts=[r['every_cut_stable_queries'] for r in a];y=np.array(counts)/n*100
            ax.plot(x,y,color=color,marker=marker,ls=ls,ms=3.3,lw=1,label=label,mec='white',mew=.35);rows.extend(a)
            if method=='fine_preview':
                ax.annotate(f'{counts[0]:,}',xy=(1,y[0]),xytext=(2,16),textcoords='offset points',fontsize=6.8,color=TEAL,fontweight='bold')
                if ds.startswith('hot'):ax.annotate('905→904',xy=(8,y[-1]),xytext=(0,6),textcoords='offset points',ha='center',fontsize=6.5,color=TEAL)
            if method=='dp_preview_fine' and y[0]>0:
                ax.annotate(f'{counts[0]:,}',xy=(1,y[0]),xytext=(3,-11),textcoords='offset points',fontsize=6.8,color=BLUE)
        ax.set(xscale='log',xlim=(.8,20),ylim=(-3,107),xticks=[1,4,16],yticks=[0,50,100],xlabel='Budget (MiB)')
        ax.set_xticklabels(['1','4','16']);ax.minorticks_off()
        ax.text(0,1.17,f'({letter}) {display}',transform=ax.transAxes,fontsize=7.4,fontweight='bold')
    axes[0].set_ylabel('All-cut STABLE (%)');axes[1].set_yticklabels([])
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.53,1.01),ncol=3,fontsize=6.5,handlelength=1.3,handletextpad=.4,columnspacing=.9)
    (OUT/'domain_budget.source.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    save(fig,'domain_budget','Compressed retention improves certificate capacity at the smallest measured budget.',
         str(v.REPORT_PATH)+' tables.C_certificate_capacity',blocks=[0],
         metric='All original questions; all three cutoffs STABLE. Actual-point witness search uniformly off.',
         preserved_nonmonotonicity='Hot Sparse: 510,905,904; real cache-history effect retained.')


if __name__=='__main__':
    selected=set(sys.argv[1:])
    for name,fn in [('example',example),('overview',overview),('mechanism',mechanism),('output',output_certificate),
                    ('performance',performance),('coupling',coupling),('controls',controls),('budget',budget)]:
        if not selected or name in selected:fn();print(name,'column export')
