"""Manuscript visual revision: evidence-first vector figures, frozen results.

Python is the established graphics backend. Diagrams are explanatory, not
measurements; all quantitative panels use every frozen question/block. The
layout-matched cohort A is normalized only to its own paired lifting control.
Source-count active costs exclude startup; controls and capacity have one block.
The original execution code, graphs, seeds and result ledgers are never modified.
"""
from pathlib import Path
from hashlib import sha256
import csv
import json
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'figures'
SKILL = Path.home()/'.codex/skills/nature-figure/scripts'
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

TEAL = '#176c74'
PALE = '#e3f0ef'
INK = '#243440'
GRAY = '#73818f'
LIGHT = '#eef1f4'
ORANGE = '#b76832'
SAND = '#f7e9db'
BLUE = '#426a9a'
PURPLE = '#82679a'
mpl.rcParams.update({
    'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7.2, 'axes.labelsize':7.5, 'axes.titlesize':8,
    'xtick.labelsize':6.7, 'ytick.labelsize':6.7,
    'text.color':INK, 'axes.labelcolor':INK, 'xtick.color':INK, 'ytick.color':INK,
    'pdf.fonttype':42, 'svg.fonttype':'none', 'axes.linewidth':.65,
    'axes.spines.top':False, 'axes.spines.right':False, 'legend.frameon':False,
    'mathtext.fontset':'dejavusans', 'savefig.facecolor':'white',
})
REPORT_PATH = ROOT/'results/domain_supplement_v3/evidence_summary_v3/report.json'
FACTS_PATH = ROOT/'paper/generated/domain_source_count_facts.json'
REPORT = json.loads(REPORT_PATH.read_text(encoding='utf8'))
TABLES = REPORT['tables']
FACTS = json.loads(FACTS_PATH.read_text(encoding='utf8'))
DATASETS = [('hotpot_pool7405','HotpotQA',1000),('musique_pool2417','MuSiQue',2417)]


def text(ax,x,y,s,size=7.2,color=INK,ha='left',va='center',weight='normal',**kw):
    s=s.replace(r'\widehat',r'\hat')
    if '$' in s and ('_' in s or '^' in s or r'\hat' in s):
        size=max(size,7.5)  # Keep live math subscripts above the 5 pt glyph floor.
    return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va=va,
                   fontweight=weight,linespacing=1.25,**kw)


def rect(ax,x,y,w,h,fc='white',ec=GRAY,lw=.75,rounding=0):
    p = (FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rounding}',
                        facecolor=fc,edgecolor=ec,linewidth=lw)
         if rounding else Rectangle((x,y),w,h,facecolor=fc,edgecolor=ec,linewidth=lw))
    ax.add_patch(p)
    return p


def arrow(ax,start,end,color=INK,lw=.85,style='-|>',rad=0,ls='-'):
    p=FancyArrowPatch(start,end,arrowstyle=style,mutation_scale=7.2,
                      linewidth=lw,color=color,connectionstyle=f'arc3,rad={rad}',
                      linestyle=ls,shrinkA=0,shrinkB=0)
    ax.add_patch(p)
    return p


def route(ax,points,color=INK,lw=.8,ls='-'):
    xs,ys=zip(*points)
    ax.plot(xs,ys,color=color,lw=lw,ls=ls)
    arrow(ax,points[-2],points[-1],color,lw)


def canvas(w,h):
    fig=plt.figure(figsize=(w/25.4,h/25.4))
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,w),ylim=(h,0));ax.axis('off')
    return fig,ax


def save(fig,name,claim,source,kind='measurement',**extra):
    stem=OUT/name
    fig.canvas.draw()
    layout_options = ({'panel_ids':[chr(97+i) for i in range(len(fig.axes))],
                       'row_groups':[[chr(97+i) for i in range(len(fig.axes))]]}
                      if len(fig.axes)>1 else {})
    require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',
                                      overlay_svg=str(stem)+'.alignment.svg',strict=True,**layout_options)
    fig.savefig(str(stem)+'.pdf')
    fig.savefig(str(stem)+'.svg')
    fig.savefig(str(stem)+'.png',dpi=600)
    contract=dict(version='visual-revision-v1',claim=claim,type=kind,
                  source=source,backend='Python/Matplotlib',
                  selection='All original questions and all recorded blocks in each declared cohort.',
                  statistics='Three-block dots/ranges are descriptive, not confidence intervals; single-block panels have no error bars.',
                  input_report_sha256=sha256(REPORT_PATH.read_bytes()).hexdigest(),
                  script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    contract.update(extra)
    Path(str(stem)+'.contract.json').write_text(json.dumps(contract,indent=2),encoding='utf8')
    plt.close(fig)


def example():
    fig,ax=canvas(89,67)
    text(ax,2,3.5,'One question, two strictly certified top-2 sets',8,weight='bold')
    text(ax,2,11.5,'Which fanzine interviewed the Sex Pistols bassist\nwhile he was in The Flowers of Romance?',7.2)
    # Exact linear mapping of the declared weight rectangle and saved policies.
    x0,y0,w,h=8,30,25,25
    rect(ax,x0,y0,w,h,PALE,TEAL)
    def xy(m,v):return x0+(m-1)/3*w,y0+h-(v-.25)/1.75*h
    for m,v,tag,offset in [(1,1,'P1',(2,3)),(2.5,1.125,'P2',(2,-3))]:
        x,y=xy(m,v);ax.plot(x,y,'o',ms=4.7,color=TEAL,mec='white',mew=.6)
        text(ax,x+offset[0],y+offset[1],tag,6.8,color=TEAL,weight='bold')
    text(ax,20.5,60,r'Mention weight $m$',6.7,ha='center')
    text(ax,2.4,42.5,r'Weak weight $w$',6.7,ha='center',rotation=90)
    for x,s in [(8,'1'),(33,'4')]:text(ax,x,57,s,6.5,ha='center')
    for y,s in [(30,'2'),(55,'0.25')]:text(ax,6.3,y,s,6.5,ha='right')
    text(ax,20.5,24.5,'Legal policy domain',7,ha='center',weight='bold')
    for y,title,second,c,fc in [(24,'P1: (1, 1)','Ex Pistols',GRAY,LIGHT),
                               (44,'P2: (2.5, 1.125)','Sid Vicious*',TEAL,PALE)]:
        rect(ax,42,y,45,18,fc,c,rounding=1)
        text(ax,44,y+3.5,title,7,color=c,weight='bold')
        text(ax,44,y+9,'1  Flowers of Romance*',7)
        text(ax,44,y+14,'2  '+second,7,color=c,weight='bold')
    save(fig,'domain_example','Two legal common-weight policies change a real top-2 membership.',
         'natural_instability_story_v3.json; HotpotQA qi264, qid5adea9ea55429939a52fe934',
         kind='selected real instability witness',
         selection='One positive support illustration selected post hoc; no accuracy claim.',
         policies=[[1,1],[2.5,1.125]],boundary_gap_lower=[1.2292842618392002e-5,5.3704523604850334e-5])


def overview():
    fig,ax=canvas(183,87)
    text(ax,2,3,'ViewRank: one audit covers a continuous policy domain',8.8,weight='bold')
    rect(ax,2,10,26,21,LIGHT,GRAY,rounding=1)
    text(ax,15,14,'Audit request',7.4,ha='center',weight='bold')
    text(ax,15,23,r'$h,\ S,\ \Theta,\ K$',8.4,ha='center')
    rect(ax,2,43,26,22,LIGHT,GRAY,rounding=1)
    text(ax,15,48,'Fixed graph',7.4,ha='center',weight='bold')
    text(ax,15,57,r'$P_0$, relation groups',6.7,ha='center')
    route(ax,[(28,20),(32,20),(32,49),(36,49)])
    arrow(ax,(28,58),(36,58))
    rect(ax,36,10,94,21,PALE,TEAL,rounding=1)
    text(ax,39,14.5,'Reusable response directory  (Sec. V)',7.6,color=TEAL,weight='bold')
    text(ax,39,21.2,r'Group/source keys: $\widehat R_j$, source stencil $\widehat H$, retained rows $I_j$',7)
    text(ax,39,27,r'Residuals $\rho_j$ and tails $\tau_j$; LRU admission + active pins',7)
    rect(ax,143,10,37,21,LIGHT,GRAY,rounding=1)
    text(ax,161.5,15,'Numerical backend',7.2,ha='center',weight='bold')
    text(ax,161.5,21,'Base / signed responses',6.6,ha='center')
    text(ax,161.5,27,'Actual policy solves',6.6,ha='center')
    arrow(ax,(143,18),(130,18),TEAL)
    text(ax,136.5,14,'fill',6.5,ha='center')
    arrow(ax,(130,27),(143,27),GRAY,ls='--')
    text(ax,136.5,23.4,'miss',6.5,ha='center')
    stages=[(36,26,'1  Prepare',r'Seed + base $\widehat y$','Current-base preview'),
            (71,28,'2  Compile (III)',r'Affine $r\!\times\!r$ core',r'$\widehat D,\widehat F_j$: degree $\leq r$'),
            (108,34,'3  Certify (IV)','Candidate + outside bounds','Bernstein cells; all cuts'),
            (151,29,'4  Witness (V)','Two actual policies','Strict top-k checks')]
    for x,w,title,line1,line2 in stages:
        rect(ax,x,43,w,22,'white',TEAL if x in (71,108) else GRAY,rounding=1)
        rect(ax,x,43,w,6,PALE if x in (71,108) else LIGHT,ec='none')
        text(ax,x+w/2,46,title,7.1,ha='center',weight='bold',color=TEAL if x in(71,108) else INK)
        text(ax,x+w/2,54,line1,6.8,ha='center')
        text(ax,x+w/2,60,line2,6.5,ha='center')
    arrow(ax,(62,54),(71,54));arrow(ax,(99,54),(108,54))
    arrow(ax,(49,31),(49,43),TEAL);text(ax,50.5,37,'rows',6.5)
    arrow(ax,(85,31),(85,43),TEAL);text(ax,86.5,37,r'$\widehat H$',7)
    arrow(ax,(120,31),(120,43),TEAL);text(ax,121.5,37,r'$I_j,\tau_j,\rho_j$',6.7)
    route(ax,[(142,54),(146.5,54),(146.5,38),(165,38),(165,43)],GRAY)
    text(ax,154,34.7,'if unresolved',6.5,ha='center')
    arrow(ax,(173,43),(173,31),GRAY,ls='--')
    for x,w,label,c,fc in [(108,27,'STABLE',TEAL,PALE),(141,18,'UNSTABLE',ORANGE,SAND),
                           (163,17,'UNKNOWN',GRAY,LIGHT)]:
        rect(ax,x,76,w,7,fc,c,rounding=.8);text(ax,x+w/2,79.5,label,6.6,color=c,ha='center',weight='bold')
    arrow(ax,(121.5,65),(121.5,76),TEAL);text(ax,123,70.5,'all cells pass',6.5)
    route(ax,[(164,65),(164,70),(150,70),(150,76)],ORANGE)
    route(ax,[(171.5,65),(171.5,76)],GRAY)
    text(ax,2,74,'Persistent responses + admitted family work',7,color=TEAL,weight='bold')
    text(ax,2,80.5,'share one byte budget; the graph/backend are separate.',6.8)
    save(fig,'domain_overview','Reusable graph responses feed an affine source compiler and a whole-output domain certificate.',
         ['main.tex Sections II-V','src/policy_domain_source.py','src/policy_domain_polynomial.py',
          'src/policy_domain_fast.py','src/retained_policy_family.py'],kind='system schematic',
         selection='Grounded module and data-flow diagram, no empirical scale implied.',
         optional_controller='Not drawn; the frozen primary uses fixed fine precision.')


def mechanism():
    fig=plt.figure(figsize=(183/25.4,70/25.4))
    axes=[fig.add_axes([.025+i*.335,.23,.285,.59]) for i in range(3)]
    a,b,c=axes
    for i,ax in enumerate(axes):
        ax.text(0,1.22,f'({chr(97+i)})  '+['One shared relation policy','Relaxation admits extra policies','A strict boundary over the domain'][i],
                transform=ax.transAxes,fontsize=7.6,fontweight='bold',va='bottom')
    a.set(xlim=(0,10),ylim=(8,0));a.axis('off')
    pts={'A':(2,2),'B':(7.7,2),'U':(2,6),'V':(7.7,6),'C':(4.85,6)}
    for u,v,g,rad in [('A','B','m',0),('A','U','w',0),('B','U','m',0),('B','V','w',0)]:
        p,q=np.array(pts[u]),np.array(pts[v]);d=(q-p)/np.linalg.norm(q-p)
        arrow(a,p+d*.55,q-d*.55,TEAL if g=='m' else ORANGE,lw=1.15)
        mid=(p+q)/2
        tx,ty=(4.1,3.0) if u=='B' and v=='U' else (mid[0]+.2,mid[1]-.35)
        text(a,tx,ty,g,7.8,color=TEAL if g=='m' else ORANGE)
    for node,(x,y) in pts.items():
        a.add_patch(Circle((x,y),.53,facecolor=PALE if node in ('A','B') else 'white',edgecolor=TEAL if node in ('A','B') else GRAY,lw=.9))
        text(a,x,y,node,8,ha='center',weight='bold')
    text(a,2,.45,r'$h_A=7/8$',7.1,ha='center');text(a,5.7,5.0,r'$h_C=1/8$',7.1,ha='center')
    for node in ('U','V','C'):
        x,y=pts[node];arrow(a,(x-.32,y+.44),(x+.32,y+.44),GRAY,rad=.95)
    text(a,5,7.65,r'$t_A=t_B=t=m/(m+w)$',7.5,ha='center',color=TEAL)
    lo,hi=1/3,16/17;t=np.linspace(lo,hi,401);ta,tb=np.meshgrid(t,t)
    u=7/16-7*ta/16+7*ta*tb/32;v=7*ta*(1-tb)/32
    scores=np.stack([np.full_like(ta,7/16),7*ta/32,u,v,np.full_like(ta,1/8)])
    excluded=np.argmin(scores,axis=0)
    b.contourf(ta,tb,(excluded==2).astype(int),levels=[-.5,.5,1.5],colors=[LIGHT,SAND])
    b.contour(ta,tb,u-v,levels=[0],colors=[ORANGE],linewidths=.65)
    b.plot(t,t,color=TEAL,lw=2)
    b.plot(hi,lo,'o',ms=4.5,color=ORANGE,mec='white',mew=.6)
    b.text(.37,.83,'A, B, U, C',fontsize=7)
    b.text(.66,.43,'A, B, V, C',fontsize=6.8,color=ORANGE)
    b.text(.37,.62,'Shared',fontsize=7,color=TEAL,rotation=39)
    b.set(xlim=(lo-.02,hi+.02),ylim=(lo-.02,hi+.02),xlabel=r'Source A fraction $t_A$',ylabel=r'Source B fraction $t_B$')
    b.set_xticks([lo,.65,hi],['1/3','0.65','16/17']);b.set_yticks([lo,.65,hi],['1/3','0.65','16/17'])
    cands=np.vstack([np.full_like(t,7/16),7*t/32,7/16-7*t/16+7*t*t/32,np.full_like(t,1/8)])
    bottom=cands.min(axis=0);outside=7*t*(1-t)/32
    assert np.all(bottom>outside)
    c.fill_between(t,outside,bottom,color=PALE)
    c.plot(t,bottom,color=TEAL,lw=1.5,label='Lowest selected score')
    c.plot(t,outside,color=ORANGE,lw=1.4,label='Outside node V')
    c.text(.66,.092,'Selected: A, B, U, C',fontsize=6.8,ha='center',color=TEAL)
    c.text(.62,.018,'Outside: V',fontsize=6.8,ha='center',color=ORANGE)
    c.set(xlim=(lo,hi),ylim=(0,.15),xlabel=r'Common fraction $t$',ylabel='Score')
    c.yaxis.set_label_coords(-.105,.5)
    c.set_xticks([lo,.65,hi],['1/3','0.65','16/17']);c.set_yticks([0,.05,.10,.15])
    fig.text(.025,.065,'Mention links: teal   |   Weak links: orange   |   Absorbing U, V, C; damping 1/2',fontsize=7)
    np.savez_compressed(OUT/'domain_mechanism.source.npz',t_A=ta,t_B=tb,scores=scores,
                        common_t=t,lowest_selected=bottom,outside_v=outside)
    save(fig,'domain_mechanism','Common weights exclude the incompatible source assignments of an independent relaxation.',
         'Exact five-node construction in main.tex; formulas exported with full 401x401 grid.',
         kind='analytic constructed mechanism',selection='Full analytic declared interval, not an empirical query sample.',
         formulas=dict(U='7/16 - 7*t_A/16 + 7*t_A*t_B/32',V='7*t_A*(1-t_B)/32',C='1/8'),
         proof='On the common diagonal, all A,B,U,C strictly exceed V; lifted point (16/17,1/3) changes membership.')


def output_certificate():
    fig,ax=canvas(183,57)
    for x,label in [(2,'(a)  Retained response coordinates'),(64,'(b)  Cover the complete output'),(127,'(c)  Cover the continuous domain')]:
        text(ax,x,4,label,7.6,weight='bold')
    text(ax,13,11,r'$2r$ signed responses',7.2,ha='center')
    x0,y0,cw,ch=4,15,3.8,3.3
    for j in range(6):
        for i in range(8):
            fc=TEAL if i<2 else (PALE if (i+j)%3==0 or i==3 else LIGHT)
            rect(ax,x0+j*cw,y0+i*ch,cw,ch,fc,'white',lw=.35)
    text(ax,30,18.3,r'Source stencil $\widehat H$',6.8)
    text(ax,30,27.8,r'Kept rows $I_j$',6.8)
    text(ax,30,37.7,r'Omitted: $|\widehat R_{ij}|\leq\tau_j$',6.8)
    text(ax,3,48.5,'Sources + retained rows + base preview',6.8)
    text(ax,3,53,r'form the candidate union $\mathcal{C}$.',6.8)
    arrow(ax,(59,29),(64,29),GRAY)
    for y,h,label,fc,c in [(12,8,r'Selected $T$: lower bound $B_T$',PALE,TEAL),
                           (25,8,r'Candidates $\mathcal{C}\!\setminus\!T$: upper bound $B_U$',LIGHT,GRAY),
                           (38,8,r'All nodes outside $\mathcal{C}$: tail bound $B_O$',SAND,ORANGE)]:
        rect(ax,65,y,55,h,fc,c,rounding=.6);text(ax,92.5,y+h/2,label,6.9,ha='center')
    text(ax,92.5,52,r'$B_T>\max(B_U,B_O)+2E$',8,ha='center',color=TEAL,weight='bold')
    arrow(ax,(121.5,29),(126.5,29),GRAY)
    # A schematic partition; no sample locations, measured cell counts or success fractions.
    x,y,w,h=135,13,39,28
    rect(ax,x,y,w,h,PALE,TEAL)
    ax.plot([x+w/2,x+w/2],[y,y+h],color='white',lw=1)
    ax.plot([x,x+w],[y+h/2,y+h/2],color='white',lw=1)
    ax.plot([x+3*w/4,x+3*w/4],[y+h/2,y+h],color='white',lw=1)
    ax.plot([x+w/2,x+w],[y+3*h/4,y+3*h/4],color='white',lw=1)
    text(ax,154.5,9,r'Bernstein cells over $\Theta$',7,ha='center')
    text(ax,131,27,r'$w$',7.5,ha='center');text(ax,154.5,44.5,r'$m$',7.5,ha='center')
    text(ax,154.5,50,r'Each cell: $\widehat D>0$ + boundary test',6.9,ha='center')
    text(ax,154.5,54.5,'Every cell passes: STABLE',7,ha='center',color=TEAL,weight='bold')
    save(fig,'domain_output_certificate','Retained coordinates plus certified tails cover every output node on every policy cell.',
         'main.tex Eqs. 8-12 and src/policy_domain_fast.py _prepare / certify',kind='certificate schematic',
         selection='Illustrative matrix and partition; no measured sparsity, cell count or frequency is implied.',
         color_semantics='Teal: mandatory source coordinates / selected set; pale teal: other retained coordinates; gray: omitted coordinates; orange: outside bound.')


def performance():
    fig,axes=plt.subplots(1,3,figsize=(183/25.4,76/25.4))
    fig.subplots_adjust(left=.075,right=.985,bottom=.26,top=.79,wspace=.40)
    rows=[]
    for i,(ds,display,n) in enumerate(DATASETS):
        ax=axes[i];a=[v for v in TABLES['A_core_matched'] if v['dataset']==ds]
        ratios=np.array([v['primary_total_seconds']/v['comparison_total_seconds'] for v in a])
        qvr=a[0]['primary_resolved_all_cut_queries'];ql=a[0]['comparison_resolved_all_cut_queries']
        p=qvr/n*100;l=ql/n*100
        ax.plot([1,np.median(ratios)],[l,p],lw=1,color=GRAY,zorder=1)
        ax.scatter(np.ones(3),[l]*3,marker='s',s=31,color=GRAY,zorder=4)
        for block,ratio in enumerate(ratios):
            ax.plot(ratio,p,'o',ms=4.3,mec='white',mew=.5,color=TEAL,zorder=5)
            rows.append(dict(dataset=ds,block=block,metric='paired_complete_cost_primary_over_lift',value=ratio,
                             primary_resolved=qvr,lift_resolved=ql,questions=n))
        ax.set(xlim=(.25,1.18),ylim=(97.5,100.15),xlabel='Complete cost / paired Lift',ylabel='All-cut audits resolved (%)')
        ax.set_xticks([.25,.5,.75,1]);ax.set_yticks([98,99,100])
        ax.text(.03,1.20,f'({chr(97+i)})  {display}: {n:,} questions',transform=ax.transAxes,fontsize=7.8,fontweight='bold')
        ax.annotate(f'VR: {qvr:,}',xy=(np.median(ratios),p),xytext=(2,9),textcoords='offset points',ha='left',fontsize=7.1,color=TEAL,fontweight='bold')
        ax.annotate(f'Lift\n{ql:,}',xy=(1,l),xytext=(3,-8),textcoords='offset points',ha='left',va='top',fontsize=7.1,color=GRAY)
        gain=100*(1-ratios)
        ax.text(.06,.13,f'{gain.min():.1f}–{gain.max():.1f}%\nless complete cost',transform=ax.transAxes,color=TEAL,fontsize=8.2,fontweight='bold')
        pvr=np.median([v['primary_p95_active_query_seconds'] for v in a])*1000
        pl=np.median([v['comparison_p95_active_query_seconds'] for v in a])*1000
        ax.text(.01,-.35,f'p95: {pvr:.0f} vs {pl:.0f} ms (VR / Lift)',transform=ax.transAxes,fontsize=6.8)
    ax=axes[2]
    for ds,display,n in DATASETS:
        f=FACTS['datasets'][ds]['strata'];rs=np.arange(1,5)
        vals=np.array([f[str(r)]['raw_paired_ratios'] for r in rs]);med=np.median(vals,axis=1)
        color,marker=(TEAL,'o') if ds.startswith('hot') else (BLUE,'s')
        ax.plot(rs,med,color=color,marker=marker,ms=3.7,lw=1.3,label=display)
        for j,r in enumerate(rs):
            ax.plot([r]*3,vals[j],'.',ms=3,color=color)
            rows.extend(dict(dataset=ds,block=b,metric=f'r{r}_active_lift_over_primary',value=v,
                             stratum_questions=f[str(r)]['questions']) for b,v in enumerate(vals[j]))
        ax.annotate(f'{med[-1]:.2f}×',xy=(4,med[-1]),xytext=(-2,5),ha='right',textcoords='offset points',color=color,fontsize=7.1,fontweight='bold')
    ax.axhline(1,color=GRAY,lw=.7,ls='--')
    ax.set(xlim=(.7,4.55),ylim=(.65,5.55),xticks=[1,2,3,4],yticks=[1,2,3,4,5],
           xlabel=r'Modified sources $r$',ylabel='Active cost: Lift / VR')
    ax.text(.03,1.20,'(c)  Source coupling',transform=ax.transAxes,fontsize=7.8,fontweight='bold')
    ax.legend(fontsize=6.8,loc='upper left',handlelength=1.5)
    ax.text(.01,-.35,'All r=1–4 strata; startup excluded here',transform=ax.transAxes,fontsize=6.8)
    fig.text(.075,.035,'Same compression and response tolerance; three complete paired blocks, all questions including UNKNOWN.',fontsize=7)
    keys=sorted({k for row in rows for k in row})
    with (OUT/'domain_performance.source.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    save(fig,'domain_performance','Shared-parameter certification lowers complete cost, resolves more audits and has its largest active-cost gains at larger r.',
         [str(REPORT_PATH),str(FACTS_PATH)],blocks=[0,1,2],
         normalization='Each A block divided only by its paired compressed SourceLift; all r strata shown. No cross-cohort absolute-time pooling.',
         visual_axis='Panels a,b use a labeled 97.5–100.15% resolution range to show the small coverage differences; complete counts are printed.')


def controls():
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,68/25.4))
    fig.subplots_adjust(left=.075,right=.98,bottom=.24,top=.80,wspace=.25)
    methods=[('fine_preview','ViewRank',TEAL,'o'),('no_core_preview','NoCore',ORANGE,'X'),
             ('scan_preview','All-row Scan',BLUE,'s'),('interval_preview','Independent intervals',GRAY,'D'),
             ('adaptive_preview','Two-tier',PURPLE,'^')]
    # Resolve exact frozen method names without altering their data.
    frozen=[r for r in TABLES['method_blocks'] if r['group']=='B']
    for ax,(ds,display,n),letter in zip(axes,DATASETS,'ab'):
        d=[r for r in frozen if r['dataset']==ds];base=next(r for r in d if r['method']=='fine_preview')
        for key,label,color,marker in methods:
            candidates=[r for r in d if r['method']==key]
            if not candidates:
                token={'no_core_preview':'no_core','scan_preview':'scan','interval_preview':'interval'}[key]
                candidates=[r for r in d if token in r['method']]
            r=candidates[0];x=r['total_seconds']/base['total_seconds'];y=r['resolved_all_cut_fraction']*100
            ax.scatter(x,y,s=42 if key=='fine_preview' else 30,marker=marker,color=color,zorder=5,edgecolors='white',linewidths=.55)
            if key=='fine_preview':
                ax.annotate('ViewRank',xy=(x,y),xytext=(8,-18),textcoords='offset points',color=color,fontweight='bold',fontsize=7.2)
            elif key=='adaptive_preview':
                ax.annotate('Two-tier',xy=(x,y),xytext=(8,7),textcoords='offset points',color=color,fontsize=7.1)
            elif key=='scan_preview':
                ax.annotate(f'All-row Scan\n{x:.2f}× cost',xy=(x,y),xytext=(-4,-27),ha='right',textcoords='offset points',color=color,fontsize=7.1)
            elif key=='no_core_preview':
                ax.annotate(f'NoCore\n{y:.1f}% resolved',xy=(x,y),xytext=(7,5),textcoords='offset points',color=color,fontsize=7.1)
            else:
                ax.annotate(f'Independent intervals\n{y:.1f}% resolved',xy=(x,y),xytext=(8,-6),textcoords='offset points',color=color,fontsize=7.1)
        ax.axvline(1,color='#c6ced4',lw=.65,ls='--')
        ax.set(xlim=(.65,5.65),ylim=(0,113),xticks=[1,2,3,4,5],yticks=[0,25,50,75,100],
               xlabel='Complete-workload cost / ViewRank',ylabel='All-cut audits resolved (%)')
        ax.text(0,1.16,f'({letter})  {display}: all {n:,} questions',transform=ax.transAxes,fontsize=8,fontweight='bold')
    fig.text(.075,.045,'Same original questions, policies and budget; one complete matched block. Read cost and resolution together.',fontsize=7)
    (OUT/'domain_controls.source.json').write_text(json.dumps(frozen,indent=2),encoding='utf8')
    save(fig,'domain_controls','The coherent source core and all-output pruning improve the measured cost–resolution operating point.',
         str(REPORT_PATH)+' tables.method_blocks group B',blocks=[0],
         tradeoffs='Independent intervals can be cheaper but resolve fewer requests; two-tier is optional and mixed across datasets.')


def budget():
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,71/25.4))
    fig.subplots_adjust(left=.075,right=.98,bottom=.27,top=.78,wspace=.25)
    data=TABLES['C_certificate_capacity']
    # Table C is one method-block row per budget, not a paired-comparison table.
    rows=[]
    for ax,(ds,display,n),letter in zip(axes,DATASETS,'ab'):
        for method,label,color,marker,ls in [('fine_preview','VR-Sparse',TEAL,'o','-'),
                  ('dp_preview_fine','VR-DensePruned',BLUE,'s','--'),
                  ('lift_sparse_fine_preview','Sparse SourceLift',GRAY,'^',':')]:
            a=sorted([r for r in data if r['dataset']==ds and r['method']==method],key=lambda r:r['budget_bytes'])
            x=np.array([r['budget_bytes']/1024**2 for r in a]);counts=[r['every_cut_stable_queries'] for r in a]
            y=np.array(counts)/n*100
            ax.plot(x,y,label=label,color=color,marker=marker,ls=ls,lw=1.35,ms=4,mec='white',mew=.5)
            rows.extend(a)
            if method=='fine_preview':
                ax.annotate(f'{counts[0]:,} / {n:,}\n({y[0]:.1f}%)',xy=(1,y[0]),xytext=(1.12,97),textcoords='data',va='top',
                            arrowprops=dict(arrowstyle='-',color=TEAL,lw=.6),color=TEAL,fontsize=7.2,fontweight='bold')
                if ds.startswith('hot'):
                    ax.annotate('905 → 904',xy=(8,y[-1]),xytext=(0,8),textcoords='offset points',ha='center',color=TEAL,fontsize=7)
            if method=='dp_preview_fine':
                ax.annotate(f'{counts[0]:,} / {n:,}',xy=(1,y[0]),xytext=(8,-13),textcoords='offset points',color=BLUE,fontsize=7)
        ax.set(xscale='log',xlim=(.82,20),ylim=(-3,103),xticks=[1,4,16],yticks=[0,25,50,75,100],
               xlabel='Admitted numerical-state budget (MiB)',ylabel='All-cut STABLE questions (%)')
        ax.set_xticklabels(['1','4','16']);ax.minorticks_off()
        ax.text(0,1.15,f'({letter})  {display}: {n:,} questions',transform=ax.transAxes,fontsize=8,fontweight='bold')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.99),ncol=3,fontsize=7.1,columnspacing=2)
    fig.text(.075,.065,'Witness search off at every budget; all original questions remain in the denominator. One block per setting.',fontsize=7)
    fig.text(.075,.018,'Budget includes retained + family-private numerical state; it is distinct from process working set.',fontsize=7)
    (OUT/'domain_budget.source.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    save(fig,'domain_budget','Compressed retention improves whole-domain certificate capacity at the smallest measured budget.',
         str(REPORT_PATH)+' tables.C_certificate_capacity',blocks=[0],
         metric='All requested cutoffs STABLE; actual-point witness search uniformly disabled.',
         preserved_nonmonotonicity='HotpotQA Sparse: 510,905,904; query-dependent cache-miss previews are nonnested.')


if __name__=='__main__':
    requested=set(sys.argv[1:])
    for name,fn in [('example',example),('overview',overview),('mechanism',mechanism),
                    ('output',output_certificate),('performance',performance),('controls',controls),('budget',budget)]:
        if not requested or name in requested:
            fn();print(name,'exported')
