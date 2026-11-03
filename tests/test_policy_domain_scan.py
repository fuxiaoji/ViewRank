from fractions import Fraction as Q
import weakref

import numpy as np
import pytest

from src.policy_domain_polynomial import compile_domain,certify
from src import policy_domain_fast as outward
from src import policy_domain_dot as dot
from src import policy_domain_scan as scan


def exact_margin(domain,y,rows,R,mask,tails,ids,bounds,box):
    D,F=domain.controls(box);selected=set(map(int,ids));row_set=set(map(int,rows))
    outside=[Q.from_float(float(y[i])) for i in range(len(y)) if i not in row_set]
    minimum=None
    for c,d in enumerate(D):
        assert d>0
        f=[poly[c] for poly in F]
        error=Q(bounds['kappa'])*(Q(bounds['base_residual'])*d+sum(Q(float(r))*abs(a) for r,a in zip(bounds['atom_residuals'],f)))
        low=[];high=[]
        for i,node in enumerate(rows):
            value=Q(float(y[node]))*d-sum(Q(float(R[i,j]))*f[j] for j in range(len(f)) if mask[i,j])
            tail=sum(Q(float(tails[j]))*abs(f[j]) for j in range(len(f)) if not mask[i,j])
            (low if node in selected else high).append(value-tail if node in selected else value+tail)
        margins=[]
        if high:margins.append(min(low)-max(high)-2*error)
        if outside:margins.append(min(low)-max(outside)*d-sum(Q(float(t))*abs(a) for t,a in zip(tails,f))-2*error)
        for margin in margins:minimum=margin if minimum is None else min(minimum,margin)
    return Q(1) if minimum is None else minimum


def verify_leaves(domain,y,rows,R,mask,tails,ids,bounds,result):
    for leaf in result['leaves']:
        if leaf['status']=='STABLE':
            box=tuple(tuple(Q(v) for v in pair) for pair in leaf['box'])
            assert exact_margin(domain,y,rows,R,mask,tails,ids,bounds,box)>0


def near_case(r,units):
    y=np.zeros(r+2);y[r]=.5+units*np.spacing(.5);y[r+1]=.5
    R=np.zeros((len(y),2*r))
    domain=compile_domain(np.zeros((r,r)),np.zeros((r,r)),[0]*r,[0]*r,[0]*r,((.5,2),(.5,2)),range(r))
    args=(domain,y,np.arange(len(y)),R,np.ones_like(R,bool),np.zeros(2*r))
    return args,dict(kappa=2,base_residual=0,atom_residuals=np.zeros(2*r)),[r]


@pytest.mark.parametrize('r,units',[(1,10),(1,11),(2,14),(2,15),(3,18),(3,19),(4,22),(4,23)])
def test_all_eight_near_boundary_coverage_losses_are_recovered(r,units):
    args,bounds,ids=near_case(r,units)
    first=dot.certify_fast(*args,ids,**bounds)
    old=outward.certify_fast(*args,ids,**bounds)
    result=scan.certify_fast(*args,ids,**bounds)
    assert first['status']=='UNKNOWN' and old['status']==result['status']=='STABLE'
    assert result['scan_backend']=='outward_fallback'
    assert result['shared_scan']['fallback_cutoffs']==[1]
    assert result['shared_scan']['dot']['control_boxes']==1
    assert result['shared_scan']['outward_fallback']['control_boxes']==1
    assert result['leaves']==old['leaves']
    verify_leaves(*args,ids,bounds,result)


def physical(r=2,n=19):
    rng=np.random.default_rng(20261120+r)
    P=rng.uniform(.1,1,(n,n));P/=P.sum(0)
    fraction=rng.choice([0.,.25,.5,.75,1.],size=(n,n));np.fill_diagonal(fraction,0.)
    M=P*fraction;W=P-M-np.diag(P.diagonal());S=np.arange(r)
    seed=np.full(n,.2/n);seed[0]+=.8
    mu,nu=M[:,S].sum(0),W[:,S].sum(0)
    solved=np.linalg.solve(np.eye(n)-.5*P,np.column_stack((.5*seed,-.5*(M[:,S]-P[:,S]*mu),-.5*(W[:,S]-P[:,S]*nu))))
    y,R=solved[:,0],solved[:,1:]
    domain=compile_domain(R[S,:r],R[S,r:],mu,nu,y[S],((.5,2),(.25,2)),S)
    return domain,y,R,(P,M,W,S,seed)


@pytest.mark.parametrize('mode',['sparse','dense_pruned','dense','scan'])
@pytest.mark.parametrize('r',[1,2,3,4])
def test_all_physical_modes_share_dot_fallback_and_exact_leaf_proof(mode,r):
    domain,y,R,data=physical(r);original=R.copy()
    order=np.lexsort((np.arange(len(y)),-y));selected={k:order[:k] for k in (1,3,5)}
    kept=[set(np.argsort(-abs(R[:,j]))[:2])|set(domain.source_ids) for j in range(2*r)]
    pruned=np.array(sorted(set(order[:5])|set.union(*kept)))
    rows=np.arange(len(y)) if mode in ('dense','scan') else pruned
    if mode=='dense':mask=np.ones((len(rows),2*r),bool);tails=np.zeros(2*r)
    else:
        mask=np.array([[node in keep for keep in kept] for node in rows])
        tails=np.array([max((abs(R[i,j]) for i in range(len(y)) if i not in keep),default=0.) for j,keep in enumerate(kept)])
    kwargs=dict(kappa=2,base_residual=1e-12,atom_residuals=np.full(2*r,1e-12),chunk_rows=3)
    if mode in ('dense','dense_pruned'):
        args=(domain,y,rows,None,None,tails);kwargs['retained_columns']=[R[:,j] for j in range(2*r)]
        proof_mask=np.ones((len(rows),2*r),bool)
    else:args=(domain,y,rows,R[rows],mask,tails);proof_mask=mask
    baseline=outward.certify_many(*args,selected,**kwargs)
    result=scan.certify_many(*args,selected,**kwargs)
    np.testing.assert_array_equal(R,original)
    for k,ids in selected.items():
        old=baseline['cuts'][k];new=result['cuts'][k]
        assert old['status']!='STABLE' or new['status']=='STABLE'
        verify_leaves(domain,y,rows,R[rows],proof_mask,tails,ids,kwargs,new)
        if new['status']=='STABLE':
            P,M,W,S,seed=data
            for m,w in ((.5,.25),(.5,2),(2,.25),(2,2),(1.125,.875)):
                view=P.copy()
                for s in S:
                    raw=m*M[:,s]+w*W[:,s];raw=raw.copy();raw[s]+=P[s,s];view[:,s]=raw/raw.sum()
                truth=np.linalg.solve(np.eye(len(P))-.5*view,.5*seed)
                assert set(ids)==set(np.argsort(-truth,kind='stable')[:k])
    estimate=scan.scratch_upper_bound(domain,len(y),3,len(selected),len(rows))
    assert result['workspace_peak_bytes']<=estimate
    assert result['workspace_peak_bytes']>=baseline['workspace_peak_bytes']
    assert result['shared_scan']['dot_seconds']>=0 and result['shared_scan']['outward_seconds']>=0


def test_fallback_only_receives_unknown_cuts_and_dot_temporary_is_released(monkeypatch):
    args,bounds,_=near_case(1,20);domain,y,rows,R,mask,tails=args
    selected={1:[1],2:[1,2]};temporary=[]
    def first(*call_args,**kwargs):
        array=np.ones(128);temporary.append(weakref.ref(array))
        return dict(cuts={1:dict(status='STABLE',selected_ids=[1]),2:dict(status='UNKNOWN')},
            workspace_peak_bytes=8000,shared_scan=dict(control_boxes=1,candidate_node_controls=12),temporary=array)
    def fallback(*call_args,**kwargs):
        assert set(call_args[6])=={2}
        assert temporary[0]() is None
        return dict(cuts={2:dict(status='STABLE',selected_ids=[1,2])},workspace_peak_bytes=4000,
            shared_scan=dict(control_boxes=3,candidate_node_controls=36))
    monkeypatch.setattr(dot,'certify_many',first);monkeypatch.setattr(outward,'certify_many',fallback)
    result=scan.certify_many(*args,selected,**bounds)
    assert result['cuts'][1]['scan_backend']=='dot' and result['cuts'][2]['scan_backend']=='outward_fallback'
    assert result['workspace_peak_bytes']==8000
    assert result['shared_scan']['candidate_node_controls']==48


def test_fraction_inputs_go_directly_to_outward_without_recursion(monkeypatch):
    domain=compile_domain([[0]],[[0]],[0],[0],[Q(1,3)],((1,2),(1,2)),[0])
    def forbidden(*args,**kwargs):raise AssertionError('Fraction input must not enter dot')
    monkeypatch.setattr(dot,'certify_many',forbidden)
    result=scan.certify_many(domain,[Q(1,3),Q(2,3)],[0,1],[[0,0],[0,0]],[[1,1],[1,1]],
        [0,0],{1:[1]},kappa=2,base_residual=0,atom_residuals=[0,0])
    assert result['cuts'][1]['status']=='STABLE'
    assert result['cuts'][1]['scan_backend']=='outward_nonfloat'
    assert result['shared_scan']['nonfloat_direct_outward']


def test_iterable_inputs_remain_available_for_fallback():
    args,bounds,ids=near_case(4,22);domain,y,rows,R,mask,tails=args
    result=scan.certify_many(domain,y,(int(x) for x in rows),R,mask,(x for x in tails),
        {1:(x for x in ids)},kappa=2,base_residual=0,atom_residuals=(0 for _ in range(8)))
    assert result['cuts'][1]['status']=='STABLE' and result['cuts'][1]['scan_backend']=='outward_fallback'


def test_dot_node_bounds_include_exact_large_cancellation_and_conversion():
    D=(Q(1,3),Q(7,11));F=((Q(1,7),Q(-3,13)),(Q(1,7),Q(3,13)))
    rounded=dot._rounded_controls(D,F)
    values=np.array([[1e150,-1e150],[1e-200,1e-200],[.125,.25]])
    y=np.array([1.,0.,.75]);missing=np.array([[0.,0.],[1e-120,2e-120],[.01,.02]])
    operations=7;g=operations*dot.U/(1-operations*dot.U)
    low,high=dot.node_bounds(y,values,missing,rounded,dot._upper(g),dot._upper(operations*dot.TINY/(1-g)))
    for i in range(3):
        for j in range(2):
            true=Q(float(y[i]))*D[j]-sum(Q(float(values[i,a]))*F[a][j] for a in range(2))
            tail=sum(Q(float(missing[i,a]))*abs(F[a][j]) for a in range(2))
            assert Q(float(low[i,j]))<=true-tail and Q(float(high[i,j]))>=true+tail


def test_negative_D_and_positive_D_subdivision_keep_original_semantics():
    negative=compile_domain([[2]],[[0]],[0],[0],[.5],((.25,.5),(1,2)),[0])
    args=(negative,np.array([.5,.1]),np.arange(2),np.array([[2.,0.],[0.,0.]]),np.ones((2,2),bool),[0,0])
    assert scan.certify_fast(*args,[0],kappa=2,base_residual=0,atom_residuals=[0,0])['status']=='UNKNOWN'
    H=np.array([[0.,-2.],[2.,0.]])
    domain=compile_domain(H,np.zeros((2,2)),[0,0],[0,0],[0,0],((.1,3),(1,1.1)),[0,1])
    y=np.array([0.,0.,.8,.2]);R=np.vstack((np.column_stack((H,np.zeros((2,2)))),np.zeros((2,4))))
    args=(domain,y,np.arange(4),R,np.ones_like(R,bool),np.zeros(4));bounds=dict(kappa=2,base_residual=0,atom_residuals=np.zeros(4))
    result=scan.certify_fast(*args,[2],subdivide_depth=2,**bounds)
    assert result['status']=='STABLE'
    verify_leaves(*args,[2],bounds,result)


def test_real_omitted_tail_counterexample_cannot_become_false_stable():
    P=np.array([[.5,0,0],[.25,1,0],[.25,0,1.]])
    M=np.zeros_like(P);M[2,0]=.25;W=P-M-np.diag(P.diagonal());seed=np.array([.3,.4,.3])
    solved=np.linalg.solve(np.eye(3)-.5*P,np.column_stack((.5*seed,-.5*(M[:,0]-.25*P[:,0]),-.5*(W[:,0]-.25*P[:,0]))))
    y,R=solved[:,0],solved[:,1:]
    domain=compile_domain(R[[0],:1],R[[0],1:],[.25],[.25],y[[0]],((1,4),(.25,1)),[0])
    args=(domain,y,np.array([0,1]),R[[0,1]],np.ones((2,2),bool),abs(R[2]))
    bounds=dict(kappa=2,base_residual=1e-12,atom_residuals=[1e-12]*2)
    result=scan.certify_fast(*args,[1],**bounds)
    assert result['status']=='UNKNOWN'
    dropped=scan.certify_fast(*args[:-1],np.zeros(2),[1],**bounds)
    assert dropped['status']=='STABLE'  # Invalid caller contract, included to expose the missing protection.
    view=P.copy();raw=4*M[:,0]+.25*W[:,0];raw=raw.copy();raw[0]+=P[0,0];view[:,0]=raw/raw.sum()
    truth=np.linalg.solve(np.eye(3)-.5*view,.5*seed)
    assert truth[2]>truth[1]
