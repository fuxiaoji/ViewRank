from fractions import Fraction as Q

import numpy as np
import pytest

from src.policy_domain_polynomial import compile_domain, certify
from src.policy_domain_fast import certify_fast, certify_many, scratch_upper_bound


def physical(n=11, r=2, seed=551):
    rng = np.random.default_rng(seed)
    P = rng.uniform(.1,1,(n,n)); P /= P.sum(axis=0)
    fraction = rng.uniform(0,1,(n,n)); np.fill_diagonal(fraction,0)
    G = P*fraction; W=P-G-np.diag(P.diagonal()); S=np.arange(r)
    seed_vector=np.full(n,.2/n);seed_vector[0]+=.8
    mu,nu=G[:,S].sum(0),W[:,S].sum(0)
    rhs=np.column_stack((.5*seed_vector,-.5*(G[:,S]-P[:,S]*mu),-.5*(W[:,S]-P[:,S]*nu)))
    solved=np.linalg.solve(np.eye(n)-.5*P,rhs)
    y,R=solved[:,0],solved[:,1:]
    domain=compile_domain(R[S,:r],R[S,r:],mu,nu,y[S],((.5,2.),(.5,2.)),S)
    order=np.lexsort((np.arange(n),-y))
    return domain,y,R,order,(P,G,W,S,seed_vector)


def arguments(domain,y,R):
    return (domain,y,np.arange(len(y)),R,np.ones_like(R,dtype=bool),np.zeros(R.shape[1]))


def bounds(domain):
    return dict(kappa=2.,base_residual=1e-12,atom_residuals=np.full(2*domain.r,1e-12))


def same_proof(left,right):
    for key in ('status','selected_ids','k','source_count','atom_count','tensor_degree','attempted_boxes','leaf_boxes','leaves'):
        assert left[key]==right[key]


@pytest.mark.parametrize('r',[1,2,3,4])
def test_matches_original_kernel_all_cutoffs_and_dense_direct_solutions(r):
    domain,y,R,order,data=physical(r=r)
    selections={k:order[:k] for k in (1,3,5)}
    fast=certify_many(*arguments(domain,y,R),selections,chunk_rows=3,**bounds(domain))
    for k,ids in selections.items():
        slow=certify(*arguments(domain,y,R),ids,**bounds(domain))
        same_proof(fast['cuts'][k],slow)
        if slow['status']=='STABLE':
            P,G,W,S,seed=data
            for m,w in ((.5,.5),(.5,2.),(2.,.5),(2.,2.),(.75,1.25),(1.5,.8)):
                view=P.copy()
                for source in S:
                    raw=m*G[:,source]+w*W[:,source]
                    raw=raw.copy();raw[source]+=P[source,source]
                    view[:,source]=raw/raw.sum()
                truth=np.linalg.solve(np.eye(len(P))-.5*view,.5*seed)
                assert set(ids)==set(np.argsort(-truth,kind='stable')[:k])
    assert fast['shared_scan']['control_boxes']==1
    assert fast['shared_scan']['candidate_node_controls']==len(y)*(r+1)**2


def test_dense_borrows_full_columns_and_matches_matrix_path():
    domain,y,R,order,_=physical(r=3)
    result=certify_many(domain,y,np.arange(len(y)),None,None,np.zeros(6),
        {1:order[:1],3:order[:3]},retained_columns=[R[:,j] for j in range(6)],chunk_rows=2,**bounds(domain))
    original=certify_many(*arguments(domain,y,R),{1:order[:1],3:order[:3]},chunk_rows=2,**bounds(domain))
    for k in result['cuts']:same_proof(result['cuts'][k],original['cuts'][k])
    np.testing.assert_array_equal(R,np.column_stack([R[:,j] for j in range(6)]))
    assert result['workspace_peak_bytes']==scratch_upper_bound(domain,len(y),2,2,len(y))


def test_sparse_masks_and_outside_nodes_match_original():
    domain,y,R,order,_=physical(n=19,r=3)
    retained=[set(np.argsort(-abs(R[:,j]))[:2])|set(domain.source_ids) for j in range(6)]
    candidates=sorted(set(order[:5])|set.union(*retained))
    mask=np.array([[node in rows for rows in retained] for node in candidates])
    tails=np.array([max((abs(R[i,j]) for i in range(len(y)) if i not in rows),default=0.) for j,rows in enumerate(retained)])
    args=(domain,y,np.array(candidates),R[candidates],mask,tails)
    for k in (1,3,5):
        same_proof(certify_fast(*args,order[:k],chunk_rows=4,**bounds(domain)),
                   certify(*args,order[:k],**bounds(domain)))


def absorbing_example():
    P=np.array([[.5,0,0],[.25,1,0],[.25,0,1.]])
    G=np.zeros_like(P);G[2,0]=.25
    W=P-G-np.diag(P.diagonal());seed=np.array([.3,.4,.3])
    rhs=np.column_stack((.5*seed,-.5*(G[:,0]-.25*P[:,0]),-.5*(W[:,0]-.25*P[:,0])))
    solved=np.linalg.solve(np.eye(3)-.5*P,rhs);y,R=solved[:,0],solved[:,1:]
    domain=compile_domain(R[[0],:1],R[[0],1:],[.25],[.25],y[[0]],((1,4),(.25,1)),[0])
    return domain,y,R,P,G,W,seed


def test_omitted_response_tail_prevents_false_stability():
    domain,y,R,P,G,W,seed=absorbing_example()
    candidates=np.array([0,1]);values=R[candidates];mask=np.ones_like(values,dtype=bool)
    tails=np.abs(R[2])
    good=certify_fast(domain,y,candidates,values,mask,tails,[1],**bounds(domain))
    dropped=certify_fast(domain,y,candidates,values,mask,np.zeros(2),[1],**bounds(domain))
    assert good['status']=='UNKNOWN'
    assert dropped['status']=='STABLE'  # Deliberately invalid omitted-coordinate contract.
    view=P.copy();raw=4*G[:,0]+.25*W[:,0];raw[0]+=P[0,0];view[:,0]=raw/raw.sum()
    truth=np.linalg.solve(np.eye(3)-.5*view,.5*seed)
    assert truth[2]>truth[1]
    same_proof(good,certify(domain,y,candidates,values,mask,tails,[1],**bounds(domain)))


def test_many_keeps_stable_cut_and_subdivides_only_unknown_cut():
    domain,y,R,*_=absorbing_example()
    selected={1:[1],2:[1,2]}
    result=certify_many(*arguments(domain,y,R),selected,subdivide_depth=1,chunk_rows=1,**bounds(domain))
    assert result['cuts'][1]['status']=='UNKNOWN'
    assert result['cuts'][2]['status']=='STABLE'
    assert result['cuts'][2]['attempted_boxes']==1
    assert result['cuts'][1]['attempted_boxes']==3
    for k,ids in selected.items():
        same_proof(result['cuts'][k],certify(*arguments(domain,y,R),ids,subdivide_depth=1,**bounds(domain)))
    assert result['shared_scan']['control_boxes']==3


def test_abs_bernstein_envelope_when_amplitudes_cross_zero():
    P=np.array([[.5,0,0,0],[.25,1,0,0],[.25,0,1,0],[0,0,0,1.]])
    G=np.zeros_like(P);G[1,0]=.25
    b=Q(500,797);seed=np.array([float(b),0,0,float(Q(297,500)*b)])
    W=P-G-np.diag(P.diagonal())
    solved=np.linalg.solve(np.eye(4)-.5*P,np.column_stack((.5*seed,-.5*(G[:,0]-.25*P[:,0]),-.5*(W[:,0]-.25*P[:,0]))))
    y,R=solved[:,0],solved[:,1:]
    domain=compile_domain(R[[0],:1],R[[0],1:],[.25],[.25],y[[0]],((.5,2),(.5,2)),[0])
    _,F=domain.controls();assert any(min(f)<0<max(f) for f in F)
    result=certify_fast(*arguments(domain,y,R),[0],**bounds(domain))
    assert result['status']=='STABLE'
    same_proof(result,certify(*arguments(domain,y,R),[0],**bounds(domain)))


def test_negative_denominator_is_unknown():
    domain=compile_domain([[2]],[[0]],[0],[0],[.5],((.25,.5),(1,2)),[0])
    y=np.array([.5,.1]);R=np.array([[2.,0.],[0.,0.]])
    result=certify_fast(*arguments(domain,y,R),[0],kappa=2,base_residual=0,atom_residuals=[0,0])
    assert result['status']=='UNKNOWN'
    assert 'D Bernstein' in result['leaves'][0]['reason']


def test_subdivision_recovers_positive_polynomial_with_negative_root_controls():
    # M=I+(m-1)*rotation has D=1+4*(m-1)^2>0; coarse controls can be negative.
    H=np.array([[0.,-2.],[2.,0.]])
    domain=compile_domain(H,np.zeros((2,2)),[0,0],[0,0],[0,0],((.1,3),(1,1.1)),[0,1])
    y=np.array([0.,0.,.8,.2]);R=np.vstack((np.column_stack((H,np.zeros((2,2)))),np.zeros((2,4))))
    kw=dict(kappa=2,base_residual=0,atom_residuals=np.zeros(4))
    assert certify_fast(*arguments(domain,y,R),[2],**kw)['status']=='UNKNOWN'
    fast=certify_fast(*arguments(domain,y,R),[2],subdivide_depth=2,chunk_rows=1,**kw)
    slow=certify(*arguments(domain,y,R),[2],subdivide_depth=2,**kw)
    assert fast['status']=='STABLE'
    same_proof(fast,slow)


def test_fraction_node_inputs_and_source_consistency():
    domain=compile_domain([[Q(0)]],[[Q(0)]],[0],[0],[Q(1,3)],((1,2),(1,2)),[0])
    y=[Q(1,3),Q(2,3)];R=[[Q(0),Q(0)],[Q(0),Q(0)]]
    fast=certify_fast(domain,y,[0,1],R,[[True,True]]*2,[0,0],[1],kappa=2,base_residual=0,atom_residuals=[0,0])
    slow=certify(domain,y,[0,1],R,[[True,True]]*2,[0,0],[1],kappa=2,base_residual=0,atom_residuals=[0,0])
    same_proof(fast,slow)
    with pytest.raises(ValueError,match='source base'):
        certify_fast(domain,[.4,.6],[0,1],R,[[True,True]]*2,[0,0],[1],kappa=2,base_residual=0,atom_residuals=[0,0])
