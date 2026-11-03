from fractions import Fraction

import numpy as np
import pytest

from src.policy_domain_polynomial import compile_domain, certify
from src.policy_domain_interval import certify_many


def dependency_model():
    P=np.array([[.5,0,0,0],[.25,1,0,0],[.25,0,1,0],[0,0,0,1.]])
    seed=np.array([500/797,0,0,297/797])
    A=np.eye(4)-.5*P
    y=np.linalg.solve(A,.5*seed)
    gm=np.array([0,.25,0,0]);gw=np.array([0,0,.25,0])
    R=np.column_stack([np.linalg.solve(A,-.5*(g-.25*P[:,0])) for g in (gm,gw)])
    domain=compile_domain(R[[0],:1],R[[0],1:],[.25],[.25],y[[0]],((1,2),(1,2)),[0])
    return P,seed,y,R,domain


def test_dependency_loss_is_sound_but_needs_more_spatial_work():
    P,seed,y,R,domain=dependency_model()
    common=dict(kappa=2,base_residual=1e-13,atom_residuals=[1e-13]*2)
    poly=certify(domain,y,list(range(4)),R,np.ones_like(R,dtype=bool),[0,0],[0],**common)
    coarse=certify_many(domain,y,list(range(4)),R,np.ones_like(R,dtype=bool),[0,0],{1:[0]},**common)
    fine=certify_many(domain,y,list(range(4)),R,np.ones_like(R,dtype=bool),[0,0],{1:[0]},subdivide_depth=8,**common)
    assert poly['status']=='STABLE'
    assert coarse['cuts'][1]['status']=='UNKNOWN'
    assert fine['cuts'][1]['status']=='STABLE'
    assert fine['attempted_boxes']>1
    for m in np.linspace(1,2,7):
        for w in np.linspace(1,2,7):
            view=P.copy();raw=np.array([.5,.25*m,.25*w,0]);view[:,0]=raw/raw.sum()
            scores=np.linalg.solve(np.eye(4)-.5*view,.5*seed)
            assert np.argmax(scores)==0


def test_borrowed_columns_chunking_and_multiple_cuts_match_owned_values():
    _,_,y,R,domain=dependency_model()
    opts=dict(kappa=2,base_residual=1e-13,atom_residuals=[1e-13]*2,subdivide_depth=4)
    a=certify_many(domain,y,list(range(4)),R,np.ones_like(R,dtype=bool),[0,0],{1:[0],2:[0,3]},chunk_rows=1,**opts)
    b=certify_many(domain,y,list(range(4)),None,None,[0,0],{1:[0],2:[0,3]},retained_columns=[R[:,0],R[:,1]],chunk_rows=4,**opts)
    assert {k:v['status'] for k,v in a['cuts'].items()}=={k:v['status'] for k,v in b['cuts'].items()}
    assert a['attempted_boxes']==b['attempted_boxes']


def test_omitted_tail_and_source_interactions_are_enforced():
    _,_,y,R,domain=dependency_model()
    rows=[0,3]
    tail=[np.max(abs(R[[1,2],j])) for j in range(2)]
    result=certify_many(domain,y,rows,R[rows],np.ones((2,2),dtype=bool),tail,{1:[0]},kappa=2,base_residual=1e-13,atom_residuals=[1e-13]*2)
    assert result['cuts'][1]['status']=='UNKNOWN'
    bad=R[rows].copy();bad[0,0]+=1
    with pytest.raises(ValueError):
        certify_many(domain,y,rows,bad,np.ones((2,2),dtype=bool),tail,{1:[0]},kappa=2,base_residual=0,atom_residuals=[0,0])
