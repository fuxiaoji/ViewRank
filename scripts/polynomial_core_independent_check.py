"""Independent mathematical checks of the affine source core and degree-r gaps.

Small graphs use dense solves, including exact Fraction checks of two hand
examples. Two unchanged first-query public packages use explicit actual-view
cold PI and signed-RHS Neumann solves, never dense full-graph inverses. No
producer engine is imported. Synthetic examples are illustrations, not
performance workloads or evidence of practical frequency.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
from itertools import permutations
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from build_multigroup_oracle import (GRID,actual_view,load_package,sha,solve_reference,write)


def add(a,b,scale=1.):
    result=a.copy()
    for k,v in b.items():result[k]=result.get(k,0.)+scale*v
    return {k:v for k,v in result.items() if v!=0}


def times(a,b):
    result={}
    for (i,j),x in a.items():
        for (k,l),y in b.items():result[i+k,j+l]=result.get((i+k,j+l),0.)+x*y
    return {k:v for k,v in result.items() if v!=0}


def evaluate(p,dm,dw):
    return sum(v*dm**i*dw**j for (i,j),v in p.items())


def determinant_polynomial(matrix):
    r=len(matrix)
    if r==0:return {(0,0):1.}
    result={}
    for perm in permutations(range(r)):
        inversions=sum(perm[i]>perm[j] for i in range(r) for j in range(i+1,r))
        term={(0,0):(-1.)**inversions}
        for i,j in enumerate(perm):term=times(term,matrix[i][j])
        result=add(result,term)
    return result


def polynomials(Cm,Cw,yS):
    r=len(yS)
    matrix=[[{(0,0):float(i==j),(1,0):float(Cm[i,j]),(0,1):float(Cw[i,j])} for j in range(r)] for i in range(r)]
    det=determinant_polynomial(matrix)
    adj_y=[]
    for i in range(r):
        value={}
        for j in range(r):
            # adj(M)[i,j] is the cofactor obtained by deleting row j, column i.
            minor=[[matrix[a][b] for b in range(r) if b!=i] for a in range(r) if a!=j]
            value=add(value,determinant_polynomial(minor),(-1.)**(i+j)*yS[j])
        adj_y.append(value)
    return det,adj_y


def gap_polynomial(model,i,j,det,adj_y):
    result={k:(model['y'][i]-model['y'][j])*v for k,v in det.items()}
    for s,p in enumerate(adj_y):
        linear={(1,0):model['Rm'][i,s]-model['Rm'][j,s],(0,1):model['Rw'][i,s]-model['Rw'][j,s]}
        result=add(result,times(linear,p),-1.)
    return result


def iterative_linear(P,rhs,alpha,tol=1e-15):
    x=np.zeros_like(rhs)
    for iteration in range(10000):
        new=rhs+alpha*(P@x)
        if np.max(np.sum(abs(new-x),axis=0))<=tol:
            x=new;break
        x=new
    else:raise RuntimeError('Signed Neumann solve did not converge')
    residual=np.sum(abs(x-alpha*(P@x)-rhs),axis=0)
    if np.max(residual)>tol*2:raise RuntimeError('Signed actual base-system residual too large')
    return x,residual


def prepare(P,G,seed,S,alpha,dense=True):
    n=P.shape[0];S=np.array(S,dtype=int)
    if sparse.issparse(P):
        W=(P-G-sparse.diags(P.diagonal(),format='csc')).tocsc()
        Ps=P[:,S].toarray();Ms=G[:,S].toarray();Ws=W[:,S].toarray()
    else:
        W=P-G-np.diag(P.diagonal());Ps=P[:,S];Ms=G[:,S];Ws=W[:,S]
    mu=Ms.sum(0);nu=Ws.sum(0);Um=Ms-Ps*mu;Uw=Ws-Ps*nu
    rhs=np.column_stack(((1-alpha)*seed,-alpha*Um,-alpha*Uw))
    if dense:
        response=np.linalg.solve(np.eye(n)-alpha*P,rhs)
        residual=np.sum(abs(response-alpha*(P@response)-rhs),axis=0)
    else:response,residual=iterative_linear(P,rhs,alpha)
    r=len(S);y=response[:,0];Rm=response[:,1:1+r];Rw=response[:,1+r:]
    Cm=np.diag(mu)+Rm[S];Cw=np.diag(nu)+Rw[S]
    return dict(P=P,G=G,W=W,seed=seed,S=S,alpha=alpha,n=n,r=r,mu=mu,nu=nu,
        Um=Um,Uw=Uw,y=y,Rm=Rm,Rw=Rw,Cm=Cm,Cw=Cw,base_atom_actual_residuals=residual)


def affine(model,m,w):
    M=np.eye(model['r'])+(m-1)*model['Cm']+(w-1)*model['Cw']
    t=np.linalg.solve(M,model['y'][model['S']])
    x=model['y']-((m-1)*model['Rm']+(w-1)*model['Rw'])@t
    return x,M,t


def dense_view(model,m,w):
    P=model['P'];view=P.copy()
    for s in model['S']:
        raw=m*model['G'][:,s]+w*model['W'][:,s]
        raw=raw.copy();raw[s]+=P[s,s]
        view[:,s]=raw/raw.sum()
    return view


def check_points(model,points,dense=True):
    det_poly,adj_y=polynomials(model['Cm'],model['Cw'],model['y'][model['S']])
    # A fixed gap pair, independent of whether its sign is favorable.
    i,j=0,min(1,model['n']-1)
    gap=gap_polynomial(model,i,j,det_poly,adj_y)
    assert max((sum(k) for k in det_poly),default=0)<=model['r']
    assert max((sum(k) for k in gap),default=0)<=model['r']
    errors=dict(affine_score_max=0.,source_identity_max=0.,det_polynomial_max=0.,adjugate_identity_max=0.,gap_polynomial_max=0.,determinant_lemma_max=0.)
    determinants=[];minimum_true_adjugate_times_base=float('inf')
    for m,w in points:
        x,M,t=affine(model,m,w);det=float(np.linalg.det(M));determinants.append(det)
        if dense:
            view=dense_view(model,m,w)
            direct=np.linalg.solve(np.eye(model['n'])-model['alpha']*view,(1-model['alpha'])*model['seed'])
            ratio=float(np.linalg.det(np.eye(model['n'])-model['alpha']*view)/np.linalg.det(np.eye(model['n'])-model['alpha']*model['P']))
            Z=1+(m-1)*model['mu']+(w-1)*model['nu']
            errors['determinant_lemma_max']=max(errors['determinant_lemma_max'],abs(det-np.prod(Z)*ratio))
        else:
            view=actual_view(model['P'],model['G'],model['S'],(m,w))
            direct,_,_=solve_reference(view,model['seed'],model['alpha'],5e-15)
            Z=1+(m-1)*model['mu']+(w-1)*model['nu']
        errors['affine_score_max']=max(errors['affine_score_max'],float(np.max(abs(x-direct))))
        errors['source_identity_max']=max(errors['source_identity_max'],float(np.max(abs(x[model['S']]-Z*t),initial=0)))
        errors['det_polynomial_max']=max(errors['det_polynomial_max'],abs(evaluate(det_poly,m-1,w-1)-det))
        adj_eval=np.array([evaluate(p,m-1,w-1) for p in adj_y])
        minimum_true_adjugate_times_base=min(minimum_true_adjugate_times_base,float(np.min(adj_eval,initial=float('inf'))))
        assert np.min(adj_eval,initial=0)>=-1e-12
        errors['adjugate_identity_max']=max(errors['adjugate_identity_max'],float(np.max(abs(adj_eval-det*t),initial=0)))
        errors['gap_polynomial_max']=max(errors['gap_polynomial_max'],abs(evaluate(gap,m-1,w-1)-(direct[i]-direct[j])*det))
        assert det>0
    assert errors['affine_score_max']<1e-11 and errors['gap_polynomial_max']<1e-9
    if not dense:errors['determinant_lemma_max']=None
    return dict(points=len(points),r=model['r'],determinant_min=min(determinants),errors=errors,
        determinant_lemma_direct_full_determinants_checked=bool(dense),
        minimum_true_adjugate_times_base_on_test_points=minimum_true_adjugate_times_base,
        det_degree=max((sum(k) for k in det_poly),default=0),gap_degree=max((sum(k) for k in gap),default=0),
        coefficients_scope='Floating determinant/cofactor expansion checks the algebra numerically; it is not a certified polynomial positivity algorithm.')


def fraction_solve(A,b):
    aug=[list(row)+[v] for row,v in zip(A,b)];n=len(b)
    for i in range(n):
        pivot=next(j for j in range(i,n) if aug[j][i])
        aug[i],aug[pivot]=aug[pivot],aug[i]
        scale=aug[i][i];aug[i]=[x/scale for x in aug[i]]
        for j in range(n):
            if j!=i:
                scale=aug[j][i];aug[j]=[x-scale*y for x,y in zip(aug[j],aug[i])]
    return [row[-1] for row in aug]


def corner_counterexample():
    P=np.zeros((5,5));G=np.zeros_like(P)
    P[1,0]=P[2,0]=P[2,1]=P[3,1]=.5
    G[1,0]=G[2,1]=.5
    for j in (2,3,4):P[j,j]=1
    seed=np.array([.95,0,0,0,.05]);model=prepare(P,G,seed,[0,1],.5)
    corners=[(F(1,4),F(99,100)),(F(1,4),F(101,100)),(F(4),F(99,100)),(F(4),F(101,100))]
    records=[]
    for m,w in corners+[(F(1),F(1))]:
        view=[[F(0) for _ in range(5)] for _ in range(5)]
        t=m/(m+w);view[1][0]=view[2][1]=t;view[2][0]=view[3][1]=1-t
        for j in (2,3,4):view[j][j]=1
        A=[[F(i==j)-F(1,2)*view[i][j] for j in range(5)] for i in range(5)]
        x=fraction_solve(A,[F(19,40),F(0),F(0),F(0),F(1,40)])
        exact=[F(19,40),F(19,80)*t,F(19,40)*(1-t)+F(19,80)*t*t,F(19,80)*t*(1-t),F(1,20)]
        assert x==exact
        top=sorted(range(5),key=lambda i:(-x[i],i))[:4]
        records.append(dict(m=str(m),w=str(w),scores=[str(v) for v in x],top4=top,top4_nodes=[['A','B','U','V','C'][i] for i in top]))
    assert all(set(r['top4'])=={0,1,2,4} for r in records[:4])
    assert set(records[4]['top4'])=={0,1,2,3}
    assert model['Cm'][1,0]<0
    det,adj=polynomials(model['Cm'],model['Cw'],model['y'][model['S']])
    gap=gap_polynomial(model,3,4,det,adj)
    check=check_points(model,[(float(m),float(w)) for m,w in corners]+[(1.,1.)])
    _,arbitraryM,_=affine(model,.25,4.)
    arbitrary_p=np.linalg.det(arbitraryM)*np.linalg.solve(arbitraryM,np.array([1.,0.]))
    assert arbitrary_p[1]<0
    return dict(status='passed',nodes=['A','B','U','V','C'],sources=['A','B'],alpha=.5,
        box=dict(m=[.25,4.],w=[.99,1.01]),records=records,check=check,
        cleared_gap_V_minus_C='-.0125*m^2+.034375*m*w-.0125*w^2',
        positive_denominator='det(M)=(m+w)^2/4',
        C_m=model['Cm'].tolist(),C_w=model['Cw'].tolist(),minimum_C_entry=float(min(model['Cm'].min(),model['Cw'].min())),
        arbitrary_nonphysical_source_base=dict(m=.25,w=4.,y_sources=[1.,0.],adjugate_times_y=arbitrary_p.tolist(),
            scope='Nonnegative source RHS alone does not imply adj(M)*y_sources >=0: these source values are not the true base vector for this graph/seed. The physical true vector is nonnegative.'),
        scope='Exact Fraction actual-view solves demonstrate corner-only top-k tests are unsound for continuous two-group boxes; synthetic mathematical counterexample, no practical-frequency claim.')


def dependency_example():
    P=np.array([[.5,0,0,0],[.25,1,0,0],[.25,0,1,0],[0,0,0,1.]])
    G=np.zeros_like(P);G[1,0]=.25;b=F(500,797)
    seed=np.array([float(b),0,0,float(F(297,500)*b)])
    model=prepare(P,G,seed,[0],.5)
    response=np.column_stack((model['Rm'][:,0],model['Rw'][:,0]))
    rank=int(np.linalg.matrix_rank(response));minor=F(1,12)*F(1,12)-F(1,12)*F(-1,6)
    assert rank==2 and minor==F(1,48)
    true_min=F(3,5)*b;competitor=F(297,500)*b;box_lower=F(10,17)*b
    assert box_lower<competitor<true_min
    return dict(status='passed',alpha=.5,sources=[0],base_seed=[str(b),'0','0',str(F(297,500)*b)],
        raw_group_weights=dict(self_loop='.5',mention='.25',weak='.25'),
        source_response_Rm_Rw=response.tolist(),source_response_rank=rank,exact_two_row_minor=str(minor),
        H_m='1/12',H_w='1/12',actual_box=dict(m=[1,2],w=[1,2]),
        independent_coefficient_box=dict(a_m=[0,.8],a_w=[0,.8]),
        impossible_independent_corner=[.8,.8],independent_box_lower_xA=str(box_lower),
        true_minimum_xA=str(true_min),constant_xC=str(competitor),
        exact_cleared_gap_A_minus_C_over_b='(109-47*((m-1)+(w-1)))/1500',minimum_cleared_gap_over_b='1/100',
        check=check_points(model,[(1.,1.),(1.,2.),(2.,1.),(2.,2.),(1.5,1.5)]),
        scope='Independent a intervals lose shared-denominator dependence. Although source interaction H values are equal, full response columns have rank two: one parameter-independent response per source is not generally lossless.')


def approximate_checks():
    rng=np.random.default_rng(9041);n=7;r=3;alpha=.5
    P=rng.uniform(.1,1,(n,n));P/=P.sum(0);frac=rng.uniform(0,1,(n,n));np.fill_diagonal(frac,0);G=P*frac
    seed=rng.dirichlet(np.ones(n));model=prepare(P,G,seed,list(range(r)),alpha)
    yt=model['y']+rng.normal(0,1e-5,n);Rm=model['Rm']+rng.normal(0,1e-4,(n,r));Rw=model['Rw']+rng.normal(0,1e-4,(n,r))
    Cm=np.diag(model['mu'])+Rm[model['S']];Cw=np.diag(model['nu'])+Rw[model['S']]
    A=np.eye(n)-alpha*P;rho0=float(np.linalg.norm(A@yt-(1-alpha)*seed,1))
    rhom=np.sum(abs(A@Rm+alpha*model['Um']),axis=0);rhow=np.sum(abs(A@Rw+alpha*model['Uw']),axis=0)
    checks=[]
    for m,w in GRID:
        M=np.eye(r)+(m-1)*Cm+(w-1)*Cw;t=np.linalg.solve(M,yt[model['S']]);h=yt[model['S']]-M@t
        Z=1+(m-1)*model['mu']+(w-1)*model['nu']
        x=yt-((m-1)*Rm+(w-1)*Rw)@t
        beta=rho0+abs(m-1)*float(abs(t)@rhom)+abs(w-1)*float(abs(t)@rhow)
        beta+=alpha*float((abs(h)/Z)@(abs(m-1)*abs(model['Um']).sum(0)+abs(w-1)*abs(model['Uw']).sum(0)))
        view=dense_view(model,m,w);actual_residual=float(np.linalg.norm((np.eye(n)-alpha*view)@x-(1-alpha)*seed,1))
        direct=np.linalg.solve(np.eye(n)-alpha*view,(1-alpha)*seed);error=float(np.max(abs(x-direct)))
        assert actual_residual<=beta+5e-14 and error<=beta/(1-alpha)+5e-14
        checks.append(dict(m=m,w=w,beta=beta,actual_l1_residual=actual_residual,actual_infinity_error=error,posterior_bound=beta/(1-alpha)))
    sign_examples=[];positive_det_negative_p=[]
    for m in (.25,1e-6):
        eps=m*m*1e-2;P=np.full((3,3),(1-eps)/2);np.fill_diagonal(P,eps)
        G=P-np.diag(P.diagonal());seed=np.array([1.,0,0]);smodel=prepare(P,G,seed,[0,1],.5)
        _,trueM,_=affine(smodel,m,1.)
        R=smodel['Rm'].copy();R[0,0]+=2*m/(1-m)
        approxM=np.eye(2)+(m-1)*(np.diag(smodel['mu'])+R[smodel['S']])
        rho=np.sum(abs((np.eye(3)-.5*P)@R+.5*smodel['Um']),axis=0)
        assert np.linalg.det(trueM)>0 and np.linalg.det(approxM)<0
        sign_examples.append(dict(m=m,w=1.,self_mass=eps,true_det=float(np.linalg.det(trueM)),approximate_det=float(np.linalg.det(approxM)),actual_atom_l1_residuals=rho.tolist(),nonzero_true_direction_norm=float(np.linalg.norm(smodel['Um'],1))))
        R2=smodel['Rm'].copy()
        for j,source in enumerate(smodel['S']):R2[source,j]+=2*m/(1-m)
        M2=np.eye(2)+(m-1)*(np.diag(smodel['mu'])+R2[smodel['S']])
        d2=float(np.linalg.det(M2));p2=d2*np.linalg.solve(M2,smodel['y'][smodel['S']])
        assert d2>0 and np.max(p2)<0
        positive_det_negative_p.append(dict(m=m,w=1.,approximate_det=d2,approximate_adjugate_times_exact_y=p2.tolist(),
            actual_atom_l1_residuals=np.sum(abs((np.eye(3)-.5*P)@R2+.5*smodel['Um']),axis=0).tolist()))
    return dict(status='passed',posterior_cases=checks,approximate_determinant_sign_counterexamples=sign_examples,
        positive_approximate_det_with_negative_adjugate_times_y=positive_det_negative_p,
        conclusion='Physical true-M positivity does not transfer to approximate H. Arbitrarily small absolute residual can still flip its determinant when positive policy weights approach zero; the fixed [.25,4] illustration requires a larger perturbation. A fixed practical box needs its own sign/nonzero bound, not this limiting argument.')


def random_checks():
    rng=np.random.default_rng(20261103);results=[]
    for r in (1,2,3,4):
        for trial in range(5):
            n=8;P=rng.uniform(.01,1,(n,n));P/=P.sum(0)
            frac=rng.uniform(0,1,(n,n));np.fill_diagonal(frac,0);G=P*frac
            seed=rng.dirichlet(np.ones(n));model=prepare(P,G,seed,list(range(r)),.5)
            results.append(dict(trial=trial,**check_points(model,list(GRID)+[(.25,.99),(4.,1.01)])))
    return results


def actual_checks(input_root):
    results=[]
    for dataset in ('hotpot_pool7405','musique_pool2417'):
        folder=Path(input_root)/dataset;meta,hashes,queries,P,G=load_package(folder);q=queries[0]
        seed=np.zeros(P.shape[0]);seed[q['seed_indices']]=q['seed_values']
        model=prepare(P,G,seed,q['sources'],float(meta['alpha']),dense=False)
        result=check_points(model,list(GRID),dense=False)
        result.update(dataset=dataset,qid=q['qid'],question_index=0,original_sources=q['sources'],nodes=P.shape[0],input_sha256=hashes,
            base_atom_max_actual_l1_residual=float(model['base_atom_actual_residuals'].max()),
            comparison='Full unchanged actual normalized sparse view cold PI versus independently prepared affine core; small graphs separately use dense direct solves. No induced subgraph or source truncation.')
        results.append(result)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',default='D:/desktop/ICDE 2027/results/natural_policy_workload')
    parser.add_argument('--output',default=str(Path(__file__).with_suffix('.json')))
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise ValueError('Use a new report filename; previous checks are preserved')
    with threadpool_limits(1):
        report=dict(status='passed',random_small_graphs=random_checks(),corner_counterexample=corner_counterexample(),
            dependency_and_two_response_rank_example=dependency_example(),approximate_variant=approximate_checks(),
            fixed_actual_first_queries=actual_checks(args.input_root),script_sha256=sha(__file__),
            independent_oracle_helper_sha256=sha(Path(__file__).with_name('build_multigroup_oracle.py')),
            scope='Independent algebra/numerics and synthetic counterexamples. No new producer benchmark, speed claim, interval positivity proof, or practical frequency claim.')
    write(output,report)
    print(json.dumps(dict(status=report['status'],output=str(output),actual=report['fixed_actual_first_queries'],
        approximate_sign_counterexamples=report['approximate_variant']['approximate_determinant_sign_counterexamples'])))


if __name__=='__main__':main()
