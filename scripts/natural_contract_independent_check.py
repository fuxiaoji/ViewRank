"""Independent exact-Fraction tests of the natural adapter's residual contract.

Only tiny synthetic graphs are used here. Fractions independently evaluate
canonical base residuals, full old-target residuals and actual solutions.
This file does not implement an adapter or change the frozen two-query scope.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits


def f(value):
    return value if isinstance(value,F) else F.from_float(float(value))


def matrix(P):
    return [[f(x) for x in row] for row in P.toarray()]


def matvec(A,x):
    return [sum((v*w for v,w in zip(row,x)),F(0)) for row in A]


def norm(x):return sum((abs(v) for v in x),F(0))


def solve(A,b):
    n=len(b);rows=[list(row)+[value] for row,value in zip(A,b)]
    for i in range(n):
        j=next(j for j in range(i,n) if rows[j][i])
        rows[i],rows[j]=rows[j],rows[i];scale=rows[i][i]
        rows[i]=[v/scale for v in rows[i]]
        for j in range(n):
            if j==i:continue
            scale=rows[j][i];rows[j]=[v-scale*w for v,w in zip(rows[j],rows[i])]
    return [row[-1] for row in rows]


def normalize(P):
    n=len(P);c=[sum((P[i][j] for i in range(n)),F(0)) for j in range(n)]
    return [[P[i][j]/c[j] for j in range(n)] for i in range(n)],c


def residual(A,v,rhs,alpha):
    Av=matvec(A,v)
    return norm([b-x+alpha*t for b,x,t in zip(rhs,v,Av)])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_adapter(path):
    name='independent_test_natural_adapter'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def residual_tests(adapter):
    rng=np.random.default_rng(2026110301);cases=[]
    for graph in range(40):
        n=2+graph%6
        raw=rng.uniform(.05,1,(n,n));raw/=raw.sum(0)
        raw*=1+rng.uniform(-1e-7,1e-7,n)
        P=sparse.csc_matrix(raw,dtype=np.float64);exactP=matrix(P);canonical,c=normalize(exactP)
        guard=adapter.SparseResidualGuard(P,.5);contract=guard.column
        assert all(contract['c_lo']<=value<=contract['c_hi'] for value in c)
        assert contract['delta']>=max(abs(value-1) for value in c)
        for exponent in (-900,0,900):
            v=rng.uniform(-1,1,n)*np.ldexp(1.,exponent)
            rhs=rng.uniform(-1,1,n)
            # Exact RHS differs slightly from the stored float RHS; its full
            # conversion error is known independently, rather than assumed 0.
            exact_rhs=[f(x)+F((-1)**i,3*2**60) for i,x in enumerate(rhs)]
            conversion=norm([x-f(y) for x,y in zip(exact_rhs,rhs)])
            upper,parts=guard.residual_upper(v,rhs,conversion)
            value=residual(canonical,list(map(f,v)),exact_rhs,F(1,2))
            assert upper>=value
            assert adapter.norm1_upper(v)>=norm(list(map(f,v)))
            old,old_parts=guard.residual_upper(v,rhs,conversion,old_background=True)
            assert old-upper==old_parts['old_raw_background_perturbation']
            assert old_parts['old_raw_background_perturbation']>=F(1,2)*max(abs(x-1) for x in c)*norm(list(map(f,v)))
            cases.append(dict(graph=graph,n=n,exponent=exponent,exact_residual=str(value),upper=str(upper),
                exact_over_upper=float(value/upper) if upper else 0.))
    # Normal external operands whose products can become subnormal. Coverage
    # is checked exactly on the current arithmetic path, without assuming FTZ.
    P=sparse.csc_matrix(np.array([[.25,.75],[.75,.25]]));guard=adapter.SparseResidualGuard(P,.5)
    canonical,_=normalize(matrix(P));tiny=np.finfo(float).tiny
    for v in (np.array([tiny,-tiny]),np.array([tiny,2*tiny]),np.array([-tiny,-3*tiny])):
        rhs=np.zeros(2);upper,_=guard.residual_upper(v,rhs)
        value=residual(canonical,list(map(f,v)),[F(0),F(0)],F(1,2))
        assert upper>=value
        cases.append(dict(name='normal_operand_subnormal_intermediate',exact_residual=str(value),upper=str(upper),exact_over_upper=float(value/upper)))
    return dict(status='passed',cases=len(cases),maximum_exact_residual_over_guard=max(c['exact_over_upper'] for c in cases),records=cases)


def rejection_tests(adapter):
    I=sparse.identity(2,format='csc',dtype=np.float64);ones=np.ones(2);zero=np.zeros(2)
    smallest=np.nextafter(0.,1.)
    duplicate=sparse.csc_matrix((np.array([.5,.5,1.]),np.array([0,0,1]),np.array([0,2,3])),shape=(2,2))
    sourceP=sparse.csc_matrix([[.5,0.],[.5,1.]])
    duplicateM=sparse.csc_matrix((np.array([.1,.1]),np.array([1,1]),np.array([0,2,2])),shape=(2,2))
    actions=[('subnormal_P',lambda:adapter.SparseResidualGuard(sparse.csc_matrix([[smallest,0.],[0.,1.]]),.5)),
        ('negative_P',lambda:adapter.SparseResidualGuard(sparse.csc_matrix([[-1.,0.],[0.,1.]]),.5)),
        ('nonfinite_P',lambda:adapter.SparseResidualGuard(sparse.csc_matrix([[float('nan'),0.],[0.,1.]]),.5)),
        ('empty_column',lambda:adapter.SparseResidualGuard(sparse.csc_matrix([[0.,0.],[0.,1.]]),.5)),
        ('duplicate_P',lambda:adapter.SparseResidualGuard(duplicate,.5)),
        ('duplicate_M',lambda:adapter.source_columns_exact(sourceP,duplicateM,[0],F(1,2))),
        ('mismatched_M_shape',lambda:adapter.source_columns_exact(sourceP,sparse.csc_matrix((3,3)),[0],F(1,2))),
        ('subnormal_alpha',lambda:adapter.SparseResidualGuard(I,smallest)),
        ('noncontractive_alpha',lambda:adapter.SparseResidualGuard(I,1.)),
        ('subnormal_v',lambda:adapter.SparseResidualGuard(I,.5).residual_upper(np.array([smallest,1.]),zero)),
        ('subnormal_rhs',lambda:adapter.SparseResidualGuard(I,.5).residual_upper(ones,np.array([smallest,0.]))),
        ('nonfinite_v',lambda:adapter.SparseResidualGuard(I,.5).residual_upper(np.array([float('inf'),1.]),zero)),
        ('gamma_out_of_range',lambda:adapter.gamma(2**53)),
        ('L1_norm_overflow',lambda:adapter.norm1_upper(np.array([1e308,1e308]))),
        ('SpMV_overflow',lambda:adapter.SparseResidualGuard(sparse.csc_matrix([[1.,1.],[0.,0.]]),.5).residual_upper(np.array([1e308,1e308]),zero))]
    records=[]
    for name,action in actions:
        try:
            with np.errstate(over='ignore',invalid='ignore'):action()
        except (ValueError,OverflowError,FloatingPointError) as exc:
            records.append(dict(name=name,rejected=True,exception=type(exc).__name__))
        else:raise AssertionError('Unsafe input was accepted: '+name)
    return dict(status='passed',cases=len(records),records=records)


def old_target_tests(adapter):
    rng=np.random.default_rng(2026110302);records=[]
    points=((F(1,4),F(1,2)),(F(1,4),F(4)),(F(1),F(1)),(F(2),F(1,4)),(F(4),F(2)),(F(3,2),F(9,8)))
    for case in range(8):
        n=5;r=2;S=[0,1];alpha=F(1,2)
        raw=rng.uniform(.1,1,(n,n));raw/=raw.sum(0);raw*=1+rng.uniform(-1e-6,1e-6,n)
        P=sparse.csc_matrix(raw);Pq=matrix(P);Pprime,c=normalize(Pq)
        frac=rng.choice([0.,.25,.5,.75,1.],size=(n,n));np.fill_diagonal(frac,0.)
        G=sparse.csc_matrix(raw*frac);Gq=matrix(G)
        guard=adapter.SparseResidualGuard(P,.5)
        source=adapter.source_columns_exact(P,G,S,alpha)
        seed=rng.dirichlet(np.ones(n));b=[(1-alpha)*f(x) for x in seed]
        rhs_exact=[b];mu=[];nu=[]
        mention=[];weak=[]
        for si,s in enumerate(S):
            mg=[Gq[i][s]/c[s] for i in range(n)]
            wg=[(Pq[i][s]-Gq[i][s]-(Pq[i][s] if i==s else 0))/c[s] for i in range(n)]
            m=sum(mg,F(0));w=sum(wg,F(0));mu.append(m);nu.append(w);mention.append(mg);weak.append(wg)
            assert source[si]['raw_total']==c[s] and source[si]['masses']==[m,w]
        for masses,groups in ((mu,mention),(nu,weak)):
            for j,s in enumerate(S):rhs_exact.append([-alpha*(groups[j][i]-masses[j]*Pprime[i][s]) for i in range(n)])
        rhs_float=[np.array([float(x) for x in column]) for column in rhs_exact]
        conversion=[norm([x-f(y) for x,y in zip(column,stored)]) for column,stored in zip(rhs_exact,rhs_float)]
        for si in range(r):
            for group in range(2):
                index=1+group*r+si;item=source[si]['atom_rhs'][group]
                assert np.array_equal(item['rhs'],rhs_float[index])
                assert item['rhs_conversion_error']==conversion[index]
        Araw=np.eye(n)-.5*raw
        vectors=[np.linalg.solve(Araw,rhs) for rhs in rhs_float]
        rho=[guard.residual_upper(v,rhs,e,old_background=True)[0] for v,rhs,e in zip(vectors,rhs_float,conversion)]
        y=list(map(f,vectors[0]));Rm=[list(map(f,v)) for v in vectors[1:1+r]];Rw=[list(map(f,v)) for v in vectors[1+r:]]
        assert guard.kappa_old>=1/(1-alpha*max([F(1)]+c))
        for m,w in points:
            M=[[F(i==j)+(m-1)*(F(i==j)*mu[j]+Rm[j][S[i]])+(w-1)*(F(i==j)*nu[j]+Rw[j][S[i]]) for j in range(r)] for i in range(r)]
            t=solve(M,[y[s] for s in S]);Z=[1+(m-1)*mu[j]+(w-1)*nu[j] for j in range(r)]
            x=[y[i]-sum(((m-1)*Rm[j][i]+(w-1)*Rw[j][i])*t[j] for j in range(r)) for i in range(n)]
            assert all(x[S[j]]==Z[j]*t[j] for j in range(r))
            old=[row.copy() for row in Pq];canonical=[row.copy() for row in Pprime]
            for j,s in enumerate(S):
                weighted=[m*Gq[i][s]+w*(Pq[i][s]-Gq[i][s]-(Pq[i][s] if i==s else 0))+(Pq[i][s] if i==s else 0) for i in range(n)]
                mass=sum(weighted,F(0))
                for i in range(n):old[i][s]=canonical[i][s]=weighted[i]/mass
            beta=rho[0]+sum(abs((m-1)*t[j])*rho[1+j]+abs((w-1)*t[j])*rho[1+r+j] for j in range(r))
            actual=residual(old,x,b,alpha)
            direct=solve([[F(i==j)-alpha*old[i][j] for j in range(n)] for i in range(n)],b)
            error=norm([a-bb for a,bb in zip(x,direct)])
            assert actual<=beta and error<=guard.kappa_old*beta
            # Exact target difference is only the non-source background.
            difference=max(sum(abs(old[i][j]-canonical[i][j]) for i in range(n)) for j in range(n))
            assert difference<=guard.column['delta']
            records.append(dict(case=case,m=str(m),w=str(w),exact_old_residual=str(actual),beta_upper=str(beta),
                exact_l1_error=str(error),error_upper=str(guard.kappa_old*beta),
                exact_residual_over_guard=float(actual/beta),exact_error_over_bound=float(error/(guard.kappa_old*beta))))
    return dict(status='passed',graphs=8,actual_views=len(records),maximum_residual_over_guard=max(r['exact_residual_over_guard'] for r in records),
        maximum_error_over_bound=max(r['exact_error_over_bound'] for r in records),records=records,
        target_scope='Non-source stored raw P columns untouched; source raw mention/weak/self columns weighted and normalized in exact rational arithmetic. No claim about literal floating source-normalization bytes.')


def preparation_and_runtime_tests(adapter):
    P=sparse.csc_matrix((np.array([.2,.1,.7,.4,.6,1.]),
        np.array([2,0,1,2,0,1]),np.array([0,3,5,6])),shape=(3,3))
    before={key:getattr(P,key).tobytes() for key in ('data','indices','indptr')}
    def entries(A):
        return sorted((j,int(A.indices[i]),float(A.data[i]).hex())
            for j in range(A.shape[1]) for i in range(A.indptr[j],A.indptr[j+1]))
    expected=entries(P);sorted_P,audit=adapter.sort_only_csc_copy(P,'independent unsorted example')
    assert sorted_P.has_canonical_format and entries(sorted_P)==expected
    assert all(getattr(P,key).tobytes()==value for key,value in before.items())
    duplicate=sparse.csc_matrix((np.array([.5,.5,1.]),np.array([0,0,1]),np.array([0,2,3])),shape=(2,2))
    try:adapter.sort_only_csc_copy(duplicate,'independent duplicate example')
    except ValueError:pass
    else:raise AssertionError('Sort-only helper coalesced a duplicate')
    runtime=adapter.kernel_runtime_guard();assert all(runtime.values())
    kernel=sys.modules[adapter.compile_domain.__module__]
    x,y=1e-200,1e-120
    product=kernel.Interval.of(x)*kernel.Interval.of(y);truth=f(x)*f(y)
    assert f(product.lo)<=truth<=f(product.hi)
    return dict(status='passed',sort_only_coordinate_multiset_preserved=True,input_arrays_unchanged=True,
        duplicate_sort_rejected=True,sort_audit=audit,kernel_runtime_guard=runtime,
        subnormal_interval_product=dict(x=x,y=y,lower=product.lo,upper=product.hi,exact=str(truth),enclosed=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter',default=str(Path(__file__).with_name('natural_domain_adapter.py')))
    parser.add_argument('--output',default=str(Path(__file__).with_suffix('.json')))
    args=parser.parse_args();path=Path(args.output)
    if path.exists():raise ValueError('Use a new independent report filename')
    adapter_hash=sha(args.adapter);adapter=load_adapter(args.adapter)
    with threadpool_limits(1):
        report=dict(status='passed',residual_guard=residual_tests(adapter),input_rejections=rejection_tests(adapter),
            old_target_contract=old_target_tests(adapter),script_sha256=sha(__file__),adapter_sha256=sha(args.adapter),
            sort_and_runtime_guards=preparation_and_runtime_tests(adapter),
            scope='Independent exact-Fraction tiny graph coverage and guard rejection tests. Does not prove an arbitrary floating library implementation follows the stated IEEE model or certify unsampled natural domains.')
    assert adapter_hash==sha(args.adapter),'Adapter source changed during independent tests'
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('residual_guard','input_rejections','old_target_contract')}|
        {'residual_guard_cases':report['residual_guard']['cases'],'guard_max_ratio':report['residual_guard']['maximum_exact_residual_over_guard'],
         'rejections':report['input_rejections']['cases'],'old_actual_views':report['old_target_contract']['actual_views'],
         'old_residual_max_ratio':report['old_target_contract']['maximum_residual_over_guard']}))


if __name__=='__main__':main()
