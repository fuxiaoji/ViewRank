"""Exact Bernstein controls using integer dot products, with geometry-only reuse.

Box endpoints and polynomial coefficients are interpreted as exact stored numbers (Fractions,
integers, or binary64 floats). Production ExactDomain polynomials/boxes are
already Fractions. No sampled interpolation or query-dependent cache is used.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math


def _exact(value):
    if isinstance(value,Fraction):return value
    if isinstance(value,bool):raise TypeError('A Boolean is not a stored scalar.')
    if isinstance(value,int):return Fraction(value)
    value=float(value)
    if not math.isfinite(value):raise ValueError('Stored coefficients must be finite.')
    return Fraction.from_float(value)


@dataclass(frozen=True)
class Geometry:
    degree:int
    box:tuple
    # Full tensor monomial order i*(degree+1)+j; sparse polynomials select
    # their nonzero columns. Each control row has its own integer denominator.
    integer_rows:tuple
    row_denominators:tuple
    integer_payload_bytes:int


def _lcm_denominators(values):
    denominator=1
    for value in values:
        denominator=math.lcm(denominator,value.denominator)
    return denominator


@lru_cache(maxsize=64)
def _geometry(box,degree):
    (m0,m1),(w0,w1)=box
    axes=[]
    for low,high in ((m0,m1),(w0,w1)):
        width=high-low
        # For x^i after x=low+width*u, this is its degree-n Bernstein
        # coefficient at position k. The bivariate transform is a tensor
        # product of the two independent one-dimensional tables.
        table=[]
        for k in range(degree+1):
            table.append(tuple(sum((Fraction(math.comb(i,a)*math.comb(k,a),math.comb(degree,a))*
                low**(i-a)*width**a for a in range(min(i,k)+1)),Fraction(0))
                for i in range(degree+1)))
        axes.append(table)
    integer_rows=[];denominators=[];payload=0
    for p in range(degree+1):
        for q in range(degree+1):
            weights=tuple(axes[0][p][i]*axes[1][q][j]
                for i in range(degree+1) for j in range(degree+1))
            denominator=_lcm_denominators(weights)
            row=tuple(v.numerator*(denominator//v.denominator) for v in weights)
            integer_rows.append(row);denominators.append(denominator)
            payload+=(denominator.bit_length()+7)//8
            payload+=sum((abs(v).bit_length()+7)//8 for v in row)
    return Geometry(degree,box,tuple(integer_rows),tuple(denominators),payload)


def prepare_geometry(box,degree):
    """Prepare/retrieve only the immutable box/degree linear transform.

    Cache construction time is real preparation work and must be charged by
    an integrating experiment. The bounded cache stores no polynomial,
    source set, seed, vector, result or numerical solver progress.
    """
    if int(degree)!=degree or degree<0:raise ValueError('Nonnegative integer degree required.')
    canonical=tuple(tuple(_exact(v) for v in pair) for pair in box)
    if len(canonical)!=2 or any(len(pair)!=2 for pair in canonical):
        raise ValueError('Box has two endpoint pairs.')
    return _geometry(canonical,int(degree))


def bernstein_fast(poly,box,degree):
    """Exact control tuple, equal to the original Fraction Bernstein map."""
    geometry=prepare_geometry(box,degree);n=geometry.degree
    terms=[]
    for (i,j),value in poly.items():
        if i<0 or j<0 or i>n or j>n:
            raise ValueError('Polynomial exceeds requested tensor degree.')
        value=_exact(value)
        if value:terms.append((int(i)*(n+1)+int(j),value))
    if not terms:return (Fraction(0),)*((n+1)**2)
    common=_lcm_denominators(value for _,value in terms)
    integers=tuple((index,value.numerator*(common//value.denominator)) for index,value in terms)
    return tuple(Fraction(sum(coefficient*row[index] for index,coefficient in integers),
                          common*denominator)
                 for row,denominator in zip(geometry.integer_rows,geometry.row_denominators))


def controls_fast(domain,box=None):
    """Exact D/F controls; no domain-dependent result is retained."""
    box=domain.box if box is None else box
    return (bernstein_fast(domain.D,box,domain.r),
            tuple(bernstein_fast(f,box,domain.r) for f in domain.F))


def clear_geometry_cache():
    _geometry.cache_clear()


def geometry_cache_info():
    return _geometry.cache_info()
