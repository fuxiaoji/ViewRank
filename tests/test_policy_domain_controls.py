from fractions import Fraction as Q
import random

from src.policy_domain_controls import bernstein_fast,clear_geometry_cache,geometry_cache_info
from src.policy_domain_polynomial import bernstein


def test_integer_controls_equal_original_fraction_transform_for_all_degrees():
    rng=random.Random(923514)
    boxes=[((Q(1),Q(4)),(Q(1,4),Q(2))),((Q(-3,2),Q(1,3)),(Q(-1),Q(7,4)))]
    for degree in range(1,5):
        for box in boxes:
            for _ in range(12):
                poly={(i,j):Q(rng.randrange(-91,92),rng.randrange(1,74))
                    for i in range(degree+1) for j in range(degree+1-i)}
                assert bernstein_fast(poly,box,degree)==bernstein(poly,box,degree)


def test_geometry_cache_contains_only_box_degree_not_query_polynomials():
    clear_geometry_cache()
    box=((Q(1),Q(2)),(Q(1,4),Q(2)))
    for value in (Q(1),Q(173,19),Q(-5,7)):
        bernstein_fast({(0,0):value},box,3)
    info=geometry_cache_info()
    assert info.currsize==1 and info.misses==1 and info.hits==2
