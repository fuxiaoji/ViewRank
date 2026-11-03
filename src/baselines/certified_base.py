"""Common adaptive base solve for every executor in the new experiment round.

No requested seed is normalized or changed. Actual residual metadata is read
immediately after its solve; nominal solver tolerance is never used as a bound.
This helper requires the iterative backend's writable tol and last_residuals.
"""
from time import perf_counter
import numpy as np

from src.policy_family_fast import top_indices


def certified_base(ranker,seed,ks,tolerances=(1e-6,1e-8,1e-10,1e-12)):
    """Return (y, actual_L1_residual, strict_flags, elapsed_seconds, levels).

    All requested cuts must separate to finish early. If the last level has
    unresolved cuts, their strict flags stay false (numerical boundary case).
    The actual base residual must accompany y into every view certificate.
    Original backend tolerance is restored even if a solve raises.
    """
    started=perf_counter();backend=ranker.base_lu;original_tol=backend.tol
    seed=np.asarray(seed,dtype=float);ks=tuple(map(int,ks));levels=[]
    try:
        for tolerance in tolerances:
            level_started=perf_counter();backend.tol=float(tolerance)
            y=backend.solve((1-ranker.alpha)*seed)
            residual=float(backend.last_residuals[0])
            padding=64*np.finfo(float).eps*(1+np.linalg.norm(y,1))/(1-ranker.alpha)
            bound=residual/(1-ranker.alpha)+padding
            ids=top_indices(y,min(max(ks)+1,len(y)))
            gaps={k:(None if k==len(y) else float(y[ids[k-1]]-y[ids[k]])) for k in ks}
            flags={k:(k==len(y) or gaps[k]>2*bound) for k in ks}
            levels.append(dict(tolerance=float(tolerance),actual_residual_l1=residual,
                score_error_bound=bound,gaps=gaps,strict_flags=flags.copy(),seconds=perf_counter()-level_started))
            if all(flags.values()):break
    finally:backend.tol=original_tol
    levels[-1]['numerical_boundary']=not all(flags.values())
    return y,residual,flags,perf_counter()-started,levels
