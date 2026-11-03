"""32-bit row identifiers for the native backend's signed-32-bit node domain.

Scores, residuals, coefficients and all numerical kernels remain float64.
Admission retains the parent's conservative upper bounds. This module does
not mutate the frozen compiler or any other process's constructor binding.
"""
import numpy as np
from .compiled_policy_cache import CompiledStencilDirectory


class PackedStencilDirectory(CompiledStencilDirectory):
    def __init__(self,ranker,G,stencil,keep,budget_bytes,**kwargs):
        if ranker.n>np.iinfo(np.int32).max:
            raise ValueError('Packed row IDs require the native 32-bit node domain.')
        packed={int(s):np.asarray(rows,dtype=np.int32) for s,rows in stencil.items()}
        super().__init__(ranker,G,packed,keep,budget_bytes,**kwargs)

    def atomic(self,source,delta):
        previous_peak=self.peak_bytes
        previous_live_peak=self.peak_live_response_bytes
        atom=super().atomic(source,delta)
        if atom.rows.dtype==np.int32:
            return atom
        before=atom.nbytes
        atom.rows=atom.rows.astype(np.int32,copy=False)
        self._payload_bytes+=atom.nbytes-before
        # The parent's insertion hook sees the temporary int64 construction
        # array. Retained-state peaks describe the packed cache; conversion
        # workspace is still included in the separately measured process RSS.
        self.peak_bytes=previous_peak
        self.peak_live_response_bytes=previous_live_peak
        self._track()
        return atom
