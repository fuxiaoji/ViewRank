"""Output-sensitive exact top-k for signed sums of cached response columns."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TopKResult:
    indices: np.ndarray
    scores: np.ndarray
    visited_nodes: int
    scanned_depth: int
    unseen_upper_bound: float


class SortedResponseIndex:
    """Sort each cached column once, then query arbitrary signed combinations.

    Coefficients may be positive, negative, or zero. Query ties use ascending
    node index. A strict frontier certificate ensures that unresolved ties
    trigger further scanning. No dense result vector is materialized.
    """

    def __init__(self, columns):
        self.columns = np.ascontiguousarray(columns, dtype=np.float64)
        if self.columns.ndim != 2:
            raise ValueError("columns must have shape (nodes, cached columns)")
        self.n, self.width = self.columns.shape
        self.ascending = np.argsort(self.columns, axis=0, kind="stable")
        self.descending = np.argsort(-self.columns, axis=0, kind="stable")
        self.column_magnitudes = np.max(np.abs(self.columns), axis=0) if self.n else np.zeros(self.width)

    @classmethod
    def concatenate(cls, *parts):
        """Join separately prepared seed/template indexes without sorting again."""
        combined = cls.__new__(cls)
        combined.n = parts[0].n
        if any(p.n != combined.n for p in parts):
            raise ValueError("index parts must describe the same node universe")
        combined.columns = np.ascontiguousarray(np.column_stack([p.columns for p in parts]))
        combined.ascending = np.column_stack([p.ascending for p in parts])
        combined.descending = np.column_stack([p.descending for p in parts])
        combined.column_magnitudes = np.concatenate([p.column_magnitudes for p in parts])
        combined.width = combined.columns.shape[1]
        return combined

    def topk(self, coeff, k):
        coeff = np.asarray(coeff, dtype=np.float64)
        if coeff.shape != (self.width,):
            raise ValueError("coeff must contain one coefficient per cached column")
        if not 1 <= k <= self.n:
            raise ValueError("k must lie between one and the number of nodes")
        active = np.flatnonzero(coeff)
        if not active.size:
            return TopKResult(np.arange(k), np.zeros(k), 0, 0, 0.0)
        # Outward padding covers roundoff in weighted sums near the frontier.
        rounding = np.finfo(float).eps * (self.width + 2) * float(np.abs(coeff) @ self.column_magnitudes)
        # Select column views, rather than copying an n-by-width order matrix.
        orders = [self.descending[:, j] if coeff[j] > 0 else self.ascending[:, j]
                  for j in active]
        seen = np.zeros(self.n, dtype=bool)
        candidates = []
        candidate_scores = []
        visited = 0
        depth = 0
        block = 1
        while depth < self.n:
            next_depth = min(self.n, depth + block)
            nodes = np.concatenate([order[depth:next_depth] for order in orders])
            fresh = np.unique(nodes[~seen[nodes]])
            if fresh.size:
                seen[fresh] = True
                scores = np.sum(self.columns[fresh] * coeff, axis=1)
                candidates.extend(fresh.tolist())
                candidate_scores.extend(scores.tolist())
                visited += len(fresh)
            depth = next_depth
            block *= 2
            if visited < k:
                continue
            # Keep only current top-k among visited nodes; discarded nodes cannot
            # re-enter since every subsequently seen node only tightens the cutoff.
            if len(candidates) > k:
                ids = np.asarray(candidates, dtype=np.int64)
                values = np.asarray(candidate_scores)
                best = np.lexsort((ids, -values))[:k]
                candidates = ids[best].tolist()
                candidate_scores = values[best].tolist()
            else:
                best = np.lexsort((np.asarray(candidates), -np.asarray(candidate_scores)))
                candidates = np.asarray(candidates)[best].tolist()
                candidate_scores = np.asarray(candidate_scores)[best].tolist()
            if visited == self.n:
                upper = float("-inf")
            else:
                next_frontier = self.columns[[order[depth] for order in orders], active]
                upper = float(np.sum(coeff[active] * next_frontier)) + rounding
            if candidate_scores[-1] > upper:
                return TopKResult(np.asarray(candidates, dtype=np.int64),
                                  np.asarray(candidate_scores), visited, depth, upper)
        raise RuntimeError("all nodes should have been evaluated")
