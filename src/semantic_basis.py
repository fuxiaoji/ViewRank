"""Exact one-direction compilation of normalized semantic strength families."""
from dataclasses import replace
from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import lu_factor

from .composable import ComposableBoundaryDirectory, ComposedView


EVIDENCE_RELATIONS = {
    "supports", "witnessed_by", "grounded_in", "causes", "causal", "requires",
    "motive", "means", "opportunity",
}
GENERIC_RELATIONS = {"", "link", "edge", "network_edge"}


def strength_scale(strength, mass):
    """p(lambda)-p = a(lambda)*(p_G-m*p), for normalized group reweighting."""
    return (float(strength) - 1.0) / (1.0 + (float(strength) - 1.0) * float(mass))


class FixedGroupIndex:
    def __init__(self, record, P):
        self.P = P
        self.n = P.shape[0]
        self.rows = {}
        self.semantic = any(e.get("relation", "") not in GENERIC_RELATIONS for e in record["edges"])
        for edge in record["edges"]:
            self.rows.setdefault(int(edge["src"]), []).append(edge)
        self.kind = "fixed evidence-relation group" if self.semantic else "fixed controlled hashed half-neighbor group"

    def group(self, source):
        p = self.P[:, int(source)].toarray().ravel()
        group = np.zeros(self.n)
        if self.semantic:
            edges = self.rows.get(int(source), [])
            total = sum(float(e["weight"]) for e in edges)
            for edge in edges:
                if edge.get("relation", "") in EVIDENCE_RELATIONS:
                    group[int(edge["dst"])] += float(edge["weight"]) / total
        else:
            destinations = np.flatnonzero(p)
            hashes = [int.from_bytes(sha256(f"strength-group:{source}:{dst}".encode()).digest()[:8], "little")
                      for dst in destinations]
            selected = destinations[np.argsort(hashes, kind="stable")[:max(1, len(destinations) // 2)]]
            group[selected] = p[selected]
        mass = float(group.sum())
        direction = group - mass * p
        direction[np.abs(direction) < 1e-16] = 0
        return sparse.csc_matrix(direction.reshape(-1, 1)), mass, sparse.csc_matrix(group.reshape(-1, 1))


class SemanticStrengthDirectory(ComposableBoundaryDirectory):
    """Cache one atomic response per fixed group direction, not per lambda.

    Scaled sparse payloads are transient view objects. The resident directory
    contains only unscaled direction responses. No full n-vector is retained.
    """
    def compose_strength(self, sources, directions, masses, strength):
        started = perf_counter()
        sources = np.asarray(sources, dtype=np.int64)
        if len(set(sources.tolist())) != len(sources):
            raise ValueError("strength composition requires distinct sources")
        before_hits, before_misses = self.hits, self.misses
        before_solves, before_setup = self.atomic_solve_calls, self.atomic_prepare_seconds
        atoms = []
        for source, direction, mass in sorted(zip(sources, directions, masses), key=lambda item: item[0]):
            scale = strength_scale(strength, mass)
            if scale == 0 or not direction.nnz:
                continue
            basis = self.atomic(int(source), direction)
            fingerprint = sha256(bytes.fromhex(basis.fingerprint) + np.float64(scale).tobytes()).hexdigest()
            atoms.append(replace(
                basis, fingerprint=fingerprint, delta=basis.delta * scale,
                boundary=basis.boundary * scale, values=basis.values * scale,
                tail_inf=abs(scale) * basis.tail_inf,
                residual_l1=abs(scale) * basis.residual_l1,
                update_l1=abs(scale) * basis.update_l1,
                prepare_seconds=0.0, solve_calls=0,
            ))
        sources = np.asarray([atom.source for atom in atoms], dtype=np.int64)
        boundaries = np.column_stack([atom.boundary for atom in atoms]) if atoms else np.empty((len(self.B), 0))
        locations = np.asarray([self.positions[int(source)] for source in sources], dtype=np.int64)
        K = np.eye(len(sources)) + boundaries[locations]
        fingerprint = sha256("|".join(atom.fingerprint for atom in atoms).encode("ascii")).hexdigest()
        new = fingerprint not in self.whole_views
        self.whole_views.add(fingerprint)
        return ComposedView(
            atoms=tuple(atoms), sources=sources, K=K, small_lu=lu_factor(K) if atoms else None,
            fingerprint=fingerprint, new_whole_view=new,
            atom_hits=self.hits - before_hits, atom_misses=self.misses - before_misses,
            atomic_solve_calls=self.atomic_solve_calls - before_solves,
            atomic_prepare_seconds=self.atomic_prepare_seconds - before_setup,
            compose_seconds=perf_counter() - started,
        )
