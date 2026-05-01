from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Iterator

class EnergyModel:
    """
    Sparse Ising model with up to 3-body interactions using COO-like storage.

    Conventions
    - Internal spins are s_i in {-1,+1}. Binary inputs map 0->-1, 1->+1.
    - Store only unique, strictly increasing index tuples:
        1-body: dense h (shape (n,))
        2-body: J_idx (m2,2) with rows [i,j] s.t. i<j, J_val (m2,)
        3-body: K_idx (m3,3) with rows [i,j,k] s.t. i<j<k, K_val (m3,)
    - No constant (0-body) term is stored.

    Energy
        E(s) = sum_i h_i s_i
             + sum_t J_val[t] * Π_{p in J_idx[t]} s_p
             + sum_u K_val[u] * Π_{p in K_idx[u]} s_p

    Alpha scaling
        alpha = sqrt(n) / sqrt( sum(h^2) + sum(J_val^2) + sum(K_val^2) )

    Notes
    - Arrays are copied, coerced (h: float64; idx: int32; val: float64), and set read-only.
    - Indices validated to be in [0, n-1] and strictly increasing per row.
    - prune_tol drops |coeff| <= prune_tol from J and K (and h if provided).
    """

    def __init__(
        self,
        n_spins: int,
        *,
        h: Optional[np.ndarray] = None,
        J_idx: Optional[np.ndarray] = None,
        J_val: Optional[np.ndarray] = None,
        K_idx: Optional[np.ndarray] = None,
        K_val: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        atol: float = 1e-12,
        prune_tol: float = 0.0,
    ):
        self.n_spins = int(n_spins)
        self.name = name
        self.atol = float(atol)

        self.h = self._prep_h(h, prune_tol)
        self.J_idx, self.J_val = self._prep_terms(J_idx, J_val, order=2, prune_tol=prune_tol)
        self.K_idx, self.K_val = self._prep_terms(K_idx, K_val, order=3, prune_tol=prune_tol)

        self._alpha: Optional[float] = None

    # --------- Public API ---------

    @property
    def alpha(self) -> float:
        if self._alpha is None:
            s2 = 0.0
            if self.h is not None:
                s2 += float(np.dot(self.h, self.h))
            if self.J_val is not None and len(self.J_val):
                s2 += float(np.dot(self.J_val, self.J_val))
            if self.K_val is not None and len(self.K_val):
                s2 += float(np.dot(self.K_val, self.K_val))
            if s2 <= 0.0:
                raise ValueError("alpha undefined: no nonzero couplings")
            self._alpha = float(np.sqrt(self.n_spins / s2))
        return self._alpha

    def energy(self, state, *, spin_type: str = "binary") -> float:
        s = self._state_to_spin(state, spin_type=spin_type)

        e = 0.0
        if self.h is not None:
            e += float(np.dot(self.h, s))

        if self.J_val is not None and len(self.J_val):
            sj = s[self._row_take(self.J_idx, s.shape[0])]  # shape (m2,2)
            e += float(np.dot(self.J_val, sj.prod(axis=1, dtype=float)))

        if self.K_val is not None and len(self.K_val):
            sk = s[self._row_take(self.K_idx, s.shape[0])]  # shape (m3,3)
            e += float(np.dot(self.K_val, sk.prod(axis=1, dtype=float)))

        return e

    def iter_terms(self) -> Iterator[Tuple[int, Tuple[int, ...], float]]:
        if self.h is not None:
            for i, c in enumerate(self.h):
                if c != 0.0:
                    yield 1, (i,), float(c)
        if self.J_val is not None and len(self.J_val):
            for (i, j), c in zip(self.J_idx, self.J_val):
                yield 2, (int(i), int(j)), float(c)
        if self.K_val is not None and len(self.K_val):
            for (i, j, k), c in zip(self.K_idx, self.K_val):
                yield 3, (int(i), int(j), int(k)), float(c)

    # --------- Helpers: input prep, validation, mapping ---------

    def _prep_h(self, h, prune_tol: float) -> Optional[np.ndarray]:
        if h is None:
            return None
        h = np.asarray(h, dtype=np.float64).reshape(-1)
        if h.shape[0] != self.n_spins:
            raise ValueError(f"h must have shape ({self.n_spins},), got {h.shape}")
        if prune_tol > 0.0:
            mask = np.abs(h) > prune_tol
            h = np.where(mask, h, 0.0)
            if not mask.any():
                return None
        h.setflags(write=False)
        return h

    def _prep_terms(self, idx, val, *, order: int, prune_tol: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if idx is None and val is None:
            return None, None
        if idx is None or val is None:
            raise ValueError(f"Both idx and val must be provided for order {order}")

        idx = np.asarray(idx, dtype=np.int32)
        val = np.asarray(val, dtype=np.float64).reshape(-1)

        if idx.ndim != 2 or idx.shape[1] != order:
            raise ValueError(f"idx for order {order} must have shape (m,{order}), got {idx.shape}")
        if idx.shape[0] != val.shape[0]:
            raise ValueError(f"idx and val length mismatch: {idx.shape[0]} vs {val.shape[0]}")

        if not np.all((0 <= idx) & (idx < self.n_spins)):
            raise ValueError("indices out of bounds")
        if not np.all(np.diff(idx, axis=1) > 0):
            raise ValueError("each index row must be strictly increasing (i<j<... )")

        if prune_tol > 0.0:
            m = np.abs(val) > prune_tol
            idx = idx[m]
            val = val[m]
            if val.size == 0:
                return None, None

        # Deduplicate by summing duplicates (rows must be hashable via structured view)
        if idx.size:
            key = np.ascontiguousarray(idx).view([("k", idx.dtype, (order,))]).reshape(-1)
            order_idx = np.argsort(key, kind="mergesort")
            idx = idx[order_idx]
            val = val[order_idx]
            uniq_mask = np.ones(len(val), dtype=bool)
            uniq_mask[1:] = np.any(idx[1:] != idx[:-1], axis=1)
            # cumulative sum within groups, then select last of each group
            grp_ids = np.cumsum(uniq_mask) - 1
            summed = np.zeros(int(grp_ids[-1]) + 1, dtype=np.float64)
            np.add.at(summed, grp_ids, val)
            last_pos = np.where(uniq_mask)[0]
            last_pos = np.append(last_pos[1:], len(val) - 1)
            idx = idx[last_pos]
            val = summed

            if prune_tol > 0.0:
                m2 = np.abs(val) > prune_tol
                idx = idx[m2]
                val = val[m2]
                if val.size == 0:
                    return None, None

        idx.setflags(write=False)
        val.setflags(write=False)
        return idx, val

    @staticmethod
    def _row_take(idx: np.ndarray, width: int) -> np.ndarray:
        # Convert (m,k) into flat indices into s, then reshape back to (m,k).
        # Here we rely on s being 1D and idx holding 0..n-1.
        return idx

    def _state_to_spin(self, state, *, spin_type: str) -> np.ndarray:
        if isinstance(state, str):
            a = np.fromiter((1 if c == "1" else 0 for c in state), dtype=np.int8, count=len(state))
            if len(a) != self.n_spins:
                raise ValueError(f"bitstring length {len(a)} != n_spins {self.n_spins}")
            s = 2 * a.astype(np.int8) - 1
            return s.astype(np.float64, copy=False)
        arr = np.asarray(state)
        if arr.shape[0] != self.n_spins:
            raise ValueError(f"state length {arr.shape[0]} != n_spins {self.n_spins}")
        if spin_type == "binary":
            s = 2 * arr.astype(np.int8, copy=False) - 1
            return s.astype(np.float64, copy=False)
        if spin_type == "spin":
            return arr.astype(np.float64, copy=False)
        raise ValueError("spin_type must be 'binary' or 'spin'")

#   Add later list:
# - Builders: from_dense(...), from_terms(...), to_dict/from_dict.
# - 4+ body support via general idx_k/val_k dict.
# - Optional CSR/adjacency view for fast local ΔE updates.
# - Batch energy(states) with vectorized gather; optional numba.
# - QUBO constructor (from_qubo) and model fingerprint/hash.
