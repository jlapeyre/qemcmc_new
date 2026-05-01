from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Iterator, Sequence, Union, Any, Mapping, Dict
import os
import json

# TODO: some funcs use only self.n_spins. These could be ordinary functions


class EnergyModel:
    """Sparse Ising model with up to 3-body interactions using COO-like storage.

    Parameters
    - n_spins (int): number of spins/qubits (>= 1).
    - h (array-like | None, optional): 1-body fields, shape (n_spins,). Coerced to float64;
      if prune_tol > 0, entries with |h_i| <= prune_tol set to 0; if all zero, treated as None.
    - J_idx (array-like[int] | None, optional): shape (m2, 2); strictly increasing pairs i<j.
    - J_val (array-like[float] | None, optional): shape (m2,); aligned with J_idx rows.
    - K_idx (array-like[int] | None, optional): shape (m3, 3); strictly increasing triples i<j<k.
    - K_val (array-like[float] | None, optional): shape (m3,); aligned with K_idx rows.
    - name (str | None, optional): optional label for the model.
    - atol (float, default 1e-12): numerical tolerance for internal comparisons.
    - prune_tol (float, default 0.0): optional pruning; if > 0, drop coefficients with
      |coeff| <= prune_tol (h entries are zeroed, sparse terms removed). If all entries in a
      section are pruned, that section becomes None. If 0, pruning is disabled.

    Validation and normalization
    - Arrays are copied, coerced (idx: int32; val, h: float64), and set read-only.
    - Indices validated to be in [0, n_spins-1] and strictly increasing per row.
    - Duplicate rows in J_idx/K_idx are summed; zeros may be dropped by pruning.
    - Empty arrays of shape (0, 2) or (0, 3) are allowed and treated as "no terms."

    - Spins s_i ∈ {-1, +1}. Binary inputs map 0 → -1 and 1 → +1 in energy(...).
    - No constant (0-body) term is stored. Alpha scaling available via .alpha.

    Conventions
    - Internal spins are s_i in {-1,+1}. Binary inputs map 0->-1, 1->+1 in energy(...).
    - Store only unique, strictly increasing index tuples:
        1-body: dense h (shape (n,))
        2-body: J_idx (m2,2) with rows [i,j] s.t. i<j, J_val (m2,)
        3-body: K_idx (m3,3) with rows [i,j,k] s.t. i<j<k, K_val (m3,)
    - No constant (0-body) term is stored.

    Energy
        E(s) = sum_i h_i s_i
             + sum_t J_val[t] * Π_{p in J_idx[t]} s_p
             + sum_u K_val[u] * Π_{p in K_idx[u]} s_p

    Raises
    - ValueError: on shape mismatches, missing J/K idx/val pairs, out-of-bounds indices,
      or non-increasing rows.

    Examples
    >>> import numpy as np
    >>> m = EnergyModel(n_spins=3,
    ...                 h=np.array([1.0, 0.0, -2.0]),
    ...                 J_idx=np.array([[0, 2]]),
    ...                 J_val=np.array([0.5]),
    ...                 prune_tol=0.0)
    >>> m.energy("101")
    2.5
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
    def num_nonzero_h(self) -> int:
        """Number of nonzero 1-body fields h_i; 0 if h is None. Pruning may have zeroed small entries."""
        return 0 if self.h is None else int(np.count_nonzero(self.h))

    @property
    def num_J(self) -> int:
        """Number of stored 2-body couplings; equals J_idx row count; 0 if no J terms."""
        return 0 if self.J_idx is None else int(self.J_idx.shape[0])

    @property
    def num_K(self) -> int:
        """Number of stored 3-body couplings; equals K_idx row count; 0 if no K terms."""
        return 0 if self.K_idx is None else int(self.K_idx.shape[0])

    def equals(self, other: "EnergyModel", *, atol: float | None = None) -> bool:
        if not isinstance(other, EnergyModel) or self.n_spins != other.n_spins:
            return False
        tol = self.atol if atol is None else float(atol)

        def _eq(a, b, *, is_float: bool) -> bool:
            if a is None and b is None:
                return True
            if (a is None) != (b is None):
                return False
            return np.allclose(a, b, atol=tol) if is_float else np.array_equal(a, b)

        return (
            _eq(self.h, other.h, is_float=True)
            and _eq(self.J_idx, other.J_idx, is_float=False)
            and _eq(self.J_val, other.J_val, is_float=True)
            and _eq(self.K_idx, other.K_idx, is_float=False)
            and _eq(self.K_val, other.K_val, is_float=True)
        )

    def __eq__(self, other: object) -> bool:
        return self.equals(other)  # type: ignore[arg-type]

    @property
    def alpha(self) -> float:
        """Return the model’s alpha scaling factor.

        Alpha is defined as:
            alpha = sqrt(n_spins) / sqrt(sum(h^2) + sum(J_val^2) + sum(K_val^2))

        - Cached on first access.
        - Raises ValueError if all couplings are zero or absent (alpha undefined).

        Returns
        - float: the scaling factor.
        """
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
        """Compute the Ising energy E(s) for a given state.

        Conventions
        - Spins are s_i in {-1,+1}.
        - If state is a bitstring (str) or spin_type="binary", map 0->-1 and 1->+1.
        - If spin_type="spin", state must already be in {-1,+1}.

        Energy
            E(s) = sum_i h_i s_i
                 + sum_t J_val[t] * Π_{p in J_idx[t]} s_p
                 + sum_u K_val[u] * Π_{p in K_idx[u]} s_p

        Parameters
        - state: str bitstring or 1D array-like.
        - spin_type: "binary" or "spin" (default "binary").

        Returns
        - float: energy value.

        Examples
        >>> import numpy as np
        >>> from qemcmc import EnergyModel
        >>> m = EnergyModel(n_spins=4,
        ...                 h=np.array([1.0, 0.0, 0.0, -2.0]),
        ...                 J_idx=np.array([[0, 2], [1, 3]]),
        ...                 J_val=np.array([0.5, -1.0]))
        >>> m.energy("1010")
        2.5
        >>> m.energy([+1, -1, +1, -1], spin_type="spin")
        2.5
        """
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
        """Iterate over nonzero model terms as (order, indices, coeff).

        Yields
        - (order, indices, coeff) where:
          * order is 1, 2, or 3 for 1-, 2-, 3-body terms,
          * indices is a strictly increasing tuple of spin indices,
          * coeff is float.

        Notes
        - Omits zero coefficients after construction/pruning.
        - Order of yields is: all h terms, then J terms, then K terms.

        Example
        >>> list(EnergyModel(n_spins=3,
        ...                   h=np.array([1.0, 0.0, -2.0]),
        ...                   J_idx=np.array([[0, 2]]),
        ...                   J_val=np.array([0.5])).iter_terms())
        [(1, (0,), 1.0), (1, (2,), -2.0), (2, (0, 2), 0.5)]
        """
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
        # Set small numbers to zero
        if prune_tol > 0.0:
            mask = np.abs(h) > prune_tol
            h = np.where(mask, h, 0.0)
            if not mask.any():
                return None
        h.setflags(write=False)
        return h

    def _prep_terms(
        self,
        idx: Optional[np.ndarray],
        val: Optional[np.ndarray],
        *,
        order: int,
        prune_tol: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Normalize and validate sparse k-body (order) terms given in COO-like form.

        Parameters
        - idx: optional (m, order) int array of strictly increasing spin indices per row.
        - val: optional (m,) float array of coefficients aligned with idx rows.
        - order: interaction order (2 for J, 3 for K); used for shape checks and views.
        - prune_tol: drop entries with |coeff| <= prune_tol; may return (None, None) if empty.
        - prune_tol (float, default 0.0): if > 0, drop values with |coeff| <= prune_tol;
          if 0, pruning is disabled.

        Returns
        - (idx, val) with:
          * in-bounds, strictly increasing rows,
          * duplicates summed, stable-ordered,
          * optional pruning applied,
          * arrays set read-only
            or (None, None) if no terms remain after summing and pruning.
        """
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

        # Drop tiny coefficients and optionally short-circuit if nothing remains.
        if prune_tol > 0.0:
            m = np.abs(val) > prune_tol
            idx = idx[m]
            val = val[m]
            if val.size == 0:
                return None, None

        # Deduplicate by summing duplicates (rows must be hashable via structured view)
        # Deduplicate identical index rows by summing coefficients (stable to preserve order).
        if idx.size:
            key = np.ascontiguousarray(idx).view([("k", idx.dtype, (order,))]).reshape(-1)

            # Sort in increasing order of keys
            order_idx = np.argsort(key, kind="mergesort")
            idx = idx[order_idx]
            val = val[order_idx]

            # After sorting, mark group starts; accumulate vals within groups; take last occurrence.
            uniq_mask = np.ones(len(val), dtype=bool)
            uniq_mask[1:] = np.any(idx[1:] != idx[:-1], axis=1)
            # cumulative sum within groups, then select last of each group
            grp_ids = np.cumsum(uniq_mask) - 1
            summed = np.zeros(int(grp_ids[-1]) + 1, dtype=np.float64)
            np.add.at(summed, grp_ids, val)

            starts = np.where(uniq_mask)[0]
            ends = np.append(starts[1:], len(val)) - 1
            idx = idx[ends]
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

    def _state_to_spin(
        self,
        state: Union[str, Sequence[int], np.ndarray],
        *,
        spin_type: str,
    ) -> np.ndarray:
        """
        Convert an input state to Ising spins s in {-1, +1} with shape (n_spins,).

        Conventions
        - Internal spin representation is s_i in {-1, +1}.
        - For bitstrings (str) or spin_type="binary", the mapping is 0 -> -1 and 1 -> +1.
        - For spin_type="spin", the input is assumed to already be in {-1, +1}.

        Parameters
        - state: either
            * str: bitstring of length n_spins over {'0','1'}, or
            * 1D array-like length n_spins.
              If spin_type="binary", elements are expected in {0,1};
              if spin_type="spin", elements are expected in {-1,+1}.
        - spin_type: "binary" to interpret numeric inputs as 0/1 and map to spins,
                     or "spin" if the numeric input is already in {-1,+1}.

        Returns
        - np.ndarray: float64 array of shape (n_spins,) with entries in {-1.0, +1.0}.

        Raises
        - ValueError: if the length does not match n_spins or spin_type is invalid.
        """
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

    def to_qubo(self, *, form="upper"):
        """
        Convert this pairwise Ising EnergyModel into an equivalent QUBO model.

        Scope and mapping
        - Supports up to 2-body interactions only (no 3-body terms).
        - Uses the standard relation between binary x in {0,1}^n and spins s in {-1,+1}^n:
            s = 2x - 1  and  x = (s + 1)/2.
        - Returns (Q, offset) such that for any binary x,
            x^T Q x + offset  =  sum_i h_i s_i + sum_{i<j} J_ij s_i s_j
          where s = 2x - 1 is the corresponding spin vector.

        Parameters
        - form: "upper" or "full"
            * "upper": Q is upper-triangular with E(x) = Σ_i Q_ii x_i + Σ_{i<j} Q_ij x_i x_j.
            * "full":  Q is symmetric with E(x) = x^T Q x (off-diagonals counted twice).

        Returns
        - Q (ndarray float64, shape (n_spins, n_spins)): the QUBO matrix in the requested form.
        - offset (float): constant term so that energies match under the above mapping.

        Raises
        - ValueError: if the model contains 3-body terms.
        """
        if self.K_val is not None and len(self.K_val):
            raise ValueError("to_qubo supports only up to 2-body models")
        hv = self.h if self.h is not None else np.zeros(self.n_spins, dtype=np.float64)
        J_idx = self.J_idx if self.J_idx is not None else np.empty((0, 2), dtype=np.int32)
        J_val = self.J_val if self.J_val is not None else np.empty((0,), dtype=np.float64)
        return ising_to_qubo(self.n_spins, hv, J_idx, J_val, form=form)

    @classmethod
    def from_qubo(cls, Q, *, interpret="auto", atol=1e-12, prune_tol=0.0, name=None):
        """
        Build a pairwise Ising EnergyModel from a QUBO matrix Q.

        Scope and mapping
        - Supports pairwise models only. Any higher-order terms must be absent.
        - Uses the standard relation s = 2x - 1 between spins s in {-1,+1} and binary x in {0,1}.
        - Internally calls qubo_to_ising(...) and sets the resulting (h, J) on the model.

        Parameters
        - Q (array-like, shape (n,n)): QUBO matrix.
        - interpret: "auto" | "upper" | "full"
            * "upper": interpret E(x) = Σ_i Q_ii x_i + Σ_{i<j} Q_ij x_i x_j.
            * "full":  interpret E(x) = x^T Q x (symmetric Q, off-diagonals counted twice).
            * "auto":  use "full" if Q is symmetric within atol; otherwise use "upper".
        - atol (float): absolute tolerance to decide symmetry when interpret="auto".
        - prune_tol (float): pruning threshold applied to constructed Ising coefficients.
        - name (str | None): optional model name.

        Returns
        - EnergyModel: A model with n_spins = Q.shape[0], populated h and J (no K terms).
        """
        h, J_idx, J_val, _ = qubo_to_ising(Q, interpret=interpret, atol=atol)
        return cls(
            n_spins=Q.shape[0],
            h=h,
            J_idx=J_idx,
            J_val=J_val,
            name=name,
            prune_tol=prune_tol,
        )

    def save(
        self,
        path: Union[str, os.PathLike],
        *,
        offset: Optional[float] = None,
        meta: Optional[Mapping[str, Any]] = None,
        name: Optional[str] = None,
        prune_tol_meta: Optional[float] = None,
        compressed: bool = True,
    ) -> None:
        """
        Write this EnergyModel to an NPZ file.

        Stored schema (v1)
        - format_version: 'qemcmc-energy-v1' (Unicode scalar).
        - n_spins: int64 scalar.
        - name: Unicode scalar (model name; override via 'name' kw if desired).
        - atol: float64 scalar (from the instance).
        - prune_tol: float64 scalar (metadata; None→NaN unless provided via prune_tol_meta).
        - offset: float64 scalar (optional metadata; None→NaN).
        - meta_json: Unicode scalar; compact JSON for arbitrary metadata.
        - h: float64, shape (n,) or empty (0,).
        - J_idx: int32, shape (m2,2) or empty (0,2).
        - J_val: float64, shape (m2,) or empty (0,).
        - K_idx: int32, shape (m3,3) or empty (0,3).
        - K_val: float64, shape (m3,) or empty (0,).

        Notes
        - Uses Unicode scalars for strings to avoid object dtype and pickling.
        - Empty arrays represent “no terms”; consumers may translate empty↔None as needed.
        """
        h = _empty_or(self.h, (0,), np.float64)
        J_idx = _empty_or(self.J_idx, (0, 2), np.int32)
        J_val = _empty_or(self.J_val, (0,), np.float64)
        K_idx = _empty_or(self.K_idx, (0, 3), np.int32)
        K_val = _empty_or(self.K_val, (0,), np.float64)
        meta_str = json.dumps(meta or {}, separators=(",", ":"), sort_keys=True)
        arrs: Dict[str, np.ndarray] = {}
        arrs["format_version"] = _as_unicode_scalar("qemcmc-energy-v1")
        arrs["n_spins"] = np.array(int(self.n_spins), dtype=np.int64)
        arrs["name"] = _as_unicode_scalar(name if name is not None else getattr(self, "name", None))
        arrs["atol"] = np.array(float(getattr(self, "atol", 1e-12)), dtype=np.float64)
        arrs["prune_tol"] = _as_float_scalar(prune_tol_meta)
        arrs["offset"] = _as_float_scalar(offset)
        arrs["meta_json"] = _as_unicode_scalar(meta_str)
        arrs["h"] = np.asarray(h, dtype=np.float64)
        arrs["J_idx"] = np.asarray(J_idx, dtype=np.int32).reshape(-1, 2)
        arrs["J_val"] = np.asarray(J_val, dtype=np.float64).reshape(-1)
        arrs["K_idx"] = np.asarray(K_idx, dtype=np.int32).reshape(-1, 3)
        arrs["K_val"] = np.asarray(K_val, dtype=np.float64).reshape(-1)
        if compressed:
            np.savez_compressed(path, **arrs)
        else:
            np.savez(path, **arrs)

    @classmethod
    def load(
        cls,
        path: Union[str, os.PathLike],
    ) -> Tuple["EnergyModel", Optional[float], Dict[str, Any]]:
        """
        Read an EnergyModel from an NPZ file written by to_npz.

        Returns
        - (model, offset, meta) where:
          * model: reconstructed EnergyModel (no further pruning applied).
          * offset: float if present (not NaN), else None.
          * meta: dict parsed from meta_json (may include prune_tol and other info).

        Validation
        - Checks version tag if present.
        - Relies on EnergyModel’s constructor to validate indices, shapes, and bounds.
        """
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files)
            fmt = str(z["format_version"]) if "format_version" in keys else "qemcmc-energy-v1"
            if not fmt.startswith("qemcmc-energy-v"):
                raise ValueError(f"Unrecognized energy model format_version: {fmt}")
            n = int(z["n_spins"])
            name = str(z["name"]) if "name" in keys else None
            atol = float(z["atol"]) if "atol" in keys else 1e-12
            h = (
                np.asarray(z["h"], dtype=np.float64)
                if "h" in keys
                else np.empty((0,), dtype=np.float64)
            )
            J_idx = (
                np.asarray(z["J_idx"], dtype=np.int32)
                if "J_idx" in keys
                else np.empty((0, 2), dtype=np.int32)
            )
            J_val = (
                np.asarray(z["J_val"], dtype=np.float64)
                if "J_val" in keys
                else np.empty((0,), dtype=np.float64)
            )
            K_idx = (
                np.asarray(z["K_idx"], dtype=np.int32)
                if "K_idx" in keys
                else np.empty((0, 3), dtype=np.int32)
            )
            K_val = (
                np.asarray(z["K_val"], dtype=np.float64)
                if "K_val" in keys
                else np.empty((0,), dtype=np.float64)
            )
            off_raw = float(z["offset"]) if "offset" in keys else float("nan")
            meta_raw = str(z["meta_json"]) if "meta_json" in keys else "{}"
        h_arg = None if h.size == 0 else h
        J_idx_arg = J_idx.reshape(-1, 2)
        J_val_arg = J_val.reshape(-1)
        K_idx_arg = K_idx.reshape(-1, 3)
        K_val_arg = K_val.reshape(-1)
        meta: Dict[str, Any] = {}
        try:
            meta = json.loads(meta_raw)
        except Exception:
            meta = {}
        model = cls(
            n_spins=n,
            h=h_arg,
            J_idx=J_idx_arg if J_idx_arg.size else None,
            J_val=J_val_arg if J_val_arg.size else None,
            K_idx=K_idx_arg if K_idx_arg.size else None,
            K_val=K_val_arg if K_val_arg.size else None,
            name=name,
            atol=atol,
            prune_tol=0.0,
        )
        offset = None if np.isnan(off_raw) else float(off_raw)
        return model, offset, meta


def qubo_to_ising(Q, *, interpret="auto", atol=1e-12):
    """
    Convert a QUBO model to an equivalent pairwise Ising model.

    Scope and mapping
    - Binary variables: x in {0,1}^n.
    - Spin variables:   s in {-1,+1}^n with s = 2x - 1 and x = (s + 1)/2.
    - Produces coefficients (h, J) and a constant offset so that for all x,
        x^T Q x + offset  =  sum_i h_i s_i + sum_{i<j} J_ij s_i s_j
      where s = 2x - 1 is the corresponding spin vector.
    - Output is suitable for constructing a pairwise Ising EnergyModel.

    Parameters
    - Q (array-like, shape (n,n)): QUBO matrix.
    - interpret: "auto" | "upper" | "full"
        * "upper": interpret E(x) = Σ_i Q_ii x_i + Σ_{i<j} Q_ij x_i x_j
                   (Q need not be symmetric; only upper triangle is used).
        * "full":  interpret E(x) = x^T Q x (Q must be symmetric; off-diagonals
                   contribute twice).
        * "auto":  treat as "full" if Q is symmetric within atol; else "upper".
    - atol (float): absolute tolerance for symmetry checks when interpret="auto".

    Returns
    - h (ndarray float64, shape (n,)): linear Ising coefficients.
    - J_idx (ndarray int32, shape (m,2)): index pairs [i,j] with i < j for nonzero J_ij.
    - J_val (ndarray float64, shape (m,)): corresponding J_ij values.
    - offset (float): constant term making the QUBO and Ising energies equal under s=2x-1.

    Raises
    - ValueError: if Q is not square, or interpret is invalid.
    """
    A = np.asarray(Q, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Q must be a square matrix")
    #    n = A.shape[0]
    if interpret not in ("auto", "upper", "full"):
        raise ValueError("interpret must be 'auto', 'upper', or 'full'")
    mode = "full" if interpret == "auto" and np.allclose(A, A.T, atol=atol) else interpret
    mode = "upper" if mode == "auto" else mode
    d = np.diag(A).astype(np.float64, copy=True)
    b = np.triu(A + A.T, 1) if mode == "full" else np.triu(A, 1)
    sum_pairs = b.sum(axis=1) + b.sum(axis=0)
    h = 0.5 * d + 0.25 * sum_pairs
    ij = np.where(b != 0.0)
    J_idx = np.stack(ij, axis=1) if ij[0].size else np.empty((0, 2), dtype=np.int32)
    J_val = (
        (b[ij] * 0.25).astype(np.float64, copy=False)
        if ij[0].size
        else np.empty((0,), dtype=np.float64)
    )
    offset = -0.5 * d.sum() - 0.25 * b.sum()
    return h, J_idx.astype(np.int32, copy=False), J_val, float(offset)


def ising_to_qubo(n_spins, h=None, J_idx=None, J_val=None, *, form="upper"):
    """
    Convert Ising (pairwise only) to QUBO.
    Inputs
    - h (n,) or None, J given by (J_idx with i<j, J_val), no 3-body allowed.
    - form='upper' returns upper-triangular Q with E(x)=sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j.
      form='full' returns symmetric Q for E(x)=x^T Q x.
    Returns
    - Q, offset such that: sum_i h_i s_i + sum_{i<j} J_ij s_i s_j = x^T Q x + offset.
    """
    if form not in ("upper", "full"):
        raise ValueError("form must be 'upper' or 'full'")
    n = int(n_spins)
    hv = np.zeros(n, dtype=np.float64) if h is None else np.asarray(h, dtype=np.float64).reshape(-1)
    if hv.shape[0] != n:
        raise ValueError("h length must match n_spins")
    if J_idx is None or J_val is None:
        J_idx = np.empty((0, 2), dtype=np.int32)
        J_val = np.empty((0,), dtype=np.float64)
    J_idx = np.asarray(J_idx, dtype=np.int32).reshape(-1, 2)
    J_val = np.asarray(J_val, dtype=np.float64).reshape(-1)
    if J_idx.shape[0] != J_val.shape[0]:
        raise ValueError("J_idx and J_val length mismatch")
    if J_idx.size and not np.all(J_idx[:, 0] < J_idx[:, 1]):
        raise ValueError("Require i<j in J_idx rows")
    sumJ = np.zeros(n, dtype=np.float64)
    if J_val.size:
        np.add.at(sumJ, J_idx[:, 0], J_val)
        np.add.at(sumJ, J_idx[:, 1], J_val)
    d = 2.0 * (hv - sumJ)
    if form == "upper":
        Q = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(Q, d)
        Q[J_idx[:, 0], J_idx[:, 1]] = 4.0 * J_val
    else:
        Q = np.diag(d)
        Q[J_idx[:, 0], J_idx[:, 1]] = 2.0 * J_val
        Q[J_idx[:, 1], J_idx[:, 0]] = 2.0 * J_val
    offset = float(J_val.sum() - hv.sum())
    return Q, offset


def _as_unicode_scalar(s: Optional[str]) -> np.ndarray:
    """Return a NumPy 0-d Unicode array for safe NPZ storage (no object dtype)."""
    return np.array("" if s is None else s, dtype=np.str_)


def _as_float_scalar(x: Optional[float]) -> np.ndarray:
    """Return a NumPy 0-d float64 array for NPZ, using NaN for None."""
    return np.array(np.nan if x is None else float(x), dtype=np.float64)


def _empty_or(x: Optional[np.ndarray], shape: Tuple[int, ...], dtype) -> np.ndarray:
    """Return x if not None, else an empty array with requested shape/dtype."""
    return x if x is not None else np.empty(shape, dtype=dtype)


#   Add later list:
# - Builders: from_dense(...), from_terms(...), to_dict/from_dict.
# - 4+ body support via general idx_k/val_k dict.
# - Optional CSR/adjacency view for fast local ΔE updates.
# - Batch energy(states) with vectorized gather; optional numba.
# - model fingerprint/hash.
# - QUBO constructor (from_qubo) and
