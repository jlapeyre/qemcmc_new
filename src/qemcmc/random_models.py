from __future__ import annotations
import numpy as np
from typing import Optional, Union
from .energy_model import EnergyModel

__all__ = ["random_pairwise_model", "random_bitstring"]


def _as_rng(rng: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng()
    return np.random.default_rng(int(rng))


def random_bitstring(n: int, rng: Optional[Union[int, np.random.Generator]] = None) -> str:
    g = _as_rng(rng)
    bits = g.integers(0, 2, size=int(n), endpoint=False)
    return "".join("1" if b else "0" for b in bits)


def _sample_coeffs(k: int, dist: str, rng: np.random.Generator) -> np.ndarray:
    if k <= 0:
        return np.empty((0,), dtype=np.float64)
    if dist == "normal":
        return rng.normal(loc=0.0, scale=1.0, size=k).astype(np.float64)
    if dist == "uniform":
        return rng.uniform(low=-1.0, high=1.0, size=k).astype(np.float64)
    raise ValueError("dist must be 'normal' or 'uniform'")


def random_pairwise_model(
    n: int,
    *,
    p1: float = 1.0,
    p2: float = 0.2,
    target_alpha: float = 1.0,
    dist: str = "normal",
    rng: Optional[Union[int, np.random.Generator]] = None,
    prune_tol: float = 0.0,
    name: Optional[str] = None,
    sigma_h: float = 1.0,
    sigma_J: float = 1.0,
) -> EnergyModel:
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0.0 <= p1 <= 1.0) or not (0.0 <= p2 <= 1.0):
        raise ValueError("p1 and p2 must be in [0,1]")
    if target_alpha <= 0.0:
        raise ValueError("target_alpha must be positive")
    g = _as_rng(rng)
    m_pairs = n * (n - 1) // 2
    n1 = int(np.round(p1 * n))
    n2 = int(np.round(p2 * m_pairs))
    n1 = int(np.clip(n1, 0, n))
    n2 = int(np.clip(n2, 0, m_pairs))
    if n1 == 0 and n2 == 0:
        raise ValueError("at least one of p1 or p2 must be > 0 to define alpha")
    h = None
    if n1 > 0:
        h = np.zeros(n, dtype=np.float64)
        idx1 = g.choice(n, size=n1, replace=False)
        vals1 = _sample_coeffs(n1, dist, g) * float(sigma_h)
        h[idx1] = vals1
    J_idx = None
    J_val = None
    if n2 > 0:
        iu, ju = np.triu_indices(n, k=1)
        pick = g.choice(m_pairs, size=n2, replace=False)
        pairs = np.stack([iu[pick], ju[pick]], axis=1).astype(np.int32)
        vals2 = _sample_coeffs(n2, dist, g) * float(sigma_J)
        J_idx = pairs
        J_val = vals2.astype(np.float64, copy=False)
    s2 = 0.0
    if h is not None:
        s2 += float(np.dot(h, h))
    if J_val is not None and J_val.size:
        s2 += float(np.dot(J_val, J_val))
    if s2 <= 0.0:
        raise ValueError("all sampled couplings are zero; try different rng or params")
    alpha_cur = float(np.sqrt(n) / np.sqrt(s2))
    k = float(alpha_cur / float(target_alpha))
    if h is not None:
        h = (h * k).astype(np.float64, copy=False)
    if J_val is not None:
        J_val = (J_val * k).astype(np.float64, copy=False)
    model = EnergyModel(
        n_spins=n,
        h=h,
        J_idx=J_idx,
        J_val=J_val,
        prune_tol=float(prune_tol),
        name=name,
    )
    return model
