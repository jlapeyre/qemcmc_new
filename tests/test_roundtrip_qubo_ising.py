import numpy as np
import pytest
from qemcmc import EnergyModel
from qemcmc import random_pairwise_model
from qemcmc.energy_model import qubo_to_ising, ising_to_qubo


def _pairs_to_dict(J_idx, J_val):
    if J_idx is None or J_val is None or J_val.size == 0:
        return {}
    d = {}
    for (i, j), v in zip(J_idx, J_val):
        d[(int(i), int(j))] = float(v)
    return d


def test_roundtrip_ising_to_qubo_to_ising():
    m = random_pairwise_model(9, p1=0.6, p2=0.25, target_alpha=1.0, rng=1234)
    Q, off1 = m.to_qubo(form="upper")
    m2 = EnergyModel.from_qubo(Q, interpret="upper")
    h1 = m.h if m.h is not None else np.zeros(m.n_spins)
    h2 = m2.h if m2.h is not None else np.zeros(m2.n_spins)
    assert np.allclose(h1, h2, atol=1e-12)
    d1 = _pairs_to_dict(m.J_idx, m.J_val)
    d2 = _pairs_to_dict(m2.J_idx, m2.J_val)
    assert set(d1.keys()) == set(d2.keys())
    for k in d1.keys():
        assert d1[k] == pytest.approx(d2[k], abs=1e-12)


def test_roundtrip_qubo_to_ising_to_qubo_upper():
    n = 7
    rng = np.random.default_rng(2025)
    Q = np.zeros((n, n), dtype=np.float64)
    diag = rng.uniform(-1.0, 1.0, size=n)
    np.fill_diagonal(Q, diag)
    m_pairs = n * (n - 1) // 2
    n2 = int(np.round(0.3 * m_pairs))
    iu, ju = np.triu_indices(n, k=1)
    pick = rng.choice(m_pairs, size=n2, replace=False)
    Q[iu[pick], ju[pick]] = rng.uniform(-2.0, 2.0, size=n2)
    h, J_idx, J_val, off1 = qubo_to_ising(Q, interpret="upper")
    Q2, off2 = ising_to_qubo(n, h, J_idx, J_val, form="upper")
    assert np.allclose(Q, Q2, atol=1e-12)
    assert off1 == pytest.approx(off2, abs=1e-12)
