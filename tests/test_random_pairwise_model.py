import numpy as np
import pytest
from qemcmc import random_pairwise_model
from qemcmc.random_models import random_bitstring


def test_random_pairwise_counts_and_alpha_repro():
    n = 10
    p1 = 0.4
    p2 = 0.2
    m = random_pairwise_model(n, p1=p1, p2=p2, target_alpha=0.75, rng=42)
    want_h = int(np.round(p1 * n))
    want_J = int(np.round(p2 * (n * (n - 1) // 2)))
    got_h = 0 if m.h is None else int((m.h != 0).sum())
    got_J = 0 if m.J_idx is None else int(m.J_idx.shape[0])
    assert got_h == want_h
    assert got_J == want_J
    assert m.alpha == pytest.approx(0.75, rel=1e-9, abs=1e-12)


def test_random_pairwise_energy_works():
    n = 8
    m = random_pairwise_model(n, p1=0.3, p2=0.25, target_alpha=1.0, rng=7)
    s = random_bitstring(n, rng=9)
    e = m.energy(s)
    assert np.isfinite(e)


def test_random_pairwise_reproducible_seed():
    a = random_pairwise_model(12, p1=0.5, p2=0.1, target_alpha=1.2, rng=2024)
    b = random_pairwise_model(12, p1=0.5, p2=0.1, target_alpha=1.2, rng=2024)
    assert a.n_spins == b.n_spins
    ha = a.h if a.h is not None else np.zeros(a.n_spins)
    hb = b.h if b.h is not None else np.zeros(b.n_spins)
    assert np.allclose(ha, hb)
    if a.J_idx is None and b.J_idx is None:
        assert True
    else:
        assert a.J_idx is not None and b.J_idx is not None
        assert a.J_idx.shape == b.J_idx.shape
        assert np.array_equal(a.J_idx, b.J_idx)
        assert np.allclose(a.J_val, b.J_val)
