import numpy as np
import pytest
from qemcmc import EnergyModel


def test_energy_and_alpha_literal():
    m = EnergyModel(
        n_spins=4,
        h=np.array([1.0, 0.0, 0.0, -2.0]),
        J_idx=np.array([[0, 2], [1, 3]]),
        J_val=np.array([0.5, -1.0]),
    )
    assert m.energy("1010") == pytest.approx(2.5, abs=1e-12)
    assert m.alpha == pytest.approx(0.8, abs=1e-12)


def test_iter_terms_content():
    m = EnergyModel(
        n_spins=4,
        h=np.array([1.0, 0.0, 0.0, -2.0]),
        J_idx=np.array([[0, 2], [1, 3], [1, 2]]),
        J_val=np.array([0.5, -1.0, 0.7]),
    )
    terms = list(m.iter_terms())
    assert (1, (0,), 1.0) in terms
    assert (1, (3,), -2.0) in terms
    assert (2, (1, 3), -1.0) in terms
    assert (2, (0, 2), 0.5) in terms


def test_pruning_and_dedup():
    m = EnergyModel(
        n_spins=3,
        h=np.array([0.0, 0.0, 0.0]),
        J_idx=np.array([[0, 1], [0, 1], [1, 2]]),
        J_val=np.array([0.1, -0.1, 0.2]),
        prune_tol=1e-14,
    )
    terms = list(m.iter_terms())
    assert (2, (1, 2), 0.2) in terms
    assert not any(t for t in terms if t[0] == 2 and t[1] == (0, 1))


def test_energy_from_spin_array():
    h = np.array([1.0, -0.5, 2.0, 0.0])
    J_idx = np.array([[0, 2], [1, 3]])
    J_val = np.array([0.25, 0.75])
    m = EnergyModel(n_spins=4, h=h, J_idx=J_idx, J_val=J_val)
    s = np.array([-1, +1, -1, +1], dtype=float)
    e_manual = h @ s + 0.25 * (s[0] * s[2]) + 0.75 * (s[1] * s[3])
    assert m.energy(s, spin_type="spin") == pytest.approx(e_manual, abs=1e-12)


def test_omit_all_couplings_energy_zero_alpha_raises():
    m = EnergyModel(n_spins=3)
    assert m.energy("000") == 0.0
    with pytest.raises(ValueError):
        _ = m.alpha


def test_only_h_provided_energy_and_alpha():
    h = np.array([1.0, -2.0, 0.5])
    m = EnergyModel(n_spins=3, h=h)
    s = np.array([1, -1, 1], dtype=float)
    assert m.energy(s, spin_type="spin") == pytest.approx(h @ s, abs=1e-12)
    assert m.alpha == pytest.approx(np.sqrt(3) / np.sqrt((h * h).sum()), abs=1e-12)


def test_only_J_provided_energy_and_alpha():
    J_idx = np.array([[0, 2], [1, 2]], dtype=np.int32)
    J_val = np.array([0.5, -1.0], dtype=float)
    m = EnergyModel(n_spins=3, J_idx=J_idx, J_val=J_val)
    s = np.array([1.0, -1.0, 1.0])
    e = 0.5 * s[0] * s[2] + (-1.0) * s[1] * s[2]
    assert m.energy(s, spin_type="spin") == pytest.approx(e, abs=1e-12)
    assert m.alpha == pytest.approx(np.sqrt(3) / np.sqrt((J_val * J_val).sum()), abs=1e-12)


def test_only_K_provided_energy_and_alpha():
    K_idx = np.array([[0, 1, 2]], dtype=np.int32)
    K_val = np.array([0.7], dtype=float)
    m = EnergyModel(n_spins=3, K_idx=K_idx, K_val=K_val)
    s = np.array([1.0, -1.0, 1.0])
    e = 0.7 * s[0] * s[1] * s[2]
    assert m.energy(s, spin_type="spin") == pytest.approx(e, abs=1e-12)
    assert m.alpha == pytest.approx(np.sqrt(3) / np.sqrt((K_val * K_val).sum()), abs=1e-12)


def test_omit_h_is_ok_and_explicit_none_is_ok_with_J():
    J_idx = np.array([[0, 1]], dtype=np.int32)
    J_val = np.array([2.0], dtype=float)
    m = EnergyModel(n_spins=2, h=None, J_idx=J_idx, J_val=J_val)
    assert m.energy("11") == pytest.approx(2.0, abs=1e-12)


def test_omitting_all_J_fields_is_ok_when_h_present():
    h = np.array([1.0, -1.5, 0.1])
    m = EnergyModel(n_spins=3, h=h)
    energy = m.energy("110")
    expected = 1.0 * 1 + (-1.5) * 1 + 0.1 * -1.0
    assert (energy) == pytest.approx(expected, abs=1e-12)


def test_only_one_of_J_idx_or_J_val_raises():
    with pytest.raises(ValueError):
        _ = EnergyModel(n_spins=3, J_idx=np.array([[0, 1]]), J_val=None)
    with pytest.raises(ValueError):
        _ = EnergyModel(n_spins=3, J_idx=None, J_val=np.array([1.0]))


def test_only_one_of_K_idx_or_K_val_raises():
    with pytest.raises(ValueError):
        _ = EnergyModel(n_spins=3, K_idx=np.array([[0, 1, 2]]), K_val=None)
    with pytest.raises(ValueError):
        _ = EnergyModel(n_spins=3, K_idx=None, K_val=np.array([1.0]))


def test_empty_J_arrays_behave_like_none():
    J_idx = np.empty((0, 2), dtype=np.int32)
    J_val = np.empty((0,), dtype=float)
    h = np.array([0.5, -0.5])
    m = EnergyModel(n_spins=2, h=h, J_idx=J_idx, J_val=J_val)
    assert m.energy("10") == pytest.approx(0.5 * 1 + (-0.5) * -1, abs=1e-12)
