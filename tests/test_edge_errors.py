import numpy as np
import pytest
import qemcmc.circuits_core as cc
from qemcmc import EnergyModel, prepare_state, build_proposer_qc, problem_op


def test_alpha_and_state_errors():
    m = EnergyModel(n_spins=3)
    with pytest.raises(ValueError):
        _ = m.alpha
    with pytest.raises(ValueError):
        _ = m.energy("10")
    with pytest.raises(ValueError):
        _ = m.energy([0, 1, 0], spin_type="oops")


def test_prepare_and_build_errors():
    with pytest.raises(ValueError):
        _ = prepare_state("102", n_qubits=3)
    m = EnergyModel(n_spins=2, h=np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        _ = build_proposer_qc(m, "00", gamma=1.1, time=1)
    with pytest.raises(ValueError):
        _ = build_proposer_qc(m, "00", gamma=0.5, time=0)


def test_problem_op_empty_returns_zero_identity():
    m = EnergyModel(n_spins=3)
    H = problem_op(m)
    assert H.paulis.to_labels() == ["III"]
    assert float(H.coeffs[0].real) == 0.0


def test_counts_normalization_and_errors():
    d = cc._normalize_counts_keys_to_bitstrings({3: 1}, width=4)
    assert d == {"0011": 1}
    with pytest.raises(ValueError):
        _ = cc._normalize_counts_keys_to_bitstrings({(0,): 1}, width=1)
    with pytest.raises(ValueError):
        _ = cc._normalize_counts_keys_to_bitstrings({"1": 1}, width=2)
