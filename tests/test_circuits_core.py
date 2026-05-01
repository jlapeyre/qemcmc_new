import numpy as np
import pytest
import importlib
from qiskit.circuit.library import PauliEvolutionGate
from qemcmc import (
    EnergyModel,
    problem_op,
    mixer_op,
    prepare_state,
    build_proposer_qc,
    evolved_statevector,
)

cc = importlib.import_module("qemcmc.circuits_core")


def test_prepare_state_big_endian():
    qc = prepare_state("1010", n_qubits=4)
    xs = []
    for instr in qc.data:
        if instr.operation.name == "x":
            xs.append(qc.find_bit(instr.qubits[0]).index)
    assert set(xs) == {3, 1}


def test_problem_op_big_endian_and_signs():
    m = EnergyModel(
        n_spins=3,
        h=np.array([1.0, 0.0, -2.0]),
        J_idx=np.array([[0, 2]]),
        J_val=np.array([0.5]),
        K_idx=np.array([[0, 1, 2]]),
        K_val=np.array([0.3]),
    )
    H = problem_op(m)
    labels = H.paulis.to_labels()
    assert np.allclose(np.asarray(H.coeffs).imag, 0.0, atol=1e-15)
    coeffs = np.asarray(H.coeffs.real, dtype=float)
    d = {lab: c for lab, c in zip(labels, coeffs)}
    assert d.get("IIZ", None) == pytest.approx(-1.0, abs=1e-12)
    assert d.get("ZII", None) == pytest.approx(2.0, abs=1e-12)
    assert d.get("ZIZ", None) == pytest.approx(0.5, abs=1e-12)
    assert d.get("ZZZ", None) == pytest.approx(-0.3, abs=1e-12)


def test_mixer_op_terms():
    Hx = mixer_op(3)
    labels = set(Hx.paulis.to_labels())
    assert np.allclose(np.asarray(Hx.coeffs).imag, 0.0, atol=1e-15)
    coeffs = np.asarray(Hx.coeffs.real, dtype=float)
    assert labels == {"IIX", "IXI", "XII"}
    assert np.allclose(coeffs, 1.0)


def test_build_proposer_qc_has_evolution_gate():
    m = EnergyModel(
        n_spins=2,
        h=np.array([0.5, -0.5]),
    )
    qc = build_proposer_qc(m, "01", gamma=0.4, time=3, delta_time=0.8)
    has_evo = any(isinstance(inst.operation, PauliEvolutionGate) for inst in qc.data)
    assert has_evo
    assert qc.num_qubits == 2


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_evolved_statevector_shape_and_norm():
    m = EnergyModel(n_spins=2, h=np.array([0.2, -0.1]))
    psi = evolved_statevector(m, "01", gamma=0.5, time=2, delta_time=0.8)
    assert psi.shape == (4,)
    assert np.isclose(np.vdot(psi, psi).real, 1.0, atol=1e-12)
