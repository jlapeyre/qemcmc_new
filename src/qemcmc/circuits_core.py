"""
Circuit builders for QeMCMC-style proposals.

Design
- Problem Hamiltonian is constructed from EnergyModel.iter_terms() as Z-strings.
- Total evolution: H_total = gamma * H_mixer + (1 - gamma) * alpha * H_problem
  with the conventional Ising sign absorbed in problem_op construction.

Exports
- problem_op(model) -> SparsePauliOp
- mixer_op(n_qubits) -> SparsePauliOp
- prepare_state(s[, n_qubits]) -> QuantumCircuit
- build_proposer_qc(model, s, gamma, time[, delta_time]) -> QuantumCircuit
- evolved_statevector(model, s, gamma, time[, delta_time]) -> np.ndarray
"""

from __future__ import annotations

from typing import Optional, Any, Dict, Mapping
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter

__all__ = [
    "problem_op",
    "mixer_op",
    "prepare_state",
    "build_proposer_qc",
    "evolved_statevector",
]


def _validate_bitstring(s: str, *, length: Optional[int] = None) -> str:
    """Validate s is a {0,1}-string and, if provided, matches length."""
    if not isinstance(s, str):
        raise TypeError("bitstring must be str")
    if set(s) - {"0", "1"}:
        raise ValueError("bitstring must contain only '0'/'1'")
    if length is not None and len(s) != length:
        raise ValueError(f"bitstring length {len(s)} != {length}")
    return s


def _normalize_counts_keys_to_bitstrings(
    counts: Mapping[Any, int], *, width: int
) -> Dict[str, int]:
    """
    Normalize a sampler counts mapping to canonical {0,1}-string keys of fixed length.

    Purpose
    - Different samplers may return outcome keys as integers or as bitstrings of
      varying width. This function standardizes them for downstream consumers.

    Behavior
    - int keys: converted via zero-padded binary, format(k, f"0{width}b").
    - str keys: must already be {0,1}-strings and match the requested width; validated.
    - Any other key type raises ValueError.

    Parameters
    - counts: mapping outcome -> count as returned by a sampler (e.g., SamplerV2).
    - width: required bit-width for normalized keys.

    Returns
    - dict[str, int] with zero-padded bitstring keys of length `width` and integer counts.
    """
    out: Dict[str, int] = {}
    for k, v in counts.items():
        if isinstance(k, str):
            ks = _validate_bitstring(k, length=width)
        elif isinstance(k, (int, np.integer)):
            ks = format(int(k), f"0{width}b")
        else:
            raise ValueError(f"Unsupported counts key type {type(k)}: {k!r}")
        out[ks] = int(v)
    return out


def problem_op(model, *, n_qubits: Optional[int] = None) -> SparsePauliOp:
    """
    Build the problem Hamiltonian H_problem as Z-strings from the model's sparse terms.

    Convention
    - Uses strictly increasing index tuples from model.iter_terms().
    - Big-endian placement: index i maps to Z on position (n-1-i).
    - Includes the Ising spin-sign factor (-1)^k for k-body terms so that
      E(s) computed by the model aligns with <s|Z...Z|s> contributions.

    Parameters
    - model: EnergyModel with iter_terms() yielding (order, indices, coeff).
    - n_qubits: override number of qubits; defaults to model.n_spins.

    Returns
    - SparsePauliOp summing all Z, ZZ, ZZZ terms (empty operator if no terms).

    Example
    >>> from qemcmc import EnergyModel
    >>> import numpy as np
    >>> m = EnergyModel(n_spins=3, h=np.array([1,0,-1.0]), J_idx=np.array([[0,2]]), J_val=np.array([0.5]))
    >>> H = problem_op(m)
    >>> len(H.paulis) > 0
    True
    """
    n = n_qubits or model.n_spins
    labels, coeffs = [], []
    for order, idxs, c in model.iter_terms():
        spin_sign = (-1) ** order
        p = ["I"] * n
        for i in idxs:
            p[n - 1 - i] = "Z"
        labels.append("".join(p))
        coeffs.append(spin_sign * float(c))
    if not labels:
        return SparsePauliOp.from_list([("I" * n, 0.0)])
    return SparsePauliOp(labels, coeffs=np.asarray(coeffs, dtype=float))


def mixer_op(n_qubits: int) -> SparsePauliOp:
    """
    Return the transverse-field mixer Hamiltonian H_mixer = Σ_i X_i (big-endian).

    Parameters
    - n_qubits: number of qubits.

    Returns
    - SparsePauliOp representing the sum of single-qubit X terms.

    Example
    >>> Hx = mixer_op(4)
    >>> Hx.num_qubits
    4
    """
    terms = [("I" * (n_qubits - 1 - i) + "X" + "I" * i, 1.0) for i in range(n_qubits)]
    return SparsePauliOp.from_list(terms)


def prepare_state(s: str, *, n_qubits: Optional[int] = None) -> QuantumCircuit:
    """
    Prepare computational basis state |s> from |0...0> using big-endian mapping.

    Parameters
    - s: bitstring of length n over {'0','1'}.
    - n_qubits: optional override for circuit width; defaults to len(s).

    Returns
    - QuantumCircuit with X gates applied where s has '1'.

    Example
    >>> qc = prepare_state("1010")
    >>> qc.num_qubits
    4
    """
    n = n_qubits or len(s)
    _validate_bitstring(s, length=n)
    qc = QuantumCircuit(n)
    for i, b in enumerate(s):
        if b == "1":
            qc.x(n - 1 - i)
    return qc


def build_proposer_qc(
    model, s: str, *, gamma: float, time: int, delta_time: float = 0.8
) -> QuantumCircuit:
    """
    Build the proposal evolution circuit (no measurements).

    Model
    - H_total = gamma * H_mixer + (1 - gamma) * alpha * H_problem,
      implemented as Hx * gamma + Hp * (-(1-gamma)*alpha) to match Ising sign conventions.
    - Uses Suzuki–Trotter with reps = floor(time / delta_time).

    Parameters
    - model: EnergyModel providing n_spins, alpha, and iter_terms().
    - s: input bitstring to prepare before evolution (big-endian).
    - gamma: mixer weight in [0,1].
    - time: positive integer evolution time.
    - delta_time: positive step controlling Trotter reps.

    Returns
    - QuantumCircuit performing the time evolution (no measure).

    Example
    >>> from qemcmc import EnergyModel
    >>> import numpy as np
    >>> m = EnergyModel(n_spins=2, h=np.array([0.5,-0.5]))
    >>> qc = build_proposer_qc(m, "01", gamma=0.4, time=3)
    >>> qc.num_qubits
    2
    """
    if not (0.0 <= float(gamma) <= 1.0):
        raise ValueError("gamma must be in [0,1]")
    t = int(time)
    if t <= 0:
        raise ValueError("time must be a positive integer")
    reps = int(np.floor(t / float(delta_time)))

    n = model.n_spins
    qc = prepare_state(s, n_qubits=n)

    Hx = mixer_op(n) * float(gamma)
    Hp = problem_op(model) * (-(1.0 - float(gamma)) * float(model.alpha))
    evo = PauliEvolutionGate(Hx + Hp, time=t, synthesis=SuzukiTrotter(reps=reps))
    qc.append(evo, range(n))
    return qc


def evolved_statevector(
    model, s: str, *, gamma: float, time: int, delta_time: float = 0.8
) -> np.ndarray:
    """
    Return the evolved statevector (no measurement) for a single proposal.

    Parameters
    - model: EnergyModel used to build H_problem and alpha scaling.
    - s: input bitstring prepared as |s>.
    - gamma: mixer weight in [0,1].
    - time: positive integer evolution time.
    - delta_time: positive step controlling Trotter reps.

    Returns
    - np.ndarray: complex statevector of length 2**n (dtype complex128).

    Example
    >>> from qemcmc import EnergyModel
    >>> import numpy as np
    >>> m = EnergyModel(n_spins=2, h=np.array([0.2, -0.1]))
    >>> psi = evolved_statevector(m, "01", gamma=0.5, time=2)
    >>> psi.shape[0] == 4
    True
    """
    qc = build_proposer_qc(model, s, gamma=gamma, time=time, delta_time=delta_time)
    return np.asarray(Statevector.from_instruction(qc).data)
