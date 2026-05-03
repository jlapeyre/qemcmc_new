from qiskit.converters import circuit_to_dag
from qiskit.converters import dag_to_circuit
from qiskit.circuit import Qubit, QuantumCircuit

__all__ = [
    "strip_idle_qubits",
    "twoq_gate_count",
    "twoq_depth",
    "active_qubit_count",
    "active_qubit_count_dag",
    "op_count_map",
    "select_ops",
    "count_map",
]


def strip_idle_qubits(qc):
    """
    Remove qubits that carry no operations and return a compacted circuit.
    """
    dag = circuit_to_dag(qc)
    idle = [w for w in dag.idle_wires() if isinstance(w, Qubit)]
    for q in idle:
        dag.remove_qarg(q)
    return dag_to_circuit(dag)


def twoq_gate_count(qc):
    """
    Count the number of two-qubit gates present in the circuit data.
    """
    return sum(1 for inst in qc.data if getattr(inst.operation, "num_qubits", 0) == 2)


def twoq_depth(qc):
    """
    Compute the number of DAG layers that contain at least one two-qubit gate.
    """
    dag = circuit_to_dag(qc)
    return sum(
        any(n.op.num_qubits == 2 for n in layer["graph"].op_nodes()) for layer in dag.layers()
    )


def active_qubit_count(qc):
    """
    Return the number of non-idle (active) qubits in the circuit.

    A qubit is considered active if it participates in at least one operation.
    """
    dag = circuit_to_dag(qc)
    return active_qubit_count_dag(dag)


def active_qubit_count_dag(dag):
    """
    Return the number of non-idle (active) qubits in the circuit.

    A qubit is considered active if it participates in at least one operation.
    """
    idle = [w for w in dag.idle_wires() if isinstance(w, Qubit)]
    return dag.num_qubits() - len(idle)


def op_count_map(qc):
    """
    Return a dict mapping operation class names to their occurrence counts.
    """
    names = [type(inst.operation).__name__ for inst in qc.data]
    return {k: names.count(k) for k in set(names)}


# --- Usage Examples ---
# 1. Single indices: select_ops(qc, 0, 5, 10)
# 2. Slices:       select_ops(qc, slice(0, 4))  # Equivalent to qc[0:4]
# 3. Mixed:        new = select_ops(qc, 2, slice(4, 7), [10, 15])
def select_ops(qc, *indices) -> QuantumCircuit:
    """
    Creates a new circuit containing only the specified operation indices.
    Accepts integers, slices, or lists of integers.
    Example: select_ops(qc, 2, slice(4, 7), [10, 12])
    """
    new_qc = qc.copy_empty_like()

    # Flatten the inputs into a single list of indices
    idx_list = []
    for item in indices:
        if isinstance(item, int):
            idx_list.append(item)
        elif isinstance(item, slice):
            # Convert slice(a, b) into actual indices based on circuit length
            idx_list.extend(range(*item.indices(len(qc))))
        elif isinstance(item, (list, tuple)):
            idx_list.extend(item)

    # Append the selected instructions
    for i in idx_list:
        new_qc.append(qc[i])

    return new_qc


# def count_map(counts: Mapping[Any, int], width: int) -> Dict[Any, int]:
#     out: Dict[Any, int] = {}
#     for k, v in counts.items():
#         out[s] = int(v)
#     return out
