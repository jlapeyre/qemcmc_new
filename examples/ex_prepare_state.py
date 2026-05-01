from qemcmc import prepare_state

# Prepare computational basis state |s> (big-endian mapping).
qc = prepare_state("1010", n_qubits=4)

print(qc)
