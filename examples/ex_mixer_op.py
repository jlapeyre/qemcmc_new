from qemcmc import mixer_op

# Transverse-field mixer: Hx = sum_i X_i on n qubits.
Hx = mixer_op(4)

print(Hx)
