# Build a tiny model and its Z-basis problem operator.
from qemcmc import EnergyModel, problem_op
import numpy as np

m = EnergyModel(
    n_spins=4,
    h=np.array([1.0, 0.0, 0.0, -2.0]),
    J_idx=np.array([[0, 2], [1, 3]]),
    J_val=np.array([0.5, -1.0]),
)

# The corresponding Pauli operator
H = problem_op(m)

print(H)
