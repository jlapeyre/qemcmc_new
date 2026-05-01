from qemcmc import EnergyModel, evolved_statevector
import numpy as np

# Statevector after evolution (no measurement).
m = EnergyModel(
    n_spins=3,
    h=np.array([0.2, -0.1, 0.0]),
    J_idx=np.array([[0, 1]]),
    J_val=np.array([0.7]),
)
psi = evolved_statevector(m, "101", gamma=0.5, time=6, delta_time=0.8)

print(psi)
