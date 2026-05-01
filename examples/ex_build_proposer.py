from qemcmc import EnergyModel, build_proposer_qc
import numpy as np

# Build the QeMCMC proposer circuit for (model, s, gamma, time).
m = EnergyModel(
    n_spins=4,
    h=np.array([1.0, 0.0, 0.0, -2.0]),
    J_idx=np.array([[0, 2], [1, 3]]),
    J_val=np.array([0.5, -1.0]),
)
qc = build_proposer_qc(m, "1010", gamma=0.5, time=6, delta_time=0.8)

print(qc)
