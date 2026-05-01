from qemcmc import EnergyModel
import numpy as np

"""
- This example builds:
  - 4-spin Ising model with h = [1, 0, 0, -2], 2-body terms: J01? no; J02=+0.5, J13=-1.0; no 3-body.
  - Bitstring "1010" maps to spins s = [1, -1, 1, -1] (0→-1, 1→+1).

- Expected energy
  - h·s = 1·1 + 0·(-1) + 0·1 + (-2)·(-1) = 3
  - 2-body: 0.5·(s0 s2) + (-1.0)·(s1 s3) = 0.5·(1·1) + (-1.0)·((-1)·(-1)) = 0.5 - 1.0 = -0.5
  - Total E = 3 + (-0.5) = 2.5

- Expected alpha
  - α = sqrt(n) / sqrt(Σ h_i^2 + Σ J_val^2) = sqrt(4) / sqrt(1^2 + (-2)^2 + 0.5^2 + (-1)^2) = 2 / sqrt(6.25) = 0.8
"""

m = EnergyModel(
    n_spins=4,
    h=np.array([1, 0, 0, -2.0]),
    J_idx=np.array([[0, 2], [1, 3]]),
    J_val=np.array([0.5, -1.0]),
)

print(m.energy("1010"), m.alpha)  # expected: 2.5 0.8
