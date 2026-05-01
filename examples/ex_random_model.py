# Example: build a random pairwise Ising model and print a few summary stats.
# - n=6 spins/qubits.
# - p1: probability a 1-body field h_i is present (nonzero).
# - p2: probability a 2-body coupling J_ij is present (nonzero).
# - target_alpha: rescale coefficients so the model’s alpha ≈ this value.
# - rng: random seed or Generator for reproducibility.

from qemcmc import random_pairwise_model

m = random_pairwise_model(6, p1=0.5, p2=0.3, target_alpha=1.0, rng=123)  # build model

print("n_spins ", m.n_spins)  # number of spins/qubits in the model
print("alpha ", m.alpha)  # computed alpha scaling after any rescaling
print(m.h is not None and (m.h != 0).sum() or 0)  # count of nonzero 1-body fields h_i
print(m.J_idx is not None and m.J_idx.shape[0] or 0)  # number of 2-body couplings
