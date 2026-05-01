# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python
#     name: qemcmc-venv
#     language: python
# ---

# %% [markdown]
# # QeMCMC MIS: build model, set backend, and run MCMC optimize
#
# This notebook / script:
# - loads a DIMACS "edge" graph,
# - builds the MIS EnergyModel,
# - selects a backend via Qiskit Runtime Service and creates a preset pass manager,
# - runs mcmc_optimize using SamplerV2,
# - shows the energy history.

# %%
from qemcmc import load_dimacs_edge, mis_energy_model, random_bitstring, mcmc_optimize
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler import generate_preset_pass_manager

# %% [markdown]
# ## Load a graph and build the MIS model
# Choose a graph filename available in qemcmc assets (or set QEMCMC_GRAPHS_DIR).

# %%
graph_file = "aves-sparrow-social.gph"
print(f"Loading graph {graph_file}")
g = load_dimacs_edge(graph_file)
print(f"nodes {g.num_nodes()}, edges {g.num_edges()}")

lam = 2.0
model, _ = mis_energy_model(g, lam=lam)
print("n_spins", model.n_spins)

# %% [markdown]
# ## Choose initial state
# Use a reproducible random bitstring as the starting configuration.

# %%
seed = 789
init_s = random_bitstring(model.n_spins, rng=seed)
print("initial bitstring:", init_s, "E0=", model.energy(init_s))

# %% [markdown]
# ## Backend and pass manager
# Initialize Qiskit Runtime Service, pick a backend, and create a preset pass manager.
# Uncomment Fake backend to run locally without an account.

# %%
service = QiskitRuntimeService()
backend = service.least_busy(min_num_qubits=model.n_spins)
# from qiskit_ibm_runtime.fake_provider import FakeFez
# backend = FakeFez()
print("backend:", backend)

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
sampler = SamplerV2(backend)

# %% [markdown]
# ## Run MCMC optimization
# Configure proposal evolution and Metropolis settings, then run the loop.

# %%
shots = 1
params = dict(gamma=0.5, time=6, delta_time=0.8, beta=1.0, patience=5, max_iters=200)
print("start:", init_s, "E=", model.energy(init_s))


# %% [markdown]
# ## Run the optimization loop

# %%
res = mcmc_optimize(
    init_s,
    model,
    pm,
    sampler,
    shots,
    gamma=params["gamma"],
    time=params["time"],
    delta_time=params["delta_time"],
    beta=params["beta"],
    patience=params["patience"],
    max_iters=params["max_iters"],
    rng=seed,
    log_every=1,
)

# %%
print("best_s:", res["best_s"], "best_E:", res["best_E"])
print("iters:", res["iters"], "accept_rate:", f'{res["accept_rate"]:.3f}')

# %% [markdown]
# ## Plot energy history

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 3))
plt.plot(res["history_E"], marker="o", ms=3, lw=1)
plt.xlabel("iteration")
plt.ylabel("energy")
plt.title("MCMC energy history")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mcmc_energy_history.png")
# plt.show()
