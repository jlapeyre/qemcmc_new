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
# # Create and run QeMCMC circuit from a graph
#

# %%
from qemcmc import mis_energy_model, random_bitstring, build_proposer_qc
from qemcmc import active_qubit_count
from qemcmc import load_dimacs_edge

from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# %% [markdown]
# We either read or generate a graph to demonstrate the workflow.
# Even the simplest graph the QOBLIB data repository results in a circuit
# that is too large for a state vector simulator.

# %%
# import rustworkx as rx
# graph = rx.generators.path_graph(6)
# graph_file = "farm.gph"
# graph_file = "mammalia-kangaroo-interactions.gph"
graph_file = "aves-sparrow-social.gph"
print(f"Loading graph  {graph_file}")
graph = load_dimacs_edge(graph_file)

print(f"n_nodes {graph.num_nodes()}, n_edges {graph.num_edges()}")

model, _ = mis_energy_model(graph, lam=2.0)
initial_spin_state = random_bitstring(model.n_spins, rng=123)
circuit = build_proposer_qc(model, initial_spin_state, gamma=0.5, time=6, delta_time=0.8)

# %% [markdown]
# If we have no measurements, there is no output to collect from shots.

# %%
circuit.measure_all()

print("Number of qubits in circuit ", circuit.num_qubits)

#%%
service = QiskitRuntimeService()

# %% [markdown]
# The number of qubits in device must be greater or equal to what we have.
# Because auxiliary qubits will be needed this check may not be sufficient.

#%%
print("Getting backend...")
backend = service.least_busy(min_num_qubits=circuit.num_qubits)

# %%
# backend = FakeFez()

# %%
print(backend)

# %%
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

print("Transpiling...")
isa_circuit = pm.run(circuit)

# %% [markdown]
# The width of the transpiled circuit is equal to the width of the device.

# %%
print("Number of qubits in isa_qc ", isa_circuit.num_qubits)

print("Number of active qubits in isa_qc ", active_qubit_count(isa_circuit))

sampler = SamplerV2(backend)

print("Running circuit...")
job = sampler.run([isa_circuit])

print("Fetching result...")
pub_result = job.result()[0]

print("Getting counts...")
counts = pub_result.data.meas.get_counts()

print("input spin state:", initial_spin_state)
# print("counts:", counts)

# %% [markdown]
# Plot a histogram of the freqencies of counts collected

# %%
plot_histogram(counts)

# %%
import matplotlib.pyplot as plt
plt.savefig('counts_hist.png')

# plt.show()
