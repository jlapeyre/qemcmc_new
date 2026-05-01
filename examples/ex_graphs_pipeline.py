from pathlib import Path
from qemcmc.graphs import read_dimacs_mis_energy_model

"""
Pipeline example:
- Reads a DIMACS 'edge' graph (karate.gph) from tests/data/graphs.
- Uses read_dimacs_mis_energy_model to get (EnergyModel, offset, graph).
- Prints a few summary stats including model alpha and sparse sizes.
"""

graph_file = "karate.gph"
# graph_file = "ibm32.gph"

lam = 2.0
graphs_dir = Path(__file__).resolve().parents[1] / "tests" / "data" / "graphs"
model, offset, g = read_dimacs_mis_energy_model(graphs_dir / graph_file, lam=lam)
print("nodes", g.num_nodes())
print("edges", g.num_edges())
print("alpha", model.alpha)
print("offset", offset)
print("h_nonzero", int((model.h is not None) and (model.h != 0).sum()))
print("J_terms", int((model.J_idx is not None) and model.J_idx.shape[0]))
