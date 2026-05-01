from pathlib import Path
import numpy as np
from qemcmc.graphs import (
    load_dimacs_edge,
    mis_qubo_upper_dense,
    mis_ising_from_graph,
    mis_energy_model,
)

"""
Builders example:
- Loads ibm32.gph.
- Runs the individual builders: load_dimacs_edge, mis_qubo_upper_dense,
  mis_ising_from_graph, and mis_energy_model.
- Confirms basic consistency among the outputs.
"""

lam = 1.7
graphs_dir = Path(__file__).resolve().parents[1] / "tests" / "data" / "graphs"
g = load_dimacs_edge(graphs_dir / "ibm32.gph", validate=True)
Q = mis_qubo_upper_dense(g, lam=lam)
h, J_idx, J_val, off = mis_ising_from_graph(g, lam=lam)
model, off2 = mis_energy_model(g, lam=lam)
print("n", g.num_nodes(), "m", g.num_edges())
print("Q shape", Q.shape, "diag all -1:", np.allclose(np.diag(Q), -1.0))
print("J_count", J_idx.shape[0], "J_val_constant", np.allclose(J_val, lam / 4.0))
print("offsets_equal", np.isclose(off, off2))
print("model_h_matches", model.h is not None and np.allclose(model.h, h))
