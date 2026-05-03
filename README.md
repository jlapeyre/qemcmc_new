# QeMCMC

QeMCMC provides tools for quantum enhanced Markov chain Monte Carlo (QeMCMC) using Qiskit.


## Modules and features

- [`src/qemcmc/energy_model.py`](./src/qemcmc/energy_model.py)
  - `EnergyModel`: sparse Ising (1-, 2-, 3-body), energy, alpha, QUBO <-> Ising.
  - Energy evaluation from bitstrings or spin vectors.
  - Alpha scaling factor: $\alpha = \sqrt{n}/\sqrt{\sum_i h_i^2 + \sum_t J_t^2 + \sum_u K_u^2}$.
  - QUBO to Ising and Ising to QUBO conversion for pairwise models.

- [`src/qemcmc/circuits_core.py`](./src/qemcmc/circuits_core.py)
  - `problem_op(model)`: builds the Z-basis Ising operator. Signs are chosen so that $\langle s|H|s\rangle$ matches the model energy.
  - `mixer_op(n)`: transverse-field mixer $\sum_i X_i$.
  - `prepare_state(s)`: prepares computational basis state |s>.
  - `build_proposer_qc(model, s, gamma, time, delta_time)`: builds a Trotterized evolution circuit for $H(\gamma) = \gamma H_x + (1-\gamma)\,\alpha\,H_z$.
  - `evolved_statevector(model, s, ...)`: returns the statevector from the above evolution.
  - Bitstring mapping is big-endian: the leftmost bit `s[0]` maps to the highest-index qubit (n-1).

- [`src/qemcmc/graphs.py`](./src/qemcmc/graphs.py) (requires rustworkx)
  - Load DIMACS "edge" graphs and build MIS models.
  - Definitions:
    - MIS (Maximum Independent Set): choose a largest subset of vertices with no edges between any chosen pair.
    - QUBO (Quadratic Unconstrained Binary Optimization): optimize a quadratic form over binary variables.
  - MIS QUBO (minimization) used here:
    - $E_Q(x) = -\sum_i x_i + \lambda \sum_{(i,j)\in E} x_i x_j$, with $\lambda > 1$.
  - Mapping to Ising with $s = 2x - 1$ ($x = (s+1)/2$):
    - $h_i = -\tfrac{1}{2} + \tfrac{\lambda}{4}\,\mathrm{deg}(i)$
    - $J_{ij} = \tfrac{\lambda}{4}$ for $(i,j)\in E$, $i<j$
    - offset $= \tfrac{n}{2} - \tfrac{\lambda}{4}\,|E|$
  - Helpers:
    - `mis_qubo_upper_dense`: dense upper-triangular Q matrix for the above QUBO.
    - `mis_ising_from_graph`: sparse $(h, J)$ and offset.
    - `mis_energy_model`: `EnergyModel` plus offset.
    - `read_dimacs_mis_energy_model`: one-shot DIMACS → EnergyModel.

- [`src/qemcmc/qiskit_utils.py`](./src/qemcmc/qiskit_utils.py)
  - `strip_idle_qubits`: remove qubits with no operations and compact the circuit.
  - `twoq_gate_count`: count two-qubit gates in circuit data.
  - `twoq_depth`: number of DAG layers that contain at least one two-qubit gate.
  - `active_qubit_count`: number of qubits that participate in at least one operation.
  - `active_qubit_count_dag`: same as above, starting from a DAG.
  - `op_count_map`: map from operation class name to occurrence count.
  - `select_ops`: create a subcircuit by selecting instruction indices, slices, or lists.

## Install

Clone the repository:
```sh
git clone <repo-url>
cd qemcmc
```

Create and activate a virtual environment:
```sh
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Editable install:
```sh
pip install -e .
```

Install requirements
```sh
pip install -r requirements.txt
```

It is a good idea to install the dev dependencies as well
```sh
pip install -r requirements-dev.txt
```

With an in-place or editable installation, installing a kernel in the environment
may be necessary if you want to run notebooks.
Choose the kernel `qemcmc-venv` when running
```sh
python -m ipykernel install --user --name qemcmc-venv --display-name "Python (qemcmc-venv)"
```

## Test

Run the test suite with:
```sh
pytest
```

## Graph files

If you want benchmark QOBLIB files, clone [qoblib here](https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library)

Set this environment variable to point to the MIS graphs files like this
```sh
export QEMCMC_GRAPH_DIR="/path/to/qoblib-quantum-optimization-benchmarking-library/07-independentset/instances/"
```
Then, the functions `load_dimacs_edge` and `read_dimacs_mis_energy_model`. Will search for `gph` files in this directory.
See the doc string for `resolve_graph_path(path)` to see the rules for resolving the path to a graph file

Two of the graphs that were studied in

* _Quantum-enhanced Markov Chain Monte Carlo for Combinatorial Optimization_, Kate V. Marshall et. al., http://arxiv.org/abs/2602.06171

are included here:
* [`assets/graphs/mammalia-kangaroo-interactions.gph`](assets/graphs/mammalia-kangaroo-interactions.gph)
* [`assets/graphs/aves-sparrow-social.gph`](assets/graphs/aves-sparrow-social.gph)

## Examples

- [examples/ex_model.py](./examples/ex_model.py): construct a small Ising model and compute energy and alpha.
- [examples/ex_build_proposer.py](./examples/ex_build_proposer.py): build a proposer circuit from a model and a bitstring.
- [examples/ex_evolved_statevector.py](./examples/ex_evolved_statevector.py): compute the evolved statevector (no measurement).
- [examples/ex_graphs_pipeline.py](./examples/ex_graphs_pipeline.py): DIMACS graph → MIS EnergyModel pipeline.
- [examples/ex_graphs_builders.py](./examples/ex_graphs_builders.py): individual MIS builders and consistency checks.
- [examples/workflows/mis_path_graph.py](./examples/workflows/mis_path_graph.py): MIS workflow with transpilation and SamplerV2.
- [examples/workflows/mis_optimize.py](./examples/workflows/mis_optimize.py): end-to-end MIS MCMC optimization using `mcmc_optimize`.

## Notebooks

- [examples/workflows/](./examples/workflows/): notebooks for workflows.
- [examples/workflows/mis_optimize.ipynb](./examples/workflows/mis_optimize.ipynb): notebook version of the MCMC MIS optimization workflow.

## Optimization

- The Metropolis loop and sampling utilities live in [src/qemcmc/optimize.py](./src/qemcmc/optimize.py).
  - `mcmc_optimize(initial_bitstr, model, pm, sampler, shots, ...)` runs proposal circuits, collects counts, evaluates energies, applies Metropolis acceptance, and tracks best found state and energy.
  - Returns a dict with `best_s`, `best_E`, `last_s`, `last_E`, `iters`, `accepted`, `accept_rate`, `history_E`, `history_s`.

Minimal usage sketch (see linked example for a full script):
```python
from qemcmc import mcmc_optimize
res = mcmc_optimize(init_s, model, pm, sampler, shots, gamma=0.5, time=6, delta_time=0.8)
print(res["best_s"], res["best_E"])
```


## Conventions

- Bitstrings are big-endian: bit 0 (leftmost) maps to qubit index $n-1$, and bit $i$ maps to $n-1-i$.
- The problem operator uses Z strings. Signs are chosen so that $\langle s|H|s\rangle$ equals the model energy for computational-basis state $|s\rangle$.
- The proposer Hamiltonian is $H(\gamma) = \gamma \sum_i X_i + (1-\gamma)\,\alpha\,H_z$; evolution uses Trotterization with `delta_time` controlling the number of repetitions.

## I/O and persistence

- `EnergyModel.save(path, offset=..., meta=...)` writes NPZ with arrays and metadata.
- `EnergyModel.load(path)` reconstructs the model and returns `(model, offset, meta)`.
- See tests for round-trip and read-only array checks.

## Development

Install "just" to use task recipes:
```sh
# macOS (Homebrew)
brew install just
# Linux (package manager) or cross-platform
cargo install just
# Windows (Scoop)
scoop install just
```

Common tasks:
- `just` (list recipes)
- `just fmt` (format with ruff)
- `just fmt-check` (format check)
- `just lint` (ruff lint)
- `just fix` (apply safe ruff fixes)
- `just test` (pytest)
- `just check` (fmt-check, lint, type, test)


## License
Apache-2.0
