"""
Graphs and MIS-to-QUBO/Ising utilities.

Acronyms and problem statement
- MIS (Maximum Independent Set):
  Given an undirected simple graph G=(V,E), an independent set S⊆V contains no
  edge with both endpoints in S. MIS seeks a set S of maximum cardinality |S|.
  Binary decision variables x_i∈{0,1} indicate vertex selection (x_i=1 if v_i∈S).

- QUBO (Quadratic Unconstrained Binary Optimization):
  Optimize over binary x∈{0,1}^n an objective that is quadratic in x:
      minimize  E_Q(x) = Σ_i Q_ii x_i + Σ_{i<j} Q_ij x_i x_j  (+ constant)
  We use the “upper” convention above (only i≤j entries are stored/used).

- MIS as a QUBO (minimization form used here):
      E_Q(x) = -Σ_i x_i + λ Σ_{(i,j)∈E} x_i x_j
  The first term maximizes the set size; the second penalizes adjacency.
  Any λ>1 ensures the optimum is an independent set; λ=2.0 is a common choice.

- Ising form and mapping:
  Spins s∈{-1,+1}^n relate to binary x via s = 2x - 1 and x = (s+1)/2.
  The Ising energy is  E_I(s) = Σ_i h_i s_i + Σ_{i<j} J_ij s_i s_j + offset.
  For the MIS QUBO above, one obtains the sparse coefficients:
      h_i   = -1/2 + (λ/4)·deg(i)
      J_ij  =  λ/4  for (i,j)∈E (with i<j)
      offset_Q = +n/2 - (λ/4)|E|
  The offset does not affect sampling or ranking; keep it only if absolute
  energies are required.

- DIMACS “edge” graph format:
  Lines starting with 'c' are comments.
  Header:  'p edge n m' declares node and edge counts (nodes are 1-indexed).
  Edges:   'e u v' for an undirected edge between u and v.

This module provides:
- load_dimacs_edge: parse DIMACS 'edge' graphs into a rustworkx.PyGraph.
- mis_qubo_upper_dense: build the dense upper-triangular MIS QUBO matrix.
- mis_ising_from_graph: derive sparse Ising (h, J) and the constant offset.
- mis_energy_model: convenience wrapper returning an EnergyModel and offset.
- read_dimacs_mis_energy_model: one-shot DIMACS → EnergyModel pipeline.
"""

from __future__ import annotations
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import os
import numpy as np

try:
    import rustworkx as rx
except Exception as exc:
    raise ImportError("rustworkx is required for qemcmc.graphs") from exc

from .energy_model import EnergyModel
from .paths import assets_graph_file

PathLike = Union[str, os.PathLike]


# This takes an already expanded path
# def _resolve_graph_path_in_repo(path: PathLike):
#     return assets_graph_file(path)
#     from pathlib import Path
#     full_path = Path(assets_graph_dir()) / path
#     return full_path


def resolve_graph_path(path: PathLike):
    """
    Resolve a DIMACS graph file path using a single, centralized policy.

    Resolution order
    1) Absolute path: expand ~ and env vars; if absolute and exists, return it; else raise.
    2) Environment: if QEMCMC_GRAPHS_DIR is set, join it with the given relative path.
    3) Packaged assets: qemcmc/assets/graphs/<path> via importlib.resources.

    Parameters
    - path: str or os.PathLike; can be a basename or a relative subpath.

    Returns
    - pathlib.Path pointing to an existing file.

    Raises
    - FileNotFoundError with a helpful message listing attempted locations.

    Notes
    - Keep tests independent of the env by passing an absolute path to the packaged assets.
    """
    import os
    from pathlib import Path

    tried = []
    path_string = os.path.expandvars(os.path.expanduser(str(path)))
    path_new = Path(path_string)
    if path_new.is_absolute():
        if path_new.exists():
            return path_new
        raise FileNotFoundError(f"Graph file not found at absolute path: {path_new}")

    env_dir = os.environ.get("QEMCMC_GRAPHS_DIR", None)
    if env_dir:
        base = Path(os.path.expandvars(os.path.expanduser(env_dir)))
        cand = base / path_new
        tried.append(str(cand))
        if cand.exists():
            return cand

    cand2 = assets_graph_file(path_new)
    tried.append(str(cand2))
    if cand2.exists():
        return cand2

    msg = (
        f"Graph file '{path}' not found. Tried:\n"
        + "\n".join(f"  - {t}" for t in tried)
        + "\nSet QEMCMC_GRAPHS_DIR to point at your benchmark graph directory, "
        + "or pass an absolute path."
    )
    raise FileNotFoundError(msg)


def load_dimacs_edge(
    path: PathLike,
    *,
    validate: bool = True,
) -> rx.PyGraph:
    """
    Read a DIMACS 'edge' format graph into an undirected rustworkx.PyGraph.

    Accepted lines
    - Comment: starts with 'c' (ignored).
    - Header:  'p edge n m' (n nodes, m edges; nodes are 1-indexed in file).
    - Edge:    'e u v' with u != v (multiple edges allowed in file; deduplicated).

    Behavior
    - Converts node ids to 0..n-1.
    - Removes self-loops and duplicate edges.
    - If no header is present, n is inferred as max node id seen.

    Parameters
    - path: file path to a DIMACS 'edge' graph. `resolve_graph_path` will be used to locate
            the file.
    - validate: if True, check that file node ids fit header n (when present).

    Returns
    - PyGraph with node indices 0..n-1 and unweighted edges.
    """
    n_declared: Optional[int] = None
    m_declared: Optional[int] = None
    edges_set: set[Tuple[int, int]] = set()
    max_id: int = -1
    path = resolve_graph_path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("c"):
                continue
            if s.startswith("p"):
                parts = s.split()
                if len(parts) != 4 or parts[1] != "edge":
                    raise ValueError(f"Invalid header line: {s}")
                n_declared = int(parts[2])
                m_declared = int(parts[3])
                continue
            if s.startswith("e"):
                parts = s.split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid edge line: {s}")
                u = int(parts[1]) - 1
                v = int(parts[2]) - 1
                if u == v:
                    continue
                a = min(u, v)
                b = max(u, v)
                edges_set.add((a, b))
                if a > max_id:
                    max_id = a
                if b > max_id:
                    max_id = b
                continue
    if n_declared is None:
        n = max_id + 1 if max_id >= 0 else 0
    else:
        if validate and max_id >= n_declared:
            raise ValueError("File contains node id exceeding header n")
        n = n_declared
    if validate and m_declared is not None and len(edges_set) > m_declared:
        # Duplicates removed can only reduce m; more unique edges is suspicious.
        raise ValueError("Unique edge count exceeds header m after deduplication")
    g = rx.PyGraph()
    for _ in range(n):
        g.add_node(None)
    for u, v in sorted(edges_set):
        g.add_edge(u, v, None)
    return g


def mis_qubo_upper_dense(
    g: rx.PyGraph,
    *,
    lam: float = 2.0,
) -> np.ndarray:
    """
    Build the dense upper-triangular Q matrix for MIS minimization QUBO:
        E_Q(x) = -sum_i x_i + λ * sum_{(i,j)∈E} x_i x_j

    Conventions
    - 'upper' form: E_Q(x) = Σ_i Q_ii x_i + Σ_{i<j} Q_ij x_i x_j with Q_ji unused.
    - Diagonal Q_ii = -1 for all i; for each edge i<j, set Q_ij = λ.

    Parameters
    - g: undirected graph.
    - lam: penalty λ > 1 to enforce independence; default 2.0.

    Returns
    - ndarray shape (n,n): dense Q in the 'upper' convention.
    """
    n = g.num_nodes()
    Q = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(Q, -1.0)
    for u, v in g.edge_list():
        i = min(u, v)
        j = max(u, v)
        Q[i, j] = float(lam)
    return Q


def mis_ising_from_graph(
    g: rx.PyGraph,
    *,
    lam: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Derive sparse Ising coefficients (h, J) and the constant offset for MIS.

    Mapping from the minimization QUBO
        E_Q(x) = -sum_i x_i + λ * sum_{(i,j)∈E} x_i x_j
    yields
        h_i = -1/2 + (λ/4) * deg(i)
        J_ij = λ/4  for (i,j) ∈ E (with i < j)
        offset_Q = +n/2 - (λ/4) * |E|   (so that E_QUBO + offset_Q = E_Ising)

    Parameters
    - g: undirected graph.
    - lam: penalty λ > 1; default 2.0.

    Returns
    - h (n,), J_idx (m,2) with i<j, J_val (m,), offset (float).
    """
    n = g.num_nodes()
    m = g.num_edges()
    deg = np.zeros(n, dtype=np.int64)
    pairs: List[Tuple[int, int]] = []
    for u, v in g.edge_list():
        i = min(u, v)
        j = max(u, v)
        deg[i] += 1
        deg[j] += 1
        pairs.append((i, j))
    h = (-0.5 + 0.25 * float(lam) * deg.astype(np.float64)).astype(np.float64, copy=False)
    if m:
        J_idx = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        J_val = np.full((m,), 0.25 * float(lam), dtype=np.float64)
    else:
        J_idx = np.empty((0, 2), dtype=np.int32)
        J_val = np.empty((0,), dtype=np.float64)
    offset = 0.5 * float(n) - 0.25 * float(lam) * float(m)
    return h, J_idx, J_val, offset


def mis_energy_model(
    g: rx.PyGraph,
    *,
    lam: float = 2.0,
    prune_tol: float = 0.0,
    name: Optional[str] = None,
) -> Tuple[EnergyModel, float]:
    """
    Build an EnergyModel for MIS from a rustworkx graph plus the offset.

    Parameters
    - g: undirected graph.
    - lam: penalty λ > 1; default 2.0.
    - prune_tol: optional pruning threshold for tiny coefficients.
    - name: optional model label.

    Returns
    - (EnergyModel, offset): model with pairwise Ising terms only, and the constant.
    """
    h, J_idx, J_val, offset = mis_ising_from_graph(g, lam=lam)
    model = EnergyModel(
        n_spins=g.num_nodes(),
        h=h,
        J_idx=J_idx,
        J_val=J_val,
        name=name,
        prune_tol=prune_tol,
    )
    return model, offset


def read_dimacs_mis_energy_model(
    path: PathLike,
    *,
    lam: float = 2.0,
    validate: bool = True,
    prune_tol: float = 0.0,
    name: Optional[str] = None,
) -> Tuple[EnergyModel, float, rx.PyGraph]:
    """
    One-shot pipeline: DIMACS 'edge' file -> PyGraph -> MIS EnergyModel.

    Parameters
    - path: DIMACS 'edge' file path.
    - lam: penalty λ > 1; default 2.0.
    - validate: if True, enforce header consistency checks.
    - prune_tol: optional pruning threshold passed to EnergyModel.
    - name: optional model label.

    Returns
    - (model, offset, graph): the pairwise Ising EnergyModel, its constant offset,
      and the parsed rustworkx.PyGraph for any further use.
    """
    g = load_dimacs_edge(path, validate=validate)
    model, offset = mis_energy_model(g, lam=lam, prune_tol=prune_tol, name=name)
    return model, offset, g
