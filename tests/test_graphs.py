import numpy as np
import pytest
from pathlib import Path

from qemcmc.graphs import (
    assets_graph_file,
    load_dimacs_edge,
    mis_qubo_upper_dense,
    mis_ising_from_graph,
    mis_energy_model,
    read_dimacs_mis_energy_model,
)
from qemcmc.energy_model import qubo_to_ising


def graph_file_path(filename) -> Path:
    return assets_graph_file(filename)

@pytest.mark.parametrize("fname", ["ibm32.gph", "karate.gph"])
def test_load_dimacs_edge_basic(fname):
    g = load_dimacs_edge(graph_file_path(fname), validate=True)
    assert g.num_nodes() > 0
    assert g.num_edges() > 0
    assert all(u != v for u, v in g.edge_list())


@pytest.mark.parametrize("fname", ["ibm32.gph", "karate.gph"])
@pytest.mark.parametrize("lam", [1.3, 2.0])
def test_qubo_and_ising_consistency(fname, lam):
    g = load_dimacs_edge(graph_file_path(fname), validate=True)
    n = g.num_nodes()
    m = g.num_edges()
    Q = mis_qubo_upper_dense(g, lam=lam)
    assert Q.shape == (n, n)
    assert np.allclose(np.diag(Q), -1.0)
    assert np.count_nonzero(np.triu(Q, 1)) == m
    assert np.isclose(np.triu(Q, 1).sum(), lam * m)
    h, J_idx, J_val, offset = mis_ising_from_graph(g, lam=lam)
    assert J_idx.shape == (m, 2)
    assert np.all(J_idx[:, 0] < J_idx[:, 1])
    assert np.allclose(J_val, lam / 4.0)
    deg = np.zeros(n, dtype=int)
    for i, j in g.edge_list():
        a = min(i, j)
        b = max(i, j)
        deg[a] += 1
        deg[b] += 1
    assert np.allclose(h, -0.5 + (lam / 4.0) * deg.astype(float))
    # QUBO-side constant: offset_Q = +n/2 - (λ/4) * |E|
    assert np.isclose(offset, 0.5 * n - (lam / 4.0) * m)


@pytest.mark.parametrize("fname", ["ibm32.gph", "karate.gph"])
@pytest.mark.parametrize("lam", [1.5, 2.0])
def test_energy_model_matches_sparse_ising(fname, lam):
    g = load_dimacs_edge(graph_file_path(fname), validate=True)
    h, J_idx, J_val, offset = mis_ising_from_graph(g, lam=lam)
    model, offset2 = mis_energy_model(g, lam=lam)
    assert np.isclose(offset, offset2)
    assert model.h is not None
    assert np.allclose(model.h, h)
    assert model.J_idx is not None and model.J_val is not None
    pairs = {tuple(map(int, r)): float(v) for r, v in zip(model.J_idx, model.J_val)}
    for (i, j), v in zip(J_idx, J_val):
        assert pairs[(int(i), int(j))] == pytest.approx(float(v), abs=1e-12)


@pytest.mark.parametrize("fname", ["ibm32.gph", "karate.gph"])
def test_read_dimacs_pipeline_equals_manual(fname):
    lam = 2.2
    path = graph_file_path(fname)
    #    path = graphs_dir() / fname
    g0 = load_dimacs_edge(path, validate=True)
    model0, off0 = mis_energy_model(g0, lam=lam)
    model1, off1, g1 = read_dimacs_mis_energy_model(path, lam=lam, validate=True)
    assert g0.num_nodes() == g1.num_nodes()
    assert g0.num_edges() == g1.num_edges()
    assert np.isclose(off0, off1)
    assert np.allclose(model0.h, model1.h)
    assert np.array_equal(model0.J_idx, model1.J_idx)
    assert np.allclose(model0.J_val, model1.J_val)


@pytest.mark.parametrize("fname", ["ibm32.gph", "karate.gph"])
def test_qubo_to_ising_matches_builder(fname):
    lam = 1.9
    g = load_dimacs_edge(graph_file_path(fname), validate=True)
    Q = mis_qubo_upper_dense(g, lam=lam)
    h1, J_idx1, J_val1, off1 = mis_ising_from_graph(g, lam=lam)
    h2, J_idx2, J_val2, off2 = qubo_to_ising(Q, interpret="upper")
    assert np.allclose(h1, h2)
    assert np.array_equal(J_idx1, J_idx2)
    assert np.allclose(J_val1, J_val2)
    assert np.isclose(off1, off2)
