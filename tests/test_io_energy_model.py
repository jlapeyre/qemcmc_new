import numpy as np
import pytest
from qemcmc import EnergyModel


def test_save_load_all_none_model(tmp_path):
    m = EnergyModel(n_spins=5)
    p = tmp_path / "none.npz"
    m.save(p)
    m2, off2, meta2 = EnergyModel.load(p)
    assert m.equals(m2)
    assert off2 is None and meta2 == {}
    assert m2.h is None and m2.J_idx is None and m2.K_idx is None


def test_save_load_unicode_name_and_prune_tol_meta(tmp_path):
    m = EnergyModel(n_spins=3, h=np.array([0.1, 0.0, -0.2]), name="naïve-Ñáme✓")
    meta = {"k": "väl", "n": 2, "nest": {"u": "Ω"}}
    p = tmp_path / "unicode.npz"
    m.save(p, offset=3.14, meta=meta, name="overridden-名", prune_tol_meta=1e-8)
    m2, off2, meta2 = EnergyModel.load(p)
    assert m2.name == "overridden-名"
    assert off2 == pytest.approx(3.14, abs=0)
    assert meta2 == meta
    with np.load(p, allow_pickle=False) as z:
        assert float(z["prune_tol"]) == pytest.approx(1e-8, abs=0)
        assert str(z["name"]) == "overridden-名"


def test_loaded_arrays_are_readonly(tmp_path):
    m = EnergyModel(
        n_spins=4,
        h=np.array([1.0, 0.0, -2.0, 0.5]),
        J_idx=np.array([[0, 2], [1, 3]], dtype=np.int32),
        J_val=np.array([0.25, -0.75], dtype=float),
        K_idx=np.array([[0, 1, 2]], dtype=np.int32),
        K_val=np.array([0.3], dtype=float),
    )
    p = tmp_path / "ro.npz"
    m.save(p)
    m2, _, _ = EnergyModel.load(p)
    if m2.h is not None:
        assert m2.h.flags.writeable is False
    if m2.J_idx is not None:
        assert m2.J_idx.flags.writeable is False and m2.J_idx.dtype == np.int32
    if m2.J_val is not None:
        assert m2.J_val.flags.writeable is False
    if m2.K_idx is not None:
        assert m2.K_idx.flags.writeable is False and m2.K_idx.dtype == np.int32
    if m2.K_val is not None:
        assert m2.K_val.flags.writeable is False


def test_save_load_with_explicit_empty_J_arrays_semantics(tmp_path):
    h = np.array([0.2, -0.3, 0.1, 0.0])
    m = EnergyModel(
        n_spins=4,
        h=h,
        J_idx=np.empty((0, 2), dtype=np.int32),
        J_val=np.empty((0,), dtype=np.float64),
    )
    p = tmp_path / "emptyJ.npz"
    m.save(p)
    m2, _, _ = EnergyModel.load(p)
    assert m2.J_idx is None and m2.J_val is None
    s = "1010"
    assert m.energy(s) == pytest.approx(m2.energy(s), abs=1e-12)


def test_load_minimal_npz_missing_optional_keys(tmp_path):
    p = tmp_path / "minimal.npz"
    np.savez(
        p,
        n_spins=np.array(6, dtype=np.int64),
        h=np.empty((0,), dtype=np.float64),
        J_idx=np.empty((0, 2), dtype=np.int32),
        J_val=np.empty((0,), dtype=np.float64),
        K_idx=np.empty((0, 3), dtype=np.int32),
        K_val=np.empty((0,), dtype=np.float64),
    )
    m, off, meta = EnergyModel.load(p)
    assert m.n_spins == 6 and m.h is None and m.J_idx is None and m.K_idx is None
    assert off is None and isinstance(meta, dict) and meta == {}
    assert m.name is None


def test_save_with_offset_none_roundtrips_to_none(tmp_path):
    m = EnergyModel(n_spins=2, h=np.array([0.1, -0.2]))
    p = tmp_path / "none_off.npz"
    m.save(p, offset=None)
    _, off2, _ = EnergyModel.load(p)
    assert off2 is None
