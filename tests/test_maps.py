import numpy as np

from embmigrate.maps import fit


def _planted(d_b, d_a, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    za = rng.standard_normal((n, d_a)).astype(np.float32)
    za /= np.linalg.norm(za, axis=1, keepdims=True)
    Q, _ = np.linalg.qr(rng.standard_normal((max(d_a, d_b), max(d_a, d_b))))
    return za, Q


def test_procrustes_recovers_planted_rotation():
    za, Q = _planted(64, 64)
    zb = za @ Q  # B = rotated A
    m = fit(zb, za, "procrustes")
    # B-space rotation is recovered exactly (centering cancels since rotation is linear)
    assert np.abs(m.W - Q.T).max() < 1e-4
    cos = (m.apply(zb) * (za / np.linalg.norm(za, axis=1, keepdims=True))).sum(1)
    assert cos.min() > 0.999


def test_semi_orthogonal_columns_orthonormal():
    rng = np.random.default_rng(1)
    zb, za = rng.standard_normal((3000, 32)), rng.standard_normal((3000, 64))
    m = fit(zb, za, "procrustes")
    assert m.W.shape == (32, 64)
    assert np.abs(m.W @ m.W.T - np.eye(32)).max() < 1e-4


def test_ridge_and_mlp_run():
    za, Q = _planted(32, 32, n=1000)
    zb = za @ Q
    r = fit(zb, za, "ridge")
    assert hasattr(r, "lam") and (r.apply(zb) * za).sum(1).mean() > 0.95
    assert fit(zb, za, "mlp").apply(zb).shape == za.shape
    assert fit(zb[:, :16], za, "identity").apply(zb[:, :16]).shape == za.shape
