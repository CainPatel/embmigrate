"""Maps from embedding space B to space A, all fit on paired passage embeddings."""
import numpy as np
import torch

RIDGE_LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]  # 1e-4..1, plus 10 and 100 so small k isn't under-regularized


def _norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


class Map:
    """x_b -> center(mu_b) -> L2 -> f -> + mu_a -> L2  (the index side stays native)."""

    def __init__(self, kind, mu_b, mu_a, W=None, mlp=None, d_a=None):
        self.kind, self.mu_b, self.mu_a, self.W, self.mlp, self.d_a = kind, mu_b, mu_a, W, mlp, d_a

    def apply(self, xb):
        xb = xb.astype(np.float32)
        if self.kind == "identity":  # floor baseline: zero-pad / truncate raw B, no fitting
            out = np.zeros((len(xb), self.d_a), np.float32)
            d = min(self.d_a, xb.shape[1])
            out[:, :d] = xb[:, :d]
            return _norm(out)
        z = _norm(xb - self.mu_b)
        if self.mlp is not None:
            with torch.no_grad():
                z = self.mlp(torch.from_numpy(z)).numpy()
        else:
            z = z @ self.W
        return _norm(_norm(z) + self.mu_a)


def fit(xb, xa, kind, seed=0):
    xb, xa = xb.astype(np.float32), xa.astype(np.float32)
    if kind == "identity":
        return Map(kind, None, None, d_a=xa.shape[1])
    mu_b, mu_a = xb.mean(0), xa.mean(0)
    zb, za = _norm(xb - mu_b), _norm(xa - mu_a)
    if kind == "procrustes":  # orthogonal if d_a == d_b, else orthonormal rows/cols (semi-orthogonal)
        U, _, Vt = np.linalg.svd(za.T @ zb, full_matrices=False)
        return Map(kind, mu_b, mu_a, W=(U @ Vt).T)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(zb))
    n_hold = max(1, len(zb) // 10)
    hold, tr = idx[:n_hold], idx[n_hold:]
    if kind == "ridge":
        G, C = zb[tr].T @ zb[tr], zb[tr].T @ za[tr]
        best = min(RIDGE_LAMBDAS, key=lambda lam: ((zb[hold] @ np.linalg.solve(G + lam * np.eye(len(G)), C) - za[hold]) ** 2).sum())
        W = np.linalg.solve(zb.T @ zb + best * np.eye(zb.shape[1]), zb.T @ za)
        m = Map(kind, mu_b, mu_a, W=W)
        m.lam = best
        return m
    if kind == "mlp":
        return Map(kind, mu_b, mu_a, mlp=_fit_mlp(zb[tr], za[tr], zb[hold], za[hold], seed))
    raise ValueError(kind)


def _fit_mlp(xtr, ytr, xho, yho, seed, epochs=20, bs=256):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], 1024), torch.nn.GELU(), torch.nn.Linear(1024, ytr.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    xtr, ytr, xho, yho = map(torch.from_numpy, (xtr, ytr, xho, yho))

    def loss_fn(p, y):
        return torch.nn.functional.mse_loss(p, y) + (1 - torch.nn.functional.cosine_similarity(p, y)).mean()

    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), bs):
            b = perm[i : i + bs]
            opt.zero_grad()
            loss_fn(net(xtr[b]), ytr[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            h = loss_fn(net(xho), yho).item()
        if h < best:
            best, best_state, bad = h, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 3:
                break
    net.load_state_dict(best_state)
    return net.eval()
