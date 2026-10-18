"""Shared paths, loaders, and statistics for the offline analyses (CPU only).

All inputs are saved test-set predictions in MW (see data/README.md).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "predictions"
RES = ROOT / "results"
SEEDS = [1, 7, 21, 42, 100]
HORIZONS = [1, 6, 24]
TARGETS = {"Load": 0, "Solar": 1}
BENCH = ["LSTM", "ST-GAP", "MTL-LSTA", "PatchTST", "DLinear"]

_Z = np.load(PRED / "frozen_predictions_all_MW.npz")


def y_true(h):
    """Observed test targets (6,576 x 2: load, solar) for horizon h; identical target timestamps across horizons."""
    return _Z[f"y_true_h{h}"]


def history_1h():
    """One-hour input windows (6,576 x 168 x 2, MW); index -1 is the hour before the target."""
    return _Z["X_dyn_h1_MW"].astype(float)


def frozen(h, model, seed):
    """Frozen (original protocol) predictions."""
    return _Z[f"h{h}__{model}__s{seed}"]


def corrected(h, model, seed):
    """Corrected benchmark: PatchTST (6/24 h) and DLinear (all h) from the uncapped Stage C runs; all others frozen."""
    if (model == "DLinear") or (model == "PatchTST" and h in (6, 24)):
        z = np.load(PRED / "stageC" / f"h{h:02d}_seed{seed:03d}_{model}_test_predictions_MW.npz")
        return z["pred_uncapped"]
    return frozen(h, model, seed)


_B = None


def degraded(model, seed, condition):
    """Stage B one-hour predictions under 'Clean', 'FFill_<g>h', or 'SeasonalFill_<g>h' (g in 1,3,5,6,12,23,24)."""
    global _B
    if _B is None:
        _B = np.load(PRED / "stageB_degraded_predictions_MW.npz")
    return _B[f"{model}__s{seed}__{condition}"].astype(float)


def rmse(e):
    return float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))


def dm_test(d, lag=24):
    """Diebold-Mariano statistic for loss differential d with a Bartlett-kernel Newey-West long-run variance."""
    d = np.asarray(d, float); n = len(d); u = d - d.mean()
    lrv = u @ u / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    t = d.mean() / np.sqrt(lrv / n)
    return t, 2 * stats.norm.sf(abs(t))


def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


def out(name, df):
    path = RES / "tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(ROOT)}")
    return df
