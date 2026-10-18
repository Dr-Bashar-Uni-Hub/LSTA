"""2x2 factorial contrasts with exact t-intervals and sign-flip tests (Table S9, Figure 5)."""
from itertools import product
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

def signflip(d):
    d = np.asarray(d); obs = abs(d.mean())
    return np.mean([abs((d * np.array(s)).mean()) >= obs - 1e-12 for s in product([-1, 1], repeat=len(d))])

rows = []
for h in HORIZONS:
    for t, i in TARGETS.items():
        y = y_true(h)[:, i]
        R = {m: np.array([rmse(frozen(h, m, s)[:, i] - y) for s in SEEDS]) for m in ["ST-GAP", "ST-LSTA", "MTL-GAP", "MTL-LSTA"]}
        C = {"LSTA-GAP | ST": R["ST-LSTA"] - R["ST-GAP"], "LSTA-GAP | MTL": R["MTL-LSTA"] - R["MTL-GAP"],
             "MTL-ST | GAP": R["MTL-GAP"] - R["ST-GAP"], "MTL-ST | LSTA": R["MTL-LSTA"] - R["ST-LSTA"],
             "Interaction": (R["MTL-LSTA"] - R["MTL-GAP"]) - (R["ST-LSTA"] - R["ST-GAP"])}
        for k, d in C.items():
            lo, hi = stats.t.interval(.95, 4, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(5))
            rows.append([h, t, k, d.mean(), lo, hi, f"{max((d < 0).sum(), (d > 0).sum())}/5", signflip(d)])
F = out("tableS9_factorial_exact.csv", pd.DataFrame(rows, columns=["h", "Target", "Contrast", "Mean MW", "t95 low", "t95 high", "Same sign", "p sign-flip"]))
I = F[F.Contrast == "Interaction"]
fig, ax = plt.subplots(1, 2, figsize=(13.9, 5.4))
for k, t in enumerate(TARGETS):
    d = I[I.Target == t].sort_values("h"); m = d["Mean MW"].values
    ax[k].errorbar(range(3), m, yerr=[m - d["t95 low"].values, d["t95 high"].values - m], fmt="o", capsize=5, ms=8)
    ax[k].axhline(0, color="k", ls="--", lw=1); ax[k].set_xticks(range(3)); ax[k].set_xticklabels(["1 h", "6 h", "24 h"])
    ax[k].set_ylabel("Interaction I (MW)"); ax[k].set_title(f"({'ab'[k]}) {t}"); ax[k].grid(alpha=.3)
fig.tight_layout(); fig.savefig(RES / "figures" / "figure5_interaction.png", dpi=150)
print(I.round(2).to_string(index=False))
