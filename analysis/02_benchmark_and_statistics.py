"""Corrected benchmark (Table 5), skill (Table 7), seed-pooled tie sets (Table 8), seed ensembles, Figures 3-4."""
from itertools import combinations
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

rows = []
for h in HORIZONS:
    for t, i in TARGETS.items():
        y = y_true(h)[:, i]
        for m in BENCH + ["ST-LSTA", "MTL-GAP"]:
            r = np.array([rmse(corrected(h, m, s)[:, i] - y) for s in SEEDS])
            ens = rmse(np.mean([corrected(h, m, s)[:, i] for s in SEEDS], 0) - y)
            rows.append(dict(h=h, Target=t, Model=m, RMSE_mean=r.mean(), RMSE_sd=r.std(ddof=1), nRMSE_pct=100 * r.mean() / y.mean(),
                             nRMSE_sd_pct=100 * (r / y.mean()).std(ddof=1), Ensemble_RMSE=ens))
T = out("table5_corrected_benchmark.csv", pd.DataFrame(rows))

best_persist = {(1, "Load"): 2391.092, (1, "Solar"): 2583.798, (6, "Load"): 3777.580, (6, "Solar"): 2633.191, (24, "Load"): 3777.580, (24, "Solar"): 2633.191}
sk = []
for (h, t), b in best_persist.items():
    d = T[(T.h == h) & (T.Target == t) & T.Model.isin(BENCH)].sort_values("RMSE_mean").iloc[0]
    sk.append([h, t, d.Model, d.RMSE_mean, b, 100 * (1 - d.RMSE_mean / b)])
out("table7_skill.csv", pd.DataFrame(sk, columns=["h", "Target", "Best model", "RMSE", "Best persistence RMSE", "Skill %"]))

tied, pairs = [], []
for h in HORIZONS:
    for t, i in TARGETS.items():
        y = y_true(h)[:, i]
        L = {m: np.mean([(corrected(h, m, s)[:, i] - y) ** 2 for s in SEEDS], 0) for m in BENCH}
        best = min(BENCH, key=lambda m: L[m].mean()); others = [m for m in BENCH if m != best]
        adj = holm([dm_test(L[m] - L[best])[1] for m in others])
        tied.append([h, t, best, ", ".join([best] + [m for m, a in zip(others, adj) if a >= 0.05])])
        sub = [[h, t, a, b, *dm_test(L[a] - L[b])] for a, b in combinations(BENCH, 2)]
        sub = pd.DataFrame(sub, columns=["h", "Target", "A", "B", "DM", "p"]); sub["p_holm"] = holm(sub.p); pairs.append(sub)
out("table8_tied_sets.csv", pd.DataFrame(tied, columns=["h", "Target", "Lowest-RMSE model", "Statistically tied set"]))
out("pairwise_seed_pooled_DM.csv", pd.concat(pairs))

M = BENCH; C = dict(zip(M, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"])); lab = lambda m: "ST" if m == "ST-GAP" else m
fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.7))
for k, t in enumerate(TARGETS):
    for m in M:
        d = T[(T.Model == m) & (T.Target == t)].sort_values("h")
        ax[k].errorbar(HORIZONS, d.nRMSE_pct, yerr=d.nRMSE_sd_pct, marker="o", capsize=3, lw=2, color=C[m], label=lab(m))
    ax[k].set_xticks(HORIZONS); ax[k].set_xlabel("Forecast horizon (hours)"); ax[k].set_title(f"{t} forecasting"); ax[k].grid(alpha=.3)
ax[0].set_ylabel("nRMSE (% of observed test mean)"); ax[1].legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout(); fig.savefig(RES / "figures" / "figure3_normalized_error.png", dpi=150)
fig, ax = plt.subplots(1, 2, figsize=(12, 5.6)); x = np.arange(5)
for k, t in enumerate(TARGETS):
    p = T[T.Target == t].pivot_table(index="Model", columns="h", values="RMSE_mean").loc[M]
    ax[k].bar(x - .18, 100 * (p[6] / p[1] - 1), .36, label="6 h"); ax[k].bar(x + .18, 100 * (p[24] / p[1] - 1), .36, label="24 h")
    ax[k].set_xticks(x); ax[k].set_xticklabels([lab(m) for m in M], rotation=20); ax[k].set_ylabel("RMSE increase vs. 1 h (%)"); ax[k].set_title(t)
ax[0].legend(frameon=False); fig.tight_layout(); fig.savefig(RES / "figures" / "figure4_relative_growth.png", dpi=150)
print(pd.DataFrame(tied, columns=["h", "Target", "Best", "Tied set"]).to_string(index=False))
