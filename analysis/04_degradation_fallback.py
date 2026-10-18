"""Condition-matched degradation analysis, lead-matched fallback comparison (Table 10), and stale-value mechanism."""
from common import *

X = history_1h(); y1 = y_true(1)
frozen_rel = pd.read_csv(RES / "reproduction" / "frozen_reliability_all_conditions.csv")
# Random masking: last-observation-carried-forward (LOCF) and seasonal fallbacks
masks = np.load(PRED / "random_degradation_masks.npz"); n = len(y1); rows = []
for lvl in [10, 30, 50]:
    for rep in range(1, 6):
        m = masks[f"replicate_{rep}_level_{lvl}"]; valid = ~m
        last = np.where(valid, np.arange(168)[None, :], -1).max(1)
        for t, i in TARGETS.items():
            locf = X[np.arange(n), np.clip(last, 0, None), i]
            seas = np.where(valid[:, -24], X[:, -24, i], np.where(valid[:, -168], X[:, -168, i], locf))
            rows.append([lvl, rep, t, rmse(locf - y1[:, i]), rmse(seas - y1[:, i])])
Rm = pd.DataFrame(rows, columns=["Level", "Rep", "Target", "LOCF", "Seasonal fallback"]).groupby(["Target", "Level"])[["LOCF", "Seasonal fallback"]].mean()
mod = frozen_rel[frozen_rel.Condition.str.startswith("RandomMean")].groupby(["Target", "Level_or_gap", "Model"]).RMSE_MW.mean().unstack()
Rm["Best model"] = mod.min(axis=1).values; Rm["Worst model"] = mod.max(axis=1).values
out("tableS8b_random_masking.csv", Rm.reset_index())

# Contiguous gaps and lead-matched fallback
gap_rows, fb_rows = [], []
for t, i in TARGETS.items():
    y = y1[:, i]
    for g in [1, 3, 5, 6, 12, 23, 24]:
        r = {"Target": t, "Gap h": g, "Lead h": g + 1}
        for imp in ["FFill", "SeasonalFill"]:
            v = [np.mean([rmse(degraded(m, s, f"{imp}_{g}h")[:, i] - y) for s in SEEDS]) for m in BENCH]
            r[f"{imp} best"], r[f"{imp} worst"] = min(v), max(v)
        r["Stale value"] = rmse(X[:, -(g + 1), i] - y); r["Daily"] = rmse(X[:, -24, i] - y) if g <= 23 else np.nan; r["Weekly"] = rmse(X[:, -168, i] - y)
        gap_rows.append(r)
    for g, h in [(5, 6), (23, 24)]:
        L = lambda f: {m: np.mean([(f(m, s)[:, i] - y) ** 2 for s in SEEDS], 0) for m in BENCH}
        ff = L(lambda m, s: degraded(m, s, f"FFill_{g}h")); sf = L(lambda m, s: degraded(m, s, f"SeasonalFill_{g}h")); d = L(lambda m, s: corrected(h, m, s))
        sq = lambda a: np.sqrt(a.mean()); bff = min(ff.values(), key=np.mean); bsf = min(sf.values(), key=np.mean)
        bd = min(d.values(), key=np.mean); wd = max(d.values(), key=np.mean)
        # ranges: five-seed mean of per-seed RMSE (as in the manuscript); DM tests and reductions: seed-pooled losses
        mr = lambda f: [np.mean([rmse(f(m, s)[:, i] - y) for s in SEEDS]) for m in BENCH]
        rf, rs, rd = mr(lambda m, s: degraded(m, s, f"FFill_{g}h")), mr(lambda m, s: degraded(m, s, f"SeasonalFill_{g}h")), mr(lambda m, s: corrected(h, m, s))
        fb_rows.append([t, h, f"{min(rf):.0f}-{max(rf):.0f}", f"{min(rs):.0f}-{max(rs):.0f}", f"{min(rd):.0f}-{max(rd):.0f}", dm_test(wd - bff)[0], dm_test(wd - bsf)[0], dm_test(bd - bsf)[0], dm_test(bd - bsf)[1],
                        100 * (1 - sq(bd) / sq(bff)), 100 * (1 - sq(bd) / sq(bsf))])
out("tableS8a_contiguous_gaps.csv", pd.DataFrame(gap_rows))
FB = out("table10_lead_matched_fallback.csv", pd.DataFrame(fb_rows, columns=["Target", "Lead h", "1-h FFill range", "1-h SeasonalFill range", "Direct range",
        "DM worst direct vs best FFill", "DM worst direct vs best SeasonalFill", "DM best direct vs best SeasonalFill", "p best direct vs best SeasonalFill",
        "Reduction vs best FFill %", "Reduction vs best SeasonalFill %"]))

# Stale-value mechanism (forward fill): regress seed-ensemble degraded forecast on filled value and target
mech = []
for t, i in TARGETS.items():
    y = y1[:, i]
    for g in [1, 3, 5, 6, 12, 23, 24]:
        filled = X[:, -(g + 1), i]; w, ratio = [], []
        for m in BENCH:
            p = np.mean([degraded(m, s, f"FFill_{g}h")[:, i] for s in SEEDS], 0)
            c, *_ = np.linalg.lstsq(np.c_[filled, y, np.ones_like(y)], p, rcond=None)
            w.append(c[0] / (c[0] + c[1])); ratio.append(rmse(p - filled) / rmse(p - y))
        mech.append([t, g, min(w), max(w), float(np.median(ratio))])
out("tableS8c_stale_value_mechanism.csv", pd.DataFrame(mech, columns=["Target", "Gap h", "Echo weight min", "Echo weight max", "Median distance ratio"]))
print(FB.round(1).to_string(index=False))
