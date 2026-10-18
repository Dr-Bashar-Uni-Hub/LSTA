"""Reproduce the original (frozen) Tables 5, 6, 9 and the 300 seed-level DM statistics from saved predictions."""
from common import *

PUBLISHED_T5 = {  # original manuscript values (mean RMSE, MW)
    (1, "Load"): [818.183, 786.593, 708.822, 697.373, 760.031], (1, "Solar"): [469.731, 524.389, 541.201, 540.160, 489.307],
    (6, "Load"): [2116.388, 1597.430, 1636.578, 1549.840, 2447.955], (6, "Solar"): [2342.239, 2173.919, 2271.276, 2003.036, 2621.256],
    (24, "Load"): [2923.848, 2788.721, 2827.920, 2374.515, 3124.898], (24, "Solar"): [2824.578, 2455.216, 2620.557, 2442.405, 2774.535]}
PUBLISHED_T6 = {(1, "Load"): (2391.092, 6161.737, 3777.580), (1, "Solar"): (2583.798, 2633.191, 4010.645),
                (6, "Load"): (10708.304, 6161.737, 3777.580), (6, "Solar"): (12149.543, 2633.191, 4010.645)}
rows = []
for (h, t), vals in PUBLISHED_T5.items():
    y = y_true(h)[:, TARGETS[t]]
    for m, pub in zip(BENCH, vals):
        r = np.mean([rmse(frozen(h, m, s)[:, TARGETS[t]] - y) for s in SEEDS])
        rows.append(["Table 5 (original)", h, t, m, r, pub, r - pub])
X = history_1h()
for (h, t), (op, dp, wp) in PUBLISHED_T6.items():
    i = TARGETS[t]; y = y_true(h)[:, i]
    for name, pred, pub in [("Ordinary", X[:, -h, i], op), ("Daily", X[:, -24, i], dp), ("Weekly", X[:, -168, i], wp)]:
        r = rmse(pred - y); rows.append(["Table 6", h, t, name + " persistence", r, pub, r - pub])
df = pd.DataFrame(rows, columns=["Table", "h", "Target", "Model", "Reproduced", "Published", "Difference"])
print(df.round(4).to_string(index=False))
assert df.Difference.abs().max() < 0.01, "reproduction failed"
s3 = pd.read_csv(RES / "reproduction" / "frozen_S3_DM_HAC_300_tests.csv"); worst = 0
for _, r in s3.iterrows():
    y = y_true(r["Horizon (h)"])[:, TARGETS[r["Target"]]]; i = TARGETS[r["Target"]]; s = int(r["Seed"])
    d = (frozen(r["Horizon (h)"], r["Model A"], s)[:, i] - y) ** 2 - (frozen(r["Horizon (h)"], r["Model B"], s)[:, i] - y) ** 2
    worst = max(worst, abs(dm_test(d)[0] - r["DM-HAC statistic"]))
print(f"\nMax |DM statistic difference| over {len(s3)} frozen tests: {worst:.2e}")
assert worst < 1e-6
out("reproduction_frozen_tables.csv", df)
print("PASS: frozen results reproduced")
