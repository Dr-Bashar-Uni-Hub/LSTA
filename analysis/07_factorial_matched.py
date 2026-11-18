"""Post hoc factorial with matched encoder blocks (Table 9, Figure 5, Supplementary Tables S9a-S9b).
ST arms: original runs; MTL arms: matched-block reruns (notebooks/06_stageE_matched_factorial.ipynb)."""
from itertools import product
from common import *
def light(h, arm, s): return np.load(PRED / 'stageE' / f'h{h:02d}_seed{s:03d}_{arm}_test_predictions_MW.npz')['pred_MW']
SRC = {'ST-GAP': lambda h, s: frozen(h, 'ST-GAP', s), 'ST-LSTA': lambda h, s: frozen(h, 'ST-LSTA', s),
       'MTL-GAP': lambda h, s: light(h, 'MTL-GAP', s), 'MTL-LSTA': lambda h, s: light(h, 'MTL-LSTA', s),
       'MTL-GAP original': lambda h, s: frozen(h, 'MTL-GAP', s), 'MTL-LSTA original': lambda h, s: frozen(h, 'MTL-LSTA', s)}
def signflip(d):
    d = np.asarray(d); o = abs(d.mean()); return np.mean([abs((d * np.array(s)).mean()) >= o - 1e-12 for s in product([-1, 1], repeat=len(d))])
tab, con = [], []
for h in HORIZONS:
    for t, i in TARGETS.items():
        y = y_true(h)[:, i]; R = {k: np.array([rmse(f(h, s)[:, i] - y) for s in SEEDS]) for k, f in SRC.items()}
        tab.append([h, t] + [f'{R[k].mean():.1f} ± {R[k].std(ddof=1):.1f}' for k in SRC])
        I = (R['MTL-LSTA'] - R['MTL-GAP']) - (R['ST-LSTA'] - R['ST-GAP'])
        for name, d in [('LSTA-GAP | ST', R['ST-LSTA'] - R['ST-GAP']), ('LSTA-GAP | MTL', R['MTL-LSTA'] - R['MTL-GAP']),
                        ('MTL-ST | GAP', R['MTL-GAP'] - R['ST-GAP']), ('MTL-ST | LSTA', R['MTL-LSTA'] - R['ST-LSTA']), ('Interaction', I)]:
            lo, hi = stats.t.interval(.95, 4, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(5)); con.append([h, t, name, d.mean(), lo, hi, signflip(d)])
out('table9_factorial_matched.csv', pd.DataFrame(tab, columns=['h', 'Target'] + list(SRC)))
C = out('factorial_matched_contrasts.csv', pd.DataFrame(con, columns=['h', 'Target', 'Contrast', 'Mean MW', 't95 low', 't95 high', 'p sign-flip']))
print(C[C.Contrast == 'Interaction'].round(1).to_string(index=False))
