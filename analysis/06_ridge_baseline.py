"""Post hoc ridge regression reference (Germany and Austria, 1/6/24 h).
Requires the raw OPSD file data/raw/time_series_60min_singleindex.csv (release 2020-10-06; see data/README.md).
Settings are selected on 2019 only; the model is fitted on 2015-2018 and evaluated once on the test period.
Targets are checked to be identical to the saved targets of the neural models."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ridge'))
from common import *
from builder import prepare_frame, build_horizon
from ridge import fit_select, predict
raw = pd.read_csv(ROOT / 'data/raw/time_series_60min_singleindex.csv', low_memory=False,
                  usecols=['utc_timestamp', 'DE_load_actual_entsoe_transparency', 'DE_solar_generation_actual', 'AT_load_actual_entsoe_transparency', 'AT_solar_generation_actual'])
ZA = np.load(PRED / 'austria' / 'AT_predictions_all_MW.npz'); rows = []; preds = {}
for cc, cols in [('DE', ('DE_load_actual_entsoe_transparency', 'DE_solar_generation_actual')), ('AT', ('AT_load_actual_entsoe_transparency', 'AT_solar_generation_actual'))]:
    P = prepare_frame(raw, *cols)
    for h in HORIZONS:
        d, _, _ = build_horizon(P, h); mean, scale = P['mean'], P['scale']; yte = d['y_test'] * scale + mean
        saved = y_true(h) if cc == 'DE' else ZA[f'y_true_h{h}']
        assert np.max(np.abs(yte - saved)) < 1e-3, 'targets not aligned'
        pr = np.zeros_like(yte)
        for s, t in [(0, 'Load'), (1, 'Solar')]:
            best, _ = fit_select(d, h, s, mean, scale); pr[:, s] = predict(d, 'test', h, s, best, mean, scale)
            rows.append([cc, h, t, best[0], best[1], best[2], rmse(pr[:, s] - yte[:, s])])
        preds[f'{cc}_h{h}'] = pr; preds[f'{cc}_y_h{h}'] = yte
out('ridge_results_reproduced.csv', pd.DataFrame(rows, columns=['country', 'h', 'target', 'features', 'alpha', 'val_RMSE', 'test_RMSE']))
print(pd.DataFrame(rows).round(3).to_string(index=False))
