# Fallback Strategy Matters More Than Architecture

Code, saved predictions, and reproduction material for the article:

> Bashar ALHAJAHMAD. *Fallback Strategy Matters More Than Architecture: A Reliability-Oriented Evaluation of Deep Learning Forecasters for Solar Generation and Electrical Load.* Submitted to the *International Journal of Green Energy* (2026).

The study compares LSTM, single- and multi-task Transformers (ST, MTL-LSTA), PatchTST, and DLinear for German electrical load and solar generation (Open Power System Data) at 1-, 6-, and 24-hour horizons across five training seeds. It adds statistical tie sets, a 2 × 2 factorial ablation of parameter sharing and temporal aggregation, and condition-matched fallback analyses for missing recent inputs.

## Repository layout

```
analysis/                  Offline (CPU) scripts that regenerate every number reported in the paper
  common.py                Paths, prediction loaders, Diebold–Mariano and Holm utilities
  01_reproduce_frozen.py   Reproduces the original Tables 5, 6, 9 and 300 seed-level DM statistics
  02_benchmark_and_statistics.py  Corrected benchmark (Table 5), skill (Table 7), tie sets (Table 8), Figures 3–4
  03_factorial_exact.py    Factorial contrasts with exact t-intervals and sign-flip tests (Table S9, Figure 5)
  04_degradation_fallback.py      Condition-matched degradation, lead-matched fallback (Table 10), stale-value mechanism
  run_all.py               Runs all of the above
notebooks/                 Google Colab notebooks (GPU) that trained the models and produced the saved predictions
data/predictions/          Saved test-set predictions in MW (inputs to analysis/)
results/
  tables/                  Outputs written by analysis/
  supplementary_tables/    Supplementary Tables S1–S16 (CSV)
  reproduction/            Reproduction gates, training histories of the corrected runs, frozen reference results
  figures/                 Figures written by analysis/
```

## Quick start (no GPU required)

```bash
git clone https://github.com/Dr-Bashar-Uni-Hub/LSTA.git
cd LSTA
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python analysis/run_all.py
```

`run_all.py` first verifies that the saved predictions reproduce the published original results (the check stops with an error otherwise). It then regenerates the corrected benchmark, the statistical tests, the factorial statistics, and the fallback analyses in `results/tables/` and `results/figures/`. The run takes about 1–2 minutes on a laptop.

## Full pipeline (GPU)

The models were trained in Google Colab (NVIDIA A100) with TensorFlow/Keras (LSTM, Transformers) and PyTorch (PatchTST, DLinear). See `notebooks/README.md` for the order of execution and the expected Google Drive layout.

## Post hoc analyses and reproducibility notes

Analyses added after the original protocol was fixed are labelled *post hoc* in the paper. They comprise the training-budget correction, seed-pooled and seed-ensemble tests, condition-matched and lead-matched fallback comparisons, and the exact factorial statistics.

- **Training-budget correction.** PatchTST (6, 24 h) and DLinear (all horizons) originally reached a 50-epoch limit while still improving. They were retrained with a 300-epoch ceiling and are otherwise identical (`notebooks/04_stageC_uncapped_retraining.ipynb`). DLinear replays the original first 50 epochs exactly. PatchTST is not bit-reproducible on GPU (non-deterministic attention kernels); same-seed reruns differ by up to 48 MW RMSE.
- **TensorFlow version effect.** TensorFlow checkpoints re-inferred under TensorFlow 2.21 / Keras 3.13 match the original predictions at the metric level (maximum RMSE difference 0.19 MW), not bit-exactly. PyTorch models match to within 0.01 MW. See `results/reproduction/`.

## Data

Raw data: Open Power System Data, *Data Package Time Series*, version 2020-10-06, https://doi.org/10.25832/time_series/2020-10-06. See `data/README.md` for details and the SHA-256 checksum of the processed one-hour dataset.

## Citation

Please cite the article (see `CITATION.cff`) and the Open Power System Data package.

## License

Code: MIT License (see `LICENSE`). The redistributed data arrays are derived from Open Power System Data and remain subject to the terms of the original sources.
