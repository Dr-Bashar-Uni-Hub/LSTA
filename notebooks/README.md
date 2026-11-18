# Notebooks (Google Colab, GPU)

The notebooks expect Google Drive mounted at `/content/gdrive` with the project folder `MyDrive/Benchmark_V5/`.

| Notebook | Purpose |
|---|---|
| `legacy/00_data_preparation_original_study.ipynb` | Earlier development notebook: OPSD loading, preprocessing, and the original single-horizon study (results superseded) |
| `01_benchmark_reliability_pipeline.ipynb` | Main pipeline: frozen datasets, five-model benchmark at 1/6/24 h, factorial ablation, persistence baselines, statistical tests, reliability experiments (outputs retained as a record) |
| `02_stageA_export_and_reproduction.ipynb` | Read-only export of all saved predictions and reproduction of the published Tables 5, 6, 9 |
| `03_stageB_fallback_inference.ipynb` | Post hoc inference: gaps of 1–24 h incl. lead-matched 5 h and 23 h, forward vs seasonal fill, reproduction gates |
| `04_stageC_uncapped_retraining.ipynb` | Post hoc training-budget correction for PatchTST and DLinear (300-epoch ceiling) with replay checks |
| `06_stageE_matched_factorial.ipynb` | Post hoc factorial rerun with matched encoder blocks (MTL-GAP, MTL-LSTA; 30 runs) and cross-version reproducibility probe |
| `05_stageD_austria_replication.ipynb` | Post hoc external replication on Austria: German dataset-rebuild gate, 90 training runs, missing-input inference |

Notebooks 02–05 write only to `Benchmark_V5/IJGE_Revision/` and never modify frozen artifacts.
