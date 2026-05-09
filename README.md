# CMSC-170---ML-Project
In fulfillment of the CMSC 170 requirements for a machine learning (Baye's) problem

## Getting Started
biological_data_dictionary.md contains interpretation of the data
features_name.json contains features of X_train.csv
```bash
pip install pyyaml pandas numpy scikit-learn joblib tqdm 
python src/data/preprocess.py
python src/data/export_simple_summary.py
  
```

```bash

```

## Role 3: Bayesian Model (ZINB + Horseshoe)

Run the implementation:

```bash
python src/model/train_zinb.py
```

Optional quick-run flags:

```bash
python src/model/train_zinb.py --sample-size 2000 --draws 500 --tune 500 --chains 2 --cores 2
```

Outputs are saved to `results/role3_model/`:
- `zinb_trace.nc`
- `zinb_trace_with_ppc.nc`
- `rhat_diagnostics.csv`
- `ess_diagnostics.csv`
- `model_metrics.json` (LOO/WAIC)
- `ppc_summary.csv`
- `ppc_plot.png`
- `feature_effects.csv`

### Baseline Models for Comparison (NB and ZIP)

```bash
python src/model/train_nb_zip_baselines.py
```

Optional faster run:

```bash
python src/model/train_nb_zip_baselines.py --sample-size 2000 --draws 400 --tune 400 --chains 2 --cores 2
```

Outputs are saved to `results/role3_model_baselines/`:
- `nb_trace.nc`
- `zip_trace.nc`
- `baseline_model_comparison.csv`
- `baseline_model_comparison.json`

### Unified Comparison (ZINB vs NB vs ZIP)

After running both scripts above, generate one ranked comparison table:

```bash
python src/model/compare_all_models.py
```

Outputs are saved to `results/role3_comparison/`:
- `all_model_comparison.csv`
- `all_model_comparison.json`
- `comparison_summary.json`
