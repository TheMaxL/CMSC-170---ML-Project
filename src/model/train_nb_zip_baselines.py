import argparse
import json
import os
from pathlib import Path

_cache_root = Path("results") / ".cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_cache_root / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(_cache_root / "xdg")
os.environ["LOCALAPPDATA"] = str(_cache_root / "localappdata")

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


def parse_args():
    parser = argparse.ArgumentParser(description="Train NB and ZIP Bayesian baselines.")
    parser.add_argument("--x-path", default="data/processed/X_train.csv")
    parser.add_argument("--y-path", default="data/processed/y_train.csv")
    parser.add_argument("--output-dir", default="results/role3_model_baselines")
    parser.add_argument("--draws", type=int, default=800)
    parser.add_argument("--tune", type=int, default=800)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(x_path: Path, y_path: Path):
    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    y = y_df["tm_helix_count"].to_numpy(dtype=int) if "tm_helix_count" in y_df.columns else y_df.iloc[:, 0].to_numpy(dtype=int)
    return x_df.to_numpy(dtype=float), y


def maybe_subsample(x, y, sample_size, seed):
    if sample_size and sample_size > 0:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=min(sample_size, len(y)), replace=False)
        return x[idx], y[idx]
    return x, y


def fit_nb(x, y, args, out_dir: Path):
    n, p = x.shape
    with pm.Model() as model:
        beta = pm.Normal("beta", 0.0, 1.0, shape=p)
        alpha = pm.Normal("alpha", 0.0, 2.0)
        mu = pm.math.exp(alpha + pm.math.dot(x, beta))
        alpha_nb = pm.HalfNormal("alpha_nb", 2.0)
        pm.NegativeBinomial("tm_count_obs", mu=mu, alpha=alpha_nb, observed=y)
        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            random_seed=args.seed,
            target_accept=args.target_accept,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )
    az.to_netcdf(idata, out_dir / "nb_trace.nc")
    return idata


def fit_zip(x, y, args, out_dir: Path):
    n, p = x.shape
    with pm.Model() as model:
        beta_count = pm.Normal("beta_count", 0.0, 1.0, shape=p)
        alpha_count = pm.Normal("alpha_count", 0.0, 2.0)
        mu = pm.math.exp(alpha_count + pm.math.dot(x, beta_count))

        beta_zi = pm.Normal("beta_zi", 0.0, 1.0, shape=p)
        alpha_zi = pm.Normal("alpha_zi", 0.0, 2.0)
        psi = pm.math.sigmoid(alpha_zi + pm.math.dot(x, beta_zi))

        pm.ZeroInflatedPoisson("tm_count_obs", psi=psi, mu=mu, observed=y)
        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            random_seed=args.seed,
            target_accept=args.target_accept,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )
    az.to_netcdf(idata, out_dir / "zip_trace.nc")
    return idata


def summarize_model(idata, name: str):
    loo = az.loo(idata)
    waic = az.waic(idata)
    return {
        "model": name,
        "elpd_loo": float(loo.elpd_loo),
        "p_loo": float(loo.p_loo),
        "loo_se": float(loo.se),
        "elpd_waic": float(waic.elpd_waic),
        "p_waic": float(waic.p_waic),
        "waic_se": float(waic.se),
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y = load_data(Path(args.x_path), Path(args.y_path))
    x, y = maybe_subsample(x, y, args.sample_size, args.seed)

    nb_idata = fit_nb(x, y, args, out_dir)
    zip_idata = fit_zip(x, y, args, out_dir)

    nb_stats = summarize_model(nb_idata, "NB")
    zip_stats = summarize_model(zip_idata, "ZIP")

    comparison = pd.DataFrame([nb_stats, zip_stats]).sort_values("elpd_loo", ascending=False)
    comparison.to_csv(out_dir / "baseline_model_comparison.csv", index=False)
    with open(out_dir / "baseline_model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison.to_dict(orient="records"), f, indent=2)

    print("Baseline training complete.")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
