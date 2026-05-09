import json
import argparse
import os
from pathlib import Path

# Keep package caches inside the project workspace to avoid permission issues.
_cache_root = Path("results") / ".cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_cache_root / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(_cache_root / "xdg")
os.environ["LOCALAPPDATA"] = str(_cache_root / "localappdata")

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


def load_training_data(x_path: Path, y_path: Path):
    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)

    if "tm_helix_count" in y_df.columns:
        y = y_df["tm_helix_count"].to_numpy(dtype=int)
    else:
        y = y_df.iloc[:, 0].to_numpy(dtype=int)

    x = x_df.to_numpy(dtype=float)
    feature_names = list(x_df.columns)
    return x, y, feature_names


def build_horseshoe_zinb(x: np.ndarray, y: np.ndarray):
    n, p = x.shape

    with pm.Model() as model:
        # Horseshoe priors (count component)
        tau_count = pm.HalfCauchy("tau_count", beta=1.0)
        lam_count = pm.HalfCauchy("lam_count", beta=1.0, shape=p)
        z_count = pm.Normal("z_count", mu=0.0, sigma=1.0, shape=p)
        beta_count = pm.Deterministic("beta_count", z_count * tau_count * lam_count)

        # Horseshoe priors (zero-inflation component)
        tau_zi = pm.HalfCauchy("tau_zi", beta=1.0)
        lam_zi = pm.HalfCauchy("lam_zi", beta=1.0, shape=p)
        z_zi = pm.Normal("z_zi", mu=0.0, sigma=1.0, shape=p)
        beta_zi = pm.Deterministic("beta_zi", z_zi * tau_zi * lam_zi)

        alpha_count = pm.Normal("alpha_count", mu=0.0, sigma=2.0)
        alpha_zi = pm.Normal("alpha_zi", mu=0.0, sigma=2.0)

        eta_count = alpha_count + pm.math.dot(x, beta_count)
        mu = pm.Deterministic("mu", pm.math.exp(eta_count))

        eta_zi = alpha_zi + pm.math.dot(x, beta_zi)
        psi = pm.Deterministic("psi", pm.math.sigmoid(eta_zi))

        alpha_nb = pm.HalfNormal("alpha_nb", sigma=2.0)

        pm.ZeroInflatedNegativeBinomial(
            "tm_count_obs",
            psi=psi,
            mu=mu,
            alpha=alpha_nb,
            observed=y,
        )

    return model


def save_feature_effects(idata: az.InferenceData, feature_names, output_path: Path):
    beta_count_mean = idata.posterior["beta_count"].mean(dim=("chain", "draw")).values
    beta_zi_mean = idata.posterior["beta_zi"].mean(dim=("chain", "draw")).values

    rows = []
    for i, name in enumerate(feature_names):
        rows.append(
            {
                "feature": name,
                "beta_count_posterior_mean": float(beta_count_mean[i]),
                "beta_zi_posterior_mean": float(beta_zi_mean[i]),
                "abs_beta_count": float(abs(beta_count_mean[i])),
                "abs_beta_zi": float(abs(beta_zi_mean[i])),
            }
        )

    effects_df = pd.DataFrame(rows).sort_values("abs_beta_count", ascending=False)
    effects_df.to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Horseshoe-ZINB Bayesian model (Role 3).")
    parser.add_argument("--x-path", default="data/processed/X_train.csv")
    parser.add_argument("--y-path", default="data/processed/y_train.csv")
    parser.add_argument("--output-dir", default="results/role3_model")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional row subsample for quick checks. Use 0 for full data.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(".")
    x_path = root / args.x_path
    y_path = root / args.y_path

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, feature_names = load_training_data(x_path, y_path)
    if args.sample_size and args.sample_size > 0:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(y), size=min(args.sample_size, len(y)), replace=False)
        x = x[idx]
        y = y[idx]

    with build_horseshoe_zinb(x, y) as model:
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

        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["tm_count_obs"],
            random_seed=args.seed,
            return_inferencedata=True,
        )

    combined = az.concat(idata, ppc, dim=None)

    rhat_df = az.rhat(idata).to_dataframe().reset_index()
    rhat_df.to_csv(output_dir / "rhat_diagnostics.csv", index=False)

    ess_df = az.ess(idata).to_dataframe().reset_index()
    ess_df.to_csv(output_dir / "ess_diagnostics.csv", index=False)

    loo_result = az.loo(idata, pointwise=True)
    waic_result = az.waic(idata, pointwise=True)

    metrics = {
        "loo_elpd": float(loo_result.elpd_loo),
        "loo_p_loo": float(loo_result.p_loo),
        "loo_se": float(loo_result.se),
        "waic_elpd": float(waic_result.elpd_waic),
        "waic_p_waic": float(waic_result.p_waic),
        "waic_se": float(waic_result.se),
        "zero_rate_observed": float((y == 0).mean()),
    }
    with open(output_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Posterior predictive summary for quick comparison with observed counts
    y_rep = combined.posterior_predictive["tm_count_obs"].stack(sample=("chain", "draw")).values
    y_rep_mean = y_rep.mean(axis=1)
    ppc_summary = pd.DataFrame(
        {
            "y_observed": y,
            "y_rep_mean": y_rep_mean,
            "residual": y - y_rep_mean,
        }
    )
    ppc_summary.to_csv(output_dir / "ppc_summary.csv", index=False)

    save_feature_effects(idata, feature_names, output_dir / "feature_effects.csv")

    az.to_netcdf(idata, output_dir / "zinb_trace.nc")
    az.to_netcdf(combined, output_dir / "zinb_trace_with_ppc.nc")

    # Quick visual PPC output for Role 4 handoff
    ax = az.plot_ppc(combined, data_pairs={"tm_count_obs": "tm_count_obs"}, num_pp_samples=100)
    fig = ax.ravel()[0].figure if hasattr(ax, "ravel") else ax.figure
    fig.tight_layout()
    fig.savefig(output_dir / "ppc_plot.png", dpi=200)

    print("Role 3 complete.")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
