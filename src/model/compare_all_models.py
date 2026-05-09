import argparse
import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ZINB, NB, and ZIP model outputs.")
    parser.add_argument("--zinb-metrics", default="results/role3_model/model_metrics.json")
    parser.add_argument("--baseline-metrics", default="results/role3_model_baselines/baseline_model_comparison.csv")
    parser.add_argument("--output-dir", default="results/role3_comparison")
    return parser.parse_args()


def load_zinb_metrics(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return {
        "model": "ZINB",
        "elpd_loo": m.get("loo_elpd"),
        "p_loo": m.get("loo_p_loo"),
        "loo_se": m.get("loo_se"),
        "elpd_waic": m.get("waic_elpd"),
        "p_waic": m.get("waic_p_waic"),
        "waic_se": m.get("waic_se"),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zinb = load_zinb_metrics(Path(args.zinb_metrics))
    base_df = pd.read_csv(args.baseline_metrics)

    # Normalize potential naming differences
    rename_map = {}
    if "model" not in base_df.columns and "Model" in base_df.columns:
        rename_map["Model"] = "model"
    if "elpd_loo" not in base_df.columns and "loo_elpd" in base_df.columns:
        rename_map["loo_elpd"] = "elpd_loo"
    if rename_map:
        base_df = base_df.rename(columns=rename_map)

    required = ["model", "elpd_loo", "p_loo", "loo_se", "elpd_waic", "p_waic", "waic_se"]
    missing = [c for c in required if c not in base_df.columns]
    if missing:
        raise ValueError(f"Baseline CSV missing required columns: {missing}")

    all_df = pd.concat([base_df[required], pd.DataFrame([zinb])], ignore_index=True)
    all_df["loo_rank"] = all_df["elpd_loo"].rank(ascending=False, method="min").astype(int)
    all_df["waic_rank"] = all_df["elpd_waic"].rank(ascending=False, method="min").astype(int)
    all_df = all_df.sort_values(["loo_rank", "waic_rank", "model"]).reset_index(drop=True)

    best_model = all_df.iloc[0]["model"]
    summary = {
        "best_model_by_loo": best_model,
        "n_models_compared": int(len(all_df)),
        "ranking_order": all_df["model"].tolist(),
    }

    all_df.to_csv(output_dir / "all_model_comparison.csv", index=False)
    with open(output_dir / "all_model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_df.to_dict(orient="records"), f, indent=2)
    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Comparison complete.")
    print(f"Best model by LOO: {best_model}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
