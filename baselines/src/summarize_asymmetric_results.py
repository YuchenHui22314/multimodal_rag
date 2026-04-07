import argparse
import json
import os
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "map",
    "ndcg_cut.1",
    "ndcg_cut.3",
    "ndcg_cut.5",
    "ndcg_cut.10",
    "P.1",
    "P.3",
    "P.5",
    "P.10",
    "recall.5",
    "recall.50",
    "recall.100",
    "recall.1000",
    "MRR@1",
    "MRR@3",
    "MRR@5",
    "MRR@10",
]


def load_metrics(metrics_path: str) -> dict:
    with open(metrics_path) as f:
        data = json.load(f)
    if "metrics" in data:
        payload = data["metrics"]
        payload["epoch"] = data.get("epoch")
    else:
        payload = data
    return payload


def summarize_run(run_dir: Path, target_metric: str) -> dict | None:
    eval_root = run_dir / "eval"
    if not eval_root.exists():
        return None

    best = None
    for metrics_path in sorted(eval_root.glob("epoch_*/metrics.json")):
        metrics = load_metrics(str(metrics_path))
        score = metrics.get(target_metric)
        if score is None:
            continue
        if best is None or score > best[target_metric]:
            best = {
                "name": run_dir.name,
                "best_epoch": metrics.get("epoch"),
                "metrics_path": str(metrics_path),
                **{col: metrics.get(col) for col in METRIC_COLUMNS},
                target_metric: score,
            }
    return best


def main():
    p = argparse.ArgumentParser(description="Summarize asymmetric bi-encoder eval results")
    p.add_argument("--runs_root", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--target_metric", default="MRR@10")
    p.add_argument("--baseline_metrics", nargs="*", default=[])
    args = p.parse_args()

    rows = []
    for baseline_arg in args.baseline_metrics:
        name, path = baseline_arg.split("=", 1)
        metrics = load_metrics(path)
        row = {
            "name": name,
            "best_epoch": None,
            "metrics_path": path,
            **{col: metrics.get(col) for col in METRIC_COLUMNS},
        }
        rows.append(row)

    runs_root = Path(args.runs_root)
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary = summarize_run(run_dir, args.target_metric)
        if summary is not None:
            rows.append(summary)

    df = pd.DataFrame(rows)
    ordered_cols = ["name", "best_epoch"] + METRIC_COLUMNS + ["metrics_path"]
    existing_cols = [col for col in ordered_cols if col in df.columns]
    df = df[existing_cols]
    df.to_csv(args.output_csv, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
