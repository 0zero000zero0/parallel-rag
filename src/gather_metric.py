import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gather all metrics.json under outputs/<method>/<model>/<dataset>/<index>/ "
            "and export one Excel file per index"
        )
    )
    parser.add_argument(
        "--method",
        type=str,
        default="adaptive-parallel-o1",
        help="Method name under outputs, e.g. parallel-rag",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3-32B",
        help="Model name under outputs/<method>, e.g. Qwen3-32B",
    )
    parser.add_argument(
        "--outputs_root",
        type=str,
        default="outputs",
        help="Outputs root dir",
    )
    return parser


def resolve_method_model_path(method: str, model: str, outputs_root: str) -> Path:
    candidate = (Path(outputs_root).expanduser() / method / model).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Path not found: {candidate}")
    return candidate


def sort_key_for_index(index_name: str) -> Any:
    if index_name.isdigit():
        return (0, int(index_name))
    return (1, index_name)


def collect_metrics(method_model_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for dataset_dir in sorted(
        [p for p in method_model_dir.iterdir() if p.is_dir()], key=lambda p: p.name
    ):
        for index_dir in sorted(
            [p for p in dataset_dir.iterdir() if p.is_dir()],
            key=lambda p: sort_key_for_index(p.name),
        ):
            metrics_file = index_dir / "metrics.json"
            if not metrics_file.exists():
                continue

            try:
                with metrics_file.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skip invalid json: {metrics_file} ({exc})")
                continue

            if not isinstance(metrics, dict):
                print(f"[WARN] Skip non-dict metrics file: {metrics_file}")
                continue

            row: dict[str, Any] = {
                "dataset": dataset_dir.name,
                "index": index_dir.name,
            }
            row.update(metrics)
            rows.append(row)

    return rows


def main() -> None:
    args = build_parser().parse_args()

    method_model_dir = resolve_method_model_path(
        args.method, args.model, args.outputs_root
    )
    if not method_model_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {method_model_dir}")

    rows = collect_metrics(method_model_dir)
    if not rows:
        raise ValueError(f"No metrics.json found under: {method_model_dir}")

    df = pd.DataFrame(rows)

    preferred_front = ["dataset"]
    other_cols = [c for c in df.columns if c not in preferred_front]
    df = df[preferred_front + other_cols]
    df = df.sort_values(by=["dataset", "index"], kind="stable").reset_index(drop=True)

    saved_files: list[Path] = []
    group: pd.DataFrame
    for index_name, group in df.groupby("index", sort=False):
        tmp_xlsx_path = method_model_dir / f"{index_name}.xlsx"
        group.to_excel(tmp_xlsx_path, index=False, float_format="%.4f")
        saved_files.append(tmp_xlsx_path)
        # tmp_csv_path = method_model_dir / f"{index_name}.csv"
        # group.to_csv(tmp_csv_path, index=False, sep="\t")
        # saved_files.append(tmp_csv_path)

    print(f"Method/Model dir: {method_model_dir}")
    print(f"Rows aggregated: {len(df)}")
    print(f"Saved {len(saved_files)} Excel file(s):")
    for p in saved_files:
        print(f"- {p.resolve()}")


if __name__ == "__main__":
    main()
