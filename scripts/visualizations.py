#!/usr/bin/env python3
"""Generate figures from saved rank-evaluation artifacts."""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from PIL import Image, ImageDraw

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = Path("reports/figures")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create figures from saved rank_eval outputs."
    )
    parser.add_argument("--rank_csv", required=True, help="Rank evaluation summary CSV from rank_eval.py.")
    parser.add_argument("--diagnostics_dir", required=True, help="Directory containing diagnostics CSV files.")
    parser.add_argument("--traces_dir", required=True, help="Directory containing trace CSV files.")
    parser.add_argument(
        "--attack_images_dir",
        default=None,
        help="Optional directory containing saved attack image PNG artifacts.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Figure output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument("--metric", default="R@1", help="Rank CSV metric to plot for the strength sweep.")
    return parser.parse_args()


def load_rank_rows(rank_csv: Path) -> pd.DataFrame:
    return pd.read_csv(rank_csv)


def attacked_rows(rank_rows: pd.DataFrame) -> pd.DataFrame:
    rows = rank_rows.copy()
    if "epsilon" not in rows.columns:
        return rows.iloc[0:0].copy()
    rows["epsilon_numeric"] = pd.to_numeric(rows["epsilon"], errors="coerce")
    attack_column = rows.get("attack", pd.Series([""] * len(rows)))
    attacked = rows[rows["epsilon_numeric"].notna() & attack_column.fillna("").astype(str).ne("")]
    return attacked.copy()


def save_empty_figure(path: Path, message: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_attack_strength_sweep(rank_rows: pd.DataFrame, metric: str, output_path: Path) -> Path:
    rows = attacked_rows(rank_rows)
    if rows.empty or metric not in rows.columns:
        return save_empty_figure(output_path, f"No attacked rows with metric {metric} were found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows[metric] = pd.to_numeric(rows[metric], errors="coerce")
    rows = rows.dropna(subset=["epsilon_numeric", metric])

    fig, ax = plt.subplots(figsize=(8, 5))
    for (dataset, model), group in rows.groupby(["dataset", "model"], dropna=False):
        group = group.sort_values("epsilon_numeric")
        ax.plot(
            group["epsilon_numeric"],
            group[metric],
            marker="o",
            linewidth=2,
            label=f"{dataset}/{model}",
        )

    ax.set_xlabel("Normalized epsilon")
    ax.set_ylabel(metric)
    ax.set_title("Attack-strength sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def find_trace_groups(traces_dir: Path, limit: int = 5) -> dict[int, list[Path]]:
    groups: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(traces_dir.rglob("*.csv")):
        try:
            rows = pd.read_csv(path, nrows=1)
        except pd.errors.EmptyDataError:
            continue
        if "query_index" not in rows.columns or rows.empty:
            continue
        groups[int(rows.iloc[0]["query_index"])].append(path)
    return dict(list(groups.items())[:limit])


def plot_query_trace(trace_paths: Sequence[Path], output_path: Path) -> Path:
    if not trace_paths:
        return save_empty_figure(output_path, "No trace rows were found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax_dist = plt.subplots(figsize=(8, 5))
    ax_rank = ax_dist.twinx()
    query_index = None

    for path in sorted(trace_paths):
        rows = pd.read_csv(path)
        if rows.empty:
            continue
        query_index = int(rows.iloc[0]["query_index"])
        epsilon = rows.iloc[0]["epsilon"]
        label = f"eps {epsilon:g}" if isinstance(epsilon, float) else f"eps {epsilon}"
        ax_dist.plot(rows["step"], rows["positive_distance"], marker="o", label=f"positive {label}")
        ax_dist.plot(rows["step"], rows["hard_negative_distance"], marker="s", label=f"hard negative {label}")
        ax_rank.plot(
            rows["step"],
            rows["nearest_positive_rank"],
            linestyle="--",
            marker="^",
            alpha=0.7,
            label=f"rank {label}",
        )

    ax_dist.set_xlabel("PGD step")
    ax_dist.set_ylabel("Descriptor distance")
    ax_rank.set_ylabel("Nearest-positive rank")
    ax_dist.set_title(f"Attack trace for query {query_index}")
    ax_dist.grid(True, alpha=0.3)

    handles, labels = ax_dist.get_legend_handles_labels()
    rank_handles, rank_labels = ax_rank.get_legend_handles_labels()
    ax_dist.legend(handles + rank_handles, labels + rank_labels, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def load_diagnostics(diagnostics_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(diagnostics_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_margin_vs_failure(diagnostics: pd.DataFrame, output_path: Path) -> Path:
    required = {"clean_margin", "attacked_nearest_positive_rank", "attack_success"}
    if diagnostics.empty or not required.issubset(diagnostics.columns):
        return save_empty_figure(output_path, "No diagnostics rows with margin and attack success were found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = diagnostics.copy()
    rows["clean_margin"] = pd.to_numeric(rows["clean_margin"], errors="coerce")
    rows["attacked_nearest_positive_rank"] = pd.to_numeric(
        rows["attacked_nearest_positive_rank"],
        errors="coerce",
    )
    rows["attack_success_bool"] = rows["attack_success"].astype(str).str.lower().isin({"true", "1", "yes"})
    rows = rows.dropna(subset=["clean_margin", "attacked_nearest_positive_rank"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for success, group in rows.groupby("attack_success_bool"):
        label = "attack success" if success else "attack failed"
        ax.scatter(
            group["clean_margin"],
            group["attacked_nearest_positive_rank"],
            alpha=0.75,
            s=36,
            label=label,
        )
    ax.set_xlabel("Clean margin")
    ax.set_ylabel("Attacked nearest-positive rank")
    ax.set_title("Margin vs attack outcome")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.text(0.01, 0.01, "CWR, when shown in diagnostics, is an empirical descriptor-space estimate.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _query_index_from_image(path: Path) -> int | None:
    match = re.match(r"(\d+)_", path.name)
    return int(match.group(1)) if match else None


def _epsilon_dir_sort_key(path: Path) -> float:
    try:
        return float(path.name.replace("eps_", ""))
    except ValueError:
        return float("inf")


def collect_attack_image_rows(attack_images_dir: Path, limit: int = 5) -> list[dict[str, object]]:
    rows_by_query: dict[int, dict[str, object]] = {}
    for epsilon_dir in sorted(attack_images_dir.rglob("eps_*"), key=_epsilon_dir_sort_key):
        for clean_path in sorted(epsilon_dir.glob("*_clean.png")):
            query_index = _query_index_from_image(clean_path)
            if query_index is None:
                continue
            row = rows_by_query.setdefault(query_index, {"query_index": query_index, "attacked": []})
            row.setdefault("clean", clean_path)
            attacked_path = epsilon_dir / f"{query_index}_attacked.png"
            if attacked_path.exists():
                row["attacked"].append((epsilon_dir.name.replace("eps_", "eps "), attacked_path))
            perturbation_paths = sorted(epsilon_dir.glob(f"{query_index}_perturbation_*.png"))
            heatmap_path = epsilon_dir / f"{query_index}_abs_heatmap.png"
            if perturbation_paths:
                row["perturbation"] = perturbation_paths[-1]
            if heatmap_path.exists():
                row["abs_heatmap"] = heatmap_path
    return [rows_by_query[key] for key in sorted(rows_by_query)[:limit]]


def _load_resized(path: Path, size: tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("RGB").resize(size)


def build_perturbation_grid(attack_images_dir: Path, output_path: Path, limit: int = 5) -> Path | None:
    rows = collect_attack_image_rows(attack_images_dir, limit=limit)
    if not rows:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    attacked_count = max(len(row.get("attacked", [])) for row in rows)
    columns = ["clean", *[f"attacked {index + 1}" for index in range(attacked_count)], "perturbation", "heatmap"]
    cell_size = (180, 120)
    header_height = 28
    width = cell_size[0] * len(columns)
    height = header_height + cell_size[1] * len(rows)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for column_index, label in enumerate(columns):
        draw.text((column_index * cell_size[0] + 6, 8), label, fill=(0, 0, 0))

    for row_index, row in enumerate(rows):
        y = header_height + row_index * cell_size[1]
        draw.text((6, y + 6), f"q{row['query_index']}", fill=(255, 255, 255))
        images: list[tuple[str, Path | None]] = [("clean", row.get("clean"))]
        attacked_images = list(row.get("attacked", []))
        images.extend((label, path) for label, path in attacked_images)
        images.extend([("perturbation", row.get("perturbation")), ("heatmap", row.get("abs_heatmap"))])
        for column_index, (_, path) in enumerate(images[: len(columns)]):
            if path is None:
                continue
            canvas.paste(_load_resized(Path(path), cell_size), (column_index * cell_size[0], y))

    canvas.save(output_path)
    return output_path


def generate_figures(
    rank_csv: Path,
    diagnostics_dir: Path,
    traces_dir: Path,
    output_dir: Path,
    metric: str = "R@1",
    attack_images_dir: Path | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = [
        plot_attack_strength_sweep(load_rank_rows(rank_csv), metric, output_dir / "attack_strength_sweep.png"),
        plot_margin_vs_failure(load_diagnostics(diagnostics_dir), output_dir / "margin_vs_failure.png"),
    ]

    for query_index, paths in find_trace_groups(traces_dir, limit=5).items():
        generated.append(plot_query_trace(paths, output_dir / f"query_trace_{query_index}.png"))

    if attack_images_dir is not None and attack_images_dir.exists():
        grid_path = build_perturbation_grid(attack_images_dir, output_dir / "perturbation_visibility_grid.png")
        if grid_path is not None:
            generated.append(grid_path)

    return generated


def main() -> None:
    args = parse_arguments()
    attack_images_dir = Path(args.attack_images_dir) if args.attack_images_dir else None
    generated = generate_figures(
        rank_csv=Path(args.rank_csv),
        diagnostics_dir=Path(args.diagnostics_dir),
        traces_dir=Path(args.traces_dir),
        output_dir=Path(args.output_dir),
        metric=args.metric,
        attack_images_dir=attack_images_dir,
    )
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
