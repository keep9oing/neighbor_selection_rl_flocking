import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_radius_csv(path: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    if not header or header[0] != "step":
        raise ValueError(f"{path} does not look like a radius CSV")

    steps = np.asarray([int(float(row[0])) for row in rows], dtype=np.int64)
    labels = header[1:]
    radii = np.full((len(rows), len(labels)), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for col_idx, value in enumerate(row[1:]):
            if value != "":
                radii[row_idx, col_idx] = float(value)
    return steps, labels, radii


def load_summary(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def summary_records(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    records = {}
    for record in summary.get("records", []):
        for key, csv_path in record.items():
            if key == "csv_path" or key.endswith("_csv_path"):
                records[Path(csv_path).name] = record
    return records


def radius_bounds(summary: Dict[str, Any], radius_min_arg: Optional[float], radius_max_arg: Optional[float]) -> Tuple[float, float]:
    radius_min = radius_min_arg
    radius_max = radius_max_arg
    if radius_min is None:
        radius_min = summary.get("radius_min", summary.get("configured_radius_min"))
    if radius_max is None:
        radius_max = summary.get("radius_max", summary.get("configured_radius_max"))
    if radius_min is None or radius_max is None:
        raise ValueError("radius min/max are required for --value-mode action. Pass --radius-min and --radius-max.")
    radius_min = float(radius_min)
    radius_max = float(radius_max)
    if radius_max <= radius_min:
        raise ValueError("radius_max must be greater than radius_min")
    return radius_min, radius_max


def infer_episode_seed(path: Path) -> Tuple[Optional[int], Optional[int]]:
    match = re.search(r"episode_(\d+)_seed_(\d+)", path.name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def transform_values(radii: np.ndarray, value_mode: str, radius_min: Optional[float], radius_max: Optional[float]) -> np.ndarray:
    if value_mode == "radius":
        return radii
    if value_mode == "action":
        if radius_min is None or radius_max is None:
            raise ValueError("radius bounds are required to convert radius to action")
        return (radii - radius_min) / (radius_max - radius_min)
    raise ValueError(f"Unsupported value_mode: {value_mode}")


def y_limits(
    csv_paths: List[Path],
    explicit_min: Optional[float],
    explicit_max: Optional[float],
    value_mode: str,
    radius_min: Optional[float],
    radius_max: Optional[float],
) -> Tuple[float, float]:
    if explicit_min is not None and explicit_max is not None:
        if explicit_max <= explicit_min:
            raise ValueError("--ymax must be greater than --ymin")
        return explicit_min, explicit_max

    if value_mode == "action":
        ymin = 0.0 if explicit_min is None else explicit_min
        ymax = 1.0 if explicit_max is None else explicit_max
        if ymax <= ymin:
            raise ValueError("--ymax must be greater than --ymin")
        return ymin, ymax

    values = []
    for path in csv_paths:
        _, _, radii = load_radius_csv(path)
        radii = transform_values(radii, value_mode, radius_min, radius_max)
        finite = radii[np.isfinite(radii)]
        if finite.size:
            values.append(finite)
    if not values:
        return 0.0, 1.0

    all_values = np.concatenate(values)
    ymin = float(np.nanmin(all_values)) if explicit_min is None else explicit_min
    ymax = float(np.nanmax(all_values)) if explicit_max is None else explicit_max
    pad = max((ymax - ymin) * 0.08, 1.0)
    if explicit_min is None:
        ymin -= pad
    if explicit_max is None:
        ymax += pad
    return ymin, ymax


def title_for(
    path: Path,
    record: Optional[Dict[str, Any]],
    values: np.ndarray,
    value_mode: str,
    title_prefix: Optional[str],
    value_label: Optional[str],
) -> str:
    episode, seed = infer_episode_seed(path)
    parts = []
    if episode is not None:
        parts.append(f"episode={episode:03d}")
    if seed is not None:
        parts.append(f"seed={seed}")
    if record is not None:
        if "train_return" in record:
            parts.append(f"train={record['train_return']:.2f}")
        if "original_return" in record:
            parts.append(f"original={record['original_return']:.2f}")
    label = value_label or ("normalized action" if value_mode == "action" else "radius")
    if value_mode == "action":
        parts.append(f"mean {label}={np.nanmean(values):.4f}")
        parts.append(f"range={np.nanmin(values):.4f}-{np.nanmax(values):.4f}")
    else:
        parts.append(f"mean {label}={np.nanmean(values):.2f}")
        parts.append(f"range={np.nanmin(values):.2f}-{np.nanmax(values):.2f}")
    prefix = title_prefix or ("learned_radius_best normalized action over time" if value_mode == "action" else "learned_radius_best radius over time")
    return prefix + " | " + " | ".join(parts)


def plot_all_agents_line(
    csv_path: Path,
    record: Optional[Dict[str, Any]],
    output_path: Path,
    ymin: float,
    ymax: float,
    value_mode: str,
    radius_min: Optional[float],
    radius_max: Optional[float],
    title_prefix: Optional[str],
    y_label: Optional[str],
    value_label: Optional[str],
    dpi: int,
):
    steps, labels, radii = load_radius_csv(csv_path)
    values = transform_values(radii, value_mode, radius_min, radius_max)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=dpi, constrained_layout=True)
    colors = plt.get_cmap("tab20")(np.arange(len(labels)) % 20)

    for agent_idx, label in enumerate(labels):
        ax.plot(
            steps,
            values[:, agent_idx],
            color=colors[agent_idx],
            linewidth=0.9,
            alpha=0.82,
            label=label.replace("agent_", "a"),
        )

    ax.plot(steps, np.nanmean(values, axis=1), color="#111827", linewidth=1.8, label="mean")
    ax.set_title(title_for(csv_path, record, values, value_mode, title_prefix, value_label), fontsize=10)
    ax.set_xlabel("timestep")
    ax.set_ylabel(y_label or ("normalized radius action" if value_mode == "action" else "selected disk radius"))
    ax.set_ylim(ymin, ymax)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(ncol=3, fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(output_path)
    plt.close(fig)


def plot_agent_small_multiples(
    csv_path: Path,
    record: Optional[Dict[str, Any]],
    output_path: Path,
    ymin: float,
    ymax: float,
    value_mode: str,
    radius_min: Optional[float],
    radius_max: Optional[float],
    title_prefix: Optional[str],
    y_label: Optional[str],
    value_label: Optional[str],
    dpi: int,
):
    steps, labels, radii = load_radius_csv(csv_path)
    values = transform_values(radii, value_mode, radius_min, radius_max)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_agents = len(labels)
    cols = 4
    rows = int(np.ceil(num_agents / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10), dpi=dpi, sharex=True, sharey=True, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    colors = plt.get_cmap("tab20")(np.arange(num_agents) % 20)

    for agent_idx, label in enumerate(labels):
        ax = axes[agent_idx]
        series = values[:, agent_idx]
        ax.plot(steps, series, color=colors[agent_idx], linewidth=1.1)
        ax.axhline(np.nanmean(series), color="#111827", linewidth=0.8, linestyle="--", alpha=0.75)
        precision = 4 if value_mode == "action" else 2
        ax.set_title(f"{label} mean={np.nanmean(series):.{precision}f}", fontsize=8)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, color="#e5e7eb", linewidth=0.6)

    for ax in axes[num_agents:]:
        ax.axis("off")

    fig.suptitle(title_for(csv_path, record, values, value_mode, title_prefix, value_label), fontsize=10)
    fig.supxlabel("timestep")
    fig.supylabel(y_label or ("normalized radius action" if value_mode == "action" else "selected disk radius"))
    fig.savefig(output_path)
    plt.close(fig)


def save_transformed_csv(
    csv_path: Path,
    output_path: Path,
    value_mode: str,
    radius_min: Optional[float],
    radius_max: Optional[float],
):
    steps, labels, radii = load_radius_csv(csv_path)
    values = transform_values(radii, value_mode, radius_min, radius_max)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + labels)
        for step, row in zip(steps, values):
            writer.writerow([int(step)] + ["" if np.isnan(value) else f"{value:.8f}" for value in row])


def main():
    parser = argparse.ArgumentParser(description="Plot radius time series from already saved radius CSVs.")
    parser.add_argument("--input-csv-dir", default="results/radius_vs_acs_best_checkpoint/media/radius_trends/csv")
    parser.add_argument("--summary-json", default="results/radius_vs_acs_best_checkpoint/media/radius_trends/radius_trend_summary.json")
    parser.add_argument("--output-dir", default="results/radius_vs_acs_best_checkpoint/media/radius_time_series")
    parser.add_argument("--value-mode", choices=("radius", "action"), default="radius")
    parser.add_argument("--radius-min", type=float, default=None)
    parser.add_argument("--radius-max", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--title-prefix", default=None)
    parser.add_argument("--y-label", default=None)
    parser.add_argument("--value-label", default=None)
    parser.add_argument("--dpi", type=int, default=130)
    args = parser.parse_args()

    input_csv_dir = Path(args.input_csv_dir)
    csv_paths = sorted(input_csv_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {input_csv_dir}")

    summary_path = Path(args.summary_json) if args.summary_json else None
    summary = load_summary(summary_path)
    records = summary_records(summary)
    radius_min = radius_max = None
    if args.value_mode == "action":
        radius_min, radius_max = radius_bounds(summary, args.radius_min, args.radius_max)
    ymin, ymax = y_limits(csv_paths, args.ymin, args.ymax, args.value_mode, radius_min, radius_max)

    output_dir = Path(args.output_dir)
    all_agents_dir = output_dir / "all_agents_line"
    small_multiples_dir = output_dir / "agent_small_multiples"
    transformed_csv_dir = output_dir / "csv"

    for csv_path in csv_paths:
        stem = csv_path.stem.replace("_radius_trend", f"_{args.value_mode}_timeseries")
        record = records.get(csv_path.name)
        plot_all_agents_line(
            csv_path=csv_path,
            record=record,
            output_path=all_agents_dir / f"{stem}_all_agents.png",
            ymin=ymin,
            ymax=ymax,
            value_mode=args.value_mode,
            radius_min=radius_min,
            radius_max=radius_max,
            title_prefix=args.title_prefix,
            y_label=args.y_label,
            value_label=args.value_label,
            dpi=args.dpi,
        )
        plot_agent_small_multiples(
            csv_path=csv_path,
            record=record,
            output_path=small_multiples_dir / f"{stem}_small_multiples.png",
            ymin=ymin,
            ymax=ymax,
            value_mode=args.value_mode,
            radius_min=radius_min,
            radius_max=radius_max,
            title_prefix=args.title_prefix,
            y_label=args.y_label,
            value_label=args.value_label,
            dpi=args.dpi,
        )
        if args.value_mode != "radius":
            save_transformed_csv(
                csv_path=csv_path,
                output_path=transformed_csv_dir / f"{stem}.csv",
                value_mode=args.value_mode,
                radius_min=radius_min,
                radius_max=radius_max,
            )
        print(f"plotted {csv_path.name}", flush=True)

    manifest = {
        "input_csv_dir": str(input_csv_dir),
        "summary_json": str(summary_path) if summary_path else None,
        "value_mode": args.value_mode,
        "radius_min": radius_min,
        "radius_max": radius_max,
        "num_csv_files": len(csv_paths),
        "ymin": ymin,
        "ymax": ymax,
        "title_prefix": args.title_prefix,
        "y_label": args.y_label,
        "value_label": args.value_label,
        "all_agents_line_dir": str(all_agents_dir),
        "agent_small_multiples_dir": str(small_multiples_dir),
        "transformed_csv_dir": str(transformed_csv_dir) if args.value_mode != "radius" else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "radius_time_series_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {output_dir / 'radius_time_series_manifest.json'}")


if __name__ == "__main__":
    main()
