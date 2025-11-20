#!/usr/bin/env python3
"""
Compute the mean metric *per .txt file*.

Expected line format (flexible):
  2025-11-25_01-44-44.json -> metric: 0.0114 (used frames: 1006 )

Usage:
  python mean_per_txt.py aa.txt bb.txt
  python mean_per_txt.py /path/to/dir_with_txts --recurse
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path
from typing import Optional


METRIC_RE = re.compile(r"\bmetric:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
FRAMES_RE = re.compile(r"\bused\s*frames:\s*(\d+)")


def iter_txt_files(inputs: list[str], recurse: bool) -> list[Path]:
    out: list[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            pattern = "**/*.txt" if recurse else "*.txt"
            out.extend(sorted(p.glob(pattern)))
        else:
            out.append(p)
    # keep only existing files
    out = [p for p in out if p.exists() and p.is_file()]
    return sorted(out, key=lambda x: (x.parent.as_posix(), x.name))


def parse_metrics_from_txt(path: Path) -> tuple[list[float], list[Optional[int]]]:
    metrics: list[float] = []
    frames: list[Optional[int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = METRIC_RE.search(line)
        if not m:
            continue
        metrics.append(float(m.group(1)))

        f = FRAMES_RE.search(line)
        frames.append(int(f.group(1)) if f else None)

    return metrics, frames


def weighted_mean(values: list[float], weights: list[int]) -> float:
    total_w = sum(weights)
    if total_w == 0:
        raise ValueError("sum(weights) == 0")
    return sum(v * w for v, w in zip(values, weights)) / total_w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="One or more .txt files and/or directories")
    ap.add_argument("--recurse", action="store_true", help="Recurse into subdirectories")
    args = ap.parse_args()

    paths = iter_txt_files(args.inputs, args.recurse)
    if not paths:
        print("No .txt files found in inputs.", file=sys.stderr)
        return 2

    for path in paths:
        try:
            metrics, frames = parse_metrics_from_txt(path)
        except Exception as e:
            print(f"[skip] {path}: failed to read/parse: {e}", file=sys.stderr)
            continue

        if not metrics:
            print(f"{path.name}: no metrics found")
            continue

        mean = statistics.fmean(metrics)
        print(f"{path.name}: records={len(metrics)}  mean_metric={mean}")

        # Optional: weighted mean if every line had used frames
        if all(f is not None for f in frames):
            wmean = weighted_mean(metrics, [f for f in frames if f is not None])
            print(f"  weighted_mean_by_used_frames={wmean}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
