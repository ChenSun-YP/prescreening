#!/usr/bin/env python3
"""
Scan all subfolders of a semiFine directory for mode_stdev_good_pairs.txt,
then plot a grid of crosscorrelograms for every listed neuron pair.

Each panel is labelled with the subfolder name and the two neuron names.

Usage:
    python plot_good_pairs.py [--semifine DIR] [--base BASE] [--session SESSION_ID] [--out OUTPUT.png]

Defaults:
    --semifine  /project2/dsong_945/BenR/prescreening/FilterFiles/
                    Jan2010-Nonstationarity_Learning/1029/semiFine
    --base      /project2/dsong_945/BenR/prescreening/data/
                    Jan2010-Nonstationarity_Learning_DUPLICATE
    --session   1029
    --out       good_pairs_ccg.png
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SEMIFINE = (
    "/project2/dsong_945/BenR/prescreening/FilterFiles/"
    "Jan2010-Nonstationarity_Learning/1029/semiFine"
)
DEFAULT_BASE = (
    "/project2/dsong_945/BenR/prescreening/data/"
    "Jan2010-Nonstationarity_Learning_DUPLICATE"
)
DEFAULT_SESSION = "1029"
PAIRS_FILENAME = "mode_stdev_good_pairs.txt"
CCG_FILENAME = "crosscorrs_edge_mean_True_semifine.pkl"
GRID_COLS = 5
# ──────────────────────────────────────────────────────────────────────────────


def find_pair_files(semifine_dir: Path) -> list[Path]:
    """Return all mode_stdev_good_pairs.txt files found one level deep."""
    files = sorted(semifine_dir.glob(f"*/{PAIRS_FILENAME}"))
    if not files:
        # also try two levels deep just in case
        files = sorted(semifine_dir.glob(f"**/{PAIRS_FILENAME}"))
    return files


def parse_pairs_txt(path: Path) -> list[tuple[str, str]]:
    """Parse tab-separated (neuronA, neuronB) pairs from a txt file."""
    pairs = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                print(
                    f"  [warn] {path.name} line {lineno}: expected 2 cols, got {len(cols)} — skipping"
                )
                continue
            pairs.append((cols[0].strip(), cols[1].strip()))
    return pairs


def ccg_pkl_path(base: Path, session_id: str, stem: str) -> Path:
    """
    Build CCG pkl path.  Tries the stem as-is first, then lowercase,
    to handle mixed-case subfolder names (K029U197 vs k029u204).
    """
    for s in [stem, stem.lower(), stem.upper()]:
        p = base / "analysis" / session_id / s / CCG_FILENAME
        if p.exists():
            return p
    # return the as-is path (will be reported as missing)
    return base / "analysis" / session_id / stem / CCG_FILENAME


def load_pkl(path: Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def lookup_pair(data: dict, neuronA: str, neuronB: str):
    """Try (A,B) then (B,A). Returns (key, value, flipped) or (None, None, False)."""
    for flipped, key in enumerate([(neuronA, neuronB), (neuronB, neuronA)]):
        if key in data:
            return key, data[key], bool(flipped)
    return None, None, False


def extract_ccg(value):
    """
    Pull (lags, counts) from whatever the dict value is.
    tuple format: (lags_array, ccg_array, mean, std, significance, threshold)
    """
    if isinstance(value, tuple):
        return np.asarray(value[0]), np.asarray(value[1])

    if isinstance(value, (np.ndarray, list)):
        arr = np.asarray(value).squeeze()
        if arr.ndim == 1:
            return None, arr
        if arr.ndim == 2:
            return arr[:, 0], arr[:, 1]

    if isinstance(value, dict):
        for ck in ("ccg", "counts", "hist", "crosscorr", "y", "values"):
            if ck in value:
                lags = value.get("lags", value.get("x", value.get("bins", None)))
                return lags, np.asarray(value[ck]).squeeze()
        for v in value.values():
            arr = np.asarray(v)
            if arr.ndim == 1 and arr.dtype.kind in "fiu":
                return None, arr

    for attr in ("ccg", "counts", "hist", "crosscorr"):
        if hasattr(value, attr):
            return (
                getattr(value, "lags", None),
                np.asarray(getattr(value, attr)).squeeze(),
            )

    raise ValueError(f"Cannot extract CCG from value of type {type(value)}")


# ── plotting ──────────────────────────────────────────────────────────────────


def plot_grid(entries: list[dict], out_path: str):
    """
    entries: list of dicts with keys:
        neuronA, neuronB, stem, pkl_path
    """
    n = len(entries)
    if n == 0:
        sys.exit("No pairs found across any mode_stdev_good_pairs.txt files.")

    cols = GRID_COLS
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 3.2, rows * 2.6),
        constrained_layout=True,
    )
    # normalise axes to a flat list
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    elif rows == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [ax for row in axes for ax in row]

    fig.suptitle("Good Pair Crosscorrelograms", fontsize=14, fontweight="bold")

    pkl_cache = {}

    def get_data(pkl_path: Path):
        if pkl_path not in pkl_cache:
            pkl_cache[pkl_path] = load_pkl(pkl_path)
        return pkl_cache[pkl_path]

    for idx, entry in enumerate(entries):
        ax = axes_flat[idx]
        neuronA = entry["neuronA"]
        neuronB = entry["neuronB"]
        stem = entry["stem"]
        pkl_path = entry["pkl_path"]

        base_title = f"[{stem}]\n{neuronA} :\n{neuronB}"

        if not pkl_path.exists():
            ax.text(
                0.5,
                0.5,
                f"PKL not found:\n{stem}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="red",
                fontsize=7,
            )
            ax.set_title(base_title, fontsize=6, color="red")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        try:
            data = get_data(pkl_path)
        except Exception as exc:
            ax.text(
                0.5,
                0.5,
                f"Load error:\n{exc}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="orange",
                fontsize=6,
            )
            ax.set_title(base_title, fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        key, value, flipped = lookup_pair(data, neuronA, neuronB)

        if key is None:
            ax.text(
                0.5,
                0.5,
                "Pair not found\nin PKL",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="red",
                fontsize=8,
            )
            ax.set_title(base_title, fontsize=6, color="red")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        try:
            lags, counts = extract_ccg(value)
        except ValueError as exc:
            ax.text(
                0.5,
                0.5,
                f"Parse error:\n{exc}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="orange",
                fontsize=6,
            )
            ax.set_title(base_title, fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        if flipped:
            counts = counts[::-1]
            if lags is not None:
                lags = -lags[::-1]

        x = lags if lags is not None else np.arange(len(counts))
        bw = (x[1] - x[0]) if len(x) > 1 else 1
        ax.bar(x, counts, width=bw, color="steelblue", edgecolor="none", alpha=0.85)
        ax.set_xlim(x[0] - bw / 2, x[-1] + bw / 2)
        ax.axvline(0, color="red", linewidth=0.8, linestyle="--")

        flip_marker = " <->" if flipped else ""
        title = f"[{stem}]{flip_marker}\n{neuronA} :\n{neuronB}"
        ax.set_title(title, fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("lag (ms)" if lags is not None else "bin", fontsize=6)
        ax.set_ylabel("CCG", fontsize=6)

    # hide unused panels
    for idx in range(n, rows * cols):
        axes_flat[idx].set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--semifine",
        default=DEFAULT_SEMIFINE,
        help=f"semiFine directory to scan (default: {DEFAULT_SEMIFINE})",
    )
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help=f"Root data directory (default: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--session",
        default=DEFAULT_SESSION,
        help=f"Session ID used to build PKL path (default: {DEFAULT_SESSION})",
    )
    parser.add_argument(
        "--out",
        default="good_pairs_ccg.png",
        help="Output PNG filename (default: good_pairs_ccg.png)",
    )
    args = parser.parse_args()

    semifine_dir = Path(args.semifine)
    base_dir = Path(args.base)

    if not semifine_dir.exists():
        sys.exit(f"ERROR: semiFine directory not found: {semifine_dir}")

    print(f"Scanning: {semifine_dir}")
    pair_files = find_pair_files(semifine_dir)
    if not pair_files:
        sys.exit(f"ERROR: no {PAIRS_FILENAME} files found under {semifine_dir}")
    print(f"  -> {len(pair_files)} subfolder(s) found\n")

    entries = []
    for txt_path in pair_files:
        stem = txt_path.parent.name
        pairs = parse_pairs_txt(txt_path)
        pkl = ccg_pkl_path(base_dir, args.session, stem)
        exists = "OK" if pkl.exists() else "NOT FOUND"
        print(f"  [{exists}] {stem}  ({len(pairs)} pairs)")
        print(f"            {pkl}")
        for a, b in pairs:
            entries.append({"neuronA": a, "neuronB": b, "stem": stem, "pkl_path": pkl})

    print(f"\nTotal pairs to plot: {len(entries)}")
    print("Plotting ...")
    plot_grid(entries, args.out)


if __name__ == "__main__":
    main()
