#!/usr/bin/env python3
"""
Plot a 4×5 grid of crosscorrelograms for 20 neuron pairs listed in a TSV file.

TSV columns (tab-separated, no header):
  0  neuron_pair     e.g.  n007_L_CA3_wire_2_cell_3:n082_R_CA1_wire_5_cell_2
  1  session_id      e.g.  1089
  2  spike_pkl       e.g.  data/.../1089/1089u196merge-clean_cutoff_5.pkl
  3  output_dir      e.g.  data/.../single_pair_analysis/1089/1089u196merge-clean_cutoff_5

The crosscorrelogram PKL is derived from columns 1 & 2:
  session_id   ->  1089
  spike_pkl stem  ->  1089u196merge-clean_cutoff_5   (basename, no .pkl)
  CCG pkl  ->  <BASE_DIR>/analysis/<session_id>/<stem>/crosscorrs_edge_mean_True_semifine.pkl

Usage:
    python plot_crosscorrelograms.py <pairs_file.tsv> [--out OUTPUT.png] [--base BASE_DIR]
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Root directory that contains the  analysis/<session>/<stem>/  tree
DEFAULT_BASE = (
    "/project2/dsong_945/BenR/prescreening/data/"
    "Jan2010-Nonstationarity_Learning_DUPLICATE"
)
CCG_FILENAME = "crosscorrs_edge_mean_True_semifine.pkl"

DEFAULT_OUT = "crosscorrelograms.png"
GRID_ROWS, GRID_COLS = 4, 5


# ── helpers ───────────────────────────────────────────────────────────────────


def ccg_pkl_path(base: str, session_id: str, spike_pkl_col: str) -> Path:
    """
    Derive the crosscorrelogram PKL path from TSV columns.

    spike_pkl_col example:
        data/Jan2010-Nonstationarity_Learning/Rodent_WFU_DNMS/1089/1089u196merge-clean_cutoff_5.pkl

    Stem = basename without extension -> '1089u196merge-clean_cutoff_5'
    Result:
        <base>/analysis/<session_id>/<stem>/crosscorrs_edge_mean_True_semifine.pkl
    """
    stem = Path(spike_pkl_col).stem  # '1089u196merge-clean_cutoff_5'
    return Path(base) / "analysis" / session_id / stem / CCG_FILENAME


def parse_tsv(path: str) -> list:
    """Return list of row dicts parsed from the TSV."""
    rows = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                print(f"  [warn] line {lineno}: fewer than 3 columns - skipping")
                continue
            pair_str = cols[0]
            if ":" not in pair_str:
                print(
                    f"  [warn] line {lineno}: no ':' in pair field '{pair_str}' - skipping"
                )
                continue
            a, b = pair_str.split(":", 1)
            rows.append(
                {
                    "neuronA": a.strip(),
                    "neuronB": b.strip(),
                    "session_id": cols[1].strip(),
                    "spike_pkl": cols[2].strip(),
                }
            )
    return rows


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
    # tuple: (lags, ccg, mean, std, significance, threshold)
    if isinstance(value, tuple):
        lags = np.asarray(value[0])
        counts = np.asarray(value[1])
        return lags, counts

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
            arr = np.asarray(getattr(value, attr)).squeeze()
            lags = getattr(value, "lags", None)
            return lags, arr

    raise ValueError(f"Cannot extract CCG from value of type {type(value)}")


# ── plotting ──────────────────────────────────────────────────────────────────


def plot_grid(rows: list, base: str, out_path: str):
    if not rows:
        sys.exit("No pairs parsed from the input file.")

    # Cache open PKL files so the same file isn't reloaded multiple times
    pkl_cache = {}

    def get_data(pkl_path: Path):
        if pkl_path not in pkl_cache:
            pkl_cache[pkl_path] = load_pkl(pkl_path)
        return pkl_cache[pkl_path]

    n = min(len(rows), GRID_ROWS * GRID_COLS)
    if len(rows) > GRID_ROWS * GRID_COLS:
        print(
            f"[warn] {len(rows)} pairs but grid holds {GRID_ROWS*GRID_COLS}; extras ignored."
        )

    fig, axes = plt.subplots(
        GRID_ROWS,
        GRID_COLS,
        figsize=(GRID_COLS * 3.2, GRID_ROWS * 2.6),
        constrained_layout=True,
    )
    fig.suptitle("Crosscorrelograms", fontsize=14, fontweight="bold")

    for idx, row in enumerate(rows[:n]):
        r, c = divmod(idx, GRID_COLS)
        ax = axes[r][c]

        neuronA = row["neuronA"]
        neuronB = row["neuronB"]
        session_id = row["session_id"]
        spike_pkl = row["spike_pkl"]

        def short(name):
            parts = name.split("_")
            return "_".join(parts[-3:]) if len(parts) >= 3 else name

        base_title = f"#{idx+1}\n{short(neuronA)} :\n{short(neuronB)}"

        pkl_path = ccg_pkl_path(base, session_id, spike_pkl)
        stem = pkl_path.parent.name  # e.g. 1089u196merge-clean_cutoff_5

        if not pkl_path.exists():
            ax.text(
                0.5,
                0.5,
                f"PKL not found:\n.../{stem}/\n{CCG_FILENAME}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="red",
                fontsize=6.5,
                wrap=True,
            )
            ax.set_title(base_title, fontsize=6.5, color="red")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        try:
            data = get_data(pkl_path)

            # DEBUG - remove after
            first_key = next(iter(data))
            print(f"DEBUG sample key: {first_key}")
            print(f"DEBUG sample value type: {type(data[first_key])}")
            print(f"DEBUG sample value: {data[first_key]}")

        except Exception as exc:
            ax.text(
                0.5,
                0.5,
                f"Load error:\n{exc}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="orange",
                fontsize=6.5,
            )
            ax.set_title(base_title, fontsize=6.5)
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
            ax.set_title(base_title, fontsize=6.5, color="red")
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
                fontsize=6.5,
            )
            ax.set_title(base_title, fontsize=6.5)
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
        title = f"#{idx+1}{flip_marker}  [{stem}]\n{short(neuronA)} :\n{short(neuronB)}"
        ax.set_title(title, fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("lag (ms)" if lags is not None else "bin", fontsize=6)
        ax.set_ylabel("count", fontsize=6)

    # hide unused panels
    for idx in range(n, GRID_ROWS * GRID_COLS):
        r, c = divmod(idx, GRID_COLS)
        axes[r][c].set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pairs_file", help="Tab-separated TSV of neuron pairs")
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help=f"Root data directory (default: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output PNG filename (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    print(f"Reading pairs from : {args.pairs_file}")
    rows = parse_tsv(args.pairs_file)
    print(f"  -> {len(rows)} pairs parsed\n")

    for i, row in enumerate(rows, 1):
        pkl = ccg_pkl_path(args.base, row["session_id"], row["spike_pkl"])
        exists = "OK" if pkl.exists() else "NOT FOUND"
        print(f"  {i:>2}. {row['neuronA']} : {row['neuronB']}")
        print(f"       [{exists}] {pkl}")

    print("\nPlotting ...")
    plot_grid(rows, args.base, args.out)


if __name__ == "__main__":
    main()
