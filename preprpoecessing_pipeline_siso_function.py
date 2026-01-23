import math
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
import glob
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import (
    check_correlation_filled_bins,
    check_correlations_unimodal,
    filter_pairs_using_correlation_filled_bins,
    check_histogram_unimodal,
    filter_pairs_using_unimodality,
)
from utils.plot_all_plots_for_siso_cc import (
    plot_all_neurons_silent_periods,
    plot_spike_raster,
    plot_neuron_correlation_matrices,
    compute_correlogram_normalized,
)

"""
Pipeline to preprocess spike train data from .pkl files
Actually does the ranking here
"""


def load_config(config_input):
    """
    Load configuration from either a file path or a dictionary.

    Args:
        config_input: Either a path to JSON config file or a config dictionary

    Returns:
        dict: Configuration dictionary
    """
    if isinstance(config_input, str):
        # If string, treat as file path
        with open(config_input, "r") as f:
            config = json.load(f)
    elif isinstance(config_input, dict):
        # If dict, use directly
        config = config_input
    else:
        raise ValueError("Config must be either a file path (str) or dictionary")

    return config


# Function to load neurons from a .pkl file
def load_neurons(pkl_path, length_of_spiketrain=None):
    """Load spike train data from a .pkl file and handle potential binary spike train format."""
    with open(pkl_path, "rb") as f:
        neurons = pickle.load(f)
    # Check if data is a binary spike train (many zeros) and convert to spike indices
    first_key = next(iter(neurons.keys()))
    if np.sum(np.array(neurons[first_key]) == 0) > 100:
        neurons = {k: np.where(v)[0] for k, v in neurons.items()}

    if length_of_spiketrain is not None:
        neurons = {k: v[v < length_of_spiketrain] for k, v in neurons.items()}

    return neurons


# Function to filter neurons based on minimum spike count
def filter_neurons(neurons, min_spikes):
    """
    Filter neurons with at least min_spikes.
    Note: Pipeline frame says 'less than MIN_SPIKES', but this is likely a typo;
    assuming 'at least MIN_SPIKES' as is standard.
    """
    return {k: v for k, v in neurons.items() if len(v) >= min_spikes}


# # Function to compute cross-correlations for all neuron pairs
# def compute_all_crosscorrs(neurons, bin_size, max_lag, sample_rate):
#     """Compute normalized cross-correlograms and z-scores for all neuron pairs."""
#     neuron_spike_trains, global_max_time, firing_rate = {}, 0, {}
#     for neuron_id, spike_times in neurons.items():
#         # Convert spike times to milliseconds
#         spike_times = np.array(spike_times) / sample_rate * 1000
#         spike_indices = np.round(spike_times).astype(int)
#         global_max_time = max(
#             global_max_time, int(np.max(spike_indices)) if len(spike_indices) > 0 else 0
#         )
#         num_bins = int(np.ceil(global_max_time / bin_size)) + 1
#         neuron_spike_trains[neuron_id] = (
#             np.histogram(spike_indices, bins=num_bins, range=(0, global_max_time))[0]
#             > 0
#         )
#         firing_rate[neuron_id] = (
#             np.sum(neuron_spike_trains[neuron_id]) / (global_max_time / 1000)
#             if global_max_time > 0
#             else 0
#         )

#     crosscorrs = {}
#     neuron_ids = list(neurons.keys())
#     for i, pre in enumerate(neuron_ids):
#         for post in neuron_ids[i + 1 :]:
#             (
#                 lags,
#                 corr_normalized,
#                 mean_normalized,
#                 std_normalized,
#                 z_scores,
#                 total_bump_score,
#             ) = compute_correlogram_normalized(
#                 neuron_spike_trains[pre].astype(float),
#                 neuron_spike_trains[post].astype(float),
#                 max_lag,
#                 bin_size,
#                 "cross",
#                 firing_rate[pre],
#                 firing_rate[post],
#                 global_max_time,
#             )
#             crosscorrs[(pre, post)] = {
#                 "lags": lags,
#                 "corr_normalized": corr_normalized,
#                 "z_scores": z_scores,
#                 "total_bump_score": total_bump_score,
#             }
#     return crosscorrs, firing_rate


# runs the full pipeline:
# 1. load config; 2. load .pkl; 3. filter neurons; 4. compute cross-correlations; 5. rank pairs; 6. save results
def run_preprocessing_pipeline(config_input, verbose=True):
    """
    Run the preprocessing pipeline with given configuration.

    Args:
        config_input: Either path to JSON config file or config dictionary
        verbose: Whether to print progress messages

    Returns:
        dict: Results summary with processed data paths and statistics
    """
    # Load configuration
    config = load_config(config_input)

    # Extract parameters
    FILE_DIR = config["paths"]["file_dir"]
    ANALYSIS_DIR = os.path.join(FILE_DIR, config["paths"]["analysis_dir"])
    PKL_FILE_PATTERN = config["paths"][
        "pkl_file"
    ]  # Now a wildcard pattern, e.g., "*.pkl"
    SAMPLE_RATE = float(config["processing"]["sample_rate"])
    MIN_SPIKES = int(config["processing"]["min_spikes"])
    N_TOP = int(
        config["processing"]["n_top"]
    )  # <-- make n_top include everything unless otherwise specified
    PLOT_ALL = config["plotting"]["plot_all"]  # JSON handles boolean directly
    RASTER_BIN_SIZE = int(config["plotting"]["raster_bin_size"])
    CONFIGS = config["cc_configs"]  # Load configs from JSON
    edge_mean = True
    RECOMPUTE = config["processing"]["recompute_cc"]

    results = {
        "processed_files": [],
        "summary_paths": [],
        "cross_correlation_paths": [],
        "ranking_paths": [],
        "statistics": {},
    }

    # Set up directories
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    pkl_files = glob.glob(os.path.join(FILE_DIR, PKL_FILE_PATTERN))

    if verbose:
        print("pkl_files include:\n  ", pkl_files)

    if not pkl_files:
        if verbose:
            print(f"No .pkl files found matching {PKL_FILE_PATTERN} in {FILE_DIR}")
        return results

    for pkl_path in pkl_files:
        if verbose:
            print(f"Processing {os.path.basename(pkl_path)}...")

        save_dir = os.path.join(
            ANALYSIS_DIR, os.path.splitext(os.path.basename(pkl_path))[0]
        )
        os.makedirs(save_dir, exist_ok=True)

        # **Stage 1: Load the spiketrain data from the .pkl file**
        neurons = load_neurons(pkl_path)
        n_before = len(neurons)

        # **Stage 2: Filter neurons for spike count**
        filtered_neurons = filter_neurons(neurons, MIN_SPIKES)
        n_after = len(filtered_neurons)

        # **Stage 3: (Optional) Generate various figures**
        if PLOT_ALL:
            plot_all_neurons_silent_periods(filtered_neurons, save_dir, SAMPLE_RATE)
            plot_spike_raster(
                filtered_neurons, save_dir, SAMPLE_RATE, bin_size=RASTER_BIN_SIZE
            )
            plot_neuron_correlation_matrices(
                filtered_neurons,
                save_dir,
                SAMPLE_RATE,
                edge_mean=edge_mean,
                configs=CONFIGS,
            )

        file_results = {
            "file": pkl_path,
            "n_neurons_before": n_before,
            "n_neurons_after": n_after,
            "configurations": [],
        }

        for item in CONFIGS:
            bin_size = item[0]
            max_lag = item[1]
            resolution = item[2]

            if verbose:
                print(f"crosscorrs for {resolution}")

            if (
                not os.path.exists(
                    os.path.join(
                        save_dir,
                        f"crosscorrs_edge_mean_{edge_mean}_{resolution.lower()}.pkl",
                    )
                )
                or RECOMPUTE
            ):
                config_single = [item]
                if verbose:
                    print(f"Computing crosscorrs for {config_single}")
                plot_neuron_correlation_matrices(
                    filtered_neurons,
                    save_dir,
                    SAMPLE_RATE,
                    edge_mean=edge_mean,
                    configs=config_single,
                    make_plots=False,
                )

            crosscorrs = pickle.load(
                open(
                    os.path.join(
                        save_dir,
                        f"crosscorrs_edge_mean_{edge_mean}_{resolution.lower()}.pkl",
                    ),
                    "rb",
                )
            )

            # Identify top N positive and negative correlation pairs
            neuron_ids = list(filtered_neurons.keys())
            pairs = [
                (pre, post)
                for i, pre in enumerate(neuron_ids)
                for post in neuron_ids[i + 1 :]
            ]

            # **Stage 4: Filter out the obviously bad pairs using "histogram base-filled"; bump score thresholds; unimodality test

            check_correlation_filled_bins(
                pkl_path="data/analysis/selected_neurons_first_200s/crosscorrs_edge_mean_True_ultra-fine.pkl",  # find a way to automatically get this path
                out_dir="selected_neurons_first_200s",
                harshness=0.10,  # at most 10% of bins can be empty
                power_threshold=-10,  # bins with value < e^-10 are considered empty
            )

            # filter pairs using correlation filled bins
            filtered_pairs_binfill = filter_pairs_using_correlation_filled_bins(
                pairs,
                bad_pairs_path="selected_neurons_first_200s\\bad_pairs.txt",
            )
            # print("filtered_pairs", filtered_pairs)

            # filter pairs using unimodality test
            check_correlations_unimodal(
                pkl_path="data/analysis/selected_neurons_first_200s/crosscorrs_edge_mean_True_ultra-fine.pkl",  # find a way to automatically get this path
                out_dir="unimodal_selected_neurons_first_200s",  # for debugging purposes
                bin_centers=None,
                smoothing_sigma=1.0,
                prominence_fraction=0.25,
                min_distance_bins=1,
            )

            filtered_pairs = filter_pairs_using_unimodality(
                filtered_pairs_binfill,
                bad_pairs_path="unimodal_selected_neurons_first_200s\\unimodal_bad_pairs.txt",
            )

            top_bump = sorted(
                [(pair, np.max(crosscorrs[pair][5])) for pair in filtered_pairs],
                key=lambda x: x[1],
                reverse=True,
            )[:N_TOP]
            bottom_bump = sorted(
                [(pair, np.min(crosscorrs[pair][5])) for pair in filtered_pairs],
                key=lambda x: x[1],
            )[:N_TOP]

            # print("correlation_filled_bins filtered top_bump", top_bump)
            # print("correlation_filled_bins filtered bottom_bump", bottom_bump)

            # **Stage 5: Generate cross-correlation plot for top N and bottom N pairs**
            plots_per_row = 10
            n_rows_top = (N_TOP + plots_per_row - 1) // plots_per_row
            n_rows_bottom = (N_TOP + plots_per_row - 1) // plots_per_row

            fig_width = max(15, min(plots_per_row * 5, N_TOP * 5))
            fig_height = 6 * (n_rows_top + n_rows_bottom)
            fig, axes = plt.subplots(
                n_rows_top + n_rows_bottom,
                min(plots_per_row, N_TOP),
                figsize=(fig_width, fig_height),
            )

            # Top bump score pairs
            for i, (pair, bump_score) in enumerate(top_bump):
                row = i // plots_per_row
                col = i % plots_per_row
                if N_TOP > 1:
                    ax = axes[row, col]
                else:
                    ax = axes[row]
                lags = crosscorrs[pair][0]
                corr_normalized = crosscorrs[pair][1]
                bump_scores = crosscorrs[pair][4]
                bump_scores_min = np.min(bump_scores)
                bump_scores_max = np.max(bump_scores)
                mean_corr = crosscorrs[pair][2]
                std_corr = crosscorrs[pair][3]
                eps = 1e-10

                bump_scores_normalized = (bump_scores - bump_scores_min) / (
                    bump_scores_max - bump_scores_min + eps
                )
                bump_scores_normalized = 0.2 + 0.2 * np.log10(
                    1 + 9 * np.clip(bump_scores_normalized, 0, 1)
                ) / np.log10(10)

                for idx in range(len(lags) - 1):
                    ax.fill_between(
                        [lags[idx], lags[idx + 1]],
                        [0, 0],
                        [corr_normalized[idx], corr_normalized[idx]],
                        color="red",
                        alpha=0.8,
                        linewidth=0.5,
                    )
                ax.set_title(
                    f"{pair[0]}-{pair[1]}\nbump_score={bump_score:.2f}", fontsize=10
                )
                ax.set_xlabel("Lag (ms)")
                ax.set_ylabel("Normalized CC")
                ax.axhline(y=mean_corr, color="gray", linestyle="-")
                ax.axhline(y=mean_corr + std_corr, color="gray", linestyle="--")
                ax.axhline(y=mean_corr - std_corr, color="gray", linestyle="--")

            # Bottom bump score pairs
            for i, (pair, bump_score) in enumerate(bottom_bump):
                row = n_rows_top + (i // plots_per_row)
                col = i % plots_per_row
                if N_TOP > 1:
                    ax = axes[row, col]
                else:
                    ax = axes[row]
                lags = crosscorrs[pair][0]
                corr_normalized = crosscorrs[pair][1]
                bump_scores = crosscorrs[pair][4]
                bump_scores_min = np.min(bump_scores)
                bump_scores_max = np.max(bump_scores)
                mean_corr = crosscorrs[pair][2]
                std_corr = crosscorrs[pair][3]
                eps = 1e-10
                bump_scores_normalized = (bump_scores - bump_scores_min) / (
                    bump_scores_max - bump_scores_min + eps
                )
                bump_scores_normalized = 0.3 + 0.3 * np.log10(
                    1 + 9 * np.clip(bump_scores_normalized, 0, 1)
                ) / np.log10(10)

                for idx in range(len(lags) - 1):
                    ax.fill_between(
                        [lags[idx], lags[idx + 1]],
                        [0, 0],
                        [corr_normalized[idx], corr_normalized[idx]],
                        color="black",
                        alpha=0.8,
                        linewidth=0.5,
                    )
                ax.set_title(
                    f"{pair[0]}-{pair[1]}\nbump_score={bump_score:.2f}", fontsize=10
                )
                ax.set_xlabel("Lag (ms)")
                ax.set_ylabel("Normalized CC")
                ax.axhline(y=mean_corr, color="gray", linestyle="-")
                ax.axhline(y=mean_corr + std_corr, color="gray", linestyle="--")
                ax.axhline(y=mean_corr - std_corr, color="gray", linestyle="--")

            # Remove empty subplots if any
            if N_TOP > 1:
                for i in range(N_TOP, n_rows_top * plots_per_row):
                    row = i // plots_per_row
                    col = i % plots_per_row
                    fig.delaxes(axes[row, col])
                for i in range(N_TOP, n_rows_bottom * plots_per_row):
                    row = n_rows_top + (i // plots_per_row)
                    col = i % plots_per_row
                    fig.delaxes(axes[row, col])

            plt.tight_layout()
            cc_plot_path = os.path.join(
                save_dir, f"top_cc_plots_{resolution.lower()}.png"
            )
            plt.savefig(cc_plot_path)
            plt.close()

            # make a figure of bump score distribution
            plt.figure()
            plt.hist(bump_scores, bins=20)
            plt.savefig(
                os.path.join(
                    save_dir, f"bump_score_distribution_{resolution.lower()}.png"
                )
            )
            plt.close()

            # **Stage 6: Save data, plots, and summary**
            summary_path = os.path.join(save_dir, "summary.txt")

            # Identify top and bottom pairs based on total bump score
            bump_score_pairs = [(pair, crosscorrs[pair][5]) for pair in pairs]
            top_bump_pairs = sorted(bump_score_pairs, key=lambda x: x[1], reverse=True)[
                :N_TOP
            ]
            bottom_bump_pairs = sorted(bump_score_pairs, key=lambda x: x[1])[:N_TOP]

            # Create DataFrame for all pairs with their bump scores
            df_ranks = pd.DataFrame(
                {
                    "Neuron1": [pair[0] for pair, _ in bump_score_pairs],
                    "Neuron2": [pair[1] for pair, _ in bump_score_pairs],
                    "BumpScore": [score for _, score in bump_score_pairs],
                }
            )
            df_ranks = df_ranks.sort_values("BumpScore", ascending=False)
            df_ranks["Rank"] = range(1, len(df_ranks) + 1)

            # Save rankings to CSV
            csv_path = os.path.join(save_dir, f"pair_rankings_{resolution.lower()}.csv")
            df_ranks.to_csv(csv_path, index=False)

            # Write summary text file
            mode = "a" if os.path.exists(summary_path) else "w"
            with open(summary_path, mode) as f:
                f.write(f"Dataset: {os.path.basename(pkl_path)}\n")
                f.write(f"Bin size: {bin_size}ms\n")
                f.write(f"Max lag: {max_lag}ms\n")
                f.write(f"Resolution: {resolution}\n")
                f.write(f"Number of neurons before filtering: {n_before}\n")
                f.write(f"Number of neurons after filtering: {n_after}\n")
                f.write(f"Number of pairs: {len(pairs)}\n")
                f.write("Top pairs by total bump score:\n")
                for pair, bump_score in top_bump_pairs:
                    f.write(f"Pair {pair}: total_bump_score={bump_score:.2f}\n")
                f.write("Bottom pairs by total bump score:\n")
                for pair, bump_score in bottom_bump_pairs:
                    f.write(f"Pair {pair}: total_bump_score={bump_score:.2f}\n")
                if PLOT_ALL:
                    f.write(
                        f"Additional plots (silent periods, raster, autocorrelations, etc.) saved in: {save_dir}\n"
                    )
                f.write(f"Pair rankings saved in: {csv_path}\n")
                f.write("-" * 50 + "\n")

            # Store configuration results
            config_result = {
                "resolution": resolution,
                "bin_size": bin_size,
                "max_lag": max_lag,
                "n_pairs": len(pairs),
                "summary_path": summary_path,
                "cc_plot_path": cc_plot_path,
                "csv_path": csv_path,
                "top_bump_pairs": top_bump_pairs,
                "bottom_bump_pairs": bottom_bump_pairs,
            }
            file_results["configurations"].append(config_result)

            if verbose:
                print(
                    f"Pipeline updated for {resolution}. Summary saved at {summary_path}"
                )
                print(f"Cross-correlation plot saved at {cc_plot_path}")
                print(f"Pair rankings saved at {csv_path}")

        results["processed_files"].append(file_results)
        results["summary_paths"].append(summary_path)

    return results


def main():
    """
    Main function for standalone script execution.
    Uses default config file paths for backward compatibility.
    """

    default_configs = ["analysis_pipeline/config_dnms.json"]

    # Try to find an existing config file
    config_path = None
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(base_dir)
    for config_file in default_configs:
        full_path = os.path.join(base_dir, config_file)
        if os.path.exists(full_path):
            config_path = full_path
            break

    print(f"Using config file: {config_path}")
    results = run_preprocessing_pipeline(config_path)
    print(f"Processing complete. Processed {len(results['processed_files'])} files.")


if __name__ == "__main__":
    main()
