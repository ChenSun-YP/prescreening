import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import argparse
import glob
import pandas as pd

# Basic: line-buffered stdout/stderr so SLURM log shows output while running
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot_all_plots_for_siso_cc import (
    plot_all_neurons_silent_periods,
    plot_spike_raster,
    plot_neuron_correlation_matrices,
    compute_correlogram_normalized
)

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
        with open(config_input, 'r') as f:
            config = json.load(f)
    elif isinstance(config_input, dict):
        # If dict, use directly
        config = config_input
    else:
        raise ValueError("Config must be either a file path (str) or dictionary")
    
    return config

# Function to load neurons from a .pkl file
def load_neurons(pkl_path, length_of_spiketrain=None):
    """Load spike train data from a .pkl file.

    Handles two formats:
    1. Simple: dict mapping neuron_id -> array of spike times (or binary spike train).
    2. Nested: dict with metadata and a 'neurons' key containing a list of dicts
       with 'name' and 'timestamps' (e.g. from NeuroSuite/neuroscope-style exports).

    Returns:
        neurons: dict of neuron_id -> spike times (full duration; caller applies cutoff if needed).
        rec_info: dict with 'tend' (total recording duration) and 'last_trial_end' (last TRIAL end time)
                  for nested format; both None for simple format.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    rec_info = {'tend': None, 'last_trial_end': None}

    # Nested format: {'neurons': [{'name': ..., 'timestamps': [...]}, ...], 'tend': ..., 'intervals': ...}
    if isinstance(data, dict) and 'neurons' in data:
        neurons_list = data['neurons']
        if not isinstance(neurons_list, (list, tuple)):
            raise ValueError("Expected 'neurons' to be a list of dicts with 'name' and 'timestamps'")
        neurons = {}
        for item in neurons_list:
            if isinstance(item, dict) and 'name' in item and 'timestamps' in item:
                name = item['name']
                ts = np.asarray(item['timestamps'], dtype=float)
                neurons[name] = ts
            else:
                raise ValueError(f"Each entry in 'neurons' must have 'name' and 'timestamps'; got keys: {item.keys() if isinstance(item, dict) else type(item)}")
        if 'tend' in data:
            rec_info['tend'] = float(data['tend'])
        if 'events' in data:
            trials = data['events'][-1] if  data['events'][-1]['name'] == 'TRIAL' else None
            if trials is not None:
                rec_info['last_trial_end'] = trials['timestamps'][-1]

    else:
        # Simple format: dict of neuron_id -> spike array
        neurons = {k: np.asarray(v, dtype=float) for k, v in data.items()}

    # Detect binary spike train (many zeros) and convert to spike indices
    if neurons:
        first_key = next(iter(neurons.keys()))
        arr = np.asarray(neurons[first_key])
        if arr.size > 100 and np.sum(arr == 0) > 100:
            neurons = {k: np.where(np.asarray(v))[0] for k, v in neurons.items()}

    # Optional caller-provided cutoff (e.g. for simple format); do not apply rec_info cutoff here
    if length_of_spiketrain is not None:
        neurons = {k: v[v < length_of_spiketrain] for k, v in neurons.items()}

    return neurons, rec_info


# Function to filter neurons based on minimum spike count
def filter_neurons(neurons, min_spikes):
    """
    Filter neurons with at least min_spikes.
    Note: Pipeline frame says 'less than MIN_SPIKES', but this is likely a typo;
    assuming 'at least MIN_SPIKES' as is standard.
    """
    return {k: v for k, v in neurons.items() if len(v) >= min_spikes}

# Function to compute cross-correlations for all neuron pairs
# def compute_all_crosscorrs(neurons, bin_size, max_lag, sample_rate):
#     """Compute normalized cross-correlograms and z-scores for all neuron pairs."""
#     neuron_spike_trains, global_max_time, firing_rate = {}, 0, {}
#     for neuron_id, spike_times in neurons.items():
#         # Convert spike times to milliseconds
#         spike_times = np.array(spike_times) / sample_rate * 1000
#         spike_indices = np.round(spike_times).astype(int)
#         global_max_time = max(global_max_time, int(np.max(spike_indices)) if len(spike_indices) > 0 else 0)
#         num_bins = int(np.ceil(global_max_time / bin_size)) + 1
#         neuron_spike_trains[neuron_id] = np.histogram(spike_indices, bins=num_bins, range=(0, global_max_time))[0] > 0
#         firing_rate[neuron_id] = np.sum(neuron_spike_trains[neuron_id]) / (global_max_time / 1000) if global_max_time > 0 else 0
    
#     crosscorrs = {}
#     neuron_ids = list(neurons.keys())
#     for i, pre in enumerate(neuron_ids):
#         for post in neuron_ids[i+1:]:
#             lags, corr_normalized, mean_normalized, std_normalized, z_scores, total_bump_score = compute_correlogram_normalized(
#                 neuron_spike_trains[pre].astype(float),
#                 neuron_spike_trains[post].astype(float),
#                 max_lag,
#                 bin_size,
#                 'cross',
#                 firing_rate[pre],
#                 firing_rate[post],
#                 global_max_time
#             )
#             crosscorrs[(pre, post)] = {
#                 'lags': lags,
#                 'corr_normalized': corr_normalized,
#                 'z_scores': z_scores,
#                 'total_bump_score': total_bump_score
#             }
#     return crosscorrs, firing_rate

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
    FILE_DIR = config['paths']['file_dir'].rstrip("/")
    # Use basename of file_dir so each job (e.g. .../1150_5_sec) writes to analysis_dir/1150_5_sec
    _output_subdir = os.path.basename(FILE_DIR)
    ANALYSIS_DIR = os.path.join(config['paths']['analysis_dir'], _output_subdir)
    PKL_FILE_PATTERN = config['paths']['pkl_file']  # Now a wildcard pattern, e.g., "*.pkl"
    SAMPLE_RATE = float(config['processing']['sample_rate'])
    MIN_SPIKES = int(config['processing']['min_spikes'])
    N_TOP = int(config['processing']['n_top'])
    PLOT_ALL = config['plotting']['plot_all']  # JSON handles boolean directly
    RASTER_BIN_SIZE = int(config['plotting']['raster_bin_size'])
    CONFIGS = config['cc_configs']  # Load configs from JSON
    edge_mean = True
    RECOMPUTE = config['processing']['recompute_cc']
    
    results = {
        'processed_files': [],
        'summary_paths': [],
        'cross_correlation_paths': [],
        'ranking_paths': [],
        'statistics': {}
    }

    # Set up directories (analysis_dir/file_dir for outputs)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    pkl_files = glob.glob(os.path.join(FILE_DIR, PKL_FILE_PATTERN))
    
    if verbose:
        print('pkl_files include:\n  ', pkl_files)

    if not pkl_files:
        if verbose:
            print(f"No .pkl files found matching {PKL_FILE_PATTERN} in {FILE_DIR}")
        return results
        
    for pkl_path in pkl_files:
        if verbose:
            print(f"Processing {os.path.basename(pkl_path)}...")

        # Path: analysis_dir/file_dir/[session_stem/]pkl_file_stem (session = pkl's parent rel to file_dir)
        pkl_dir = os.path.dirname(pkl_path)
        pkl_stem = os.path.splitext(os.path.basename(pkl_path))[0]
        try:
            session_part = os.path.relpath(pkl_dir, FILE_DIR)
        except ValueError:
            session_part = ""
        if session_part in (".", ""):
            save_dir = os.path.join(ANALYSIS_DIR, pkl_stem)
        else:
            save_dir = os.path.join(ANALYSIS_DIR, session_part, pkl_stem)
        os.makedirs(save_dir, exist_ok=True)

        # **Stage 1: Load the spiketrain data from the .pkl file**
        neurons_full, rec_info = load_neurons(pkl_path)
        n_before = len(neurons_full)

        # Cutoff for CC, AC, and silence: use last TRIAL end, else recording tend
        cutoff = rec_info.get('last_trial_end') or rec_info.get('tend')
        if cutoff is not None:
            neurons = {k: v[v < cutoff] for k, v in neurons_full.items()}
        else:
            neurons = neurons_full

        if verbose:
            print(f"  Total recording duration (tend): {rec_info['tend']}")
            print(f"  Last TRIAL end (cutoff for CC/AC/silence): {rec_info['last_trial_end']}")
            if cutoff is not None:
                print(f"  Using spikes with t < {cutoff} for cross-correlation, autocorrelation, and silence-period analysis.")

        # **Stage 2: Filter neurons**
        filtered_neurons = filter_neurons(neurons, MIN_SPIKES)
        n_after = len(filtered_neurons)

        # **Stage 3: (Optional) Generate various figures**
        if PLOT_ALL:
            plot_all_neurons_silent_periods(filtered_neurons, save_dir, SAMPLE_RATE)
            # Raster: same cut data as all other analysis; dashed line at last trial end
            plot_spike_raster(filtered_neurons, save_dir, SAMPLE_RATE, bin_size=RASTER_BIN_SIZE, cutoff_time=rec_info.get('last_trial_end'))
            plot_neuron_correlation_matrices(filtered_neurons, save_dir, SAMPLE_RATE,edge_mean=edge_mean,configs=CONFIGS)

        file_results = {
            'file': pkl_path,
            'n_neurons_before': n_before,
            'n_neurons_after': n_after,
            'configurations': []
        }

        # Write recording/trial info to summary once
        summary_path = os.path.join(save_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f'Dataset: {os.path.basename(pkl_path)}\n')
            f.write(f'Total recording duration (tend): {rec_info["tend"]}\n')
            f.write(f'Last TRIAL end: {rec_info["last_trial_end"]}\n')
            if rec_info['last_trial_end'] is not None or rec_info['tend'] is not None:
                f.write('All analysis (CC, AC, silence-period, raster) use only spikes with t < last TRIAL end (cutoff).\n')
                f.write('Spike raster plot shows data up to cutoff with a dashed vertical line at last TRIAL end.\n')
            f.write('-' * 50 + '\n')

        for item in CONFIGS:
            bin_size = item[0]
            max_lag = item[1]
            resolution = item[2]
            
            if verbose:
                print(f'crosscorrs for {resolution}')

            if not os.path.exists(os.path.join(save_dir, f'crosscorrs_edge_mean_{edge_mean}_{resolution.lower()}.pkl')) or RECOMPUTE:
                config_single = [item]
                if verbose:
                    print(f'Computing crosscorrs for {config_single}')
                plot_neuron_correlation_matrices(filtered_neurons, save_dir, SAMPLE_RATE,edge_mean=edge_mean,configs=config_single,make_plots=False)

            crosscorrs = pickle.load(open(os.path.join(save_dir, f'crosscorrs_edge_mean_{edge_mean}_{resolution.lower()}.pkl'), 'rb'))

            # Identify top N positive and negative correlation pairs
            neuron_ids = list(filtered_neurons.keys())
            pairs = [(pre, post) for i, pre in enumerate(neuron_ids) for post in neuron_ids[i+1:]]
            
            top_bump = sorted(
                [(pair, np.max(crosscorrs[pair][5]))
                for pair in pairs],
                key=lambda x: x[1],
                reverse=True
            )[:N_TOP]
            bottom_bump = sorted(
                [(pair, np.min(crosscorrs[pair][5]))
                for pair in pairs],
                key=lambda x: x[1]
            )[:N_TOP]

            # **Stage 5: Generate cross-correlation plot for top N and bottom N pairs**
            plots_per_row = 10
            n_rows_top = (N_TOP + plots_per_row - 1) // plots_per_row
            n_rows_bottom = (N_TOP + plots_per_row - 1) // plots_per_row
            
            fig_width = max(15, min(plots_per_row * 5, N_TOP * 5))
            fig_height = 6 * (n_rows_top + n_rows_bottom)
            fig, axes = plt.subplots(n_rows_top + n_rows_bottom, min(plots_per_row, N_TOP), figsize=(fig_width, fig_height))
            
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

                bump_scores_normalized = (bump_scores - bump_scores_min) / (bump_scores_max - bump_scores_min + eps)
                bump_scores_normalized = 0.2 + 0.2 * np.log10(1 + 9 * np.clip(bump_scores_normalized, 0, 1)) / np.log10(10)
                
                for idx in range(len(lags)-1):
                    ax.fill_between(
                        [lags[idx], lags[idx+1]],
                        [0, 0],
                        [corr_normalized[idx], corr_normalized[idx]],
                        color='red',
                        alpha = 0.8,
                        linewidth=0.5,
                                )
                ax.set_title(f'{pair[0]}-{pair[1]}\nbump_score={bump_score:.2f}', fontsize=10)
                ax.set_xlabel('Lag (ms)')
                ax.set_ylabel('Normalized CC')
                ax.axhline(y=mean_corr, color='gray', linestyle='-')
                ax.axhline(y=mean_corr+std_corr, color='gray', linestyle='--')
                ax.axhline(y=mean_corr-std_corr, color='gray', linestyle='--')
                
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
                bump_scores_normalized = (bump_scores - bump_scores_min) / (bump_scores_max - bump_scores_min + eps)
                bump_scores_normalized = 0.3 + 0.3 * np.log10(1 + 9 * np.clip(bump_scores_normalized, 0, 1)) / np.log10(10)
                
                for idx in range(len(lags)-1):
                    ax.fill_between(
                        [lags[idx], lags[idx+1]],
                        [0, 0],
                        [corr_normalized[idx], corr_normalized[idx]],
                        color='black',
                        alpha = 0.8,
                        linewidth=0.5,
                                )
                ax.set_title(f'{pair[0]}-{pair[1]}\nbump_score={bump_score:.2f}', fontsize=10)
                ax.set_xlabel('Lag (ms)')
                ax.set_ylabel('Normalized CC')
                ax.axhline(y=mean_corr, color='gray', linestyle='-')
                ax.axhline(y=mean_corr+std_corr, color='gray', linestyle='--')
                ax.axhline(y=mean_corr-std_corr, color='gray', linestyle='--')
                
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
            cc_plot_path = os.path.join(save_dir, f'top_cc_plots_{resolution.lower()}.png')
            plt.savefig(cc_plot_path)
            plt.close()

            # make a figure of bump score distribution
            plt.figure()
            plt.hist(bump_scores, bins=20)
            plt.savefig(os.path.join(save_dir, f'bump_score_distribution_{resolution.lower()}.png'))
            plt.close()

            # **Stage 6: Save data, plots, and summary**
            # Identify top and bottom pairs based on total bump score
            bump_score_pairs = [(pair, crosscorrs[pair][5]) for pair in pairs]
            top_bump_pairs = sorted(bump_score_pairs, key=lambda x: x[1], reverse=True)[:N_TOP]
            bottom_bump_pairs = sorted(bump_score_pairs, key=lambda x: x[1])[:N_TOP]
            
            # Create DataFrame for all pairs with their bump scores
            df_ranks = pd.DataFrame({
                'Neuron1': [pair[0] for pair, _ in bump_score_pairs],
                'Neuron2': [pair[1] for pair, _ in bump_score_pairs],
                'BumpScore': [score for _, score in bump_score_pairs]
            })
            df_ranks = df_ranks.sort_values('BumpScore', ascending=False)
            df_ranks['Rank'] = range(1, len(df_ranks) + 1)
            
            # Save rankings to CSV
            csv_path = os.path.join(save_dir, f'pair_rankings_{resolution.lower()}.csv')
            df_ranks.to_csv(csv_path, index=False)
            
            # Append per-resolution summary
            with open(summary_path, 'a') as f:
                f.write(f'Bin size: {bin_size}ms\n')
                f.write(f'Max lag: {max_lag}ms\n')
                f.write(f'Resolution: {resolution}\n')
                f.write(f'Number of neurons before filtering: {n_before}\n')
                f.write(f'Number of neurons after filtering: {n_after}\n')
                f.write(f'Number of pairs: {len(pairs)}\n')
                f.write('Top pairs by total bump score:\n')
                for pair, bump_score in top_bump_pairs:
                    f.write(f'Pair {pair}: total_bump_score={bump_score:.2f}\n')
                f.write('Bottom pairs by total bump score:\n')
                for pair, bump_score in bottom_bump_pairs:
                    f.write(f'Pair {pair}: total_bump_score={bump_score:.2f}\n')
                if PLOT_ALL:
                    f.write(f'Additional plots (silent periods, raster, autocorrelations, etc.) saved in: {save_dir}\n')
                f.write(f'Pair rankings saved in: {csv_path}\n')
                f.write('-' * 50 + '\n')

            # Store configuration results
            config_result = {
                'resolution': resolution,
                'bin_size': bin_size,
                'max_lag': max_lag,
                'n_pairs': len(pairs),
                'summary_path': summary_path,
                'cc_plot_path': cc_plot_path,
                'csv_path': csv_path,
                'top_bump_pairs': top_bump_pairs,
                'bottom_bump_pairs': bottom_bump_pairs
            }
            file_results['configurations'].append(config_result)
            
            if verbose:
                print(f"Pipeline updated for {resolution}. Summary saved at {summary_path}")
                print(f"Cross-correlation plot saved at {cc_plot_path}")
                print(f"Pair rankings saved at {csv_path}")

        results['processed_files'].append(file_results)
        results['summary_paths'].append(summary_path)
        
    return results

def main():
    """
    Main function for standalone script execution.
    Accepts one or more config JSON paths via --config; runs the pipeline for each.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline with one or more config files."
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=None,
        help="One or more config JSON paths.",
    )
    args = parser.parse_args()

    config_paths = list(args.config) if args.config else []

    if not config_paths:
        # Backward compatibility: default config
        default = os.path.join(base_dir, "analysis_pipeline", "config_dnms.json")
        if os.path.exists(default):
            config_paths = [default]
        else:
            parser.error("No config provided and default config_dnms.json not found. Use --config.")
        print(f"Using default config file: {config_paths[0]}")
    else:
        # Resolve relative paths
        resolved = []
        for p in config_paths:
            if not os.path.isabs(p):
                p = os.path.join(os.getcwd(), p) if not p.startswith("analysis_pipeline") else os.path.join(base_dir, p)
            if not os.path.exists(p):
                p_alt = os.path.join(base_dir, p)
                p = p_alt if os.path.exists(p_alt) else p
            resolved.append(p)
        config_paths = resolved

    for i, config_path in enumerate(config_paths):
        if len(config_paths) > 1:
            print(f"[{i+1}/{len(config_paths)}] Using config: {config_path}")
        else:
            print(f"Using config file: {config_path}")
        results = run_preprocessing_pipeline(config_path)
        print(f"Config done. Processed {len(results['processed_files'])} files.")
    print("All configs complete.")

if __name__ == "__main__":
    main()