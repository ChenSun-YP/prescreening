#!/usr/bin/env python3
from fileinput import filename
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.stats import norm
import textwrap

"""
Utility functions to plot various analyses for single-input single-output
"""


# Helper function to determine color intensity based on z-score
def get_color_intensity(value, mean, std):
    if std == 0:
        return "red"
    z_score = abs(value - mean) / std
    alpha = min(1.0, max(0.2, z_score / 3))
    return (1, 0, 0, alpha)


def plot_all_neurons_silent_periods(neurons, save_dir, sample_rate):
    neuron_spike_trains = {}
    global_max_time = 0
    for neuron_id, spike_times in neurons.items():
        spike_times = np.array(spike_times) / sample_rate * 1000
        spike_indices = np.round(spike_times).astype(int)
        global_max_time = max(
            global_max_time, int(np.max(spike_indices)) if len(spike_indices) > 0 else 0
        )
        spike_train = np.zeros(global_max_time + 1)
        spike_train[spike_indices] = 1
        neuron_spike_trains[neuron_id] = spike_train

    silent_periods = {}
    for neuron_id, spike_train in neuron_spike_trains.items():
        spike_indices = np.where(spike_train == 1)[0]
        if len(spike_indices) == 0:
            silent_periods[neuron_id] = [(0, global_max_time)]
            continue
        isis = np.diff(np.concatenate([spike_indices, [global_max_time]]))
        top_3_indices = np.argsort(isis)[-3:][::-1]
        silent_periods[neuron_id] = [
            (
                (
                    spike_indices[idx]
                    if idx != len(spike_indices) - 1
                    else spike_indices[-1]
                ),
                (
                    spike_indices[idx + 1]
                    if idx != len(spike_indices) - 1
                    else global_max_time
                ),
            )
            for idx in top_3_indices
        ]

    fig, ax = plt.subplots(figsize=(15, len(neurons) * 0.5))
    neuron_ids = list(neurons.keys())
    for i, neuron_id in enumerate(neuron_ids):
        for j, (start, end) in enumerate(silent_periods[neuron_id][:3]):
            ax.fill_betweenx(
                [i - 0.4, i + 0.4],
                start,
                end,
                color=plt.cm.viridis(j / 3),
                alpha=0.6,
                label=f"Top {j+1}" if i == 0 else "",
            )
    ax.set_yticks(np.arange(len(neuron_ids)))
    ax.set_yticklabels(neuron_ids)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron ID")
    ax.set_title("Top 3 Longest Silent Periods per Neuron")
    ax.set_xlim(0, global_max_time)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_neurons_silent_periods.png"))
    print(f'Saved at {os.path.join(save_dir, "all_neurons_silent_periods.png")}')
    plt.close()


def plot_spike_raster(neurons, save_dir, sample_rate, bin_size):
    neuron_spike_indices = {}
    global_max_time = 0
    firing_rates = {}
    for neuron_id, spike_times in neurons.items():
        spike_times = np.array(spike_times) / sample_rate * 1000
        spike_indices = np.round(spike_times).astype(int)
        global_max_time = max(
            global_max_time, int(np.max(spike_indices)) if len(spike_indices) > 0 else 0
        )
        neuron_spike_indices[neuron_id] = spike_indices
        # Calculate firing rate in Hz (spikes per second)
        # Use global_max_time in milliseconds, convert to seconds for Hz
        firing_rates[neuron_id] = (
            len(spike_indices) / (global_max_time / 1000) if global_max_time > 0 else 0
        )

    num_bins = int(np.ceil(global_max_time / bin_size)) + 1
    binned_spikes = {
        neuron_id: np.histogram(
            spike_indices, bins=num_bins, range=(0, global_max_time)
        )[0]
        > 0
        for neuron_id, spike_indices in neuron_spike_indices.items()
    }

    fig, ax = plt.subplots(figsize=(15, len(neurons) * 0.5))
    neuron_ids = list(neurons.keys())
    for i, neuron_id in enumerate(neuron_ids):
        spike_bins = np.where(binned_spikes[neuron_id])[0]
        ax.plot(
            spike_bins * bin_size, [i] * len(spike_bins), "k.", markersize=4, alpha=0.5
        )
    ax.set_yticks(np.arange(len(neuron_ids)))
    # Create labels with neuron ID and firing rate
    labels = [
        f"{neuron_id} ({firing_rates[neuron_id]:.2f} Hz)" for neuron_id in neuron_ids
    ]
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron ID")
    ax.set_title(f"Spike Raster Plot (Binned at {bin_size} ms)")
    ax.set_xlim(0, global_max_time)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "spike_raster_binned.png"))
    print(f'Saved at {os.path.join(save_dir, "spike_raster_binned.png")}')
    plt.close()


def compute_correlogram(
    x, y, max_lag_ms, bin_size, mode, firing_rate_x, firing_rate_y, T
):
    """
    x and y are the spike trains of the two neurons
    max_lag_ms is the maximum lag in milliseconds
    bin_size is the bin size in milliseconds
    mode is either 'auto' or 'cross'
    firing_rate_x and firing_rate_y are the firing rates of the two neurons
    T is the total time of the spike trains
    mean after normalizing by the expected coincidences is 1 as we did not remove the baseline firing rate
    also return the std after normalizing by the expected coincidences
    """
    # Convert max_lag from milliseconds to number of bins
    max_lag_bins = int(max_lag_ms / bin_size)

    # Center the signals by subtracting the mean
    # x_centered = x - np.mean(x)
    # y_centered = y - np.mean(y) if mode == 'cross' else x_centered
    x_centered = x
    y_centered = y

    # Compute correlation using numpy's correlate
    corr = scipy.signal.correlate(x_centered, y_centered, mode="full")
    # Find the center (zero lag) index
    center = len(corr) // 2

    # Extract only the desired lags
    start_idx = center - max_lag_bins
    end_idx = center + max_lag_bins + 1
    corr_trimmed = corr[start_idx:end_idx]

    # Zero out the center bin for autocorrelation
    if mode == "auto":
        corr_trimmed[max_lag_bins] = 0

    # Create lag values in milliseconds
    lags = np.arange(-max_lag_bins, max_lag_bins + 1) * bin_size
    # normalize the correlogram by variance of x squareroot * y squareroot
    # print(np.var(x), np.var(y))
    # corr_trimmed = corr_trimmed / np.std(x) * np.std(y)

    # to normalize the correlogram by expected coincidences we divide by the expected coincidences
    # expected coincidences is the product of the firing rates of the two neurons and the bin width
    # then y value become the normalized correlogram

    # E = firing_rate_x * firing_rate_y  * T * bin_size  # Expected coincidences hz * hz * ms = number of coincidences
    # corr_normalized = corr_trimmed / E if E > 0 else corr_trimmed  # Avoid division by zero
    mean = np.mean(corr_trimmed)  # Empirical mean (should be ~1)
    std = np.std(corr_trimmed)  # Empirical std
    z_scores = (corr_trimmed - mean) / std if std > 0 else np.zeros_like(corr_trimmed)

    # std is rootsqure of E
    # std_normalized = np.sqrt(1/E)
    # empirical_std = np.std(corr_normalized)  # Uses all lag bins after normalization
    return lags, corr_trimmed, mean, std, z_scores


def compute_correlogram_normalized_cc_first(
    x,
    y,
    max_lag_ms,
    bin_size,
    mode,
    firing_rate_x,
    firing_rate_y,
    T,
    edge_mean=True,
    score_type="bump",
    preNeuron=0,
    postNeuron=0,
):
    """
    x and y are the spike trains of the two neurons
    max_lag_ms is the maximum lag in milliseconds
    bin_size is the bin size in milliseconds
    mode is either 'auto' or 'cross'
    firing_rate_x and firing_rate_y are the firing rates of the two neurons
    T is the total time of the spike trains
    mean after normalizing by the expected coincidences is 1 as we did not remove the baseline firing rate
    also return the std after normalizing by the expected coincidences
    """
    # Convert max_lag from milliseconds to number of bins
    max_lag_bins = int(max_lag_ms / bin_size)  # number of bins

    # Center the signals by subtracting the mean
    # x_centered = x - np.mean(x)
    # y_centered = y - np.mean(y) if mode == 'cross' else x_centered
    x_centered = x
    y_centered = y
    # pad x and y entill they are T long
    x_centered = np.pad(x_centered, (0, T - len(x_centered)), "constant")
    y_centered = np.pad(y_centered, (0, T - len(y_centered)), "constant")

    corr = scipy.signal.correlate(y_centered, x_centered, mode="full")
    # Find the center (zero lag) index
    center = len(corr) // 2

    start_idx = center - max_lag_ms
    end_idx = center + max_lag_ms + 1
    corr_trimmed = corr[start_idx:end_idx]

    filename = f"corr_trimmed_{preNeuron}_{postNeuron}.txt"
    np.savetxt(filename, corr_trimmed)

    # Zero out the center bin for autocorrelation
    if mode == "auto":
        num_spikes = np.sum(x)
        corr_trimmed[
            max_lag_bins
        ] -= num_spikes  # Subtract self-pairs from zero-lag bin

    # Create lag values in milliseconds
    lags = np.arange(-max_lag_bins, max_lag_bins + 1) * bin_size
    binned_corr = np.zeros(2 * max_lag_bins)
    for i in range(2 * max_lag_bins):
        start_lag = lags[i] + max_lag_ms
        end_lag = lags[i + 1] + max_lag_ms
        binned_corr[i] = np.sum(corr_trimmed[start_lag:end_lag])

    # normalize the correlogram by variance of x squareroot * y squareroot
    # corr_trimmed = corr_trimmed / np.std(x) * np.std(y)

    # to normalize the correlogram by expected coincidences we divide by the expected coincidences
    # expected coincidences is the product of the firing rates of the two neurons and the bin width
    # then y value become the normalized correlogram

    E = (
        firing_rate_x * firing_rate_y * T * bin_size
    )  # Expected coincidences hz * hz * ms = number of coincidences
    if E <= 0:
        print(f"Warning: Expected coincidences (E) is {E}")
    corr_normalized = (
        binned_corr / E if E > 0 else binned_corr
    )  # Avoid division by zero

    # mean_normalized = np.mean(corr_normalized)  # Empirical mean (should be ~1)
    mean_normalized = np.mean(corr_normalized)  # 1 as 1 is the expected coincidences
    std_normalized = np.std(
        corr_normalized
    )  # Empirical std on local bins! so this actualy use local mean which is not 1.

    z_scores = (
        (corr_normalized - mean_normalized) / mean_normalized
        if mean_normalized > 0
        else np.zeros_like(corr_normalized)
    )
    baseline = np.mean(
        corr_normalized
    )  # As per your normalization, baseline is 1 but here we are using the global mean
    # bump_scores = np.sqrt(np.abs(corr_normalized - baseline)) / std_normalized if std_normalized > 0 else np.zeros_like(corr_normalized)
    # bump_scores = np.sqrt(np.abs(corr_normalized - baseline)) /baseline
    total_z_score = np.sum(np.abs(z_scores))

    # bumpt measure
    from scipy.signal import find_peaks

    dev = np.abs(corr_normalized - baseline)
    peaks, props = find_peaks(
        dev, prominence=0.01 * baseline
    )  # Adjust prominence threshold
    bump_scores = np.zeros_like(corr_normalized)
    bump_scores[peaks] = props["prominences"] / baseline

    total_bump_score = np.sum(
        bump_scores
    )  # Total score across all lags to identify significant pairs

    # std is rootsqure of E
    score_type = "z_score"
    if score_type == "bump":
        return (
            lags,
            corr_normalized,
            mean_normalized,
            std_normalized,
            bump_scores,
            total_bump_score,
        )
    elif score_type == "z_score":
        return (
            lags,
            corr_normalized,
            mean_normalized,
            std_normalized,
            z_scores,
            total_z_score,
        )


def compute_correlogram_normalized(
    x,
    y,
    max_lag_ms,
    bin_size,
    mode,
    firing_rate_x,
    firing_rate_y,
    T,
    edge_mean=True,
    score_type="bump",
):
    """
    x and y are the spike trains of the two neurons
    max_lag_ms is the maximum lag in milliseconds
    bin_size is the bin size in milliseconds
    mode is either 'auto' or 'cross'
    firing_rate_x, firing_rate_y, T, edge_mean, score_type are included for compatibility
    Returns correlogram with normalization and scoring to match compute_correlogram_normalized output
    """
    # Resample and binarize
    n_original = len(x)
    resampling_factor = int(bin_size / 1)  # Assuming original data is in 1ms bins
    x_resampled = np.array(
        [
            np.sum(x[i : i + resampling_factor])
            for i in range(0, n_original, resampling_factor)
        ]
    )
    y_resampled = np.array(
        [
            np.sum(y[i : i + resampling_factor])
            for i in range(0, n_original, resampling_factor)
        ]
    )
    x_resampled = (x_resampled > 0).astype(float)
    y_resampled = (y_resampled > 0).astype(float)

    # Compute correlation
    max_lag_bins = int(max_lag_ms / bin_size)
    corr = scipy.signal.correlate(y_resampled, x_resampled, mode="full")
    center = len(corr) // 2
    start_idx = center - max_lag_bins
    end_idx = center + max_lag_bins + 1
    corr_trimmed = corr[start_idx:end_idx]

    # Zero out zero-lag bin for autocorrelation
    if mode == "auto":
        corr_trimmed[max_lag_bins] = 0

    # Create lag values in milliseconds
    lags = np.arange(-max_lag_bins, max_lag_bins + 1) * bin_size

    # Normalize by number of bins (simple normalization to produce corr_normalized)
    corr_normalized = corr_trimmed

    # normalize by the expected coincidences
    # E = firing_rate_x * firing_rate_y  * T * bin_size  # Expected coincidences hz * hz * ms = number of coincidences
    # print(f'E: {E}')
    # corr_normalized = corr_trimmed / E if E > 0 else corr_trimmed  # Avoid division by zero

    # Compute mean and std
    mean_normalized = np.mean(corr_normalized)
    std_normalized = np.std(corr_normalized)

    # Compute scores
    if score_type == "z_score":
        scores = (
            (corr_normalized - mean_normalized) / std_normalized
            if std_normalized > 0
            else np.zeros_like(corr_normalized)
        )
        total_score = np.sum(np.abs(scores))
    else:  # score_type == 'bump'
        baseline = mean_normalized
        dev = np.abs(corr_normalized - baseline)
        from scipy.signal import find_peaks

        peaks, props = find_peaks(dev, prominence=0.01 * baseline)
        scores = np.zeros_like(corr_normalized)
        scores[peaks] = props["prominences"] / baseline
        total_score = np.sum(scores)

    return lags, corr_normalized, mean_normalized, std_normalized, scores, total_score


# def plot_neuron_autocorrelations_combined(neurons, save_dir, sample_rate):
#     settings = [(40, 1000), (20, 500), (8, 250), (2, 50)]  # (bin_size, max_lag) in ms
#     all_autocorrs, all_lags = {}, {}

#     for bin_size, max_lag in settings:
#         neuron_spike_trains, global_max_time = {}, 0
#         for neuron_id, spike_times in neurons.items():
#             spike_times = np.array(spike_times) / sample_rate * 1000
#             spike_indices = np.round(spike_times).astype(int)
#             global_max_time = max(global_max_time, int(np.max(spike_indices)) if len(spike_indices) > 0 else 0)
#             num_bins = int(np.ceil(global_max_time / bin_size)) + 1
#             neuron_spike_trains[neuron_id] = np.histogram(spike_indices, bins=num_bins, range=(0, global_max_time))[0] > 0

#         autocorrs = {}
#         for neuron_id, spike_train in neuron_spike_trains.items():
#             lags, corr = compute_correlogram(spike_train.astype(float), spike_train.astype(float), max_lag, bin_size, 'auto')
#             autocorrs[neuron_id] = corr
#             if (bin_size, max_lag) not in all_lags:
#                 all_lags[(bin_size, max_lag)] = lags
#         all_autocorrs[(bin_size, max_lag)] = autocorrs

#     neuron_ids = list(neurons.keys())
#     neurons_per_page = 16
#     n_pages = int(np.ceil(len(neuron_ids) / neurons_per_page))

#     for page in range(n_pages):
#         page_neurons = neuron_ids[page * neurons_per_page:min((page + 1) * neurons_per_page, len(neuron_ids))]
#         n_rows = int(np.ceil(len(page_neurons) / 4))
#         fig, axes = plt.subplots(n_rows, 16, figsize=(15, 1.5 * n_rows))
#         if n_rows == 1: axes = [axes]

#         for i, neuron_id in enumerate(page_neurons):
#             row, col_start = i // 4, (i % 4) * 4
#             for s, (bin_size, max_lag) in enumerate(settings):
#                 ax = axes[row][col_start + s]
#                 lags = all_lags[(bin_size, max_lag)]
#                 corr = all_autocorrs[(bin_size, max_lag)][neuron_id]
#                 ax.step(lags, corr, where='mid', color='b', linewidth=0.8)
#                 if s == 0:
#                     ax.text(0.05, 0.95, f'{neuron_id}', transform=ax.transAxes, fontsize=6, va='top', ha='left',
#                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
#                 if row == 0:
#                     ax.set_title(f'±{max_lag}ms, {bin_size}ms', fontsize=7)
#                 ax.set_xlim(-max_lag, max_lag)
#                 y_data = corr
#                 ax.set_ylim(np.min(y_data) - 0.1 * (np.max(y_data) - np.min(y_data)),
#                            np.max(y_data) + 0.1 * (np.max(y_data) - np.min(y_data)))
#                 ax.tick_params(axis='both', labelsize=5)
#                 if row < n_rows - 1:
#                     ax.set_xticklabels([])
#                 else:
#                     ax.set_xlabel('Lag (ms)', fontsize=6)
#                 if col_start == 0 and s == 0:
#                     ax.set_ylabel('Correlation', fontsize=6)

#         for i in range(len(page_neurons), n_rows * 4):
#             for s in range(4):
#                 axes[i // 4][(i % 4) * 4 + s].set_visible(False)

#         fig.suptitle(f'Autocorrelation of Neuron Spike Trains at Multiple Resolutions{" (Page " + str(page+1) + "/" + str(n_pages) + ")" if n_pages > 1 else ""}',
#                     fontsize=10, y=0.98)
#         plt.tight_layout()
#         plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)
#         plt.savefig(os.path.join(save_dir, f'neuron_autocorrelations_combined_page{page+1}.png'), dpi=150)
#         print(f'Saved at {os.path.join(save_dir, f"neuron_autocorrelations_combined_page{page+1}.png")}')
#         plt.close()


# def plot_neuron_cross_correlations_combined(neurons, save_dir, sample_rate):
#     settings = [(50, 1000), (25, 500), (13, 250), (3, 50)]  # (bin_size, max_lag) in ms
#     neuron_ids = list(neurons.keys())
#     pairs = [(n1, n2) for i, n1 in enumerate(neuron_ids) for n2 in neuron_ids[i + 1 :]]
#     if not pairs:
#         print("No neuron pairs to plot cross-correlations")
#         return

#     all_autocorrs, all_crosscorrs, all_lags = {}, {}, {}
#     for bin_size, max_lag in settings:
#         neuron_spike_trains, global_max_time = {}, 0
#         for neuron_id, spike_times in neurons.items():
#             spike_times = np.array(spike_times) / sample_rate * 1000
#             spike_indices = np.round(spike_times).astype(int)
#             global_max_time = max(
#                 global_max_time,
#                 int(np.max(spike_indices)) if len(spike_indices) > 0 else 0,
#             )
#             num_bins = int(np.ceil(global_max_time / bin_size)) + 1
#             neuron_spike_trains[neuron_id] = (
#                 np.histogram(spike_indices, bins=num_bins, range=(0, global_max_time))[
#                     0
#                 ]
#                 > 0
#             )

#         autocorrs, crosscorrs = {}, {}
#         for neuron_id in neuron_ids:
#             spike_train = neuron_spike_trains[neuron_id].astype(float)
#             lags, corr = compute_correlogram(
#                 spike_train, spike_train, max_lag, bin_size, "auto"
#             )
#             autocorrs[neuron_id] = corr
#             if (bin_size, max_lag) not in all_lags:
#                 all_lags[(bin_size, max_lag)] = lags
#         for n1, n2 in [(n1, n2) for n1 in neuron_ids for n2 in neuron_ids if n1 != n2]:
#             lags, corr = compute_correlogram(
#                 neuron_spike_trains[n1].astype(float),
#                 neuron_spike_trains[n2].astype(float),
#                 max_lag,
#                 bin_size,
#                 "cross",
#             )
#             crosscorrs[(n1, n2)] = corr
#         all_autocorrs[(bin_size, max_lag)] = autocorrs
#         all_crosscorrs[(bin_size, max_lag)] = crosscorrs

#     pairs_per_page = 10
#     n_pages = int(np.ceil(len(pairs) / pairs_per_page))
#     for page in range(n_pages):
#         page_pairs = pairs[
#             page * pairs_per_page : min((page + 1) * pairs_per_page, len(pairs))
#         ]
#         fig, axes = plt.subplots(
#             len(page_pairs), 12, figsize=(20, 1.25 * len(page_pairs))
#         )
#         if len(page_pairs) == 1:
#             axes = [axes]

#         for i, (n1, n2) in enumerate(page_pairs):
#             for s, (bin_size, max_lag) in enumerate(settings):
#                 col_offset = s * 3
#                 lags = all_lags[(bin_size, max_lag)]
#                 # Autocorrelations
#                 axes[i][col_offset].step(
#                     lags,
#                     all_autocorrs[(bin_size, max_lag)][n1],
#                     where="mid",
#                     color="b",
#                     linewidth=0.8,
#                 )
#                 axes[i][col_offset + 1].step(
#                     lags,
#                     all_autocorrs[(bin_size, max_lag)][n2],
#                     where="mid",
#                     color="g",
#                     linewidth=0.8,
#                 )
#                 # Cross-correlation
#                 cross_corr = all_crosscorrs[(bin_size, max_lag)][(n1, n2)]
#                 mean_corr = np.mean(cross_corr)
#                 std_corr = np.std(cross_corr)
#                 for idx in range(len(lags) - 1):
#                     color = get_color_intensity(cross_corr[idx], mean_corr, std_corr)
#                     axes[i][col_offset + 2].step(
#                         lags[idx : idx + 2],
#                         cross_corr[idx : idx + 2],
#                         where="mid",
#                         color=color,
#                         linewidth=0.8,
#                     )
#                 if s == 0:
#                     for j, text in enumerate([f"A:{n1}", f"B:{n2}", "A→B"]):
#                         axes[i][col_offset + j].text(
#                             0.05,
#                             0.95,
#                             text,
#                             transform=axes[i][col_offset + j].transAxes,
#                             fontsize=6,
#                             va="top",
#                             ha="left",
#                             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
#                         )
#                 if i == 0:
#                     axes[i][col_offset].set_title(
#                         f"±{max_lag}ms, {bin_size}ms", fontsize=7
#                     )
#                 for j in range(3):
#                     ax = axes[i][col_offset + j]
#                     ax.set_xlim(-max_lag, max_lag)
#                     y_data = ax.lines[0].get_ydata()
#                     ax.set_ylim(
#                         np.min(y_data) - 0.1 * (np.max(y_data) - np.min(y_data)),
#                         np.max(y_data) + 0.1 * (np.max(y_data) - np.min(y_data)),
#                     )
#                     ax.tick_params(labelsize=5)
#                     if i < len(page_pairs) - 1:
#                         ax.set_xticklabels([])
#             axes[i][0].set_ylabel(f"Pair {i+1}", fontsize=7)

#         fig.suptitle(
#             f'Neuron Pair Correlations at Multiple Resolutions{" (Page " + str(page+1) + "/" + str(n_pages) + ")" if n_pages > 1 else ""}',
#             fontsize=10,
#             y=0.98,
#         )
#         plt.tight_layout()
#         plt.subplots_adjust(top=0.95, hspace=0.4, wspace=0.4)
#         plt.savefig(
#             os.path.join(
#                 save_dir, f"neuron_pair_correlations_combined_page{page+1}.png"
#             ),
#             dpi=150,
#         )
#         print(
#             f'Saved at {os.path.join(save_dir, f"neuron_pair_correlations_combined_page{page+1}.png")}'
#         )
#         plt.close()


# def plot_z_score_distribution(neurons, save_dir, sample_rate):
#     configs = [
#         (80, 1000, "Broad"),
#         (40, 500, "Medium"),
#         (20, 250, "Fine"),
#         (2, 50, "Ultra-fine"),
#     ]
#     configs = [(80, 1000, "Broad"), (20, 250, "Fine")]
#     neuron_ids = list(neurons.keys())

#     for bin_size, max_lag, resolution in configs:
#         neuron_spike_trains, global_max_time, firing_rate, global_max_times = (
#             {},
#             0,
#             {},
#             {},
#         )
#         for neuron_id, spike_times in neurons.items():
#             spike_times = np.array(spike_times) / sample_rate * 1000
#             spike_indices = np.round(spike_times).astype(int)
#             global_max_time = max(
#                 global_max_time,
#                 int(np.max(spike_indices)) if len(spike_indices) > 0 else 0,
#             )
#             global_max_times[neuron_id] = global_max_time
#             num_bins = int(np.ceil(global_max_time / bin_size)) + 1
#             neuron_spike_trains[neuron_id] = (
#                 np.histogram(spike_indices, bins=num_bins, range=(0, global_max_time))[
#                     0
#                 ]
#                 > 0
#             )
#             firing_rate[neuron_id] = np.sum(neuron_spike_trains[neuron_id]) / (
#                 global_max_time / 1000
#             )

#         crosscorrs = {}
#         for i, pre in enumerate(neuron_ids):
#             for post in neuron_ids[i + 1 :]:
#                 crosscorrs[(pre, post)] = compute_correlogram_normalized(
#                     neuron_spike_trains[pre].astype(float),
#                     neuron_spike_trains[post].astype(float),
#                     max_lag,
#                     bin_size,
#                     "cross",
#                     firing_rate[pre],
#                     firing_rate[post],
#                     max(global_max_times[pre], global_max_times[post]),
#                 )

#         z_scores = np.concatenate(
#             [
#                 crosscorrs[(pre, post)][4]
#                 for i, pre in enumerate(neuron_ids)
#                 for post in neuron_ids[i + 1 :]
#             ]
#         )

#         counts, bins = np.histogram(z_scores, bins=100)
#         normalized_counts = counts / np.sum(counts)
#         print(f"Sum total count: {np.sum(counts)}")
#         print(f"Sum total normalized count: {np.sum(normalized_counts):.3f}")

#         plt.figure(figsize=(8, 8))  # Square figure
#         plt.bar(
#             bins[:-1],
#             normalized_counts,
#             width=np.diff(bins),
#             align="edge",
#             alpha=0.7,
#             color="skyblue",
#         )  # No edgecolor
#         bin_width = np.diff(bins)[0]
#         x = np.linspace(min(bins), max(bins), 100)
#         gaussian = norm.pdf(x, loc=0, scale=1) * bin_width
#         plt.plot(x, gaussian, "r-", lw=2, label="Theoretical Gaussian (N(0,1))")

#         plt.ylabel("Proportion of Pairs", fontsize=20)
#         plt.xlabel("Z-score", fontsize=20)
#         plt.title(
#             f"{save_dir}\nZ-score Distribution of Cross-Correlations\n({resolution}: ±{max_lag}ms, {bin_size}ms bins)",
#             fontsize=20,
#             pad=20,
#         )
#         plt.ylim(0, 0.05)  # Global ylim

#         num_pairs = len(crosscorrs)
#         plt.text(
#             0.95,
#             0.95,
#             f"N={num_pairs} pairs",
#             transform=plt.gca().transAxes,
#             fontsize=20,
#             ha="right",
#             va="top",
#             bbox=dict(facecolor="white", alpha=0.8),
#         )

#         plt.legend(fontsize=20)
#         plt.grid(True, linestyle=":", alpha=0.5)
#         plt.tick_params(axis="both", labelsize=20)
#         plt.tight_layout()

#         save_path = os.path.join(
#             save_dir, f"z_score_distribution_{resolution.lower()}.png"
#         )
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#         print(f"Saved {resolution} resolution z-score distribution at {save_path}")
#         plt.close()


# def plot_z_score_distributions_multiple(
#     pkl_list, sample_rate, save_dir, dataset_names=None
# ):
#     """Plot z-score distributions for multiple datasets as both subplots and overlay, for each resolution."""
#     if dataset_names is None or len(dataset_names) != len(pkl_list):
#         dataset_names = [f"Dataset {i+1}" for i in range(len(pkl_list))]

#     configs = [(80, 1000, "Broad"), (20, 250, "Fine")]

#     for bin_size, max_lag, resolution in configs:
#         # Calculate grid size for subplots
#         n_datasets = len(pkl_list)
#         n_cols = int(np.ceil(np.sqrt(n_datasets)))
#         n_rows = int(np.ceil(n_datasets / n_cols))

#         # Create two figures: one for subplots, one for overlay
#         fig_subplots = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
#         fig_overlay = plt.figure(figsize=(8, 8))

#         colors = plt.cm.viridis(np.linspace(0, 1, len(pkl_list)))
#         all_z_scores = []  # Store all z_scores for consistent binning

#         # First pass: collect all z-scores and create individual subplots
#         for idx, (pkl_path, dataset_name) in enumerate(zip(pkl_list, dataset_names)):
#             with open(pkl_path, "rb") as f:
#                 neurons = pickle.load(f)

#             neuron_ids = list(neurons.keys())
#             neuron_spike_trains, global_max_time, firing_rate, global_max_times = (
#                 {},
#                 0,
#                 {},
#                 {},
#             )

#             # Process spike trains
#             for neuron_id, spike_times in neurons.items():
#                 spike_times = np.array(spike_times) / sample_rate * 1000
#                 spike_indices = np.round(spike_times).astype(int)
#                 global_max_time = max(
#                     global_max_time,
#                     int(np.max(spike_indices)) if len(spike_indices) > 0 else 0,
#                 )
#                 global_max_times[neuron_id] = global_max_time
#                 num_bins = int(np.ceil(global_max_time / bin_size)) + 1
#                 neuron_spike_trains[neuron_id] = (
#                     np.histogram(
#                         spike_indices, bins=num_bins, range=(0, global_max_time)
#                     )[0]
#                     > 0
#                 )
#                 firing_rate[neuron_id] = np.sum(neuron_spike_trains[neuron_id]) / (
#                     global_max_time / 1000
#                 )

#             # Compute cross-correlations
#             crosscorrs = {}
#             for i, pre in enumerate(neuron_ids):
#                 for post in neuron_ids[i + 1 :]:
#                     crosscorrs[(pre, post)] = compute_correlogram(
#                         neuron_spike_trains[pre].astype(float),
#                         neuron_spike_trains[post].astype(float),
#                         max_lag,
#                         bin_size,
#                         "cross",
#                         firing_rate[pre],
#                         firing_rate[post],
#                         max(global_max_times[pre], global_max_times[post]),
#                     )

#             z_scores = np.concatenate(
#                 [
#                     crosscorrs[(pre, post)][4]
#                     for i, pre in enumerate(neuron_ids)
#                     for post in neuron_ids[i + 1 :]
#                 ]
#             )
#             all_z_scores.append(z_scores)

#             # Create subplot
#             plt.figure(fig_subplots.number)
#             ax = plt.subplot(n_rows, n_cols, idx + 1)
#             counts, bins = np.histogram(z_scores, bins=100)
#             normalized_counts = counts / np.sum(counts)
#             ax.bar(
#                 bins[:-1],
#                 normalized_counts,
#                 width=np.diff(bins),
#                 align="edge",
#                 color=colors[idx],
#                 alpha=0.7,
#             )

#             # Add Gaussian curve to each subplot
#             bin_width = np.diff(bins)[0]
#             x = np.linspace(min(bins), max(bins), 100)
#             gaussian = norm.pdf(x, loc=0, scale=1) * bin_width
#             ax.plot(x, gaussian, "r-", lw=1, alpha=0.8)

#             # put text on the sum bins of the z-scores thats abs +=3
#             extreme_z = np.mean(np.abs(z_scores) >= 3)
#             # Add text to plot
#             ax.text(
#                 0.95,
#                 0.95,
#                 f"P(|z|≥3): {100*extreme_z:.3f}%",
#                 transform=ax.transAxes,
#                 ha="right",
#                 va="top",
#                 bbox=dict(facecolor="white", alpha=0.8),
#                 fontsize=12,
#             )

#             # ax.set_title(dataset_name, fontsize=20)
#             # Wrap long dataset names to multiple lines
#             wrapped_name = "\n".join(
#                 textwrap.wrap(dataset_name, width=20)
#             )  # adjust width as needed
#             ax.set_title(wrapped_name, fontsize=20)
#             ax.set_xlabel(
#                 "Z-score" if idx >= (n_rows - 1) * n_cols else "", fontsize=24
#             )
#             ax.set_xlabel("Z-score" if idx >= (n_rows - 1) * n_cols else "")
#             ax.set_ylabel("Proportion" if idx % n_cols == 0 else "")
#             ax.grid(True, linestyle=":", alpha=0.5)
#             ax.set_ylim(0, 0.05)

#         # Save subplots figure
#         plt.figure(fig_subplots.number)
#         plt.suptitle(
#             f"Z-score Distributions by Dataset\n({resolution}: ±{max_lag}ms, {bin_size}ms bins)",
#             fontsize=16,
#             y=1.02,
#         )
#         plt.tight_layout()
#         save_path_subplots = os.path.join(
#             save_dir, f"z_score_distributions_subplots_{resolution.lower()}.png"
#         )
#         plt.savefig(save_path_subplots, dpi=300, bbox_inches="tight")

#         # Create overlay plot
#         plt.figure(fig_overlay.number)
#         for idx, (z_scores, dataset_name) in enumerate(
#             zip(all_z_scores, dataset_names)
#         ):
#             counts, bins = np.histogram(z_scores, bins=100)
#             normalized_counts = counts / np.sum(counts)
#             plt.bar(
#                 bins[:-1],
#                 normalized_counts,
#                 width=np.diff(bins),
#                 align="edge",
#                 alpha=0.5,
#                 color=colors[idx],
#                 label=dataset_name,
#             )

#         # Add Gaussian curve to overlay
#         bin_width = np.diff(bins)[0]
#         x = np.linspace(min(bins), max(bins), 100)
#         gaussian = norm.pdf(x, loc=0, scale=1) * bin_width
#         plt.plot(x, gaussian, "r-", lw=2, label="N(0,1)")

#         plt.ylabel("Proportion of Pairs", fontsize=20)
#         plt.xlabel("Z-score", fontsize=20)
#         plt.title(
#             f"Z-score Distributions Across Datasets\n({resolution}: ±{max_lag}ms, {bin_size}ms bins)",
#             fontsize=20,
#             pad=20,
#         )
#         plt.ylim(0, 0.05)
#         plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.grid(True, linestyle=":", alpha=0.5)
#         plt.tick_params(axis="both", labelsize=12)

#         # Save overlay figure
#         plt.tight_layout()
#         save_path_overlay = os.path.join(
#             save_dir, f"z_score_distributions_overlay_{resolution.lower()}.png"
#         )
#         plt.savefig(save_path_overlay, dpi=300, bbox_inches="tight")

#         print(f"Saved {resolution} subplot figure at {save_path_subplots}")
#         print(f"Saved {resolution} overlay figure at {save_path_overlay}")
#         plt.close("all")


def plot_neuron_correlation_matrices(
    neurons, save_dir, sample_rate, edge_mean=True, configs=None, make_plots=True
):
    if configs is None:
        configs = [
            (80, 1000, "Broad"),
            (40, 500, "Medium"),
            (20, 250, "Fine"),
            (10, 100, "semiFine"),
            (2, 50, "Ultra-fine"),
        ]
    neuron_ids = list(neurons.keys())

    for bin_size, max_lag, resolution in configs:
        neuron_spike_trains, global_max_time, firing_rate, global_max_times = (
            {},
            0,
            {},
            {},
        )
        for neuron_id, spike_times in neurons.items():
            spike_times = np.array(spike_times) / sample_rate * 1000
            spike_indices = np.round(spike_times).astype(int)
            global_max_time = max(
                global_max_time,
                int(np.max(spike_indices)) if len(spike_indices) > 0 else 0,
            )
            global_max_times[neuron_id] = global_max_time
            num_bins = global_max_time
            neuron_spike_trains[neuron_id] = (
                np.histogram(spike_indices, bins=num_bins, range=(0, global_max_time))[
                    0
                ]
                > 0
            )
            firing_rate[neuron_id] = (
                np.sum(neuron_spike_trains[neuron_id]) / global_max_time
            )

        autocorrs = {
            n: compute_correlogram_normalized(
                neuron_spike_trains[n].astype(float),
                neuron_spike_trains[n].astype(float),
                max_lag,
                bin_size,
                "auto",
                firing_rate[n],
                firing_rate[n],
                global_max_times[n],
                edge_mean=edge_mean,
            )
            for n in neuron_ids
        }
        # crosscorrs = {(pre, post): compute_correlogram_normalized(neuron_spike_trains[pre].astype(float), neuron_spike_trains[post].astype(float), max_lag, bin_size, 'cross', firing_rate[pre], firing_rate[post], max(global_max_times[pre], global_max_times[post]), edge_mean=edge_mean)
        #              for pre in neuron_ids for post in neuron_ids if pre != post}
        crosscorrs = {
            (pre, post): compute_correlogram_normalized_cc_first(
                neuron_spike_trains[pre].astype(float),
                neuron_spike_trains[post].astype(float),
                max_lag,
                bin_size,
                "cross",
                firing_rate[pre],
                firing_rate[post],
                max(global_max_times[pre], global_max_times[post]),
                edge_mean=edge_mean,
                preNeuron=pre,
                postNeuron=post,
            )
            for pre in neuron_ids
            for post in neuron_ids
            if pre != post
        }
        if make_plots:
            fig_size = min(24, 1.2 * (len(neuron_ids) + 1))
            fig = plt.figure(figsize=(fig_size, fig_size))
            gs = fig.add_gridspec(
                len(neuron_ids) + 1, len(neuron_ids) + 1, hspace=0.15, wspace=0.15
            )

            for i, pre_id in enumerate(neuron_ids):
                for j, post_id in enumerate(neuron_ids):
                    ax = fig.add_subplot(gs[i + 1, j + 1])
                    if pre_id != post_id:
                        lags, corr, mean_corr, std_corr, z_score, total_bump_score = (
                            crosscorrs[(pre_id, post_id)]
                        )
                        for idx in range(len(lags) - 1):
                            color = get_color_intensity(corr[idx], mean_corr, std_corr)
                            ax.fill_between(
                                [lags[idx], lags[idx + 1]],
                                [0, 0],
                                [corr[idx], corr[idx]],
                                color=color,
                                alpha=0.6,
                                linewidth=0.5,
                            )
                        ax.axhline(
                            y=mean_corr,
                            color="black",
                            linestyle="--",
                            linewidth=0.2,
                            alpha=0.8,
                            label="Mean CC" if i == 1 and j == 1 else None,
                        )
                        ax.fill_between(
                            [lags[0], lags[-1]],
                            [mean_corr - std_corr, mean_corr - std_corr],
                            [mean_corr + std_corr, mean_corr + std_corr],
                            color="gray",
                            alpha=0.2,
                        )
                    ax.set_xlim(-max_lag, max_lag)
                    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.3)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.axhline(
                        y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5
                    )
                    if i < len(neuron_ids) - 1:
                        ax.set_xticklabels([])
                    else:
                        ax.set_xlabel("ms", fontsize=6)
                    ax.tick_params(labelsize=5)
                    if i == 0:
                        # Truncate to max 5 letters for title, removing "neuron" prefix
                        post_id_short = post_id.replace("neuron", "")[:5]
                        ax.set_title(post_id_short, fontsize=6, pad=2)
                    if j == 0:
                        # Truncate to max 5 letters for ylabel, removing "neuron" prefix
                        pre_id_short = pre_id.replace("neuron", "")[:5]
                        ax.set_ylabel(pre_id_short, fontsize=6)
                    ax.set_box_aspect(1)

            for idx, neuron_id in enumerate(neuron_ids):
                lags, corr, mean_corr, std_corr, z_score, total_bump_score = autocorrs[
                    neuron_id
                ]
                y_range = np.max(corr) - np.min(corr)
                ax_top = fig.add_subplot(gs[0, idx + 1])
                ax_top.step(lags, corr, where="mid", color="b", linewidth=0.5)
                ax_top.set_xlim(-max_lag, max_lag)
                ax_top.set_ylim(
                    np.min(corr) - 0.1 * y_range, np.max(corr) + 0.1 * y_range
                )
                ax_top.grid(True, linestyle=":", linewidth=0.5, alpha=0.3)
                ax_top.tick_params(labelsize=5)
                ax_top.spines["top"].set_visible(False)
                ax_top.spines["right"].set_visible(False)
                ax_top.axhline(
                    y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5
                )
                # Truncate to max 5 letters for title, removing "neuron" prefix
                neuron_id_short = neuron_id.replace("neuron", "")[:5]
                ax_top.set_title(neuron_id_short, fontsize=6, pad=2)
                ax_top.set_box_aspect(1)

                ax_left = fig.add_subplot(gs[idx + 1, 0])
                ax_left.step(lags, corr, where="mid", color="b", linewidth=0.5)
                ax_left.set_xlim(-max_lag, max_lag)
                ax_left.set_ylim(
                    np.min(corr) - 0.1 * y_range, np.max(corr) + 0.1 * y_range
                )
                ax_left.grid(True, linestyle=":", linewidth=0.5, alpha=0.3)
                ax_left.tick_params(labelsize=5)
                ax_left.spines["top"].set_visible(False)
                ax_left.spines["right"].set_visible(False)
                ax_left.axhline(
                    y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5
                )
                # Truncate to max 5 letters for ylabel, removing "neuron" prefix
                neuron_id_short = neuron_id.replace("neuron", "")[:5]
                ax_left.set_ylabel(neuron_id_short, fontsize=6)
                ax_left.set_box_aspect(1)

            ax_corner = fig.add_subplot(gs[0, 0])
            ax_corner.axis("off")
            ax_corner.set_box_aspect(1)

            # Add global note about input-output relationship
            ax_corner.text(
                0.5,
                0.5,
                "Output TopRow→\n↓\nLeftColInput",
                transform=ax_corner.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"),
            )

            plt.suptitle(
                f"Cross-correlation Matrix ({resolution} Resolution: ±{max_lag}ms, {bin_size}ms bins)",
                fontsize=10,
                y=0.98,
            )
            plt.tight_layout()
            filedir = os.path.join(
                save_dir,
                f"correlation_matrix_edge_mean_{edge_mean}_{resolution.lower()}.png",
            )
            plt.savefig(filedir, dpi=300, bbox_inches="tight")
            print(f"Saved {resolution} resolution matrix at {filedir}")
            plt.close()
        with open(
            os.path.join(
                save_dir, f"crosscorrs_edge_mean_{edge_mean}_{resolution.lower()}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(crosscorrs, f)
        with open(
            os.path.join(
                save_dir, f"autocorrs_edge_mean_{edge_mean}_{resolution.lower()}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(autocorrs, f)
        print(f"Saved {resolution} resolution crosscorrs and autocorrs at {save_dir}")
    return configs
