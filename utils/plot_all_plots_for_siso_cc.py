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

    filename = f"corr_trimmed_folder/corr_trimmed_{preNeuron}_{postNeuron}.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
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
