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

import numpy as np
import diptest
from diptest import diptest
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from scipy.stats import gaussian_kde


# check all intervals on base of the histogram is full
def check_correlation_filled_bins(
    pkl_path="data/analysis/selected_neurons_first_200s/crosscorrs_edge_mean_True_ultra-fine.pkl",  # find a way to automatically get this path
    out_dir="selected_neurons_first_200s",  # for debugging purposes
    harshness=0.10,  # at most 10% of bins can be empty
    power_threshold=-10,  # bins with value < e^-10 are considered empty
):
    """
    Saves (pre, post) pairs where at least `fraction_threshold` of corr bins
    have |corr| < e^(power_threshold).

    Parameters
    ----------
    pkl_path : str
        Path to crosscorrs pickle file
    fraction_threshold : float
        Fraction of bins required (e.g. 0.10 = 10%)
    power_threshold : float
        Power for exponential threshold (e.g. -10 → e^-10)
    """

    os.makedirs(out_dir, exist_ok=True)

    bad_pairs_path = os.path.join(out_dir, "fill_bin_bad_pairs.txt")
    good_pairs_path = os.path.join(out_dir, "fill_bin_good_pairs.txt")

    threshold = math.exp(power_threshold)

    with open(pkl_path, "rb") as f:
        crosscorrs = pickle.load(f)

    bad_pairs = []
    good_pairs = []

    for (pre, post), value in crosscorrs.items():
        # value is a tuple:
        # (lags, corr, mean, std, scores, total_score)

        # print(f"Filtering pair: {pre}, {post}")

        corr = value[1]

        if corr is None or len(corr) == 0:
            continue

        max_small = len(corr) * harshness
        # print(f"Max small bins allowed: {max_small}")

        num_small = np.sum(np.abs(corr) < threshold)
        # print(f"Number of small bins: {num_small}")

        if num_small >= max_small:
            print(
                f"{pre}, {post} — " f"{num_small} of bins |corr| < e^{power_threshold}"
            )
            bad_pairs.append((pre, post))
        else:
            good_pairs.append((pre, post))

        # Write bad pairs
    with open(bad_pairs_path, "w") as f:
        for pre, post in bad_pairs:
            f.write(f"{pre}\t{post}\n")

    # Write good pairs
    with open(good_pairs_path, "w") as f:
        for pre, post in good_pairs:
            f.write(f"{pre}\t{post}\n")


# can be combined with other filter methods
def filter_pairs_using_correlation_filled_bins(
    pairs,
    bad_pairs_path="selected_neurons_first_200s\\bad_pairs.txt",
):
    """
    Remove any pair that appears in bad_pairs.txt
    from the provided list of pairs.

    Parameters
    ----------
    pairs : list of tuple
        Original list of (pre, post) pairs
    good_pairs_path : str
    bad_pairs_path : str

    Returns
    -------
    filtered_pairs : list of tuple
    """

    def load_pairs(path):
        loaded = set()
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                pre, post = line.strip().split()
                loaded.add((pre, post))
        return loaded

    bad_pairs = load_pairs(bad_pairs_path)
    print(f"Loaded {len(bad_pairs)} bad pairs from {bad_pairs_path}")
    # print("bad_pairs", bad_pairs)

    filtered_pairs = [pair for pair in pairs if pair not in bad_pairs]

    return filtered_pairs


def check_firing_rate(
    pkl_path="data/analysis/selected_neurons_first_200s/crosscorrs_edge_mean_True_ultra-fine.pkl",  # find a way to automatically get this path
    out_dir="selected_neurons_first_200s",  # for debugging purposes
    min_rate=0.1,  # in Hz
    max_rate=5.0,  # in Hz
):
    os.makedirs(out_dir, exist_ok=True)

    bad_pairs_path = os.path.join(out_dir, "firing_rate_bad_pairs.txt")
    good_pairs_path = os.path.join(out_dir, "firing_rate_good_pairs.txt")

    with open(pkl_path, "rb") as f:
        crosscorrs = pickle.load(f)

    bad_pairs = []
    good_pairs = []

    for (pre, post), value in crosscorrs.items():
        # value is a tuple:
        # (lags, corr, mean, std, scores, total_score)

        # print(f"Checking firing rate for pair: {pre}, {post}")

        mean_rate = value[2]  # mean firing rate of pre neuron

        if mean_rate < min_rate or mean_rate > max_rate:
            print(
                f"{pre}, {post} — Mean firing rate {mean_rate:.2f} Hz outside range [{min_rate}, {max_rate}]"
            )
            bad_pairs.append((pre, post))
        else:
            good_pairs.append((pre, post))

    # Write bad pairs
    with open(bad_pairs_path, "w") as f:
        for pre, post in bad_pairs:
            f.write(f"{pre}\t{post}\n")

    # Write good pairs
    with open(good_pairs_path, "w") as f:
        for pre, post in good_pairs:
            f.write(f"{pre}\t{post}\n")


def filter_pairs_using_firing_rate(
    pairs,
    bad_pairs_path="selected_neurons_first_200s\\firing_rate_bad_pairs.txt",
):
    """
    Remove any pair that appears in bad_pairs.txt
    from the provided list of pairs.

    Parameters
    ----------
    pairs : list of tuple
        Original list of (pre, post) pairs
    good_pairs_path : str
    bad_pairs_path : str

    Returns
    -------
    filtered_pairs : list of tuple
    """

    def load_pairs(path):
        loaded = set()
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                pre, post = line.strip().split()
                loaded.add((pre, post))
        return loaded

    bad_pairs = load_pairs(bad_pairs_path)
    # print("bad_pairs", bad_pairs)

    filtered_pairs = [pair for pair in pairs if pair not in bad_pairs]

    return filtered_pairs


def check_histogram_unimodal(
    preNeuron=0,
    postNeuron=0,
    alpha=0.05,
):
    """
    Hartigan's Dip Test for Unimodality

    Must use continuous data.

    """

    # print(f"Checking unimodality for pair: {preNeuron}, {postNeuron}")

    filename = f"corr_trimmed_folder/corr_trimmed_{preNeuron}_{postNeuron}.txt"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Correlogram file not found: {filename}")

    # Load per-ms coincidence counts
    corr_ms = np.loadtxt(filename)

    # Construct lag axis
    n = len(corr_ms)

    if n == 0:
        return False, 0.0

    # Infer lag axis from length
    # Example: n=17 → lags = [-8, ..., 0, ..., +8]
    half = n // 2
    lags_ms = np.arange(-half, half + 1)

    if len(lags_ms) != n:
        raise ValueError("Correlogram length must be odd and centered at zero lag")

    # Reconstruct raw lag samples
    samples = np.repeat(lags_ms, corr_ms.astype(int))

    if len(samples) <= 3:  # too few data to run test; exclude
        return False, 0.0

    _, p_value = diptest(samples)

    # print(f"Unimodality p-value: {p_value}")
    test_result = False
    if p_value >= alpha:
        test_result = True

    return test_result, p_value


def check_correlations_unimodal(
    pkl_path="data/analysis/selected_neurons_first_200s/crosscorrs_edge_mean_True_ultra-fine.pkl",  # find a way to automatically get this path
    out_dir="selected_neurons_first_200s",  # for debugging purposes
    bin_centers=None,
    smoothing_sigma=1.0,
    prominence_fraction=0.25,
    min_distance_bins=1,
):

    os.makedirs(out_dir, exist_ok=True)

    bad_pairs_path = os.path.join(out_dir, "unimodal_bad_pairs.txt")
    good_pairs_path = os.path.join(out_dir, "unimodal_good_pairs.txt")

    with open(pkl_path, "rb") as f:
        crosscorrs = pickle.load(f)

    bad_pairs = []
    good_pairs = []

    for (pre, post), value in crosscorrs.items():
        # value is a tuple:
        # (lags, corr, mean, std, scores, total_score)

        # print(f"Filtering pair: {pre}, {post}")

        counts = value[1]

        if counts is None or len(counts) == 0:
            continue

        unimodality, significance = check_histogram_unimodal(
            preNeuron=pre,
            postNeuron=post,
            alpha=prominence_fraction,
        )

        if unimodality is False:
            print(f"{pre}, {post} — Correlogram is not unimodal p={significance:.4f}")
            bad_pairs.append((pre, post))
        else:
            good_pairs.append((pre, post))

        # Write bad pairs
    with open(bad_pairs_path, "w") as f:
        for pre, post in bad_pairs:
            f.write(f"{pre}\t{post}\n")

    # Write good pairs
    with open(good_pairs_path, "w") as f:
        for pre, post in good_pairs:
            f.write(f"{pre}\t{post}\n")


# can be combined with other filter methods
def filter_pairs_using_unimodality(
    pairs,
    bad_pairs_path="selected_neurons_first_200s\\unimodal_bad_pairs.txt",
):
    """
    Remove any pair that appears in bad_pairs.txt
    from the provided list of pairs.

    Parameters
    ----------
    pairs : list of tuple
        Original list of (pre, post) pairs
    good_pairs_path : str
    bad_pairs_path : str

    Returns
    -------
    filtered_pairs : list of tuple
    """

    def load_pairs(path):
        loaded = set()
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                pre, post = line.strip().split()
                loaded.add((pre, post))
        return loaded

    bad_pairs = load_pairs(bad_pairs_path)
    # print("bad_pairs", bad_pairs)

    filtered_pairs = [pair for pair in pairs if pair not in bad_pairs]

    return filtered_pairs


# find mode using kde
def calc_mode_using_kde(
    lags,
    corr,
    grid_size=1000,
):

    # ensure no negative weights
    weights = np.clip(corr, a_min=0.0, a_max=None)

    kde = gaussian_kde(dataset=lags, weights=weights, bw_method=None)
    grid = np.linspace(lags.min(), lags.max(), grid_size)
    density = kde(grid)
    mode = grid[np.argmax(density)]
    return float(mode)


# find mode of correlogram & stdev of correlogram
def check_stdev_around_mode(
    pkl_path="data/analysis/selected_neurons_first_200s/crosscorrs_edge_mean_True_ultra-fine.pkl",  # find a way to automatically get this path
    out_dir="selected_neurons_first_200s",  # for debugging purposes
    stdev_threshold=15.0,  # in ms
):

    os.makedirs(out_dir, exist_ok=True)

    bad_pairs_path = os.path.join(out_dir, "mode_stdev_bad_pairs.txt")
    good_pairs_path = os.path.join(out_dir, "mode_stdev_good_pairs.txt")

    with open(pkl_path, "rb") as f:
        crosscorrs = pickle.load(f)

    bad_pairs = []
    good_pairs = []

    for (pre, post), value in crosscorrs.items():
        # value is a tuple:
        # (lags, corr, mean, std, scores, total_score)

        # print(f"Filtering pair: {pre}, {post}")

        lags = value[0]
        corr = value[1]

        if corr is None or len(corr) == 0:
            continue

        # adjust lag if need be
        if len(lags) > len(corr):
            lags = 0.5 * (lags[:-1] + lags[1:])

        mode = calc_mode_using_kde(lags, corr)
        print(f"{pre}, {post} — Mode of correlogram: {mode:.2f} ms")

        if mode is None:
            print(f"{pre}, {post} — Could not calculate mode, skipping stdev check")
            stdev = stdev_threshold + 1.0  # force it to be bad
        else:
            stdev = np.sqrt(np.sum(corr * (lags - mode) ** 2) / np.sum(corr))
        print(f"{pre}, {post} — Stdev around mode: {stdev:.2f} ms")

        if stdev > stdev_threshold:
            print(
                f"{pre}, {post} — Stdev around mode {stdev:.2f} ms exceeds threshold {stdev_threshold} ms"
            )
            bad_pairs.append((pre, post))
        else:
            good_pairs.append((pre, post))

    # Write bad pairs
    with open(bad_pairs_path, "w") as f:
        for pre, post in bad_pairs:
            f.write(f"{pre}\t{post}\n")
    # Write good pairs
    with open(good_pairs_path, "w") as f:
        for pre, post in good_pairs:
            f.write(f"{pre}\t{post}\n")


def filter_pairs_using_mode_stdev(
    pairs,
    bad_pairs_path="selected_neurons_first_200s\\mode_stdev_bad_pairs.txt",
):
    """
    Remove any pair that appears in bad_pairs.txt
    from the provided list of pairs.

    Parameters
    ----------
    pairs : list of tuple
        Original list of (pre, post) pairs
    good_pairs_path : str
    bad_pairs_path : str

    Returns
    -------
    filtered_pairs : list of tuple
    """

    def load_pairs(path):
        loaded = set()
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                pre, post = line.strip().split()
                loaded.add((pre, post))
        return loaded

    bad_pairs = load_pairs(bad_pairs_path)
    # print("bad_pairs", bad_pairs)

    filtered_pairs = [pair for pair in pairs if pair not in bad_pairs]

    return filtered_pairs
