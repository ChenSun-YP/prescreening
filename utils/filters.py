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

    bad_pairs_path = os.path.join(out_dir, "bad_pairs.txt")
    good_pairs_path = os.path.join(out_dir, "good_pairs.txt")

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
        return True

    # Infer lag axis from length
    # Example: n=17 → lags = [-8, ..., 0, ..., +8]
    half = n // 2
    lags_ms = np.arange(-half, half + 1)

    if len(lags_ms) != n:
        raise ValueError("Correlogram length must be odd and centered at zero lag")

    # Reconstruct raw lag samples
    samples = np.repeat(lags_ms, corr_ms.astype(int))

    if len(samples) <= 3:  # too few data to run test; exclude
        return False

    _, p_value = diptest(samples)

    # print(f"Unimodality p-value: {p_value}")

    return p_value >= alpha


def check_correlations_unimodal(
    pkl_path="data/analysis/selected_neurons_first_200s/crosscorrs_edge_mean_True_ultra-fine.pkl",  # find a way to automatically get this path
    out_dir="selected_neurons_first_200s",  # for debugging purposes
    bin_centers=None,
    smoothing_sigma=1.0,
    prominence_fraction=0.05,
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

        unimodality = check_histogram_unimodal(
            preNeuron=pre,
            postNeuron=post,
            alpha=prominence_fraction,
        )

        if unimodality is True:
            # print(
            #     f"{pre}, {post} — " f"{num_small} of bins |corr| < e^{power_threshold}"
            # )
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


def calculate_mode_weighted_kde(lags, corr):
    """
    Estimate the mode lag using KDE weighted by corr values.

    Parameters
    ----------
    lags : 1D array-like (numeric)
        Lag values (e.g., ms).
    corr : 1D array-like (numeric)
        Corresponding correlation strengths (can be negative).

    Returns
    -------
    mode_lag : float
        Estimated mode (lag with highest weighted density).
    density_x : np.ndarray
        grid of x values used for KDE
    density_vals : np.ndarray
        KDE values on grid
    """
    lags = np.asarray(lags, dtype=float)
    corr = np.asarray(corr, dtype=float)

    if lags.size == 0 or corr.size == 0 or lags.size != corr.size:
        raise ValueError("lags and corr must be non-empty arrays of same length")

    # Convert corr into nonnegative weights for KDE:
    # shift by min and add small eps to avoid zeros
    min_corr = np.min(corr)
    weights = corr - min_corr
    eps = 1e-8
    weights = weights + eps

    # If weights are effectively zero (flat corr), fallback to argmax
    if np.allclose(weights, 0):
        mode_lag = float(lags[np.argmax(corr)])
        return mode_lag, np.array([mode_lag]), np.array([1.0])

    # Use scipy gaussian_kde with weights (available on modern scipy)
    try:
        kde = gaussian_kde(lags, weights=weights)
        xs = np.linspace(np.min(lags), np.max(lags))
        dens = kde(xs)
        mode_lag = float(xs[np.argmax(dens)])
        return mode_lag, xs, dens
    except TypeError:
        # Older scipy may not support weights in gaussian_kde.
        # Fallback: approximate weighted KDE by repeating values proportional to normalized weights.
        # To avoid huge arrays, scale weights to a manageable integer multiplier.
        scaled = weights - np.min(weights)
        scaled = scaled / np.max(scaled)
        repeats = (scaled * 100).astype(int) + 1  # at least 1 repetition
        expanded = np.repeat(lags, repeats)
        # compute KDE on expanded samples (unweighted)
        kde = gaussian_kde(expanded)
        xs = np.linspace(np.min(lags), np.max(lags))
        dens = kde(xs)
        mode_lag = float(xs[np.argmax(dens)])
        return mode_lag, xs, dens


def weighted_std_around_mode(lags, corr, mode_lag):
    """
    Compute weighted RMS deviation of lags around mode_lag using corr-derived weights.

    Uses nonnegative weights derived from corr by shifting to make min == 0
    (so stronger positive correlations contribute more).

    Returns standard deviation in same units as lags.
    """
    lags = np.asarray(lags, dtype=float)
    corr = np.asarray(corr, dtype=float)

    if lags.size == 0 or corr.size == 0 or lags.size != corr.size:
        raise ValueError("lags and corr must be non-empty arrays of same length")

    # weights: shift so min is zero, then clip to non-negative
    weights = corr - np.min(corr)
    # tiny epsilon to avoid all-zero weights if corr is constant
    if np.allclose(weights, 0):
        # fallback: unweighted stdev of lags around mode
        return float(np.sqrt(np.mean((lags - mode_lag) ** 2)))
    # compute weighted variance
    w = weights.astype(float)
    wsum = np.sum(w)
    var = np.sum(w * (lags - mode_lag) ** 2) / wsum
    return float(np.sqrt(var))


def check_pairs_using_mode_stdev(
    pkl_path="data/analysis/selected_neurons_first_200s/crosscorrs_edge_mean_True_ultra-fine.pkl",
    out_dir="selected_neurons_first_200s",
    stdev_threshold=15.0,  # units same as lags array (ms)
):
    os.makedirs(out_dir, exist_ok=True)
    bad_pairs_path = os.path.join(out_dir, "mode_stdev_bad_pairs.txt")
    good_pairs_path = os.path.join(out_dir, "mode_stdev_good_pairs.txt")

    with open(pkl_path, "rb") as f:
        crosscorrs = pickle.load(f)

    bad_pairs = []
    good_pairs = []

    for (pre, post), value in crosscorrs.items():
        # value expected: (lags, corr, mean, std, scores, total_score)
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            print(f"Skipping {pre},{post}: unexpected value format")
            continue

        lags = np.asarray(value[0], dtype=float)
        corr = np.asarray(value[1], dtype=float)
        print(lags.size, corr.size)

        # skip empty or mismatched arrays
        if lags.size == 0 or corr.size == 0 or lags.size != corr.size:
            print(f"Skipping {pre},{post}: empty or mismatched lags/corr")
            print(f"lags: {lags}, corr: {corr}")
            continue

        # Try weighted KDE mode, fallback to argmax
        try:
            mode_lag, _, _ = calculate_mode_weighted_kde(lags, corr)
        except Exception:
            # fallback: pick lag with maximum corr
            mode_lag = float(lags[np.argmax(corr)])

        # compute weighted stdev around mode (in same units as lags)
        stdev = weighted_std_around_mode(lags, corr, mode_lag)

        print(f"Pair {pre} -> {post}: mode_lag = {mode_lag:.3f}, stdev = {stdev:.3f}")

        if stdev > stdev_threshold:
            bad_pairs.append((pre, post))
        else:
            good_pairs.append((pre, post))

    # write files
    with open(bad_pairs_path, "w") as f:
        for pre, post in bad_pairs:
            f.write(f"{pre}\t{post}\n")

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
